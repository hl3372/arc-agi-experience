"""P4 ARC-AGI-2 evaluation with vLLM: sample, force answers for capped samples, score, summarize.

    python -m arcsolver.eval.run_eval --model MODEL_DIR --tasks-dir ARC-AGI-2/data/evaluation --out-dir OUT \
        [--items ITEMS] [--n 4] [--kv-cache-dtype fp8] [--mtp 3] [--dp 8] [--tp 1] [--no-think]

Protocol (P4):
  * split tests: one prompt per test input (the task's training pairs + that test input);
  * prompt arm r2k_zb_max (arc_render.build_messages), rendered with the model's chat template,
    add_generation_prompt=True, thinking on (--no-think: enable_thinking=False, for answer-only models);
  * k samples per test input (--n 4) at temperature 1.0, top_p 1.0, top-k off, each sample its own request
    with its own seed; max new tokens 147,000 (never past --max-model-len 196,608); prefix caching off;
  * a sample stopped by the token cap with its reasoning still open is continued after the force suffix
    (default "\\n\\nNow final.\\n</think>\\n\\n") for up to --force-budget 1,024 tokens, sampled with the same
    settings; it is scored separately as the "forced" view (natural = headline);
  * scoring and averaging: arcsolver.eval.scoring.

Outputs in --out-dir: samples.jsonl (one row per test input, appended as each finishes; a rerun skips the test
inputs already there), summary.json, config.json (the settings a resume must match). --dp N runs N engine
processes (each on --tp GPUs) over disjoint test inputs; their rows land in samples.shard<r>.jsonl and are
merged into samples.jsonl when they end (or at the next start, after an interruption).
"""
from __future__ import annotations

import argparse
import codecs
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

from arcsolver.eval import scoring as S
from arcsolver.prompt import arc_render as ar

SAMPLES = "samples.jsonl"
SUMMARY = "summary.json"
CONFIG = "config.json"
MIN_FORCE_ROOM = 64          # a capped sample with less context room than this left gets no forced answer
RESUME_KEYS = ("model", "n", "temperature", "top_p", "top_k", "max_new_tokens", "max_model_len", "force_budget",
               "force_suffix", "think", "seed")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [eval] {msg}", flush=True)


def derive_seed(*parts) -> int:
    """A stable 30-bit seed from the run seed and a sample's identity (independent of order, sharding, resume)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x3FFFFFFF


def decode_escapes(s: str) -> str:
    """Backslash escapes in a command-line string ("\\n") -> characters."""
    return codecs.decode(s, "unicode_escape") if "\\" in s else s


# --- files ----------------------------------------------------------------------------------------

def repair_jsonl(path: str) -> None:
    """Drop a truncated last line (an interrupted write) so appends start on a fresh line."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    rows, bad = S.read_jsonl(path)
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        ends_clean = f.read(1) == b"\n"
    if bad or not ends_clean:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        log(f"repaired {path}: kept {len(rows)} rows, dropped {bad} unreadable lines")


def append_rows(f, rows) -> None:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def merge_shards(out_dir: str) -> None:
    """Move the rows of samples.shard*.jsonl into samples.jsonl (first row per test input kept)."""
    main = os.path.join(out_dir, SAMPLES)
    files = sorted(glob.glob(os.path.join(out_dir, "samples.shard*.jsonl")))
    if not files:
        return
    repair_jsonl(main)
    have = {S.row_key(r) for r in S.read_jsonl(main)[0]}
    n = 0
    with open(main, "a", encoding="utf-8") as f:
        for p in files:
            rows = [r for r in S.read_jsonl(p)[0] if S.row_key(r) not in have]
            append_rows(f, rows)
            have.update(S.row_key(r) for r in rows)
            n += len(rows)
            os.remove(p)
    for p in glob.glob(os.path.join(out_dir, "shard*.units.json")):
        os.remove(p)
    log(f"merged {n} rows from {len(files)} shard files into {main}")


def run_config(args) -> dict:
    return {"model": os.path.abspath(args.model) if os.path.isdir(args.model) else args.model,
            "tokenizer": args.tokenizer or None, "arm": S.ARM, "n": args.n, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len, "force_budget": args.force_budget,
            "force_suffix": args.force_suffix, "think": not args.no_think, "seed": args.seed,
            "kv_cache_dtype": args.kv_cache_dtype, "mtp": args.mtp, "mtp_method": args.mtp_method,
            "draft_sample_method": args.draft_sample_method or None, "quantization": args.quantization or None,
            "dtype": args.dtype, "dp": args.dp, "tp": args.tp, "prefix_caching": False}


def check_config(args, out_dir: str) -> None:
    """Write config.json on the first run; on a resume, refuse settings that would mix incomparable samples."""
    path = os.path.join(out_dir, CONFIG)
    cfg = run_config(args)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diff = {k: (old.get(k), cfg[k]) for k in RESUME_KEYS if old.get(k) != cfg[k]}
        if diff and not args.allow_config_change:
            raise SystemExit(f"{out_dir} holds samples drawn with other settings {diff} (old, new); use a new "
                             f"--out-dir or pass --allow-config-change")
        if not diff:
            return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# --- engine ---------------------------------------------------------------------------------------

def has_vision_tower(model: str) -> bool:
    p = os.path.join(model, "config.json")
    if not os.path.isfile(p):
        return False
    with open(p, encoding="utf-8") as f:
        return "vision_config" in json.load(f)


def make_llm(args):
    from vllm import LLM
    kw = dict(model=args.model, tokenizer=args.tokenizer or args.model, dtype=args.dtype,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tp, seed=args.seed, enable_prefix_caching=False,
              kv_cache_dtype=args.kv_cache_dtype, enforce_eager=args.enforce_eager)
    if args.max_num_seqs:
        kw["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.quantization:
        kw["quantization"] = args.quantization
    if args.mtp > 0:          # the checkpoint's own MTP head as the drafter
        spec = {"method": args.mtp_method, "num_speculative_tokens": args.mtp}
        if args.draft_sample_method:
            spec["draft_sample_method"] = args.draft_sample_method
        kw["speculative_config"] = spec
    if has_vision_tower(args.model):      # text-only prompts: no encoder budget for images / video
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    log(f"vLLM engine: { {k: v for k, v in kw.items() if k != 'model'} }")
    return LLM(**kw)


def render_prompt(tok, task: dict, test_index: int, think: bool) -> list[int]:
    msgs = ar.build_messages(S.single_test_task(task, test_index), S.ARM)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=think)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def run_worker(args, tasks: dict, units: list, out_path: str) -> None:
    """Generate, force and score the given test inputs; append one row per test input to out_path as it ends."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm
    try:                      # only finished requests come out of step() (what LLM.generate asks for too)
        from vllm.sampling_params import RequestOutputKind
        final_only = {"output_kind": RequestOutputKind.FINAL_ONLY}
    except ImportError:
        final_only = {}

    think = not args.no_think
    force = think and args.force_budget > 0
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    prompts = [render_prompt(tok, tasks[t], j, think) for t, j in units]
    tail = tok.decode(prompts[0][-8:])
    want = "<think>\n" if think else "</think>\n\n"
    if not tail.endswith(want):
        log(f"WARNING: the rendered prompt ends with {tail!r}, not {want!r}: check the chat template / thinking mode")
    suffix_ids = [int(x) for x in tok(args.force_suffix, add_special_tokens=False)["input_ids"]] if force else []
    plens = sorted(len(p) for p in prompts)
    log(f"{len(units)} test inputs x {args.n} samples; prompt tokens {plens[0]}..{plens[-1]} "
        f"(median {plens[len(plens) // 2]}); force {'on, ' + str(len(suffix_ids)) + ' suffix tokens' if force else 'off'}")

    t_load = time.time()
    llm = make_llm(args)
    engine = llm.llm_engine
    log(f"vLLM {vllm.__version__} ready in {time.time() - t_load:.0f}s")

    def params(max_tokens: int, seed: int):
        return SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                              max_tokens=max_tokens, seed=seed, **final_only)

    pending, counter = {}, itertools.count()

    def submit(ids: list, sp, tag: tuple) -> None:
        prompt = TokensPrompt(prompt_token_ids=list(ids))
        if hasattr(llm, "enqueue"):
            rid = llm.enqueue([prompt], [sp], use_tqdm=False)[0]
        else:
            rid = f"arc{next(counter)}"
            engine.add_request(rid, prompt, sp)
        # enqueue may return the engine's internal id "<external>-<suffix>"; outputs carry the external id
        pending[str(rid).rsplit("-", 1)[0]] = tag

    def take(out_id: str) -> tuple:
        for key in (out_id, out_id.rsplit("-", 1)[0]):
            if key in pending:
                return pending.pop(key)
        raise RuntimeError(f"output for an unknown request id {out_id!r}")

    n = args.n
    state = [{"samples": [None] * n, "forced": [None] * n, "forced_tokens": [None] * n, "forced_finish": [None] * n,
              "open": 0} for _ in units]
    stats = {"rows": 0, "samples": 0, "gen": 0, "forced_sub": 0, "forced_done": 0, "forced_gen": 0}
    out = open(out_path, "a", encoding="utf-8")

    def finish_unit(i: int) -> None:
        t, j = units[i]
        st = state[i]
        texts = [s["text"] for s in st["samples"]]
        truth = ar.grid_key(tasks[t]["test"][j]["output"])
        sc = S.score_samples(texts, st["forced"], truth, think)
        row = {"task_id": t, "test_index": j, "n_test": len(tasks[t]["test"]), "prompt_tokens": len(prompts[i]),
               "think": think, "texts": texts, "gen_tokens": [s["tokens"] for s in st["samples"]],
               "finish": [s["finish"] for s in st["samples"]], "closed": sc["closed"],
               "forced": st["forced"], "forced_tokens": st["forced_tokens"], "forced_finish": st["forced_finish"],
               "correct": sc["correct"], "correct_forced": sc["correct_forced"],
               "n_correct": sum(sc["correct"]), "n_correct_forced": sum(sc["correct_forced"]),
               "seeds": [derive_seed(args.seed, t, j) + k for k in range(n)]}
        append_rows(out, [row])
        stats["rows"] += 1
        state[i] = None                                    # free the texts

    for i, (t, j) in enumerate(units):                     # prompt-major: rows complete early, in order
        cap = min(args.max_new_tokens, args.max_model_len - len(prompts[i]))
        if cap < 1:
            log(f"WARNING: {t} test {j}: prompt of {len(prompts[i])} tokens leaves no room under --max-model-len")
            state[i]["samples"] = [{"text": "", "tokens": 0, "finish": "prompt_too_long"}] * n
            finish_unit(i)
            continue
        base = derive_seed(args.seed, t, j)
        for k in range(n):
            submit(prompts[i], params(cap, base + k), ("s", i, k))
        state[i]["open"] = n

    t0 = last = time.time()
    while pending:
        for o in engine.step():
            if not o.finished:
                continue
            kind, i, k = take(o.request_id)
            c = o.outputs[0] if o.outputs else None
            text = (c.text or "") if c is not None else ""
            ids = [int(x) for x in c.token_ids] if c is not None else []
            fin = (c.finish_reason or "") if c is not None else "abort"
            st = state[i]
            st["open"] -= 1
            if kind == "s":
                st["samples"][k] = {"text": text, "tokens": len(ids), "finish": fin}
                stats["samples"] += 1
                stats["gen"] += len(ids)
                if force and S.needs_forced_answer(fin, text, think):
                    fp = prompts[i] + ids + suffix_ids
                    room = args.max_model_len - len(fp)
                    if room >= MIN_FORCE_ROOM:
                        t, j = units[i]
                        submit(fp, params(min(args.force_budget, room), derive_seed(args.seed, t, j, "forced") + k),
                               ("f", i, k))
                        st["open"] += 1
                        stats["forced_sub"] += 1
            else:
                st["forced"][k] = args.force_suffix + text
                st["forced_tokens"][k] = len(ids)
                st["forced_finish"][k] = fin
                stats["forced_done"] += 1
                stats["forced_gen"] += len(ids)
            if st["open"] == 0:
                finish_unit(i)
        if time.time() - last >= args.log_every:
            last = time.time()
            dt = last - t0
            log(f"rows {stats['rows']}/{len(units)}  samples {stats['samples']}/{len(units) * n}  forced "
                f"{stats['forced_done']}/{stats['forced_sub']}  generated {(stats['gen'] + stats['forced_gen']) / 1e6:.2f}M "
                f"tokens ({(stats['gen'] + stats['forced_gen']) / max(dt, 1e-9):,.0f} tok/s)  {dt / 60:.0f} min")
    out.close()
    left = [units[i] for i, st in enumerate(state) if st is not None]
    if left:
        raise RuntimeError(f"{len(left)} test inputs never completed, e.g. {left[:3]}")
    log(f"done: {stats['rows']} rows, {stats['forced_sub']} forced answers, "
        f"{(stats['gen'] + stats['forced_gen']) / 1e6:.2f}M tokens in {(time.time() - t0) / 60:.1f} min")


# --- data parallel --------------------------------------------------------------------------------

def run_data_parallel(args, argv: list, todo: list, out_dir: str) -> int:
    """--dp N: N child processes of this module (each `--dp 1` on its own --tp GPUs) over disjoint test inputs."""
    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpus = [g.strip() for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if g.strip()]
    else:
        gpus = [str(g) for g in range(args.dp * args.tp)]
    if len(gpus) < args.dp * args.tp:
        raise SystemExit(f"--dp {args.dp} x --tp {args.tp} needs {args.dp * args.tp} GPUs, have {gpus}")
    env_path = REPO_ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    procs = []
    for r in range(args.dp):
        part = todo[r::args.dp]
        if not part:
            continue
        units_file = os.path.join(out_dir, f"shard{r}.units.json")
        with open(units_file, "w", encoding="utf-8") as f:
            json.dump([list(u) for u in part], f)
        cmd = [sys.executable, "-m", "arcsolver.eval.run_eval", *argv, "--dp", "1",
               "--worker-units", units_file, "--worker-out", os.path.join(out_dir, f"samples.shard{r}.jsonl")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus[r * args.tp:(r + 1) * args.tp]), PYTHONPATH=env_path)
        log_file = open(os.path.join(out_dir, f"shard{r}.log"), "a", encoding="utf-8")
        procs.append((r, subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT), log_file))
        log(f"shard {r}: {len(part)} test inputs on GPUs {env['CUDA_VISIBLE_DEVICES']} (log {log_file.name})")
        if args.dp_stagger > 0 and r < args.dp - 1:
            time.sleep(args.dp_stagger)
    failed = []
    for r, p, lf in procs:
        if p.wait() != 0:
            failed.append(r)
        lf.close()
    merge_shards(out_dir)
    if failed:
        log(f"shards {failed} failed (see shard<r>.log); finished rows are kept, a rerun continues")
    return 1 if failed else 0


# --- main -----------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model directory (or hub id)")
    ap.add_argument("--tokenizer", default="", help="tokenizer / chat template source (default: --model)")
    ap.add_argument("--tasks-dir", required=True, help="directory of ARC task JSONs (train/test pairs, test outputs)")
    ap.add_argument("--items", default="", help="optional item list restricting the tasks (JSON list / {'items': [...]}"
                                                " / one task id per line)")
    ap.add_argument("--limit-tasks", type=int, default=0, help="only the first N selected tasks (smoke tests)")
    ap.add_argument("--out-dir", required=True)
    g = ap.add_argument_group("sampling")
    g.add_argument("--n", type=int, default=4, help="samples per test input")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0, help="0 (or -1) = off, as in the reference runs")
    g.add_argument("--max-new-tokens", type=int, default=147000)
    g.add_argument("--max-model-len", type=int, default=196608, help="prompt + generation context of the engine")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-think", action="store_true",
                   help="enable_thinking=False (answer-only models): the reply is the answer, no forced answers")
    g = ap.add_argument_group("forced answers")
    g.add_argument("--force-budget", type=int, default=1024, help="max tokens of a forced answer (0 = off)")
    g.add_argument("--force-suffix", default=S.DEFAULT_FORCE_SUFFIX,
                   help="text appended to a capped sample before its forced answer (backslash escapes decoded)")
    g = ap.add_argument_group("engine")
    g.add_argument("--dtype", default="bfloat16")
    g.add_argument("--kv-cache-dtype", default="auto", help="'fp8' for an FP8 KV cache (reference runs)")
    g.add_argument("--quantization", default="", help="vLLM quantization, e.g. 'fp8' (default: from the checkpoint)")
    g.add_argument("--mtp", type=int, default=0, help="MTP speculative tokens from the model's own MTP head "
                                                      "(reference runs: 3; 0 = off)")
    g.add_argument("--mtp-method", default="mtp", help="speculative_config method (vLLM 0.30: 'mtp')")
    g.add_argument("--draft-sample-method", default="", choices=["", "greedy", "probabilistic"],
                   help="speculative_config draft_sample_method ('' = vLLM default; both are exact sampling)")
    g.add_argument("--tp", "--tensor-parallel-size", dest="tp", type=int, default=1)
    g.add_argument("--dp", "--data-parallel-size", dest="dp", type=int, default=1,
                   help="engine processes over disjoint test inputs, each on --tp GPUs")
    g.add_argument("--gpus", default="", help="GPU ids for --dp (default: $CUDA_VISIBLE_DEVICES or 0..dp*tp-1)")
    g.add_argument("--dp-stagger", type=float, default=10.0, help="seconds between engine starts with --dp")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    g.add_argument("--max-num-seqs", type=int, default=0, help="0 = vLLM default")
    g.add_argument("--max-num-batched-tokens", type=int, default=0, help="0 = vLLM default")
    g.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--log-every", type=float, default=120.0, help="seconds between progress lines")
    ap.add_argument("--allow-config-change", action="store_true",
                    help="resume an out dir whose config.json has other sampling settings")
    ap.add_argument("--worker-units", default="", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default="", help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    args.force_suffix = decode_escapes(args.force_suffix)
    if args.n < 1 or args.max_new_tokens < 1 or args.dp < 1 or args.tp < 1 or args.force_budget < 0:
        raise SystemExit("--n, --max-new-tokens, --dp and --tp must be >= 1, --force-budget >= 0")
    if args.force_budget > 0 and not args.no_think and "</think>" not in args.force_suffix:
        raise SystemExit("--force-suffix must close the reasoning ('</think>'), or the forced answer never parses")

    if args.worker_units:                                  # a --dp child
        with open(args.worker_units, encoding="utf-8") as f:
            units = [(str(t), int(j)) for t, j in json.load(f)]
        tasks = S.load_tasks(args.tasks_dir, sorted({t for t, _ in units}))
        repair_jsonl(args.worker_out)
        run_worker(args, tasks, units, args.worker_out)
        return 0

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    task_ids = S.read_items(args.items) if args.items else S.task_ids_in(args.tasks_dir)
    if args.limit_tasks:
        task_ids = task_ids[:args.limit_tasks]
    if not task_ids:
        raise SystemExit(f"no tasks selected from {args.tasks_dir}")
    tasks = S.load_tasks(args.tasks_dir, task_ids)
    units = [(t, j) for t in task_ids for j in range(len(tasks[t]["test"]))]
    check_config(args, out_dir)
    merge_shards(out_dir)                                  # rows of an interrupted --dp run
    main_path = os.path.join(out_dir, SAMPLES)
    repair_jsonl(main_path)
    done = {S.row_key(r) for r in S.read_jsonl(main_path)[0]}
    todo = [u for u in units if u not in done]
    log(f"{len(task_ids)} tasks, {len(units)} test inputs: {len(units) - len(todo)} already in {main_path}, "
        f"{len(todo)} to run")

    rc = 0
    if todo:
        if args.dp == 1:
            run_worker(args, tasks, todo, main_path)
        else:
            rc = run_data_parallel(args, argv, todo, out_dir)

    rows, seen = [], set()
    for r in S.read_jsonl(main_path)[0]:
        if S.row_key(r) not in seen:
            seen.add(S.row_key(r))
            rows.append(r)
    summary = S.summarize(rows, {t: len(tasks[t]["test"]) for t in task_ids})
    summary["config"] = run_config(args)
    with open(os.path.join(out_dir, SUMMARY), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(S.format_summary(summary), flush=True)
    log(f"summary written to {os.path.join(out_dir, SUMMARY)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
