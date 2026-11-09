#!/usr/bin/env python
"""Build an SFT row cache: reasoning rows (CoT and/or render traces) + optional aux rows, with an ARC eval leak check.

Output (--output, one JSON): {"examples": [{"messages", "length", "prov"}, ...], "metadata": {...}, "order": [[row
indices of epoch 0], [epoch 1], ...]}, plus <output>.report.json (the metadata). tokenize_cache.py turns it into the
trainer's format. Rows longer than --max-seq-length (templated, in tokens, with --tokenizer) are dropped, never
truncated. Rows that look like an ARC-AGI-2 evaluation task are dropped (task id, or >= 2 of the eval task's grids
appearing verbatim in the prompt, in the indexed-row rendering or as plain digit rows).

Inputs:
- --cot FILE(s): reasoning rows, one JSON per line (see README.md "Input row format"). Accepted shapes:
    {"prompt_messages": [system?, user], "reasoning" (or "body"): str, "answer": str, "task_id": str, ...}
    {"messages": [system?, user, {"role": "assistant", "content": "<think>\\n...\\n</think>\\n\\n..."}], ...}
  The assistant turn is "<think>\\n{reasoning}\\n</think>\\n\\n{answer}", which is what the Qwen3.5 chat template
  renders for a thinking turn and what generation with enable_thinking=True continues from.
- --traces FILE(s) (render traces): {"prompt_messages": [system, user], "body", "answer", "arm_id", "task_id",
  "cohort", "slot"}; the arm's channel decides where the body goes (arc_render.channel).
- --aux-cache: a cache in this script's own output format (non-thinking rows with an empty think block).

Epoch layouts:
- --epochs N with --cot ... --cot-repeat (the recipes in README.md): every CoT row is stored once and shown in every
  epoch, reshuffled each epoch (seed + 1 + e).
- --cot without --cot-repeat: CoT rows are split into N disjoint per-epoch blocks (each seen once over the run).
- Any of --cot, --aux-all or --fixed-trace-order selects the three-source layout: epoch e = the traces, with aux
  block e and CoT block e each shuffled and spread evenly through the epoch. --aux-all uses every aux row of
  --aux-cache (<= the cap) split into --epochs disjoint blocks; --trace-tag TEXT puts TEXT on its own line at the
  start of every trace's reasoning; --fixed-trace-order keeps one trace permutation in every epoch.
- Otherwise (traces + --aux-frac sample): with --epochs N > 1 the cache holds every trace once plus N disjoint blocks
  of aux rows; epoch e = all traces + aux block e, shuffled (traces repeat, aux rows never do). --group-order shows
  trace cohorts in a fixed order per epoch.
- --exclude-ids FILE(s) drops traces and CoT rows whose task id is listed (held-out and flagged tasks).
- A block holds len // epochs rows (every epoch must have the same length); the remainder is reported and dropped.

    python -m arcsolver.sft.build_cache --cot rows.jsonl --cot-repeat --aux-frac 0 --epochs 3 \\
        --max-seq-length 262144 --tokenizer MODEL_DIR --eval-dir ARC-AGI-2/data/evaluation --workers 64 \\
        --output work/cache.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import time
from collections import Counter

from arcsolver.prompt import arc_render as ar

THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"  # what the template writes for a non-thinking turn (and aux rows)


def assistant_text(trace, tag: str | None = None) -> str:
    """The trace's arm decides the channel: 'think' puts the body in the think block and the answer after it;
    'output' (render_output) leaves the think block empty and writes body, the ANSWER_MARKER line, then the answer.
    tag (e.g. "<alpha>"): its own first line of the reasoning, in either channel."""
    body = trace["body"].strip("\n")
    if tag:
        body = tag + "\n" + body
    if trace.get("arm_id") in ar.ARMS and ar.channel(trace["arm_id"]) == "output":
        return (EMPTY_THINK + body + "\n\n" + ar.ANSWER_MARKER + "\n" + trace["answer"].strip("\n"))
    return THINK_OPEN + body + THINK_CLOSE + trace["answer"].strip("\n")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def token_length(tok, messages) -> int:
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


def eval_fingerprints(eval_dir, min_cells=9):
    """{task_id: set of rendered grid-row blocks} for every train/test grid of every eval task
    with at least `min_cells` cells (tiny grids collide by chance)."""
    fps = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        task = ar.load_task(path)
        grids = [p[r] for p in task["train"] for r in ("input", "output")] + [t["input"] for t in task["test"]]
        fps[os.path.basename(path)[:-5]] = {
            "\n".join(ar._indexed_rows(g, "")) for g in grids if len(g) * len(g[0]) >= min_cells}
    return fps


def leak_hits(prompt_text, task_id, fps, min_grids=2):
    """Eval task ids whose id is the trace's id or whose grids appear >= min_grids times."""
    base = task_id.split("__")[0]
    hits = [e for e in fps if e == task_id or e == base]
    for e, blocks in fps.items():
        if e not in hits and sum(b in prompt_text for b in blocks) >= min_grids:
            hits.append(e)
    return hits


def interleave_evenly(main: list, extra: list) -> list:
    """Spread `extra` evenly through `main`, keeping both orders (item i of a list of n sits at (i + 0.5) / n)."""
    keyed = [((i + 0.5) / len(main), 0, x) for i, x in enumerate(main)]
    keyed += [((k + 0.5) / len(extra), 1, x) for k, x in enumerate(extra)]
    return [x for _, _, x in sorted(keyed, key=lambda t: (t[0], t[1]))]


def interleave_streams(streams: list) -> list:
    """Spread several ordered lists evenly through one sequence, keeping each list's order: item i of a list of n
    sits at (i + 0.5) / n; ties go to the earlier list. interleave_evenly(a, b) == interleave_streams([a, b])."""
    keyed = [((i + 0.5) / len(st), k, i, x) for k, st in enumerate(streams) if st for i, x in enumerate(st)]
    return [x for *_, x in sorted(keyed, key=lambda t: t[:3])]


def split_blocks(items: list, epochs: int) -> tuple:
    """(epochs disjoint blocks of len(items) // epochs, in order; the leftover items)."""
    size = len(items) // epochs
    return [items[e * size:(e + 1) * size] for e in range(epochs)], items[epochs * size:]


def block_orders(n_traces: int, blocks: list, epochs: int, seed: int, fixed_trace_order: bool, repeat=()) -> list:
    """Row order per epoch for the three-source layout. blocks: one entry per extra source (aux, CoT), each a list
    of `epochs` lists of row indices. Epoch e = the traces (one permutation reused every epoch when
    fixed_trace_order, else reshuffled with seed + 1 + e) with each source's block e shuffled and spread evenly.
    repeat: further row indices shown in every epoch together with the traces (--cot-repeat)."""
    base = list(range(n_traces)) + list(repeat)
    random.Random(seed).shuffle(base)
    orders = []
    for e in range(epochs):
        rng = random.Random(seed + 1 + e)
        traces = list(base)
        if not fixed_trace_order:
            rng.shuffle(traces)
        streams = [traces]
        for src in blocks:
            b = list(src[e])
            rng.shuffle(b)
            streams.append(b)
        orders.append(interleave_streams(streams))
    return orders


def compact_fingerprints(eval_dir, min_cells=20):
    """{task_id: set of compact digit-row blocks} for the eval tasks' non-uniform grids with >= min_cells cells:
    the leak check for prompts that render grids as plain digit rows (raw CoT sources)."""
    fps = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        task = ar.load_task(path)
        grids = [p[r] for p in task["train"] for r in ("input", "output")] + [t["input"] for t in task["test"]]
        fps[os.path.basename(path)[:-5]] = {
            "\n".join("".join(str(c) for c in row) for row in g) for g in grids
            if len(g) * len(g[0]) >= min_cells and len({c for row in g for c in row}) > 1}
    return fps


_COT = {}   # per-run state of the CoT row check (set in main; inherited by --workers processes through fork)


def _cot_worker_init():
    from transformers import AutoTokenizer
    _COT["tok"] = AutoTokenizer.from_pretrained(_COT["tokenizer"])


def _cot_one(job):
    """(kind, value) for one raw CoT line: malformed / excluded / eval_overlap / too_long / ok (the row dict)."""
    base, line_no, line = job
    t = json.loads(line)
    msgs = load_cot_row(t)
    if msgs is None:
        return "malformed", None
    tid = str(t.get("task_id", ""))
    if tid and (tid in _COT["excluded"] or tid.split("arc2-training__")[-1] in _COT["excluded"]):
        return "excluded", None
    user = msgs[-2]["content"]
    hits = leak_hits(user, tid, _COT["fps"]) if tid else []
    hits += [e for e, blocks in _COT["cfps"].items() if e not in hits and sum(b in user for b in blocks) >= 2]
    if hits:
        return "eval_overlap", {"task_id": tid, "eval_ids": hits}
    n = token_length(_COT["tok"], msgs)
    if n > _COT["max_len"]:
        return "too_long", n
    return "ok", {"messages": msgs, "length": n, "prov": f"cot|{base}|{tid}|l{line_no}"}


def load_cot_row(t: dict) -> list | None:
    """Messages [system?, user, assistant] of one raw CoT record, or None when malformed. Accepted shapes:
    {"messages": [..., {"role": "assistant", ...}]} with the reasoning inside <think>...</think> in the assistant
    turn, or {"prompt_messages": [...], "body"|"reasoning": ..., "answer": ...}."""
    if t.get("messages"):
        msgs = [{"role": m["role"], "content": m["content"]} for m in t["messages"]]
    elif t.get("prompt_messages") and (t.get("body") or t.get("reasoning")) and t.get("answer") is not None:
        body = (t.get("body") or t.get("reasoning")).strip("\n")
        msgs = [{"role": m["role"], "content": m["content"]} for m in t["prompt_messages"]]
        msgs.append({"role": "assistant", "content": THINK_OPEN + body + THINK_CLOSE + str(t["answer"]).strip("\n")})
    else:
        return None
    roles = [m["role"] for m in msgs]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        return None
    a = msgs[-1]["content"]
    if not a.startswith("<think>") or "</think>" not in a:
        return None
    return msgs


def epoch_orders(n_traces: int, n_aux_per_epoch: int, epochs: int, seed: int, groups=None) -> list:
    """Row order for each epoch of a multi-epoch cache laid out as [traces..., aux block 0, aux block 1, ...].
    Epoch e = every trace index + the indices of aux block e, shuffled with seed + 1 + e.
    groups (lists of trace indices covering every trace once): each epoch shows the groups in this order, each
    shuffled on its own, with the epoch's aux rows spread evenly through the whole epoch."""
    traces = list(range(n_traces))
    if groups is not None and sorted(i for g in groups for i in g) != traces:
        raise ValueError("groups must cover every trace index exactly once")
    orders = []
    for e in range(epochs):
        start = n_traces + e * n_aux_per_epoch
        aux = list(range(start, start + n_aux_per_epoch))
        rng = random.Random(seed + 1 + e)
        if groups is None:
            idx = traces + aux
            rng.shuffle(idx)
        else:
            seq = []
            for g in groups:
                g = list(g)
                rng.shuffle(g)
                seq += g
            rng.shuffle(aux)
            idx = interleave_evenly(seq, aux) if aux else seq
        orders.append(idx)
    return orders


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="*", default=[],
                    help="render-trace jsonl file(s) (none: aux + CoT only, three-source layout)")
    ap.add_argument("--aux-cache", default=None, help="a cache in this script's output format (aux rows)")
    ap.add_argument("--aux-frac", type=float, default=0.10, help="aux share of the final rows")
    ap.add_argument("--max-seq-length", type=int, default=20480)
    ap.add_argument("--tokenizer", required=True, help="model dir (or hub id) whose chat template and tokenizer the "
                                                        "trainer uses (the vocab-pruned model when pruning)")
    ap.add_argument("--eval-dir", required=True,
                    help="ARC-AGI-2 evaluation task dir (*.json) for the leak check, e.g. ARC-AGI-2/data/evaluation")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--epochs", type=int, default=1,
                    help="> 1: fresh aux rows every epoch (disjoint blocks) and a per-epoch 'order'; the traces repeat")
    ap.add_argument("--group-order", default=None,
                    help="comma-separated trace cohorts, e.g. 'training,bank': every epoch shows the cohorts in this "
                         "order (each shuffled) with aux rows spread evenly; needs --epochs; every cohort must be listed")
    ap.add_argument("--trace-tag", default=None, help="own first line of every trace's reasoning, e.g. '<alpha>'")
    ap.add_argument("--fixed-trace-order", action="store_true",
                    help="three-source layout: one trace permutation in every epoch (repeats exactly an epoch apart)")
    ap.add_argument("--aux-all", action="store_true",
                    help="three-source layout: every aux row of --aux-cache, split into --epochs disjoint blocks")
    ap.add_argument("--cot", nargs="+", default=None,
                    help="three-source layout: raw CoT jsonl file(s), split into --epochs disjoint blocks")
    ap.add_argument("--cot-repeat", action="store_true",
                    help="with --cot: every CoT row in every epoch (stored once, reshuffled each epoch with the "
                         "traces) instead of disjoint per-epoch blocks")
    ap.add_argument("--exclude-ids", nargs="+", default=None,
                    help="file(s) of task ids (one per line) whose traces and CoT rows are dropped")
    ap.add_argument("--workers", type=int, default=0,
                    help="CoT rows: check and count tokens in N processes (same output; 0 = serial, the default)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if not 0 <= args.aux_frac < 1:
        raise SystemExit("--aux-frac must be in [0, 1)")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    three = bool(args.cot or args.aux_all or args.fixed_trace_order)
    if not args.traces and not (args.cot or args.aux_all):
        raise SystemExit("without --traces the cache needs --cot and/or --aux-all")
    if args.cot_repeat and not args.cot:
        raise SystemExit("--cot-repeat needs --cot")
    if three and args.group_order:
        raise SystemExit("--group-order does not combine with the three-source layout")
    excluded = set()
    for path in args.exclude_ids or []:
        excluded |= {ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    fps = eval_fingerprints(args.eval_dir)
    t0 = time.time()

    rows, dropped_long, dropped_leak, bad, dropped_excluded = [], [], [], [], []
    arms, cohorts = Counter(), Counter()
    sources = []
    for path in args.traces:
        sources.append({"path": os.path.abspath(path), "sha256": sha256_file(path)})
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                t = json.loads(line)
                pm = t.get("prompt_messages") or []
                if ([m.get("role") for m in pm] != ["system", "user"] or not t.get("body")
                        or not t.get("answer")):
                    bad.append(f"{os.path.basename(path)}:{line_no}")
                    continue
                if t.get("arm_id") in ar.ARMS and pm[0]["content"] != ar.system_prompt(t["arm_id"]):
                    bad.append(f"{os.path.basename(path)}:{line_no} (system prompt differs from arm {t['arm_id']})")
                    continue
                if t["task_id"] in excluded or t["task_id"].split("arc2-training__")[-1] in excluded:
                    dropped_excluded.append(t["task_id"])
                    continue
                hits = leak_hits(pm[1]["content"], t["task_id"], fps)
                if hits:
                    dropped_leak.append({"task_id": t["task_id"], "eval_ids": hits})
                    continue
                messages = [dict(pm[0]), dict(pm[1]),
                            {"role": "assistant", "content": assistant_text(t, args.trace_tag)}]
                n = token_length(tok, messages)
                if n > args.max_seq_length:
                    dropped_long.append(n)
                    continue
                prov = f"trace|{t.get('cohort')}|{t['task_id']}|{t.get('arm_id')}|s{t.get('slot')}"
                rows.append({"messages": messages, "length": n, "prov": prov})
                arms[t.get("arm_id")] += 1
                cohorts[t.get("cohort")] += 1
    n_traces = len(rows)
    if n_traces == 0 and args.traces:
        raise SystemExit("no usable traces")

    n_aux = 0
    aux_meta = {}
    aux_blocks = None       # three-source layout: per-epoch lists of row indices
    aux_leftover = 0
    if args.aux_all or args.aux_frac > 0:
        if not args.aux_cache:
            raise SystemExit("--aux-frac > 0 / --aux-all needs --aux-cache")
        with open(args.aux_cache, encoding="utf-8") as f:
            payload = json.load(f)
        aux_all = payload["examples"] if isinstance(payload, dict) else payload
        aux_meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if int(aux_meta.get("max_seq_length", args.max_seq_length)) > args.max_seq_length:
            print(f"note: aux cache was built at max_seq_length={aux_meta.get('max_seq_length')}; "
                  f"rows over {args.max_seq_length} are skipped")
        pool = [r for r in aux_all if int(r.get("length") or 0) <= args.max_seq_length]
        want = len(pool) // args.epochs if args.aux_all else round(n_traces * args.aux_frac / (1 - args.aux_frac))
        aux_leftover = len(pool) - want * args.epochs if args.aux_all else 0
        if want * args.epochs > len(pool):
            raise SystemExit(f"need {want} x {args.epochs} epochs = {want * args.epochs} aux rows <= "
                             f"{args.max_seq_length} tokens, cache has {len(pool)}")
        rng = random.Random(args.seed)
        for i, r in enumerate(rng.sample(pool, want * args.epochs)):
            block = f"e{i // want}|" if args.epochs > 1 else ""
            rows.append({"messages": r["messages"], "length": int(r["length"]),
                         "prov": "aux|" + block + str(r.get("prov", ""))})
        n_aux = want
        del aux_all, payload, pool
        if three:
            aux_blocks = [list(range(n_traces + e * want, n_traces + (e + 1) * want)) for e in range(args.epochs)]

    # raw CoT rows (three-source layout): validated, leak- and exclusion-checked, capped, split into epoch blocks
    n_cot, cot_blocks, cot_sources, cot_leftover = 0, None, [], 0
    cot_repeat_idx = []     # --cot-repeat: CoT row indices shown in every epoch
    cot_drop = {"malformed": 0, "too_long": [], "excluded": 0, "eval_overlap": []}
    if args.cot:
        cfps = compact_fingerprints(args.eval_dir)
        cot_rows = []
        _COT.update(tok=tok, fps=fps, cfps=cfps, excluded=excluded, max_len=args.max_seq_length, tokenizer=args.tokenizer)
        for path in args.cot:
            cot_sources.append({"path": os.path.abspath(path), "sha256": sha256_file(path)})
            jobs = ((os.path.basename(path), line_no, line) for line_no, line in
                    enumerate(open(path, encoding="utf-8"), 1) if line.strip())
            if args.workers > 0:     # opt-in: same results in the same order, rows checked in parallel
                import multiprocessing
                pool = multiprocessing.get_context("fork").Pool(args.workers, initializer=_cot_worker_init)
                results = pool.imap(_cot_one, jobs, chunksize=16)
            else:
                pool, results = None, map(_cot_one, jobs)
            for kind, val in results:
                if kind == "malformed":
                    cot_drop["malformed"] += 1
                elif kind == "excluded":
                    cot_drop["excluded"] += 1
                elif kind == "eval_overlap":
                    cot_drop["eval_overlap"].append(val)
                elif kind == "too_long":
                    cot_drop["too_long"].append(val)
                else:
                    cot_rows.append(val)
            if pool is not None:
                pool.close(); pool.join()
        random.Random(args.seed + 7).shuffle(cot_rows)
        parts, rest = ([cot_rows], []) if args.cot_repeat else split_blocks(cot_rows, args.epochs)
        cot_leftover = len(rest)
        n_cot = len(parts[0])
        start = len(rows)
        cot_blocks = []
        for e, part in enumerate(parts):
            cot_blocks.append(list(range(start, start + len(part))))
            for r in part:
                if not args.cot_repeat:
                    r["prov"] = r["prov"].replace("cot|", f"cot|e{e}|", 1)
                rows.append(r)
            start += len(part)
        if args.cot_repeat:
            cot_repeat_idx, cot_blocks = cot_blocks[0], None

    order = None
    groups = None
    if args.group_order:
        names = [x.strip() for x in args.group_order.split(",") if x.strip()]
        cohort_of = [r["prov"].split("|")[1] for r in rows[:n_traces]]
        missing = sorted(set(cohort_of) - set(names))
        if missing:
            raise SystemExit(f"--group-order {names} does not list trace cohorts {missing}")
        groups = [[i for i, c in enumerate(cohort_of) if c == name] for name in names]
        print("group order: " + ", ".join(f"{n} {len(g)}" for n, g in zip(names, groups)))
    if three:
        order = block_orders(n_traces, [b for b in (aux_blocks, cot_blocks) if b], args.epochs, args.seed,
                             args.fixed_trace_order, cot_repeat_idx)
        epoch_rows = [rows[i] for i in order[0]]
    elif args.epochs > 1 or groups is not None:
        order = epoch_orders(n_traces, n_aux, args.epochs, args.seed, groups)
        epoch_rows = [rows[i] for i in order[0]]  # stats below describe one epoch (epoch 0)
    else:
        random.Random(args.seed + 1).shuffle(rows)
        epoch_rows = rows
    lengths = sorted(r["length"] for r in epoch_rows)
    trace_tokens = sum(r["length"] for r in epoch_rows if r["prov"].startswith("trace|"))
    cot_tokens = sum(r["length"] for r in epoch_rows if r["prov"].startswith("cot|"))
    meta = {
        "stage": 2,
        "max_seq_length": args.max_seq_length,
        "n_examples": len(rows),
        "epochs": args.epochs,
        "rows_per_epoch": n_traces + n_aux + n_cot,
        "n_traces": n_traces,
        "n_aux": n_aux,
        "n_aux_total": n_aux * args.epochs,
        "aux_frac": args.aux_frac,
        "aux_cache": os.path.abspath(args.aux_cache) if args.aux_cache else None,
        "aux_cache_generator_digest": aux_meta.get("generator_digest"),
        "sources": sources,
        "arms": dict(arms),
        "cohorts": dict(cohorts),
        "dropped_too_long": len(dropped_long),
        "dropped_too_long_max": max(dropped_long) if dropped_long else None,
        "dropped_eval_overlap": dropped_leak,
        "dropped_excluded": len(dropped_excluded),
        "layout": "three-source" if three else ("groups" if groups is not None else "epochs"),
        "trace_tag": args.trace_tag,
        "fixed_trace_order": bool(args.fixed_trace_order),
        "aux_all": bool(args.aux_all),
        "aux_leftover_dropped": aux_leftover,
        "n_cot": n_cot,
        "n_cot_total": n_cot if args.cot_repeat else n_cot * args.epochs,
        "cot_repeat": bool(args.cot_repeat),
        "cot_sources": cot_sources,
        "cot_dropped": {"malformed": cot_drop["malformed"], "excluded": cot_drop["excluded"],
                        "eval_overlap": cot_drop["eval_overlap"], "too_long": len(cot_drop["too_long"]),
                        "too_long_max": max(cot_drop["too_long"]) if cot_drop["too_long"] else None,
                        "leftover": cot_leftover},
        "tokens_cot": cot_tokens,  # per epoch (epoch 0)
        "exclude_ids": [os.path.abspath(x) for x in args.exclude_ids or []],
        "malformed": bad,
        "tokens_total": sum(lengths),  # per epoch (epoch 0 when epochs > 1)
        "tokens_traces": trace_tokens,
        "length_p50": lengths[len(lengths) // 2],
        "length_max": lengths[-1],
        "tokenizer": args.tokenizer,
        "assistant_format": ("<think>\\n{body}\\n</think>\\n\\n{answer}" if not any(
            a in ar.OUTPUT_CHANNEL_ARMS for a in arms) else
            "<think>\\n\\n</think>\\n\\n{body}\\n\\n" + ar.ANSWER_MARKER + "\\n{answer}"),
        "group_order": args.group_order,
        "seed": args.seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        payload = {"examples": rows, "metadata": meta}
        if order is not None:
            payload["order"] = order
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, args.output)
    with open(args.output + ".report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if args.epochs > 1:
        print(f"{args.epochs} epochs: {n_traces + n_aux} rows per epoch ({n_traces} traces repeated + {n_aux} fresh aux); "
              f"{len(rows)} unique rows stored ({n_aux * args.epochs} aux); stats below are per epoch")
    if three:
        print(f"three-source layout, {args.epochs} epochs: {n_traces} traces (tag {args.trace_tag!r}, "
              f"{'fixed' if args.fixed_trace_order else 'reshuffled'} order) + {n_aux} aux + {n_cot} CoT rows per epoch; "
              f"per epoch tokens: traces {trace_tokens / 1e6:.1f}M, CoT {cot_tokens / 1e6:.1f}M, "
              f"aux {(sum(lengths) - trace_tokens - cot_tokens) / 1e6:.1f}M; dropped aux leftover {aux_leftover}, "
              f"CoT {cot_drop['malformed']} malformed / {cot_drop['excluded']} excluded / "
              f"{len(cot_drop['eval_overlap'])} eval-overlap / {len(cot_drop['too_long'])} too long / {cot_leftover} leftover; "
              f"{len(dropped_excluded)} traces excluded")
    print(f"wrote {args.output}: {len(epoch_rows)} rows = {n_traces} traces + {n_aux} aux "
          f"({100 * n_aux / len(epoch_rows):.1f}% rows, {100 * (1 - trace_tokens / max(1, sum(lengths))):.1f}% tokens); "
          f"{sum(lengths) / 1e6:.2f}M tokens, median {meta['length_p50']}, max {meta['length_max']}; "
          f"dropped {len(dropped_long)} too long (> {args.max_seq_length}), {len(dropped_leak)} eval-overlap, "
          f"{len(bad)} malformed; arms {dict(arms)}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
