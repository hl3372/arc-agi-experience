#!/usr/bin/env python
"""vLLM engine process for the RL trainer (run with the vLLM venv's python, not the training venv).

Standalone script: it imports nothing from arcsolver, so it runs in the vLLM environment. One process per engine,
role "sampler" (the policy: rollouts, eval samples, scoring probes, weight reloads). train.py (directly, or through
node_service.py on every node) spawns it through workers.EngineClient and talks to it over an AF_UNIX
multiprocessing.connection socket (authkey from $RL_ENGINE_AUTHKEY) with pickled plain-python messages, so the two
venvs (different torch builds) never share objects. Each request is a dict {"cmd": ..., ...}; each reply a dict, or
{"error": traceback} which the client re-raises.

Optional environment hooks (read at engine start):
  ARC_DRAFT_SAMPLE_METHOD       default of --draft-sample-method (MTP drafts: "greedy" | "probabilistic")
  ARC_REJECTION_SAMPLE_METHOD   default of --rejection-sample-method ("standard" | "block")
  ARC_GDN_PREFILL_BACKEND       vLLM gdn_prefill_backend for hybrid GatedDeltaNet models (flashinfer / triton / cutedsl)

Commands
  info                      engine facts (max_model_len, gpu memory, versions)
  generate                  prompts (token-id lists) + sampling params -> per prompt, n samples of
                            {token_ids, logprobs (sampled token, raw model distribution), finish, text}
  score                     seqs (token-id lists) + starts -> log p(seq[i] | seq[:i]) for i >= start
                            (prompt_logprobs=0, i.e. just the actual token; raw logits, T = 1)
  reload                    weights_path (a dir with *.safetensors in checkpoint format) ->
                            collective_rpc("reload_weights", weights_path=...) + reset_prefix_cache()
  sleep / wake              llm.sleep(level) / llm.wake_up(tags): time-share the GPU with the trainer (level 2
                            drops weights and KV; after wake(["weights"]) a reload refills them, then
                            wake(["kv_cache"]))
  generate params.topk = k  also writes per-sample top-k arrays (sampled-token log-probs, top-k ids and
                            log-probs) to msg["topk_dir"] as npz and returns their paths (score centering)
  digest                    exact bit fingerprints of a few weight tensors (tensor_digest), compared
                            with the trainer's after every sync
  stream                    continuous refill: items = single samples {key, j, prompt, seed,
                            max_tokens}; the engine keeps about target_seqs of them in flight, sends
                            {"type": "sample", ...} as each finishes and {"type": "need", "n": k} when it has room,
                            to which the client replies {"items": [...]} ([] = no more work); ends with the
                            final reply {"type": "end", ...}. Same requests and seeds as generate (a sample j of a
                            prompt with seed s is the n=1 request with seed s + j, which is how vLLM seeds the
                            children of an n > 1 request). Optional msg["force"] (train.py
                            --train-force-inline / --eval-force-inline): a sample cut at its token cap gets its forced answer
                            (prompt + sample + suffix_ids, up to `budget` new tokens, the forced sampling params)
                            queued at once on this engine, while the sample's KV and GDN states are still in the
                            prefix cache (engines with prefix caching); the sample is sent with rec["forced"] =
                            {token_ids, finish, cached_tokens} once its answer is done
  shutdown

Weight sync choice (vLLM 0.30.0): the trainer writes a bf16 safetensors file to /dev/shm and the
sampler calls the worker's built-in reload_weights(weights_path=...), which runs the model's own
load_weights with layerwise reload/processing (the same code as the initial load). Alternatives
considered: NCCL weight transfer needs two GPUs (NCCL refuses two ranks on one device); vLLM's
CUDA-IPC engine needs vLLM on the trainer side (a different torch build here); restarting the
engine costs minutes of compile/graph capture per step. The prefix cache is reset after every
reload, otherwise cached KV from the old weights would be reused.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener


def token_logprobs_from(container, token_ids, offset: int = 0) -> list:
    """Log-prob of token_ids[i] at position offset + i from a vLLM logprobs container (a
    FlatLogprobs or a list of {token_id: Logprob} dicts)."""
    out = []
    if hasattr(container, "start_indices"):          # FlatLogprobs
        s_idx, e_idx, toks, lps = container.start_indices, container.end_indices, container.token_ids, container.logprobs
        for i, tok in enumerate(token_ids):
            s, e = s_idx[offset + i], e_idx[offset + i]
            for j in range(s, e):
                if toks[j] == tok:
                    out.append(float(lps[j]))
                    break
            else:
                raise KeyError(f"token {tok} missing from logprobs at position {offset + i}")
    else:
        for i, tok in enumerate(token_ids):
            out.append(float(container[offset + i][tok].logprob))
    return out


def topk_arrays(container, token_ids, k: int):
    """From a FlatLogprobs container of a sample generated with logprobs=k: (sampled-token log-probs [T] float32,
    top-k ids [T, k] int32 padded with -1, top-k log-probs [T, k] float32 padded with -inf). vLLM stores, per
    position, the sampled token and then the top-k (k + 1 entries, the sampled token twice when it is among the
    top k); the distinct entries with rank <= k form the head. Vectorised: no per-entry Python loop."""
    import numpy as np
    T = len(token_ids)
    starts = np.asarray(container.start_indices[:T], dtype=np.int64)
    ends = np.asarray(container.end_indices[:T], dtype=np.int64)
    toks = np.asarray(container.token_ids, dtype=np.int64)
    lps = np.asarray(container.logprobs, dtype=np.float32)
    ranks = np.nan_to_num(np.asarray(container.ranks, dtype=np.float64), nan=1e18)   # None -> nan -> never head
    counts = ends - starts
    pos = np.repeat(np.arange(T), counts)
    if T and (starts[1:] == ends[:-1]).all():            # FlatLogprobs appends positions contiguously
        ent = np.arange(starts[0], ends[-1])
    else:
        ent = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if T else np.zeros(0, np.int64)
    sampled = np.asarray(token_ids, dtype=np.int64)
    hit = toks[ent] == sampled[pos]
    first = np.full(T, -1, dtype=np.int64)
    hit_ent = ent[hit][::-1]
    first[pos[hit][::-1]] = hit_ent                       # first matching entry per position wins (reversed write)
    if (first < 0).any():
        raise KeyError(f"sampled token missing from the logprobs at {int((first < 0).sum())} positions")
    lp_y = lps[first]
    head = ranks[ent] <= k
    hp, he = pos[head], ent[head]
    order = np.lexsort((ranks[he], hp))
    hp, he = hp[order], he[order]
    # FlatLogprobs lists the sampled token first and then the top-k, so a sampled token inside the top-k appears
    # twice with the same rank (almost every position); keep one entry per (position, token) so the head
    # is k distinct tokens.
    dup = np.zeros(len(hp), dtype=bool)
    if len(hp) > 1:
        tok_sorted = np.lexsort((toks[he], hp))
        a, b = hp[tok_sorted], toks[he][tok_sorted]
        rep = np.zeros(len(hp), dtype=bool)
        rep[1:] = (a[1:] == a[:-1]) & (b[1:] == b[:-1])
        dup[tok_sorted] = rep
    hp, he = hp[~dup], he[~dup]
    slot = np.arange(len(hp)) - np.searchsorted(hp, hp)    # rank order within each position
    keep = slot < k
    ids = np.full((T, k), -1, dtype=np.int32)
    out = np.full((T, k), -np.inf, dtype=np.float32)
    ids[hp[keep], slot[keep]] = toks[he[keep]]
    out[hp[keep], slot[keep]] = lps[he[keep]]
    return lp_y, ids, out


DIGEST_PARAMS = ("embed_tokens.weight", "layers.0.linear_attn.out_proj.weight",
                 "layers.3.self_attn.o_proj.weight", "layers.5.mlp.down_proj.weight",
                 "layers.30.mlp.down_proj.weight")


def tensor_digest(t) -> int:
    """Exact integer fingerprint of a 2-byte tensor's bits (bf16): sum_i bits_i * (i % 1021 + 1),
    in int64 chunks. Equal iff the tensors are (with overwhelming probability) bitwise equal; the
    same function runs in the trainer (torch 2.10) and in the vLLM worker (torch 2.13)."""
    import torch
    flat = t.detach().contiguous().view(torch.int16).flatten()
    total, step = 0, 1 << 24
    for s in range(0, flat.numel(), step):
        v = flat[s:s + step].to(torch.int64)
        w = torch.arange(s, s + v.numel(), device=v.device, dtype=torch.int64) % 1021 + 1
        total += int((v * w).sum())
    return total


def named_digests(named_params: dict, suffixes=DIGEST_PARAMS) -> dict:
    """{suffix: digest} for the parameter whose name is `suffix` or ends in "." + suffix."""
    out = {}
    for suf in suffixes:
        hits = [n for n in named_params if n == suf or n.endswith("." + suf)]
        out[suf] = tensor_digest(named_params[hits[0]]) if len(hits) == 1 else f"no unique match: {hits[:3]}"
    return out


def _cuda_mem_rpc(self, empty):           # runs inside vLLM's GPU worker process
    import gc
    import torch
    def snap():
        free, total = torch.cuda.mem_get_info()
        return dict(torch_alloc_gb=round(torch.cuda.memory_allocated() / 2**30, 2),
                    torch_reserved_gb=round(torch.cuda.memory_reserved() / 2**30, 2),
                    device_free_gb=round(free / 2**30, 2), device_total_gb=round(total / 2**30, 2))
    before = snap()
    if empty:
        gc.collect()
        torch.cuda.empty_cache()
    return dict(before=before, after=snap())


def _free_host_rpc(self):                  # runs inside vLLM's GPU worker process
    import torch
    fn = getattr(torch._C, "_host_emptyCache", None)
    if fn is not None:
        fn()
    return fn is not None


def _digest_rpc(self, suffixes):          # runs inside vLLM's GPU worker (self = the worker)
    return named_digests(dict(self.model_runner.get_model().named_parameters()), suffixes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["sampler"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    # With prefix caching on a hybrid GDN model, vLLM 0.30 keeps GDN ("mamba align")
    # states only at semantic checkpoints (prompt boundary) by default (0). "none" keeps one at every block boundary,
    # also during decode, so a forced answer queued as its rollout ends (--train-force-inline) hits the cache up to the
    # rollout's last block instead of only its prompt. A positive value = one state per that many tokens.
    ap.add_argument("--prefix-cache-retention-interval", default="",
                    help="'' = vLLM's default (0), 'none' = dense, or a multiple of the KV block size")
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=0,
                    help="0 = from --gpu-memory-utilization; else the KV pool size (vLLM kv_cache_memory_bytes)")
    # FP8 KV cache, MTP speculative decoding, sleep mode (time-sharing the GPU with the trainer),
    # top-k logprobs for score centering (max_logprobs, logprobs_mode), and no tokenizer work in the engine (the
    # caller decodes; top-k logprob tokens are then never detokenized).
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--mtp-tokens", type=int, default=0, help="MTP speculative tokens (0 = off)")
    ap.add_argument("--enable-sleep-mode", action="store_true")
    ap.add_argument("--max-logprobs", type=int, default=20)
    ap.add_argument("--logprobs-mode", default="processed_logprobs")
    ap.add_argument("--skip-tokenizer-init", action="store_true")
    ap.add_argument("--log-stats", action="store_true")
    ap.add_argument("--quantization", default="", help="'' = from the checkpoint's config (an FP8 checkpoint is FP8)")
    ap.add_argument("--block-size", type=int, default=0, help="KV block size (0 = vLLM's choice)")
    ap.add_argument("--attention-backend", default="", help="'' = vLLM's choice, e.g. FLASHINFER")
    # vLLM drafts greedily by default (draft = argmax q, verified as one-hot). "probabilistic" samples drafts from q
    # and runs the full p/q rejection test: still exact, and at T=1 it accepts more (about +20% tokens/s measured with
    # 3 MTP tokens on an FP8 4B model at 65k context). Opt-in; the env var lets a launcher set it without a trainer
    # flag.
    ap.add_argument("--draft-sample-method", default=os.environ.get("ARC_DRAFT_SAMPLE_METHOD", ""),
                    choices=["", "greedy", "probabilistic"], help="'' = vLLM's default (greedy)")
    # Block verification (Sun et al. 2024; vLLM V2 rejection sampler) is lossless like
    # token-wise verification and accepts at least as much. Opt-in, also through the env var like the draft method.
    ap.add_argument("--rejection-sample-method", default=os.environ.get("ARC_REJECTION_SAMPLE_METHOD", ""),
                    choices=["", "standard", "block"], help="'' = vLLM's default (standard)")
    # the policy identity (identity.policy_id of what --model / the last reload holds): stamped on every
    # sample record and switched only by a successful reload, so the caller can reject records of another policy
    ap.add_argument("--policy-id", default="", help="identity of the weights this engine starts with ('' = unknown)")
    args = ap.parse_args()
    authkey = bytes.fromhex(os.environ["RL_ENGINE_AUTHKEY"])
    # Never outlive the trainer: on SIGTERM, or when the trainer process dies (PR_SET_PDEATHSIG), kill
    # this engine's whole process group (it is the session leader; vLLM's EngineCore is in the group) so no
    # orphaned EngineCore keeps a GPU.
    import signal

    def _die(*_a):
        os.killpg(os.getpgid(0), signal.SIGKILL)

    signal.signal(signal.SIGTERM, _die)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm

    t0 = time.time()
    extra = {}
    if args.mtp_tokens > 0:
        extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.mtp_tokens}
        if args.draft_sample_method:
            extra["speculative_config"]["draft_sample_method"] = args.draft_sample_method
        if args.rejection_sample_method:
            extra["speculative_config"]["rejection_sample_method"] = args.rejection_sample_method
    if args.quantization:
        extra["quantization"] = args.quantization
    if args.block_size:
        extra["block_size"] = args.block_size
    if args.attention_backend:
        extra["attention_config"] = {"backend": args.attention_backend}
    if os.environ.get("ARC_GDN_PREFILL_BACKEND"):      # vLLM gdn_prefill_backend (flashinfer / triton / cutedsl)
        extra["gdn_prefill_backend"] = os.environ["ARC_GDN_PREFILL_BACKEND"]
    if args.prefix_cache_retention_interval:
        v = args.prefix_cache_retention_interval.lower()
        extra["prefix_cache_retention_interval"] = None if v == "none" else int(v)
    if args.kv_cache_memory_bytes:                  # benchmarks: a fixed KV pool size
        extra["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
              enforce_eager=args.enforce_eager, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              enable_prefix_caching=not args.no_prefix_caching,
              limit_mm_per_prompt={"image": 0, "video": 0}, kv_cache_dtype=args.kv_cache_dtype,
              enable_sleep_mode=args.enable_sleep_mode, max_logprobs=max(args.max_logprobs, 1),
              logprobs_mode=args.logprobs_mode, skip_tokenizer_init=args.skip_tokenizer_init,
              disable_log_stats=not args.log_stats, **extra)
    load_s = time.time() - t0
    print(f"[{args.role}] vLLM {vllm.__version__} ready in {load_s:.0f}s: {args.model}", flush=True)
    state = {"policy": args.policy_id or None}

    def sample_record(c, topk, path):
        tids = [int(x) for x in c.token_ids]
        rec = {"token_ids": tids, "finish": c.finish_reason, "text": c.text,
               "stop_reason": c.stop_reason if isinstance(c.stop_reason, (int, str, type(None))) else str(c.stop_reason),
               "logprobs": None, "policy": state["policy"]}
        if topk:
            import numpy as np
            lp_y, h_ids, h_lp = topk_arrays(c.logprobs, tids, topk)
            np.savez(path, y=np.asarray(tids, dtype=np.int32), lp_y=lp_y, ids=h_ids, lp=h_lp)
            rec["logprobs"] = lp_y.tolist()
            rec["topk_path"] = path
        elif c.logprobs is not None:
            rec["logprobs"] = token_logprobs_from(c.logprobs, tids)
        return rec

    def sampling_params(p, n, seed, max_tokens):
        topk = int(p.get("topk") or 0)
        lb = p.get("logit_bias")                            # train.py --eval-logit-bias
        if lb and args.mtp_tokens > 0:
            raise ValueError("logit_bias is silently ignored by vLLM with speculative decoding: start the "
                             "engines with --mtp-tokens 0")
        return SamplingParams(
            n=n, temperature=p["temperature"], top_p=p["top_p"], top_k=p.get("top_k", 0),
            min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=max_tokens, seed=seed, logprobs=topk if topk else (0 if p.get("logprobs") else None),
            flat_logprobs=True, detokenize=not args.skip_tokenizer_init,
            ignore_eos=bool(p.get("ignore_eos", False)),        # benchmarks only
            logit_bias={int(k): float(v) for k, v in lb.items()} if lb else None)

    def stream(msg):
        """Continuous refill (see the module docstring). Runs vLLM's own engine loop (LLM.enqueue + LLMEngine.step,
        exactly what LLM.generate does) so requests can be added while others decode."""
        p = msg["params"]
        topk = int(p.get("topk") or 0)
        topk_dir, tag = msg.get("topk_dir"), msg.get("tag", "g")
        if topk and not topk_dir:
            raise ValueError("topk needs topk_dir (where the per-sample arrays are written)")
        if topk:
            os.makedirs(topk_dir, exist_ok=True)       # a remote node's <topk_dir>/n<node> (workers.RemoteEngineClient)
        target = max(1, int(msg["target_seqs"]))
        kv_high = float(msg.get("kv_high", 0.9))
        # dynamic sampling (off unless the client sends heartbeat_s > 0): while samples decode, ask
        # at least every heartbeat_s (n = free slots, 0 when full), so the client can always answer; its reply
        # {"abort": true} aborts every in-flight sample and ends the command, {"hold": true} means "no work right
        # now but more may come" (not exhausted; the engine then re-asks only every heartbeat_s, and idles rather
        # than ending when nothing decodes). Strict request/reply: the client only ever answers a "need".
        heartbeat_s = float(msg.get("heartbeat_s") or 0.0)
        force = msg.get("force") or None
        conn = state["conn"]
        engine = llm.llm_engine
        pending = {}
        t0 = time.time()
        gen, done, asked, exhausted = 0, 0, False, False
        n_forced, forced_gen, forced_cached, forced_prompt = 0, 0, 0, 0
        forced_inflight = [0]        # forced answers run on top of target_seqs (short; their KV is the cached sample's)
        aborted, last_ask = 0, time.time()
        kv = {"t": 0.0, "usage": 0.0, "max": 0.0}

        def kv_full():
            """KV-cache usage (from vLLM's metrics, refreshed every 2 s) at or above kv_high: adding samples now
            would only make vLLM preempt (and later recompute) running ones."""
            if time.time() - kv["t"] > 2.0:
                kv["t"] = time.time()
                try:
                    for m in llm.get_metrics():
                        if m.name == "vllm:kv_cache_usage_perc":
                            kv["usage"] = float(m.value)
                            kv["max"] = max(kv["max"], kv["usage"])
                except Exception:  # noqa: BLE001 - metrics are advisory (off without --log-stats)
                    kv["usage"] = 0.0
            return kv["usage"] >= kv_high

        def add(items):
            if not items:
                return
            reqs = [TokensPrompt(prompt_token_ids=list(it["prompt"])) for it in items]
            params = [sampling_params(p, 1, it.get("seed"), it["max_tokens"]) for it in items]
            for rid, it in zip(llm.enqueue(reqs, params, use_tqdm=False), items):
                # enqueue returns vLLM's internal id ("<external>-<8 random chars>"); outputs carry the external one
                pending[rid.rsplit("-", 1)[0]] = it

        def add_forced(it, rec):
            """Queue the forced answer of a sample cut at its cap (see the module docstring); False if there is no room
            left in the context (as train.force_answers: room < 64 skips it)."""
            fp = list(it["prompt"]) + rec["token_ids"] + list(force["suffix_ids"])
            room = int(force["max_model_len"]) - len(fp)
            if room < 64:
                rec["forced_skipped"] = True
                return False
            seed = it.get("seed")
            fseed = None if seed is None else (int(seed) * 7919 + int(force.get("seed_salt", 13))) % (2 ** 31)
            sp = sampling_params(force["params"], 1, fseed, min(int(force["budget"]), room))
            rid = llm.enqueue([TokensPrompt(prompt_token_ids=fp)], [sp], use_tqdm=False)[0]
            pending[rid.rsplit("-", 1)[0]] = {"forced_of": it, "rec": rec, "n_prompt": len(fp)}
            forced_inflight[0] += 1
            return True

        def abort_all():
            """Abort every in-flight sample (dynamic sampling's stop) and drain the engine, so the next command
            starts with nothing queued. pending is keyed by the external request id (what abort_request takes)."""
            ids = list(pending)
            if ids:
                engine.abort_request(ids)
            pending.clear()
            while engine.has_unfinished_requests():
                engine.step()
            return len(ids)

        held = False                                     # the last reply was "hold": re-ask only every heartbeat_s
        add(msg.get("items") or [])
        while True:
            want = target - (len(pending) - forced_inflight[0])
            room = not exhausted and want > 0 and (not pending or not kv_full())
            due = heartbeat_s > 0 and time.time() - last_ask >= heartbeat_s
            if not asked and ((room and (not held or due)) or (pending and due)):
                conn.send({"type": "need", "n": want if room else 0})
                asked, last_ask = True, time.time()
            if asked and (not pending or conn.poll(0)):
                reply = conn.recv()                      # blocks only when nothing is decoding
                asked = False
                if reply.get("abort"):
                    aborted = abort_all()
                    break
                held = bool(reply.get("hold"))
                if reply.get("items"):
                    add(reply["items"])
                elif not held:
                    exhausted = True
                continue
            if not pending:
                if held and not exhausted:               # no work right now: idle until the client has some
                    time.sleep(0.2)
                    continue
                break
            for out in engine.step():
                if not out.finished:
                    continue
                it = pending.pop(out.request_id)
                c = out.outputs[0]
                if "forced_of" in it:                    # a forced answer: send its sample with it
                    forced_inflight[0] -= 1
                    src, rec = it["forced_of"], it["rec"]
                    ftids = [int(x) for x in c.token_ids]
                    cached = int(getattr(out, "num_cached_tokens", 0) or 0)
                    rec["forced"] = {"token_ids": ftids, "finish": c.finish_reason, "text": c.text,
                                     "cached_tokens": cached}
                    n_forced += 1
                    forced_gen += len(ftids)
                    forced_cached += cached
                    forced_prompt += it["n_prompt"]
                    conn.send({"type": "sample", "key": src["key"], "j": src["j"], "rec": rec})
                    continue
                path = os.path.join(topk_dir, f"{tag}_p{it['key']}_s{it['j']}.npz") if topk else None
                rec = sample_record(c, topk, path)
                gen += len(rec["token_ids"])
                done += 1
                if (force and rec["finish"] == "length" and rec["token_ids"]
                        and (force.get("keys") is None or it["key"] in force["keys"])   # keys: replay bench only
                        and add_forced(it, rec)):
                    continue                             # sent once its forced answer is done
                conn.send({"type": "sample", "key": it["key"], "j": it["j"], "rec": rec})
        return {"type": "end", "seconds": time.time() - t0, "gen_tokens": gen, "samples": done,
                "kv_usage_max": kv["max"], "aborted": aborted,
                **({"forced": n_forced, "forced_gen_tokens": forced_gen, "forced_cached_tokens": forced_cached,
                    "forced_prompt_tokens": forced_prompt} if force else {})}

    def generate(msg):
        p = msg["params"]
        prompts = msg["prompts"]
        seeds = msg.get("seeds") or [None] * len(prompts)
        max_tokens = msg["max_tokens"]
        max_tokens = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        topk = int(p.get("topk") or 0)
        topk_dir = msg.get("topk_dir")
        ns = msg.get('n_per_prompt') or [p['n']]*len(prompts)
        if len(ns)!=len(prompts) or any(not isinstance(n,int) or n<1 for n in ns):
            raise ValueError('n_per_prompt must have one positive sample count per prompt')
        if topk and not topk_dir:
            raise ValueError("topk needs topk_dir (where the per-sample arrays are written)")
        if topk:
            os.makedirs(topk_dir, exist_ok=True)
        reqs = [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompts]
        params = [sampling_params(p, n_samples, seed, mt) for seed, mt, n_samples in zip(seeds, max_tokens, ns)]
        t = time.time()
        outs = llm.generate(reqs, params, use_tqdm=False)
        dt = time.time() - t
        res, gen = [], 0
        t_x = time.time()
        for i, o in enumerate(outs):
            samples = []
            for j, c in enumerate(o.outputs):
                path = os.path.join(topk_dir, f"{msg.get('tag', 'g')}_{i}_{j}.npz") if topk else None
                rec = sample_record(c, topk, path)
                gen += len(rec["token_ids"])
                samples.append(rec)
            res.append(samples)
        return {"outputs": res, "seconds": dt, "gen_tokens": gen, "extract_seconds": time.time() - t_x}

    def score(msg):
        seqs, starts = msg["seqs"], msg["starts"]
        sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=0, prompt_logprobs=0,
                            detokenize=False, flat_logprobs=True)
        reqs = [TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
        t = time.time()
        outs = llm.generate(reqs, sp, use_tqdm=False)
        dt = time.time() - t
        res = []
        for o, s, start in zip(outs, seqs, starts):
            if not 0 < start < len(s):
                raise ValueError(f"start {start} outside (0, {len(s)})")
            res.append(token_logprobs_from(o.prompt_logprobs, s[start:], offset=start))
        return {"logprobs": res, "seconds": dt, "tokens": sum(len(s) for s in seqs)}

    def reload(msg):
        t = time.time()
        state["policy"] = None                       # unknown until the reload completed
        llm.collective_rpc("reload_weights", kwargs={"weights_path": msg["weights_path"]})
        llm.reset_prefix_cache()
        state["policy"] = msg.get("policy")
        return {"seconds": time.time() - t, "policy": state["policy"]}

    def digest(msg):
        return {"digests": llm.collective_rpc(_digest_rpc, args=(tuple(msg.get("suffixes") or DIGEST_PARAMS),))[0]}

    def info(_msg):
        import torch
        free, total = torch.cuda.mem_get_info()
        cfg = llm.llm_engine.vllm_config
        return {"role": args.role, "model": args.model, "vllm": vllm.__version__, "torch": torch.__version__,
                "policy": state["policy"],
                "max_model_len": cfg.model_config.max_model_len, "load_seconds": load_s,
                "gpu_free_gb": free / 2**30, "gpu_total_gb": total / 2**30,
                "prefix_caching": bool(cfg.cache_config.enable_prefix_caching),
                "kv_cache_dtype": str(cfg.cache_config.cache_dtype),
                "quantization": str(cfg.model_config.quantization),
                "speculative": (str(cfg.speculative_config.method) if cfg.speculative_config else None),
                "draft_sample_method": (str(getattr(cfg.speculative_config, "draft_sample_method", ""))
                                        if cfg.speculative_config else None),
                "num_gpu_blocks": getattr(cfg.cache_config, "num_gpu_blocks", None),
                "block_size": getattr(cfg.cache_config, "block_size", None)}

    def sleep(msg):
        t = time.time()
        llm.sleep(level=int(msg.get("level", 2)))
        return {"seconds": time.time() - t}

    def cuda_mem(msg):
        t = time.time()
        r = llm.collective_rpc(_cuda_mem_rpc, args=(bool(msg.get("empty", True)),))[0]
        return {"seconds": time.time() - t, **r}

    def free_host_cache(_msg):
        t = time.time()
        ok = llm.collective_rpc(_free_host_rpc)
        return {"seconds": time.time() - t, "released": ok}

    def wake(msg):
        """Before waking the KV cache, return the GPU memory torch cached in this engine (an in-place weight reload
        leaves temporaries there that vLLM's sleep allocator does not manage; the engine with the least headroom
        could then fail to re-map its KV cache). On an out-of-memory error, empty the cache and retry once."""
        t = time.time()
        tags = msg.get("tags")
        if tags is None or "kv_cache" in tags:
            llm.collective_rpc(_cuda_mem_rpc, args=(True,))
        try:
            llm.wake_up(tags=tags)
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower():
                raise
            print(f"[{args.role}] wake {tags} hit OOM; emptying the CUDA cache and retrying once", flush=True)
            time.sleep(5)
            mem = llm.collective_rpc(_cuda_mem_rpc, args=(True,))[0]
            print(f"[{args.role}] engine CUDA memory before the retry: {mem}", flush=True)
            llm.wake_up(tags=tags)
        return {"seconds": time.time() - t}

    handlers = {"generate": generate, "score": score, "reload": reload, "info": info, "digest": digest,
                "sleep": sleep, "wake": wake, "stream": stream, "free_host_cache": free_host_cache,
                "cuda_mem": cuda_mem}
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    listener = Listener(args.socket, family="AF_UNIX", authkey=authkey)
    print(f"[{args.role}] listening on {args.socket}", flush=True)
    conn = listener.accept()
    state["conn"] = conn
    while True:
        try:
            msg = conn.recv()
        except (EOFError, ConnectionResetError, OSError):
            print(f"[{args.role}] client went away; exiting", flush=True)
            break
        cmd = msg.get("cmd")
        if cmd == "shutdown":
            conn.send({"ok": True})
            break
        try:
            reply = handlers[cmd](msg)
        except Exception:  # noqa: BLE001 - reported to the client, which raises
            reply = {"error": traceback.format_exc()}
            print(f"[{args.role}] {cmd} failed:\n{reply['error']}", flush=True)
        conn.send(reply)
    conn.close()
    listener.close()
    del llm
    sys.exit(0)


if __name__ == "__main__":
    main()
