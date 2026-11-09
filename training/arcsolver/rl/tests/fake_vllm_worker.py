#!/usr/bin/env python
"""Stand-in for arcsolver/rl/vllm_worker.py in CPU tests (stdlib only; the multi-node sampler tests).

Same CLI (unknown arguments are ignored; --socket and --policy-id are honoured), same transport (AF_UNIX
multiprocessing.connection, authkey from $RL_ENGINE_AUTHKEY), same commands and message shapes as the real worker: info,
generate, stream (continuous refill incl. heartbeat / hold / abort and the optional inline forced answers), score,
reload, sleep, wake, digest, free_host_cache, cuda_mem, shutdown.

Samples are a pure function of (prompt token ids, seed, policy, max_tokens): never of which engine or node runs them, so
placement-independence can be checked bit for bit. Sample j of a generate() request with seed s is the n=1 sample with
seed s + j (how vLLM seeds the children of an n > 1 request, and what sampler_dispatch.generate_stream relies on).

Fault injection (read at every command, so a test can change it while the engine runs; later sources win):
  environment   FAKE_RELOAD_FAIL=1, FAKE_ACK_POLICY=<id>, FAKE_CRASH_AFTER=<samples>, FAKE_CRASH_ON=<cmd>,
                FAKE_START_DELAY=<s>, FAKE_STEP_S=<s per decode step>, FAKE_PDEATHSIG=0 (default 1, as the real worker)
  files         $FAKE_CTL_DIR/all.json, then $FAKE_CTL_DIR/<socket basename without .sock>.json, e.g.
                {"ack_policy": "STALE"}, {"reload_fail": true}, {"crash_after_samples": 2}, {"crash_on": "reload"},
                {"step_s": 0.01}
  crash_after_samples K: the process exits (code 3, no reply) instead of producing its (K+1)-th sample;
  crash_on CMD: the process exits (code 4) when it receives CMD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from multiprocessing.connection import Listener

TOKENS_PER_STEP = 4          # simulated decode: a sample of L tokens finishes after ceil(L / 4) engine steps
VOCAB = 1000


def _rng(*parts) -> random.Random:
    h = hashlib.sha256(json.dumps(parts, separators=(",", ":"), default=str).encode()).digest()
    return random.Random(int.from_bytes(h[:16], "big"))


def sample(prompt, seed, policy, max_tokens: int, params: dict | None = None) -> dict:
    """The record a sampler of `policy` produces for `prompt` with `seed` (deterministic; placement-free)."""
    params = params or {}
    max_tokens = max(1, int(max_tokens))
    r = _rng("sample", [int(t) for t in prompt], seed, policy, max_tokens, params.get("temperature"),
             params.get("top_p"), params.get("top_k", 0))
    length = r.randint(1, max_tokens + max_tokens // 4)          # about 20% of the samples hit the cap
    length = min(length, max_tokens)
    tids = [r.randrange(VOCAB) for _ in range(length)]
    finish = "length" if length == max_tokens else "stop"
    rec = {"token_ids": tids, "finish": finish, "text": " ".join(f"t{t}" for t in tids),
           "stop_reason": None if finish == "length" else 151645, "logprobs": None, "policy": policy}
    if params.get("logprobs", True):
        rec["logprobs"] = [-round(0.01 + 3.0 * r.random(), 6) for _ in tids]
    return rec


def score_logprobs(seq, start: int, policy) -> list:
    """log p(seq[i] | seq[:i]) for i >= start under `policy` (deterministic, finite, <= 0)."""
    return [-round(0.001 + 4.0 * _rng("score", [int(t) for t in seq[:i + 1]], policy).random(), 6)
            for i in range(start, len(seq))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--policy-id", default="")
    ap.add_argument("--role", default="sampler")
    ap.add_argument("--model", default="(fake)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args, _unknown = ap.parse_known_args()
    authkey = bytes.fromhex(os.environ["RL_ENGINE_AUTHKEY"])
    name = os.path.basename(args.socket)
    name = name[:-5] if name.endswith(".sock") else name

    def _die(*_a):                                     # as vllm_worker: SIGTERM kills the whole process group
        os.killpg(os.getpgid(0), signal.SIGKILL)

    signal.signal(signal.SIGTERM, _die)
    if os.environ.get("FAKE_PDEATHSIG", "1") != "0":
        try:
            import ctypes
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
        except (OSError, AttributeError):
            pass

    def ctl() -> dict:
        c = {}
        env = os.environ
        if env.get("FAKE_RELOAD_FAIL", "0") not in ("", "0"):
            c["reload_fail"] = True
        if env.get("FAKE_ACK_POLICY"):
            c["ack_policy"] = env["FAKE_ACK_POLICY"]
        if env.get("FAKE_CRASH_AFTER"):
            c["crash_after_samples"] = int(env["FAKE_CRASH_AFTER"])
        if env.get("FAKE_CRASH_ON"):
            c["crash_on"] = env["FAKE_CRASH_ON"]
        if env.get("FAKE_STEP_S"):
            c["step_s"] = float(env["FAKE_STEP_S"])
        d = env.get("FAKE_CTL_DIR")
        if d:
            for fn in ("all.json", f"{name}.json"):
                try:
                    with open(os.path.join(d, fn)) as f:
                        c.update(json.load(f))
                except (OSError, ValueError):
                    pass
        return c

    t0 = time.time()
    start_delay = float(os.environ.get("FAKE_START_DELAY") or ctl().get("start_delay_s") or 0.0)
    time.sleep(start_delay)                            # "model load"
    load_s = time.time() - t0
    print(f"[{args.role}] fake engine ready in {load_s:.1f}s: {args.model} policy {args.policy_id!r}", flush=True)
    state = {"policy": args.policy_id or None, "awake": {"weights", "kv_cache"}, "sent": 0, "weights_path": None}

    def produce(prompt, seed, max_tokens, params):
        """One sample of the current policy; honours crash_after_samples."""
        k = ctl().get("crash_after_samples")
        if k is not None and state["sent"] >= int(k):
            print(f"[{args.role}] injected crash after {state['sent']} samples", flush=True)
            os._exit(3)
        state["sent"] += 1
        rec = sample(prompt, seed, state["policy"], max_tokens, params)
        rec["fake_engine"] = name                       # metadata only (tests strip it before comparing)
        return rec

    def need_awake(what):
        missing = {"weights", "kv_cache"} - state["awake"]
        if missing:
            raise RuntimeError(f"{what} while {sorted(missing)} sleep")

    def check_params(p):
        if int(p.get("topk") or 0):
            raise ValueError("fake worker: top-k logprob files are not simulated")

    def stream(msg):
        """vllm_worker.stream's control loop, with a simulated engine step instead of vLLM's."""
        need_awake("stream")
        p = msg["params"]
        check_params(p)
        target = max(1, int(msg["target_seqs"]))
        heartbeat_s = float(msg.get("heartbeat_s") or 0.0)
        force = msg.get("force") or None
        conn = state["conn"]
        pending = {}
        rid_next = [0]
        t0 = time.time()
        gen, done, asked, exhausted = 0, 0, False, False
        n_forced, forced_gen, forced_cached, forced_prompt = 0, 0, 0, 0
        forced_inflight = [0]
        aborted, last_ask = 0, time.time()

        def enqueue(entry, n_tokens):
            entry["steps"] = max(1, math.ceil(n_tokens / TOKENS_PER_STEP))
            pending[rid_next[0]] = entry
            rid_next[0] += 1

        def add(items):
            for it in items or []:
                rec = produce(it["prompt"], it.get("seed"), it["max_tokens"], p)
                enqueue({"it": it, "rec": rec}, len(rec["token_ids"]))

        def add_forced(it, rec):
            fp = list(it["prompt"]) + rec["token_ids"] + list(force["suffix_ids"])
            room = int(force["max_model_len"]) - len(fp)
            if room < 64:
                rec["forced_skipped"] = True
                return False
            seed = it.get("seed")
            fseed = None if seed is None else (int(seed) * 7919 + int(force.get("seed_salt", 13))) % (2 ** 31)
            frec = sample(fp, fseed, state["policy"], min(int(force["budget"]), room), force["params"])
            enqueue({"forced_of": it, "rec": rec, "frec": frec, "n_prompt": len(fp)}, len(frec["token_ids"]))
            forced_inflight[0] += 1
            return True

        def step():
            time.sleep(float(ctl().get("step_s", 0.002)))
            finished = []
            for rid in list(pending):
                pending[rid]["steps"] -= 1
                if pending[rid]["steps"] <= 0:
                    finished.append(pending.pop(rid))
            return finished

        held = False
        add(msg.get("items") or [])
        while True:
            want = target - (len(pending) - forced_inflight[0])
            room = not exhausted and want > 0
            due = heartbeat_s > 0 and time.time() - last_ask >= heartbeat_s
            if not asked and ((room and (not held or due)) or (pending and due)):
                conn.send({"type": "need", "n": want if room else 0})
                asked, last_ask = True, time.time()
            if asked and (not pending or conn.poll(0)):
                reply = conn.recv()                      # blocks only when nothing is decoding
                asked = False
                if reply.get("abort"):
                    aborted = len(pending)
                    pending.clear()
                    break
                held = bool(reply.get("hold"))
                if reply.get("items"):
                    add(reply["items"])
                elif not held:
                    exhausted = True
                continue
            if not pending:
                if held and not exhausted:
                    time.sleep(0.05)
                    continue
                break
            for ent in step():
                if "forced_of" in ent:
                    forced_inflight[0] -= 1
                    src, rec, frec = ent["forced_of"], ent["rec"], ent["frec"]
                    cached = len(src["prompt"]) + len(rec["token_ids"])
                    rec["forced"] = {"token_ids": frec["token_ids"], "finish": frec["finish"], "text": frec["text"],
                                     "cached_tokens": cached}
                    n_forced += 1
                    forced_gen += len(frec["token_ids"])
                    forced_cached += cached
                    forced_prompt += ent["n_prompt"]
                    conn.send({"type": "sample", "key": src["key"], "j": src["j"], "rec": rec})
                    continue
                it, rec = ent["it"], ent["rec"]
                gen += len(rec["token_ids"])
                done += 1
                if (force and rec["finish"] == "length" and rec["token_ids"]
                        and (force.get("keys") is None or it["key"] in force["keys"])
                        and add_forced(it, rec)):
                    continue
                conn.send({"type": "sample", "key": it["key"], "j": it["j"], "rec": rec})
        return {"type": "end", "seconds": time.time() - t0, "gen_tokens": gen, "samples": done,
                "kv_usage_max": 0.0, "aborted": aborted,
                **({"forced": n_forced, "forced_gen_tokens": forced_gen, "forced_cached_tokens": forced_cached,
                    "forced_prompt_tokens": forced_prompt} if force else {})}

    def generate(msg):
        need_awake("generate")
        p = msg["params"]
        check_params(p)
        prompts = msg["prompts"]
        seeds = msg.get("seeds") or [None] * len(prompts)
        max_tokens = msg["max_tokens"]
        max_tokens = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        ns = msg.get("n_per_prompt") or [p["n"]] * len(prompts)
        if len(ns) != len(prompts) or any(not isinstance(n, int) or n < 1 for n in ns):
            raise ValueError("n_per_prompt must have one positive sample count per prompt")
        t = time.time()
        res, gen = [], 0
        for prompt, seed, mt, n in zip(prompts, seeds, max_tokens, ns):
            group = [produce(prompt, None if seed is None else seed + j, mt, p) for j in range(n)]
            gen += sum(len(r["token_ids"]) for r in group)
            res.append(group)
        return {"outputs": res, "seconds": time.time() - t, "gen_tokens": gen, "extract_seconds": 0.0}

    def score(msg):
        need_awake("score")
        res = []
        for s, start in zip(msg["seqs"], msg["starts"]):
            if not 0 < start < len(s):
                raise ValueError(f"start {start} outside (0, {len(s)})")
            res.append(score_logprobs(s, start, state["policy"]))
        return {"logprobs": res, "seconds": 0.0, "tokens": sum(len(s) for s in msg["seqs"])}

    def reload(msg):
        t = time.time()
        state["policy"] = None                         # unknown until the reload completed (as vllm_worker)
        if "weights" not in state["awake"]:
            raise RuntimeError("reload while the weights sleep")
        if not os.path.isdir(msg["weights_path"]):
            raise FileNotFoundError(msg["weights_path"])
        c = ctl()
        if c.get("reload_fail"):
            raise RuntimeError("injected reload failure")
        state["weights_path"] = msg["weights_path"]
        state["policy"] = c["ack_policy"] if "ack_policy" in c else msg.get("policy")
        return {"seconds": time.time() - t, "policy": state["policy"]}

    def digest(msg):
        return {"digests": {s: int(hashlib.sha256(f"{state['policy']}|{s}".encode()).hexdigest()[:15], 16)
                            for s in (msg.get("suffixes") or ("embed_tokens.weight",))}}

    def info(_msg):
        return {"role": args.role, "model": args.model, "vllm": "fake", "torch": None, "policy": state["policy"],
                "max_model_len": args.max_model_len, "load_seconds": load_s, "gpu_free_gb": 0.0, "gpu_total_gb": 0.0,
                "prefix_caching": True, "kv_cache_dtype": "auto", "quantization": "None", "speculative": None,
                "draft_sample_method": None, "num_gpu_blocks": None, "block_size": None,
                "fake": True, "engine": name, "pid": os.getpid(), "awake": sorted(state["awake"]),
                "weights_path": state["weights_path"], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def sleep(msg):
        state["awake"].clear()
        return {"seconds": 0.0, "level": int(msg.get("level", 2))}

    def wake(msg):
        tags = msg.get("tags")
        state["awake"].update(tags if tags is not None else ("weights", "kv_cache"))
        return {"seconds": 0.0}

    def free_host_cache(_msg):
        return {"seconds": 0.0, "released": [True]}

    def cuda_mem(_msg):
        snap = dict(torch_alloc_gb=0.0, torch_reserved_gb=0.0, device_free_gb=0.0, device_total_gb=0.0)
        return {"seconds": 0.0, "before": snap, "after": dict(snap)}

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
        if cmd and cmd == ctl().get("crash_on"):
            print(f"[{args.role}] injected crash on {cmd}", flush=True)
            os._exit(4)
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
    sys.exit(0)


if __name__ == "__main__":
    main()
