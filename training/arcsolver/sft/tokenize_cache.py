#!/usr/bin/env python
"""Pre-tokenize a row cache (build_cache.py output) for the multi-GPU trainer (train_fsdp.py).

Each row is rendered with the model's chat template, and its loss span is everything after the last
"<|im_start|>assistant\\n" (the assistant turn from "<think>" through the end-of-turn token). Every row must
tokenize to exactly the length the cache recorded (same tokenizer), else nothing is written. Outputs, in --out:
    ids.u32        all rows' token ids, concatenated (uint32)
    offsets.npy    int64 [n + 1] row boundaries into ids
    label_start.npy int32 [n] first label position of each row
    source.npy     int8 [n]: 0 trace (prov "trace|..."), 1 aux ("aux|..."), 2 CoT (anything else)
    order.npy      int32 [epochs, rows_per_epoch]: the cache's per-epoch row order
    meta.json      counts and token totals
    python -m arcsolver.sft.tokenize_cache --cache CACHE.json --tokenizer MODEL_DIR --out DIR [--workers 64]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

ROWS = None
TOK = None
MARK = None


def _init(tok_dir):
    global TOK, MARK
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(tok_dir)
    MARK = TOK.encode("<|im_start|>assistant\n", add_special_tokens=False)


def _one(i):
    r = ROWS[i]
    ids = TOK.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = list(ids)
    m = len(MARK)
    start = -1
    for j in range(len(ids) - m, -1, -1):
        if ids[j:j + m] == MARK:
            start = j + m
            break
    return i, np.asarray(ids, dtype=np.uint32), start


def main():
    global ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    payload = json.load(open(a.cache))
    ROWS = payload["examples"]
    order = payload["order"]
    meta_in = payload["metadata"]
    n = len(ROWS)
    print(f"loaded {n} rows, {len(order)} epochs x {len(order[0])} in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    src = np.array([0 if r["prov"].startswith("trace|") else 1 if r["prov"].startswith("aux|") else 2 for r in ROWS],
                   dtype=np.int8)
    lens = np.array([int(r["length"]) for r in ROWS], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    ids_mm = np.memmap(os.path.join(a.out, "ids.u32.part"), dtype=np.uint32, mode="w+", shape=(int(offsets[-1]),))
    label_start = np.zeros(n, dtype=np.int32)
    mismatch, nomark = [], []
    with Pool(a.workers, initializer=_init, initargs=(a.tokenizer,)) as pool:
        for k, (i, ids, start) in enumerate(pool.imap_unordered(_one, range(n), chunksize=64)):
            if len(ids) != lens[i]:
                mismatch.append((i, int(lens[i]), len(ids)))
                continue
            if start <= 0:
                nomark.append(i)
                continue
            ids_mm[offsets[i]:offsets[i + 1]] = ids
            label_start[i] = start
            if (k + 1) % 20000 == 0:
                print(f"  {k + 1}/{n} rows ({time.time() - t0:.0f}s)", flush=True)
    if mismatch or nomark:
        raise SystemExit(f"{len(mismatch)} rows tokenize to a different length than the cache records "
                         f"(first {mismatch[:3]}), {len(nomark)} without an assistant marker: refusing to write")
    ids_mm.flush()
    del ids_mm
    os.replace(os.path.join(a.out, "ids.u32.part"), os.path.join(a.out, "ids.u32"))
    np.save(os.path.join(a.out, "offsets.npy"), offsets)
    np.save(os.path.join(a.out, "label_start.npy"), label_start)
    np.save(os.path.join(a.out, "source.npy"), src)
    np.save(os.path.join(a.out, "order.npy"), np.asarray(order, dtype=np.int32))
    lab = lens - label_start
    meta = {"cache": os.path.abspath(a.cache), "tokenizer": a.tokenizer, "n_rows": n, "epochs": len(order),
            "rows_per_epoch": len(order[0]), "tokens_stored": int(offsets[-1]),
            "tokens_per_epoch": [int(lens[np.asarray(o)].sum()) for o in order],
            "label_tokens_per_epoch_by_source": [
                {s: int(lab[np.asarray(o)][src[np.asarray(o)] == c].sum()) for c, s in ((0, "trace"), (1, "aux"), (2, "cot"))}
                for o in order],
            "max_len": int(lens.max()), "cache_metadata": {k: v for k, v in meta_in.items() if k != "sources"},
            "seconds": round(time.time() - t0)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("n_rows", "epochs", "rows_per_epoch", "tokens_stored", "tokens_per_epoch",
                                           "label_tokens_per_epoch_by_source", "max_len", "seconds")}))


if __name__ == "__main__":
    main()
