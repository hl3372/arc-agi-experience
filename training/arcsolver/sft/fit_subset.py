#!/usr/bin/env python
"""Memory fit check data: a tokenized dataset (tokenize_cache.py format) holding only the K longest rows of DATA, one
epoch, longest first. train_fsdp.py with --global-rows K on K ranks then gives every rank one of the longest rows in
its first step: the per-rank peak (gathered parameters + gradients + one row's activations + the optimizer step) that
the real run will hit. Run it with the real run's memory/kernel flags and --save-every 1 to also test a save + export.

    python -m arcsolver.sft.fit_subset --data DATA --out FIT_DIR [--rows 8]
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--first", action="store_true",
                    help="the first K rows of epoch 0's order instead of the K longest (numerics A/B tests)")
    ap.add_argument("--max-len", type=int, default=None, help="with --first: only rows of at most this many tokens")
    a = ap.parse_args()
    offsets = np.load(os.path.join(a.data, "offsets.npy"))
    label_start = np.load(os.path.join(a.data, "label_start.npy"))
    source = np.load(os.path.join(a.data, "source.npy"))
    ids = np.memmap(os.path.join(a.data, "ids.u32"), dtype=np.uint32, mode="r")
    lengths = np.diff(offsets)
    if a.first:
        first = np.load(os.path.join(a.data, "order.npy"))[0].astype(np.int64)
        if a.max_len:
            first = first[lengths[first] <= a.max_len]
        pick = first[:a.rows]
    else:
        pick = np.argsort(-lengths, kind="stable")[:a.rows]
    os.makedirs(a.out, exist_ok=True)
    new_off = np.zeros(len(pick) + 1, dtype=np.int64)
    new_off[1:] = np.cumsum(lengths[pick])
    with open(os.path.join(a.out, "ids.u32"), "wb") as f:
        for i in pick:
            f.write(np.asarray(ids[offsets[i]:offsets[i + 1]], dtype=np.uint32).tobytes())
    np.save(os.path.join(a.out, "offsets.npy"), new_off)
    np.save(os.path.join(a.out, "label_start.npy"), label_start[pick].astype(label_start.dtype))
    np.save(os.path.join(a.out, "source.npy"), source[pick].astype(source.dtype))
    np.save(os.path.join(a.out, "order.npy"), np.arange(len(pick), dtype=np.int32)[None, :])
    meta = {"fit_check_of": os.path.abspath(a.data), "n_rows": int(len(pick)), "epochs": 1,
            "rows_per_epoch": int(len(pick)), "tokens_stored": int(new_off[-1]),
            "lengths": [int(x) for x in lengths[pick]], "rows": [int(x) for x in pick]}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("FIT_SUBSET", json.dumps({k: meta[k] for k in ("n_rows", "tokens_stored", "lengths")}))


if __name__ == "__main__":
    main()
