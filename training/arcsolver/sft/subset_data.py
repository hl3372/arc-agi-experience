#!/usr/bin/env python
"""Length-capped copy of a tokenized SFT dataset (tokenize_cache.py format), with the epoch orders extended.
  * rows longer than --max-len are dropped (never truncated); the kept rows keep their ids / label starts / sources;
  * each existing epoch's order keeps its sequence of kept rows (so runs compare with the uncapped order);
  * epochs beyond the source's are new permutations of the kept rows, seed = --base-seed + epoch;
  * meta.json records the cap, the dropped rows, the length distribution and per-epoch token counts.
The source order must show every row once per epoch (build_cache.py --cot-repeat layout).
    python -m arcsolver.sft.subset_data --data SRC_DIR --out OUT_DIR --max-len 163840 [--epochs 10] [--base-seed 20261025]
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="total epochs in the new order (default: the source's)")
    ap.add_argument("--base-seed", type=int, default=20261025)
    a = ap.parse_args()
    offsets = np.load(os.path.join(a.data, "offsets.npy"))
    label_start = np.load(os.path.join(a.data, "label_start.npy"))
    source = np.load(os.path.join(a.data, "source.npy"))
    order = np.load(os.path.join(a.data, "order.npy"))
    ids = np.memmap(os.path.join(a.data, "ids.u32"), dtype=np.uint32, mode="r")
    src_meta = json.load(open(os.path.join(a.data, "meta.json")))
    lengths = np.diff(offsets)
    keep = np.flatnonzero(lengths <= a.max_len)
    old2new = np.full(len(lengths), -1, dtype=np.int64)
    old2new[keep] = np.arange(len(keep))
    os.makedirs(a.out, exist_ok=True)
    new_off = np.zeros(len(keep) + 1, dtype=np.int64)
    new_off[1:] = np.cumsum(lengths[keep])
    with open(os.path.join(a.out, "ids.u32.part"), "wb") as f:
        for i in keep:
            f.write(np.asarray(ids[offsets[i]:offsets[i + 1]], dtype=np.uint32).tobytes())
    os.replace(os.path.join(a.out, "ids.u32.part"), os.path.join(a.out, "ids.u32"))
    E = a.epochs or order.shape[0]
    new_order = np.zeros((E, len(keep)), dtype=np.int32)
    for e in range(E):
        if e < order.shape[0]:
            o = old2new[order[e]]
            o = o[o >= 0]
        else:
            o = np.random.default_rng(a.base_seed + e).permutation(len(keep))
        assert len(o) == len(keep) and len(np.unique(o)) == len(keep), e
        new_order[e] = o
    np.save(os.path.join(a.out, "offsets.npy"), new_off)
    np.save(os.path.join(a.out, "label_start.npy"), label_start[keep].astype(label_start.dtype))
    np.save(os.path.join(a.out, "source.npy"), source[keep].astype(source.dtype))
    np.save(os.path.join(a.out, "order.npy"), new_order)
    kl = lengths[keep]
    labels = kl - label_start[keep]
    meta = {k: v for k, v in src_meta.items() if k not in ("tokens_per_epoch", "label_tokens_per_epoch_by_source")}
    meta.update({"subset_of": os.path.abspath(a.data), "max_len_cap": a.max_len, "n_rows": int(len(keep)),
                 "rows_dropped": int(len(lengths) - len(keep)), "dropped_rows": [int(x) for x in np.flatnonzero(lengths > a.max_len)],
                 "epochs": int(E), "rows_per_epoch": int(len(keep)), "tokens_stored": int(new_off[-1]),
                 "tokens_per_epoch": [int(new_off[-1])] * E, "label_tokens_per_epoch": int(labels.sum()),
                 "max_len": int(kl.max()), "epoch_seeds_beyond_source": {str(e): a.base_seed + e for e in range(order.shape[0], E)},
                 "length_pct": {str(q): int(np.percentile(kl, q)) for q in (1, 50, 90, 99)}})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("SUBSET", json.dumps({k: meta[k] for k in ("n_rows", "rows_dropped", "epochs", "tokens_stored", "max_len", "length_pct")}))


if __name__ == "__main__":
    main()
