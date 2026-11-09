#!/usr/bin/env python
"""Dataset for a run restarted mid-way on the union of its data and new rows (every row every epoch from the restart
on). The trained part stays byte-identical:
  * rows 0..N_old-1 = the old dataset unchanged; then each --extra set's rows of at most --max-len tokens (tokenize_cache.py
    output, uncapped) with its source code (0 trace, 1 aux, 2 CoT, 3 hinted CoT: logged separately as loss_hint);
  * epochs < --keep-epochs keep the old order exactly; epoch --keep-epochs keeps only the first --partial-rows rows of its
    old order (what was trained of it; a multiple of the rows per step, so its steps are the trained ones); all padded
    with -1 to the new width (train_fsdp.py's build_plan skips -1);
  * then --new-epochs epochs, each a permutation of ALL rows (seed --seed + epoch);
  * checks: the old prefix of every array and of ids.u32 (sha256) equals the old dataset; meta.json records it all.
Resume the run on the new dataset with train_fsdp.py --lr-stretch-from <restart step> --lr-stretch-total <old total>
to stretch the rest of the original cosine over the new length.
    python -m arcsolver.sft.union_data --old OLD_DIR --extra DIR,SOURCE [--extra ...] --out OUT_DIR --max-len 163840 \\
        --keep-epochs 2 --partial-rows 16000 --new-epochs 5 [--seed 20261028]
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np


def sha_file(path, limit=None):
    h, left = hashlib.sha256(), limit
    with open(path, "rb") as f:
        while left is None or left > 0:
            b = f.read(1 << 24 if left is None else min(1 << 24, left))
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--extra", action="append", required=True, help="DIR,SOURCE (tokenize_cache.py output, in order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--keep-epochs", type=int, required=True)
    ap.add_argument("--partial-rows", type=int, required=True)
    ap.add_argument("--new-epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()
    ld = lambda d, n: np.load(os.path.join(d, n))
    o_off, o_ls, o_src, o_ord = ld(a.old, "offsets.npy"), ld(a.old, "label_start.npy"), ld(a.old, "source.npy"), ld(a.old, "order.npy")
    o_meta = json.load(open(os.path.join(a.old, "meta.json")))
    N = len(o_off) - 1
    K = a.keep_epochs
    assert 0 <= K < o_ord.shape[0] and 0 < a.partial_rows <= N
    assert all(sorted(o_ord[e].tolist()) == list(range(N)) for e in range(K + 1))
    extras, new_lens, new_ls, new_src = [], [], [], []
    os.makedirs(a.out, exist_ok=True)
    part = os.path.join(a.out, "ids.u32.part")
    shutil.copyfile(os.path.join(a.old, "ids.u32"), part)
    with open(part, "ab") as f:
        for spec in a.extra:
            d, src = spec.split(",")
            off, ls = ld(d, "offsets.npy"), ld(d, "label_start.npy")
            lens = np.diff(off)
            keep = np.flatnonzero(lens <= a.max_len)
            ids = np.memmap(os.path.join(d, "ids.u32"), dtype=np.uint32, mode="r")
            for i in keep:
                f.write(np.asarray(ids[off[i]:off[i + 1]], dtype=np.uint32).tobytes())
                new_lens.append(int(lens[i])); new_ls.append(int(ls[i])); new_src.append(int(src))
            extras.append({"dir": os.path.abspath(d), "source": int(src), "rows": int(len(lens)), "kept": int(len(keep)),
                           "dropped_over_cap": [int(x) for x in np.flatnonzero(lens > a.max_len)]})
    old_bytes = os.path.getsize(os.path.join(a.old, "ids.u32"))
    assert old_bytes == int(o_off[-1]) * 4
    off = np.concatenate([o_off, o_off[-1] + np.cumsum(np.asarray(new_lens, dtype=np.int64))]).astype(o_off.dtype)
    assert os.path.getsize(part) == int(off[-1]) * 4
    assert sha_file(part, old_bytes) == sha_file(os.path.join(a.old, "ids.u32")), "old ids prefix differs"
    os.replace(part, os.path.join(a.out, "ids.u32"))
    ls = np.concatenate([o_ls, np.asarray(new_ls, dtype=o_ls.dtype)])
    src = np.concatenate([o_src, np.asarray(new_src, dtype=o_src.dtype)])
    M = len(new_lens)
    W = N + M
    E = K + 1 + a.new_epochs
    order = np.full((E, W), -1, dtype=np.int32)
    for e in range(K):
        order[e, :N] = o_ord[e]
    order[K, :a.partial_rows] = o_ord[K][:a.partial_rows]
    for e in range(K + 1, E):
        order[e] = np.random.default_rng(a.seed + e).permutation(W)
    for e in range(K):
        assert np.array_equal(order[e][order[e] >= 0], o_ord[e]), e
    assert np.array_equal(order[K][order[K] >= 0], o_ord[K][:a.partial_rows])
    for e in range(K + 1, E):
        assert len(np.unique(order[e])) == W and order[e].min() == 0, e
    assert np.array_equal(off[:N + 1], o_off) and np.array_equal(ls[:N], o_ls) and np.array_equal(src[:N], o_src)
    np.save(os.path.join(a.out, "offsets.npy"), off)
    np.save(os.path.join(a.out, "label_start.npy"), ls)
    np.save(os.path.join(a.out, "source.npy"), src)
    np.save(os.path.join(a.out, "order.npy"), order)
    lens = np.diff(off)
    lab = lens - ls
    rows_e = [int((order[e] >= 0).sum()) for e in range(E)]
    meta = {k: v for k, v in o_meta.items() if k not in ("tokens_per_epoch", "dropped_rows", "label_tokens_per_epoch")}
    meta.update({"union_from": os.path.abspath(a.old), "extras": extras, "n_rows": int(W), "n_old": int(N), "n_new": int(M),
                 "max_len_cap": a.max_len, "epochs": int(E), "keep_epochs": K, "partial_epoch": K, "partial_rows": a.partial_rows,
                 "new_epochs": a.new_epochs, "rows_per_epoch": rows_e, "tokens_stored": int(off[-1]),
                 "tokens_per_epoch": [int(lens[order[e][order[e] >= 0]].sum()) for e in range(E)],
                 "label_tokens_per_epoch": [int(lab[order[e][order[e] >= 0]].sum()) for e in range(E)],
                 "new_tokens_per_epoch": int(lens[N:].sum()), "max_len": int(lens.max()),
                 "epoch_seeds_new": {str(e): a.seed + e for e in range(K + 1, E)},
                 "old_ids_sha256": sha_file(os.path.join(a.old, "ids.u32"))})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("UNION", json.dumps({k: meta[k] for k in ("n_old", "n_new", "rows_per_epoch", "tokens_per_epoch", "max_len")}))


if __name__ == "__main__":
    main()
