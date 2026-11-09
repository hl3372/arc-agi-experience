"""Merge packing_v2 shard caches into one stage-1 cache.

    python -m pv2.merge_caches --output out/aux_pv2.json out/shard_1.json out/shard_2.json ...

Shards come from ``pv2.build_dataset`` runs with different ``--seed`` values (row seeds are
``seed * 10_000_000 + i``, so shards of fewer than 10M rows never share a row).  The merge refuses
shards that overlap (a repeated prov), were built with a different max_seq_length, config digest,
generator digest or builder version, and sums the per-shard stats.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

MUST_MATCH = ("builder", "version", "max_seq_length", "config_digest", "generator_digest")


def merge(paths: list[str]) -> dict:
    examples, provs, stats, failures = [], set(), Counter(), Counter()
    first = None
    seeds = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload["metadata"]
        if first is None:
            first = meta
        for key in MUST_MATCH:
            if meta.get(key) != first.get(key):
                raise SystemExit(f"{path}: {key}={meta.get(key)!r} differs from {first.get(key)!r}")
        for ex in payload["examples"]:
            if ex["prov"] in provs:
                raise SystemExit(f"{path}: row {ex['prov']} already in an earlier shard (reuse of a --seed?)")
            provs.add(ex["prov"])
            examples.append(ex)
        stats.update(meta.get("stats", {}))
        failures.update(meta.get("failures", {}))
        seeds.append(meta.get("seed"))
    lengths = sorted(e["length"] for e in examples)
    meta = {k: v for k, v in first.items() if k not in ("stats", "failures", "seed", "seconds")}
    meta.update({"merged_from": [os.path.basename(p) for p in paths], "seeds": seeds,
                 "stats": dict(stats), "failures": dict(failures),
                 "length_p50": lengths[len(lengths) // 2], "length_max": lengths[-1],
                 "tokens_total": sum(lengths)})
    return {"metadata": meta, "examples": examples}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args(argv)
    payload = merge(args.shards)
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, args.output)
    m = payload["metadata"]
    print(f"merged {len(args.shards)} shards: {len(payload['examples'])} rows, {m['tokens_total']} tokens")


if __name__ == "__main__":
    main()
