#!/usr/bin/env python
"""bf16 Qwen3.5-family checkpoint -> block-FP8 checkpoint in the official Qwen3.8-27B-FP8 format (fp8_quant.py: e4m3
weights with 128x128 bf16 *_scale_inv for the decoder / MTP projections, everything else bf16), e.g. for serving the
SFT exports with vLLM in FP8.
    python -m arcsolver.tools.make_fp8 --src MODEL_BF16 --dst MODEL_FP8 [--check-against OFFICIAL_FP8 --check-n 40]
        [--device cuda]
Streams shard by shard (never the whole model in memory). --check-against compares --check-n quantized tensors (and
their scales) byte for byte with an official FP8 checkpoint; tensors whose shapes differ (a pruned vocab) are skipped.
config.json gets fp8_quant.fp8_quantization_config() with its defaults (the Qwen3.8-27B module list: a skip list for
loaders, which only consult it for Linear layers; the quantized set is fixed by fp8_quant.should_quantize).
"""
import argparse
import json
import os
import random
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from arcsolver.tools import fp8_quant as fq


def entries(name, w):
    if fq.should_quantize(name, tuple(w.shape)):
        q, s = fq.quantize_weight(w)
        return {name: q, fq.scale_name(name): s}
    return {name: w}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check-against", default="")
    ap.add_argument("--check-n", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for fn in os.listdir(a.src):
        p = os.path.join(a.src, fn)
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or os.path.isdir(p):
            continue
        shutil.copy2(p, os.path.join(a.dst, fn))
    idx_path = os.path.join(a.src, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx_path))["weight_map"].values())) if os.path.exists(idx_path) \
        else ["model.safetensors"]
    weight_map, n_q, total = {}, 0, 0
    for fn in files:
        out = {}
        with safe_open(os.path.join(a.src, fn), "pt") as f:
            for k in f.keys():
                ents = entries(k, f.get_tensor(k).to(a.device))
                for kk, v in ents.items():
                    out[kk] = v.contiguous().cpu()
                    weight_map[kk] = fn
                    total += out[kk].numel() * out[kk].element_size()
                n_q += len(ents) > 1
        save_file(out, os.path.join(a.dst, fn), metadata={"format": "pt"})
        print(f"{fn}: {len(out)} tensors", flush=True)
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)
    cfg_path = os.path.join(a.dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["quantization_config"] = fq.fp8_quantization_config()
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"FP8 checkpoint: {n_q} quantized tensors, {total / 2**30:.1f} GiB -> {a.dst}", flush=True)
    if a.check_against:
        ref_map = json.load(open(os.path.join(a.check_against, "model.safetensors.index.json")))["weight_map"]
        cand = [k for k in weight_map if k in ref_map and weight_map[k] and "embed_tokens" not in k and "lm_head" not in k]
        random.Random(0).shuffle(cand)
        ok = bad = skip = 0
        for k in cand[:a.check_n]:
            with safe_open(os.path.join(a.dst, weight_map[k]), "pt") as f:
                mine = f.get_tensor(k)
            with safe_open(os.path.join(a.check_against, ref_map[k]), "pt") as f:
                ref = f.get_tensor(k)
            if mine.shape != ref.shape or mine.dtype != ref.dtype:
                skip += 1
                continue
            same = torch.equal(mine.view(torch.uint8) if mine.element_size() == 1 else mine,
                               ref.view(torch.uint8) if ref.element_size() == 1 else ref)
            ok += same
            bad += not same
            if not same:
                print(f"MISMATCH {k}: {mine.dtype} {tuple(mine.shape)}", flush=True)
        print(f"check vs {a.check_against}: {ok} identical, {bad} different, {skip} skipped (shape/dtype)", flush=True)
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
