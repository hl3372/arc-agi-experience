"""Hugging Face model helpers shared with the FSDP SFT trainer: the SDPA causal-mask shortcut, the text decoder
layers of a Qwen3.5 checkpoint, AdamW parameter groups and the full bf16 HF export of an FSDP-sharded model."""
from __future__ import annotations

import json
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn.functional as F


def install_sdpa_causal():
    """At micro-batch 1 a verified purely causal attention mask becomes is_causal=True, so PyTorch takes the flash SDPA
    kernel instead of the masked (memory-efficient) one."""
    orig = F.scaled_dot_product_attention
    verified = {}

    def exact_causal(mask):
        S = mask.shape[-1]
        m = mask[0, 0] if mask.dim() == 4 else mask
        if m.shape[-2] != S:
            return False
        allowed = m if m.dtype == torch.bool else (m == 0)
        if S <= 4096:
            return bool(torch.equal(allowed, torch.ones(S, S, dtype=torch.bool, device=m.device).tril()))
        idx = torch.tensor([0, 1, S // 3, S // 2, S - 2, S - 1], device=m.device)
        want = torch.arange(S, device=m.device)[None, :] <= idx[:, None]
        return bool(torch.equal(allowed[idx], want)) and bool(allowed[:, 0].all()) and bool(allowed.diagonal().all())

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if (attn_mask is not None and not is_causal and q.dim() == 4 and q.shape[0] == 1
                and q.shape[-2] == k.shape[-2] and q.shape[-2] > 1):
            key = (tuple(attn_mask.shape), str(attn_mask.dtype))
            if key not in verified:
                verified[key] = exact_causal(attn_mask)
            if verified[key]:
                return orig(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True, scale=scale, **kw)
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = sdpa
    torch.nn.functional.scaled_dot_product_attention = sdpa


def text_layers(hf):
    return hf.model.language_model.layers


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "norm" not in n.lower() else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def export_hf(model, base, dest, rank, meta, dtype=None):
    """Full bf16 HF checkpoint (the base's key set and shards layout; vision weights from the base). dtype: the trained
    tensors' dtype, default bf16; base-copied tensors keep the base's dtype."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        trained = {k[len("hf."):]: v for k, v in sd.items() if k.startswith("hf.")}
        idx_path = os.path.join(base, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            wmap = json.load(open(idx_path))["weight_map"]
        else:
            with safe_open(os.path.join(base, "model.safetensors"), "pt") as f:
                wmap = {k: "model.safetensors" for k in f.keys()}
        missing = [k for k in trained if k not in wmap and k != "lm_head.weight"]   # lm_head: tied, absent in base
        if missing:
            raise RuntimeError(f"trained keys absent from the base checkpoint: {missing[:5]}")
        lm_untrained = [k for k in wmap if k not in trained and (k.startswith("model.language_model.") or k == "lm_head.weight")]
        if lm_untrained:
            raise RuntimeError(f"language-model base keys without a trained tensor: {lm_untrained[:5]}")
        copied = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in wmap if k not in trained})
        print(f"[export] base tensors copied unchanged (untrained parts): {copied}", flush=True)
        tmp = dest + ".part"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        by_file = {}
        for k, fn in wmap.items():
            by_file.setdefault(fn, []).append(k)
        total = 0
        for fn, keys in sorted(by_file.items()):
            tensors = {}
            base_keys = [k for k in keys if k not in trained]
            if base_keys:
                with safe_open(os.path.join(base, fn), "pt") as f:
                    for k in base_keys:
                        tensors[k] = f.get_tensor(k)
            for k in keys:
                if k in trained:
                    tensors[k] = trained[k].to(dtype or torch.bfloat16).contiguous()
            total += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(tensors, os.path.join(tmp, fn), metadata={"format": "pt"})
        if os.path.exists(idx_path):
            idx = json.load(open(idx_path))
            idx["metadata"] = {"total_size": total}
            json.dump(idx, open(os.path.join(tmp, "model.safetensors.index.json"), "w"), indent=2)
        for f in os.listdir(base):
            if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(os.path.join(base, f)):
                shutil.copy2(os.path.join(base, f), os.path.join(tmp, f))
        json.dump(meta, open(os.path.join(tmp, "milestone.json"), "w"), indent=1)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    dist.barrier()
