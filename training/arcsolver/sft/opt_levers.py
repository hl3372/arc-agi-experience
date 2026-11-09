"""Opt-in training-speed levers for train_fsdp.py (every default of the trainer is unchanged).

- --recompute selective: the full-attention decoder layers stay checkpointed, but the attention kernel's forward
  outputs (FA3 / FA4 / SDPA out + logsumexp) are kept from the forward (trainer_speed.sac_attention: bit-identical
  values, the recompute skips the quadratic attention forward; ~1.06x at 65k tokens with FA3, more on longer rows).
  Why this split and not "keep every token mixer's output, recompute only the MLP": the attention forward is the
  dominant part of the recompute at long rows and its outputs cost S x heads x head_dim per layer for only the
  attention layers (1 in 4 for Qwen3.5); keeping the GatedDeltaNet / attention block outputs of all layers would add
  S x hidden per layer for a recompute saving that is mostly projections and the linear-time GDN kernels. The cached
  tensors live in torch.utils.checkpoint's own storage, so saved-tensor hooks (BudgetOffload) never see them:
  sac_reserve_bytes() sizes them and the trainer lowers each row's offload budget by that much.
- --fp8 rowwise|tensorwise (experimental): torchao float8 training (convert_to_float8_training before FSDP2 sharding)
  for the big decoder-layer projections only: fp8_filter() keeps an nn.Linear under model.language_model.layers.N with
  both dims divisible by 16 and at least --fp8-min-dim (default 1024: the GDN in_proj_a / in_proj_b with 32 outputs
  gain nothing and pay the dynamic-scaling casts), skips the first and last decoder layers unless
  --no-fp8-skip-edge-layers, and never touches lm_head (the trainer's chunked CE calls it directly), embeddings, the
  vision tower or MTP modules. --cpu-test runs torchao's emulated fp8.
- --fused-kernels compile_dynamic: trainer_speed's compiled RMSNorm (exact forward, fused backward) + SwiGLU with
  torch.compile(dynamic=True) and a raised recompile limit (static shapes recompile per row length). Forward
  bit-equal to eager (~1.09-1.12x on top of FA3 + selective recompute).
- --fused-kernels liger: Liger-kernel's RMSNorm (offset 1.0, casting "gemma", not in place: Liger's own
  apply_liger_kernel_to_qwen3_5 settings, = Qwen3_5RMSNorm's (x_fp32 * rstd) * (1 + w_fp32) cast once) and SwiGLU
  (silu in fp32 cast to bf16 times up, = eager). Rope is not patched (partial rotary, Liger has no Qwen3.5 rope).
  GatedDeltaNet's gated norm is fla's FusedRMSNormGated whenever fla is installed and is left alone. Liger's row
  reduction is not eager's, so its norm forward differs by bf16 flips; a resume A/B on the 9B matched the eager
  path within rel 7e-4 on the gradient norm, at ~1.14x speed. enable_liger() checks both kernels against the eager
  modules on the device before patching and refuses on a mismatch.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

_LAYER_RE = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.")
_NEVER = ("lm_head", "embed_tokens", "visual", "mtp")


def _trainer_speed():
    from arcsolver.sft import trainer_speed
    return trainer_speed


def _text_cfg(cfg):
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)


# --- selective recompute ------------------------------------------------------------------------------------------

def enable_selective_recompute(hf) -> int:
    """Keep the full-attention layers' attention-kernel outputs from the forward (call after
    gradient_checkpointing_enable). Returns the number of layers switched. With FA4 loaded (--attn fa4) its kernels
    run as custom ops (attn_backends.enable_fa4_ops) so the policy can keep them."""
    ts = _trainer_speed()
    from arcsolver.sft import attn_backends
    if getattr(attn_backends, "_FA4", None) is not None:
        attn_backends.enable_fa4_ops()
    return ts.sac_attention(hf)


def sac_reserve_bytes(cfg, n_tokens: int) -> int:
    """GPU bytes the selective recompute keeps for one row: per full-attention layer the bf16 attention output
    (S x heads x head_dim) and the fp32 logsumexp (heads x S), plus 1% for the kernels' extra outputs."""
    t = _text_cfg(cfg)
    types = getattr(t, "layer_types", None) or ["full_attention"] * t.num_hidden_layers
    n_attn = sum(1 for x in types if x == "full_attention")
    heads = t.num_attention_heads
    hd = getattr(t, "head_dim", None) or t.hidden_size // heads
    return int(n_attn * n_tokens * heads * (hd * 2 + 4) * 1.01)


# --- fp8 ---------------------------------------------------------------------------------------------------------

def fp8_filter(module, fqn: str, n_layers: int, skip_edge: bool = True, min_dim: int = 1024) -> bool:
    """True for the nn.Linear modules that become Float8Linear (see the module docstring)."""
    if not isinstance(module, nn.Linear) or any(p in fqn.split(".") for p in _NEVER):
        return False
    m = _LAYER_RE.search(fqn)
    if m is None:
        return False
    i = int(m.group(1))
    if skip_edge and i in (0, n_layers - 1):
        return False
    a, b = module.in_features, module.out_features
    return a % 16 == 0 and b % 16 == 0 and min(a, b) >= min_dim


def convert_fp8(hf, recipe: str, skip_edge: bool = True, min_dim: int = 1024, emulate: bool = False) -> list:
    """Swap the selected nn.Linear modules of the HF model for torchao Float8Linear (before fully_shard).
    Returns the converted module names."""
    import dataclasses
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    if recipe not in ("rowwise", "tensorwise"):
        raise ValueError(f"fp8 recipe must be rowwise or tensorwise, got {recipe!r}")
    cfg = Float8LinearConfig.from_recipe_name(recipe)
    if emulate:
        cfg = dataclasses.replace(cfg, emulate=True)
    n_layers = len(hf.model.language_model.layers)
    picked = []

    def keep(mod, fqn):
        ok = fp8_filter(mod, fqn, n_layers, skip_edge, min_dim)
        if ok:
            picked.append(fqn)
        return ok

    convert_to_float8_training(hf, config=cfg, module_filter_fn=keep)
    return picked


def summarize_modules(names: list) -> str:
    """'84 modules: mlp.down_proj x28, mlp.gate_proj x28, ...' (layer index stripped)."""
    kinds = {}
    for n in names:
        k = _LAYER_RE.split(n)[-1]
        kinds[k] = kinds.get(k, 0) + 1
    return f"{len(names)} modules: " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items()))


# --- fused kernels -------------------------------------------------------------------------------------------------

_LIGER_ORIG = {}


def _liger_norm_forward(self, x):
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    return LigerRMSNormFunction.apply(x, self.weight, self.eps, 1.0, "gemma", False)


def _liger_mlp_forward(self, x):
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


def _check(name, ref, got, rel):
    err = (ref.float() - got.float()).abs().max().item()
    scale = max(ref.float().abs().max().item(), 1e-6)
    if err > rel * scale:
        raise RuntimeError(f"--fused-kernels liger: {name} differs from the eager module by {err:.3g} (scale {scale:.3g}); "
                           f"not patched")
    return err / scale


def enable_liger(device, hidden_act: str = "silu", hidden: int = 256, check: bool = True) -> list:
    """Patch Qwen3_5RMSNorm.forward and Qwen3_5MLP.forward (process-wide) with Liger's kernels after checking each
    against the eager forward and input gradient on `device` (fp32: ~1e-5 relative; bf16: within 2 bf16 ulps).
    Returns the patched class names."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if hidden_act != "silu":
        raise RuntimeError(f"--fused-kernels liger: hidden_act {hidden_act!r} is not SwiGLU's silu")
    import liger_kernel  # noqa: F401  (pip install liger-kernel==0.8.5; imported here, not at load)
    if check:
        torch.manual_seed(0)
        for dt, rel in ((torch.float32, 1e-5), (torch.bfloat16, 2 ** -6)):
            norm = M.Qwen3_5RMSNorm(hidden, eps=1e-6).to(device)
            with torch.no_grad():
                norm.weight.normal_(0, 0.1)
            x = torch.randn(1, 37, hidden, device=device, dtype=dt, requires_grad=True)
            ref = M.Qwen3_5RMSNorm.forward(norm.to(dt), x)
            got = _liger_norm_forward(norm.to(dt), x)
            _check(f"RMSNorm forward ({dt})", ref, got, rel)
            gr, = torch.autograd.grad(ref.float().square().sum(), x)
            gg, = torch.autograd.grad(got.float().square().sum(), x)
            _check(f"RMSNorm input grad ({dt})", gr, gg, rel * 4)
            g = torch.randn(1, 37, 2 * hidden, device=device, dtype=dt)
            u = torch.randn(1, 37, 2 * hidden, device=device, dtype=dt)
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction
            _check(f"SwiGLU ({dt})", torch.nn.functional.silu(g) * u, LigerSiLUMulFunction.apply(g, u), rel)
    if "norm" not in _LIGER_ORIG:
        _LIGER_ORIG["norm"] = M.Qwen3_5RMSNorm.forward
        M.Qwen3_5RMSNorm.forward = _liger_norm_forward
    if "mlp" not in _LIGER_ORIG:
        _LIGER_ORIG["mlp"] = M.Qwen3_5MLP.forward
        M.Qwen3_5MLP.forward = _liger_mlp_forward
    return ["Qwen3_5RMSNorm (decoder, q/k and final norms)", "Qwen3_5MLP (silu(gate) * up)"]


def disable_liger() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if "norm" in _LIGER_ORIG:
        M.Qwen3_5RMSNorm.forward = _LIGER_ORIG.pop("norm")
    if "mlp" in _LIGER_ORIG:
        M.Qwen3_5MLP.forward = _LIGER_ORIG.pop("mlp")


def enable_compile_dynamic(recompile_limit: int = 64) -> None:
    _trainer_speed().enable_compiled_norm_mlp(dynamic=True, recompile_limit=recompile_limit)


# --- one entry point for train_fsdp.load_model ----------------------------------------------------------------------

def apply(hf, cfg, device, recompute="full", fused="none", fp8="none", fp8_skip_edge=True, fp8_min_dim=1024,
          fp8_emulate=False) -> dict:
    """Apply the requested levers to a loaded (unsharded) model; returns what was done, for the [model] log line."""
    info = {}
    if recompute == "selective":
        info["recompute"] = f"selective ({enable_selective_recompute(hf)} attention layers keep their kernel outputs)"
    if fused == "liger":
        info["fused"] = "liger: " + ", ".join(enable_liger(device, getattr(_text_cfg(cfg), "hidden_act", "silu")))
    elif fused == "compile_dynamic":
        enable_compile_dynamic()
        info["fused"] = "compile_dynamic (exact norm + SwiGLU, dynamic shapes)"
    if fp8 != "none":
        names = convert_fp8(hf, fp8, fp8_skip_edge, fp8_min_dim, fp8_emulate)
        if not names:
            raise RuntimeError("--fp8: no Linear module passed the filter (check --fp8-min-dim)")
        info["fp8"] = f"{fp8}{' (emulated)' if fp8_emulate else ''}: " + summarize_modules(names)
    return info
