"""Opt-in speedups for the FSDP2 trainer (train_fsdp.py --recompute selective / --fused-kernels compile_dynamic /
--compile-norm-mlp). Nothing here changes what the trainer computes beyond bf16 rounding order; every piece is off
unless asked for.

- sac_attention(hf): selective activation checkpointing for the softmax-attention decoder layers. The layers stay
  checkpointed, but the attention kernel's forward outputs (out + logsumexp; SDPA flash/efficient/cuDNN, FA3 or FA4
  as custom ops) are kept from the forward, so the backward's recompute of those layers skips the quadratic
  attention forward. Extra memory: S x heads x head_dim x 2 B (+ lse) per attention layer. The recompute is
  otherwise unchanged and the cached tensors are the forward's own, so values are bit-identical to plain
  checkpointing with the same kernel.
- enable_compiled_norm_mlp(): torch.compile of RMSNorm (RMSNormExactFwd: forward bit-equal to eager, fused
  backward) and silu(gate) * up. A fully compiled norm (norm_mode="fused") is faster but changes bf16 roundings of
  the forward, which this model family amplifies to gradient differences far above the run-to-run noise floor.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

# --- selective activation checkpointing ---------------------------------------------------------------------------

_ATTN_FWD_OPS = ("aten._scaled_dot_product_flash_attention.", "aten._scaled_dot_product_efficient_attention.",
                 "aten._scaled_dot_product_flash_attention_for_cpu.",   # CPU tests only (train_fsdp --cpu-test --recompute)
                 "aten._scaled_dot_product_cudnn_attention.", "aten._flash_attention_forward.",
                 "aten._efficient_attention_forward.", "::_flash_attn_forward", "._flash_attn_forward.",
                 "fa4sac.fwd.")     # attn_backends.enable_fa4_ops (FA4 as a custom op)


def _is_attention_forward(func) -> bool:
    s = str(func)
    return any(k in s for k in _ATTN_FWD_OPS)


def sac_policy(save_hidden_mm: int = 0):
    """Policy for torch.utils.checkpoint's selective checkpointing: keep the attention kernel's forward outputs;
    with save_hidden_mm = hidden size, also the outputs of the matmuls that project back to the hidden size
    (o_proj / out_proj / down_proj: S x hidden each), recomputing everything else."""
    from torch.utils.checkpoint import CheckpointPolicy
    aten = torch.ops.aten
    mm_ops = (aten.mm.default, aten.addmm.default)

    def policy(ctx, func, *args, **kwargs):
        if _is_attention_forward(func):
            return CheckpointPolicy.MUST_SAVE
        if save_hidden_mm and func in mm_ops:
            b = args[-1]
            if b.dim() == 2 and b.shape[-1] == save_hidden_mm:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE
    return policy


def sac_checkpoint_fn(save_hidden_mm: int = 0):
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
    ctx_fn = functools.partial(create_selective_checkpoint_contexts, sac_policy(save_hidden_mm))
    return functools.partial(checkpoint, use_reentrant=False, context_fn=ctx_fn)


def text_layers(hf):
    return hf.model.language_model.layers


def sac_attention(hf, layer_types=("full_attention",), save_hidden_mm: bool = False) -> int:
    """Switch the decoder layers of these types to selective checkpointing (call after
    gradient_checkpointing_enable; works before or after fully_shard). Returns the number of layers changed."""
    hidden = hf.config.get_text_config().hidden_size if hasattr(hf.config, "get_text_config") else 0
    fn = sac_checkpoint_fn(hidden if save_hidden_mm else 0)
    n = 0
    for layer in text_layers(hf):
        if getattr(layer, "layer_type", None) in layer_types:
            if not getattr(layer, "gradient_checkpointing", False):
                raise RuntimeError("sac_attention: gradient checkpointing must be enabled first")
            layer._gradient_checkpointing_func = fn
            n += 1
    return n


# --- compiled RMSNorm / SwiGLU (elementwise fusion) -------------------------------------------------------------------

_COMPILE_ORIG = {}
_COMPILED_FNS = {}
_COMPILE_DYNAMIC = False     # enable_compiled_norm_mlp(dynamic=True): symbolic shapes from the first compile
# The compiled SwiGLU rounds silu(g) to bf16 before the product, as eager does (inductor's emulate_precision_casts).
# torch 2.10's CPU (C++) codegen crashes on that option ("to_dtype() got an unexpected keyword argument
# 'use_compute_types'"), so the CPU test turns it off; the GPU (Triton) path keeps it.
SWIGLU_EMULATE_CASTS = True


def _rms_eager(x, w, eps: float):
    """Qwen3_5RMSNorm.forward's math (fp32 throughout, one rounding at the end)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + w.float())).type_as(x)


def _rms_sq(x):
    """x.float() ** 2 in fp32, bit-equal to eager's x.float().pow(2) (CUDA pow with exponent 2 is base * base)."""
    xf = x.float()
    return xf * xf


def _rms_tail(x, rstd, w):
    """Eager's (x.float() * rstd) * (1 + w.float()) and the one bf16 rounding: the same fp32 ops in the same
    order, so bit-equal given the same rstd."""
    return ((x.float() * rstd) * (1.0 + w.float())).to(x.dtype)


def _rms_bwd(x, w, rstd, gy):
    """d/dx and d/dw of y = x * rstd * (1 + w), rstd = (mean(x^2) + eps)^-1/2, in fp32; one rounding each."""
    xf, g = x.float(), gy.float()
    g1 = g * (1.0 + w.float())
    s = (g1 * xf).sum(-1, keepdim=True)
    dx = rstd * g1 - (rstd * rstd * rstd) * (s / x.shape[-1]) * xf
    dw = (g * (xf * rstd)).reshape(-1, x.shape[-1]).sum(0)
    return dx.to(x.dtype), dw.to(w.dtype)


def _swiglu_eager(g, u):
    return F.silu(g) * u


def _compiled(name):
    """One torch.compile wrapper per function for the whole process (a new wrapper per enable would recompile).
    Static shapes except the sequence dim, which the callers mark dynamic: the hidden-size reductions stay
    persistent kernels and one graph serves every row length. The SwiGLU emulates eager's bf16 rounding of
    silu(g) before the product (emulate_precision_casts), so its output matches eager up to the fp32 silu itself."""
    if name not in _COMPILED_FNS:
        fn = {"norm": _rms_eager, "rms_sq": _rms_sq, "rms_tail": _rms_tail, "rms_bwd": _rms_bwd}.get(name)
        dyn = True if _COMPILE_DYNAMIC else None
        if fn is not None:
            _COMPILED_FNS[name] = torch.compile(fn, dynamic=dyn)
        else:
            _COMPILED_FNS[name] = torch.compile(_swiglu_eager, dynamic=dyn,
                                                options={"emulate_precision_casts": SWIGLU_EMULATE_CASTS})
    return _COMPILED_FNS[name]


def _seq_dynamic(*ts):
    """Mark the sequence dim ([B, S, ...] -> 1; [S, H] -> 0) dynamic before the first compile, also on a view's
    base (dynamo guards the base's strides too: q_norm gets a chunk of q_proj's output, which otherwise
    recompiles once per new length)."""
    for t in ts:
        for x in (t, t._base):
            if x is not None:
                torch._dynamo.maybe_mark_dynamic(x, 1 if x.dim() >= 3 else 0)


class RMSNormExactFwd(torch.autograd.Function):
    """Qwen3_5RMSNorm with a forward bit-equal to eager and a fused backward.

    Why: in this model any bf16-level change of the forward (a handful of 1-ulp flips per norm call, which is all a
    fully compiled RMSNorm gives: inductor's row reduction sums in another order) spreads through the 98k-token
    GatedDeltaNet recurrences to the level FA3 shows vs SDPA: log-probs 0.008 mean, gradients 20-45% vs the
    baseline, far above the 1% run-to-run floor. The backward is linear, so reordering it
    stays at the floor.
    Forward: squares by inductor (bit-equal), the row mean by eager's own reduction kernel (same order), rsqrt
    eager, the scale chain by inductor (bit-equal): 3 kernels reading 14 B per element instead of eager's 7-8 and
    40 B. Backward: one fused inductor kernel pair (dx, dw) instead of eager autograd's ~12 fp32 kernels."""

    @staticmethod
    def forward(ctx, x, w, eps):
        _seq_dynamic(x)
        sq = _compiled("rms_sq")(x).contiguous()          # eager's x.float() is contiguous for these inputs too
        rstd = torch.rsqrt(sq.mean(-1, keepdim=True) + eps)
        del sq
        _seq_dynamic(rstd)
        y = _compiled("rms_tail")(x, rstd, w)
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, w, rstd = ctx.saved_tensors
        _seq_dynamic(x, rstd, gy)
        dx, dw = _compiled("rms_bwd")(x, w, rstd, gy)
        return dx, dw, None


def _norm_forward_exact(self, x):
    return RMSNormExactFwd.apply(x, self.weight, self.eps)


def _norm_forward_compiled(self, x):
    _seq_dynamic(x)
    return _compiled("norm")(x, self.weight, self.eps)


def _mlp_forward_compiled(self, x):
    g, u = self.gate_proj(x), self.up_proj(x)
    _seq_dynamic(g, u)
    return self.down_proj(_compiled("mlp")(g, u))


def enable_compiled_norm_mlp(norm: bool = True, mlp: bool = True, norm_mode: str = "exact", dynamic: bool = False,
                             recompile_limit: int = 0) -> None:
    """Process-wide: Qwen3_5RMSNorm and the silu(gate) * up of Qwen3_5MLP through torch.compile.
    norm_mode "exact" (default): RMSNormExactFwd, forward bit-equal to eager, fused backward. "fused": the whole
    norm compiled (fastest; forward differs by bf16 flips, which this model amplifies far beyond the floor, so it
    is for measurements only). The SwiGLU forward matched eager bit for bit at 32k/65k (emulate_precision_casts).

    Inside FSDP2 every call sees the gathered (plain) parameters. sac_attn's checkpoint dispatch modes declare
    ignore_compile_internals, so the graphs compile and run inside the attention layers too, in the forward and
    the recompute alike. The sequence dim is marked dynamic, so a new length should not recompile the graphs
    (tested on CPU with FSDP2 + checkpointing + SAC).

    dynamic (opt-in, train_fsdp --fused-kernels compile_dynamic): torch.compile(dynamic=True), i.e. symbolic shapes
    from the first compile instead of maybe_mark_dynamic hints. The SFT trainer's rows change length every
    micro-step, and without it a run hit recompile_limit 8 in _rms_bwd (steps ~3x slower for 10+ steps).
    recompile_limit > 0 raises dynamo's per-frame limit (the norm runs at several call sites and layouts: decoder
    norms, q/k norms on [B, S, heads, d] views, the final norm). Must be set before the first compiled call."""
    global _COMPILE_DYNAMIC
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if dynamic:
        if _COMPILED_FNS and not _COMPILE_DYNAMIC:
            raise RuntimeError("enable_compiled_norm_mlp(dynamic=True) after static graphs were built in this process")
        _COMPILE_DYNAMIC = True
    if recompile_limit:
        for k in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, k):
                setattr(torch._dynamo.config, k, max(getattr(torch._dynamo.config, k), recompile_limit))
    if norm and "norm" not in _COMPILE_ORIG:
        if norm_mode not in ("exact", "fused"):
            raise ValueError(f"norm_mode must be 'exact' or 'fused', got {norm_mode!r}")
        _COMPILE_ORIG["norm"] = M.Qwen3_5RMSNorm.forward
        M.Qwen3_5RMSNorm.forward = _norm_forward_exact if norm_mode == "exact" else _norm_forward_compiled
    if mlp and "mlp" not in _COMPILE_ORIG:
        _COMPILE_ORIG["mlp"] = M.Qwen3_5MLP.forward
        M.Qwen3_5MLP.forward = _mlp_forward_compiled


def disable_compiled_norm_mlp() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if "norm" in _COMPILE_ORIG:
        M.Qwen3_5RMSNorm.forward = _COMPILE_ORIG.pop("norm")
    if "mlp" in _COMPILE_ORIG:
        M.Qwen3_5MLP.forward = _COMPILE_ORIG.pop("mlp")
