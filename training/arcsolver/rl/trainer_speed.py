"""Opt-in speedups for the FSDP2 RL trainer (fsdp.py + train.py).

Nothing here changes what the trainer computes beyond bf16 rounding order; every piece is off unless
fsdp.load(..., speed=...) / --trainer-speed asks for it. The equivalence notes below were measured on Qwen3.5 at
32k-98k tokens against the default path (log-probs and gradients vs the run-to-run noise floor).

Pieces:
- sac_attention(hf): selective activation checkpointing for the softmax-attention decoder layers. The layers stay
  checkpointed, but the attention kernel's forward outputs (out + logsumexp; SDPA flash/efficient/cuDNN or FA3) are
  kept from the forward, so the backward's recompute of those layers skips the quadratic attention forward. Extra
  memory: S x 16 x 256 x 2 B (+ lse) per attention layer = 0.8 GB per layer at 98k tokens (6.4 GB for the 8 attention layers of Qwen3.5-4B).
  The recompute is otherwise unchanged and the cached tensors are the forward's own, so values are bit-identical to
  plain checkpointing with the same kernel.
- FusedHeadLogprobs / token_and_head_logprobs_fused: lm_head + fp32 log-softmax + gather at the sampled token and
  the sampler's top-k, without fp32 logits. Same precision path as sc._chunk_token_and_head (bf16 logits from
  the same cuBLAS call, fp32 logsumexp, dlogits formed in fp32 and rounded to bf16 once, the same two backward
  GEMMs); only the fp32 reduction order inside the logsumexp differs, and the weight gradient is summed over chunks
  in fp32 (the default path sums its per-chunk bf16 products in bf16, which alone moves it ~0.4-1% when the chunk
  size changes). Saves only the fp32 logsumexp per row, so chunks can be 4-8x larger. Measured at 65k: log-probs
  identical, gradient at the run-to-run noise floor, 1.026x.
- enable_gdn_gva(): GatedDeltaNet without repeating q/k to the 32 value heads (fla >= 0.5 GVA; on Hopper with
  install_gva_dqkwg_backend, without which fla 0.5.0 refuses the GVA backward). Forward bit-equal, gradient at the
  floor, but no measured speed-up (1.000x): works, not worth enabling.
- enable_compiled_norm_mlp(): torch.compile of RMSNorm (RMSNormExactFwd: forward bit-equal to eager, fused
  backward) and silu(gate) * up. Log-probs bit-equal, gradient at the floor. The speed-up depends on the setup (in the
  FSDP2 SFT trainer at ~131k tokens it was measured slower than eager): measure before enabling. A fully compiled norm
  (norm_mode="fused") is faster but changes bf16 roundings of the forward, which this model amplifies to FA3-sized
  gradient differences.
- assign_balanced: rollouts to ranks by estimated cost (LPT, no equal-count constraint) for gradient accumulation
  without a per-rollout reduce-scatter (FSDP2 no-sync: the unsharded gradient accumulates in fp32, reduce_dtype,
  and is reduce-scattered once per minibatch; train.py --balance lpt).
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

# --- selective activation checkpointing ---------------------------------------------------------------------------

_ATTN_FWD_OPS = ("aten._scaled_dot_product_flash_attention.", "aten._scaled_dot_product_efficient_attention.",
                 "aten._scaled_dot_product_flash_attention_for_cpu.",   # CPU tests only
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


# --- Gated DeltaNet without the q/k head repeat (fla GVA) -----------------------------------------------------------

_GDN_ORIG = None


def _gdn_forward_gva(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    """Qwen3_5GatedDeltaNet.forward for training rows (no cache): identical except that q/k keep their 16 heads and
    fla's chunk_gated_delta_rule maps v-head h to k-head h // 2 itself (GVA), which is exactly what
    repeat_interleave(2, dim=2) feeds it today. Saves the repeated copies and their backward reduction."""
    if cache_params is not None:
        return _GDN_ORIG(self, hidden_states, cache_params=cache_params, cache_position=cache_position,
                         attention_mask=attention_mask)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    bsz, seq_len, _ = hidden_states.shape
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(bsz, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    if self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                                          activation=self.activation, seq_idx=None)
    else:
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(bsz, seq_len, -1, self.head_k_dim)
    key = key.reshape(bsz, seq_len, -1, self.head_k_dim)
    value = value.reshape(bsz, seq_len, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    core, _ = self.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None,
                                          output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(bsz, seq_len, -1))


_GVA_BACKEND = None


def install_gva_dqkwg_backend():
    """fla 0.5.0 on Hopper with Triton >= 3.4: the gated chunk_bwd_dqkwg must run in TileLang (fla refuses its own
    Triton kernel there: wrong results, fla #640), but TileLang's verifier rejects GVA (k with fewer heads than v)
    and fla then falls back to the refused Triton kernel, so gdn_gva fails on Hopper without this.
    Fix: one more `common` backend, ahead of TileLang, that takes exactly that case: q/k are expanded to the value
    heads for this one kernel (value head h reads key head h // G, fla's own GVA mapping and what
    repeat_interleave(G, dim=2) feeds the baseline), TileLang runs as in the baseline, and dq/dk are summed over
    the G heads of each group, which is what fla's Triton GVA path returns. Every other GVA kernel (l2norm, kkt,
    wy, fwd_h, fwd_o, dhu, wy_bwd) runs with the 16 key heads. Idempotent; returns the backend."""
    global _GVA_BACKEND
    if _GVA_BACKEND is not None:
        return _GVA_BACKEND
    from fla.ops.backends import BackendRegistry, BaseBackend
    BackendRegistry.ensure_initialized("common")
    reg = BackendRegistry._registries.get("common")
    if reg is None:
        raise RuntimeError("fla: no 'common' backend registry (fla version without backend dispatch?)")

    class TileLangGVABackend(BaseBackend):
        backend_type = "tilelang_gva"
        package_name = "tilelang"
        env_var = "FLA_TILELANG"
        priority = 4                                  # ahead of fla's TileLang (5); other ops: no impl -> skipped

        @classmethod
        def is_available(cls) -> bool:
            try:
                import tilelang  # noqa: F401  (availability probe)
                return True
            except ImportError:
                return False

        def chunk_bwd_dqkwg_verifier(self, q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, scale=None,
                                     cu_seqlens=None, chunk_size=64, chunk_indices=None, use_exp2=False,
                                     transpose_state_layout=False):
            if v.shape[2] == k.shape[2]:
                return False, "not GVA (fla's TileLang backend takes it)"
            if v.shape[2] % k.shape[2]:
                return False, "value heads not a multiple of key heads"
            if g is None or g_gamma is not None:
                return False, "only the gated case (g, no g_gamma)"
            if h.dtype != q.dtype:
                return False, "TileLang needs h.dtype == q.dtype"
            return True, None

        def chunk_bwd_dqkwg(self, q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, scale=None,
                            cu_seqlens=None, chunk_size=64, chunk_indices=None, use_exp2=False,
                            transpose_state_layout=False):
            from fla.ops.common.backends.tilelang.chunk_bwd import chunk_bwd_dqkwg_tilelang
            self.calls = getattr(self, "calls", 0) + 1
            B, T, H, K = k.shape
            G = v.shape[2] // H
            if scale is None:                         # the expanded K is the same, but keep fla's default explicit
                scale = K ** -0.5
            dq, dk, dw, dg = chunk_bwd_dqkwg_tilelang(
                q=q.repeat_interleave(G, dim=2), k=k.repeat_interleave(G, dim=2), v=v, do=do, h=h, dh=dh, w=w, g=g,
                g_gamma=g_gamma, dv=dv, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size,
                chunk_indices=chunk_indices, use_exp2=use_exp2, transpose_state_layout=transpose_state_layout)
            dq = dq.view(B, T, H, G, K).sum(3)
            dk = dk.view(B, T, H, G, K).sum(3)
            return dq, dk, dw, dg

    _GVA_BACKEND = TileLangGVABackend()
    reg.register(_GVA_BACKEND)
    return _GVA_BACKEND


def enable_gdn_gva(on: bool = True) -> None:
    """Process-wide switch of Qwen3_5GatedDeltaNet.forward to the GVA path (needs fla's chunk_gated_delta_rule
    with GVA support: fla >= 0.5; on Hopper also tilelang, see install_gva_dqkwg_backend)."""
    global _GDN_ORIG
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if _GDN_ORIG is None:
        _GDN_ORIG = M.Qwen3_5GatedDeltaNet.forward
    if on and M.chunk_gated_delta_rule is None:
        raise RuntimeError("GVA path needs fla's chunk_gated_delta_rule")
    if on:
        install_gva_dqkwg_backend()
    M.Qwen3_5GatedDeltaNet.forward = _gdn_forward_gva if on else _GDN_ORIG


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
    persistent kernels and one graph serves every rollout length. The SwiGLU emulates eager's bf16 rounding of
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
    the recompute alike. No graph recompiles for a new length (the sequence dim is marked dynamic; checked on CPU
    with FSDP2 + checkpointing + SAC, torch 2.10 and 2.14). A gdn_gva backward error earlier in the same process
    leaves FSDP2's state half-resharded and makes later calls fail with "DTensor" errors unrelated to compile.

    dynamic (opt-in, used by the SFT trainer): torch.compile(dynamic=True), i.e. symbolic shapes from the first
    compile instead of maybe_mark_dynamic hints. Rows that change length every micro-step otherwise hit
    recompile_limit 8 in _rms_bwd (steps ~3x slower for a while).
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


# --- fused lm_head + log-softmax + gather ----------------------------------------------------------------------

_TRITON = None


def _kernels():
    global _TRITON
    if _TRITON is not None:
        return _TRITON
    import triton
    import triton.language as tl

    @triton.jit
    def lse_kernel(L_ptr, stride, LSE_ptr, V, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        base = L_ptr + row.to(tl.int64) * stride
        m = tl.full([BLOCK], -float("inf"), tl.float32)
        s = tl.zeros([BLOCK], tl.float32)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(base + cols, mask=cols < V, other=-float("inf")).to(tl.float32)
            m_new = tl.maximum(m, x)
            safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            s = s * tl.exp(m - safe) + tl.exp(x - safe)
            m = m_new
        mx = tl.max(m, 0)
        tot = tl.sum(tl.where(m == -float("inf"), 0.0, s * tl.exp(m - mx)), 0)
        tl.store(LSE_ptr + row, mx + tl.log(tot))

    @triton.jit
    def dlogits_kernel(L_ptr, stride, LSE_ptr, G_ptr, V, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        msk = cols < V
        ptr = L_ptr + row.to(tl.int64) * stride + cols
        x = tl.load(ptr, mask=msk, other=0.0).to(tl.float32)
        lse = tl.load(LSE_ptr + row)
        g = tl.load(G_ptr + row)
        tl.store(ptr, (tl.exp(x - lse) * g).to(L_ptr.dtype.element_ty), mask=msk)

    _TRITON = (triton, lse_kernel, dlogits_kernel)
    return _TRITON


def _lse_rows(logits, vocab, out):
    if not logits.is_cuda:                           # CPU tests: the same math in torch
        out.copy_(logits[:, :vocab].float().logsumexp(-1))
        return
    triton, lse_kernel, _ = _kernels()
    n = logits.shape[0]
    if n:
        lse_kernel[(n,)](logits, logits.stride(0), out, vocab, BLOCK=4096, num_warps=8)


def _dlogits_rows(logits, lse, gneg, vocab):
    if not logits.is_cuda:
        logits[:, :vocab] = (torch.exp(logits[:, :vocab].float() - lse[:, None]) * gneg[:, None]).to(logits.dtype)
        return
    triton, _, dl_kernel = _kernels()
    n = logits.shape[0]
    if n:
        BLOCK = 4096
        dl_kernel[(n, triton.cdiv(vocab, BLOCK))](logits, logits.stride(0), lse, gneg, vocab, BLOCK=BLOCK,
                                                   num_warps=8)


def _mm_f32(a, b):
    """a @ b with an fp32 result (bf16 inputs, fp32 accumulate, no bf16 rounding of the product)."""
    if a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
    return a.float().mm(b.float()) if not a.is_cuda else a.mm(b).float()


class FusedHeadLogprobs(torch.autograd.Function):
    """(log p(targets) [T], log p(head_ids) [T, k]) = gathers of log_softmax(hidden @ weight.T) in fp32.
    hidden [T, H] bf16, weight [Vp, H] bf16 (rows >= vocab are padding: excluded from the softmax, zero grad),
    targets [T] int64, head_ids [T, k] int64 (entries < 0 are padding; clamped to 0 like sc.py)."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, head_ids, chunk: int, vocab: int):
        T = hidden.shape[0]
        cols = torch.cat([targets[:, None], head_ids.clamp(min=0)], 1)
        lse = torch.empty(T, dtype=torch.float32, device=hidden.device)
        sel = torch.empty(cols.shape, dtype=torch.float32, device=hidden.device)
        for s in range(0, T, chunk):
            lg = F.linear(hidden[s:s + chunk], weight)
            _lse_rows(lg, vocab, lse[s:s + chunk])
            sel[s:s + chunk] = lg.gather(1, cols[s:s + chunk]).float()
            del lg
        lp = sel - lse[:, None]
        ctx.save_for_backward(hidden, weight, cols, lse)
        ctx.chunk, ctx.vocab = chunk, vocab
        return lp[:, 0].contiguous(), lp[:, 1:].contiguous()

    @staticmethod
    def backward(ctx, g_y, g_h):
        hidden, weight, cols, lse = ctx.saved_tensors
        T, chunk, vocab = hidden.shape[0], ctx.chunk, ctx.vocab
        if g_y is None:
            g_y = torch.zeros(T, dtype=torch.float32, device=hidden.device)
        if g_h is None:
            g_h = torch.zeros(T, cols.shape[1] - 1, dtype=torch.float32, device=hidden.device)
        coef = torch.cat([g_y.float()[:, None], g_h.float()], 1)          # d loss / d (logit - lse) per gathered entry
        gneg = (-g_y.float() - g_h.float().sum(1)).contiguous()           # d loss / d lse
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dh = torch.empty_like(hidden) if need_h else None
        dw = None                                     # fp32 sum over chunks (the default path sums them in bf16)
        starts = list(range(0, T, chunk))
        for s in reversed(starts):
            e = min(T, s + chunk)
            hc = hidden[s:e]
            lg = F.linear(hc, weight)
            c = cols[s:e]
            x_sel = lg.gather(1, c).float()
            _dlogits_rows(lg, lse[s:e], gneg[s:e], vocab)                 # dense part, in place, bf16
            dense_sel = torch.exp(x_sel - lse[s:e, None]) * gneg[s:e, None]
            same = c[:, :, None] == c[:, None, :]
            csum = (same.to(coef.dtype) * coef[s:e, None, :]).sum(-1)    # duplicate ids get every coefficient
            lg.scatter_(1, c, (dense_sel + csum).to(lg.dtype))
            if lg.shape[1] > vocab:
                lg[:, vocab:] = 0
            if need_h:
                torch.mm(lg, weight, out=dh[s:e])
            if need_w:
                dwc = _mm_f32(lg.t(), hc)
                dw = dwc if dw is None else dw.add_(dwc)
                del dwc
            del lg
        if dw is not None:
            dw = dw.to(weight.dtype)
        return dh, dw, None, None, None, None


def token_and_head_logprobs_fused(hidden, weight, targets, head_ids, chunk: int = 8192):
    """Drop-in for sc.token_and_head_logprobs (same returns); `chunk` rows of bf16 logits at a time."""
    if not (hidden.shape[0] == targets.shape[0] == head_ids.shape[0]):
        raise ValueError(f"{hidden.shape[0]} hidden states, {targets.shape[0]} targets, {head_ids.shape[0]} head rows")
    vocab = weight.shape[0]
    pad = -vocab % 128
    w = F.pad(weight, (0, 0, 0, pad)) if pad else weight
    if not (torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad)):
        with torch.no_grad():
            return FusedHeadLogprobs.forward(_NoCtx(), hidden, w, targets, head_ids, chunk, vocab)
    return FusedHeadLogprobs.apply(hidden, w, targets, head_ids, chunk, vocab)


class _NoCtx:
    def save_for_backward(self, *a):
        pass


# --- rank balancing ---------------------------------------------------------------------------------------------

def rollout_cost(n_tokens: int, a: float = 1.0, b: float = 0.0) -> float:
    """Estimated forward+backward time of one rollout of n tokens: a*n + b*n^2 (only ratios matter; train.py uses
    cost(n) = n + n^2 / --balance-quad-tokens)."""
    return a * n_tokens + b * n_tokens * n_tokens


def assign_balanced(lengths: list, world: int, cost=None) -> list:
    """Rollout indices per rank minimising the slowest rank's estimated total (LPT: longest first, each to the rank
    with the least load so far). Counts per rank may differ: only valid when the per-rollout backward needs no
    collective (no-sync accumulation). Every rank gets at least one rollout when there are >= world of them."""
    cost = cost or (lambda n: rollout_cost(n))
    order = sorted(range(len(lengths)), key=lambda i: (-cost(lengths[i]), i))
    load = [0.0] * world
    out = [[] for _ in range(world)]
    for i in order:
        r = min(range(world), key=lambda k: (load[k], len(out[k]), k))
        out[r].append(i)
        load[r] += cost(lengths[i])
    return out
