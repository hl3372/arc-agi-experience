"""Bit-exact re-creation of the official Qwen/Qwen3.8-27B-FP8 checkpoint from bf16 weights.

The trainers keep bf16/fp32 weights. An FP8 vLLM sampler (RL weight sync) or an FP8 export (make_fp8.py) must get
weights in exactly the format of Qwen/Qwen3.8-27B-FP8, and the FP8 weights we serve must be byte-for-byte what the
trainer wrote. Applied to the official bf16 checkpoint (Qwen/Qwen3.8-27B), this module reproduces the official FP8
checkpoint bit for bit: every e4m3 byte and every scale (verified by comparing against the official files).

Format (read from the safetensors headers of the FP8 repo):
  * 407 tensors are float8_e4m3fn: the q/k/v/o_proj of the 16 full-attention layers, the in_proj_qkv/in_proj_z/
    out_proj of the 48 Gated-DeltaNet layers, gate/up/down_proj of all 64 MLPs, and the same 7 Linear weights of
    the single MTP layer (mtp.layers.0.*). Every one of them has both dims divisible by 128.
  * Each one has a companion `<name>_scale_inv`, which is BFLOAT16 (not float32), with shape
    [ceil(out/128), ceil(in/128)]. Dequantization is w ~= fp8 * scale_inv, broadcast over 128x128 blocks.
  * The other 1,199 - 407 = 792 tensors stay bf16 and are byte-identical to the bf16 repo: embed_tokens,
    lm_head, every norm, GDN in_proj_a/in_proj_b/conv1d/A_log/dt_bias/norm, q_norm/k_norm, mtp.fc,
    mtp.*norm*, and the whole vision tower (model.visual.*).

Recipe (found empirically; matches 100% of bytes on every tensor checked):
    amax  = max |w| over each 128x128 block            (w = the bf16 weight; max is exact in any precision)
    s     = amax * fl32(1/448)                          (fp32 multiply by the fp32-rounded reciprocal of 448)
    q     = fp8_e4m3fn_rne( fl32(w / s) )               (IEEE fp32 division by the UNROUNDED fp32 scale)
    scale_inv = bf16_rne(s)                             (the rounded scale is only stored, never used to quantize)
The `* fl32(1/448)` matters: the official files were evidently made with eager PyTorch on CUDA, where
`amax / 448.0` (a tensor divided by a CPU scalar) is computed as `amax * (1/448.0)` in fp32. On CPU the same
expression is a true division, and it changes about 0.1% of the fp8 bytes (ties in blocks whose amax/448 is exact
in fp32). Here the reciprocal is spelled out, so CPU and GPU give the same bytes.

Plain torch only (no vLLM), and device-agnostic. Do not torch.compile `quantize_weight`: Triton's fp32 division
is not correctly rounded, so compiled code can flip fp8 ties.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Mapping

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0                       # torch.finfo(torch.float8_e4m3fn).max
BLOCK = (128, 128)                    # quantization_config.weight_block_size
SCALE_SUFFIX = "_scale_inv"           # "<weight name>_scale_inv"
SCALE_DTYPE = torch.bfloat16          # dtype of the official *_scale_inv tensors
PASSTHROUGH_DTYPE = torch.bfloat16    # dtype of every unquantized tensor in the official checkpoint

# The Linear weights the official checkpoint stores in fp8 (decoder layers and the MTP layer). Everything
# else stays bf16, including in_proj_a/in_proj_b ([48, 5120]), mtp.fc, lm_head, embeddings and the vision tower.
_QUANT_RE = re.compile(
    r"^(?:model\.language_model\.layers|mtp\.layers)\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
    r"\.weight$")


# --- which tensors ---------------------------------------------------------------------------------

def should_quantize(name: str, shape: Iterable[int]) -> bool:
    """True iff the official FP8 checkpoint stores `name` as float8_e4m3fn (with a `<name>_scale_inv`).

    `name` uses checkpoint naming (model.language_model.layers.N..., mtp.layers.0..., model.visual...), which is
    also the naming of Qwen3_5ForConditionalGeneration.state_dict(). Matches the official choice on all 1,199
    weight names of the checkpoint."""
    return len(tuple(shape)) == 2 and _QUANT_RE.match(name) is not None


def scale_name(name: str) -> str:
    return name + SCALE_SUFFIX


# --- one tensor ------------------------------------------------------------------------------------

def _blocks(rows: int, cols: int, block) -> tuple[int, int]:
    return math.ceil(rows / block[0]), math.ceil(cols / block[1])


def quantize_weight(w: torch.Tensor, block=BLOCK, *, input_dtype: torch.dtype | None = torch.bfloat16,
                    check_finite: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D weight -> (w_fp8 [out, in] float8_e4m3fn, scale_inv [ceil(out/b0), ceil(in/b1)] bfloat16).

    Applied to the official bf16 weights, this reproduces the official bytes exactly (see the module docstring
    for the recipe). The result lives on w's device, and CPU and CUDA give identical bytes.

    input_dtype: the weight is first rounded to this dtype (default bf16). The official pipeline quantizes
      the bf16 checkpoint, so with the default, quantizing fp32 master weights gives the same bytes as
      quantizing the bf16 export of those weights. Pass None to quantize the given values as they are.
    Edge blocks (dims not divisible by the block) take amax over their valid part only. The official
      checkpoint has no such tensor, so this case is our own (standard) choice.
    All-zero blocks: s = 0 would give 0/0. They get fp8 code 0 (signed zero kept) and scale_inv = bf16(1/448),
      a positive finite scale (also safe for kernels that take log2 of scales). Again, the official
      checkpoint has no such block (its smallest scale is 7.39e-5).
    check_finite: raise ValueError on inf/NaN weights instead of writing NaN fp8 codes (one host sync)."""
    if w.dim() != 2:
        raise ValueError(f"quantize_weight expects a 2-D weight, got shape {tuple(w.shape)}")
    if not w.is_floating_point():
        raise ValueError(f"quantize_weight expects a floating-point weight, got {w.dtype}")
    b0, b1 = int(block[0]), int(block[1])
    rows, cols = w.shape
    nb0, nb1 = _blocks(rows, cols, (b0, b1))
    if input_dtype is not None and w.dtype != input_dtype:
        w = w.to(input_dtype)
    x = w.float()                                            # exact for bf16 input
    pr, pc = nb0 * b0 - rows, nb1 * b1 - cols
    if pr or pc:                                              # zero padding cannot change max|w|
        x = torch.nn.functional.pad(x, (0, pc, 0, pr))
    xb = x.view(nb0, b0, nb1, b1)
    amax = xb.abs().amax(dim=(1, 3), keepdim=True)            # [nb0, 1, nb1, 1] fp32
    if check_finite and not bool(torch.isfinite(amax).all()):
        raise ValueError("quantize_weight: weight contains inf/NaN")
    inv_max = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32, device=x.device)  # fl32(1/448)
    s = amax * inv_max                                        # == CUDA eager `amax / 448.0`
    s = torch.where(amax == 0, inv_max, s)                    # all-zero block -> s = fl32(1/448), codes stay 0
    q = (xb / s).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)      # IEEE fp32 division, RNE cast (clamp is a no-op on real data)
    q = q.view(nb0 * b0, nb1 * b1)
    if pr or pc:
        q = q[:rows, :cols]
    return q.contiguous(), s.view(nb0, nb1).to(SCALE_DTYPE)


def dequantize_weight(w_fp8: torch.Tensor, scale_inv: torch.Tensor, block=BLOCK,
                      out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """fp8 weight * per-block scale_inv (computed in fp32), returned in out_dtype. This is the
    weight the sampler's block-fp8 GEMM effectively uses, so it can serve as a trainer-side fake-quant
    forward: dequantize_weight(*quantize_weight(w)). Use it as a straight-through estimator if
    gradients must flow."""
    b0, b1 = int(block[0]), int(block[1])
    rows, cols = w_fp8.shape
    nb0, nb1 = _blocks(rows, cols, (b0, b1))
    if tuple(scale_inv.shape) != (nb0, nb1):
        raise ValueError(f"scale_inv shape {tuple(scale_inv.shape)} != {(nb0, nb1)} for weight {(rows, cols)}")
    s = scale_inv.float().repeat_interleave(b0, 0)[:rows].repeat_interleave(b1, 1)[:, :cols]
    return (w_fp8.float() * s).to(out_dtype)


# --- whole state dict ------------------------------------------------------------------------------

def quantize_state_dict(sd: Mapping[str, torch.Tensor], *, block=BLOCK,
                        passthrough_dtype: torch.dtype | None = PASSTHROUGH_DTYPE,
                        check_finite: bool = True) -> dict[str, torch.Tensor]:
    """{checkpoint name: bf16/fp32 tensor} -> {name: tensor} in the official FP8 checkpoint format.

    Quantized weights keep their name (float8_e4m3fn) and gain `<name>_scale_inv` (bf16). Every other
    floating tensor is passed through as passthrough_dtype (bf16, as in the official checkpoint; None keeps
    the dtype). Tensors stay on their device. Key order: input order, each scale right after its weight
    (order does not matter: safetensors sorts the header on save).

    Partial dicts are fine (the vLLM sampler's reload_weights accepts a subset). For a full checkpoint,
    pass every tensor, including the frozen vision tower and MTP layer: the trainer's HF model may lack mtp.*,
    and those tensors must still be in the file."""
    out: dict[str, torch.Tensor] = {}
    for name, t in sd.items():
        if name.endswith(SCALE_SUFFIX):
            raise ValueError(f"{name}: input already contains a scale tensor (already quantized?)")
        if name.startswith(("model.layers.", "layers.")):   # would silently match nothing and stay bf16
            raise ValueError(f"{name}: text-only/unprefixed naming; the FP8 checkpoint uses "
                             "model.language_model.layers.* (Qwen3_5ForConditionalGeneration state_dict keys)")
        if should_quantize(name, t.shape):
            q, s = quantize_weight(t, block, check_finite=check_finite)
            out[name] = q
            out[scale_name(name)] = s
        elif passthrough_dtype is not None and t.is_floating_point() and t.dtype != passthrough_dtype:
            out[name] = t.to(passthrough_dtype)
        else:
            out[name] = t
    return out


def dequantize_state_dict(sd: Mapping[str, torch.Tensor], block=BLOCK,
                          out_dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Inverse format map (for tests and debugging): fp8 + scale_inv -> dequantized out_dtype, other tensors unchanged."""
    out = {}
    for name, t in sd.items():
        if name.endswith(SCALE_SUFFIX):
            continue
        if t.dtype == FP8_DTYPE:
            out[name] = dequantize_weight(t, sd[scale_name(name)], block, out_dtype)
        else:
            out[name] = t
    return out


# --- config.json -----------------------------------------------------------------------------------

def _modules_to_not_convert(layer_types: list[str], vision_depth: int, n_deepstack: int, n_mtp: int) -> list[str]:
    """Rebuilds the official modules_to_not_convert list in its exact order (882 entries for
    Qwen3.8-27B). It lists vision and MTP modules under both HF (model.visual.*) and vLLM (visual.*) names,
    plus MoE leftovers (mlp.gate, mlp.shared_expert_gate) and the fused vLLM in_proj_ba. It is a skip list for
    loaders. The quantized set is defined by should_quantize: embed_tokens/in_proj_qkv etc. are not listed
    under their real names, which is harmless because loaders only consult it for Linear layers."""
    m: list[str] = []
    for i in range(vision_depth):
        m += [f"model.visual.blocks.{i}.attn.proj", f"model.visual.blocks.{i}.attn.qkv",
              f"model.visual.blocks.{i}.mlp.linear_fc1", f"model.visual.blocks.{i}.mlp.linear_fc2",
              f"visual.blocks.{i}.attn.proj", f"visual.blocks.{i}.attn.qkv_proj",
              f"visual.blocks.{i}.mlp.linear_fc1", f"visual.blocks.{i}.mlp.linear_fc2"]
    for i in range(n_deepstack):
        for p in ("model.visual", "visual"):
            m += [f"{p}.deepstack_merger_list.{i}.linear_fc1", f"{p}.deepstack_merger_list.{i}.linear_fc2",
                  f"{p}.deepstack_merger_list.{i}.norm"]
    for p in ("model.visual", "visual"):
        m += [f"{p}.merger.linear_fc1", f"{p}.merger.linear_fc2", f"{p}.merger.norm", f"{p}.patch_embed.proj",
              f"{p}.pos_embed"]
    m += ["visual", "model.visual", "lm_head", "model.embed_tokens"]
    for i, kind in enumerate(layer_types):
        p = f"model.language_model.layers.{i}"
        m += [f"{p}.input_layernorm", f"{p}.mlp.shared_expert_gate", f"{p}.post_attention_layernorm", f"{p}.mlp.gate"]
        if kind == "linear_attention":
            m += [f"{p}.linear_attn.{n}" for n in ("A_log", "conv1d", "dt_bias", "in_proj_ba", "in_proj_b",
                                                   "in_proj_a", "norm")]
        else:
            m += [f"{p}.self_attn.k_norm", f"{p}.self_attn.q_norm"]
    for i in range(n_mtp):
        p = f"mtp.layers.{i}"
        m += [f"{p}.input_layernorm", f"{p}.mlp.gate", f"{p}.mlp.shared_expert_gate", f"{p}.post_attention_layernorm",
              f"{p}.self_attn.k_norm", f"{p}.self_attn.q_norm"]
    m += ["mtp.fc", "mtp.norm", "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"]
    return m


_QWEN38_27B_LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 16


def fp8_quantization_config(layer_types: list[str] | None = None, vision_depth: int = 27, n_deepstack: int = 3,
                            n_mtp: int = 1) -> dict:
    """The `quantization_config` block of the official Qwen3.8-27B-FP8 config.json (equal to it, list order included,
    with the defaults). Put it at the top level of config.json; the rest of the config is the bf16 model's,
    unchanged. For another layout, pass text_config.layer_types / vision depth / mtp_num_hidden_layers."""
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "modules_to_not_convert": _modules_to_not_convert(
            list(layer_types) if layer_types is not None else _QWEN38_27B_LAYER_TYPES, vision_depth, n_deepstack,
            n_mtp),
        "weight_block_size": [BLOCK[0], BLOCK[1]],
    }
