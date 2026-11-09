#!/usr/bin/env python
"""Attention backends for the softmax (full-attention) layers of Qwen3.5-family models in the RL trainer
(fsdp.load(attn_impl=...), --trainer-attn).

Backends, registered with transformers' AttentionInterface so `attn_implementation=<name>` selects them:
- "sdpa"          transformers' own SDPA path (the default, with hf_utils.install_sdpa_causal).
- FA3_NAME        FlashAttention-3 from the HF kernels hub build (kernels-community/flash-attn3), called directly.
                  NOT transformers' native `attn_implementation="kernels-community/flash-attn3"`: in transformers
                  5.2 that path hands Qwen3.5's 3-D mrope position_ids ([3, B, S]) to _flash_attention_forward,
                  whose _is_packed_sequence() then reports a packed batch and builds cu_seqlens [0, S, 2S, 3S] for
                  an S-token tensor (out-of-bounds varlen call).
- FA4_NAME        FlashAttention-4 (flash_attn.cute, CuTe DSL; Blackwell GPUs, head dim 256 included), optionally as
                  torch custom ops (enable_fa4_ops) so selective activation checkpointing can keep its output.
- REF_NAME        the fp32 reference (used by `selftest`): q, k, v cast to fp32, GQA expanded, torch SDPA forced to the
                  memory-efficient backend, output cast back to the compute dtype. If the efficient kernel refuses
                  the shape, a chunked exact fp32 implementation (ExactAttention) is used instead.
All of them take unpadded batch-1 rows only (the trainer's rollouts); packed rows or padding masks are refused.

FA3 kernel source (load_fa3), in this order:
1. a local copy: --fa3-dir / $ARC_FA3_DIR (made with `fetch-fa3` below; no `kernels` package and no hub access at
   run time);
2. the `kernels` package: kernels.get_kernel(FA3_REPO, revision=FA3_REVISION) (needs the hub or a warm cache).

FA4 source (load_fa4): $ARC_FA4_DIR, if set, is added as a site directory first (a `pip install --target DIR` tree
kept out of the training environment, see load_fa4); otherwise the installed `flash_attn.cute` package is used.

CLI:
    python -m arcsolver.rl.attn_backends fetch-fa3 --dest DIR [--revision v1] [--variant VARIANT]   # then ARC_FA3_DIR=DIR
    python -m arcsolver.rl.attn_backends selftest [--fa3-dir DIR]   # loads FA3 and checks it against fp64 (GPU)
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Optional     # enable_fa4_ops: torch.library reads the annotations

import torch
import torch.nn.functional as F

FA3_REPO = "kernels-community/flash-attn3"
FA3_REVISION = "v1"               # branch of the kernel repo (main == v1 for the torch210-cu128 build)
FA3_NAME = "arc_fa3"              # no "flash"/"sdpa" substrings: transformers keys checks on those
REF_NAME = "arc_fp32_ref"

FA3_INFO: dict = {}               # how the kernel was loaded
REF_INFO: dict = {"mode": "efficient", "fallbacks": 0, "fallback_reason": None}
FA3_DETERMINISTIC = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
_FA3 = None


# --- FA3 kernel: variant, fetch, load ------------------------------------------------------------------------

def build_variant() -> str:
    """The kernels-hub build variant of this torch, e.g. torch210-cxx11-cu128-x86_64-linux (as kernels computes it)."""
    if torch.version.cuda is None:
        raise RuntimeError("FA3 needs a CUDA build of torch")
    tv = torch.__version__.split("+")[0].split(".")
    cu = torch.version.cuda.split(".")
    abi = "cxx11" if torch.compiled_with_cxx11_abi() else "cxx98"
    return f"torch{tv[0]}{tv[1]}-{abi}-cu{cu[0]}{cu[1]}-{platform.machine()}-{platform.system().lower()}"


def _http_get(url: str, dest: Path | None = None, timeout: float = 60.0):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "arc-attn-backends/1"})
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if dest is None:
            return r.read()
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dest)
        return None


def _sha256_b64(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return base64.b64encode(h.digest()).decode()


def fetch_fa3(dest: str, revision: str = FA3_REVISION, variant: str | None = None, repo: str = FA3_REPO) -> Path:
    """Download one build variant of the hub kernel to <dest>/build/<variant>/ and verify it against the sha256
    digests in the kernel repo's metadata.json. Tries the kernel-type repo first, then the legacy model repo."""
    variant = variant or build_variant()
    out = Path(dest) / "build" / variant
    out.mkdir(parents=True, exist_ok=True)
    bases = [f"https://huggingface.co/kernels/{repo}/resolve/{revision}/build/{variant}",
             f"https://huggingface.co/{repo}/resolve/{revision}/build/{variant}"]
    meta, base, errs = None, None, []
    for b in bases:
        try:
            meta = json.loads(_http_get(f"{b}/metadata.json"))
            base = b
            break
        except Exception as e:  # noqa: BLE001 - try the next source
            errs.append(f"{b}: {e!r}")
    if meta is None:
        raise RuntimeError("could not fetch metadata.json:\n  " + "\n  ".join(errs))
    digests = (meta.get("digest") or {}).get("files") or {}
    files = sorted(digests) if digests else ["__init__.py", "_ops.py", "flash_attn_interface.py",
                                             "flash_attn_config.py", "flash_attn3/__init__.py",
                                             f"{meta['id']}.abi3.so"]
    for fn in files:
        p = out / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        want = digests.get(fn)
        if p.exists() and want and _sha256_b64(p) == want:
            print(f"[fetch-fa3] ok (cached) {fn}", flush=True)
            continue
        t0 = time.time()
        _http_get(f"{base}/{fn}", p, timeout=600)
        if want and _sha256_b64(p) != want:
            raise RuntimeError(f"sha256 mismatch for {fn}")
        print(f"[fetch-fa3] {fn} {p.stat().st_size / 2**20:.1f} MiB {time.time() - t0:.0f}s"
              f"{' sha256 ok' if want else ' (no digest to verify)'}", flush=True)
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(Path(dest) / "FETCHED.json", "w") as f:
        json.dump({"repo": repo, "revision": revision, "variant": variant, "source": base, "files": files,
                   "verified": bool(digests), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f,
                  indent=1)
    print(f"[fetch-fa3] done: {out}  (export ARC_FA3_DIR={dest})", flush=True)
    return out


def _variant_dir(root: str) -> Path:
    """Accept the variant dir itself, <root>/build/<variant>, or a repo-like root holding build/."""
    r = Path(root)
    if (r / "flash_attn_interface.py").exists():
        return r
    cands = [r / "build" / build_variant(), r / build_variant()] if torch.version.cuda else []
    for c in cands:
        if (c / "flash_attn_interface.py").exists():
            return c
    found = sorted(glob.glob(str(r / "build" / "*")))
    want = build_variant() if torch.version.cuda else "(CPU torch: pass the variant dir itself)"
    raise FileNotFoundError(f"no FA3 build for {want} under {root} (found: {found})")


def _import_package(vdir: Path):
    name = "arc_flash_attn3_" + hashlib.sha1(str(vdir.resolve()).encode()).hexdigest()[:10]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, vdir / "__init__.py", submodule_search_locations=[str(vdir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def load_fa3(fa3_dir: str | None = None, repo: str = FA3_REPO, revision: str = FA3_REVISION):
    """The FA3 kernel module (flash_attn_func, ...); see the module docstring for the source order."""
    global _FA3
    if _FA3 is not None:
        return _FA3
    fa3_dir = fa3_dir or os.environ.get("ARC_FA3_DIR") or None
    if fa3_dir:
        vdir = _variant_dir(fa3_dir)
        mod = _import_package(vdir)
        info = {"source": "local", "path": str(vdir)}
        fetched = Path(fa3_dir) / "FETCHED.json"
        if fetched.exists():
            info["fetched"] = json.loads(fetched.read_text())
    else:
        try:
            import kernels
        except ImportError as e:
            raise ImportError("FA3: set $ARC_FA3_DIR (see `python -m arcsolver.rl.attn_backends fetch-fa3`) or install "
                              "the `kernels` package") from e
        try:
            mod = kernels.get_kernel(repo, revision=revision)
        except TypeError:
            mod = kernels.get_kernel(repo)
        info = {"source": "kernels", "kernels_version": getattr(kernels, "__version__", "?"), "repo": repo,
                "revision": revision}
    if not hasattr(mod, "flash_attn_func"):
        raise ImportError(f"FA3 module {mod} has no flash_attn_func")
    mf = getattr(mod, "__file__", None)
    info["module_file"] = mf
    if mf:
        md = Path(mf).parent / "metadata.json"
        if md.exists():
            try:
                m = json.loads(md.read_text())
                info["metadata"] = {k: m.get(k) for k in ("name", "id", "version", "backend")}
            except ValueError:
                pass
    try:
        info["variant"] = build_variant()
    except RuntimeError as e:
        info["variant"] = f"unknown ({e})"
    FA3_INFO.clear()
    FA3_INFO.update(info)
    _FA3 = mod
    return mod


# --- attention functions (transformers AttentionInterface signature) ------------------------------------------

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


_UNPACKED_OK: set = set()


def _check_unpacked(position_ids, S: int) -> None:
    """The FA3 path ignores position ids (rotary is applied before the kernel), so it must not see packed rows
    (position resets): those would need varlen cu_seqlens. Checked once per shape (one host sync)."""
    if position_ids is None:
        return
    p = position_ids[0] if position_ids.dim() == 3 else position_ids          # mrope [3, B, S] -> text row
    key = (tuple(p.shape), str(p.device))
    if key in _UNPACKED_OK:
        return
    if p.shape[-1] != S:
        raise ValueError(f"position_ids length {p.shape[-1]} != query length {S}")
    ar = torch.arange(S, device=p.device, dtype=p.dtype)
    if not bool(torch.equal(p, ar[None, :].expand_as(p) + p[:, :1])):
        raise ValueError("FA3 path: packed sequences (position-id resets) are not supported; use sdpa")
    _UNPACKED_OK.add(key)


def fa3_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                          sliding_window=None, softcap=None, is_causal=None, **kwargs):
    """FlashAttention-3 for one unpadded batch (the trainers' batch-1 rows). query [B, Hq, S, D], key/value
    [B, Hkv, S, D] (GQA handled by the kernel); returns ([B, S, Hq, D], None) like transformers' flash path."""
    if attention_mask is not None:
        raise ValueError("FA3 path: an attention mask (padding) was passed; only unpadded rows are supported")
    if dropout:
        raise ValueError("FA3 path: attention dropout is not supported (FA3 has none)")
    if sliding_window is not None or softcap:
        raise NotImplementedError("FA3 path: sliding window / softcap are not wired (not used by Qwen3.5)")
    if any(kwargs.get(k) is not None for k in ("cu_seq_lens_q", "cu_seq_lens_k")):
        raise ValueError("FA3 path: varlen (packed) kwargs are not supported")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"FA3 path expects bf16/fp16 q/k/v, got {query.dtype}")
    S = query.shape[2]
    _check_unpacked(kwargs.get("position_ids"), S)
    causal = (is_causal if is_causal is not None else getattr(module, "is_causal", True)) and S > 1
    fa3 = _FA3 if _FA3 is not None else load_fa3()
    out = fa3.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                              softmax_scale=scaling, causal=causal, deterministic=FA3_DETERMINISTIC)
    if isinstance(out, tuple):
        out = out[0]
    return out, None


FA4_NAME = "arc_fa4"
_FA4 = None


def load_fa4(fa4_dir: str | None = None):
    """FlashAttention-4 (flash-attn-4, CuTe DSL; Blackwell, head dim 256 included). fa4_dir / $ARC_FA4_DIR: a side
    directory made with `pip install --no-deps --target DIR flash-attn-4==4.0.0b36 nvidia-cutlass-dsl==4.7.1
    nvidia-cutlass-dsl-libs-base==4.7.1 nvidia-cutlass-dsl-libs-core==4.7.1 nvidia-cutlass-dsl-libs-cu12==4.7.1
    apache-tvm-ffi torch-c-dlpack-ext quack-kernels==0.6.5`, which keeps it out of the training environment
    (site.addsitedir runs the DSL's .pth). Neither given: the installed flash_attn.cute package."""
    global _FA4
    if _FA4 is None:
        import site
        d = fa4_dir or os.environ.get("ARC_FA4_DIR") or ""
        if d and os.path.isdir(d):
            site.addsitedir(d)
        from flash_attn.cute import flash_attn_func
        _FA4 = flash_attn_func
    return _FA4


_FA4_OPS = False


def enable_fa4_ops() -> None:
    """FA4's forward and backward as torch custom ops fa4sac::fwd / fa4sac::bwd (opt-in, used with --trainer-attn fa4
    and --trainer-speed sac_attn): the same flash_attn.cute kernels with the same arguments FlashAttnFunc passes for these
    unpadded rows (no window / softcap / sink / mods, num_splits 1), so values are unchanged, but the forward is now an
    operator that selective checkpointing (trainer_speed.sac_policy) can keep: the backward's recompute of a checkpointed
    attention layer reuses the forward's out + lse instead of rerunning the kernel. FlashAttnFunc itself is a Python
    autograd.Function around CuTe launches, which the checkpoint policy never sees. Idempotent."""
    global _FA4_OPS
    if _FA4_OPS:
        return
    load_fa4()
    from flash_attn.cute import interface as fai
    _fwd, _bwd = fai._flash_attn_fwd, fai._flash_attn_bwd

    @torch.library.custom_op("fa4sac::fwd", mutates_args=())
    def fa4sac_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: Optional[float],
                   causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # FA4 chooses its kernel variant and allocates lse from requires_grad: as FlashAttnFunc does in training
        qq, kk, vv = (t.detach().requires_grad_() for t in (q, k, v))
        out, lse = _fwd(qq, kk, vv, qv=None, softmax_scale=softmax_scale, causal=causal, window_size_left=None,
                        window_size_right=None, learnable_sink=None, softcap=0.0, num_splits=1, pack_gqa=None,
                        score_mod=None, mask_mod=None, aux_tensors=None, aux_scalars=None, block_sparse_tensors=None,
                        return_lse=False, gather_kv_indices=None, gather_bwd_recompute_p=False)[:2]
        return out, lse

    @torch.library.custom_op("fa4sac::bwd", mutates_args=())
    def fa4sac_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                   lse: torch.Tensor, softmax_scale: Optional[float], causal: bool
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = _bwd(q, k, v, out, dout, lse, softmax_scale, causal, 0.0, window_size_left=None,
                          window_size_right=None, deterministic=False, score_mod=None, score_mod_bwd=None,
                          mask_mod=None, aux_tensors=None, aux_scalars=None, block_sparse_tensors=None, dlse=None,
                          learnable_sink=None)
        return dq, dk, dv

    def setup(ctx, inputs, output):
        q, k, v, scale, causal = inputs
        ctx.save_for_backward(q, k, v, *output)
        ctx.scale, ctx.causal = scale, causal

    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = torch.ops.fa4sac.bwd(dout, q, k, v, out, lse, ctx.scale, ctx.causal)
        return dq, dk, dv, None, None

    torch.library.register_autograd("fa4sac::fwd", backward, setup_context=setup)
    _FA4_OPS = True


def fa4_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                          sliding_window=None, softcap=None, is_causal=None, **kwargs):
    """FlashAttention-4 for one unpadded batch, as fa3_attention_forward (query [B, Hq, S, D], key/value [B, Hkv, S, D],
    GQA in the kernel). On Blackwell at head dim 256, causal: forward ~4x / backward ~2.3x torch's flash backend, with
    the same error against an fp32 reference (rel 2-6e-3 on out / dq / dk / dv at 4k tokens)."""
    if attention_mask is not None:
        raise ValueError("FA4 path: an attention mask (padding) was passed; only unpadded rows are supported")
    if dropout:
        raise ValueError("FA4 path: attention dropout is not supported")
    if sliding_window is not None or softcap:
        raise NotImplementedError("FA4 path: sliding window / softcap are not wired (not used by Qwen3.5)")
    if any(kwargs.get(k) is not None for k in ("cu_seq_lens_q", "cu_seq_lens_k")):
        raise ValueError("FA4 path: varlen (packed) kwargs are not supported")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"FA4 path expects bf16/fp16 q/k/v, got {query.dtype}")
    S = query.shape[2]
    _check_unpacked(kwargs.get("position_ids"), S)
    causal = (is_causal if is_causal is not None else getattr(module, "is_causal", True)) and S > 1
    if _FA4_OPS:     # enable_fa4_ops: the same kernels as custom ops the selective checkpoint can keep
        out, _ = torch.ops.fa4sac.fwd(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scaling, causal)
        return out, None
    out = load_fa4()(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), softmax_scale=scaling, causal=causal)
    if isinstance(out, tuple):
        out = out[0]
    return out, None


def ref_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None,
                          is_causal=None, **kwargs):
    """fp32 reference: q/k/v -> fp32, GQA expanded, SDPA forced to the memory-efficient kernel (on CPU: math);
    output cast back to the input dtype. Returns ([B, S, Hq, D], None)."""
    if attention_mask is not None:
        raise ValueError("fp32 reference: masks are not supported (unpadded batch-1 rows only)")
    if dropout:
        raise ValueError("fp32 reference: dropout must be 0")
    S = query.shape[2]
    causal = (is_causal if is_causal is not None else getattr(module, "is_causal", True)) and S > 1
    if causal and key.shape[2] != S:
        raise ValueError("fp32 reference: causal attention needs square q/k lengths")
    n_rep = query.shape[1] // key.shape[1]
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if REF_INFO["mode"] != "exact":
        q = query.float()
        k = _repeat_kv(key.float(), n_rep)
        v = _repeat_kv(value.float(), n_rep)
        backend = SDPBackend.EFFICIENT_ATTENTION if q.is_cuda else SDPBackend.MATH   # CPU: smoke tests only
        try:
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
            return out.to(query.dtype).transpose(1, 2).contiguous(), None
        except RuntimeError as e:          # "No available kernel" for this shape/dtype: fall back, but say so
            REF_INFO["fallbacks"] += 1
            REF_INFO["fallback_reason"] = str(e)[:500]
            del q, k, v
    out = ExactAttention.apply(query, key, value, scale, causal, torch.float32)
    return out.transpose(1, 2).contiguous(), None


# --- exact chunked attention (fp64 / fp32), forward and backward -----------------------------------------------

def _rows_per_chunk(rep: int, kv_len: int, acc: torch.dtype, budget: int, n_tmp: int = 3) -> int:
    per_row = max(1, n_tmp * rep * kv_len * torch.finfo(acc).bits // 8)
    c = max(16, budget // per_row)
    return int(c // 16 * 16)


def _causal_fill_(sc: torch.Tensor, s0: int, s1: int) -> None:
    """sc [rep, c, s1] holds rows s0..s1-1 against keys 0..s1-1: mask keys j > i (only the last c columns)."""
    c = s1 - s0
    tri = torch.ones(c, c, dtype=torch.bool, device=sc.device).triu_(1)
    sc[..., s0:s1].masked_fill_(tri, float("-inf"))


@torch.no_grad()
def exact_forward(q, k, v, scale: float, causal: bool = True, acc=torch.float64, budget: int = 4 << 30):
    """Chunked softmax attention in `acc` precision. q [B, H, S, D], k/v [B, Hkv, S, D] (any dtype).
    Returns out [B, H, S, D] and lse [B, H, S] in `acc`. Never holds more than a [rep, rows, S] score block."""
    B, H, S, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    if causal and Sk != S:
        raise ValueError("causal exact attention needs square q/k")
    rep = H // Hkv
    out = torch.empty(B, H, S, D, dtype=acc, device=q.device)
    lse = torch.empty(B, H, S, dtype=acc, device=q.device)
    c = _rows_per_chunk(rep, Sk, acc, budget)
    for b in range(B):
        for g in range(Hkv):
            hs = slice(g * rep, (g + 1) * rep)
            kg, vg = k[b, g].to(acc), v[b, g].to(acc)
            for s0 in range(0, S, c):
                s1 = min(S, s0 + c)
                ke = s1 if causal else Sk
                sc = torch.matmul(q[b, hs, s0:s1].to(acc), kg[:ke].T).mul_(scale)
                if causal:
                    _causal_fill_(sc, s0, s1)
                m = sc.amax(-1, keepdim=True)
                sc.sub_(m).exp_()
                l = sc.sum(-1, keepdim=True)
                sc.div_(l)
                out[b, hs, s0:s1] = torch.matmul(sc, vg[:ke])
                lse[b, hs, s0:s1] = (m + l.log()).squeeze(-1)
                del sc
    return out, lse


@torch.no_grad()
def exact_backward(q, k, v, out, lse, dout, scale: float, causal: bool = True, acc=torch.float64,
                   budget: int = 4 << 30):
    """Gradients of exact_forward for upstream dout [B, H, S, D]: (dq [B,H,S,D], dk, dv [B,Hkv,S,D]) in `acc`.
    Standard softmax backward: P = exp(S - lse), dV = P^T dO, dP = dO V^T, Delta = rowsum(dO * O),
    dS = P * (dP - Delta), dQ = scale dS K, dK = scale dS^T Q."""
    B, H, S, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    rep = H // Hkv
    dq = torch.zeros(B, H, S, D, dtype=acc, device=q.device)
    dk = torch.zeros(B, Hkv, Sk, D, dtype=acc, device=q.device)
    dv = torch.zeros(B, Hkv, Sk, D, dtype=acc, device=q.device)
    c = _rows_per_chunk(rep, Sk, acc, budget)
    for b in range(B):
        for g in range(Hkv):
            hs = slice(g * rep, (g + 1) * rep)
            kg, vg = k[b, g].to(acc), v[b, g].to(acc)
            for s0 in range(0, S, c):
                s1 = min(S, s0 + c)
                ke = s1 if causal else Sk
                n = s1 - s0
                qc = q[b, hs, s0:s1].to(acc)
                doc = dout[b, hs, s0:s1].to(acc)
                p = torch.matmul(qc, kg[:ke].T).mul_(scale)
                if causal:
                    _causal_fill_(p, s0, s1)
                p.sub_(lse[b, hs, s0:s1, None].to(acc)).exp_()
                dv[b, g, :ke] += torch.matmul(p.reshape(rep * n, ke).T, doc.reshape(rep * n, D))
                dp = torch.matmul(doc, vg[:ke].T)
                delta = (doc * out[b, hs, s0:s1].to(acc)).sum(-1, keepdim=True)
                dp.sub_(delta).mul_(p)                       # dS
                del p
                dq[b, hs, s0:s1] = torch.matmul(dp, kg[:ke]).mul_(scale)
                dk[b, g, :ke] += torch.matmul(dp.reshape(rep * n, ke).T, qc.reshape(rep * n, D)).mul_(scale)
                del dp
    return dq, dk, dv


def exact_attention(q, k, v, dout, scale: float, causal: bool = True, acc=torch.float64, budget: int = 4 << 30):
    """(out, dq, dk, dv) of causal GQA softmax attention in `acc` (fp64 = the paper's reference)."""
    out, lse = exact_forward(q, k, v, scale, causal, acc, budget)
    dq, dk, dv = exact_backward(q, k, v, out, lse, dout, scale, causal, acc, budget)
    return out, dq, dk, dv


class ExactAttention(torch.autograd.Function):
    """Autograd wrapper of exact_forward/backward (the fp32 reference's fallback). In: q [B,H,S,D], k/v [B,Hkv,S,D];
    out [B, H, S, D] in the input dtype; gradients in the input dtypes."""

    @staticmethod
    def forward(ctx, q, k, v, scale, causal, acc):
        out, lse = exact_forward(q, k, v, scale, causal, acc)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale, ctx.causal, ctx.acc = scale, causal, acc
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = exact_backward(q, k, v, out, lse, dout, ctx.scale, ctx.causal, ctx.acc)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


# --- error measure (selftest) ------------------------------------------------------------------------------------

@torch.no_grad()
def rel_err_pct(x: torch.Tensor, ref: torch.Tensor, chunk: int = 1 << 26) -> float:
    """100 * ||x - ref||_2 / ||ref||_2, accumulated in fp64 over flat chunks (x and ref may live on different
    devices; computed on x's device)."""
    if x.shape != ref.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(ref.shape)}")
    xf, rf = x.reshape(-1), ref.reshape(-1)
    num = torch.zeros((), dtype=torch.float64, device=x.device)
    den = torch.zeros((), dtype=torch.float64, device=x.device)
    for i in range(0, xf.numel(), chunk):
        a = xf[i:i + chunk].to(torch.float64)
        b = rf[i:i + chunk].to(device=x.device, dtype=torch.float64)
        num += (a - b).square().sum()
        den += b.square().sum()
    num, den = float(num), float(den)
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return 100.0 * math.sqrt(num / den)


# --- registration ------------------------------------------------------------------------------------------------

def _register(name: str, fn) -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(name, fn)
    # flash-style mask: None for unpadded rows (the functions above refuse anything else)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, flash_attention_mask)


def register_fa3(fa3_dir: str | None = None, revision: str = FA3_REVISION) -> str:
    """Load the FA3 kernel and register FA3_NAME; returns the attn_implementation string for from_pretrained."""
    load_fa3(fa3_dir, revision=revision)
    _register(FA3_NAME, fa3_attention_forward)
    return FA3_NAME


def register_fa4(fa4_dir: str | None = None) -> str:
    """Load FA4 and register FA4_NAME; returns the attn_implementation string for from_pretrained."""
    load_fa4(fa4_dir)
    _register(FA4_NAME, fa4_attention_forward)
    return FA4_NAME


def register_ref() -> str:
    _register(REF_NAME, ref_attention_forward)
    return REF_NAME


def raw_function(backend: str):
    """The attention function of a backend key ("sdpa" | "fa3" | "fa4" | "ref")."""
    if backend == "sdpa":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward
    return {"fa3": fa3_attention_forward, "fa4": fa4_attention_forward, "ref": ref_attention_forward}[backend]


# --- CLI -------------------------------------------------------------------------------------------------------------

def _selftest(args) -> int:
    """Load FA3 and compare it (and SDPA bf16, and the fp32 reference) to fp64 on a small causal GQA problem."""
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    register_fa3(args.fa3_dir, args.revision)
    print("FA3:", json.dumps(FA3_INFO, indent=1, default=str))
    g = torch.Generator(device=dev).manual_seed(0)
    B, H, Hkv, S, D = 1, 16, 4, args.len, 256
    q = torch.randn(B, H, S, D, device=dev, generator=g).bfloat16()
    k = torch.randn(B, Hkv, S, D, device=dev, generator=g).bfloat16()
    v = torch.randn(B, Hkv, S, D, device=dev, generator=g).bfloat16()
    do = torch.randn(B, S, H, D, device=dev, generator=g).bfloat16()
    scale = D ** -0.5

    class M(torch.nn.Module):
        is_causal, num_key_value_groups, layer_idx = True, H // Hkv, 0
    m = M()
    o64, dq64, dk64, dv64 = exact_attention(q, k, v, do.transpose(1, 2), scale)
    for nm in ("sdpa", "fa3", "ref"):
        ql, kl, vl = (t.clone().requires_grad_(True) for t in (q, k, v))
        o = raw_function(nm)(m, ql, kl, vl, None, dropout=0.0, scaling=scale)[0]
        o.backward(do)
        print(f"{nm:5s} out {rel_err_pct(o.transpose(1, 2), o64):.4f}%  dq {rel_err_pct(ql.grad, dq64):.4f}%  "
              f"dk {rel_err_pct(kl.grad, dk64):.4f}%  dv {rel_err_pct(vl.grad, dv64):.4f}%")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-fa3", help="download + verify the FA3 hub build for this torch into --dest")
    f.add_argument("--dest", required=True)
    f.add_argument("--revision", default=FA3_REVISION)
    f.add_argument("--variant", default=None, help="default: this torch's variant, e.g. torch210-cxx11-cu128-x86_64-linux")
    f.add_argument("--repo", default=FA3_REPO)
    s = sub.add_parser("selftest", help="load FA3 (GPU) and check it against fp64 on a small problem")
    s.add_argument("--fa3-dir", default=None)
    s.add_argument("--revision", default=FA3_REVISION)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--len", type=int, default=2048)
    a = ap.parse_args(argv)
    if a.cmd == "fetch-fa3":
        fetch_fa3(a.dest, a.revision, a.variant, a.repo)
        return 0
    return _selftest(a)


if __name__ == "__main__":
    sys.exit(main())
