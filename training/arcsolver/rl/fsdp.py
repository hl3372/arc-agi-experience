"""FSDP2 policy for multi-GPU RL (train.py).

fp32 master parameters sharded across ranks, bf16 compute, fp32 gradient reduction, gradient checkpointing, the SDPA
causal hook; the vision tower is dropped for training (frozen, unused for text) and its tensors (and the MTP head's)
come from the base checkpoint whenever weights are written for vLLM.

Gradient accumulation is an fp32 gradient sum: every rollout's bf16 gradient is reduce-scattered in fp32 and added to
the sharded fp32 gradient (requires_gradient_sync stays on); parameters stay gathered across a rank's rollouts and are
resharded after its last one. Every rank must therefore run the same number of backward passes per step (padding with
zero-loss dummies), unless --balance lpt accumulates unsharded and reduce-scatters once per minibatch.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time

import torch
import torch.nn as nn

from arcsolver.rl import hf_utils as tf
from arcsolver.rl.loss import token_logprobs
from arcsolver.rl.sc import token_and_head_logprobs


class _Offloaded:
    __slots__ = ("host", "d2h", "gpu", "h2d", "device", "idx")

    def __init__(self, host, d2h, device, idx):
        self.host, self.d2h, self.gpu, self.h2d, self.device, self.idx = host, d2h, None, None, device, idx


class ActivationOffload:
    """Saved-tensor hooks (for large models at long contexts) that keep large saved activations in pinned host memory
    between forward and backward. With non-reentrant checkpointing the tensors saved outside the checkpointed decoder
    layers are the layers' inputs (T x hidden bf16 each: 64 x 1.3 GB at 130k tokens for a 64-layer model, the bulk of
    the backward peak); inside a layer the checkpoint's own hooks take precedence, so recompute buffers stay on the GPU.
    Forward: each tensor >= min_bytes that is not a parameter is copied to a pinned slab on a side stream (record_stream keeps its GPU block
    alive until the copy lands). Backward: it is copied back on the side stream on first use, and the tensor saved
    just before it is prefetched, so the transfers overlap the previous layer's recompute and backward. The values
    are bit-identical; only where they wait changes. Pinned slabs are 4 GiB chunks (a power of two, so the caching
    host allocator does not round them up), reused from the start by every forward."""

    CHUNK = 4 << 30
    ALIGN = 512

    def __init__(self, min_bytes: int = 64 << 20):
        self.min_bytes = min_bytes
        self.stream = None
        self.chunks = []
        self.items = []
        self.pos = (0, 0)
        self.bytes_out = 0

    def begin(self):
        """Start a forward: reuse the slabs from the top (the previous backward has consumed its tensors)."""
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        self.items, self.pos, self.bytes_out = [], (0, 0), 0

    def release(self):
        """Drop the pinned slabs after a step's last backward: kept, they would sit pinned through checkpointing and
        sampling and starve the host. The next forward re-pins (~10 s)."""
        torch.cuda.synchronize()
        self.chunks, self.items = [], []
        release_host_cache()

    def _carve(self, nbytes: int):
        if nbytes > self.CHUNK:
            raise ValueError(f"activation of {nbytes} bytes exceeds the {self.CHUNK}-byte offload chunk")
        ci, off = self.pos
        if ci < len(self.chunks) and off + nbytes > self.CHUNK:
            ci, off = ci + 1, 0
        if ci == len(self.chunks):
            self.chunks.append(torch.empty(self.CHUNK, dtype=torch.uint8, pin_memory=True))
        self.pos = (ci, off + -(-nbytes // self.ALIGN) * self.ALIGN)
        return self.chunks[ci][off:off + nbytes]

    def pack(self, t):
        nbytes = t.numel() * t.element_size()
        if not t.is_cuda or nbytes < self.min_bytes or not t.is_contiguous() or (t.requires_grad and t.grad_fn is None):
            return t            # parameters (e.g. the gathered lm_head every logprob chunk saves) stay with FSDP
        host = self._carve(nbytes).view(t.dtype).view(t.shape)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            host.copy_(t, non_blocking=True)
            d2h = torch.cuda.Event()
            d2h.record()
        t.record_stream(self.stream)
        item = _Offloaded(host, d2h, t.device, len(self.items))
        self.items.append(item)
        self.bytes_out += nbytes
        return item

    def _fetch(self, item):
        if item.gpu is not None:
            return
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(item.d2h)
            g = torch.empty(item.host.shape, dtype=item.host.dtype, device=item.device)
            g.copy_(item.host, non_blocking=True)
            item.h2d = torch.cuda.Event()
            item.h2d.record()
        item.gpu = g

    def unpack(self, item):
        if not isinstance(item, _Offloaded):
            return item
        self._fetch(item)
        cur = torch.cuda.current_stream()
        cur.wait_event(item.h2d)
        g, item.gpu = item.gpu, None
        g.record_stream(cur)
        if item.idx > 0:                                   # backward runs the layers in reverse
            self._fetch(self.items[item.idx - 1])
        return g


class CompletionLogprobs(nn.Module):
    """log p(completion token j | prompt, completion[:j]) in fp32 [C], lm_head applied in checkpointed chunks."""

    def __init__(self, hf, chunk: int = 2048):
        super().__init__()
        self.hf = hf
        self.chunk = chunk
        self.offload = None              # an ActivationOffload (set_activation_offload)
        self.head_fn = None              # trainer_speed.token_and_head_logprobs_fused with --trainer-speed fused_head

    def forward(self, input_ids, prompt_len: int, head_ids=None):
        """log p(completion tokens) [C]; with head_ids [C, k] (the sampler's top-k at each completion position)
        also log p at those ids: (lp_y [C], lp_head [C, k]) for score centering (sc.py)."""
        if self.offload is not None and torch.is_grad_enabled():
            # only the decoder stack: the head's chunks each save the (padded) lm_head, which must stay put
            self.offload.begin()
            with torch.autograd.graph.saved_tensors_hooks(self.offload.pack, self.offload.unpack):
                h = self.hf.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        else:
            h = self.hf.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        S = input_ids.shape[1]
        if self.head_fn is not None:     # fused head (fp32 logsumexp without fp32 logits); no head ids = k of 0
            hid = head_ids if head_ids is not None else input_ids.new_empty((S - prompt_len, 0))
            lp_y, lp_h = self.head_fn(h[prompt_len - 1:S - 1], self.hf.lm_head.weight, input_ids[0, prompt_len:S], hid,
                                      chunk=self.chunk)
            return (lp_y, lp_h) if head_ids is not None else lp_y
        if head_ids is not None:
            return token_and_head_logprobs(h[prompt_len - 1:S - 1], self.hf.lm_head.weight, input_ids[0, prompt_len:S],
                                           head_ids, chunk=self.chunk)
        return token_logprobs(h[prompt_len - 1:S - 1], self.hf.lm_head.weight, input_ids[0, prompt_len:S],
                              chunk=self.chunk)


SPEED_OPTIONS = ("sac_attn", "fused_head", "gdn_gva", "compile_norm_mlp")


def load(path: str, device, chunk: int = 2048, keep_gathered: bool = True, freeze_embed: bool = False,
         attn_impl: str = "sdpa", speed: str = "", frozen: bool = False):
    """The policy as an FSDP2 module: fp32 master parameters sharded over the ranks, bf16 compute, fp32 gradient
    reduction. Loaded in the checkpoint's bf16 on CPU, then each decoder layer is moved to the GPU in fp32 and
    sharded at once, so a GPU never holds more than one full fp32 layer. The vision tower is dropped.
    keep_gathered=False reshards every layer after its forward and backward (for models whose gathered bf16 weights
    do not fit next to the activations); freeze_embed freezes the input embedding table.
    attn_impl: "sdpa" (default: transformers' SDPA + the causal hook), "fa3" (FlashAttention-3, kernels-hub build) or
    "fa4" (FlashAttention-4 CuTe, Blackwell), via attn_backends.py; unpadded batch-1 rows only.
    speed: comma-separated opt-in speedups from trainer_speed.py, each equal
    to the default path up to bf16 rounding order: sac_attn (the attention layers' checkpoint keeps the attention
    kernel outputs, so the backward recompute skips the quadratic forward; +0.8 GB per attention layer at 98k),
    fused_head (lm_head + fp32 log-softmax + top-k gather without fp32 logits; use a larger chunk, e.g. 8192),
    gdn_gva (GatedDeltaNet q/k stay at 16 heads; fla maps the 32 value heads itself), compile_norm_mlp
    (torch.compile of the RMSNorm and SwiGLU elementwise chains).
    frozen (the reference model for --kl-coef): every parameter frozen, kept in bf16 (no fp32 masters),
    eval mode, no gradient checkpointing, never resharded by a forward: ranks run different numbers of reference
    forwards (--balance lpt), so the gather must be one explicit collective (gather_frozen) and the release a local
    one (reshard_frozen); call it under torch.no_grad()."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoModelForImageTextToText
    attn = "sdpa"
    opts = [o for o in (speed or "").split(",") if o]
    bad = [o for o in opts if o not in SPEED_OPTIONS]
    if bad:
        raise ValueError(f"unknown --trainer-speed options {bad} (known: {SPEED_OPTIONS})")
    if attn_impl == "fa3":
        from arcsolver.rl import attn_backends
        attn = attn_backends.register_fa3()      # loads the kernel ($ARC_FA3_DIR, else the `kernels` package)
    elif attn_impl == "fa4":                     # Blackwell: FlashAttention-4 ($ARC_FA4_DIR or the installed flash_attn)
        from arcsolver.rl import attn_backends
        attn = attn_backends.register_fa4()
        if "sac_attn" in opts:                   # FA4 as custom ops, so selective checkpointing can keep its output
            attn_backends.enable_fa4_ops()
    elif attn_impl != "sdpa":
        raise ValueError(f"attn_impl must be 'sdpa', 'fa3' or 'fa4', got {attn_impl!r}")
    hf = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn)
    if getattr(hf.model, "visual", None) is not None:
        hf.model.visual = None
    hf.config.use_cache = False
    if frozen:
        hf.eval()
        hf.requires_grad_(False)
    else:
        hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hf.train()
    if freeze_embed:
        hf.get_input_embeddings().weight.requires_grad_(False)
    wrapper = CompletionLogprobs(hf, chunk)
    if opts:
        from arcsolver.rl import trainer_speed as TS
        if "gdn_gva" in opts:
            TS.enable_gdn_gva(True)
        if "compile_norm_mlp" in opts:
            TS.enable_compiled_norm_mlp()
        if "sac_attn" in opts:
            TS.sac_attention(hf)
        if "fused_head" in opts:
            wrapper.head_fn = TS.token_and_head_logprobs_fused
    wrapper.speed = tuple(opts)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    if frozen:
        keep_gathered = True

    def bf16_params(mod):                      # frozen: bf16 weights (= the trainer's bf16 compute copy), fp32 buffers
        for prm in mod.parameters():
            if not hasattr(prm, "full_tensor"):
                prm.data = prm.data.to(torch.bfloat16)

    for layer in tf.text_layers(hf):
        layer.to(device=device, dtype=torch.float32)
        if frozen:
            bf16_params(layer)
        fully_shard(layer, mp_policy=mp, reshard_after_forward=not keep_gathered)
    if frozen:                                 # as the trainer's wrapper.to(float32) for the buffers (rotary inv_freq)
        wrapper.to(device=device)
        for mod in wrapper.modules():
            for k, buf in list(mod._buffers.items()):
                if buf is not None and buf.is_floating_point():
                    mod._buffers[k] = buf.to(device=device, dtype=torch.float32)
        bf16_params(wrapper)
    else:
        wrapper.to(device=device, dtype=torch.float32)
    fully_shard(wrapper, mp_policy=mp, reshard_after_forward=not keep_gathered)
    wrapper.keep_gathered = keep_gathered
    return wrapper


def trainer_logprobs(model, device, items):
    """Per-token log-probs of each item's completion under the trainer's current weights, no grad (--ppo-old-logp
    trainer: DAPO clip-higher against the policy itself, not the FP8 sampler). Keep-gathered FSDP:
    gather_frozen gathers the parameters (collective), the per-item forwards need no communication (any count per
    rank), released locally after; otherwise every forward is a collective: call with the same
    number of items on every rank (None = a padding forward). Returns [fp32 tensor [T] on device | None]."""
    out = []
    keep = bool(getattr(model, "keep_gathered", True))
    with torch.no_grad():
        if keep:
            gather_frozen(model, device)
        for it in items:
            if it is None:
                model(torch.tensor([[0, 0, 0]], device=device), 1)
                out.append(None)
                continue
            ids = torch.tensor([list(it["prompt_ids"]) + list(it["completion"])], device=device)
            out.append(model(ids, len(it["prompt_ids"])).float())
        if keep:
            reshard_frozen(model)
    return out


@torch.no_grad()
def gather_frozen(model, device):
    """Collective (every rank, the same number of calls): one tiny forward all-gathers a frozen keep-gathered model's
    parameters, which then stay unsharded, so per-rollout forwards need no communication."""
    model(torch.tensor([[0, 0]], device=device), 1)


def reshard_frozen(model):
    """Local: free the unsharded parameters of every FSDP module of a frozen model (before offload / between steps)."""
    from torch.distributed.fsdp import FSDPModule
    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.reshard()


def release_host_cache():
    """Return torch's cached pinned host blocks to the OS. Freed pinned memory (offload slabs, the CPU copies of
    the Adam moments after they move back to the GPU) is otherwise kept for reuse and counts against host RAM."""
    fn = getattr(torch._C, "_host_emptyCache", None)
    if fn is not None:
        fn()


def set_activation_offload(model, min_bytes: int = 64 << 20):
    """Offload the large saved activations of every grad-enabled forward to pinned host memory (ActivationOffload);
    min_bytes <= 0 turns it off."""
    model.offload = ActivationOffload(min_bytes) if min_bytes > 0 else None
    return model.offload


def optimizer(model, lr: float, eps: float, beta2: float = 0.999, weight_decay: float = 0.0):
    """AdamW over the sharded fp32 parameters. weight_decay 0 (default): no decay, one param group. weight_decay > 0:
    decoupled decay on the matrices only, as the SFT trainer (hf_utils.param_groups: ndim >= 2, no norms); two
    groups."""
    if weight_decay > 0:
        groups = tf.param_groups(model, weight_decay)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, beta2), eps=eps, foreach=True)


def begin_step(model):
    """Every rollout's gradient is reduce-scattered (fp32) as it is produced. With keep_gathered the parameters stay
    gathered across a rank's rollouts (resharded after the last one); without, every layer reshards after its
    backward."""
    model.set_requires_gradient_sync(True)
    model.set_reshard_after_backward(not getattr(model, "keep_gathered", True))


def last_micro(model):
    model.set_reshard_after_backward(True)


# --- time-sharing the GPU with the sampler ---------------------------------------------------------------

@torch.no_grad()
def offload(model, opt):
    """Move the sharded fp32 parameters and the AdamW state to CPU (gradients must be None), freeing the GPU for
    the sampler; FSDP2's module.to() keeps its sharded views consistent."""
    model.to("cpu")
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v) and v.device.type != "cpu":
                st[k] = v.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@torch.no_grad()
def reload(model, opt, device, *, optimizer_state=True):
    model.to(device)
    if optimizer_state:
        reload_optimizer(opt, device)
    torch.cuda.synchronize()


@torch.no_grad()
def offload_optimizer(opt):
    """AdamW state back to CPU between minibatch updates (the moments stay off the GPU through backward)."""
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v) and v.device.type != "cpu":
                st[k] = v.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@torch.no_grad()
def reload_optimizer(opt, device):
    """Restore Adam moments only after backward releases its long-sequence activations."""
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v) and v.device.type == "cpu" and k != "step":
                st[k] = v.to(device, non_blocking=True)
    torch.cuda.synchronize()


def dummy_backward(model, device):
    ids = torch.tensor([[0, 0, 0]], device=device)
    (model(ids, 1).sum() * 0.0).backward()


def grad_norm_clip(model, max_norm: float) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    value = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite gradient norm {value}; refusing optimizer update")
    return value


# --- weights out --------------------------------------------------------------------------------------------

def _base_map(base: str) -> dict:
    idx = os.path.join(base, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    from safetensors import safe_open
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def full_bf16_state(model, rank: int) -> dict | None:
    """Collective: every rank calls; rank 0 gets {checkpoint key: bf16 cpu tensor} of the trained parameters."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank != 0:
        return None
    return {k[len("hf."):]: v.to(torch.bfloat16).contiguous() for k, v in sd.items() if k.startswith("hf.")}


class SyncWriter:
    """Weights for vLLM's reload_weights: <dir>/model-trained.safetensors (rewritten every step) next to
    <dir>/model-frozen.safetensors (the base's vision + MTP tensors, written once), in checkpoint naming."""

    def __init__(self, base: str, sync_dir: str):
        self.base, self.dir = base, sync_dir
        self.wmap = _base_map(base)

    def write(self, state: dict) -> dict:
        from safetensors import safe_open
        from safetensors.torch import save_file
        t0 = time.time()
        os.makedirs(self.dir, exist_ok=True)
        keep = {k: v for k, v in state.items() if k in self.wmap}
        extra = sorted(set(state) - set(self.wmap) - {"lm_head.weight"})
        if extra:
            raise RuntimeError(f"trained keys absent from the base checkpoint: {extra[:5]}")
        frozen = os.path.join(self.dir, "model-frozen.safetensors")
        if not os.path.exists(frozen):
            need = sorted(k for k in self.wmap if k not in keep)
            tens = {}
            by_file = {}
            for k in need:
                by_file.setdefault(self.wmap[k], []).append(k)
            for fn, keys in by_file.items():
                with safe_open(os.path.join(self.base, fn), "pt") as f:
                    for k in keys:
                        tens[k] = f.get_tensor(k)
            if tens:
                save_file(tens, frozen + ".part", metadata={"format": "pt"})
                os.replace(frozen + ".part", frozen)
        path = os.path.join(self.dir, "model-trained.safetensors")
        save_file(keep, path + ".part", metadata={"format": "pt"})
        os.replace(path + ".part", path)
        return {"write_s": time.time() - t0, "bytes": sum(v.numel() * v.element_size() for v in keep.values())}


class SyncWriterFP8:
    """Weights for the FP8 sampler, in exactly the format of its FP8 checkpoint (`base`, the bf16 model quantized with
    fp8_quant): every trained parameter is gathered (bf16) one at a time,
    quantized on rank 0's GPU with fp8_quant (128x128 blocks, Qwen's bf16 exceptions) and written to
    <dir>/model-trained-NNNNN.safetensors; <dir>/model-frozen.safetensors holds the base's tensors the trainer does
    not have (vision tower, MTP head), written once. The engines' reload_weights then loads them with the FP8
    model's own loader, so what the sampler runs, and what we would serve, is the trainer's quantization."""

    def __init__(self, base: str, sync_dir: str, shard_bytes: int = 4 << 30):
        from arcsolver.rl import fp8_quant as fq
        self.fq, self.base, self.dir, self.shard_bytes = fq, base, sync_dir, shard_bytes
        self.wmap = _base_map(base)

    def _entries(self, key: str, w):
        q = self.fq.quantize_tensor_entries(key, w) if hasattr(self.fq, "quantize_tensor_entries") else None
        if q is not None:
            return q
        if self.fq.should_quantize(key, tuple(w.shape)):
            wq, scale = self.fq.quantize_weight(w)
            return {key: wq, self.fq.scale_name(key) if hasattr(self.fq, "scale_name") else key + "_scale_inv": scale}
        return {key: w}

    def write_stream(self, model, rank: int, writer: bool | None = None, dest: str | None = None) -> dict | None:
        """Collective (every rank calls). Returns write stats on the writing ranks.
        `writer` = this rank writes a copy (default: rank 0 only); with node-local samplers (multi-node)
        every node's local rank 0 writes its own, from the same gathered tensors, and the copies' manifests are compared
        before any engine reloads. `dest` = a fresh version directory (default the sync dir itself); the base's frozen
        tensors are then written once to <sync dir>/frozen/ and hard-linked into each version."""
        from safetensors import safe_open
        from safetensors.torch import save_file
        t0 = time.time()
        writer = (rank == 0) if writer is None else bool(writer)
        out = dest or self.dir
        if writer:
            os.makedirs(out, exist_ok=True)
        buf, nbytes, idx, produced, total = {}, 0, 0, set(), 0

        def flush():
            nonlocal buf, nbytes, idx
            if not buf:
                return
            path = os.path.join(out, f"model-trained-{idx:05d}.safetensors")
            save_file(buf, path + ".part", metadata={"format": "pt"})
            os.replace(path + ".part", path)
            idx += 1
            buf, nbytes = {}, 0

        for name, p in model.named_parameters():
            t = p.detach().to(torch.bfloat16)
            full = t.full_tensor() if hasattr(t, "full_tensor") else t
            if writer:
                key = name[len("hf."):] if name.startswith("hf.") else name
                if key not in self.wmap:
                    raise RuntimeError(f"trained tensor {key} is not in the sampler checkpoint {self.base}")
                for k, v in self._entries(key, full).items():
                    v = v.contiguous().cpu()
                    buf[k] = v
                    produced.add(k)
                    nbytes += v.numel() * v.element_size()
                    total += v.numel() * v.element_size()
                if nbytes >= self.shard_bytes:
                    flush()
            del full, t
        if not writer:
            return None
        flush()
        for old in os.listdir(out):                       # shards left from a run with more of them
            if old.startswith("model-trained-") and int(old.split("-")[2].split(".")[0]) >= idx:
                os.remove(os.path.join(out, old))
        frozen = os.path.join(self.dir if dest is None else os.path.join(self.dir, "frozen"), "model-frozen.safetensors")
        os.makedirs(os.path.dirname(frozen), exist_ok=True)
        if not os.path.exists(frozen):
            need = sorted(k for k in self.wmap if k not in produced)
            by_file, tens = {}, {}
            for k in need:
                by_file.setdefault(self.wmap[k], []).append(k)
            for fn, keys in by_file.items():
                with safe_open(os.path.join(self.base, fn), "pt") as f:
                    for k in keys:
                        tens[k] = f.get_tensor(k)
            if tens:
                save_file(tens, frozen + ".part", metadata={"format": "pt"})
                os.replace(frozen + ".part", frozen)
        if dest is not None and os.path.exists(frozen):
            link = os.path.join(out, "model-frozen.safetensors")
            if not os.path.exists(link):
                try:
                    os.link(frozen, link)
                except OSError:
                    shutil.copy2(frozen, link)
        return {"write_s": time.time() - t0, "bytes": total, "shards": idx}


class SyncWriterBF16Stream:
    """The bf16 SyncWriter behind the same collective write_stream interface."""

    def __init__(self, base: str, sync_dir: str):
        self.w = SyncWriter(base, sync_dir)

    def write_stream(self, model, rank: int, writer: bool | None = None, dest: str | None = None):
        if dest is not None or (writer is not None and bool(writer) != (rank == 0)):
            raise ValueError("the bf16 sync writer writes rank 0's single sync dir only (versioned / per-node sync "
                             "needs --sampler-fp8)")
        full = full_bf16_state(model, rank)
        return self.w.write(full) if rank == 0 else None


def export_fp8_dir(sync_dir: str, fp8_base: str, dest: str, info: dict):
    """Rank 0: a vLLM-loadable FP8 model dir from the last sync (weights) and the FP8 base's config/tokenizer."""
    import json as _json
    os.makedirs(dest, exist_ok=True)
    for fn in os.listdir(fp8_base):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or os.path.isdir(os.path.join(fp8_base, fn)):
            continue
        shutil.copy2(os.path.join(fp8_base, fn), os.path.join(dest, fn))
    weight_map = {}
    from safetensors import safe_open
    for fn in sorted(os.listdir(sync_dir)):
        if fn.endswith(".safetensors"):
            shutil.copy2(os.path.join(sync_dir, fn), os.path.join(dest, fn))
            with safe_open(os.path.join(dest, fn), "pt") as f:
                for k in f.keys():
                    weight_map[k] = fn
    with open(os.path.join(dest, "model.safetensors.index.json"), "w") as f:
        _json.dump({"metadata": {}, "weight_map": weight_map}, f)
    with open(os.path.join(dest, "export_info.json"), "w") as f:
        _json.dump(info, f)


def export_model_dir(model, base: str, dest: str, rank: int, info: dict):
    """A complete vLLM/HF-loadable model dir (bf16; base's side files, vision and MTP tensors). Collective."""
    tf.export_hf(model, base, dest, rank, info)


# --- resumable checkpoints (DCP) ---------------------------------------------------------------------------

def save_dcp(model, opt, path: str):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
    dcp.save({"model": get_model_state_dict(model), "optim": get_optimizer_state_dict(model, opt)}, checkpoint_id=path)


def load_dcp(model, opt, path: str, progress=None):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (get_model_state_dict, get_optimizer_state_dict,
                                                         set_model_state_dict, set_optimizer_state_dict)
    progress = progress or (lambda message: None)
    progress("DCP: building model state mapping")
    state = {"model": get_model_state_dict(model)}
    progress("DCP: building optimizer state mapping")
    state["optim"] = get_optimizer_state_dict(model, opt)
    progress(f"DCP: reading checkpoint {path}")
    dcp.load(state, checkpoint_id=path)
    progress("DCP: applying model state")
    set_model_state_dict(model, state["model"])
    progress("DCP: applying optimizer state")
    set_optimizer_state_dict(model, opt, state["optim"])
    progress("DCP: restoration complete")


def save_dcp_split(model, opt, ckpt_dir: str):
    """DCP in two checkpoints: <ckpt>/dcp_model (fp32 masters, 4 bytes per parameter) and <ckpt>/dcp_optim (AdamW,
    8 bytes per parameter). Same bytes as save_dcp, but the weights alone can be kept as a fallback."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
    dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=os.path.join(ckpt_dir, "dcp_model"))
    dcp.save({"optim": get_optimizer_state_dict(model, opt)}, checkpoint_id=os.path.join(ckpt_dir, "dcp_optim"))


from arcsolver.rl.fsdp_layout import dcp_layout  # noqa: F401(torch-free, shared with the resume scan)


def load_checkpoint(model, opt, ckpt_dir: str, weights_only: bool = False, progress=None):
    """Restore a checkpoint in either layout. weights_only (or a weights-only checkpoint) restores the fp32 masters
    and leaves AdamW fresh. Returns 'full' or 'weights'."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (get_model_state_dict, get_optimizer_state_dict,
                                                         set_model_state_dict, set_optimizer_state_dict)
    progress = progress or (lambda message: None)
    layout = dcp_layout(ckpt_dir)
    if layout is None:
        raise FileNotFoundError(f"no DCP state in {ckpt_dir}")
    if layout == "joint" and not weights_only:
        load_dcp(model, opt, os.path.join(ckpt_dir, "dcp"), progress)
        return "full"
    progress(f"DCP: reading model state ({layout} layout)")
    state = {"model": get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=os.path.join(ckpt_dir, "dcp" if layout == "joint" else "dcp_model"))
    set_model_state_dict(model, state["model"])
    if weights_only or layout == "weights":
        progress("DCP: weights restored; optimizer state starts fresh")
        return "weights"
    progress("DCP: reading optimizer state")
    ostate = {"optim": get_optimizer_state_dict(model, opt)}
    dcp.load(ostate, checkpoint_id=os.path.join(ckpt_dir, "dcp_optim"))
    set_optimizer_state_dict(model, opt, ostate["optim"])
    progress("DCP: restoration complete")
    return "full"


def assign_equal(lengths: list, world: int) -> list:
    """Rollout indices per rank: equal counts (the per-rollout gradient reduction is a collective), longest first
    dealt in a snake so each rank's total is close. Returns [[idx...] per rank]."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out = [[] for _ in range(world)]
    for k, i in enumerate(order):
        rnd, pos = divmod(k, world)
        out[pos if rnd % 2 == 0 else world - 1 - pos].append(i)
    return out
