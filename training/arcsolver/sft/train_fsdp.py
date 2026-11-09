#!/usr/bin/env python
"""Multi-GPU full fine-tuning (FSDP2, fp32 master weights) on a pre-tokenized SFT dataset.

- data: a dataset written by tokenize_cache.py (chat template applied, loss on the assistant turn), trained in the
  dataset's per-epoch order; a step takes the next --global-rows rows of the epoch (the last step of an epoch takes
  the remainder, like a partial gradient-accumulation window);
- loss: token mean over every label token of the step, chunked cross-entropy so 100k+-token rows never materialise
  full logits;
- optimizer: AdamW (betas 0.9/0.999, weight decay on matrices only) on fp32 master weights; compute in bf16 (FSDP2
  mixed precision: bf16 params for compute, fp32 gradient reduction); grad-norm clip; linear warmup then cosine to
  --min-lr-ratio x peak (HF cosine_with_min_lr);
- layout: every decoder layer and the root are sharded; the vision tower is dropped for training (it is frozen and
  unused for text) and copied back from the base checkpoint at export.

Load balance: rows of one step go to ranks by longest-processing-time on an attention-aware cost, and a rank runs
its rows back to back with no collectives in between (parameters stay gathered and gradient reduction waits for its
last row), so a long CoT row costs one rank time instead of stalling every micro-step.

Memory (opt-in; defaults unchanged): the gathered bf16 params and the fp32 unsharded gradient accumulation (FSDP2
upcasts to the reduce dtype between rows) are static per rank; --cpu-offload-optim moves the sharded fp32
params/grads/AdamW states to the CPU (FSDP2 CPUOffloadPolicy; the step runs there), --offload-activations-min-len
keeps long rows' checkpointed layer inputs in a pinned CPU arena until backward, --act-gpu-budget-gb keeps a row's
saved activations on the GPU up to a budget and offloads only the excess. None of them changes the math.

Speed (opt-in): --attn fa3 (Hopper) / fa4 (Blackwell) runs the softmax-attention layers with FlashAttention-3/4
(attn_backends.py); --recompute selective keeps the attention kernels' forward outputs through the checkpoint
recompute (exact); --fused-kernels compile_dynamic|liger fuses RMSNorm/SwiGLU; --fp8 trains the big projections in
torchao float8 (experimental). Kernel choices change bf16 rounding order only, so a run can switch at a checkpoint.

Stopping: --stop-file PATH saves a resumable checkpoint and exits after the step in which PATH appeared (on any
rank); --stop-at-step N saves and exits at step N. A restart with the same --out resumes from the newest checkpoint.

    torchrun --nproc_per_node 8 -m arcsolver.sft.train_fsdp --data DIR --base-model DIR --out RUN_DIR [...]
Outputs in RUN_DIR: ckpt/step_N (resumable, torch.distributed.checkpoint), milestone-N (HF bf16 export with
milestone.json), train_log.jsonl, trainer_state.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SRC_NAMES = ("trace", "aux", "cot")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="tokenize_cache.py output directory")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--global-rows", type=int, default=64, help="rows per optimizer step (all ranks together)")
    ap.add_argument("--epochs", type=int, default=None, help="default: every epoch the data holds")
    ap.add_argument("--peak-lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--warmup-steps", type=int, default=None, help="absolute warmup length (overrides --warmup-frac)")
    ap.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="cosine",
                    help="cosine (default: warmup, cosine to --min-lr-ratio x peak over the run) or wsd (opt-in, "
                         "experimental: warmup (step+1)/warmup x peak, then constant; with --decay-start S "
                         "linear from peak at update S to 0 at S + --decay-steps). A function of the global step only, "
                         "so a resume reproduces it")
    ap.add_argument("--decay-start", type=int, default=None, help="with --lr-schedule wsd: update index where decay starts")
    ap.add_argument("--decay-steps", type=int, default=None, help="with --lr-schedule wsd: decay length in updates")
    ap.add_argument("--lr-stretch-from", type=int, default=None,
                    help="opt-in (data added mid-run, see union_data.py): from this update index S on, the rest of the original "
                         "cosine (length --lr-stretch-total T0) is stretched over the new run length T: update s uses the "
                         "original schedule's LR at S + (s - S) * (T0 - S) / (T - S). Continuous at S, floor at T; updates "
                         "before S keep the original schedule exactly")
    ap.add_argument("--lr-stretch-total", type=int, default=None, help="with --lr-stretch-from: the original schedule length T0")
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--milestone-steps", type=int, nargs="*", default=None,
                    help="HF exports after these step counts (overrides --milestone-fracs)")
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--milestone-fracs", type=float, nargs="*", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--save-every", type=int, default=100, help="resumable checkpoint every N steps (latest kept)")
    ap.add_argument("--stop-at-step", type=int, default=None, help="save a resumable checkpoint there and exit")
    ap.add_argument("--save-at-milestones", action="store_true",
                    help="opt-in (one process through several milestones): also save a resumable checkpoint "
                         "at every milestone step, kept out of the rolling rotation")
    ap.add_argument("--stop-at-milestone-file", default=None, metavar="PATH",
                    help="opt-in: when PATH exists (on any rank) at a milestone step, export, save and exit")
    ap.add_argument("--export-steps", type=int, nargs="*", default=None,
                    help="opt-in: weights-only HF exports after these step counts -> <out>/snapshot-<step> "
                         "(snapshot.json written last); no resumable checkpoint, no stop check, not milestones (the "
                         "milestones and the schedule are unchanged)")
    ap.add_argument("--export-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="with --export-steps: fp32 = the master weights as trained (lossless, for averaging / merging; "
                         "parts copied from the base are upcast exactly and their dtypes recorded in snapshot.json); "
                         "bf16 = the milestone format")
    ap.add_argument("--export-min-free-gb", type=float, default=150.0,
                    help="with --export-steps: a snapshot is skipped (logged) when its run dir's disk has less free space "
                         "(the resumable checkpoints need the room)")
    ap.add_argument("--max-steps", type=int, default=None, help="testing: cap the schedule (the LR curve still uses "
                                                                   "the full length unless --schedule-steps is set)")
    ap.add_argument("--schedule-steps", type=int, default=None)
    ap.add_argument("--ce-chunk", type=int, default=8192)
    ap.add_argument("--no-sdpa-causal", action="store_true")
    ap.add_argument("--wandb-project", default=None,
                    help="log to Weights & Biases (rank 0) when set and WANDB_API_KEY is in the environment")
    ap.add_argument("--wandb-run-name", default=None, help="W&B run name (also the run id, so a restart resumes it)")
    ap.add_argument("--no-export", action="store_true", help="testing: skip milestone exports")
    ap.add_argument("--plan-only", type=int, default=None, metavar="WORLD",
                    help="write <out>/run_plan.json (steps, warmup, milestones) for WORLD ranks and exit")
    ap.add_argument("--offload-activations-min-len", type=int, default=0, metavar="TOKENS",
                    help="rows of at least TOKENS tokens keep their large saved tensors (the checkpointed layers' "
                         "inputs) in pinned CPU memory until backward (0 = off; 9B rows up to 131k on 80 GB GPUs)")
    ap.add_argument("--cpu-offload-optim", action="store_true",
                    help="FSDP2 CPUOffloadPolicy: the sharded fp32 params, grads and AdamW states live in CPU memory "
                         "and the optimizer step runs there (same math); unsharded bf16 params and the fp32 gradient "
                         "accumulation stay on the GPU")
    ap.add_argument("--attn", choices=["sdpa", "fa3", "fa4"], default="sdpa",
                    help="softmax-attention kernel: sdpa (default, + the causal hook), fa3 (FlashAttention-3, Hopper: "
                         "from --fa3-dir / $ARC_FA3_DIR, else the `kernels` package) or fa4 (FlashAttention-4, "
                         "Blackwell: flash_attn.cute, importable or from --fa4-dir / $ARC_FA4_DIR); unpadded batch-1 rows")
    ap.add_argument("--fa3-dir", default=None,
                    help="local FA3 kernel copy (python -m arcsolver.sft.attn_backends fetch-fa3 --dest DIR)")
    ap.add_argument("--fa4-dir", default=None, help="FA4 side directory (pip install --target DIR ...; see "
                         "attn_backends.load_fa4); default $ARC_FA4_DIR, else FA4 must be importable")
    ap.add_argument("--sdpa-backend", choices=["auto", "flash", "cudnn", "efficient"], default="auto",
                    help="with --attn sdpa (opt-in): allow only this PyTorch SDPA backend; auto = PyTorch's own choice")
    ap.add_argument("--select-top-frac", type=float, default=None, metavar="FRAC",
                    help="opt-in, experimental: backprop only each row's FRAC highest-loss label tokens under the current "
                         "model (context unchanged); the step loss is the mean over the selected tokens")
    ap.add_argument("--select-weight", type=float, default=None, metavar="W",
                    help="with --select-top-frac (opt-in, experimental hybrid): every label token keeps weight 1 and the "
                         "selected ones get weight W; the step loss is the weighted token mean")
    ap.add_argument("--max-row-tokens", type=int, default=None,
                    help="opt-in: rows longer than this many tokens are left out of their step; the plan keeps "
                         "its steps, epochs and milestones (a resumed run keeps its schedule and data position)")
    ap.add_argument("--merge-tail", action="store_true",
                    help="an epoch's last step with fewer rows than ranks is merged into the previous step (opt-in)")
    ap.add_argument("--merge-tail-min", type=int, default=0, metavar="N",
                    help="with --merge-tail: merge an epoch's last step when it has fewer than N rows (default: the number "
                         "of ranks). A run moved from 16 to 8 ranks keeps its 16-rank plan with 16")
    ap.add_argument("--stop-file", default=None, metavar="PATH",
                    help="when PATH exists (on any rank) after a step: save a resumable checkpoint there and exit")
    ap.add_argument("--compile-norm-mlp", action="store_true",
                    help="torch.compile the RMSNorm and SwiGLU elementwise chains (trainer_speed.py; static shapes: "
                         "recompiles per row length and was ~3x slower in SFT; prefer --fused-kernels compile_dynamic)")
    ap.add_argument("--zero3", action="store_true",
                    help="opt-in, experimental (large models): every layer reshards after its forward and backward, every row's "
                         "gradient is reduce-scattered in fp32 as it is produced (no unsharded accumulation), so the "
                         "ranks run their rows in lockstep micro-steps (a step's rows sorted by length, one per rank "
                         "per micro-step, zero-loss dummies pad); the model loads in bf16 on the CPU and moves to the "
                         "GPU layer by layer")
    ap.add_argument("--act-gpu-budget-gb", type=float, default=0.0, metavar="GIB",
                    help="opt-in: a row's large saved activations stay on the GPU up to GIB, the "
                         "rest go to a pinned CPU arena sized for the longest row's excess (0 = off)")
    ap.add_argument("--act-arena-gb", type=float, default=0.0, metavar="GIB",
                    help="with --act-gpu-budget-gb: cap the pinned arena per rank at GIB (host RAM); a row's excess "
                         "beyond it goes to pageable memory (slower copies; 0 = no cap)")
    ap.add_argument("--layerwise-load", action="store_true",
                    help="opt-in (large models): load the checkpoint's bf16 on the CPU and move each layer to the GPU "
                         "in fp32 as it is sharded (implied by --zero3)")
    ap.add_argument("--recompute", choices=["full", "selective"], default="full",
                    help="opt-in (opt_levers.py): selective = the full-attention layers keep their attention "
                         "kernel's forward outputs, so the recompute skips the quadratic attention forward (same values); "
                         "with --act-gpu-budget-gb each row's offload budget is lowered by the kept bytes")
    ap.add_argument("--fp8", choices=["none", "rowwise", "tensorwise"], default="none",
                    help="opt-in, experimental (opt_levers.py): torchao float8 training for the decoder layers' large "
                         "projections (not lm_head / embeddings / MTP); --cpu-test uses torchao's emulation")
    ap.add_argument("--fp8-skip-edge-layers", action=argparse.BooleanOptionalAction, default=True,
                    help="with --fp8: keep the first and last decoder layers in bf16 (default on)")
    ap.add_argument("--fp8-min-dim", type=int, default=1024, help="with --fp8: smallest in/out size converted")
    ap.add_argument("--fused-kernels", choices=["none", "liger", "compile_dynamic"], default="none",
                    help="opt-in (opt_levers.py): compile_dynamic = trainer_speed's exact compiled RMSNorm + "
                         "SwiGLU with dynamic shapes (bit-equal forward); liger = Liger-kernel RMSNorm + SwiGLU (needs "
                         "pip install liger-kernel; norm forward not bit-equal)")
    ap.add_argument("--cpu-test", action="store_true", help="testing: run on the CPU with gloo (no CUDA)")
    ap.add_argument("--offload-min-bytes", type=int, default=32 << 20, help="testing: smallest saved tensor offloaded")
    a = ap.parse_args(argv)
    if a.select_weight is not None and (not a.select_top_frac or a.select_weight < 1):
        ap.error("--select-weight W needs --select-top-frac and W >= 1")
    if a.lr_schedule == "wsd" and (a.decay_start is None) != (a.decay_steps is None):
        ap.error("--decay-start and --decay-steps go together")
    if a.lr_schedule != "wsd" and a.decay_start is not None:
        ap.error("--decay-start needs --lr-schedule wsd")
    if (a.lr_stretch_from is None) != (a.lr_stretch_total is None):
        ap.error("--lr-stretch-from and --lr-stretch-total go together")
    if a.lr_stretch_from is not None and (a.lr_schedule != "cosine" or not 0 < a.lr_stretch_from < a.lr_stretch_total):
        ap.error("--lr-stretch-from S needs the cosine schedule and 0 < S < --lr-stretch-total")
    if a.act_gpu_budget_gb and a.offload_activations_min_len:
        ap.error("--act-gpu-budget-gb replaces --offload-activations-min-len (use one)")
    if a.zero3 and (a.cpu_offload_optim or a.select_top_frac):
        ap.error("--zero3 is not combined with --cpu-offload-optim / --select-top-frac")
    if a.zero3 and a.offload_activations_min_len:
        ap.error("--zero3 offloads with --act-gpu-budget-gb")
    if a.fused_kernels != "none" and a.compile_norm_mlp:
        ap.error("--fused-kernels replaces --compile-norm-mlp (use one)")
    if a.fused_kernels == "compile_dynamic" and a.zero3:
        ap.error("--fused-kernels compile_dynamic is not supported with --zero3")
    if a.fused_kernels == "liger" and a.cpu_test:
        ap.error("--fused-kernels liger needs CUDA (Triton)")
    a.layerwise_load = a.layerwise_load or a.zero3
    return a


# --- schedule and plan -------------------------------------------------------------------------------

def lr_at(step: int, peak: float, floor: float, warmup: int, total: int) -> float:
    """HF cosine_with_min_lr: update `step` (0-based) uses this LR."""
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def stretch_step(step: int, stretch_from, stretch_total, total: int) -> float:
    """--lr-stretch-from S --lr-stretch-total T0: the update index of the original T0-update schedule whose LR update `step`
    of the T-update run uses (identity before S; S..T maps linearly onto S..T0)."""
    if stretch_from is None or step < stretch_from:
        return step
    return stretch_from + (step - stretch_from) * (stretch_total - stretch_from) / max(1, total - stretch_from)


def lr_wsd(step: int, peak: float, warmup: int, decay_start=None, decay_steps=None) -> float:
    """Warmup-stable-decay: update `step` (0-based) uses this LR."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if decay_start is None or step < decay_start:
        return peak
    return peak * (1.0 - min(1.0, (step - decay_start) / decay_steps))


def row_cost(n_tokens: int) -> float:
    """Relative compute of one row: linear layers ~ tokens, the full-attention layers add ~ tokens^2."""
    return n_tokens * (1.0 + n_tokens / 65536.0)


def assign(rows, lengths, world):
    """Longest-processing-time assignment of one step's rows to ranks (deterministic on every rank)."""
    loads = [0.0] * world
    out = [[] for _ in range(world)]
    for i in sorted(rows, key=lambda r: (-lengths[r], r)):
        k = min(range(world), key=lambda j: (loads[j], j))
        out[k].append(int(i))
        loads[k] += row_cost(int(lengths[i]))
    return out, loads


def assign_lockstep(rows, lengths, world):
    """--zero3: a step's rows sorted longest first; micro-step j runs rows j*world .. (j+1)*world-1, one per rank, so
    the rows that run together have similar lengths (sorting minimises the summed per-micro-step maxima). None = a
    zero-loss dummy row (a partial last micro-step). Every rank gets the same number of entries. Returns the rank lists
    and the load ratio (summed per-micro-step maximum) / (mean rank load)."""
    s = sorted(rows, key=lambda r: (-lengths[r], r))
    out = [[] for _ in range(world)]
    span = 0.0
    for j in range(0, len(s), world):
        slot = s[j:j + world]
        span += max(row_cost(int(lengths[r])) for r in slot)
        for k in range(world):
            out[k].append(int(slot[k]) if k < len(slot) else None)
    mean = sum(row_cost(int(lengths[r])) for r in rows) / world
    return out, span / max(1e-9, mean)


def build_plan(order, lengths, labels, G, world, epochs, merge_tail=False, lockstep=False, max_row_tokens=None,
               merge_tail_min=0):
    """[(epoch, rank lists, label tokens of the step, max/mean load)] for every step of the run.
    merge_tail (opt-in): an epoch's last chunk with fewer rows than ranks joins the previous step (no update from a
    handful of rows, and no rank runs the zero-row dummy pass); merge_tail_min replaces the rank count in that test, so
    a run moved to fewer ranks keeps its steps. lockstep (--zero3): assign_lockstep."""
    plan = []
    for e in range(epochs):
        o = order[e]
        o = o[o >= 0]           # -1 pads an epoch shorter than the order's width (rows added mid-run, union_data.py)
        chunks = [o[k * G:(k + 1) * G] for k in range(math.ceil(len(o) / G))]
        if merge_tail and len(chunks) > 1 and len(chunks[-1]) < (merge_tail_min or world):
            chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
            chunks.pop()
        for chunk in chunks:
            rows = [int(x) for x in chunk if max_row_tokens is None or lengths[int(x)] <= max_row_tokens]
            if lockstep:
                ranks, ratio = assign_lockstep(rows, lengths, world)
                plan.append((e, ranks, int(sum(labels[r] for r in rows)), ratio))
                continue
            ranks, loads = assign(rows, lengths, world)
            plan.append((e, ranks, int(sum(labels[r] for r in rows)), max(loads) / max(1e-9, sum(loads) / world)))
    return plan


def epoch_position(done, steps_per_epoch):
    """Finished epochs as a float after `done` steps (2.5 = halfway through the third epoch of the plan)."""
    c = 0
    for e, n in enumerate(steps_per_epoch):
        if done <= c + n:
            return e + (done - c) / max(1, n)
        c += n
    return float(len(steps_per_epoch))


def milestone_steps(fracs, total):
    """{step (1-based count of finished steps): frac} for milestone fractions of the run."""
    out = {}
    for f in sorted(fracs):
        s = max(1, min(total, round(f * total)))
        while s in out and s < total:
            s += 1
        out[s] = f
    return out


# --- model ---------------------------------------------------------------------------------------------

def install_sdpa_causal():
    """At micro-batch 1 a verified purely causal mask becomes is_causal=True, so PyTorch takes the flash SDPA kernel
    instead of the masked (memory-efficient) one (transformers passes an explicit causal mask otherwise)."""
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


class CausalLMLoss(nn.Module):
    """Summed next-token cross-entropy over a row's label span, lm_head applied in checkpointed chunks.
    select_frac (opt-in, experimental: train on only the most surprising tokens within a trace): only the
    ceil(select_frac * n) label tokens with the highest cross-entropy under the current model (picked per row, on
    detached losses) contribute to the returned sum; every token stays in the context. select_weight (opt-in hybrid):
    all label tokens count with weight 1 and the selected ones with weight W. last_stats = (summed loss over ALL label
    tokens, the row's share of the step normaliser = its summed token weights, summed loss over the selected tokens,
    number selected) of the latest forward."""

    def __init__(self, hf, chunk, select_frac=None, select_weight=None):
        super().__init__()
        self.hf = hf
        self.chunk = chunk
        self.select_frac = select_frac
        self.select_weight = select_weight
        self.last_stats = None

    def _ce(self, h, t):
        return F.cross_entropy(self.hf.lm_head(h).float(), t, reduction="sum")

    def _ce_tok(self, h, t):
        return F.cross_entropy(self.hf.lm_head(h).float(), t, reduction="none")

    def forward(self, input_ids, label_start: int):
        h = self.hf.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        S = input_ids.shape[1]
        hs, tgt = h[label_start - 1:S - 1], input_ids[0, label_start:S]
        if not self.select_frac:
            total = h.new_zeros((), dtype=torch.float32)
            for i in range(0, tgt.shape[0], self.chunk):
                total = total + checkpoint(self._ce, hs[i:i + self.chunk], tgt[i:i + self.chunk], use_reentrant=False)
            self.last_stats = (total.detach(), tgt.shape[0], total.detach(), tgt.shape[0])
            return total
        tok = torch.cat([checkpoint(self._ce_tok, hs[i:i + self.chunk], tgt[i:i + self.chunk], use_reentrant=False)
                         for i in range(0, tgt.shape[0], self.chunk)])
        n = tgt.shape[0]
        k = n_selected(n, self.select_frac)
        idx = torch.topk(tok.detach(), k, sorted=False).indices
        sel = tok[idx].sum()
        w0, w1 = select_weights(self.select_weight)
        self.last_stats = (tok.detach().sum(), row_norm(n, self.select_frac, self.select_weight), sel.detach(), k)
        return sel if not w0 else tok.sum() + (w1 - w0) * sel


def n_selected(n_label: int, frac) -> int:
    """Label tokens kept per row by --select-top-frac (same rule in the loss and in the step normaliser)."""
    return n_label if not frac else max(1, min(n_label, math.ceil(frac * n_label)))


def select_weights(select_weight):
    """(weight of the unselected label tokens, weight of the selected ones): --select-top-frac alone = (0, 1),
    with --select-weight W = (1, W)."""
    return (0.0, 1.0) if select_weight is None else (1.0, float(select_weight))


def row_norm(n_label: int, frac, select_weight=None):
    """A row's share of the step normaliser = its summed token weights, so the step loss is a weighted token mean."""
    if not frac:
        return n_label
    k = n_selected(n_label, frac)
    w0, w1 = select_weights(select_weight)
    return w0 * (n_label - k) + w1 * k


def text_layers(hf):
    return hf.model.language_model.layers


def load_model(base, device, chunk, attn="sdpa", fa3_dir=None, compile_norm_mlp=False, select_frac=None,
               select_weight=None, layerwise=False, levers=None):
    """layerwise (--layerwise-load, implied by --zero3): the checkpoint's bf16 stays on the CPU (shard(layerwise=True)
    moves each layer to the GPU in fp32 and shards it at once, so no GPU holds the whole fp32 model, and no rank holds
    an fp32 CPU copy); exact when the checkpoint is bf16.
    levers (opt-in): opt_levers.apply() keyword arguments (--recompute / --fused-kernels / --fp8), applied after
    checkpointing is enabled and before sharding; what was done is in the returned loss module's .levers."""
    from transformers import AutoConfig, AutoModelForImageTextToText
    if layerwise:
        impl = "sdpa"
        if attn in ("fa3", "fa4"):
            from arcsolver.sft import attn_backends
            impl = attn_backends.register_fa3(fa3_dir) if attn == "fa3" else attn_backends.register_fa4()
        elif attn != "sdpa":
            raise ValueError(f"attn must be 'sdpa', 'fa3' or 'fa4', got {attn!r}")
        if compile_norm_mlp:
            raise ValueError("--compile-norm-mlp is not supported with --zero3")
        cfg = AutoConfig.from_pretrained(base)
        hf = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=impl)
        if getattr(hf.model, "visual", None) is not None:
            hf.model.visual = None
        hf.config.use_cache = False
        hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hf.train()
        info = _apply_levers(hf, cfg, device, levers)
        m = CausalLMLoss(hf, chunk, select_frac, select_weight)
        m.levers = info
        return m, cfg
    impl = "sdpa"
    if attn in ("fa3", "fa4"):
        from arcsolver.sft import attn_backends
        impl = attn_backends.register_fa3(fa3_dir) if attn == "fa3" else attn_backends.register_fa4()
    elif attn != "sdpa":
        raise ValueError(f"attn must be 'sdpa', 'fa3' or 'fa4', got {attn!r}")
    if compile_norm_mlp:
        from arcsolver.sft import trainer_speed
        trainer_speed.enable_compiled_norm_mlp()
    cfg = AutoConfig.from_pretrained(base)
    hf = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32, attn_implementation=impl)
    if getattr(hf.model, "visual", None) is not None:
        hf.model.visual = None       # frozen and unused for text rows; restored from the base at export
    hf.config.use_cache = False
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hf.train()           # from_pretrained returns eval mode, where HF layers skip gradient checkpointing
    info = _apply_levers(hf, cfg, device, levers)
    hf.to(device)
    m = CausalLMLoss(hf, chunk, select_frac, select_weight)
    m.levers = info
    return m, cfg


def _apply_levers(hf, cfg, device, levers):
    if not levers:
        return {}
    from arcsolver.sft import opt_levers
    return opt_levers.apply(hf, cfg, device, **levers)


def shard(model, mesh=None, cpu_offload=False, zero3=False, device=None, layerwise=False, param_dtype=torch.bfloat16):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    off = CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy()
    if layerwise:  # from the CPU bf16 copy (load_model(layerwise=True)): one fp32 layer on the GPU at a time
        for layer in text_layers(model.hf):     # (with CPUOffloadPolicy fully_shard moves the shards back to the CPU)
            layer.to(device=device, dtype=torch.float32)
            fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=zero3, offload_policy=off)
        model.to(device=device, dtype=torch.float32)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=zero3, offload_policy=off)
        return model
    for layer in text_layers(model.hf):
        fully_shard(layer, mp_policy=mp, reshard_after_forward=False, offload_policy=off)
    fully_shard(model, mp_policy=mp, reshard_after_forward=False, offload_policy=off)
    return model


class PinnedArena:
    """One pinned CPU buffer per rank, handed out front to back for one row's saved tensors and reset before the next
    offloaded row: a row's saved tensors are all consumed by its backward before the next row's forward starts, and
    every copy into or out of the buffer is queued on the same stream. (The caching host allocator would round each
    1 GiB layer input up to 2 GiB and keep every size class it ever saw pinned.)"""

    def __init__(self, nbytes: int, pin: bool = True, overflow_pin: bool = True, exact: bool = False):  # pin=False: CPU tests
        self.pin = pin
        self.registered = False
        if pin and exact:
            # exact size: torch's pinned allocator rounds up to a power of two (a 150 GiB arena takes 256 GiB per
            # rank); plain memory + cudaHostRegister pins exactly nbytes (falls back to the rounding allocator if
            # registration fails)
            self.buf = torch.empty(nbytes, dtype=torch.uint8)
            rc = torch.cuda.cudart().cudaHostRegister(self.buf.data_ptr(), nbytes, 0)
            self.registered = rc == 0 or getattr(rc, "value", None) == 0 or str(rc).lower().endswith("success")
            if not self.registered:
                self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        else:
            self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
        self.overflow_pin = overflow_pin and pin     # False (--act-arena-gb): overflow goes to pageable memory
        self.off = 0
        self.overflow = 0

    def reset(self):
        self.off = 0

    def take(self, shape, dtype):
        n = math.prod(shape) * dtype.itemsize
        start = (self.off + 511) // 512 * 512
        if start + n > self.buf.numel():          # larger than sized for: fall back to the caching host allocator
            self.overflow += 1
            return torch.empty(shape, dtype=dtype, pin_memory=self.overflow_pin)
        self.off = start + n
        return self.buf[start:start + n].view(dtype).view(shape)


class ActivationOffload(torch.autograd.graph.saved_tensors_hooks):
    """Saved-tensor hooks that keep large saved CUDA tensors in pinned CPU memory (an arena) until backward.

    With non-reentrant gradient checkpointing the tensors saved inside a checkpointed layer are placeholders (the
    checkpoint's own hooks are innermost), so what reaches these hooks is each layer's checkpointed input (S x hidden
    in bf16: 1 GiB per layer for a 131k row of a 4096-hidden model), the final norm's and the CE chunks' inputs and a few small
    tensors. Tensors under min_bytes (the rotary tables every layer shares, ids, norm weights) stay where they are.
    The copies are queued on the current stream both ways, so they are ordered with the compute that produces /
    consumes them."""

    def __init__(self, arena: PinnedArena, min_bytes: int = 32 << 20, only_cuda: bool = True):  # False: CPU tests
        def pack(t):
            if (only_cuda and not t.is_cuda) or t.numel() * t.element_size() < min_bytes:
                return t
            cpu = arena.take(t.shape, t.dtype)
            cpu.copy_(t, non_blocking=True)
            return t.device, cpu

        def unpack(p):
            if isinstance(p, torch.Tensor):
                return p
            dev, cpu = p
            return cpu.to(dev, non_blocking=True)

        super().__init__(pack, unpack)


class BudgetOffload(torch.autograd.graph.saved_tensors_hooks):
    """--act-gpu-budget-gb: a row's large saved tensors (the checkpointed layers' inputs, S x hidden bf16 per layer)
    stay on the GPU until they total budget_bytes; later ones go to the pinned arena until backward. Only the excess
    of rows above the budget is pinned, so the arena is the longest row's excess per rank (the plain
    ActivationOffload arena is the whole longest row). Parameters (FSDP's gathered weights: leaf tensors that require
    grad) are never moved. reset() before every row's forward."""

    def __init__(self, arena, budget_bytes: int, min_bytes: int = 32 << 20, only_cuda: bool = True):
        self.kept = 0
        self.moved = 0

        def pack(t):
            n = t.numel() * t.element_size()
            if (only_cuda and not t.is_cuda) or n < min_bytes or (t.requires_grad and t.grad_fn is None):
                return t
            if self.kept + n <= budget_bytes or arena is None:
                self.kept += n
                return t
            cpu = arena.take(t.shape, t.dtype)
            cpu.copy_(t, non_blocking=True)
            self.moved += n
            return t.device, cpu

        def unpack(p):
            if isinstance(p, torch.Tensor):
                return p
            dev, cpu = p
            return cpu.to(dev, non_blocking=True)

        super().__init__(pack, unpack)

    def reset(self, reserve: int = 0):
        """reserve (--recompute selective): bytes the row keeps on the GPU outside these hooks, counted against the
        budget from the start."""
        self.kept = reserve
        self.moved = 0


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "norm" not in n.lower() else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


# --- checkpoints and exports ------------------------------------------------------------------------------

def release_cached_memory():
    """Hand the caching allocator's free blocks back before a save/export. Their collectives (DCP's planning gather,
    the full-state-dict gather) open new NCCL connections whose buffers NCCL allocates outside PyTorch; after a
    long-row step on an 80 GB GPU the cache held nearly all of it and the gather failed with "NCCL Error 1: unhandled
    cuda error". Changes no numbers."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def node_info():
    """(node index, local rank, number of nodes) under torchrun (LOCAL_WORLD_SIZE ranks per node)."""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    lws = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    return rank // lws, int(os.environ.get("LOCAL_RANK", 0)), max(1, world // lws)


def save_resumable(model, opt, out, step, rank, keep=1, protect=()):
    """DCP checkpoint. Multi-node (works without a shared filesystem): every rank writes its own shard files on its
    node's disk; rank 0's .metadata is broadcast and written on every node, and each node's local rank 0
    writes COMPLETE and rotates that node's old checkpoints (single node: exactly as before, local rank 0 = rank 0)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict
    path = os.path.join(out, "ckpt", f"step_{step:06d}")
    release_cached_memory()
    dcp.save({"model": get_model_state_dict(model), "optim": get_optimizer_state_dict(model, opt)}, checkpoint_id=path)
    dist.barrier()
    node, local_rank, n_nodes = node_info()
    if n_nodes > 1:
        meta = [open(os.path.join(path, ".metadata"), "rb").read() if rank == 0 else None]
        dist.broadcast_object_list(meta, src=0)
        mp = os.path.join(path, ".metadata")
        if local_rank == 0 and (not os.path.exists(mp) or open(mp, "rb").read() != meta[0]):
            with open(mp + ".tmp", "wb") as f:
                f.write(meta[0])
            os.replace(mp + ".tmp", mp)
        dist.barrier()
    if local_rank == 0:
        with open(os.path.join(path, "COMPLETE"), "w") as f:
            f.write(str(step))
        done = sorted(glob.glob(os.path.join(out, "ckpt", "step_*")))
        for old in done[:-keep]:
            if old != path and os.path.basename(old) not in protect:     # protect: milestone checkpoints (--save-at-milestones)
                shutil.rmtree(old, ignore_errors=True)
    dist.barrier()


def latest_ckpt(out):
    """The newest COMPLETE checkpoint; multi-node: the newest one COMPLETE on every node (each keeps its own shards)."""
    done = [d for d in sorted(glob.glob(os.path.join(out, "ckpt", "step_*"))) if os.path.exists(os.path.join(d, "COMPLETE"))]
    if dist.is_initialized() and node_info()[2] > 1:
        names = [None] * dist.get_world_size()
        dist.all_gather_object(names, [os.path.basename(d) for d in done])
        common = sorted(set.intersection(*[set(n) for n in names]))
        return os.path.join(out, "ckpt", common[-1]) if common else None
    return done[-1] if done else None


def load_resumable(model, opt, path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (get_model_state_dict, get_optimizer_state_dict,
                                                         set_model_state_dict, set_optimizer_state_dict)
    state = {"model": get_model_state_dict(model), "optim": get_optimizer_state_dict(model, opt)}
    dcp.load(state, checkpoint_id=path)
    set_model_state_dict(model, state["model"])
    set_optimizer_state_dict(model, opt, state["optim"])
    return int(os.path.basename(path).split("_")[1])


def export_hf(model, base, dest, rank, meta, dtype=torch.bfloat16, meta_name="milestone.json"):
    """Full HF checkpoint (the base's key set and shards layout; vision weights from the base): bf16 by default; with
    dtype=torch.float32 the trained tensors are the fp32 master weights as they are and the base-copied ones are upcast
    (exact), their original dtypes recorded in meta["base_copied_dtypes"]."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    release_cached_memory()
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
        upcast = dtype == torch.float32
        copied_dtypes = {}
        for fn, keys in sorted(by_file.items()):
            tensors = {}
            base_keys = [k for k in keys if k not in trained]
            if base_keys:
                with safe_open(os.path.join(base, fn), "pt") as f:
                    for k in base_keys:
                        tensors[k] = f.get_tensor(k)
                        if upcast:
                            copied_dtypes[k] = str(tensors[k].dtype).replace("torch.", "")
                            tensors[k] = tensors[k].to(torch.float32)
            for k in keys:
                if k in trained:
                    tensors[k] = trained[k].to(dtype).contiguous()
            total += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(tensors, os.path.join(tmp, fn), metadata={"format": "pt"})
        if os.path.exists(idx_path):
            idx = json.load(open(idx_path))
            idx["metadata"] = {"total_size": total}
            json.dump(idx, open(os.path.join(tmp, "model.safetensors.index.json"), "w"), indent=2)
        for f in os.listdir(base):
            if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(os.path.join(base, f)):
                shutil.copy2(os.path.join(base, f), os.path.join(tmp, f))
        if upcast:
            # proof that the snapshot holds the fp32 masters, not bf16 values: of up to 64 trained matrices, the share whose first 65,536 values are not all exactly representable in bf16
            names = sorted(k for k in trained if k in wmap and trained[k].dim() == 2)[:64]
            finer = sum(1 for k in names if not torch.equal(trained[k].flatten()[:65536],
                                                            trained[k].flatten()[:65536].to(torch.bfloat16).float()))
            meta = {**meta, "base_copied_dtypes": copied_dtypes,
                    "trained_tensors_finer_than_bf16": f"{finer}/{len(names)}"}
        json.dump(meta, open(os.path.join(tmp, meta_name), "w"), indent=1)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    dist.barrier()


# --- training ----------------------------------------------------------------------------------------------

def plan_only(a):
    offsets = np.load(os.path.join(a.data, "offsets.npy"))
    label_start = np.load(os.path.join(a.data, "label_start.npy"))
    order = np.load(os.path.join(a.data, "order.npy"))
    lengths = np.diff(offsets)
    epochs = a.epochs or order.shape[0]
    plan = build_plan(order, lengths, lengths - label_start, a.global_rows, a.plan_only, epochs, a.merge_tail,
                      max_row_tokens=a.max_row_tokens, merge_tail_min=a.merge_tail_min)
    total = a.schedule_steps or len(plan)
    imb = [p[3] for p in plan]
    ms = {s: s / total for s in a.milestone_steps} if a.milestone_steps else milestone_steps(a.milestone_fracs, total)
    out = {"total_steps": total, "warmup": a.warmup_steps or max(1, round(a.warmup_frac * total)), "global_rows": a.global_rows,
           "world": a.plan_only, "epochs": epochs,
           "steps_per_epoch": [sum(1 for p in plan if p[0] == e) for e in range(epochs)],
           "milestones": {str(k): v for k, v in ms.items()}, "lr_schedule": a.lr_schedule,
           "decay_start": a.decay_start, "decay_steps": a.decay_steps,
           "imbalance_median": float(np.median(imb)), "imbalance_p90": float(np.percentile(imb, 90)),
           "label_tokens": int(sum(p[2] for p in plan))}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "run_plan.json"), "w"), indent=1)
    print(json.dumps(out))


def main(argv=None):
    a = parse_args(argv)
    if a.plan_only:
        return plan_only(a)
    # CPU-offloaded DTensors (grad-norm clip, exports) need a CPU backend next to NCCL
    dist.init_process_group("gloo" if a.cpu_test else "cpu:gloo,cuda:nccl" if a.cpu_offload_optim else "nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if a.cpu_test and os.environ.get("ARC_TEST_NODE_OUT") and node_info()[0] > 0:
        a.out = f"{a.out}.node{node_info()[0]}"      # testing: node-local run dirs (no shared filesystem)
    local = int(os.environ.get("LOCAL_RANK", 0))
    if a.cpu_offload_optim:    # the AdamW step runs on the CPU in every rank: split the cores between them
        torch.set_num_threads(max(4, (os.cpu_count() or 8) // int(os.environ.get("LOCAL_WORLD_SIZE", world))))
    if a.cpu_test:
        dev = torch.device("cpu")
    else:
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    torch.manual_seed(1234)
    if not a.no_sdpa_causal:
        install_sdpa_causal()
    if a.sdpa_backend != "auto" and not a.cpu_test:
        torch.backends.cuda.enable_flash_sdp(a.sdpa_backend == "flash")
        torch.backends.cuda.enable_cudnn_sdp(a.sdpa_backend == "cudnn")
        torch.backends.cuda.enable_mem_efficient_sdp(a.sdpa_backend == "efficient")
        torch.backends.cuda.enable_math_sdp(False)       # fail loudly instead of an O(S^2)-memory fallback

    D = a.data
    offsets = np.load(os.path.join(D, "offsets.npy"))
    label_start = np.load(os.path.join(D, "label_start.npy"))
    source = np.load(os.path.join(D, "source.npy"))
    order = np.load(os.path.join(D, "order.npy"))
    ids_mm = np.memmap(os.path.join(D, "ids.u32"), dtype=np.uint32, mode="r")
    lengths = np.diff(offsets)
    labels = lengths - label_start
    epochs = a.epochs or order.shape[0]
    plan = build_plan(order, lengths, labels, a.global_rows, world, epochs, a.merge_tail, lockstep=a.zero3,
                      max_row_tokens=a.max_row_tokens, merge_tail_min=a.merge_tail_min)
    total = a.schedule_steps or len(plan)
    run_steps = min(len(plan), a.max_steps or len(plan))
    warmup = a.warmup_steps or max(1, round(a.warmup_frac * total))
    floor = a.peak_lr * a.min_lr_ratio
    milestones = {s: s / total for s in a.milestone_steps} if a.milestone_steps else milestone_steps(a.milestone_fracs, total)
    export_steps = set(a.export_steps or ())
    steps_per_epoch = [sum(1 for p in plan if p[0] == e) for e in range(epochs)]
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        if export_steps:
            print(f"[snapshot] weights-only {a.export_dtype} exports after steps {sorted(export_steps)} -> "
                  f"{a.out}/snapshot-<step>", flush=True)
        imb = [p[3] for p in plan]
        print(f"[plan] {len(plan)} steps ({epochs} epochs x {order.shape[1]} rows, {a.global_rows} rows/step, "
              f"{world} ranks{', lockstep micro-steps (--zero3)' if a.zero3 else ''}); warmup {warmup}; milestones "
              f"{milestones}; load imbalance max/mean median "
              f"{np.median(imb):.2f} p90 {np.percentile(imb, 90):.2f}", flush=True)

    t_load = time.time()
    levers = None
    if a.recompute != "full" or a.fp8 != "none" or a.fused_kernels != "none":
        levers = {"recompute": a.recompute, "fused": a.fused_kernels, "fp8": a.fp8, "fp8_skip_edge": a.fp8_skip_edge_layers,
                  "fp8_min_dim": a.fp8_min_dim, "fp8_emulate": a.cpu_test}
    if a.fa4_dir:      # attn_backends.load_fa4 reads it
        os.environ["ARC_FA4_DIR"] = a.fa4_dir
    model, cfg = load_model(a.base_model, dev, a.ce_chunk, a.attn, a.fa3_dir, a.compile_norm_mlp, a.select_top_frac,
                            a.select_weight, layerwise=a.layerwise_load, levers=levers)
    model = shard(model, cpu_offload=a.cpu_offload_optim, zero3=a.zero3, device=dev, layerwise=a.layerwise_load,
                  param_dtype=torch.float32 if a.cpu_test else torch.bfloat16)   # CPU tests: no bf16 backward on oneDNN
    opt = torch.optim.AdamW(param_groups(model, a.weight_decay), lr=a.peak_lr, betas=(a.adam_beta1, a.adam_beta2),
                            eps=a.adam_eps, foreach=True)
    start = 0
    ck = latest_ckpt(a.out)
    if ck:
        start = load_resumable(model, opt, ck)
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"[model] {a.base_model}: {n_params / 1e9:.2f}B trainable params (vision tower dropped), loaded in "
              f"{time.time() - t_load:.0f}s; start step {start}" + (f" (resumed from {ck})" if ck else "")
              + f"; attention {a.attn}" + (" + compiled norm/mlp" if a.compile_norm_mlp else ""), flush=True)
        for k, v in getattr(model, "levers", {}).items():
            print(f"[levers] {k}: {v}", flush=True)
    sac_res = None
    if a.recompute == "selective":    # the kept attention outputs bypass the offload hooks: count them against the budget
        import opt_levers
        sac_res = lambda n: opt_levers.sac_reserve_bytes(cfg, int(n))
    arena = None
    if a.offload_activations_min_len:    # sized for the longest row: every layer input + final norm + CE inputs
        hidden = getattr(cfg, "text_config", cfg).hidden_size
        nbytes = int(lengths.max()) * hidden * 2 * (len(text_layers(model.hf)) + 4)
        arena = PinnedArena(nbytes, pin=not a.cpu_test)
        if rank == 0:
            print(f"[offload] rows >= {a.offload_activations_min_len} tokens keep saved activations in a pinned arena "
                  f"of {nbytes / 2**30:.1f} GiB per rank" + ("; optimizer state on the CPU" if a.cpu_offload_optim
                                                              else ""), flush=True)
    offl = None
    if a.act_gpu_budget_gb:    # only a row's excess over the budget is pinned: arena = the longest row's excess
        hidden = getattr(cfg, "text_config", cfg).hidden_size
        row_bytes = int(lengths.max()) * hidden * 2 * (len(text_layers(model.hf)) + 4)
        budget = int(a.act_gpu_budget_gb * 2**30)
        excess = row_bytes - budget + (sac_res(lengths.max()) if sac_res else 0)
        size = excess + (1 << 30)
        if a.act_arena_gb:
            size = min(size, int(a.act_arena_gb * 2**30))
        arena = PinnedArena(size, pin=not a.cpu_test, overflow_pin=False, exact=True) if excess > 0 else None
        offl = BudgetOffload(arena, budget, min_bytes=a.offload_min_bytes, only_cuda=not a.cpu_test)
        if rank == 0:
            print(f"[offload] saved activations beyond {a.act_gpu_budget_gb:g} GiB per row go to a pinned arena of "
                  f"{(arena.buf.numel() if arena is not None else 0) / 2**30:.1f} GiB per rank (longest row "
                  f"{row_bytes / 2**30:.1f} GiB; overflow to pageable memory; "
                  f"{'exact cudaHostRegister' if arena is not None and arena.registered else 'torch pinned allocator'})", flush=True)

    wb = None
    if rank == 0 and a.wandb_project and os.environ.get("WANDB_API_KEY"):
        import wandb
        wb = wandb.init(project=a.wandb_project, name=a.wandb_run_name, id=a.wandb_run_name, resume="allow",
                        config={**vars(a), "total_steps": total, "warmup": warmup, "world": world})
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a") if rank == 0 else None

    params = [p for p in model.parameters() if p.requires_grad]
    tok_window, t_window = 0, time.time()
    bad_steps = 0                   # consecutive steps whose gradient norm was not finite (update skipped)
    for step in range(start, run_steps):
        if a.stop_at_step is not None and step >= a.stop_at_step:
            break
        t0 = time.time()
        epoch, ranks, n_label, imbalance = plan[step]
        if a.select_top_frac:   # the step's loss is the (weighted) mean over the selected tokens of all its rows
            n_label = sum(row_norm(int(labels[r]), a.select_top_frac, a.select_weight) for rk in ranks for r in rk)
        mine = ranks[rank]
        lr = (lr_wsd(step, a.peak_lr, warmup, a.decay_start, a.decay_steps) if a.lr_schedule == "wsd"
              else lr_at(stretch_step(step, a.lr_stretch_from, a.lr_stretch_total, total), a.peak_lr, floor, warmup,
                         a.lr_stretch_total or total))
        for g in opt.param_groups:
            g["lr"] = lr
        stats = torch.zeros(14, device=dev, dtype=torch.float64)  # loss sums x3, label tokens x3, tokens, rows, all-token loss, all labels, selected loss, selected, hint loss, hint labels
        # default: a rank's rows accumulate unsharded gradients locally and reduce after its last row; --zero3: every
        # row's gradient is reduce-scattered as it is produced (all ranks run the same number of rows, lockstep)
        model.set_requires_gradient_sync(a.zero3)
        model.set_reshard_after_backward(a.zero3)
        work = mine if mine else [None]
        moved = 0
        for j, r in enumerate(work):
            last = j == len(work) - 1
            if last and not a.zero3:
                model.set_requires_gradient_sync(True)
                model.set_reshard_after_backward(True)
            if r is None:          # a rank without rows (only in an epoch's last partial step): join the collectives
                ids = torch.tensor([[0, 0, 0]], device=dev)
                (model(ids, 1) * 0.0).backward()
                continue
            ids = torch.from_numpy(ids_mm[offsets[r]:offsets[r + 1]].astype(np.int64)).to(dev, non_blocking=True)[None]
            ls = int(label_start[r])
            if offl is not None:
                offl.reset(sac_res(lengths[r]) if sac_res else 0)
                if arena is not None:
                    arena.reset()
                with offl:
                    loss_sum = model(ids, ls)
                moved = max(moved, offl.moved)
            elif arena is not None and lengths[r] >= a.offload_activations_min_len:
                arena.reset()
                with ActivationOffload(arena):
                    loss_sum = model(ids, ls)
            else:
                loss_sum = model(ids, ls)
            (loss_sum * (world / n_label)).backward()
            s = int(source[r])
            hint = s == 3                                # source 3 = hinted-prompt CoT rows: counted as cot, also logged apart
            s = 2 if hint else s
            stats[s] += loss_sum.detach().double()
            full_sum, w_sum, sel_sum, k_sel = model.last_stats
            stats[3 + s] += float(w_sum)                 # = all label tokens unless --select-top-frac
            if hint:
                stats[12] += loss_sum.detach().double()
                stats[13] += float(w_sum)
            stats[8] += full_sum.double()
            stats[9] += float(lengths[r] - ls)
            stats[10] += sel_sum.double()
            stats[11] += float(k_sel)
            stats[6] += float(lengths[r])
            stats[7] += 1
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        # A NaN/inf gradient would make every weight NaN and the next saves would replace the good checkpoints. The norm
        # is global (same on every rank), so all ranks skip the update together; three in a row stop the run (a restart
        # resumes from the last checkpoint).
        skipped = not bool(torch.isfinite(gn))
        if skipped:
            bad_steps += 1
            if rank == 0:
                print(f"[step {step + 1}] non-finite grad norm {float(gn)}: optimizer step skipped ({bad_steps} in a row)",
                      flush=True)
            if bad_steps >= 3:
                raise RuntimeError(f"{bad_steps} non-finite gradient norms in a row at step {step + 1}: stopping")
        else:
            bad_steps = 0
            opt.step()
        opt.zero_grad(set_to_none=True)
        dist.all_reduce(stats)
        done = step + 1
        dt = time.time() - t0
        tok_window += int(stats[6].item())
        if rank == 0:
            lt = stats[3:6].sum().item()
            rec = {"step": done, "epoch": epoch, "lr": lr, "loss": stats[:3].sum().item() / max(1.0, lt),
                   "grad_norm": float(gn), "step_s": round(dt, 2), "tokens": int(stats[6].item()),
                   "label_tokens": int(lt), "rows": int(stats[7].item()), "imbalance": round(imbalance, 3),
                   "tok_per_s": round(stats[6].item() / max(dt, 1e-9)),
                   "mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 1) if dev.type == "cuda" else 0.0}
            if skipped:
                rec["skipped_update"] = 1
            if a.select_top_frac:     # "loss" = the objective (weighted mean); loss_all = over every label token
                rec["loss_all"] = stats[8].item() / max(1.0, stats[9].item())
                rec["loss_sel"] = stats[10].item() / max(1.0, stats[11].item())
                rec["selected_tokens"] = int(stats[11].item())
            if arena is not None:
                rec["offload_overflow"] = arena.overflow
            if offl is not None:
                rec["offload_gb_r0"] = round(moved / 2**30, 3)      # rank 0's largest per-row offload this step
                rec["offload_bytes_r0"] = int(moved)
            for c, nm in enumerate(SRC_NAMES):
                if stats[3 + c] > 0:
                    rec[f"loss_{nm}"] = stats[c].item() / stats[3 + c].item()
            if stats[13] > 0:
                rec["loss_hint"] = stats[12].item() / stats[13].item()
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if done <= 20 or done % 10 == 0:
                print(f"[step {done}/{total}] loss {rec['loss']:.4f} lr {lr:.2e} gn {rec['grad_norm']:.3f} "
                      f"{dt:.1f}s {rec['tok_per_s'] / 1e3:.1f}k tok/s rows {rec['rows']} imb {imbalance:.2f} "
                      f"mem {rec['mem_gb']}G", flush=True)
            if wb:
                wb.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=done)
        if done in milestones and not a.no_export:
            dest = os.path.join(a.out, f"milestone-{done}")
            # rank 0 decides (the export lands on its node; other nodes' run dirs never hold milestones): a collective
            need = torch.tensor([0.0 if os.path.exists(os.path.join(dest, "milestone.json")) else 1.0], device=dev)
            dist.broadcast(need, src=0)
            if need.item() > 0:
                export_hf(model, a.base_model, dest, rank, {"step": done, "frac": milestones[done], "total_steps": total})
                if rank == 0:
                    print(f"[export] milestone-{done} ({milestones[done]:.0%})", flush=True)
        stop_now = stop_ms = False
        if a.stop_file or a.stop_at_milestone_file:    # every rank checks the stop files; any rank -> all act
            flags = torch.tensor([1.0 if a.stop_file and os.path.exists(a.stop_file) else 0.0,
                                  1.0 if a.stop_at_milestone_file and os.path.exists(a.stop_at_milestone_file) else 0.0],
                                 device=dev)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            stop_now = flags[0].item() > 0
            stop_ms = flags[1].item() > 0 and done in milestones
        stop_here = (a.stop_at_step is not None and done >= a.stop_at_step) or stop_now or stop_ms
        at_ms = a.save_at_milestones and done in milestones
        if done % a.save_every == 0 or stop_here or done == run_steps or at_ms:
            protect = {f"step_{m:06d}" for m in milestones} if a.save_at_milestones else ()
            save_resumable(model, opt, a.out, done, rank, keep=2 if node_info()[2] > 1 else 1, protect=protect)
            if rank == 0:
                json.dump({"step": done, "total_steps": total, "time": time.time()},
                          open(os.path.join(a.out, "trainer_state.json"), "w"))
                if stop_now:
                    print(f"[stop-file] {a.stop_file}: saved ckpt/step_{done:06d}, exiting", flush=True)
                elif stop_ms:
                    print(f"[stop-at-milestone] {a.stop_at_milestone_file}: milestone {done} exported, ckpt/step_{done:06d} "
                          f"saved, exiting", flush=True)
        if done in export_steps and not a.no_export:
            # after the checkpoint: the resumable save comes first
            dest = os.path.join(a.out, f"snapshot-{done}")
            need = 0.0 if os.path.exists(os.path.join(dest, "snapshot.json")) else 1.0
            if need and rank == 0 and shutil.disk_usage(a.out).free < a.export_min_free_gb * 2**30:   # checkpoints need the room
                print(f"[snapshot] WARNING snapshot-{done} skipped: {shutil.disk_usage(a.out).free / 2**30:.0f} GiB free "
                      f"in {a.out} (< {a.export_min_free_gb:g})", flush=True)
                need = 0.0
            need = torch.tensor([need], device=dev)
            dist.broadcast(need, src=0)
            if need.item() > 0:
                t_x = time.time()
                meta = {"step": done, "epoch": round(epoch_position(done, steps_per_epoch), 4), "total_steps": total,
                        "lr": lr, "dtype": {"fp32": "float32", "bf16": "bfloat16"}[a.export_dtype],
                        "weights_only": True, "created": time.time()}
                export_hf(model, a.base_model, dest, rank, meta,
                          dtype=torch.float32 if a.export_dtype == "fp32" else torch.bfloat16, meta_name="snapshot.json")
                if rank == 0:
                    print(f"[snapshot] snapshot-{done} (epoch {meta['epoch']}, {a.export_dtype}, "
                          f"{time.time() - t_x:.0f}s)", flush=True)
        if stop_now or stop_ms:
            break
    if wb:
        wb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
