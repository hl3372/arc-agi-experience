#!/usr/bin/env python
"""Multi-GPU RL with verifiable rewards (GRPO family) for long-reasoning ARC solvers: vLLM samplers + an FSDP2 trainer.

    torchrun --nproc_per_node 8 -m arcsolver.rl.train --out RUN --student MODEL [flags]   (see arcsolver/rl/README.md)

One step:
- sampling: rank 0 owns one vLLM engine per GPU (CUDA_VISIBLE_DEVICES=i, --engine-gpu-util of each GPU), or, with
  --node-sampler-services 1, reaches every node's engines through that node's sampler service (node_service.py). A
  step's prompt groups are dealt to the engines (static split, a shared queue, or continuous streaming) and generated
  concurrently; eval prompts likewise.
- rewards: rank 0 scores every rollout (rewards.score_rollout: exact-match outcome, optional shaping), forms the group
  advantages and drops the groups without a learning signal.
- training: every rank holds the FSDP2 policy (fsdp.py). Rank 0 scatters the rollouts that need a forward to the ranks
  (equal counts, or cost-balanced with --balance lpt); each rank runs its rollouts' loss (scaled by world size: the
  fp32 reduce-scatter averages) and pads with zero-loss dummies. The per-rollout gradients are summed in fp32; the
  token-mean normaliser is the step's total. Clip, AdamW (fp32 masters), constant LR after warmup.
- time-sharing (--timeshare 1): the engines sleep while the trainer trains and the trainer's state goes to CPU while
  the engines sample, so both share the same GPUs.
- sync: the full bf16 state is gathered (and, with --sampler-fp8, re-quantised to the sampler checkpoint's FP8 format)
  into --sync-dir; every engine reloads it; a probe compares engine 0 with the trainer.
- checkpoints: checkpoints/step_N/{dcp/ or dcp_model/ + dcp_optim/ (sharded fp32 masters + AdamW), model/ (HF bf16
  export), model_fp8/ (sampler export), state.json, COMPLETE}; a rerun resumes from the latest complete one.
The other ranks idle while rank 0 samples and evaluates; the process group timeout is --dist-timeout-h.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import core as oc
from arcsolver.rl import data as ot
from arcsolver.rl import rewards as rr
from arcsolver.rl import config as RT


def load_logit_bias(path):
    """--eval-logit-bias file -> {str(token_id): float(bias)} (JSON keys stay strings over the sampler socket; the
    worker converts them)."""
    with open(path) as f:
        d = json.load(f)
    lb = {str(int(k)): float(v) for k, v in (d.get("logit_bias", d) if isinstance(d, dict) else {}).items()}
    if not lb or not all(math.isfinite(v) for v in lb.values()):
        raise SystemExit(f"--eval-logit-bias {path}: no finite token biases")
    return lb


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--engine-gpu-util", type=float, default=0.40)
    pre.add_argument("--dist-timeout-h", type=float, default=4.0)
    pre.add_argument("--min-free-gb-trainer", type=float, default=60.0)
    # time-share every GPU between its vLLM engine and the trainer
    pre.add_argument("--timeshare", type=int, default=0,
                     help="1 = engines sleep while the trainer trains; the trainer's state goes to CPU while they sample")
    pre.add_argument("--defer-optimizer-reload", action="store_true",
                     help="keep Adam moments on CPU through forward/backward; restore before the optimizer update")
    pre.add_argument("--sampler-model", default="", help="model dir the engines start from ('' = --student)")
    pre.add_argument("--sampler-fp8", type=int, default=0,
                     help="1 = sync the weights in the sampler checkpoint's FP8 format (fp8_quant); --sampler-model FP8")
    pre.add_argument("--kv-cache-dtype", default="auto")
    pre.add_argument("--sampler-log-stats", action="store_true")
    pre.add_argument("--save-first-step", action="store_true",
                     help="checkpoint step 1 as well as the regular save interval")
    pre.add_argument("--restore-only", action="store_true",
                     help="restore a completed checkpoint on all ranks, record evidence, exit without rollouts")
    pre.add_argument("--sampler-dispatch-prompts", type=int, default=0,
                     help="positive = engines draw this many prompt groups from a shared queue; 0 = static split")
    pre.add_argument("--sampler-stream-seqs", type=int, default=0,
                     help="positive = continuous refill: every engine keeps this many single samples decoding, "
                          "drawing from one global queue as samples finish (overrides --sampler-dispatch-prompts)")
    pre.add_argument("--sampler-kv-high", type=float, default=0.9,
                     help="streaming: an engine asks for more samples only while its KV cache is below this fraction")
    pre.add_argument("--sampler-control-file", default="",
                     help="JSON {stream_seqs, kv_high} re-read before every generate call (tune without a restart)")
    pre.add_argument("--engine-sleep-level", type=int, default=2, choices=(1, 2),
                     help="vLLM sleep level while the trainer trains. 2 discards the weights and the reload refills the "
                          "target model only, which leaves the MTP draft head garbage (acceptance falls to ~1 token, "
                          "half-speed sampling); 1 backs the weights up to CPU so the draft head survives")
    pre.add_argument("--engine-no-prefix-caching", action="store_true",
                     help="engines without the prefix cache (hybrid GDN models then skip the 'align' state copies)")
    pre.add_argument("--engine-block-size", type=int, default=0, help="vLLM KV block size (0 = vLLM's choice)")
    pre.add_argument("--engine-attention-backend", default="", help="vLLM attention backend ('' = vLLM's choice)")
    pre.add_argument("--dcp-split", type=int, default=0,
                     help="1 = checkpoints hold dcp_model/ (fp32 masters) and dcp_optim/ (AdamW) separately, so the "
                          "weights alone can be kept as a fallback (--resume-weights-only)")
    pre.add_argument("--train-all-rollouts", action="store_true",
                     help="rehearsal only: run backward and the optimizer on every kept rollout, zero advantage "
                          "included (zero loss and gradient: the weights do not move, every training path runs)")
    pre.add_argument("--minibatch-max", type=int, default=1,
                     help="optimizer updates per rollout batch, at most: the mixed groups are split into "
                          "min(this, groups // --minibatch-min-groups) minibatches (>= 1), each its own token-mean "
                          "update; 1 = one update per batch. The control file's minibatch_max can lower it per step")
    pre.add_argument("--minibatch-min-groups", type=int, default=8)
    pre.add_argument("--resume-weights-only", action="store_true",
                     help="resume the newest checkpoint's fp32 masters with a fresh AdamW (a weights-only fallback); "
                          "the learning rate re-warms over --rewarmup-steps")
    pre.add_argument("--rewarmup-steps", type=int, default=5)
    pre.add_argument("--bf16-export-every", type=int, default=1,
                     help="write the HF bf16 model/ export at every Nth checkpoint step (1 = every save, 0 = never; the "
                          "final step always). With --sampler-fp8, resume needs only dcp/ + model_fp8/ (the export's "
                          "full-state gather on rank 0 costs host RAM at every save)")
    pre.add_argument("--activation-offload-mb", type=int, default=0,
                     help="positive = keep saved activations of at least this many MB in pinned host memory "
                          "between forward and backward (fsdp.ActivationOffload; for large models at long contexts)")
    pre.add_argument("--mtp-tokens", type=int, default=0, help="MTP speculative tokens in the engines (0 = off)")
    pre.add_argument("--pg-loss", choices=["ppo", "tis_sc"], default="ppo",
                     help="ppo (clipped ratio against the behaviour log-probs) or tis_sc (truncated importance sampling "
                          "+ score centering, sc.py)")
    pre.add_argument("--pool-drop-solved", type=int, default=0,
                     help="1 = a task whose rollout group is all correct (natural outcomes, every "
                          "sample of the group) is never drawn again (WeightedOrder.mark_solved, kept in the checkpoint; "
                          "seeded from the rollout records when the order starts fresh on a resume). Needs --pool-order "
                          "weighted without --bank-share")
    pre.add_argument("--pool-drop-solved-seed", default="",
                     help="DIR:UPTO[,DIR2:UPTO2...] (a fresh restart that keeps what earlier runs learned about solved "
                          "tasks): also mark the tasks with an all-correct group in "
                          "DIR/rollouts/step_N.jsonl.gz, N <= UPTO (other runs' records, same group size), as solved at a "
                          "fresh start")
    pre.add_argument("--kl-coef", type=float, default=0.0,
                     help="reference KL penalty coefficient (per-token k3 estimator of KL(policy || reference), in the "
                          "same token mean as the policy-gradient term; loss.kl_k3); 0 = off. Needs --pg-loss ppo or "
                          "tis_sc")
    pre.add_argument("--ref-model", default="",
                     help="the frozen KL reference (HF bf16 dir, on every node); '' = --student (the run's start "
                          "checkpoint: a resume restores the trainer from --student + DCP, so this stays the start)")
    pre.add_argument("--eval-bank-all", action="store_true", help="evaluate every task in the bank holdout split")
    pre.add_argument("--sc-topk", type=int, default=0, help="sampler top-k for score centering (0 = no centering)")
    pre.add_argument("--tis-cap", type=float, default=2.0, help="TIS weight cap C (<= 0 = no TIS)")
    pre.add_argument("--topk-dir", default="", help="where engines write top-k arrays ('' = /dev/shm/rl_topk_<run>)")
    pre.add_argument("--fsdp-keep-gathered", type=int, default=1,
                     help="1 = keep parameters gathered across a rank's rollouts (faster; fits 4B/9B); 0 = reshard per "
                          "layer (less memory, for larger models)")
    pre.add_argument("--freeze-embed", type=int, default=0)
    pre.add_argument("--mask-truncated", type=int, default=0,
                     help="1 = DAPO overlong filtering: a rollout cut at the token budget still "
                          "counts as a failure in its group's advantage statistics, but leaves the loss and its "
                          "token-mean normaliser (0 = off: truncated rollouts train with their negative advantage)")
    pre.add_argument("--trunc-neg-scale", type=float, default=1.0,
                     help="multiply the sequence advantage of every truncated rollout (finish "
                          "length) of a kept group by this, after the group statistics (where it counted as R = 0); "
                          "with --mask-truncated 0 it stays in the loss with the smaller push-down. 1 = unchanged")
    # dynamic sampling (experimental, off by default): keep adding prompt rounds into free engine slots until the
    # finished rounds hold --dyn-target-groups groups with an outcome signal, then abort the rest (see
    # sampler_dispatch.generate_rounds). Needs --sampler-stream-seqs and --reward-stream.
    pre.add_argument("--dyn-target-groups", type=int, default=0,
                     help="0 = off; else stop sampling once whole rounds (in dispatch order) hold this many groups with "
                          "outcome variance (DAPO dynamic sampling); round 0 is --prompts-per-step prompts")
    pre.add_argument("--dyn-round-prompts", type=int, default=32, help="prompts per extra round")
    pre.add_argument("--dyn-max-prompts", type=int, default=0, help="cap on prompts sampled per step (0 = 4 x prompts)")
    pre.add_argument("--dyn-margin", type=float, default=1.1,
                     help="add a round only while (informative so far + unjudged prompts x running informative "
                          "fraction) < target x margin, so the tail of the needed rounds is not slowed for nothing")
    pre.add_argument("--dyn-frac-init", type=float, default=0.4, help="prior informative fraction (8 pseudo-prompts)")
    pre.add_argument("--dyn-heartbeat-s", type=float, default=2.0, help="engines ask at least this often while decoding")
    # rank-0 work that would otherwise run serially while every GPU idles
    pre.add_argument("--reward-stream", type=int, default=0,
                     help="positive = with --sampler-stream-seqs, score each rollout (detokenise, rewards) "
                          "in this many rank-0 threads as its sample arrives, while the other samples still "
                          "decode, instead of all of them after the last one; the same scores, rollouts and order. "
                          "0 = serial after sampling")
    pre.add_argument("--rollout-write-async", action="store_true",
                     help="write the step's rollout record (jsonl.gz) and token file (npz) in a rank-0 thread while "
                          "the ranks train; joined before a checkpoint commits and at the end of "
                          "the step. The same files")
    pre.add_argument("--trainer-attn", choices=["sdpa", "fa3", "fa4"], default="sdpa",
                     help="attention kernel of the trainer's softmax layers: sdpa (default), fa3 (FlashAttention-3 from "
                          "the kernels hub; ARC_FA3_DIR) or fa4 (FlashAttention-4 CuTe; ARC_FA4_DIR); attn_backends.py")
    pre.add_argument("--trainer-speed", default="",
                     help="comma-separated opt-in trainer speedups (trainer_speed.py): sac_attn, fused_head, gdn_gva, "
                          "compile_norm_mlp; '' = none")
    pre.add_argument("--balance", choices=["equal", "lpt"], default="equal",
                     help="rollouts to ranks: equal = equal counts, every backward reduce-scatters (default); lpt = "
                          "by estimated cost with no-sync fp32 accumulation and one reduce-scatter per minibatch "
                          "(needs --fsdp-keep-gathered 1; +1 unsharded fp32 gradient copy per rank)")
    pre.add_argument("--balance-quad-tokens", type=float, default=60000.0,
                     help="lpt cost model: cost(n) = n + n^2 / this (the length at which the quadratic attention "
                          "term equals the linear one)")
    pre.add_argument("--train-budget-force", type=int, default=0,
                     help="0 = off; else every on-policy "
                          "rollout cut at the token cap gets its reasoning closed (the eval's budget-forcing suffix) "
                          "and up to this many tokens to answer, sampled as the eval samples; a right answer earns "
                          "--forced-credit (rewards.forced_credit_rewards / scale_forced_advantages). The forced "
                          "tokens are only scored, never trained")
    pre.add_argument("--train-budget-force-from", type=int, default=0,
                     help="first step whose truncated rollouts get forced answers (0 = every step)")
    pre.add_argument("--train-force-inline", type=int, default=0,
                     help="1 = a rollout cut at the cap gets its forced answer queued on its "
                          "engine as soon as it ends (vllm_worker stream msg force), while its KV and GDN states are "
                          "still in the engines' prefix cache, instead of a separate pass that re-prefills every "
                          "~60-80k-token context. Same requests, room check, budget and sampling settings as "
                          "force_answers; only the forced answers' seeds differ. Turns the engines' prefix caching on "
                          "(overrides --engine-no-prefix-caching); needs streaming dispatch (--sampler-stream-seqs)")
    pre.add_argument("--eval-force-inline", type=int, default=0,
                     help="--train-force-inline for the evals' forced answers "
                          "(--eval-budget-force): each cut eval sample's answer is queued on its engine as it ends")
    pre.add_argument("--forced-credit", type=float, default=0.5,
                     help="share of a natural success's credit that a right forced answer earns, in (0, 1]")
    pre.add_argument("--eval-steps", default="",
                     help="comma-separated extra eval steps on top "
                          "of --eval-every (also run on resume when their eval is missing); '' = none")
    pre.add_argument("--pool-extend", action="store_true",
                     help="resume a frontier_mix run whose pool GREW (new bank tasks): when "
                          "the checkpoint's strata sizes differ from this pool's, the never-drawn bank tasks (by task "
                          "id, from the rollout records) are drawn first in a seeded shuffle and the revisit queue is "
                          "re-seeded from the records (mixorder.extend_from_records); a checkpoint of this pool's "
                          "size resumes as usual")
    pre.add_argument("--pool-extend-tiers", action="store_true",
                     help="with --pool-extend, the bank "
                          "tasks drawn fewer times than the most-drawn ones also go to pending, by draw count (never "
                          "drawn first), so the pass in progress finishes before any task repeats; inert on a resume "
                          "without a pool change")
    pre.add_argument("--pool-focus-file", default="",
                     help="JSON list of task ids (a .json suffix is ignored) of which each step draws at least "
                          "--pool-focus-per-step")
    pre.add_argument("--pool-focus-per-step", type=int, default=0,
                     help="with --pool-order frontier_mix: each step draws at least this many focus tasks (their "
                          "never-drawn ones from the new-task queue first, then revisits: fewest draws first, then the "
                          "oldest latest draw); other fresh draws never pick a focus task")
    pre.add_argument("--pool-focus-order", choices=["fewest", "shuffle"], default="fewest",
                     help="fewest: never-drawn focus tasks first, then fewest draws / oldest; shuffle: "
                          "seeded shuffled cycles over the whole focus set (sorted by task id), new and drawn tasks mixed")
    pre.add_argument("--pool-focus-shuffle-from", type=int, default=0,
                     help="with --pool-focus-order shuffle: the first step drawn in shuffle order (its cycle position is "
                          "re-seeded at every start from the focus draws recorded since this step)")
    pre.add_argument("--pool-focus-max-frac", type=float, default=0.5,
                     help="the largest share of the bank the focus set may cover (the other fresh draws skip it)")
    pre.add_argument("--pool-focus-cap-file", default="",
                     help="JSON list of focus task ids (e.g. ones never solved so far) of which a step takes at most "
                          "--pool-focus-cap; further ones in the shuffled "
                          "cycle are passed over until the next cycle and the next focus tasks fill the step")
    pre.add_argument("--pool-focus-cap", type=int, default=10,
                     help="with --pool-focus-cap-file: at most this many of its tasks per step")
    pre.add_argument("--pool-focus-cap-schedule", default="",
                     help="with --pool-focus-cap-file: 'STEP:CAP,STEP:CAP,...' (increasing steps; each cap holds until the "
                          "next knot); replaces "
                          "--pool-focus-cap / --pool-focus-cap-from. Keep the earlier knots when adding one: a resume "
                          "replays the cursor with every recorded step's own cap")
    pre.add_argument("--pool-focus-cap-from", type=int, default=1,
                     help="with --pool-focus-cap-file: the first step drawn with the cap (the shuffle cursor of a resume "
                          "is replayed from the records with the cap from this step on)")
    pre.add_argument("--pool-focus-from-step", type=int, default=1,
                     help="focus draw counts are seeded at start from the rollout records of this step on (no focus "
                          "task was drawn before it)")
    pre.add_argument("--pool-new-per-step", type=int, default=0,
                     help="spread new bank tasks over the steps: with --pool-order frontier_mix, "
                          "never-drawn bank tasks found by a --pool-extend wait in a queue and each step takes at most "
                          "this many of them (inside its bank quota); 0 = they are drawn first, all at once")
    pre.add_argument("--train-split-tests", action="store_true",
                     help="each rollout group of a multi-test task sees one of its tests (seeded by run seed, step and "
                          "group) as a single-test task, rewarded on that test alone (config.split_test_view)")
    pre.add_argument("--eval-tests", choices=["joint", "split", "both"], default="joint",
                     help="joint = one prompt per task with all its test inputs; "
                          "split = one prompt per test input of a multi-test task (n samples each), merged per task "
                          "(pass@k per output averaged, as score_task), written to evals/step_N.split.*; both = joint "
                          "then split")
    pre.add_argument("--prefix-groups", type=int, default=0,
                     help="this many of each step's groups continue a frozen prefix of one of "
                          "the previous step's rollouts (select_prefix_sources): group_size continuations sampled from "
                          "the prefix, rewarded on prefix + continuation, trained on the continuation only; the fresh "
                          "draws shrink by the same number")
    pre.add_argument("--prefix-from", type=int, default=0, help="first step with --prefix-groups (0 = every step)")
    pre.add_argument("--prefix-every", type=int, default=1,
                     help="prefix groups only on steps divisible by N (1 = every step); the other steps draw "
                          "--prompts-per-step fresh prompts")
    pre.add_argument("--prefix-extra-prompts", type=int, default=0,
                     help="on steps with prefix groups, this many more fresh prompts (the step has --prompts-per-step + "
                          "N groups)")
    pre.add_argument("--prefix-max-zero", type=int, default=8,
                     help="at most this many prefix sources from groups with no fully right rollout (never from groups "
                          "whose rollouts were all right)")
    pre.add_argument("--prefix-min-start", type=int, default=2500, help="the cut is at least this many tokens in")
    pre.add_argument("--prefix-early-frac", type=float, default=0.0,
                     help="with --prefix-early-prob the cut lands in the first this fraction of the "
                          "source's reasoning, otherwise in the rest (0 = uniform over the allowed range, as before)")
    pre.add_argument("--prefix-early-prob", type=float, default=0.8)
    pre.add_argument("--prefix-zero-max-frac", type=float, default=1.0,
                     help="a source from a group with no fully right rollout is cut at most "
                          "this fraction of the way through its reasoning (1.0 = no extra limit)")
    pre.add_argument("--prefix-end-margin", type=int, default=3000,
                     help="the cut is at least this many tokens before the source's </think> (or its end if truncated)")
    pre.add_argument("--len-pair-lambda", type=float, default=0.0,
                     help="thresholded pairwise length shaping (rewards.pairwise_length_shaping) with this maximum "
                          "coefficient lambda_0; 0 = off")
    pre.add_argument("--len-pair-min-correct", type=int, default=4,
                     help="only groups with at least this many fully right rollouts get the length shaping")
    pre.add_argument("--len-pair-delta", type=float, default=10000, help="pairwise length tolerance (tokens)")
    pre.add_argument("--len-pair-span", type=float, default=20000,
                     help="extra gap beyond the tolerance that reaches the full pairwise preference (tokens)")
    pre.add_argument("--len-pair-gamma", type=float, default=0.5,
                     help="at most this share of a right rollout's advantage can be removed in a group with failures")
    pre.add_argument("--len-pair-allright", type=int, choices=[0, 1], default=1,
                     help="1: all-right groups get the length shaping too (and are kept for it); 0: only groups that "
                          "also have failures")
    pre.add_argument("--eval-logit-bias", default="",
                     help="JSON file {\"logit_bias\": {token_id: bias}} applied to every eval "
                          "generation (not training); vLLM ignores logit_bias under speculative decoding, so it needs "
                          "--mtp-tokens 0")
    pre.add_argument("--eval-joint-steps", default="",
                     help="with --eval-tests split: comma-separated eval steps that ALSO run the joint eval (a bridge "
                          "to the earlier joint evals)")
    pre.add_argument("--eval-partial-dir", default="",
                     help="resumable evals: every finished eval sample and "
                          "forced-answer continuation is saved to DIR/step_N[.split]/ as it arrives (gzip JSON, tmp + "
                          "fsync + rename); a rerun loads them and generates only the missing samples with the same "
                          "seeds. One DIR per evaluated model; MANIFEST.json pins the model, settings and prompts. "
                          "Needs streaming dispatch (--sampler-stream-seqs). '' = off (unchanged)")
    pre.add_argument("--rollout-store-dir", default="",
                     help="preemption-safe sampling: every finished training sample (record + top-k arrays) and forced "
                          "answer of the "
                          "step being drawn is saved to DIR/step_N/ as it arrives; a restart that redraws step N with "
                          "the same policy (sampler weight digest), prompts, seeds and caps loads them and samples only "
                          "the rest. A step's directory is removed once the next step's checkpoint is complete. Needs streaming "
                          "dispatch and --per-request-seeds. '' = off (unchanged)")
    pre.add_argument("--node-sampler-services", type=int, default=0,
                     help="1 = every node's local rank 0 runs node_service.py owning its local engines; rank 0 "
                          "reaches all of them over authenticated TCP (needed for more than one node)")
    pre.add_argument("--sampler-service-addrs", default="",
                     help="comma-separated address per node (node-rank order) its service binds to, e.g. ADDR0,ADDR1")
    pre.add_argument("--sampler-service-port", type=int, default=29760, help="node n's service listens on this + n")
    pre.add_argument("--sync-versioned", type=int, default=0,
                     help="1 = each sync writes a fresh <sync-dir>/v<step> (always on with --node-sampler-services); the "
                          "previous version is removed only after every engine acknowledged the new one")
    pre.add_argument("--objective-contract", default="", choices=["", "grpo_token_v1", "grpo_dapo_kl_v1", "grpo_dapo_kl_v2",
                                                                   "grpo_dapo_kl_sc_v1", "grpo_dapo_kl_sc_prefix_v1", "grpo_dapo_kl_sc_prefix_v2",
                                                                   "grpo_dapo_kl_sc_prefix_v3", "grpo_dapo_clip_v1"],
                     help="refuse to launch unless every objective/workload flag matches the named contract "
                          "(contract.py; grpo_token_v1 = vanilla GRPO, token-level loss, binary outcome); '' = off")
    pre.add_argument("--allow-objective-change", default="",
                     help="resume: comma-separated objective keys allowed to differ from the checkpoint's snapshot "
                          "(e.g. lr); 'legacy' accepts a checkpoint saved without one. Default: any change is refused")
    pre.add_argument("--adam-beta2", type=float, default=0.999, help="AdamW beta2")
    pre.add_argument("--weight-decay", type=float, default=0.0,
                     help="decoupled AdamW weight decay on the matrices only (as SFT); 0 = none (one param group)")
    pre.add_argument("--ppo-old-logp", choices=["sampler", "trainer"], default="sampler",
                     help="--pg-loss ppo: the ratio's denominator: the FP8 sampler's log-probs (as before) or the bf16 "
                          "trainer's own, recomputed at the start of each step before any update (DAPO clip-higher over "
                          "several minibatch updates per step without the FP8/bf16 mismatch in the clip)")
    pre.add_argument("--lr-ramp-to", type=float, default=0.0,
                     help="0 = off; else the LR rises linearly from --lr to this over --lr-ramp-steps steps after "
                          "--lr-ramp-from")
    pre.add_argument("--lr-ramp-from", type=int, default=0, help="the last step at --lr (the ramp starts after it)")
    pre.add_argument("--lr-ramp-steps", type=int, default=10)
    # DAPO-style soft overlong punishment: rewards.overlong_penalty
    pre.add_argument("--len-overlong-start", type=float, default=0.0,
                     help="0 = off; else the share of --max-new-tokens where the penalty starts (0.8: from 117.6k of 147k)")
    pre.add_argument("--len-overlong-p0", type=float, default=0.05, help="penalty just past the start")
    pre.add_argument("--len-overlong-p1", type=float, default=0.2, help="penalty at the cap and for truncated rollouts")
    pre.add_argument("--len-overlong-mixed-only", type=int, default=1,
                     help="1 = only rollouts of groups with mixed natural outcomes are penalised")
    pre.add_argument("--init-checkpoint", default="",
                     help="a fresh run starts from this checkpoint dir (dcp_model + dcp_optim + model_fp8: weights AND "
                          "AdamW), e.g. a step of another run; ignored once the run has its own checkpoint")
    extra, rest = pre.parse_known_args(sys.argv[1:] if argv is None else argv)
    args = RT.parse_args(rest)
    for k, v in vars(extra).items():
        setattr(args, k, v)
    if args.balance == "lpt" and not args.fsdp_keep_gathered:
        raise SystemExit("--balance lpt needs --fsdp-keep-gathered 1 (resharding collectives every micro-batch "
                         "would need equal counts)")
    if args.sc_topk and args.pg_loss != "tis_sc":
        raise SystemExit("--sc-topk needs --pg-loss tis_sc")
    if args.sampler_fp8 and not args.sampler_model:
        raise SystemExit("--sampler-fp8 needs --sampler-model (the FP8 checkpoint the engines load)")
    args.sampler_model = args.sampler_model or args.student
    if args.sampler_dispatch_prompts < 0:
        raise SystemExit('--sampler-dispatch-prompts must be nonnegative')
    if args.defer_optimizer_reload and not args.timeshare:
        raise SystemExit('--defer-optimizer-reload requires --timeshare 1')
    if args.restore_only and (args.dry_run or args.no_resume or args.new_pool_on_resume):
        raise SystemExit('--restore-only requires real recovery with the saved prompt order')
    if args.mask_truncated and getattr(args, "train_budget_force", 0) > 0:
        raise SystemExit("--mask-truncated excludes --train-budget-force (forced answers credit truncated rollouts)")
    if not 0.0 <= args.trunc_neg_scale <= 1.0:
        raise SystemExit("--trunc-neg-scale must be in [0, 1]")
    if args.trunc_neg_scale != 1.0 and args.mask_truncated:
        raise SystemExit("--trunc-neg-scale excludes --mask-truncated (masked rollouts leave the loss)")
    if args.train_budget_force < 0 or not 0.0 < args.forced_credit <= 1.0:
        raise SystemExit("--train-budget-force must be >= 0 and --forced-credit in (0, 1]")
    try:
        eval_due(args, 0)
    except ValueError:
        raise SystemExit(f"--eval-steps must be comma-separated integers: {args.eval_steps!r}")
    try:
        {int(x) for x in str(args.eval_joint_steps or "").split(",") if x.strip()}
    except ValueError:
        raise SystemExit(f"--eval-joint-steps must be comma-separated integers: {args.eval_joint_steps!r}")
    if args.pool_focus_per_step and (args.pool_focus_per_step < 0 or args.pool_order != "frontier_mix"
                                     or not args.pool_focus_file
                                     or args.pool_focus_per_step >= args.prompts_per_step - args.prefix_groups):
        raise SystemExit("--pool-focus-per-step needs --pool-order frontier_mix, --pool-focus-file and fewer than "
                         "--prompts-per-step minus --prefix-groups")
    if args.pool_focus_order == "shuffle" and args.pool_focus_shuffle_from < 1:
        raise SystemExit("--pool-focus-order shuffle needs --pool-focus-shuffle-from (the first shuffled step)")
    if getattr(args, "pool_focus_cap_file", "") and not (args.pool_focus_per_step and args.pool_focus_order == "shuffle"
                                                         and args.pool_focus_cap >= 0 and args.pool_focus_cap_from >= 1):
        raise SystemExit("--pool-focus-cap-file needs --pool-focus-per-step, --pool-focus-order shuffle, "
                         "--pool-focus-cap >= 0 and --pool-focus-cap-from >= 1")
    if getattr(args, "pool_focus_cap_schedule", ""):
        try:
            knots = [tuple(int(v) for v in k.split(":")) for k in args.pool_focus_cap_schedule.split(",") if k.strip()]
            ok = bool(knots) and all(len(k) == 2 and k[0] >= 1 and k[1] >= 0 for k in knots) \
                and all(knots[j][0] > knots[j - 1][0] for j in range(1, len(knots)))
        except ValueError:
            ok = False
        if not ok or not args.pool_focus_cap_file:
            raise SystemExit(f"--pool-focus-cap-schedule must be STEP:CAP,... with increasing steps >= 1 and caps >= 0, "
                             f"with --pool-focus-cap-file: {args.pool_focus_cap_schedule!r}")
    if not (0 <= args.prefix_early_frac < 1 and 0 <= args.prefix_early_prob <= 1):
        raise SystemExit("--prefix-early-frac must be in [0, 1) and --prefix-early-prob in [0, 1]")
    if not 0 < args.prefix_zero_max_frac <= 1:
        raise SystemExit("--prefix-zero-max-frac must be in (0, 1]")
    if args.pool_new_per_step < 0 or (args.pool_new_per_step and args.pool_order != "frontier_mix"):
        raise SystemExit("--pool-new-per-step must be >= 0 and needs --pool-order frontier_mix")
    if getattr(args, "pool_extend_tiers", False) and not args.pool_extend:
        raise SystemExit("--pool-extend-tiers needs --pool-extend")
    if getattr(args, "train_force_inline", 0) and not (args.train_budget_force > 0 and args.sampler_stream_seqs > 0):
        raise SystemExit("--train-force-inline needs --train-budget-force > 0 and streaming dispatch "
                         "(--sampler-stream-seqs > 0)")
    if getattr(args, "eval_force_inline", 0) and not (args.eval_budget_force > 0 and args.sampler_stream_seqs > 0):
        raise SystemExit("--eval-force-inline needs --eval-budget-force > 0 and streaming dispatch "
                         "(--sampler-stream-seqs > 0)")
    if args.eval_logit_bias:
        if args.mtp_tokens > 0:
            raise SystemExit("--eval-logit-bias needs --mtp-tokens 0 (vLLM ignores logit_bias with speculative decoding)")
        args.eval_logit_bias_map = load_logit_bias(args.eval_logit_bias)
    if args.len_pair_lambda and (args.len_pair_lambda < 0 or not 0 < args.len_pair_gamma < 1 or args.len_pair_span <= 0
                                 or args.len_pair_delta < 0 or args.len_pair_min_correct < 2 or args.len_allcorrect_beta):
        raise SystemExit("--len-pair-lambda needs lambda >= 0, 0 < gamma < 1, span > 0, delta >= 0, min-correct >= 2 "
                         "and excludes --len-allcorrect-beta")
    if args.prefix_every < 1 or args.prefix_extra_prompts < 0:
        raise SystemExit("--prefix-every must be >= 1 and --prefix-extra-prompts >= 0")
    if args.prefix_groups and not (0 < args.prefix_groups < args.prompts_per_step and args.save_rollouts
                                   and not args.dyn_target_groups):
        raise SystemExit("--prefix-groups needs 0 < N < --prompts-per-step, --save-rollouts (the previous step's records "
                         "and token ids) and excludes --dyn-target-groups")
    if args.dyn_target_groups > 0 and (not args.sampler_stream_seqs or not getattr(args, "reward_stream", 0)
                                       or args.pool_order != "frontier_mix"):
        raise SystemExit("--dyn-target-groups needs --sampler-stream-seqs, --reward-stream (rounds are judged on "
                         "streamed rewards) and --pool-order frontier_mix (take_more)")
    if args.node_sampler_services:
        if not (args.sampler_fp8 and args.timeshare and args.sampler_service_addrs):
            raise SystemExit("--node-sampler-services needs --sampler-fp8 1, --timeshare 1 and --sampler-service-addrs")
        # --sc-topk works across nodes: each node's engines write their top-k arrays under
        # <topk_dir>/n<node>/ on that node (RemoteEngineClient), and every rollout with a top-k file is trained by a
        # rank of the node that sampled it (plan_minibatches(nodes=...))
        args.sync_versioned = 1
    if args.pool_drop_solved and (args.pool_order != "weighted" or args.bank_share):
        raise SystemExit("--pool-drop-solved needs --pool-order weighted and no --bank-share (one WeightedOrder)")
    if args.pool_drop_solved_seed:
        if not args.pool_drop_solved:
            raise SystemExit("--pool-drop-solved-seed needs --pool-drop-solved 1")
        parse_solved_seed(args.pool_drop_solved_seed, check_dirs=False)
    if args.kl_coef < 0 or (args.kl_coef > 0 and args.pg_loss not in ("ppo", "tis_sc")):
        raise SystemExit("--kl-coef must be >= 0 and needs --pg-loss ppo or tis_sc")
    if args.kl_coef > 0 and not args.ref_model:
        args.ref_model = args.student
    if args.sync_versioned and not args.sampler_fp8:
        raise SystemExit("--sync-versioned needs --sampler-fp8 1 (the bf16 sync writer has one fixed directory)")
    if args.ppo_old_logp == "trainer" and args.pg_loss != "ppo":
        raise SystemExit("--ppo-old-logp trainer needs --pg-loss ppo")
    if not (0 < args.adam_beta2 < 1 and args.weight_decay >= 0):
        raise SystemExit("--adam-beta2 must be in (0, 1) and --weight-decay >= 0")
    if args.lr_ramp_to and not (args.lr_ramp_to > 0 and args.lr_ramp_steps >= 1 and args.lr_ramp_from >= 0):
        raise SystemExit("--lr-ramp-to needs > 0, --lr-ramp-steps >= 1 and --lr-ramp-from >= 0")
    if args.len_overlong_start and not (0 < args.len_overlong_start < 1 and 0 <= args.len_overlong_p0 <= args.len_overlong_p1):
        raise SystemExit("--len-overlong-start must be in (0, 1) with 0 <= --len-overlong-p0 <= --len-overlong-p1")
    if args.objective_contract:
        from arcsolver.rl import contract as rl_contract
        bad = rl_contract.check_contract(args, args.objective_contract)
        if args.objective_contract.startswith("grpo_dapo_kl") and os.path.realpath(args.ref_model) != os.path.realpath(args.student):
            bad.append(f"ref_model={args.ref_model!r}: contract {args.objective_contract} requires the start checkpoint "
                       "(--student)")
        if bad:
            raise SystemExit(f"--objective-contract {args.objective_contract}: {len(bad)} flag(s) differ:\n  "
                             + "\n  ".join(bad))
    args.topk_dir = args.topk_dir or f"/dev/shm/rl_topk_{args.run_name}"
    return args


class MultiEngine:
    """Drop-in for one sampler EngineClient over N engines: generate splits prompts round-robin and merges the
    outputs back in order; score goes to engine 0; reload goes to every engine concurrently."""

    def __init__(self, engines, dispatch_prompts=0, stream_seqs=0, kv_high=0.9, control_file="", policy=None):
        self.engines = engines
        # the committed policy identity (identity.policy_id); every record must carry it, and it changes
        # only when every engine acknowledged a reload of it. None = untracked (records are not checked).
        self.committed_policy = policy
        self.reload_failed = None                      # a tracked reload that did not commit: sampling is refused
        self.dispatch_prompts = dispatch_prompts
        self.stream_seqs = stream_seqs
        self.kv_high = kv_high
        self.control_file = control_file
        self.pool = cf.ThreadPoolExecutor(max_workers=len(engines))

    def stream_settings(self):
        """(stream_seqs, kv_high): the control file's values when it exists and parses, else the launch flags."""
        seqs, kv = self.stream_seqs, self.kv_high
        if self.control_file and os.path.isfile(self.control_file):
            try:
                with open(self.control_file) as f:
                    c = json.load(f)
                seqs = int(c.get("stream_seqs", seqs))
                kv = float(c.get("kv_high", kv))
                if seqs < 1 or not 0 < kv <= 1:
                    raise ValueError(c)
            except (OSError, ValueError, TypeError) as e:
                print(f"[sampler] ignoring control file {self.control_file}: {e}", flush=True)
                seqs, kv = self.stream_seqs, self.kv_high
        return seqs, kv

    def check_record(self, rec, where="sample"):
        if self.committed_policy is not None and rec.get("policy") != self.committed_policy:
            raise RuntimeError(f"{where}: a record of policy {str(rec.get('policy'))[:16]!r}, but the committed policy "
                               f"is {self.committed_policy[:16]!r} (mixed-policy rollouts are never accepted)")

    def _checked(self, on_sample):
        if on_sample is None or self.committed_policy is None:
            return on_sample

        def cb(e, j, rec):
            self.check_record(rec)
            return on_sample(e, j, rec)
        return cb

    def _check_outputs(self, r):
        if self.committed_policy is not None:
            for outs in r.get("outputs") or []:
                for rec in outs or []:
                    if rec is not None:
                        self.check_record(rec)
        return r

    def call(self, cmd, **kw):
        if cmd == "reload":
            want = kw.get("policy")
            self.committed_policy = None                       # unknown until every engine acknowledged (untracked without one)
            if want is not None:
                self.reload_failed = want                      # cleared only when every engine acknowledged it
            rs = list(self.pool.map(lambda e: e.call(cmd, **kw), self.engines))
            if want is not None:
                bad = [e.name for e, r in zip(self.engines, rs) if r.get("policy") != want]
                if bad:
                    raise RuntimeError(f"reload: engines {bad} did not acknowledge policy {want[:16]}")
                self.committed_policy = want
                self.reload_failed = None
            return {"seconds": max(r["seconds"] for r in rs), "acks": len(rs)}
        if cmd in ("generate_rounds", "generate", "score") and self.reload_failed is not None:
            raise RuntimeError(f"{cmd} refused: the reload of policy {self.reload_failed[:16]} never committed")
        if cmd in ("generate_rounds", "generate") and "on_sample" in kw:
            kw["on_sample"] = self._checked(kw["on_sample"])
        if cmd in ("generate_rounds", "generate"):
            return self._check_outputs(self._call(cmd, **kw))
        return self._call(cmd, **kw)

    def _call(self, cmd, **kw):
        if cmd == "generate_rounds":                   # dynamic sampling: streaming dispatch only
            from arcsolver.rl.sampler_dispatch import generate_rounds
            seqs, kv = self.stream_settings()
            hooks = {k: kw.pop(k, None) for k in ("on_sample", "more", "on_round")}
            hb = kw.pop("heartbeat_s", 2.0)
            return generate_rounds(self.engines, self.pool, kw, seqs, kv_high=kv, heartbeat_s=hb, **hooks)
        if cmd == "generate":
            on_sample = kw.pop("on_sample", None)      # --reward-stream: per-sample callback (streaming dispatch only)
            target = kw.pop("stream_target", None)     # a lower in-flight cap for this call (long forced prompts)
            if self.stream_seqs:
                from arcsolver.rl.sampler_dispatch import generate_stream
                seqs, kv = self.stream_settings()
                if target:
                    seqs = min(seqs, int(target))
                return generate_stream(self.engines, self.pool, kw, seqs, kv_high=kv, on_sample=on_sample)
            if self.dispatch_prompts:
                from arcsolver.rl.sampler_dispatch import generate_dynamic
                return generate_dynamic(self.engines, self.pool, kw, self.dispatch_prompts)
            n = len(self.engines)
            prompts = kw["prompts"]
            mts = kw.get("max_tokens")
            seeds = kw.get("seeds")
            ns = kw.get("n_per_prompt")
            parts = [list(range(e, len(prompts), n)) for e in range(n)]
            futs = []
            for e, idx in enumerate(parts):
                if not idx:
                    futs.append(None)
                    continue
                sub = {"prompts": [prompts[i] for i in idx], "params": kw["params"],
                       "max_tokens": [mts[i] for i in idx] if isinstance(mts, list) else mts,
                       "seeds": [seeds[i] for i in idx] if seeds else None,
                       "n_per_prompt": [ns[i] for i in idx] if ns is not None else None,
                       "topk_dir": kw.get("topk_dir"), "tag": f"{kw.get('tag', 'g')}e{e}"}
                futs.append(self.pool.submit(self.engines[e].call, "generate", **sub))
            outputs = [None] * len(prompts)
            gen, secs = 0, 0.0
            for idx, f in zip(parts, futs):
                if f is None:
                    continue
                r = f.result()
                for i, o in zip(idx, r["outputs"]):
                    outputs[i] = o
                gen += r["gen_tokens"]
                secs = max(secs, r["seconds"])
            return {"outputs": outputs, "gen_tokens": gen, "seconds": secs}
        if cmd == "score":
            return self.engines[0].call("score", **kw)
        if cmd in ("sleep", "wake", "free_host_cache"):
            rs = list(self.pool.map(lambda e: e.call(cmd, **kw), self.engines))
            return {"seconds": max(r["seconds"] for r in rs)}
        raise ValueError(cmd)


def engine_cli(args, model_path, log=print):
    """vllm_worker.py arguments shared by every sampler engine (the --policy-id is appended by the caller)."""
    cli = ["--role", "sampler", "--model", model_path, "--gpu-memory-utilization", args.engine_gpu_util,
           "--max-model-len", args.sampler_max_model_len, "--max-num-seqs", args.sampler_max_num_seqs,
           "--max-num-batched-tokens", args.sampler_max_num_batched_tokens, "--seed", args.seed,
           "--kv-cache-dtype", args.kv_cache_dtype, "--mtp-tokens", args.mtp_tokens,
           "--max-logprobs", max(20, args.sc_topk)]
    if args.sampler_enforce_eager:
        cli.append("--enforce-eager")
    if args.sampler_log_stats:
        cli.append("--log-stats")
    if args.timeshare:
        cli.append("--enable-sleep-mode")
    if args.sc_topk:
        cli.append("--skip-tokenizer-init")    # no detokenizing of k tokens per position
    inline_force = bool(args.train_force_inline or args.eval_force_inline)
    if args.engine_no_prefix_caching and not inline_force:
        cli.append("--no-prefix-caching")
    elif args.engine_no_prefix_caching:
        log("--train/eval-force-inline: engines keep prefix caching on (forced answers reuse their sample's KV)")
    if inline_force:                          # GDN states at every block boundary, also those reached in decode
        cli += ["--prefix-cache-retention-interval", "none"]
    if args.engine_block_size:
        cli += ["--block-size", args.engine_block_size]
    if args.engine_attention_backend:
        cli += ["--attention-backend", args.engine_attention_backend]
    return cli


def main(argv=None):
    import torch
    import torch.distributed as dist
    args = parse_args(argv)
    backend = "gloo" if args.dry_run else "nccl"          # a dry run builds the pool and eval set on CPU only
    dist.init_process_group(backend, timeout=datetime.timedelta(hours=args.dist_timeout_h))
    # Object/control traffic waits while rank 0 samples. Use CPU collectives so
    # waiting trainer ranks do not occupy the GPUs shared with vLLM. FSDP tensor
    # collectives still use the default NCCL group.
    control_group = (dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=args.dist_timeout_h))
                     if backend == "nccl" else None)
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = None
    if not args.dry_run:
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    ar.configure_arm(args.arm)
    from arcsolver.rl import fsdp as F
    if args.sdpa_mode == "causal":
        F.tf.install_sdpa_causal()
    import numpy as np
    from arcsolver.rl import loss as rl
    from arcsolver.rl import sc as rl_sc
    from arcsolver.rl.loss import reduce_stats

    def bcast(obj):
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=control_group)
        return box[0]

    # ---- rank 0: pool, eval set, resume state ------------------------------------------------------------
    log = lambda *a: None  # noqa: E731
    logf = None
    stok = cfg = tasks = eval_items = pool = counts = None
    resume = state = objective = pool_mf = pool_mf_rel = None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        for sub in ("rollouts", "evals", "checkpoints", "engines"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        logf = open(os.path.join(args.out, "log.txt"), "a", buffering=1)

        def log(*parts):  # noqa: F811
            line = " ".join(str(p) for p in parts)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[rl {stamp}] {line}", flush=True)
            logf.write(f"[{stamp}] {line}\n")

        _RUN_LOG["fn"] = log                       # module-level helpers (the rollout store) log here too
        log(f"args: {json.dumps(vars(args))}")
        log(f"distributed: {world} ranks, one vLLM engine per GPU at {args.engine_gpu_util} of its memory")
        from transformers import AutoTokenizer
        stok = AutoTokenizer.from_pretrained(args.student)
        if args.chat_template or args.reasoning_effort:
            log(f"chat template: {oc.set_chat_template(args.chat_template or None, args.reasoning_effort or None)}")
        if args.think_prefix:
            log(f"think prefix {args.think_prefix!r} -> ids {oc.set_think_prefix(stok, args.think_prefix)}")
        cfg = RT.reward_config(args)
        tasks = ot.TaskCache()
        eval_items = ot.build_eval_set(args, stok, tasks, log)
        pool, counts = RT.build_pool(args, stok, tasks, {i.task_id for i in eval_items if i.source == "bank200"}, log)
        from arcsolver.rl import identity as RI
        pool_mf = RI.pool_manifest(pool, counts, tasks)       # ordered (source, id, content, prompt, copies)
        pool_mf_rel = RI.write_pool_manifest(args.out, pool_mf)
        log(f"pool manifest {pool_mf['sha256'][:16]} ({len(pool)} prompts) -> {pool_mf_rel}")
        log(f"rewards: {json.dumps(cfg.__dict__)}")
        resume = latest_resumable(args.out, args.sampler_fp8, weights_ok=args.resume_weights_only)
        if resume and args.no_resume:
            raise SystemExit(f"--no-resume but {resume} exists")
        if resume:
            with open(os.path.join(resume, "state.json")) as f:
                state = json.load(f)
            log(f"resuming from {resume} (step {state['step']})")
        from arcsolver.rl import contract as rl_contract
        objective = rl_contract.objective_snapshot(args, cfg)
        with open(os.path.join(args.out, "launches.jsonl"), "a") as f:   # every launch's resolved objective
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:] if argv is None
                                else list(argv), "resume": resume, "objective": objective}, default=str) + "\n")
        if state is not None and (args.objective_contract or "objective" in state):
            diffs = rl_contract.check_resume(state, objective, args.allow_objective_change)
            if diffs:
                raise SystemExit(f"resume from {resume}: the objective differs from the checkpoint's "
                                 f"({len(diffs)}):\n  " + "\n  ".join(diffs)
                                 + "\n(--allow-objective-change KEY,... to accept named changes)")
            log(f"resume objective check passed ({objective['values'].get('objective_contract') or 'no contract'}; "
                f"allowed changes: {args.allow_objective_change or 'none'})")
        if state is not None and (args.objective_contract or "pool_manifest_sha256" in state):
            allow = {a.strip() for a in (args.allow_objective_change or "").split(",") if a.strip()}
            reasons = RI.check_pool_resume(args.out, state, pool_mf, allow_change=bool(args.new_pool_on_resume)
                                           or "pool" in allow, legacy_ok="legacy" in allow)
            if reasons:
                raise SystemExit(f"resume from {resume}: the prompt pool differs from the checkpoint's, so its saved "
                                 f"order indices would point at other tasks:\n  " + "\n  ".join(reasons)
                                 + "\n(--new-pool-on-resume for a fresh order, or --allow-objective-change pool)")
        if args.restore_only and (state is None or state['step'] != args.max_steps):
            raise SystemExit('--restore-only requires --max-steps equal to the latest completed checkpoint')
    if args.dry_run:
        if rank == 0:
            log("dry run: done")
        dist.barrier(group=control_group)
        dist.destroy_process_group()
        return
    resume = bcast(resume)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- engines (rank 0), then the trainer (all ranks) -------------------------------------------------
    from arcsolver.rl import workers
    engines = []
    sampler = None
    wb = None
    rollout_writer = {"w": None}                     # --rollout-write-async: the step's writer thread (rank 0)
    if rank == 0:
        import signal

        def on_term(signum, _frame):
            # torchrun SIGTERMs the survivors when a rank fails: take the engines (their own sessions) down too
            for e in engines:
                e.kill_now()
            os._exit(128 + signum)

        signal.signal(signal.SIGTERM, on_term)
    svc = {"proc": None}                             # this node's sampler service (--node-sampler-services)
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    node, nnodes = rank // local_ws, world // local_ws
    services = bool(args.node_sampler_services)
    writer_rank = (local == 0) if services else (rank == 0)       # who writes a sync copy (one per node)
    ckpt_writer = rank == 0                          # who writes a checkpoint's non-collective files (per node: below)
    if nnodes > 1 and not services:
        raise SystemExit(f"{nnodes} nodes: rank 0 cannot start engines on other nodes' GPUs; use "
                         "--node-sampler-services 1 --sampler-service-addrs ADDR0,ADDR1,...")
    if services and local == 0:
        import signal

        def on_term_svc(signum, _frame):
            if svc["proc"] is not None:
                try:
                    os.killpg(svc["proc"].pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            for e in engines:
                e.kill_now()
            os._exit(128 + signum)

        signal.signal(signal.SIGTERM, on_term_svc)
    try:
        from arcsolver.rl import identity as RI
        init_ckpt = args.init_checkpoint if (args.init_checkpoint and not resume) else ""
        start_dir = resume or init_ckpt                # --init-checkpoint: a fresh run starts from another run's checkpoint
        model_path = (os.path.join(start_dir, "model_fp8" if args.sampler_fp8 else "model") if start_dir
                      else args.sampler_model)
        serving = RI.serving_settings(args)
        cli_common = init_policy = init_sha = None
        if rank == 0:
            cli_common = engine_cli(args, model_path, log)
            t_pid = time.time()
            init_manifest = RI.dir_manifest(model_path)
            init_sha = init_manifest["sha256"]
            init_policy = RI.policy_id(init_sha, serving)
            cli_common += ["--policy-id", init_policy]
            log(f"initial policy {init_policy[:16]}: {model_path} ({len(init_manifest['files'])} files, manifest "
                f"{init_sha[:16]}; hashed in {time.time() - t_pid:.0f}s)")
        if services:
            addrs = [a.strip() for a in args.sampler_service_addrs.split(",") if a.strip()]
            if len(addrs) != nnodes:
                raise SystemExit(f"--sampler-service-addrs has {len(addrs)} addresses for {nnodes} nodes")
            key, init_policy, init_sha, cli_common = bcast((os.urandom(16).hex(), init_policy, init_sha, cli_common)
                                                           if rank == 0 else None)
            # every other node's staged sampler model must be byte-identical to rank 0's (same initial policy)
            mine = (RI.dir_manifest(model_path)["sha256"] if (local == 0 and rank != 0) else init_sha) if local == 0 else None
            seen = [None] * world
            dist.all_gather_object(seen, mine, group=control_group)
            bad = [r for r, v in enumerate(seen) if v is not None and v != init_sha]
            if bad:
                raise SystemExit(f"ranks {bad}: the local {model_path} differs from rank 0's (manifest {init_sha[:16]})")
            # OUT: one coherent shared directory (rank 0 writes the checkpoint's single-copy files) or node-local disks
            # at the same path (each node's local rank 0 writes its node's copy; the launcher unites the halves before
            # a resume). Never a FUSE object store mount: those can serve other nodes stale or truncated copies.
            from arcsolver.rl import fsdp_layout as L
            kind = L.fs_type(args.out)
            if kind.startswith("fuse"):
                raise SystemExit(f"--out {args.out} is on {kind}: multi-node checkpoints need node-local disk "
                                 "or a coherent shared filesystem")
            probe = os.path.join(args.out, bcast(f".node_probe_{os.urandom(6).hex()}" if rank == 0 else None))
            if rank == 0:
                with open(probe, "w") as f:
                    f.write("probe")
            dist.barrier(group=control_group)
            sees = [None] * world
            dist.all_gather_object(sees, os.path.exists(probe) if (local == 0 and rank != 0) else None,
                                   group=control_group)
            dist.barrier(group=control_group)
            if rank == 0:
                os.remove(probe)
            out_shared = any(bool(v) for v in sees if v is not None)
            ckpt_writer = (rank == 0) if out_shared else writer_rank
            if rank == 0:
                log(f"OUT {args.out} ({kind or '?'}): {'shared by' if out_shared else 'local to'} each node")
            if resume:                                # every node needs the whole checkpoint (restore first)
                miss = L.dcp_missing(resume, world) if local == 0 else []
                allmiss = [None] * world
                dist.all_gather_object(allmiss, miss, group=control_group)
                bad = {r // local_ws: m[:3] for r, m in enumerate(allmiss) if m}
                if bad:
                    raise SystemExit(f"resume {resume}: incomplete on node(s) {bad}; restore the checkpoint "
                                     "(every rank's shards + .metadata + model_fp8) to every node first")
            if local == 0:                            # this node's service: owns its local engines
                import subprocess
                edir = os.path.join(args.out, "engines")
                os.makedirs(edir, exist_ok=True)
                cli_path = os.path.join(edir, f"node{node}_cli.json")
                with open(cli_path, "w") as f:
                    json.dump([str(x) for x in cli_common], f)
                cmd = [sys.executable, "-m", "arcsolver.rl.node_service", "--node-rank", str(node),
                       "--bind", addrs[node], "--port", str(args.sampler_service_port + node), "--engines", str(local_ws),
                       "--run-name", args.run_name, "--log-dir", edir, "--vllm-python", args.vllm_python,
                       "--cli-json", cli_path, "--parent-pid", str(os.getpid())]
                if args.vllm_env_script:
                    cmd += ["--env-script", args.vllm_env_script]
                svc["log"] = open(os.path.join(edir, f"node{node}_service.log"), "ab")
                pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                pypath = os.pathsep.join(p for p in (pkg_root, os.environ.get("PYTHONPATH", "")) if p)
                svc["proc"] = subprocess.Popen(cmd, env=dict(os.environ, RL_SERVICE_KEY=key, PYTHONPATH=pypath),
                                               stdout=svc["log"], stderr=subprocess.STDOUT, start_new_session=True)
                print(f"[rank {rank}] node {node} sampler service started (pid {svc['proc'].pid}, "
                      f"{addrs[node]}:{args.sampler_service_port + node})", flush=True)
            if rank == 0:
                clients = [workers.NodeServiceClient(n, (addrs[n], args.sampler_service_port + n), bytes.fromhex(key))
                           for n in range(nnodes)]
                for n, c in enumerate(clients):
                    for g in range(local_ws):
                        engines.append(workers.RemoteEngineClient(f"n{n}s{g}", c, g))
        elif rank == 0:
            for g in range(world):
                engines.append(workers.EngineClient(
                    f"sampler{g}", args.vllm_python, cli_common, os.path.join(args.out, "engines", f"sampler{g}.log"),
                    f"/tmp/rl_{args.run_name}_sampler{g}.sock", env_script=args.vllm_env_script or None,
                    lock_path=None, extra_env={"CUDA_VISIBLE_DEVICES": str(g)}))
                time.sleep(5)
        if rank == 0:
            with cf.ThreadPoolExecutor(max_workers=len(engines)) as ex:
                infos = list(ex.map(lambda e: e.wait_ready(), engines))
            if any(i.get("policy") != init_policy for i in infos):
                raise RuntimeError(f"engines report policies {[str(i.get('policy'))[:16] for i in infos]}, "
                                   f"expected {init_policy[:16]}")
            log(f"{len(engines)} sampler engines ready ({nnodes} node(s)"
                f"{', node services' if services else ''}): " + json.dumps(
                    [{k: i.get(k) for k in ("load_seconds", "gpu_free_gb", "num_gpu_blocks")} for i in infos]))
            sampler = MultiEngine(engines, args.sampler_dispatch_prompts, args.sampler_stream_seqs,
                                  args.sampler_kv_high, args.sampler_control_file, policy=init_policy)
            if args.timeshare:
                log(f"engines asleep for the trainer load ({sampler.call('sleep', level=1)['seconds']:.0f}s)")
        dist.barrier(group=control_group)
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < args.min_free_gb_trainer:
            raise SystemExit(f"rank {rank}: only {free:.0f} GiB free for the trainer (needs {args.min_free_gb_trainer:.0f})")
        t0 = time.time()
        def startup_progress(message):
            print(f"[startup rank={rank} elapsed={time.time()-t0:.1f}s] {message}", flush=True)

        startup_progress(f"loading and sharding trainer base (attention: {args.trainer_attn})")
        model = F.load(args.student, dev, chunk=args.logprob_chunk, keep_gathered=bool(args.fsdp_keep_gathered),
                       freeze_embed=bool(args.freeze_embed), attn_impl=args.trainer_attn,
                       speed=args.trainer_speed)
        if args.activation_offload_mb > 0:
            F.set_activation_offload(model, args.activation_offload_mb << 20)
        startup_progress("trainer base loaded; creating optimizer")
        opt = F.optimizer(model, args.lr, args.adam_eps, beta2=args.adam_beta2, weight_decay=args.weight_decay)
        ref_model = None
        if args.kl_coef > 0:                           # frozen bf16 reference, sharded like the trainer
            startup_progress(f"loading the KL reference {args.ref_model}")
            ref_speed = ",".join(o for o in (args.trainer_speed or "").split(",") if o == "fused_head")
            ref_model = F.load(args.ref_model, dev, chunk=args.logprob_chunk, keep_gathered=True,
                               attn_impl=args.trainer_attn, speed=ref_speed, frozen=True)
        start_step = 0
        restored = None
        if resume:
            restored = F.load_checkpoint(model, opt, resume, weights_only=args.resume_weights_only,
                                         progress=startup_progress)
            start_step = bcast(state["step"] if rank == 0 else None)
        elif init_ckpt:                                # weights AND AdamW of the other run; this run's step count starts at 0
            startup_progress(f"--init-checkpoint {init_ckpt}")
            if F.load_checkpoint(model, opt, init_ckpt, progress=startup_progress) != "full":
                raise SystemExit(f"--init-checkpoint {init_ckpt}: no AdamW state (dcp_optim) to continue from")
            if rank == 0:
                log(f"started from --init-checkpoint {init_ckpt} (weights + AdamW; engines from its model_fp8)")
        rewarm_from = start_step if restored == "weights" else None
        if rank == 0 and rewarm_from is not None:
            log(f"resumed WEIGHTS ONLY at step {start_step}: fresh AdamW, lr re-warms over {args.rewarmup_steps} steps")
        if args.timeshare:
            startup_progress("offloading restored trainer and optimizer to CPU")
            F.offload(model, opt)
            if ref_model is not None:
                ref_model.to("cpu")
            # the DCP restore leaves garbage holding GPU memory; without this the startup wake can run out of memory
            # (same cleanup as after the per-step offload)
            import gc                                  # local, like the step loop's (gc is a local name in main)
            gc.collect()
            torch.cuda.empty_cache()
            startup_progress(f"trainer offloaded ({torch.cuda.memory_reserved(dev) / 2**30:.2f} GiB still reserved, "
                             f"{(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30:.1f} GiB used "
                             f"on the device); waiting for all ranks")
            dist.barrier(group=control_group)
            if rank == 0:
                log(f"trainer state on CPU; engines awake ({sampler.call('wake')['seconds']:.0f}s)")
                # level-1 sleep backed the weights up in pinned host memory (the full weights per engine), which torch
                # then caches; give it back to the OS
                try:
                    log(f"engine host cache released ({sampler.call('free_host_cache')['seconds']:.1f}s)")
                except RuntimeError as e:
                    log(f"engine host cache release failed (continuing): {str(e)[-300:]}")
            dist.barrier(group=control_group)
        order = None
        if rank == 0:
            log(f"trainer: FSDP2 over {world} GPUs, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params "
                f"(fp32 masters sharded), loaded in {time.time() - t0:.0f}s; AdamW lr {args.lr} (warmup "
                f"{args.warmup_steps}), eps {args.adam_eps}, no weight decay, clip {args.grad_clip}; "
                f"{torch.cuda.mem_get_info()[0] / 2**30:.0f} GiB free on GPU 0 after load")
            order = RT.make_order(args, pool, counts)
            if state is not None:
                restore_prompt_order(args, state, order, log, pool=pool, start_step=start_step,
                                     pool_changed=state.get("pool_manifest_sha256") not in (None, pool_mf["sha256"]))
            if args.pool_drop_solved:
                if args.pool_drop_solved_seed and start_step == 0:      # a fresh run seeded from earlier runs
                    for spec in parse_solved_seed(args.pool_drop_solved_seed):
                        sdir, supto = spec
                        seeded = solved_from_records(sdir, supto, {it.task_id: i for i, it in enumerate(pool)},
                                                     args.group_size)
                        n_new = order.mark_solved(seeded)
                        log(f"--pool-drop-solved-seed: {len(seeded)} tasks solved (all {args.group_size} correct) in "
                            f"{sdir} steps <= {supto} ({n_new} new): never drawn")
                if not order.solved and start_step > 0:   # a fresh order on a resume (or an older checkpoint)
                    seeded = solved_from_records(args.out, start_step, {it.task_id: i for i, it in enumerate(pool)},
                                                 args.group_size)
                    order.mark_solved(seeded)
                    log(f"--pool-drop-solved: {len(seeded)} tasks solved (all {args.group_size} correct) in the rollout "
                        f"records of steps 1-{start_step}: never drawn again")
                log(f"--pool-drop-solved: {len(order.solved)} of {len(pool)} pool tasks excluded")
            if getattr(args, "pool_focus_per_step", 0):
                setup_focus(args, order, pool, start_step, log)
            if args.restore_only and order.state_dict() != state['order']:
                raise RuntimeError('restored prompt order differs from checkpoint')
            writer = None
            if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
                import wandb
                id_path = os.path.join(args.out, "wandb_id.txt")
                wid = open(id_path).read().strip() if os.path.exists(id_path) else wandb.util.generate_id()
                with open(id_path, "w") as f:
                    f.write(wid)
                wb = wandb.init(project=args.wandb_project, name=args.run_name, id=wid, resume="allow",
                                config={**vars(args), "pool_size": len(pool), "eval_tasks": len(eval_items),
                                        "world": world}, dir=args.out)
        if args.restore_only:
            # Reached only after every rank loads model + Adam, offloads, and wakes
            # its sampler. No rollout, backward pass, or optimizer step is run.
            steps = []
            for entry in opt.state.values():
                if not {'step', 'exp_avg', 'exp_avg_sq'} <= entry.keys():
                    raise RuntimeError('restored Adam state is incomplete')
                value = entry['step']
                value = value.to_local() if hasattr(value, 'to_local') else value
                steps.append(float(value))
            if not steps or max(steps) != start_step or min(steps) < 0:
                raise RuntimeError('restored Adam step counters do not match checkpoint')
            report = dict(rank=rank, step=start_step, optimizer_states=len(steps),
                          optimizer_step_min=min(steps), optimizer_step_max=max(steps))
            reports = [None] * world
            dist.all_gather_object(reports, report, group=control_group)
            if rank == 0:
                from pathlib import Path
                checkpoint = Path(resume)
                proof = dict(kind='restore_only', checkpoint=str(checkpoint), step=start_step,
                             world_size=world, ranks=reports, prompt_order_restored=True,
                             checkpoint_state_sha256=hashlib.sha256((checkpoint/'state.json').read_bytes()).hexdigest(),
                             dcp_metadata_sha256=hashlib.sha256(((checkpoint/'dcp/.metadata') if (checkpoint/'dcp/.metadata').exists()
                                                                 else (checkpoint/'dcp_model/.metadata')).read_bytes()).hexdigest())
                ot.write_json(os.path.join(args.out, 'RESTORE_OK.json'), proof)
                log(f'RESTORE_ONLY_OK step={start_step} ranks={world}; no rollouts or optimizer update')
            dist.barrier(group=control_group)
            return
        if args.sync_versioned and writer_rank:        # a fresh export tree per launch (no stale versions / frozen file)
            shutil.rmtree(args.sync_dir, ignore_errors=True)
        writer = (F.SyncWriterFP8(args.sampler_model, args.sync_dir) if args.sampler_fp8
                  else F.SyncWriterBF16Stream(args.student, args.sync_dir))
        torch.cuda.reset_peak_memory_stats()

        def emit(metrics, step):
            with open(os.path.join(args.out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")
            if wb is not None:
                wb.log(metrics, step=step)

        run_eval = make_eval(args, sampler, stok, tasks, eval_items, emit, log) if rank == 0 else None
        from arcsolver.rl.hostmem import HostMemMonitor
        hostmem = HostMemMonitor() if rank == 0 else None
        aligner = rr.TokenAligner(stok) if rank == 0 else None
        if (rank == 0 and start_step == 0 and not args.no_eval_at_start and args.eval_every > 0
                and not os.path.isfile(os.path.join(args.out, "evals", "step_000000.COMPLETE"))):
            run_eval(0)                          # the base model's evaluation is kept across fresh restarts
        if rank == 0 and resume_eval_pending(args, start_step):
            log(f"finishing interrupted evaluation at checkpoint step {start_step}")
            run_eval(start_step)

        sync_dirs = {"cur": None}                       # the newest export (a version dir with --sync-versioned)

        def save_checkpoint(step):
            """Collective: DCP (fp32 masters + AdamW), the bf16 HF export, and with --sampler-fp8 the FP8 export of
            the last sync (what the engines restart from on resume, and what would be served)."""
            t0 = time.time()
            d = os.path.join(args.out, "checkpoints", f"step_{step:06d}")
            if ckpt_writer:                            # node-local OUT: the DCP dirs must exist on every node
                for sub in (("dcp_model", "dcp_optim") if args.dcp_split else ("dcp",)):
                    os.makedirs(os.path.join(d, sub), exist_ok=True)
            dist.barrier(group=control_group)
            if args.dcp_split:
                F.save_dcp_split(model, opt, d)
            else:
                F.save_dcp(model, opt, os.path.join(d, "dcp"))
            info = {"step": step, "base": args.student, "run": args.run_name, "trainer": "arcsolver.rl.train"}
            if bf16_export_due(args, step):
                F.export_model_dir(model, args.student, os.path.join(d, "model"), rank, info)
            if ckpt_writer and args.sampler_fp8:      # each node exports from its own (manifest-checked) sync copy
                F.export_fp8_dir(sync_dirs["cur"] or args.sync_dir, args.sampler_model, os.path.join(d, "model_fp8"),
                                 {**info, "format": "fp8 (fp8_quant)", "sampler_base": args.sampler_model})
            if rank == 0:
                # the step's rollout records exist before its checkpoint commits (pool refresh / resume need them)
                rollout_writer["w"] = join_rollout_writer(rollout_writer["w"])
                ot.write_json(os.path.join(d, "state.json"), {"step": step, "order": order.state_dict(),
                                                              "args": vars(args), "objective": objective,
                                                              "pool_manifest_sha256": pool_mf["sha256"],
                                                              "policy": getattr(sampler, "committed_policy", None),
                                                              "pool_manifest_file": pool_mf_rel})
            dist.barrier(group=control_group)          # every node's shards and exports are on disk
            if ckpt_writer:
                open(os.path.join(d, "COMPLETE"), "w").close()
                done = ot.ckpt_dirs(args.out)
                for old in done[:-max(1, args.keep_optimizer_states)]:
                    for sub in ("dcp", "dcp_model", "dcp_optim"):
                        shutil.rmtree(os.path.join(old, sub), ignore_errors=True)
                if args.keep_model_dirs > 0:
                    for old in done[:-args.keep_model_dirs]:
                        shutil.rmtree(os.path.join(old, "model"), ignore_errors=True)
                        shutil.rmtree(os.path.join(old, "model_fp8"), ignore_errors=True)
            if rank == 0:
                log(f"checkpoint {d} ({time.time() - t0:.0f}s)")
                emit({"time/save_s": time.time() - t0}, step)
            dist.barrier(group=control_group)

        probe = None
        for step in range(start_step + 1, args.max_steps + 1):
            t_step = time.time()
            msg = None
            if rank == 0:
                out = rollout_step(args, step, order, pool, tasks, stok, sampler, cfg, aligner)
                rollouts, kept, gaugs, prompts, res, times, aug_retries = out
                if args.pool_order in ("frontier", "frontier_mix"):   # this step's outcomes steer the next draws
                    order.update(frontier_results(rollouts))
                if args.pool_drop_solved:                  # all-correct groups: their tasks are not drawn again
                    times["solved_new"] = order.mark_solved(
                        [pi for pi, c, n, _t in frontier_results(rollouts) if n == args.group_size and c == n])
                hostmem.mark("after_rollout")
                # Pool refresh needs these records even if the process stops
                # just after committing its model/optimizer checkpoint.
                if args.save_rollouts and not args.rollout_write_async:
                    save_rollout_record(args.out, step, rollouts)
                    save_rollout_tokens(args.out, step, rollouts)
                work = [j for j, r in enumerate(kept) if r["seq_adv"] != 0.0 or r["dense"] or args.train_all_rollouts]
                norms = rl.normalisers([len(r["completion"]) for r in kept], args.loss_agg)
                # the control file can lower the cap per step, never raise it (a fallback to 1 always holds)
                try:
                    mb_max = min(args.minibatch_max,
                                 int(read_control(args.sampler_control_file).get("minibatch_max", args.minibatch_max)))
                except (TypeError, ValueError):
                    mb_max = args.minibatch_max
                if args.balance == "lpt":
                    import functools
                    from arcsolver.rl import trainer_speed as TS
                    q = args.balance_quad_tokens
                    assign_fn = functools.partial(TS.assign_balanced, cost=lambda n: n + n * n / q)
                else:
                    assign_fn = F.assign_equal
                plan = plan_minibatches(kept, work, world, norms, args.loss_agg, int(mb_max),
                                        args.minibatch_min_groups, assign_fn,
                                        length_only=times.get("allc_groups", set()),
                                        local_ws=local_ws if args.node_sampler_services else None)
                if probe is None and work:
                    r0 = kept[work[0]]
                    probe = (list(r0["prompt_ids"]), list(r0["completion"][:256]))
                do_probe = bool(work) and args.probe_every > 0 and step % args.probe_every == 0 and probe is not None
                shards = []
                for rk in range(world):
                    mbs = []
                    for mb in plan:
                        items = []
                        for j in mb["assign"][rk]:
                            r = kept[j]
                            items.append({"prompt_ids": r["prompt_ids"], "completion": r["completion"],
                                          "sampler_lp": r["sampler_lp"], "seq_adv": r["seq_adv"], "dense": r["dense"],
                                          "norm": mb["norm"][j], "topk_path": r.get("topk_path")})
                        mbs.append({"items": items, "n_micro": mb["n_micro"]})
                    shards.append({"minibatches": mbs, "probe": probe if do_probe else None,
                                   "stepped": bool(work)})
                times["minibatches"] = len(plan)
                msg = shards
                before = None
                if do_probe:
                    pseq = probe[0] + probe[1]
                    before = sampler.call("score", seqs=[pseq], starts=[len(probe[0])])["logprobs"][0]
                if args.timeshare:
                    t_sl = time.time()
                    sampler.call("sleep", level=args.engine_sleep_level)
                    times["sleep"] = time.time() - t_sl
            box = [None]
            dist.scatter_object_list(box, msg, src=0, group=control_group)
            mine = box[0]
            if rank == 0 and args.save_rollouts and args.rollout_write_async:
                # after the scatter, so the writer's GIL time does not delay the ranks' shards
                rollout_writer["w"] = start_rollout_writer(args.out, step, rollouts)
            t_rl = 0.0
            if args.timeshare:
                t_rl = time.time()
                F.reload(model, opt, dev, optimizer_state=not args.defer_optimizer_reload)
                if ref_model is not None:
                    ref_model.to(dev)
                t_rl = time.time() - t_rl
            # ---- train (all ranks) -------------------------------------------------------------------------
            t0 = time.time()
            pg_acc, adv_acc = {}, {}
            pg_acc0 = {}                               # first minibatch only: the sampler mismatch without own updates
            loss_total, n_fwd, tokens = 0.0, 0, 0
            if mine["stepped"]:
                grad_norms, n_mb = [], len(mine["minibatches"])
                lr = step_lr(args, step, rewarm_from)
                if args.pg_loss == "ppo" and args.ppo_old_logp == "trainer":
                    # the step's old log-probs under the weights that start it (before minibatch 1): the PPO ratio's
                    # denominator for every minibatch, so the clip sees only the policy's own movement within the step
                    t_old = time.time()
                    all_items = [it for mb in mine["minibatches"] for it in mb["items"]]
                    if not getattr(model, "keep_gathered", True):     # every forward a collective: equal counts
                        counts = [None] * world
                        dist.all_gather_object(counts, len(all_items), group=control_group)
                        all_items = all_items + [None] * (max(counts) - len(all_items))
                    for it, lp_old in zip(all_items, F.trainer_logprobs(model, dev, all_items)):
                        if it is not None:
                            it["_old_lp"] = lp_old
                            slp = torch.tensor(it["sampler_lp"], dtype=torch.float32, device=dev)
                            reduce_stats(pg_acc, {"smis_abs_sum": float((slp - lp_old).abs().sum()),
                                                  "smis_n": float(lp_old.numel())})
                    t_old = time.time() - t_old
                for mi, mb in enumerate(mine["minibatches"]):
                    F.begin_step(model)
                    if ref_model is not None:          # collective: same count on every rank, whatever its rollouts
                        F.gather_frozen(ref_model, dev)
                    # lpt: this rank's own count; only its last backward reduce-scatters (the gradient accumulates
                    # unsharded in fp32 until then), so ranks need not run the same number of backwards
                    n_run = max(1, len(mb["items"])) if args.balance == "lpt" else mb["n_micro"]
                    for j in range(n_run):
                        if args.balance == "lpt":
                            model.set_requires_gradient_sync(j == n_run - 1)
                        if j == n_run - 1:
                            F.last_micro(model)
                        if j >= len(mb["items"]):
                            F.dummy_backward(model, dev)
                            continue
                        it = mb["items"][j]
                        T = len(it["completion"])
                        ids = torch.tensor([list(it["prompt_ids"]) + list(it["completion"])], device=dev)
                        mask = torch.ones(T, dtype=torch.float32, device=dev)
                        dense = rl.dense_tensor(T, it["dense"], device=dev)
                        if args.pg_loss == "tis_sc":
                            ref_lp = None
                            if ref_model is not None:          # before the policy forward: no saved activations yet
                                with torch.no_grad():
                                    ref_lp = ref_model(ids, len(it["prompt_ids"])).float()
                            if args.sc_topk:
                                z = np.load(it["topk_path"])
                                q_y = torch.from_numpy(z["lp_y"]).to(dev)
                                h_ids = torch.from_numpy(z["ids"].astype(np.int64)).to(dev)
                                q_h = torch.from_numpy(z["lp"]).to(dev)
                                if "y" in z.files and not np.array_equal(z["y"], np.asarray(it["completion"])):
                                    raise RuntimeError(f"top-k file {it['topk_path']} is not this rollout's")
                                logp, lp_h = model(ids, len(it["prompt_ids"]), head_ids=h_ids)
                                valid = rl_sc.head_valid(h_ids)
                            else:
                                q_y = torch.tensor(it["sampler_lp"], dtype=torch.float32, device=dev)
                                logp = model(ids, len(it["prompt_ids"]))
                                lp_h = logp.new_zeros((T, 0))
                                q_h = lp_h.detach()
                                valid = torch.zeros((T, 0), dtype=torch.bool, device=dev)
                            adv = rl.assemble_advantage(it["seq_adv"], dense, args.adv_clip)
                            loss, stats = rl_sc.sc_tis_loss(logp, lp_h, q_y, q_h, valid, adv, mask, it["norm"],
                                                            tis_cap=args.tis_cap, score_centering=bool(args.sc_topk))
                            if ref_lp is not None:
                                kl_loss, kl_stats = rl.kl_k3(logp, ref_lp, mask, it["norm"], args.kl_coef)
                                loss = loss + kl_loss
                                stats = {**stats, **kl_stats}
                                del ref_lp, kl_loss
                        else:
                            ref_lp = None
                            if ref_model is not None:          # before the policy forward: no saved activations yet
                                with torch.no_grad():
                                    ref_lp = ref_model(ids, len(it["prompt_ids"])).float()
                            logp = model(ids, len(it["prompt_ids"]))
                            adv = rl.assemble_advantage(it["seq_adv"], dense, args.adv_clip)
                            old_lp = it.get("_old_lp")
                            if old_lp is None:
                                old_lp = torch.tensor(it["sampler_lp"], dtype=torch.float32, device=dev)
                            loss, stats = rl.pg_loss(logp, old_lp, adv, mask, it["norm"], clip_eps=args.clip_eps,
                                                     clip_eps_high=args.clip_eps_high)
                            if ref_lp is not None:
                                kl_loss, kl_stats = rl.kl_k3(logp, ref_lp, mask, it["norm"], args.kl_coef)
                                loss = loss + kl_loss
                                stats = {**stats, **kl_stats}
                                del ref_lp, kl_loss
                        (loss * world).backward()
                        loss_total += float(loss)
                        reduce_stats(pg_acc, stats)
                        if mi == 0:
                            reduce_stats(pg_acc0, stats)
                        rl.reduce_adv_stats(adv_acc, rl.adv_stats(adv.detach(), mask))
                        n_fwd += 1
                        tokens += ids.shape[1]
                        del logp, loss, stats, adv, ids
                    model.set_requires_gradient_sync(True)
                    if ref_model is not None:
                        F.reshard_frozen(ref_model)
                    if mi == n_mb - 1 and getattr(model, "offload", None) is not None:
                        model.offload.release()        # host RAM for checkpointing and the engines
                    grad_norms.append(F.grad_norm_clip(model, args.grad_clip))
                    for g in opt.param_groups:
                        g["lr"] = lr
                    if args.defer_optimizer_reload:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        F.reload_optimizer(opt, dev)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    if args.defer_optimizer_reload and mi < n_mb - 1:
                        F.offload_optimizer(opt)       # Adam back on the CPU through the next minibatch's backward
                F.release_host_cache()                 # the Adam moments' pinned CPU copies, now on the GPU
                grad_norm = sum(grad_norms) / len(grad_norms)
                grad_norm_max = max(grad_norms)
                grad_clipped = sum(args.grad_clip > 0 and gn > args.grad_clip for gn in grad_norms)
            else:
                grad_norm, lr, grad_norm_max, grad_clipped = 0.0, 0.0, 0.0, 0
            torch.cuda.synchronize()
            t_train = time.time() - t0
            gathered = [None] * world if rank == 0 else None
            dist.gather_object({"pg": pg_acc, "pg0": pg_acc0, "adv": adv_acc, "loss": loss_total, "n_fwd": n_fwd,
                                "tokens": tokens,
                                "mem": torch.cuda.max_memory_reserved(dev) / 2**30, "t_train": t_train}, gathered, dst=0,
                               group=control_group)
            if rank == 0:
                hostmem.mark("after_train")
            # ---- sync (collective gather; rank 0 writes and reloads the engines) -----------------------------
            pm, t_sync, wstat, rl_s = {}, 0.0, {"write_s": 0.0}, 0.0
            do_save = args.save_every > 0 and (step % args.save_every == 0 or step == args.max_steps
                                              or (args.save_first_step and step == 1))
            saved_now = False
            if mine["stepped"] or args.timeshare:      # time-sharing engines slept (level 2): always refill them
                t0 = time.time()
                # export of policy v<step>: one copy per node (each node's local rank 0 quantizes the same
                # gathered tensors) into a fresh version dir; every copy's full manifest must agree before any reload
                vdir = os.path.join(args.sync_dir, f"v{step:06d}") if args.sync_versioned else args.sync_dir
                wst = writer.write_stream(model, rank, writer=writer_rank, dest=vdir if args.sync_versioned else None)
                if rank == 0 and wst:
                    wstat = wst
                t_h = time.time()
                my_sha = RI.dir_manifest(vdir)["sha256"] if writer_rank else None
                shas = [None] * world
                dist.all_gather_object(shas, my_sha, group=control_group)
                new_policy = None
                if rank == 0:
                    got = {r: v for r, v in enumerate(shas) if v is not None}
                    if len(set(got.values())) != 1:
                        raise RuntimeError(f"sync v{step}: the node copies of the export differ "
                                           f"({ {r: v[:12] for r, v in got.items()} }): no engine reloads")
                    new_policy = RI.policy_id(got[0], serving)
                    wstat = dict(wstat, hash_s=time.time() - t_h, policy=new_policy, writers=len(got))
                sync_dirs["cur"] = vdir
                tr = None
                if mine["probe"]:
                    t1 = time.time()
                    with torch.no_grad():
                        pids = torch.tensor([mine["probe"][0] + mine["probe"][1]], device=dev)
                        tr = model(pids, len(mine["probe"][0])).tolist()
                if args.timeshare:
                    if do_save:
                        save_checkpoint(step)                  # collective; needs the trainer on the GPU
                        saved_now = True
                    F.offload(model, opt)
                    if ref_model is not None:
                        ref_model.to("cpu")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    free_b, total_b = torch.cuda.mem_get_info()
                    resid = [None] * world      # what each trainer rank still holds when the engines wake
                    dist.all_gather_object(resid, ((total_b - free_b) / 2**30, torch.cuda.memory_reserved() / 2**30),
                                           group=control_group)
                    if rank == 0:
                        times["resid"] = resid
                    dist.barrier(group=control_group)
                    if rank == 0:
                        sampler.call("wake", tags=["weights"])
                        rl_s = sampler.call("reload", weights_path=vdir, policy=new_policy)["seconds"]
                        sampler.call("wake", tags=["kv_cache"])
                        if args.engine_sleep_level == 1:
                            try:                       # the level-1 weight backup, now restored, stays cached pinned
                                sampler.call("free_host_cache")
                            except RuntimeError as e:
                                log(f"engine host cache release failed (continuing): {str(e)[-200:]}")
                    dist.barrier(group=control_group)
                elif rank == 0:
                    rl_s = sampler.call("reload", weights_path=vdir, policy=new_policy)["seconds"]
                if args.sync_versioned:                # every engine acknowledged v<step>: the older versions can go
                    if not args.timeshare:
                        dist.barrier(group=control_group)
                    if writer_rank:
                        import glob
                        if args.sc_topk:                   # this node's engines' top-k files of the step (all trained)
                            shutil.rmtree(os.path.join(args.topk_dir, f"step_{step:06d}"), ignore_errors=True)
                        for old in glob.glob(os.path.join(args.sync_dir, "v[0-9]*")):
                            if os.path.abspath(old) != os.path.abspath(vdir):
                                shutil.rmtree(old, ignore_errors=True)
                t_sync = time.time() - t0
                if mine["probe"]:
                    if rank == 0:
                        pseq = mine["probe"][0] + mine["probe"][1]
                        after = sampler.call("score", seqs=[pseq], starts=[len(mine["probe"][0])])["logprobs"][0]
                        mean_abs = lambda a, b: sum(abs(x - y) for x, y in zip(a, b)) / len(a)  # noqa: E731
                        pm = {"sync/probe_trainer_vs_sampler": mean_abs(tr, after),
                              "sync/probe_trainer_vs_stale_sampler": mean_abs(tr, before),
                              "sync/probe_sampler_delta": mean_abs(after, before), "time/probe_s": time.time() - t1}
            # ---- log, eval, save (rank 0; save is collective) --------------------------------------------------
            if rank == 0:
                pg_all, adv_all, pg0_all = {}, {}, {}
                for gth in gathered:
                    reduce_stats(pg_all, gth["pg"])          # sums add, *_max keys take the max
                    reduce_stats(pg0_all, gth["pg0"])
                    if gth["adv"]:
                        rl.reduce_adv_stats(adv_all, gth["adv"])
                for r in kept:
                    if r["seq_adv"] == 0.0 and not r["dense"]:
                        T = len(r["completion"])
                        rl.reduce_adv_stats(adv_all, rl.adv_stats(torch.zeros(T), torch.ones(T)))
                n_fwd_all = sum(g_["n_fwd"] for g_ in gathered)
                tok_all = sum(g_["tokens"] for g_ in gathered)
                metrics = {**rr.step_metrics(rollouts), **rl.pg_metrics(pg_all), **rl.adv_metrics(adv_all), **pm,
                           **rl.kl_metrics(pg_all),
                           **({"pool/solved_new": times.get("solved_new", 0), "pool/solved_total": len(order.solved),
                               "pool/live": len(pool) - len(order.solved)} if args.pool_drop_solved else {}),
                           **RT.slice_metrics(rollouts), **(rl_sc.sc_metrics(pg_all) if args.pg_loss == "tis_sc" else {})}
                if pg_all.get("smis_n"):                     # --ppo-old-logp trainer: the FP8 sampler vs the step's bf16 start
                    metrics["ppo/sampler_vs_trainer_abs"] = pg_all["smis_abs_sum"] / pg_all["smis_n"]
                    metrics.update({"pg_mb0/" + k.split("/", 1)[1]: v for k, v in rl.pg_metrics(pg0_all).items()})
                if args.pg_loss == "tis_sc":                 # minibatches 2..k add the policy's own drift
                    metrics.update({"sc_mb0/" + k.split("/", 1)[1]: v for k, v in rl_sc.sc_metrics(pg0_all).items()})
                if args.pg_loss == "tis_sc":
                    metrics["pg/clip_frac"] = metrics["sc/tis_clip_frac"]       # the log line's "clip" = TIS clipping
                metrics.update({"time/trainer_reload_s": t_rl, "time/engine_sleep_s": times.get("sleep", 0.0)})
                if args.reward_stream:
                    metrics["time/reward_streamed_frac"] = times.get("reward_streamed", 0) / max(1, len(rollouts))
                if times.get("dyn") is not None:               # dynamic sampling
                    dd = (res.get("dispatch") or {}).get("dyn") or {}
                    metrics.update({"dyn/rounds": dd.get("rounds", 0), "dyn/rounds_included": dd.get("rounds_included", 0),
                                    "dyn/prompts_dispatched": dd.get("prompts_dispatched", 0),
                                    "dyn/prompts_included": dd.get("prompts_included", 0),
                                    "dyn/informative_groups": times["dyn"]["judged_informative"],
                                    "dyn/samples_aborted": dd.get("samples_aborted", 0),
                                    "dyn/stopped": int(bool(dd.get("stopped"))), "dyn/holds": dd.get("holds", 0)})
                if times.get("resid"):
                    metrics.update({"mem/device_used_at_wake_gb_max": max(r[0] for r in times["resid"]),
                                    "mem/trainer_reserved_at_wake_gb_max": max(r[1] for r in times["resid"])})
                hostmem.mark("after_sync")
                metrics.update(hostmem.take())
                if args.sc_topk:
                    shutil.rmtree(os.path.join(args.topk_dir, f"step_{step:06d}"), ignore_errors=True)
                metrics.update({
                    "train/loss": sum(g_["loss"] for g_ in gathered), "train/grad_norm": grad_norm, "train/lr": lr,
                    "train/loss_per_update": sum(g_["loss"] for g_ in gathered) / max(1, times.get("minibatches", 1)),
                    "train/grad_norm_max": grad_norm_max, "train/grad_clipped_updates": grad_clipped,
                    "train/stepped": int(mine["stepped"]), "train/rollouts_fwd": n_fwd_all,
                    "train/rollouts_zero_adv": len(kept) - len(work), "train/tokens": tok_all,
                    "train/tok_per_s": tok_all / max(t_train, 1e-9),
                    "train/micro_per_rank": sum((max(1, len(m_["items"])) if args.balance == "lpt" else m_["n_micro"])
                                                for m_ in mine["minibatches"]),
                    "train/minibatches": times.get("minibatches", 1),
                    "rollout/prompts": len(prompts), "rollout/text_mismatch": times["n_mism"],
                    **({"rollout/len_share_groups": times["len_share_groups"]} if args.len_share_lambda else {}),
                    **({"train/trunc_masked": times["trunc_masked"][0],
                        "train/trunc_masked_tokens": times["trunc_masked"][1]} if args.mask_truncated else {}),
                    **({"train/trunc_scaled": times["trunc_scaled"][0],
                        "train/trunc_scaled_tokens": times["trunc_scaled"][1]} if args.trunc_neg_scale != 1.0 else {}),
                    **({**times["forced"], "time/force_s": times["force"]} if times.get("forced") else {}),
                    **adv_weight_metrics(rollouts, kept, args.group_size),
                    "rollout/entropy_est": sampled_nll_mean(rollouts),
                    **({"rollout/len_allcorrect_groups": times["len_allcorrect_groups"]}
                       if args.len_allcorrect_beta else {}),
                    **({"rollout/measured_success_of_tasks": times["measured"]} if times["measured"] is not None else {}),
                    "aug/identity_d8_frac": sum(a.d8 == "id" for a in gaugs) / len(gaugs),
                    "aug/identity_frac": sum(a.is_identity() for a in gaugs) / len(gaugs), "aug/retries": aug_retries,
                    "aug/split_test_groups": len({r["group"] for r in rollouts if r.get("test_idx") is not None}),
                    **prefix_metrics(rollouts),
                    **(times.get("lenpair") or {}),
                    **(times.get("overlong") or {}),
                    "aug/prompt_tokens_mean": sum(map(len, prompts)) / len(prompts),
                    "aug/prompt_tokens_max": max(map(len, prompts)),
                    "time/aug_s": times["aug"], "time/sample_s": times["sample"], "time/reward_s": times["reward"],
                    "time/train_s": t_train, "time/sync_s": t_sync, "time/sync_write_s": wstat["write_s"],
                    "time/sync_hash_s": wstat.get("hash_s", 0.0),
                    "time/sync_reload_s": rl_s, "time/sample_tok_per_s": res["gen_tokens"] / max(res["seconds"], 1e-9),
                    "mem/trainer_max_reserved_gb": max(g_["mem"] for g_ in gathered), "data/epoch": order.epoch})
                if args.pool_order == "frontier":
                    metrics.update({**{f"frontier/n_{s}": n for s, n in order.counts().items()},
                                    "frontier/epoch": order.epoch})
                elif args.pool_order == "frontier_mix":
                    metrics.update(order.metrics())
                metrics["time/step_s"] = time.time() - t_step
                if res.get('dispatch'):
                    scheduling = res['dispatch']
                    if scheduling.get('target_seqs'):
                        metrics.update({'sampler/stream_target_seqs': scheduling['target_seqs'],
                                        'sampler/kv_high': scheduling.get('kv_high', 0.0),
                                        'sampler/kv_usage_max': max(e.get('kv_usage_max') or 0.0
                                                                    for e in scheduling['engines'])})
                    metrics.update({'sampler/dispatch_prompts': scheduling['chunk_prompts'],
                                    'sampler/dispatch_tail_s': scheduling['tail_s'],
                                    'sampler/engine_active_s_max': max(e['active_s'] for e in scheduling['engines']),
                                    'sampler/engine_active_s_min': min(e['active_s'] for e in scheduling['engines'])})
                    fpt = sum(e.get('forced_prompt_tokens') or 0 for e in scheduling['engines'])
                    if fpt:                                  # --train-force-inline: share of forced prompts from cache
                        metrics.update({'forced/inline_n': sum(e.get('forced') or 0 for e in scheduling['engines']),
                                        'forced/inline_cached_frac': sum(e.get('forced_cached_tokens') or 0
                                                                         for e in scheduling['engines']) / fpt,
                                        'forced/inline_gen_tokens': sum(e.get('forced_gen_tokens') or 0
                                                                        for e in scheduling['engines'])})
                emit(metrics, step)
                log(f"step {step}: aug [{' '.join(a.d8 + '/' + a.meta()['perm'] for a in gaugs)}]")
                log(f"step {step}: slices {RT.slice_line(metrics)}")
                if times.get("dyn") is not None:
                    dd = (res.get("dispatch") or {}).get("dyn") or {}
                    log(f"step {step}: dyn rounds {dd.get('rounds_included')}/{dd.get('rounds')} prompts "
                        f"{dd.get('prompts_included')}/{dd.get('prompts_dispatched')} informative "
                        f"{times['dyn']['judged_informative']} (per round {times['dyn']['rounds']}) aborted samples "
                        f"{dd.get('samples_aborted')} stopped {dd.get('stopped')} round times {dd.get('round_times')}")
                log(f"step {step}: adv weight by correct-of-{args.group_size} k1 "
                    f"{100 * metrics['adv/weight_share_k1']:.0f}% k2-3 {100 * metrics['adv/weight_share_k2_3']:.0f}% "
                    f"k4-5 {100 * metrics['adv/weight_share_k4_5']:.0f}% k6-7 {100 * metrics['adv/weight_share_k6_7']:.0f}% "
                    f"other {100 * metrics['adv/weight_share_other']:.0f}% | truncated "
                    f"{100 * metrics['adv/weight_share_truncated']:.0f}% | ess {metrics['adv/ess_rollouts']:.1f} of "
                    f"{len(kept)} | net {metrics['adv/net_sum']:+.2f} of {metrics['adv/sum_abs']:.1f} | entropy_est "
                    f"{metrics['rollout/entropy_est']:.3f} trunc {100 * metrics['rollout/truncated_rate']:.0f}%")
                if "lenpair/groups" in metrics:
                    log(f"step {step}: length pairs: {metrics['lenpair/groups']} groups shaped "
                        f"({metrics['lenpair/allright_groups']} all-right kept for it) | mean |adjust| "
                        f"{metrics.get('lenpair/adj_abs_mean', 0):.3f} max {metrics.get('lenpair/adj_abs_max', 0):.3f} | "
                        f"right-length range {metrics.get('lenpair/right_len_range_mean', 0):.0f}")
                if "prefix/groups" in metrics:
                    log(f"step {step}: prefix groups {metrics['prefix/groups']} ({metrics['prefix/from_zero']} from 0-right "
                        f"groups) outcome {metrics['prefix/outcome']:.3f} (source right "
                        f"{metrics.get('prefix/outcome_src_right', float('nan')):.3f}, source wrong "
                        f"{metrics.get('prefix/outcome_src_wrong', float('nan')):.3f}) with signal "
                        f"{metrics['prefix/informative']}/{metrics['prefix/groups']} cut mean "
                        f"{metrics['prefix/cut_mean']:.0f} | fresh outcome {metrics['reward/outcome_fresh']:.3f}")
                if times.get("forced"):
                    fs_ = times["forced"]
                    log(f"step {step}: forced answers {fs_['forced/tried']} tried, {fs_['forced/correct']} right "
                        f"({100 * fs_['forced/correct_rate']:.1f}%), {fs_['forced/credited']} credited, "
                        f"{fs_['forced/rescued_groups']} groups rescued, {fs_['forced/cut_at_budget']} cut at the "
                        f"budget, {fs_['forced/skipped_no_room']} without room | {times['force']:.0f}s")
                log(f"step {step}: R {metrics['reward/total']:.3f} (outcome {metrics['reward/outcome']:.3f} "
                    f"loop_tok {metrics['reward/loop_tok_sum']:.0f}) em {100 * metrics['rollout/em_rate']:.1f}% "
                    f"(em_frac {100 * metrics['rollout/em_frac']:.1f}% vs measured unaugmented "
                    f"{100 * metrics.get('rollout/measured_success_of_tasks', float('nan')):.1f}%) "
                    f"| len {metrics['rollout/len_mean']:.0f} close {100 * metrics['rollout/close_rate']:.0f}% "
                    f"loop {100 * metrics['rollout/loop_rate']:.0f}% | groups dropped "
                    f"{100 * metrics['group/dropped_frac']:.0f}% fwd {n_fwd_all} skip {len(kept) - len(work)} | loss "
                    f"{metrics['train/loss']:.4f} gnorm {grad_norm:.3f} clip {100 * metrics['pg/clip_frac']:.2f}% mismatch "
                    f"{metrics['pg/mismatch_abs']:.4f} | sample {times['sample']:.0f}s reward {times['reward']:.0f}s train "
                    f"{t_train:.0f}s sync {t_sync:.1f}s | peak {metrics['mem/trainer_max_reserved_gb']:.0f} GB"
                    + (f" | probe {pm['sync/probe_trainer_vs_sampler']:.4f} (stale "
                       f"{pm['sync/probe_trainer_vs_stale_sampler']:.4f})" if pm else ""))
                rollout_writer["w"] = join_rollout_writer(rollout_writer["w"])
                del rollouts, kept
            if rank == 0 and eval_due(args, step):
                run_eval(step)
            if do_save and not saved_now:
                save_checkpoint(step)
            if rank == 0 and getattr(args, "rollout_store_dir", "") and (do_save or saved_now):
                # the previous step's samples go once this step's checkpoint exists (an external copy of the checkpoint
                # may lag a little, so a restore can still need the previous step's samples)
                RolloutStore.cleanup(args.rollout_store_dir, step - 1, log)
        dist.barrier(group=control_group)
        if rank == 0:
            # COMPLETE covers the weights; TRAIN_DONE also covers the window's
            # scheduled evaluation and rollout writes.
            final_checkpoint = os.path.join(args.out, "checkpoints", f"step_{args.max_steps:06d}")
            if os.path.isfile(os.path.join(final_checkpoint, "COMPLETE")):
                ot.write_json(os.path.join(final_checkpoint, "TRAIN_DONE"), {"step": args.max_steps})
            log("done")
    except BaseException as exc:
        # Fail fast: a rank that fails (e.g. out of memory) would otherwise sit in destroy_process_group's NCCL
        # teardown while its peers wait out the process-group timeout in a collective. Report, kill this rank's
        # engines and exit at once; torchrun then stops the other ranks.
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        import traceback
        print(f"[rank {rank}] FATAL, exiting without distributed teardown:\n"
              + "".join(traceback.format_exception(exc)), flush=True)
        for e in engines:
            e.kill_now()
        if svc["proc"] is not None:
            import signal
            for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 0)):
                try:
                    os.killpg(svc["proc"].pid, sig)
                    svc["proc"].wait(timeout=grace) if grace else None
                    break
                except (ProcessLookupError, PermissionError):
                    break
                except Exception:  # noqa: BLE001 - still alive after the grace period
                    continue
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if rank == 0 and rollout_writer.get("w") is not None:
            rollout_writer["w"][0].join()            # a normal exit keeps the last step's records
        for e in engines:
            try:
                e.close()
            except Exception as ex:  # noqa: BLE001
                print(f"closing {e.name}: {ex}", flush=True)
        if wb is not None:
            wb.finish()
        if writer_rank:
            shutil.rmtree(args.sync_dir, ignore_errors=True)
        if svc["proc"] is not None:                    # rank 0's close() asked every service to shut down
            import signal
            try:
                svc["proc"].wait(timeout=180)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(svc["proc"].pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def step_lr(args, step, rewarm_from=None) -> float:
    """The policy LR of a step: --lr after --warmup-steps of linear warmup (and a re-warmup after a weights-only
    resume), then with --lr-ramp-to linearly from --lr to --lr-ramp-to over the --lr-ramp-steps steps after
    --lr-ramp-from (step from + k gets k / steps of the way)."""
    lr = args.lr * (min(1.0, step / args.warmup_steps) if args.warmup_steps > 0 else 1.0)
    if rewarm_from is not None and args.rewarmup_steps > 0:
        lr *= min(1.0, (step - rewarm_from) / args.rewarmup_steps)
    to = getattr(args, "lr_ramp_to", 0.0) or 0.0
    if to > 0:
        frac = min(1.0, max(0.0, (step - args.lr_ramp_from) / max(1, args.lr_ramp_steps)))
        lr *= 1.0 + (to / args.lr - 1.0) * frac
    return lr


def latest_resumable(out, sampler_fp8=False, weights_ok=False):
    """The newest complete checkpoint with its DCP state (dcp/, or dcp_model/ + dcp_optim/; dcp_model/ alone when
    weights_ok) and the model dir the engines restart from: model_fp8/ with --sampler-fp8 (the trainer itself
    resumes from the base + DCP), else the HF bf16 model/."""
    from arcsolver.rl import fsdp_layout as L
    for d in reversed(ot.ckpt_dirs(out)):
        engine_dir = os.path.join(d, "model_fp8" if sampler_fp8 else "model")
        layout = L.dcp_layout(d)
        if layout in ("joint", "split") or (weights_ok and layout == "weights"):
            if os.path.isdir(engine_dir):
                return d
    return None


def bf16_export_due(args, step):
    every = getattr(args, "bf16_export_every", 1)
    return step == args.max_steps or (every > 0 and step % every == 0)


def setup_focus(args, order, pool, start_step, log):
    """--pool-focus-*: map the focus task ids to pool indices and seed their draw counts from the rollout records."""
    from arcsolver.rl import mixorder as rmo
    with open(args.pool_focus_file) as f:
        ids = {str(x).removesuffix(".json") for x in json.load(f)}
    index_of = {it.task_id: i for i, it in enumerate(pool) if it.task_id in ids and it.source != "arc_eval"}
    t0 = time.time()
    draws = rmo.focus_draws_from_records(args.out, args.pool_focus_from_step, start_step, index_of)
    waiting = len(set(index_of.values()) & set(order.new_queue))
    mode = getattr(args, "pool_focus_order", "fewest")
    cursor = 0
    if mode == "shuffle":
        cursor = rmo.focus_draw_count(args.out, args.pool_focus_shuffle_from, start_step, index_of)
    order.set_focus([index_of[t] for t in sorted(index_of)], args.pool_focus_per_step, draws, order=mode, cursor=cursor,
                    max_frac=getattr(args, "pool_focus_max_frac", 0.5))
    cap_file = getattr(args, "pool_focus_cap_file", "") or ""
    if cap_file:
        with open(cap_file) as f:
            capped = {str(x).removesuffix(".json") for x in json.load(f)}
        sched = [tuple(int(v) for v in k.split(":")) for k in str(getattr(args, "pool_focus_cap_schedule", "") or "").split(",")
                 if k.strip()] or None
        order.set_focus_cap([index_of[t] for t in capped if t in index_of], args.pool_focus_cap, args.pool_focus_cap_from,
                            schedule=sched)
        per_step = rmo.focus_draws_from_records(args.out, args.pool_focus_shuffle_from, start_step, index_of,
                                                per_step_total=True)
        order.focus_cursor = cursor = order.replay_focus_cursor(per_step)
        log(f"prompt order: focus cap of {len(order.focus_cap_set)} capped tasks per step: "
            f"{', '.join(f'{b} from step {a}' for a, b in order.focus_cap_sched)} ({len(capped)} ids in {cap_file}); "
            f"cursor replayed to {cursor} "
            f"({sum(per_step.values())} focus draws recorded)")
    n = [v[0] for v in draws.values()]
    log(f"prompt order: focus set {len(index_of)} of {len(ids)} ids in the pool, at least {args.pool_focus_per_step} "
        f"per step, order {mode}{f' (cycle position {cursor} of {len(index_of)}, from step {args.pool_focus_shuffle_from})' if mode == 'shuffle' else ''}; "
        f"drawn so far {len(draws)} (draws {min(n) if n else 0}..{max(n) if n else 0}), {waiting} were waiting in the "
        f"new-task queue; records of steps {args.pool_focus_from_step}..{start_step} read in {time.time() - t0:.0f}s")


def restore_prompt_order(args, state, order, log, pool=None, start_step=0, pool_changed=False):
    """pool_changed: the checkpoint's pool manifest differs from this launch's (e.g. a source dropped);
    with --new-pool-on-resume the order then starts fresh instead of failing to load the old pool's indices."""
    so = state.get("order", {})
    if (getattr(args, "pool_extend", False) and getattr(args, "pool_order", "weighted") == "frontier_mix"
            and so.get("kind") == "frontier_mix" and so.get("sizes") != [len(x) for x in order.strata]):
        # the pool grew, so its indices changed: rebuild the order from the records by task id
        info = order.extend_from_records(so, args.out, start_step, {it.task_id: i for i, it in enumerate(pool)},
                                         args.seed, tiers=getattr(args, "pool_extend_tiers", False))
        log(f"prompt order: pool extended (--pool-extend): {json.dumps(info)}")
        return
    # a --pool-order switch (weighted <-> frontier) cannot reuse the saved order's state: fresh only on request
    saved = state.get("args", {}).get("pool_order", "weighted")
    switched = saved != getattr(args, "pool_order", "weighted")
    if (switched and args.pool_order == "frontier_mix" and saved == "weighted" and not args.new_pool_on_resume
            and state.get("order", {}).get("kind") == "stratified"):
        # a stratified (bank / other) order converts: its fresh draws continue, and the frontier queue
        # is seeded from the run's rollout records so far
        from arcsolver.rl import mixorder as rmo
        order.load_state_dict(state["order"])
        hist = rmo.history_from_records(args.out, start_step, {it.task_id: i for i, it in enumerate(pool)})
        n = order.seed_history(hist)
        log(f"prompt order: stratified -> frontier_mix; fresh draws continue; frontier queue seeded from "
            f"{len(hist)} steps of rollout records: {n} tasks")
    elif args.new_pool_on_resume and (switched or pool_changed
                                      or state.get("args", {}).get("pool_file") != args.pool_file):
        log("new pool on resume: fresh prompt order")
    elif switched:
        raise SystemExit(f"checkpoint holds a --pool-order {saved} prompt order, this run uses {args.pool_order}: "
                         "add --new-pool-on-resume to start a fresh one")
    else:
        order.load_state_dict(state["order"])


def eval_due(args, step):
    """An eval after this step: every --eval-every steps, plus the --eval-steps list."""
    extra = {int(x) for x in str(getattr(args, "eval_steps", "") or "").split(",") if x.strip()}
    return (args.eval_every > 0 and step % args.eval_every == 0) or step in extra


def resume_eval_pending(args, step):
    return (step > 0 and eval_due(args, step)
            and not os.path.isfile(os.path.join(args.out, "evals", f"step_{step:06d}.COMPLETE")))


def save_rollout_record(out, step, rollouts):
    ot.write_jsonl_gz(os.path.join(out, "rollouts", f"step_{step:06d}.jsonl.gz"), [
        {"task_id": r["item"].task_id, "source": r["item"].source, "seen": r["seen"], "group": r["group"],
         "n_prompt": len(r["prompt_ids"]), "aug": r["aug"].meta(), "finish": r["finish"], "seq": r["seq"],
         "r_seq": r["r_seq"], "dense_sum": r["dense_sum"], "seq_adv": r["seq_adv"],
         "dropped": r["dropped"], **({"trunc_masked": True} if r.get("trunc_masked") else {}),
         "stats": r["stats"], "text": r["tt"].text,
         **({"r_seq_natural": r["r_seq_natural"], "forced_outcome": r["forced_outcome"],
             "forced_finish": r.get("forced_finish"), "forced_text": r["forced_text"],
             "forced_credit": r.get("forced_credit", False)} if "forced_outcome" in r else {}),
         **({"test_idx": r["test_idx"]} if r.get("test_idx") is not None else {}),
         **({"prefix": r["prefix"]} if r.get("prefix") else {})}
        for r in rollouts])


def save_rollout_tokens(out, step, rollouts):
    """The step's exact prompt and completion token ids, in rollout-record order (the text record alone cannot be
    re-tokenized back to the sampled ids, so a failed update could not be replayed from it)."""
    import numpy as np
    ps = [np.asarray(r["prompt_ids"], dtype=np.int32) for r in rollouts]
    cs = [np.asarray(r["completion"], dtype=np.int32) for r in rollouts]
    path = os.path.join(out, "rollouts", f"step_{step:06d}.tokens.npz")
    with open(path + ".part", "wb") as f:
        np.savez_compressed(f, prompt=np.concatenate(ps) if ps else np.zeros(0, np.int32),
                            prompt_len=np.asarray([len(x) for x in ps], dtype=np.int64),
                            completion=np.concatenate(cs) if cs else np.zeros(0, np.int32),
                            completion_len=np.asarray([len(x) for x in cs], dtype=np.int64))
    os.replace(path + ".part", path)


def start_rollout_writer(out, step, rollouts):
    """--rollout-write-async: save_rollout_record + save_rollout_tokens in a thread (gzip and zlib release the GIL).
    Returns (thread, box); join_rollout_writer re-raises a failure in the caller."""
    import threading
    box = {}

    def run():
        try:
            save_rollout_record(out, step, rollouts)
            save_rollout_tokens(out, step, rollouts)
        except BaseException as e:  # noqa: BLE001 - re-raised by join_rollout_writer
            box["error"] = e

    t = threading.Thread(target=run, name=f"rollout-writer-{step}", daemon=True)
    t.start()
    return t, box


def join_rollout_writer(w):
    """Wait for a start_rollout_writer thread (None = nothing pending); returns None for the caller's holder."""
    if w is None:
        return None
    t, box = w
    t.join()
    if "error" in box:
        raise RuntimeError("rollout record write failed") from box["error"]
    return None


def read_control(path):
    """The per-step control file (JSON) as a dict; {} when absent or unreadable."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def topk_node(r):
    """The node whose engine wrote a rollout's top-k file (<topk_dir>/step_N/n<node>/...), or None (no file, or a
    single-node run whose files carry no node directory)."""
    p = r.get("topk_path")
    if not p:
        return None
    d = os.path.basename(os.path.dirname(p))
    return int(d[1:]) if d.startswith("n") and d[1:].isdigit() else None


def node_affine(assign_equal, nodes, local_ws):
    """Wrap a rank assignment so each item goes to a rank of its node (items without a node: anywhere). nodes: per item
    in the order the lengths are given; local_ws: ranks per node."""
    def assign(lengths, world):
        n_nodes = world // local_ws
        out = [[] for _ in range(world)]
        free = [i for i, n in enumerate(nodes) if n is None]
        for n in range(n_nodes):
            idx = [i for i, m in enumerate(nodes) if m == n]
            if any(m is not None and m >= n_nodes for m in nodes):
                raise RuntimeError(f"top-k files from node {max(m for m in nodes if m is not None)} but {n_nodes} nodes")
            if not idx:
                continue
            for r, part in enumerate(assign_equal([lengths[i] for i in idx], local_ws)):
                out[n * local_ws + r].extend(idx[a] for a in part)
        if free:
            for r, part in enumerate(assign_equal([lengths[i] for i in free], world)):
                out[r].extend(free[a] for a in part)
        return out
    return assign


def plan_minibatches(kept, work, world, norms, loss_agg, mb_max, min_groups, assign_equal, length_only=frozenset(),
                     local_ws=None):
    """Split a step's trained rollouts into optimizer minibatches. Groups stay whole; the number of
    minibatches is min(mb_max, groups // min_groups), at least 1; groups go to the minibatch with the fewest tokens
    so far (largest first). Each minibatch is a token-mean update over its own kept rollouts (rl.normalisers on that
    minibatch); with one minibatch the step's original normalisers are used unchanged. Returns
    [{"assign": [[kept index] per rank], "n_micro": int, "norm": {kept index: normaliser}}]. Groups in length_only
    (all-correct groups kept only for --len-allcorrect-beta) are placed like any other but not counted, so the
    minibatch count still means >= min_groups groups with an outcome signal each."""
    from arcsolver.rl import loss as rl
    groups = []
    for j in work:
        g = kept[j]["group"]
        if g not in groups:
            groups.append(g)
    n_count = sum(1 for g in groups if g not in length_only)
    n_mb = max(1, min(int(mb_max), n_count // max(1, int(min_groups)))) if groups else 1
    if n_mb == 1:
        bins = [set(groups)]
    else:
        tok = {g: 0 for g in groups}
        for j in work:
            tok[kept[j]["group"]] += len(kept[j]["prompt_ids"]) + len(kept[j]["completion"])
        load, bins = [0] * n_mb, [set() for _ in range(n_mb)]
        for g in sorted(groups, key=lambda g: (-tok[g], g)):
            k = min(range(n_mb), key=lambda i: (load[i], i))
            bins[k].add(g)
            load[k] += tok[g]
    plan = []
    for b in bins:
        mb_work = [j for j in work if kept[j]["group"] in b]
        if n_mb == 1:
            norm = {j: norms[j] for j in mb_work}
        else:
            mb_kept = [j for j, r in enumerate(kept) if r["group"] in b]
            nk = rl.normalisers([len(kept[j]["completion"]) for j in mb_kept], loss_agg)
            norm = dict(zip(mb_kept, nk))
        lengths = [len(kept[j]["prompt_ids"]) + len(kept[j]["completion"]) for j in mb_work]
        fn = assign_equal
        if local_ws and world > local_ws:              # top-k files are node-local: train each rollout on its node
            nodes = [topk_node(kept[j]) for j in mb_work]
            if any(n is not None for n in nodes):
                fn = node_affine(assign_equal, nodes, local_ws)
        assign = fn(lengths, world) if mb_work else [[] for _ in range(world)]
        plan.append({"assign": [[mb_work[a] for a in ranks] for ranks in assign],
                     "n_micro": max(1, max(len(a) for a in assign)), "norm": norm})
    return plan


def check_behaviour_logprobs(s, sampler=None):
    """A sample's behaviour log probabilities (the sampler's own, the PPO denominator; never substituted by rescored
    values) must exist, align with its tokens, be finite and come from the committed policy."""
    if not s["token_ids"]:
        return
    lp = s.get("logprobs")
    if lp is None or len(lp) != len(s["token_ids"]):
        raise RuntimeError("sampler returned logprobs misaligned with token ids")
    if not all(math.isfinite(x) for x in lp):
        raise RuntimeError("sampler returned non-finite behaviour logprobs")
    if sampler is not None and hasattr(sampler, "check_record"):
        sampler.check_record(s, "rollout")


def kept_rollouts(rollouts, length_only=frozenset()):
    """The rollouts that enter the loss (and its token-mean normaliser): not in a dropped group, non-empty, and, in an
    all-correct group kept only for its length signal (--len-allcorrect-beta), only the rollouts that got one; the
    longer ones would add untrained tokens to the normaliser and dilute every other signal)."""
    return [r for r in rollouts if not r["dropped"] and r["completion"] and not r.get("trunc_masked")
            and not (r["group"] in length_only and r["seq_adv"] == 0.0)]


def mask_truncated_rollouts(rollouts):
    """--mask-truncated (DAPO overlong filtering): every truncated rollout of a kept group is marked trunc_masked,
    so kept_rollouts() leaves it out of the loss and the token-mean normaliser. Its group statistics are unchanged
    (it counted as a failure when the advantages were computed). Returns (rollouts masked, their tokens)."""
    n = toks = 0
    for r in rollouts:
        if r["finish"] == "length" and not r["dropped"]:
            r["trunc_masked"] = True
            n += 1
            toks += len(r["completion"])
    return n, toks


def scale_truncated_advantages(rollouts, scale):
    """--trunc-neg-scale: every truncated rollout of a kept group keeps its place in the loss
    and its normaliser, with its sequence advantage (<= 0: it scored R = 0 in its group's statistics) times scale.
    Its group then no longer sums to 0 but gets a net push of (1 - scale) x the sum of |A| over its truncated
    rollouts, toward the rollouts that finished. A truncated rollout credited by its forced answer (--train-budget-force)
    with a positive advantage is left alone; one with a negative advantage (its group's mean is above the forced
    credit, e.g. 6-7 natural successes of 8) is scaled like the rest, so a right forced answer never gets pushed down
    harder than a wrong one. Returns (rollouts scaled, their tokens)."""
    n = toks = 0
    for r in rollouts:
        if r["finish"] == "length" and not r["dropped"] and not (r.get("forced_credit") and r["seq_adv"] > 0):
            r["seq_adv"] *= scale
            r["trunc_scaled"] = True
            n += 1
            toks += len(r["completion"])
    return n, toks


FORCE_STREAM_SEQS = 24      # forced answers in flight per engine (their prompts are whole ~60-150k-token rollouts)


_RUN_LOG = {"fn": lambda *a: print(*a, flush=True)}   # main() puts its run logger here


class RolloutStore:
    """--rollout-store-dir (preemption-safe sampling): the finished samples of the training step being drawn are saved
    under
    DIR/step_N/ as they arrive: s_<group>_<sample>.json.gz (the engine's record) with its top-k arrays in topk/, and
    f_<rollout>.json.gz (a forced answer). One writer thread (tmp, fsync, rename; the top-k copy lands before its
    record), so a preemption leaves at most a stray .tmp. MANIFEST.json pins the step, the policy (the sampler's weight
    digest), the prompts, seeds, caps and sampling parameters: a restart that redraws step N identically loads the
    saved samples and samples only the rest; a different manifest discards the directory. A step's directory is
    removed once the NEXT step's checkpoint is complete (cleanup). Write failures are logged, never fatal: durability
    is a bonus."""

    def __init__(self, root, step, manifest, log):
        import queue
        import threading
        self.dir, self.step, self.log = os.path.join(root, f"step_{step:06d}"), step, log
        mf = os.path.join(self.dir, "MANIFEST.json")
        if os.path.isdir(self.dir):
            try:
                with open(mf) as f:
                    old = json.load(f)
            except (OSError, ValueError):
                old = None
            if old != manifest:
                bad = sorted(k for k in set(old or {}) | set(manifest) if (old or {}).get(k) != manifest.get(k))
                log(f"rollout store {self.dir}: manifest differs ({bad}); discarding its samples")
                shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(os.path.join(self.dir, "topk"), exist_ok=True)
        if not os.path.isfile(mf):
            EvalStore._write(mf, manifest, gz=False)
        self.q, self.errors, self.n_put, self.closed = queue.Queue(), [], 0, False
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, obj, src, dst = item
            try:
                if src:
                    tmp = dst + ".tmp"
                    shutil.copyfile(src, tmp)
                    with open(tmp, "rb") as f:
                        os.fsync(f.fileno())
                    os.replace(tmp, dst)
                EvalStore._write(path, obj)
            except Exception as e:                         # logged at close(), never fatal
                self.errors.append(f"{os.path.basename(path)}: {e!r}")

    def put_sample(self, i, j, rec):
        name = f"{i:04d}_{j:02d}"
        src = rec.get("topk_path")
        obj = {"i": i, "j": j, "rec": dict(rec, topk_path=None), "topk": bool(src)}
        self.q.put((os.path.join(self.dir, f"s_{name}.json.gz"), obj, src,
                    os.path.join(self.dir, "topk", f"{name}.npz") if src else None))
        self.n_put += 1

    def put_forced(self, key, seed, rec):
        self.q.put((os.path.join(self.dir, f"f_{key:05d}.json.gz"), {"key": key, "seed": seed, "rec": rec}, None, None))
        self.n_put += 1

    def _read(self, kind):
        import glob
        import gzip
        out = []
        for fn in sorted(glob.glob(os.path.join(self.dir, f"{kind}_*.json.gz"))):
            try:
                with gzip.open(fn, "rt") as f:
                    out.append(json.load(f))
            except (OSError, ValueError, EOFError) as e:
                self.log(f"rollout store: ignoring unreadable {os.path.basename(fn)} ({e!r})")
        return out

    def load_samples(self, ns, topk_dir):
        """{(group, sample): record}; a saved top-k file is copied back into this step's top-k dir (records whose
        top-k copy is missing are dropped, so they are sampled again)."""
        out = {}
        for o in self._read("s"):
            i, j, rec = int(o["i"]), int(o["j"]), o["rec"]
            if not (0 <= i < len(ns) and 0 <= j < ns[i]):
                continue
            if o.get("topk"):
                src = os.path.join(self.dir, "topk", f"{i:04d}_{j:02d}.npz")
                if not topk_dir or not os.path.isfile(src):
                    continue
                dst = os.path.join(topk_dir, f"restored_{i:04d}_{j:02d}.npz")
                shutil.copyfile(src, dst)
                rec = dict(rec, topk_path=dst)
            out[(i, j)] = rec
        return out

    def load_forced(self):
        return {int(o["key"]): o for o in self._read("f")}

    def close(self):
        if self.closed:                                    # idempotent: also called on an error path
            return
        self.closed = True
        self.q.put(None)
        self.thread.join()
        if self.errors:
            self.log(f"rollout store {self.dir}: {len(self.errors)} of {self.n_put} writes failed, e.g. {self.errors[:2]}")

    @staticmethod
    def cleanup(root, done_step, log):
        """Remove the stores of steps <= done_step (their samples are in a complete checkpoint)."""
        import glob
        for d in glob.glob(os.path.join(root, "step_*")):
            try:
                s = int(os.path.basename(d)[5:])
            except ValueError:
                continue
            if s <= done_step:
                shutil.rmtree(d, ignore_errors=True)


def open_rollout_store(args, step, sampler, prompts, ns, caps, seeds, gen_params, log):
    """The RolloutStore of this step, or None when --rollout-store-dir is off or the sampling cannot be redone exactly
    (no streaming dispatch or no per-request seeds)."""
    root = getattr(args, "rollout_store_dir", "") or ""
    if not root:
        return None
    if not getattr(sampler, "stream_seqs", 0) or seeds is None:
        log("rollout store off: needs streaming dispatch (--sampler-stream-seqs) and --per-request-seeds")
        return None
    from array import array
    h = lambda xs: hashlib.sha256(json.dumps(xs, sort_keys=True, default=str).encode()).hexdigest()[:24]
    digest = getattr(sampler, "committed_policy", None)              # the full committed policy identity
    if digest is None:
        try:                                               # untracked sampler: engine 0's tensor digest (weaker)
            digest = sampler.engines[0].call("digest")["digests"]
        except Exception as e:                             # no digest = no safe reuse
            log(f"rollout store off this step: sampler digest failed ({e!r})")
            return None
    psha = [hashlib.sha1(array("i", [int(t) for t in p]).tobytes()).hexdigest()[:16] for p in prompts]
    manifest = {"step": step, "run": args.run_name, "policy": h(digest), "prompts_sha": h(psha), "n_prompts": len(prompts),
                "n_per_prompt": h(list(ns)), "caps": h(list(caps)), "seeds": h(list(seeds)), "gen_params": h(gen_params),
                "train_budget_force": getattr(args, "train_budget_force", 0), "seed": args.seed}
    try:
        return RolloutStore(root, step, manifest, log)
    except OSError as e:
        log(f"rollout store off this step: {e!r}")
        return None


def generate_durable_step(sampler, store, prompts, ns, caps, seeds, gen_params, topk_dir, tag, on_sample, log, force=None):
    """sampler.call("generate", ...) for a training step with every finished sample saved to `store` and saved ones
    reused: the same requests streaming dispatch makes (sample j of prompt i = an n=1 request with seed seeds[i] + j),
    sample-major, so a full redraw gives the same samples as the unsaved path. `force` (--train-force-inline): the
    engines attach each cut sample's forced answer (seeded from the sample's seed), saved with its sample."""
    t0 = time.time()
    loaded = store.load_samples(ns, topk_dir)
    pol = getattr(sampler, "committed_policy", None)
    if pol is not None:                                    # a saved record of another policy is sampled again
        loaded = {k: v for k, v in loaded.items() if v.get("policy") == pol}
    outputs = [[None] * n for n in ns]
    for (i, j), rec in loaded.items():
        outputs[i][j] = rec
    todo = sorted(((i, j) for i in range(len(ns)) for j in range(ns[i]) if outputs[i][j] is None), key=lambda x: (x[1], x[0]))
    log(f"rollout store {store.dir}: {len(loaded)} of {sum(ns)} samples restored ({time.time() - t0:.0f}s), "
        f"{len(todo)} to generate")
    # the main loop's step metrics read gen_tokens, seconds and dispatch like the plain call's result
    r = {"gen_tokens": 0, "seconds": 0.0}
    if todo:
        def cb(e, _j, rec):
            i, j = todo[e]
            store.put_sample(i, j, rec)
            if on_sample is not None:
                on_sample(i, j, rec)
        t1 = time.time()
        r = sampler.call("generate", prompts=[prompts[i] for i, _ in todo], topk_dir=topk_dir, tag=tag,
                         n_per_prompt=[1] * len(todo), params=dict(gen_params, n=1),
                         max_tokens=[caps[i] for i, _ in todo], seeds=[seeds[i] + j for i, j in todo], on_sample=cb,
                         **({"force": force} if force else {}))
        for (i, j), outs in zip(todo, r["outputs"]):
            outputs[i][j] = outs[0]
        r.setdefault("seconds", time.time() - t1)
        r.setdefault("gen_tokens", sum(len(outs[0]["token_ids"]) for outs in r["outputs"]))
    out = {k: v for k, v in r.items() if k != "outputs"}
    out.update(outputs=outputs, restored=len(loaded))
    return out


def force_suffix(stok):
    """The budget-forcing suffix (training and eval forced answers) and its token ids. Default: close the reasoning
    ("\n</think>\n\n", or the ANSWER_MARKER line on the output channel). $ARC_EVAL_FORCE_SUFFIX (backslash escapes
    decoded) overrides it: e.g. models whose training traces close with "\n\nNow final.\n</think>\n\n" need that line;
    with the bare tag they can keep reasoning past the answer budget so that no forced answer parses."""
    env = os.environ.get("ARC_EVAL_FORCE_SUFFIX")
    if env:
        import codecs
        suffix = codecs.decode(env, "unicode_escape")
    else:
        suffix = "\n</think>\n\n" if ar.default_channel() == "think" else "\n\n" + ar.ANSWER_MARKER + "\n"
    return suffix, stok(suffix, add_special_tokens=False)["input_ids"]


def force_params(args):
    """Sampling settings of the forced answers (as the eval samples)."""
    return {"n": 1, "temperature": args.eval_temperature, "top_p": args.eval_top_p, "top_k": args.eval_top_k,
            "logprobs": False}


def score_forced(task, completion, suf_ids, fids, ffinish, aligner, cfg):
    """A forced answer's scores (force_answers' caller, as a pure function for the --reward-stream pool)."""
    return rr.score_rollout(task, aligner(list(completion) + suf_ids + list(fids)), ffinish, cfg)


def inline_forced_answers(rollouts, suffix, suf_ids):
    """--train-force-inline: force_answers' result from the forced answers the engines attached to the samples
    (vllm_worker stream msg force): {rollout index: (answer token ids, finish, suffix, suffix ids)}."""
    out = {}
    for j, r in enumerate(rollouts):
        f = r.pop("_forced", None)                    # private to this hand-over: not kept in the rollout records
        if f is not None:
            out[j] = (f["token_ids"], f["finish"], suffix, suf_ids)
    return out


def force_answers(args, step, sampler, stok, rollouts, store=None, only=None):
    """--train-budget-force: every on-policy rollout cut at the token cap gets its reasoning closed (the eval's
    budget-forcing suffix) and up to N tokens to answer, sampled as the eval samples. Returns {rollout index:
    (answer token ids, finish, suffix, suffix ids)}; the answer is only scored, never trained. `only` (a set of
    rollout indices; --train-force-inline's fallback for samples that came without an inline answer, e.g. restored from
    a store drawn without it) limits it to those rollouts, each seeded by its rollout index."""
    suffix, suf_ids = force_suffix(stok)
    fprompts, keys, fmax = [], [], []
    for j, r in enumerate(rollouts):
        if only is not None and j not in only:
            continue
        if r["finish"] != "length" or not r["completion"]:
            continue
        p = list(r["prompt_ids"]) + list(r["completion"]) + suf_ids
        room = args.sampler_max_model_len - len(p)
        if room >= 64:
            fprompts.append(p)
            keys.append(j)
            fmax.append(min(args.train_budget_force, room))
        else:
            r["forced_skipped"] = True                     # no room left in the context for an answer
    if not fprompts:
        return {}
    fparams = force_params(args)
    fseeds = [oc.request_seed(args.seed + 13, step, j, 1) for j in (range(len(fprompts)) if only is None else keys)]
    if store is None:
        res = sampler.call("generate", prompts=fprompts, params=fparams, max_tokens=fmax, seeds=fseeds,
                           stream_target=FORCE_STREAM_SEQS)
        return {j: (outs[0]["token_ids"], outs[0]["finish"], suffix, suf_ids) for j, outs in zip(keys, res["outputs"])}
    # --rollout-store-dir: forced answers saved as they arrive; saved ones with the same seed are reused
    saved = store.load_forced()
    done = {}
    for e, k in enumerate(keys):
        o = saved.get(k)
        if o is not None and int(o.get("seed", -1)) == fseeds[e]:
            done[e] = o["rec"]
    todo = [e for e in range(len(keys)) if e not in done]
    if todo:
        def cb(t, _j, rec):
            e = todo[t]
            store.put_forced(keys[e], fseeds[e], {"token_ids": rec["token_ids"], "finish": rec["finish"]})
        res = sampler.call("generate", prompts=[fprompts[e] for e in todo], params=fparams, max_tokens=[fmax[e] for e in todo],
                           seeds=[fseeds[e] for e in todo], stream_target=FORCE_STREAM_SEQS, on_sample=cb)
        for e, outs in zip(todo, res["outputs"]):
            done[e] = outs[0]
    return {keys[e]: (done[e]["token_ids"], done[e]["finish"], suffix, suf_ids) for e in range(len(keys))}


def forced_stats(rollouts, mask):
    """Step metrics of --train-budget-force."""
    tried = [r for r in rollouts if "forced_outcome" in r]
    groups = {}
    for r, m in zip(rollouts, mask):
        g = groups.setdefault(r["group"], [0, 0])
        g[0] += r["seq"]["outcome"] > 0                    # natural successes
        g[1] += m                                          # forced-credited
    return {"forced/tried": len(tried), "forced/correct": sum(r["forced_outcome"] > 0 for r in tried),
            "forced/credited": sum(mask), "forced/cut_at_budget": sum(r.get("forced_finish") == "length" for r in tried),
            "forced/skipped_no_room": sum(bool(r.get("forced_skipped")) for r in rollouts),
            "forced/correct_rate": sum(r["forced_outcome"] > 0 for r in tried) / max(1, len(tried)),
            "forced/rescued_groups": sum(1 for nat, fc in groups.values() if nat == 0 and fc > 0)}


def adv_weight_metrics(rollouts, kept, group_size):
    """Where a step's update weight goes, over the trained rollouts (kept) with their final
    sequence advantage A: the share of sum |A| by how many of the group's on-policy rollouts got an outcome reward
    > 0 (buckets 1, 2-3, 4-5, 6-7 of 8; other group sizes scaled to 8; "other" = groups whose rollouts all got
    credit, or none did, but unequal amounts), the share from truncated rollouts, the effective sample size
    (sum |A|)^2 / sum A^2 over rollouts, and the net sum of A (> 0 when truncated rollouts are scaled down). With
    --loss-agg seq each rollout weighs |A| / n in the update whatever its length."""
    correct = {}
    for r in rollouts:
        correct[r["group"]] = correct.get(r["group"], 0) + (r["seq"]["outcome"] > 0)

    def bucket(k):
        kk = k * 8.0 / max(1, group_size)
        if k == 0 or kk >= 8:
            return "other"
        return "k1" if kk <= 1 else ("k2_3" if kk <= 3 else ("k4_5" if kk <= 5 else "k6_7"))

    names = ("k1", "k2_3", "k4_5", "k6_7", "other")
    w = {b: 0.0 for b in names}
    groups = {b: set() for b in names}
    tot = sq = net = trunc = 0.0
    for r in kept:
        a = float(r["seq_adv"])
        b = bucket(correct.get(r["group"], 0))
        w[b] += abs(a)
        groups[b].add(r["group"])
        tot += abs(a)
        sq += a * a
        net += a
        if r["finish"] == "length":
            trunc += abs(a)
    m = {f"adv/weight_share_{b}": w[b] / tot if tot > 0 else 0.0 for b in names}
    m.update({f"adv/groups_{b}": len(groups[b]) for b in names})
    m.update({"adv/weight_share_truncated": trunc / tot if tot > 0 else 0.0,
              "adv/ess_rollouts": tot * tot / sq if sq > 0 else 0.0,
              "adv/sum_abs": tot, "adv/net_sum": net})
    return m


def sampled_nll_mean(rollouts):
    """Mean -log q(y_t) over the on-policy rollouts' sampled tokens under the sampler (temperature 1): a Monte-Carlo
    estimate of the policy's per-token entropy (the sampler's, off the trainer's by the sampler/trainer mismatch)."""
    n, s = 0, 0.0
    for r in rollouts:
        if r.get("sampler_lp"):
            n += len(r["sampler_lp"])
            s -= math.fsum(r["sampler_lp"])
    return s / n if n else 0.0


def dropped_groups(mode, rollouts, gids):
    """Groups with no learning signal. off: none; total / seq: zero variance of R_seq (+ dense rewards for total);
    outcome: zero variance of the outcome before length shaping. A length term gives all-correct groups a tiny reward
    spread (advantages ~0.001), so total would keep and train them at full compute cost."""
    if mode == "off":
        return set()
    if mode == "outcome":
        return rr.zero_variance_groups([r["r_outcome"] for r in rollouts], gids)
    return rr.zero_variance_groups([r["r_seq"] + (r["dense_sum"] if mode == "total" else 0.0) for r in rollouts], gids)


def parse_solved_seed(spec: str, check_dirs: bool = True) -> list:
    """--pool-drop-solved-seed "DIR:UPTO[,DIR2:UPTO2...]" -> [(DIR, UPTO), ...]; with check_dirs each DIR must hold
    rollout records (checked on rank 0, which builds the order; the argument check only parses)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        sdir, _, supto = part.rpartition(":")
        if not sdir or not supto.isdigit():
            raise SystemExit(f"--pool-drop-solved-seed: bad entry {part!r} (want DIR:UPTO)")
        if check_dirs and not os.path.isdir(os.path.join(sdir, "rollouts")):
            raise SystemExit(f"--pool-drop-solved-seed: {sdir}/rollouts does not exist")
        out.append((sdir, int(supto)))
    return out


def solved_from_records(out, upto_step, index_of, group_size):
    """--pool-drop-solved seed: pool indices of the tasks with an all-correct group (natural outcomes, group_size
    on-policy samples) in OUT/rollouts/step_N.jsonl.gz for N <= upto_step (tasks no longer in the pool are ignored)."""
    import glob
    import gzip
    found = set()
    for path in sorted(glob.glob(os.path.join(out, "rollouts", "step_*.jsonl.gz"))):
        try:
            step = int(os.path.basename(path)[5:11])
        except ValueError:
            continue
        if step > upto_step:
            continue
        groups = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("prefix"):
                    continue
                g = groups.setdefault(r["group"], [r["task_id"], 0, 0])
                g[1] += (r.get("seq") or {}).get("outcome", 0) > 0
                g[2] += 1
        for tid, c, n in groups.values():
            if n == group_size and c == n and tid in index_of:
                found.add(index_of[tid])
    return found


def frontier_results(rollouts):
    """FrontierOrder.update's input for one step (--pool-order frontier): per rollout group, in group order,
    (pool index, correct = outcome reward > 0, samples, truncated = finish "length") over its on-policy rollouts.
    The outcome component already applies --outcome-require-stop and --outcome-all-tests (partial credit counts) and,
    unlike r_seq, no length shaping."""
    out = {}
    for r in rollouts:
        if r.get("prefix"):                                   # prefix groups are not draws of the order
            continue
        g = out.setdefault(r["group"], [r["pool_index"], 0, 0, 0])
        g[1] += r["seq"]["outcome"] > 0
        g[2] += 1
        g[3] += r["finish"] == "length"
    return [tuple(out[k]) for k in sorted(out)]


def score_sample(task, s, aligner, cfg):
    """One sample's (TokenText, rr.score_rollout scores): a pure function of the task, the sample's token ids and
    finish reason, so it can run in any thread at any time (--reward-stream)."""
    tt = aligner(s["token_ids"])
    return tt, rr.score_rollout(task, tt, s["finish"], cfg)


_TOKEN_SETS = {}


def prefix_token_sets(stok):
    """(</think> token id, ids of tokens containing a newline), cached per tokenizer."""
    key = id(stok)
    if key not in _TOKEN_SETS:
        nl = {i for t, i in stok.get_vocab().items() if "\u010a" in t or "\n" in t}
        _TOKEN_SETS[key] = (stok.convert_tokens_to_ids(ar.THINK_END), nl)
    return _TOKEN_SETS[key]


def choose_prefix_sources(rows, comps, n, max_zero, min_start, end_margin, think_id, nl_ids, rng, zero_max_frac=1.0,
                          early_frac=0.0, early_prob=0.8):
    """--prefix-groups source choice from the previous step's rollout records `rows` and completion token ids `comps`
    (same order). Candidate groups: fresh on-policy groups (no "prefix" key) whose rollouts were not all fully right
    (outcome >= 1); at most max_zero chosen groups had no fully right rollout; one group per task. In a chosen group one
    random rollout whose reasoning (tokens before its first </think>, all of it if it never closed) is long enough for
    a cut in [min_start, R - end_margin]; the cut is uniform in that range, then moved back to just after the nearest
    newline token within 512 tokens (never below min_start). zero_max_frac (--prefix-zero-max-frac): for a source from
    a group with no fully right rollout the cut is also at most zero_max_frac * R (1.0 = no limit). early_frac > 0
    (--prefix-early-frac): with probability early_prob the cut is uniform in [min_start, early_frac * R], otherwise uniform in
    [early_frac * R, hi] (only when early_frac * R lies inside the allowed range; else uniform as before).
    Returns [(row index, cut, group's fully-right count)]."""
    groups = {}
    for i, r in enumerate(rows):
        if r.get("prefix"):
            continue
        groups.setdefault(r["group"], []).append(i)
    cands = []
    for g in sorted(groups):
        idx = groups[g]
        k = sum(float((rows[i].get("seq") or {}).get("outcome", 0) or 0) >= 1 for i in idx)
        if k < len(idx):
            cands.append((idx, k))
    rng.shuffle(cands)
    out, tasks_seen, n_zero = [], set(), 0
    for idx, k in cands:
        if len(out) >= n:
            break
        t = rows[idx[0]]["task_id"]
        if (k == 0 and n_zero >= max_zero) or t in tasks_seen:
            continue
        ok = []
        for i in idx:
            c = comps[i]
            try:
                R = c.index(think_id)
            except ValueError:
                R = len(c)
            hi = R - end_margin if k else min(R - end_margin, int(zero_max_frac * R))
            if hi > min_start:
                ok.append((i, hi, R))
        if not ok:
            continue
        i, hi, R = rng.choice(ok)
        c = comps[i]
        split = int(early_frac * R)
        if early_frac > 0 and min_start < split < hi:
            cut = rng.randint(min_start, split) if rng.random() < early_prob else rng.randint(split, hi)
        else:
            cut = rng.randint(min_start, hi)
        for q in range(cut, max(min_start, cut - 512), -1):
            if c[q - 1] in nl_ids:
                cut = q
                break
        out.append((i, cut, k))
        tasks_seen.add(t)
        n_zero += k == 0
    return out


def build_prefix_groups(args, step, pool, tasks, stok, log=print, n=None):
    """The prefix groups of `step` from the records and token ids of step - 1 (also right after a resume): per chosen
    source the exact token sequence it was sampled with (its prompt ids + the first `cut` completion ids) and the task
    view it was scored on (its split test, if any, then its augmentation, rebuilt from the record). [] when the
    previous step's files are missing."""
    import gzip
    import numpy as np
    from arcsolver.rl import augment as ra
    base = os.path.join(args.out, "rollouts", f"step_{step - 1:06d}")
    for _ in range(24):                                   # the async writer saves the tokens last (atomic replace)
        if os.path.exists(base + ".tokens.npz") and os.path.exists(base + ".jsonl.gz"):
            break
        time.sleep(5)
    else:
        log(f"step {step}: prefix groups skipped: no rollout records / token ids of step {step - 1}")
        return []
    with gzip.open(base + ".jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    with np.load(base + ".tokens.npz") as z:              # an NpzFile re-reads (and decompresses) on EVERY access:
        plen, clen = z["prompt_len"], z["completion_len"]  # load each array once (per-row access takes minutes)
        prompt_all, comp_all = z["prompt"], z["completion"]
    po, co = np.cumsum(np.r_[0, plen]), np.cumsum(np.r_[0, clen])
    if len(rows) != len(plen):
        log(f"step {step}: prefix groups skipped: {len(rows)} records vs {len(plen)} token rows")
        return []
    comps = [comp_all[co[i]:co[i + 1]].tolist() for i in range(len(rows))]
    think_id, nl = prefix_token_sets(stok)
    rng = random.Random((args.seed * 1_000_003 + step) * 7_919 + 20_261_015)
    chosen = choose_prefix_sources(rows, comps, args.prefix_groups if n is None else n, args.prefix_max_zero,
                                   args.prefix_min_start,
                                   args.prefix_end_margin, think_id, nl, rng,
                                   zero_max_frac=getattr(args, "prefix_zero_max_frac", 1.0),
                                   early_frac=getattr(args, "prefix_early_frac", 0.0),
                                   early_prob=getattr(args, "prefix_early_prob", 0.8))
    index_of = {it.task_id: i for i, it in enumerate(pool)}
    out = []
    for i, cut, k in chosen:
        r = rows[i]
        if r["task_id"] not in index_of:
            continue
        pi = index_of[r["task_id"]]
        task = tasks(pool[pi].path)
        ti = r.get("test_idx")
        if ti is not None:
            task = dict(task)
            task["test"], task["_test_index"], task["_n_test"] = [task["test"][ti]], ti, len(tasks(pool[pi].path)["test"])
        meta = r.get("aug") or {}
        aug = ra.Aug(meta.get("d8", "id"), tuple(int(c) for c in meta.get("perm", "0123456789")))
        prefix = comps[i][:cut]
        out.append({"item": pool[pi], "pool_index": pi, "task": aug.task(task), "aug": aug,
                    "prompt": prompt_all[po[i]:po[i + 1]].tolist() + prefix, "prefix": prefix,
                    "meta": {"src_step": step - 1, "src_group": r["group"], "cut": cut, "src_len": len(comps[i]),
                             "src_outcome": float((r.get("seq") or {}).get("outcome", 0) or 0),
                             "src_finish": r.get("finish"), "src_group_right": k}})
    return out


def shift_dense(sc, P):
    """Scores of prefix + continuation -> token rewards indexed in the continuation (prefix tokens dropped)."""
    if not P or not sc.get("dense"):
        return sc
    dense = {k - P: v for k, v in sc["dense"].items() if k >= P}
    return dict(sc, dense=dense, dense_sum=sum(dense.values()))


def prefix_metrics(rollouts) -> dict:
    """--prefix-groups per step: groups, sources without a right rollout, continuation outcome (by source outcome
    and cut tercile of the source's reasoning), groups with a learning signal, and the fresh draws' outcome."""
    pr = [r for r in rollouts if r.get("prefix")]
    if not pr:
        return {}
    fr = [r for r in rollouts if not r.get("prefix")]
    gs = {}
    for r in pr:
        gs.setdefault(r["group"], []).append(r)
    out = {"prefix/groups": len(gs), "prefix/rollouts": len(pr),
           "prefix/from_zero": sum(v[0]["prefix"]["src_group_right"] == 0 for v in gs.values()),
           "prefix/outcome": sum(r["seq"]["outcome"] for r in pr) / len(pr),
           "prefix/informative": sum(not v[0]["dropped"] for v in gs.values()),
           "prefix/cut_mean": sum(v[0]["prefix"]["cut"] for v in gs.values()) / len(gs),
           "reward/outcome_fresh": sum(r["seq"]["outcome"] for r in fr) / max(1, len(fr))}
    for lab, pick in (("src_right", lambda m: m["src_outcome"] >= 1), ("src_wrong", lambda m: m["src_outcome"] < 1)):
        xs = [r["seq"]["outcome"] for r in pr if pick(r["prefix"])]
        if xs:
            out[f"prefix/outcome_{lab}"] = sum(xs) / len(xs)
    return out


def is_prefix_step(args, step) -> bool:
    """--prefix-every: prefix groups only on steps divisible by N."""
    every = getattr(args, "prefix_every", 1) or 1
    return every <= 1 or step % every == 0


def rollout_step(args, step, order, pool, tasks, stok, sampler, cfg, aligner):
    """One step's augment, sample, rewards and groups on rank 0. With --prefix-groups the last groups continue frozen
    prefixes of the previous step's rollouts (build_prefix_groups)."""
    t0 = time.time()
    pstep = is_prefix_step(args, step)
    pre = (build_prefix_groups(args, step, pool, tasks, stok, n=args.prefix_groups)
           if pstep and getattr(args, "prefix_groups", 0) > 0 and step >= getattr(args, "prefix_from", 0) and step > 1
           else [])
    taken = order.take(args.prompts_per_step + (getattr(args, "prefix_extra_prompts", 0) if pre else 0) - len(pre))
    items = [pool[i] for i in taken]
    gtasks, prompts, gaugs, aug_retries = [], [], [], 0
    for gi, it in enumerate(items):
        atask, ids, aug, n_retry = RT.augment_group(args, tasks(it.path), it, stok, step, gi)
        gtasks.append(atask)
        prompts.append(ids)
        gaugs.append(aug)
        aug_retries += n_retry
    pre_ids, pre_meta = [[] for _ in items], [None for _ in items]
    for p in pre:
        items.append(p["item"])
        taken.append(p["pool_index"])
        gtasks.append(p["task"])
        prompts.append(p["prompt"])
        gaugs.append(p["aug"])
        pre_ids.append(p["prefix"])
        pre_meta.append(p["meta"])
    t_aug = time.time() - t0
    t0 = time.time()
    topk_step_dir = None
    if args.sc_topk:
        topk_step_dir = os.path.join(args.topk_dir, f"step_{step:06d}")
        os.makedirs(topk_step_dir, exist_ok=True)
    ns = [args.group_size] * len(items)
    scored, scored_f, scorer, extra = {}, {}, None, {}
    force_on = args.train_budget_force > 0 and step >= args.train_budget_force_from
    inline = force_on and bool(getattr(args, "train_force_inline", 0))
    f_suffix, f_suf_ids = force_suffix(stok) if force_on else (None, None)
    if getattr(args, "reward_stream", 0) > 0:
        # score each sample as it arrives; (group, j) -> future of (TokenText, scores)
        scorer = cf.ThreadPoolExecutor(max_workers=args.reward_stream, thread_name_prefix="reward")

        def on_sample(i, j, rec):
            f = rec.get("forced")
            if f is not None:        # --train-force-inline: its forced answer came along (scored as force_answers')
                scored_f[(i, j)] = scorer.submit(score_forced, gtasks[i], rec["token_ids"], f_suf_ids, f["token_ids"],
                                                 f["finish"], aligner, cfg)
            if i < len(pre_ids) and pre_ids[i]:          # prefix group: scored on prefix + continuation
                rec = dict(rec, token_ids=list(pre_ids[i]) + list(rec["token_ids"]))
            scored[(i, j)] = scorer.submit(score_sample, gtasks[i], rec, aligner, cfg)
        extra["on_sample"] = on_sample
    gen_params = {"n": ns[0], "temperature": args.temperature, "top_p": 1.0, "top_k": 0, "logprobs": True,
                  "topk": args.sc_topk}
    if inline:                                           # --train-force-inline: the engines force cut rollouts at once
        extra["force"] = dict(suffix_ids=list(f_suf_ids), budget=args.train_budget_force,
                              max_model_len=args.sampler_max_model_len, params=force_params(args), seed_salt=args.seed + 13)

    def caps(ps):
        return [min(args.max_new_tokens, args.sampler_max_model_len - len(p)) for p in ps]

    def caps_all():
        """Per prompt: a prefix group's continuation gets the rollout budget minus its prefix."""
        return [min(args.max_new_tokens - len(pre_ids[i]), args.sampler_max_model_len - len(p))
                for i, p in enumerate(prompts)]

    def seeds_for(lo, hi):
        return [oc.request_seed(args.seed, step, i, args.group_size) for i in range(lo, hi)] \
            if args.per_request_seeds else None

    dyn, rstore, log_fn = None, None, _RUN_LOG["fn"]
    try:
        if getattr(args, "dyn_target_groups", 0) > 0:
            # dynamic sampling: rounds of prompts are added into free engine slots and judged in dispatch order on
            # their streamed rewards; sampling stops once the judged rounds hold the target number of groups with
            # outcome variance (see sampler_dispatch.generate_rounds). Extra rounds are drawn,
            # augmented and seeded exactly like round 0 (global prompt index = seed / augmentation index).
            from arcsolver.rl.sampler_dispatch import HOLD
            cap = args.dyn_max_prompts or 4 * args.prompts_per_step
            dyn = dict(judged_prompts=0, judged_informative=0, rounds=[])

            def frac():
                return (dyn["judged_informative"] + 8 * args.dyn_frac_init) / (dyn["judged_prompts"] + 8)

            def more():
                nonlocal aug_retries
                n_now = len(prompts)
                expected = dyn["judged_informative"] + (n_now - dyn["judged_prompts"]) * frac()
                if expected >= args.dyn_target_groups * args.dyn_margin:
                    return HOLD
                k = min(args.dyn_round_prompts, cap - n_now)
                if k <= 0:
                    return None
                for idx in order.take_more(k):              # same step: no schedule advance, no repeats
                    it = pool[idx]
                    atask, ids, aug, n_retry = RT.augment_group(args, tasks(it.path), it, stok, step, len(prompts))
                    items.append(it)
                    taken.append(idx)
                    gtasks.append(atask)
                    prompts.append(ids)
                    gaugs.append(aug)
                    ns.append(args.group_size)
                    pre_ids.append([])
                    pre_meta.append(None)
                    aug_retries += n_retry
                return dict(prompts=prompts[n_now:], max_tokens=caps(prompts[n_now:]), seeds=seeds_for(n_now, len(prompts)),
                            n_per_prompt=ns[n_now:])

            def on_round(r, outs):
                lo = dyn["judged_prompts"]
                inf = 0
                for i in range(lo, lo + len(outs)):
                    vals = [scored[(i, j)].result()[1]["r_seq"] for j in range(ns[i])]
                    inf += max(vals) > min(vals)
                dyn["judged_prompts"] += len(outs)
                dyn["judged_informative"] += inf
                dyn["rounds"].append((len(outs), inf))
                return dyn["judged_informative"] >= args.dyn_target_groups

            res = sampler.call("generate_rounds", prompts=list(prompts), topk_dir=topk_step_dir, tag=f"s{step}",
                               n_per_prompt=list(ns), params=gen_params, max_tokens=caps(prompts),
                               seeds=seeds_for(0, len(prompts)), more=more, on_round=on_round,
                               heartbeat_s=args.dyn_heartbeat_s, **extra)
            n_incl = len(res["outputs"])                # whole rounds 0..stop: drop the aborted / unjudged ones
            for lst in (items, taken, gtasks, prompts, gaugs, ns):
                del lst[n_incl:]
        else:
            rstore = open_rollout_store(args, step, sampler, prompts, ns, caps_all(), seeds_for(0, len(prompts)),
                                        gen_params, log_fn)
            if rstore is not None:
                res = generate_durable_step(sampler, rstore, prompts, ns, caps_all(), seeds_for(0, len(prompts)),
                                            gen_params, topk_step_dir, f"s{step}", extra.get("on_sample"), log_fn,
                                            force=extra.get("force"))
            else:
                res = sampler.call("generate", prompts=prompts, topk_dir=topk_step_dir, tag=f"s{step}", n_per_prompt=ns,
                                   params=gen_params, max_tokens=caps_all(), seeds=seeds_for(0, len(prompts)), **extra)
    except BaseException:
        if scorer is not None:
            scorer.shutdown(wait=False, cancel_futures=True)
        if rstore is not None:
            rstore.close()
        raise
    t_sample = time.time() - t0
    if len(res["outputs"]) != len(items) or any(len(samples) != n for samples, n in zip(res["outputs"], ns)):
        raise RuntimeError("sampler returned the wrong number of prompt groups or samples")
    t0 = time.time()
    rollouts, n_mism, n_streamed = [], 0, 0
    try:
        for gi, (it, samples) in enumerate(zip(items, res["outputs"])):
            task = gtasks[gi]
            P = len(pre_ids[gi])
            for j, s in enumerate(samples):
                check_behaviour_logprobs(s, sampler)
                fut = scored.get((gi, j))
                if fut is not None:                      # --reward-stream: scored while the sampler ran
                    tt, sc = fut.result()
                    n_streamed += 1
                else:
                    tt, sc = score_sample(task, s if not P else dict(s, token_ids=list(pre_ids[gi]) + list(s["token_ids"])),
                                          aligner, cfg)
                sc = shift_dense(sc, P)                  # prefix groups: rewards on prefix + continuation, tokens after it
                n_mism += (not P) and bool(s["text"]) and tt.text != s["text"]     # engines may return no text
                rollouts.append({"item": it, "group": gi, "completion": s["token_ids"], "sampler_lp": s["logprobs"],
                                 "policy": s.get("policy"),
                                 "prompt_ids": prompts[gi], "aug": gaugs[gi], "finish": s["finish"], "tt": tt,
                                 "seen": it.seen, "topk_path": s.get("topk_path"), "pool_index": taken[gi],
                                 "test_idx": task.get("_test_index"),
                                 # resolved here: the scorer pool is shut down (cancelling queued work) below
                                 **({"_forced": s["forced"], "_forced_sc": scored_f[(gi, j)].result()
                                     if (gi, j) in scored_f else None} if s.get("forced") else {}),
                                 **({"forced_skipped": True} if s.get("forced_skipped") else {}),
                                 **({"prefix": pre_meta[gi]} if P else {}), **sc})
    except BaseException:
        if rstore is not None:                       # flush the saved samples before the error ends the process
            rstore.close()
        raise
    finally:
        if scorer is not None:                       # also on a scoring or check failure
            scorer.shutdown(wait=True, cancel_futures=True)
    gids = [r["group"] for r in rollouts]
    t_force, forced_mask = 0.0, [False] * len(rollouts)
    if force_on:
        t1 = time.time()
        try:
            if inline:      # answers the engines attached; the separate pass only for samples that came without one
                fans = inline_forced_answers(rollouts, f_suffix, f_suf_ids)
                rest = {j for j, r in enumerate(rollouts) if j not in fans and not r.get("forced_skipped")}
                if rest:
                    fans.update(force_answers(args, step, sampler, stok, rollouts, rstore, only=rest))
            else:
                for r in rollouts:                         # (samples restored from a store drawn with inline forcing)
                    r.pop("_forced", None)
                    r.pop("_forced_sc", None)
                fans = force_answers(args, step, sampler, stok, rollouts, rstore)
            for j, (fids, ffinish, suffix, suf_ids) in fans.items():
                r = rollouts[j]
                fsc = r.pop("_forced_sc", None) or score_forced(gtasks[r["group"]], r["completion"], suf_ids, fids,
                                                                ffinish, aligner, cfg)
                r["forced_outcome"] = fsc["seq"]["outcome"]
                r["forced_finish"] = ffinish
                r["forced_text"] = suffix + aligner(list(fids)).text
            for r in rollouts:
                r["r_seq_natural"] = r["r_seq"]
            combined, forced_mask = rr.forced_credit_rewards([r["r_seq"] for r in rollouts],
                                                             [r.get("forced_outcome") for r in rollouts], gids,
                                                             args.forced_credit)
            for r, v, m in zip(rollouts, combined, forced_mask):
                r["forced_credit"] = m
                r["r_seq"] = v                                 # a right forced answer counts for the group statistics
        except BaseException:
            if rstore is not None:                   # flush the saved samples / forced answers before exiting
                rstore.close()
            raise
        t_force = time.time() - t1
    if rstore is not None:
        rstore.close()                                     # every saved sample / forced answer is on the volume now
    for r in rollouts:
        r["r_outcome"] = r["r_seq"]                        # before length shaping (--drop-zero-var outcome)
    if args.len_abs_alpha:
        scaled = rr.length_reward_absolute([r["r_seq"] for r in rollouts], [len(r["completion"]) for r in rollouts],
                                           args.len_abs_alpha, args.len_abs_budget)
        for r, v in zip(rollouts, scaled):
            r["r_seq"] = v
    if args.len_correct_alpha:
        scaled = rr.length_scaled_rewards([r["r_seq"] for r in rollouts], [len(r["completion"]) for r in rollouts],
                                          gids, args.len_correct_alpha)
        for r, v in zip(rollouts, scaled):
            r["r_seq"] = v
    overlong = None
    if getattr(args, "len_overlong_start", 0):
        # prefix groups: the response is prefix + continuation, against the full cap
        lens = [len(r["completion"]) + (len(pre_ids[r["group"]]) if r.get("prefix") else 0) for r in rollouts]
        vals, n_pen, g_pen, s_pen = rr.overlong_penalty(
            [r["r_seq"] for r in rollouts], lens, [r["finish"] == "length" for r in rollouts],
            [r["r_outcome"] for r in rollouts], gids, args.max_new_tokens, args.len_overlong_start,
            args.len_overlong_p0, args.len_overlong_p1, bool(args.len_overlong_mixed_only))
        for r, v in zip(rollouts, vals):
            r["overlong_pen"] = r["r_seq"] - v
            r["r_seq"] = v
        overlong = {"overlong/rollouts": n_pen, "overlong/groups": g_pen, "overlong/penalty_sum": s_pen,
                    "overlong/penalty_mean": s_pen / max(1, n_pen)}
    seq_adv = rr.group_advantages([r["r_seq"] for r in rollouts], gids, args.adv_norm, args.adv_eps,
                                  floor=args.adv_floor)
    n_share = 0
    if args.len_share_lambda:
        before = seq_adv
        seq_adv = rr.length_share_positive(seq_adv, [0.0 if m else r["r_seq"] for r, m in zip(rollouts, forced_mask)],
                                           [len(r["completion"]) for r in rollouts], gids, args.len_share_lambda,
                                           args.len_abs_budget)
        n_share = len({g for g, a0, a1 in zip(gids, before, seq_adv) if a0 != a1})
    allc = set()
    if args.len_allcorrect_beta:
        seq_adv, allc = rr.length_advantage_all_correct(seq_adv, [r["r_outcome"] for r in rollouts],
                                                        [len(r["completion"]) for r in rollouts], gids,
                                                        args.len_allcorrect_beta, args.w_outcome,
                                                        args.len_allcorrect_floor)
    pair_stats = {}
    if getattr(args, "len_pair_lambda", 0):
        seq_adv, pair_allright, pair_stats = rr.pairwise_length_shaping(
            seq_adv, [r["seq"]["outcome"] for r in rollouts], [len(r["completion"]) for r in rollouts], gids,
            args.len_pair_lambda, args.len_pair_min_correct, args.len_pair_delta, args.len_pair_span,
            args.len_pair_gamma, shape_allright=bool(getattr(args, "len_pair_allright", 1)))
        allc = set(allc) | pair_allright                   # all-right groups with a length signal are kept
    if any(forced_mask):
        seq_adv = rr.scale_forced_advantages(seq_adv, forced_mask, args.forced_credit, gids,
                                             [r["r_seq_natural"] for r in rollouts])
    dropped = dropped_groups(args.drop_zero_var, rollouts, gids) - allc    # all-correct groups with a length signal stay
    for r, a_ in zip(rollouts, seq_adv):
        r["dropped"] = r["group"] in dropped
        r["seq_adv"] = 0.0 if r["dropped"] else a_
    n_trunc_masked = mask_truncated_rollouts(rollouts) if args.mask_truncated else (0, 0)
    n_trunc_scaled = (scale_truncated_advantages(rollouts, args.trunc_neg_scale) if args.trunc_neg_scale != 1.0
                      else (0, 0))
    kept = kept_rollouts(rollouts, allc)
    measured = None
    if args.pool_file:
        rated = [it.rate for it in items if it.rate is not None]
        measured = sum(rated) / max(1, len(rated))
    times = {"aug": t_aug, "sample": t_sample, "reward": time.time() - t0 - t_force, "n_mism": n_mism,
             "measured": measured, "reward_streamed": n_streamed, "dyn": dyn,
             "force": t_force, "forced": forced_stats(rollouts, forced_mask) if force_on else None,
             "len_share_groups": n_share, "len_allcorrect_groups": len(allc), "allc_groups": allc, "lenpair": pair_stats,
             "trunc_masked": n_trunc_masked, "trunc_scaled": n_trunc_scaled, "overlong": overlong}
    return rollouts, kept, gaugs, prompts, res, times, aug_retries


def merge_split_scores(scores: list) -> dict:
    """One task's score from the arc_render.score_task results of its per-test single-test prompts (--eval-tests
    split), in test order: per_output concatenated, pass@k / vote@2 averaged over the outputs exactly as score_task
    averages them, sample counts (closed, well_formed, n_samples) summed over the prompts."""
    out = {"n_samples": sum(x["n_samples"] for x in scores), "n_test": len(scores),
           "closed": sum(x["closed"] for x in scores), "with_second_attempt": sum(x.get("with_second_attempt", 0) for x in scores),
           "well_formed": sum(x["well_formed"] for x in scores), "per_output": [x["per_output"][0] for x in scores]}
    for key in [k for k in scores[0] if k.startswith("pass@") or k == "vote@2"]:
        out[key] = sum(o.get(key, 0.0) for o in out["per_output"]) / len(scores)
    return out


def split_eval_prompts(args, stok, tasks, eval_items):
    """[(item index, test index, n_test, task view, prompt ids)]: a single-test task as itself (its usual prompt), a
    multi-test task as one single-test view per test input (all training pairs kept)."""
    out = []
    for ii, it in enumerate(eval_items):
        task = tasks(it.path)
        nt = len(task["test"])
        if nt == 1:
            out.append((ii, 0, 1, task, it.student_ids))
            continue
        for j in range(nt):
            sub = dict(task)
            sub["test"] = [task["test"][j]]
            out.append((ii, j, nt, sub, oc.render_ids(stok, oc.student_messages(sub, args.arm))))
    return out


def model_fingerprint(d: str) -> str:
    """Cheap content fingerprint of a model dir: config + index, and each safetensors file's size, first and last MiB."""
    h = hashlib.sha256()
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if fn.endswith(".safetensors"):
            size = os.path.getsize(p)
            h.update(f"{fn}:{size}".encode())
            with open(p, "rb") as f:
                h.update(f.read(1 << 20))
                f.seek(max(0, size - (1 << 20)))
                h.update(f.read())
        elif fn in ("config.json", "model.safetensors.index.json"):
            with open(p, "rb") as f:
                h.update(fn.encode() + f.read())
    return h.hexdigest()[:24]


class EvalStore:
    """--eval-partial-dir: one eval pass's finished samples on durable storage. Each record is its own gzip JSON file
    <kind>_<prompt>_<sample>.json.gz (kind s = main sample, f = forced continuation), written by one background thread
    (tmp, fsync, rename: a preemption leaves at most a stray .tmp, never a torn record). MANIFEST.json pins what the
    records belong to; a different manifest is refused rather than mixed."""

    def __init__(self, root, name, manifest, log):
        import queue
        import threading
        self.name, self.dir, self.log = name, os.path.join(root, name), log
        os.makedirs(self.dir, exist_ok=True)
        mf = os.path.join(self.dir, "MANIFEST.json")
        if os.path.isfile(mf):
            with open(mf) as f:
                old = json.load(f)
            if old != manifest:
                bad = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
                raise SystemExit(f"--eval-partial-dir {self.dir}: MANIFEST.json differs from this eval in {bad}; "
                                 f"use a fresh directory")
        else:
            self._write(mf, manifest, gz=False)
        self.q, self.errors = queue.Queue(), []
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    @staticmethod
    def _write(path, obj, gz=True):
        import gzip
        tmp = path + ".tmp"
        data = json.dumps(obj).encode()
        with open(tmp, "wb") as raw:
            raw.write(gzip.compress(data, compresslevel=3) if gz else data)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self._write(*item)
            except Exception as e:                       # reported (and fatal) at close()
                self.errors.append(f"{item[0]}: {e!r}")

    def put(self, kind, i, j, obj):
        self.q.put((os.path.join(self.dir, f"{kind}_{i:05d}_{j:03d}.json.gz"), obj))

    def load(self, kind):
        import glob
        import gzip
        out = {}
        for fn in glob.glob(os.path.join(self.dir, f"{kind}_*.json.gz")):
            try:
                with gzip.open(fn, "rt") as f:
                    obj = json.load(f)
                out[(int(obj["i"]), int(obj["j"]))] = obj
            except (OSError, ValueError, KeyError, EOFError) as e:
                self.log(f"eval store {self.name}: ignoring unreadable {os.path.basename(fn)} ({e!r})")
        return out

    def close(self):
        self.q.put(None)
        self.thread.join()
        if self.errors:
            raise RuntimeError(f"eval store {self.dir}: {len(self.errors)} writes failed, e.g. {self.errors[:2]}")


def eval_store_manifest(args, tag, model_dir, policy, psha, n, seed_base, forced_seed_base, lb):
    """What a resumable eval pass's saved samples are valid for (`policy` = the committed policy identity of the
    engines, or the full manifest digest of a static model dir; the logit bias by its full map's hash)."""
    from arcsolver.rl import identity as RI
    return {"tag": tag, "model": os.path.abspath(model_dir), "model_fingerprint": policy,
            "prompts_sha": hashlib.sha256("".join(psha).encode()).hexdigest()[:24], "n_prompts": len(psha),
            "n": n, "temperature": args.eval_temperature, "top_p": args.eval_top_p, "top_k": args.eval_top_k,
            "max_tokens": args.eval_max_tokens, "max_model_len": args.sampler_max_model_len,
            "budget_force": args.eval_budget_force, "seed_base": seed_base, "forced_seed_base": forced_seed_base,
            "logit_bias": RI.logit_bias_sha(lb)}


def make_eval(args, sampler, stok, tasks, eval_items, emit, log):
    """The periodic eval with the multi-engine sampler (fixed eval set, eval sampling settings, optional budget forcing).
    --eval-tests: joint (one prompt per task with all its test inputs), split (one prompt per test input of a
    multi-test task, merged per task; files evals/step_N.split.*, metrics eval_split/...), or both; with split,
    --eval-joint-steps adds the joint pass at those steps."""

    aligner = rr.TokenAligner(stok)
    joint_steps = {int(x) for x in str(getattr(args, "eval_joint_steps", "") or "").split(",") if x.strip()}
    lb = getattr(args, "eval_logit_bias_map", None)
    bias = {"logit_bias": lb} if lb else {}               # --eval-logit-bias: every eval generation, main + forced
    if lb:
        log(f"eval logit bias: {len(lb)} tokens, bias {min(lb.values()):g}..{max(lb.values()):g} "
            f"({args.eval_logit_bias})")

    def group_of(it):
        return "arc_eval" if it.source == "arc_eval" else ("bank_seen" if it.seen else "bank_unseen")

    partial_dir = getattr(args, "eval_partial_dir", "") or ""
    if partial_dir and not getattr(sampler, "stream_seqs", 0):
        raise SystemExit("--eval-partial-dir needs streaming dispatch (--sampler-stream-seqs > 0)")
    fingerprint = {}

    def generate_durable(prompts, seed_base, forced_seed_base, tag):
        """generate() with every finished sample on durable storage (--eval-partial-dir): the same seeds (sample j of
        prompt i = request seed + j, as streaming dispatch gives it), caps and forcing; a rerun loads the saved samples
        and generates only the rest. Forced continuations are seeded per sample (forced_seed_base + i * n + j) rather
        than by their position in the forced list, so a rerun gives a saved sample the same seed. --eval-force-inline:
        the engines attach a cut sample's forced answer (seeded from the sample's seed), saved with the sample; the
        separate forced pass then covers only samples without one (e.g. restored from a pass drawn without it)."""
        from array import array
        n = args.eval_n
        model_dir = args.student if os.path.isdir(str(args.student)) else args.sampler_model
        from arcsolver.rl import identity as RI
        pol = getattr(sampler, "committed_policy", None)            # what the engines serve now, not the base dir
        if pol is None and model_dir not in fingerprint:
            fingerprint[model_dir] = RI.dir_manifest(model_dir)["sha256"]   # every byte, not the first/last MiB
        psha = [hashlib.sha1(array("i", [int(t) for t in p]).tobytes()).hexdigest()[:16] for p in prompts]
        manifest = eval_store_manifest(args, tag, model_dir, pol if pol is not None else fingerprint[model_dir], psha,
                                       n, seed_base, forced_seed_base, lb)
        store = EvalStore(partial_dir, tag, manifest, log)
        params = {"n": 1, "temperature": args.eval_temperature, "top_p": args.eval_top_p, "top_k": args.eval_top_k,
                  "logprobs": False, **bias}
        seeds = [oc.request_seed(seed_base, 0, i, n) for i in range(len(prompts))]
        caps = [min(args.eval_max_tokens, args.sampler_max_model_len - len(p)) for p in prompts]

        def valid(v, i, seed):
            return 0 <= i < len(prompts) and v.get("psha") == psha[i] and v.get("seed") == seed

        fextra = {}
        if args.eval_budget_force > 0 and getattr(args, "eval_force_inline", 0):
            fextra["force"] = dict(suffix_ids=list(force_suffix(stok)[1]), budget=args.eval_budget_force,
                                   max_model_len=args.sampler_max_model_len, params=params, seed_salt=forced_seed_base)

        outputs = [[None] * n for _ in prompts]
        for (i, j), v in store.load("s").items():
            if j < n and valid(v, i, seeds[i] + j if i < len(prompts) else None):
                outputs[i][j] = v["rec"]
        todo = sorted(((i, j) for i in range(len(prompts)) for j in range(n) if outputs[i][j] is None),
                      key=lambda x: (x[1], x[0]))       # sample-major, as streaming dispatch orders a batch
        log(f"eval {tag}: {len(prompts) * n - len(todo)} of {len(prompts) * n} samples restored from {store.dir}, "
            f"{len(todo)} to generate")
        if todo:
            def on_sample(e, _j, rec):
                i, j = todo[e]
                store.put("s", i, j, {"i": i, "j": j, "seed": seeds[i] + j, "psha": psha[i], "rec": rec})
            res = sampler.call("generate", prompts=[prompts[i] for i, _ in todo], params=params,
                               max_tokens=[caps[i] for i, _ in todo], seeds=[seeds[i] + j for i, j in todo],
                               on_sample=on_sample, **fextra)
            for (i, j), outs in zip(todo, res["outputs"]):
                outputs[i][j] = outs[0]
        forced = {}
        if args.eval_budget_force > 0:
            suffix = "\n</think>\n\n" if ar.default_channel() == "think" else "\n\n" + ar.ANSWER_MARKER + "\n"
            suf_ids = stok(suffix, add_special_tokens=False)["input_ids"]
            saved = store.load("f")
            frecs, fprompts, fkeys, fmax, fseeds, n_inline = {}, [], [], [], [], 0
            for i, (p, samples) in enumerate(zip(prompts, outputs)):
                for k, smp in enumerate(samples):
                    if smp["finish"] != "length":
                        continue
                    if smp.get("forced") is not None:        # --eval-force-inline: came with its sample
                        frecs[(i, k)] = smp["forced"]
                        n_inline += 1
                        continue
                    room = args.sampler_max_model_len - len(p) - len(smp["token_ids"]) - len(suf_ids)
                    if room < 64:
                        continue
                    fs = forced_seed_base + i * n + k
                    v = saved.get((i, k))
                    if v is not None and valid(v, i, fs):
                        frecs[(i, k)] = v["rec"]
                        continue
                    fprompts.append(list(p) + list(smp["token_ids"]) + suf_ids)
                    fkeys.append((i, k))
                    fmax.append(min(args.eval_budget_force, room))
                    fseeds.append(fs)
            log(f"eval {tag}: {n_inline} forced answers inline, {len(frecs) - n_inline} restored, "
                f"{len(fprompts)} to generate")
            if fprompts:
                def on_forced(e, _j, rec):
                    i, k = fkeys[e]
                    store.put("f", i, k, {"i": i, "j": k, "seed": fseeds[e], "psha": psha[i], "rec": rec})
                fres = sampler.call("generate", prompts=fprompts, params=params, max_tokens=fmax, seeds=fseeds,
                                    on_sample=on_forced)
                for key, outs in zip(fkeys, fres["outputs"]):
                    frecs[key] = outs[0]
            for key, rec in frecs.items():
                forced[key] = suffix + (rec["text"] or aligner(rec["token_ids"]).text)
        store.close()
        return {"outputs": outputs}, forced

    def generate(prompts, seed_base, forced_seed_base, tag=None):
        if partial_dir and tag:
            return generate_durable(prompts, seed_base, forced_seed_base, tag)
        fparams = {"n": 1, "temperature": args.eval_temperature, "top_p": args.eval_top_p,
                   "top_k": args.eval_top_k, "logprobs": False, **bias}
        inline = args.eval_budget_force > 0 and bool(getattr(args, "eval_force_inline", 0))
        extra = {}
        if inline:                                     # --eval-force-inline: answers queued as cut samples end
            suffix, suf_ids = force_suffix(stok)
            extra["force"] = dict(suffix_ids=list(suf_ids), budget=args.eval_budget_force,
                                  max_model_len=args.sampler_max_model_len, params=fparams, seed_salt=forced_seed_base)
        res = sampler.call("generate", prompts=prompts,
                           params={"n": args.eval_n, "temperature": args.eval_temperature,
                                   "top_p": args.eval_top_p, "top_k": args.eval_top_k, "logprobs": False, **bias},
                           max_tokens=[min(args.eval_max_tokens, args.sampler_max_model_len - len(p)) for p in prompts],
                           seeds=[oc.request_seed(seed_base, 0, i, args.eval_n) for i in range(len(prompts))], **extra)
        forced = {}
        if inline:
            for i, samples in enumerate(res["outputs"]):
                for k, smp in enumerate(samples):
                    f = smp.get("forced")
                    if f is not None:
                        forced[(i, k)] = suffix + (f.get("text") or aligner(f["token_ids"]).text)
        elif args.eval_budget_force > 0:
            suffix = "\n</think>\n\n" if ar.default_channel() == "think" else "\n\n" + ar.ANSWER_MARKER + "\n"
            suf_ids = stok(suffix, add_special_tokens=False)["input_ids"]
            fprompts, fkeys, fmax = [], [], []
            for i, (p, samples) in enumerate(zip(prompts, res["outputs"])):
                for k, smp in enumerate(samples):
                    if smp["finish"] == "length":
                        room = args.sampler_max_model_len - len(p) - len(smp["token_ids"]) - len(suf_ids)
                        if room >= 64:
                            fprompts.append(list(p) + list(smp["token_ids"]) + suf_ids)
                            fkeys.append((i, k))
                            fmax.append(min(args.eval_budget_force, room))
            if fprompts:
                fres = sampler.call("generate", prompts=fprompts, params=fparams,
                                    max_tokens=fmax, seeds=[forced_seed_base + j for j in range(len(fprompts))])
                for (i, k), outs in zip(fkeys, fres["outputs"]):
                    forced[(i, k)] = suffix + (outs[0]["text"] or aligner(outs[0]["token_ids"]).text)
        return res, forced

    def score_prompt(task, samples, forced, i, ks):
        texts = [s["text"] or aligner(s["token_ids"]).text for s in samples]   # engines may skip the tokenizer
        fsc, nf, ftexts = None, 0, None
        if forced:
            ftexts = [t + forced.get((i, k), "") for k, t in enumerate(texts)]
            fsc = ar.score_task(task, ftexts, ks=ks)
            nf = sum((i, k) in forced for k in range(len(texts)))
        sc = ar.score_task(task, texts, ks=ks)
        loops = [oc.loop_kind(t) for t in texts]
        lens = [len(s["token_ids"]) for s in samples]
        row = {"texts": texts, "gen_tokens": lens, "finish": [s["finish"] for s in samples], "loops": loops,
               "pass@1": sc.get("pass@1"), "pass@2": sc.get("pass@2"), "closed": sc["closed"],
               "correct": [o.get("correct") for o in sc["per_output"]],
               **({"forced": [forced.get((i, k)) for k in range(len(texts))]} if forced else {})}
        return sc, fsc, nf, loops, lens, row

    def run_joint(ks, tag=None):
        prompts = [it.student_ids for it in eval_items]
        res, forced = generate(prompts, args.seed + 7, args.seed + 11, tag)
        groups, rows, n_forced = {}, [], 0
        for ii, (it, samples) in enumerate(zip(eval_items, res["outputs"])):
            g = group_of(it)
            sc, fsc, nf, loops, lens, row = score_prompt(tasks(it.path), samples, forced, ii, ks)
            n_forced += nf
            if fsc is not None:
                FG = groups.setdefault("forced_" + g, {"scores": [], "loops": [], "lens": []})
                FG["scores"].append(fsc)
                FG["loops"] += ["none"] * len(loops)
                FG["lens"] += lens
            G = groups.setdefault(g, {"scores": [], "loops": [], "lens": []})
            G["scores"].append(sc)
            G["loops"] += loops
            G["lens"] += lens
            rows.append({"task_id": it.task_id, "group": g, **row})
        return groups, rows, n_forced

    def run_split(ks, tag=None):
        subs = split_eval_prompts(args, stok, tasks, eval_items)
        res, forced = generate([x[4] for x in subs], args.seed + 13, args.seed + 17, tag)
        per, rows, n_forced = {}, [], 0
        for si, ((ii, j, nt, task, _), samples) in enumerate(zip(subs, res["outputs"])):
            it = eval_items[ii]
            sc, fsc, nf, loops, lens, row = score_prompt(task, samples, forced, si, ks)
            n_forced += nf
            per.setdefault(ii, []).append((j, sc, fsc, loops, lens))
            rows.append({"task_id": it.task_id, "test_index": j, "n_test": nt, "group": group_of(it), **row})
        groups = {}
        for ii in sorted(per):
            parts = sorted(per[ii], key=lambda x: x[0])
            g = group_of(eval_items[ii])
            G = groups.setdefault(g, {"scores": [], "loops": [], "lens": []})
            G["scores"].append(merge_split_scores([x[1] for x in parts]))
            G["loops"] += [lk for x in parts for lk in x[3]]
            G["lens"] += [n for x in parts for n in x[4]]
            if parts[0][2] is not None:
                FG = groups.setdefault("forced_" + g, {"scores": [], "loops": [], "lens": []})
                FG["scores"].append(merge_split_scores([x[2] for x in parts]))
                FG["loops"] += ["none"] * sum(len(x[3]) for x in parts)
                FG["lens"] += [n for x in parts for n in x[4]]
        return groups, rows, n_forced

    def run_eval(step):
        ks = [k for k in (1, 2, 4, 8) if k <= args.eval_n]
        mode = getattr(args, "eval_tests", "joint")
        passes = []
        if mode in ("joint", "both") or (mode == "split" and step in joint_steps):
            passes.append(("eval", "", run_joint))
        if mode in ("split", "both"):
            passes.append(("eval_split", ".split", run_split))
        metrics = {}
        for pre, sfx, fn in passes:
            t0 = time.time()
            groups, rows, n_forced = fn(ks, f"step_{step:06d}{sfx}")
            summary = {}
            for g, G in sorted(groups.items()):
                gm = oc.eval_group_metrics(G["scores"], G["loops"], G["lens"], args.eval_n, ks)
                summary[g] = gm
                metrics.update({f"{pre}/{g}/{k}": v for k, v in gm.items()})
            metrics[f"time/{pre}_s"] = time.time() - t0
            if args.eval_budget_force > 0:
                metrics[f"{pre}/n_budget_forced"] = n_forced
            ot.write_jsonl_gz(os.path.join(args.out, "evals", f"step_{step:06d}{sfx}.jsonl.gz"), rows)
            ot.write_json(os.path.join(args.out, "evals", f"step_{step:06d}{sfx}.json"),
                          {"step": step, "summary": summary, **({"tests": "split"} if sfx else {})})
            log(f"{pre.replace('_', '-')} step {step} ({metrics[f'time/{pre}_s']:.0f}s): " + "; ".join(
                f"{g} pass@1 {100 * s.get('pass@1', 0):.1f} pass@2 {100 * s.get('pass@2', 0):.1f} "
                f"pass@{ks[-1]} {100 * s.get(f'pass@{ks[-1]}', 0):.1f} "
                f"closed {100 * s['closed_rate']:.0f}% loop {100 * s['loop_rate']:.0f}%" for g, s in summary.items()))
        emit(metrics, step)
        ot.write_json(os.path.join(args.out, "evals", f"step_{step:06d}.COMPLETE"), {"step": step})

    return run_eval


if __name__ == "__main__":
    main()
