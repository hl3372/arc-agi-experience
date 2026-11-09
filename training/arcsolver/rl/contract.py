"""Objective contracts for train.py.

A contract is the complete set of objective/workload arguments a run must have, so a launch cannot silently inherit a
parser default (config.py defaults to --beta-loop 0.3 and --drop-zero-var total, neither of which is GRPO on binary
correctness) or a historical launcher's shaping. --objective-contract NAME makes the trainer refuse any differing flag.

grpo_token_v1: vanilla GRPO with token-level loss
  - binary exact outcome, capped (non-EOS) responses count as wrong and stay in their groups, no forced-answer credit;
  - group-standardised advantages (sample std, ddof 1, + 1e-4), no clipping of advantages;
  - token mean over the completion tokens of every kept rollout; PPO clip 0.2 (symmetric) against the sampler's own
    behaviour log probabilities; one optimizer update per fresh rollout batch;
  - outcome zero-variance groups dropped; no loop / length / format shaping; no prefix or dynamic-sampling paths;
    split test prompts; per-request seeds.

grpo_dapo_kl_v1: grpo_token_v1 plus
  - reference KL: k3 per token, coefficient 0.001, in the same DAPO token mean (loss.kl_k3), reference = the start;
  - budget-forced answers for capped rollouts (2048-token answer budget): a right forced answer earns 0.5 of a
    natural success (rewards.forced_credit_rewards / scale_forced_advantages); the capped rollout's own reward is
    still 0 and it stays in its group; the forced tokens are scored, never trained;
  - one uniformly shuffled order over the whole pool (no tiers, focus lists, caps, strata or oversampling), 272 x 6
    rollouts per step, LR 2e-6 after 10 warmup steps; a task whose group is all correct is never drawn again.

The later contracts are listed below. The 9B GRPO recipe of the README is grpo_dapo_kl_sc_v1 followed by
grpo_dapo_kl_sc_prefix_v1.

The resolved objective (contract values + the workload/optimiser settings that change what a step trains) is stored in
every checkpoint's state.json and compared on resume: checkpoints do not restore the CLI, so a resume with different
flags would otherwise change the objective without notice.
"""
from __future__ import annotations

import hashlib
import json
import os

# attribute -> required value (a tuple = any of these values)
CONTRACTS = {
    "grpo_token_v1": {
        "adv_norm": "std", "adv_eps": 1e-4, "adv_clip": 0.0, "loss_agg": "token",
        "pg_loss": "ppo", "sc_topk": 0, "clip_eps": 0.2, "clip_eps_high": (None, 0.2),
        "drop_zero_var": "outcome", "minibatch_max": 1, "temperature": 1.0,
        "w_outcome": 1.0, "outcome_require_stop": 1, "w_format": 0.0,
        "beta_loop": 0.0, "w_loop_seq": 0.0, "w_len": 0.0,
        "len_abs_alpha": 0.0, "len_correct_alpha": 0.0, "len_share_lambda": 0.0, "len_allcorrect_beta": 0.0,
        "len_pair_lambda": 0.0,
        "mask_truncated": 0, "trunc_neg_scale": 1.0, "train_budget_force": 0,
        "dyn_target_groups": 0, "prefix_groups": 0, "prefix_extra_prompts": 0,
        "train_force_inline": 0, "train_all_rollouts": False,
        "train_split_tests": True, "per_request_seeds": True,
    },
}
CONTRACTS["grpo_dapo_kl_v1"] = dict(
    CONTRACTS["grpo_token_v1"], kl_coef=0.001, train_budget_force=2048, train_budget_force_from=0, forced_credit=0.5,
    pool_order="weighted", bank_share=0.0, pool_oversample=1, pool_file=(None, ""), pool_focus_file="",
    pool_focus_per_step=0, pool_new_per_step=0,
    prompts_per_step=272, group_size=6, lr=2e-6, warmup_steps=10, grad_clip=1.0, adam_eps=1e-8,
    pool_drop_solved=1,             # 272 x 6 (17 prompts per engine on 16 GPUs); a 6/6 task is never drawn again
    eval_temperature=1.0, eval_top_p=1.0, eval_top_k=0)             # the forced answers sample with the eval settings
# grpo_dapo_kl_v2: v1 without forced answers (they were right on only 1-3% of capped rollouts, rescuing 0-1 groups
# per step, for ~2 min per step). Capped rollouts keep reward 0 in their groups.
CONTRACTS["grpo_dapo_kl_v2"] = {k: v for k, v in CONTRACTS["grpo_dapo_kl_v1"].items()
                                if k not in ("forced_credit", "eval_temperature", "eval_top_p", "eval_top_k")}
CONTRACTS["grpo_dapo_kl_v2"].update(train_budget_force=0, train_budget_force_from=0)
# grpo_dapo_kl_sc_v1: score centering + TIS instead of the PPO ratio against the FP8 sampler (whose 0.2 clip zeroed
# ~5.5% of tokens per step purely from the sampler/trainer mismatch): v2 with --pg-loss tis_sc, the sampler's top-32 for
# the centering, TIS cap 2 (sc.sc_tis_loss; Marek & Ryabinin, Eq. 11-12); KL and the DAPO token mean unchanged.
CONTRACTS["grpo_dapo_kl_sc_v1"] = {k: v for k, v in CONTRACTS["grpo_dapo_kl_v2"].items()
                                   if k not in ("clip_eps", "clip_eps_high")}
CONTRACTS["grpo_dapo_kl_sc_v1"].update(pg_loss="tis_sc", sc_topk=32, tis_cap=2.0)

# grpo_dapo_kl_sc_prefix_v1: sc_v1 plus prefix groups on even steps, sized like the 4B recipe (24 of 96 -> 68 of 272,
# replacing fresh draws): sources from the previous step's not-all-right groups, at most 11 from 0-right groups (cut in
# their first half), cuts in [2.5k in, 10k before </think>] with 80% in the first 60% of the reasoning, snapped back to
# a newline; 6 continuations rewarded on prefix + continuation, trained on the continuation only.
CONTRACTS["grpo_dapo_kl_sc_prefix_v1"] = {k: v for k, v in CONTRACTS["grpo_dapo_kl_sc_v1"].items()
                                          if k not in ("prefix_groups", "prefix_extra_prompts")}
CONTRACTS["grpo_dapo_kl_sc_prefix_v1"].update(
    prefix_groups=68, prefix_every=2, prefix_from=0, prefix_extra_prompts=0, prefix_max_zero=11,
    prefix_zero_max_frac=0.5, prefix_early_frac=0.6, prefix_early_prob=0.8, prefix_min_start=2500,
    prefix_end_margin=10000)           # cuts end at least 10k tokens before </think> (the 4B recipe used 3k)

# grpo_dapo_kl_sc_prefix_v2 (experimental; tried after later evals fell with ~10% of samples at the cap): prefix_v1
# plus a DAPO-style soft overlong punishment in mixed groups only: 0 up to 80% of the cap, then -0.05 rising linearly to
# -0.2 at the cap (truncated rollouts: -0.2), on every rollout of a group with mixed natural outcomes.
CONTRACTS["grpo_dapo_kl_sc_prefix_v2"] = dict(CONTRACTS["grpo_dapo_kl_sc_prefix_v1"], len_overlong_start=0.8,
                                              len_overlong_p0=0.05, len_overlong_p1=0.2, len_overlong_mixed_only=1)

# grpo_dapo_kl_sc_prefix_v3 (experimental): prefix_v2 with the LR ramped linearly from 2e-6 to 4e-6 over steps 40-49,
# 4e-6 from step 49 on.
CONTRACTS["grpo_dapo_kl_sc_prefix_v3"] = dict(CONTRACTS["grpo_dapo_kl_sc_prefix_v2"], lr_ramp_to=4e-6, lr_ramp_from=39,
                                              lr_ramp_steps=10)

# grpo_dapo_clip_v1 (experimental, untested beyond one step): prefix_v2 with the PPO clipped surrogate instead of TIS +
# score centering, DAPO clip-higher (0.2 low / 0.28 high) against the bf16 trainer's own log-probs at the start of the
# step (--ppo-old-logp trainer, not the FP8 sampler's), up to 2 minibatch updates per step; AdamW with beta2 0.99 and
# decoupled weight decay 0.01 on the matrices; LR warmed up linearly over 20 steps to 1e-5. KL, the overlong punishment
# and the prefix groups as prefix_v2.
CONTRACTS["grpo_dapo_clip_v1"] = {k: v for k, v in CONTRACTS["grpo_dapo_kl_sc_prefix_v2"].items()
                                  if k not in ("tis_cap", "minibatch_max")}
CONTRACTS["grpo_dapo_clip_v1"].update(pg_loss="ppo", sc_topk=0, clip_eps=0.2, clip_eps_high=0.28, minibatch_max=2,
                                      ppo_old_logp="trainer", lr=1e-5, warmup_steps=20, adam_beta2=0.99,
                                      weight_decay=0.01)

# besides the contract keys, what a resume must not change silently (the optimiser and the rollout shape)
RESUME_KEYS = ("prompts_per_step", "group_size", "max_new_tokens", "lr", "warmup_steps", "adam_eps", "grad_clip",
               "freeze_embed", "seed", "augment", "aug_colors", "aug_identity_prob", "aug_original_prob",
               "outcome_all_tests", "pool_order", "pool_file", "pool", "student", "sampler_model", "sampler_fp8",
               "kv_cache_dtype", "mtp_tokens", "objective_contract", "ref_model", "kl_coef", "forced_credit",
               "train_budget_force", "pool_drop_solved", "pg_loss", "sc_topk", "tis_cap", "adam_beta2", "weight_decay",
               "ppo_old_logp")


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def check_contract(args, name: str) -> list:
    """Violations of contract `name` as readable strings (empty = the args satisfy it)."""
    if name not in CONTRACTS:
        raise SystemExit(f"unknown --objective-contract {name!r} (known: {', '.join(sorted(CONTRACTS))})")
    bad = []
    for k, want in CONTRACTS[name].items():
        if not hasattr(args, k):
            bad.append(f"{k}: missing from the parsed arguments (the harness and contract disagree)")
            continue
        have = getattr(args, k)
        options = want if isinstance(want, tuple) else (want,)
        if not any(_same(have, w) for w in options):
            bad.append(f"{k}={have!r}, contract {name} requires {' or '.join(repr(w) for w in options)}")
    return bad


def objective_snapshot(args, reward_cfg=None) -> dict:
    """The resolved objective of this launch: every contract key of every contract, the resume keys and the reward
    config with its hash (stored in state.json, compared on resume)."""
    keys = sorted({k for c in CONTRACTS.values() for k in c} | set(RESUME_KEYS))
    snap = {"values": {k: getattr(args, k, None) for k in keys}}
    snap["values"]["force_suffix_env"] = os.environ.get("ARC_EVAL_FORCE_SUFFIX")   # the forced answers' suffix
    if reward_cfg is not None:
        rc = {k: (sorted(v.items()) if isinstance(v, dict) else v) for k, v in vars(reward_cfg).items()}
        snap["reward_config"] = json.loads(json.dumps(rc, default=str))
        snap["reward_config_sha256"] = hashlib.sha256(json.dumps(snap["reward_config"], sort_keys=True,
                                                                 default=str).encode()).hexdigest()
    return snap


def compare_snapshots(saved: dict, current: dict, allow=()) -> list:
    """Differences between a checkpoint's objective snapshot and this launch's, except the keys in `allow`."""
    allow = set(allow)
    diffs = []
    sv, cv = (saved or {}).get("values", {}), current.get("values", {})
    for k in sorted(set(sv) | set(cv)):
        if k in allow:
            continue
        if k not in sv or k not in cv:
            diffs.append(f"{k}: {'absent from the checkpoint' if k not in sv else 'absent now'}")
        elif not _same(sv[k], cv[k]):
            diffs.append(f"{k}: checkpoint {sv[k]!r}, now {cv[k]!r}")
    if "reward_config_sha256" not in allow and saved.get("reward_config_sha256") != current.get("reward_config_sha256"):
        diffs.append("reward_config: the reward configuration changed")
    return diffs


def check_resume(state: dict, current: dict, allow_spec: str) -> list:
    """Resume check: [] when the checkpoint's objective matches (or the differing keys are explicitly allowed).
    allow_spec: comma-separated keys allowed to differ; 'legacy' accepts a checkpoint saved without a snapshot."""
    allow = [a.strip() for a in (allow_spec or "").split(",") if a.strip()]
    saved = state.get("objective")
    if saved is None:
        return [] if "legacy" in allow else ["the checkpoint has no objective snapshot (saved before contracts): "
                                             "add --allow-objective-change legacy to resume it anyway"]
    return compare_snapshots(saved, current, allow)
