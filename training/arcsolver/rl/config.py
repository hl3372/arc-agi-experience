"""Command-line flags of the RL trainer (the base set; train.py adds the distributed / sampler / objective flags) and
the pieces of a step that do not need torch: the reward configuration, the prompt pool and its order, the per-group
task view (split tests, augmentation) and the per-slice metrics.

Pool: --pool arc_train|bank|mixed; --pool-file (measured success rates, see pool.load_pool_rates) + --pool-budget
pick the measured success per task; --pool-filter learnable keeps 0 < success < 1; --pool-oversample W
repeats those W times per epoch. ARC-AGI-2 evaluation tasks are never in the pool (checked by directory, id and grid
content).
"""
from __future__ import annotations

import argparse
import json
import os
import random

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import augment as ra
from arcsolver.rl import core as oc
from arcsolver.rl import data as ot
from arcsolver.rl import frontier as rf
from arcsolver.rl import mixorder as rmo
from arcsolver.rl import pool as rp
from arcsolver.rl import rewards as rr

DEFAULTS = dict(ot.DEFAULTS)
FRONTIER_PARAMS = ("w_frontier", "w_solved", "w_dead", "w_failed", "trunc_hi", "revisit_after")   # --frontier-*


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.add_argument
    a("--out", required=True)
    a("--run-name", default=None)
    for k, v in DEFAULTS.items():
        a("--" + k.replace("_", "-"), default=v)
    a("--arm", default=oc.ARM, choices=sorted(ar.ARMS))
    # pool
    a("--pool", choices=rp.POOLS, default="arc_train")
    a("--pool-file", default=None, help="measured success rates (optional): {task_id: rate}, {task_id: {\"success\": "
      "rate}} or {\"tasks\": {task_id: {\"budgets\": {\"<B>\": {\"success\": rate}}}}} (pool.load_pool_rates)")
    a("--pool-budget", type=int, default=0, help="budget key in --pool-file (0 = --max-new-tokens)")
    a("--pool-lo", type=float, default=0.0, help="--pool-filter learnable keeps pool-lo < rate < pool-hi")
    a("--pool-hi", type=float, default=1.0)
    a("--pool-filter", choices=["learnable", "positive", "none"], default="learnable",
      help="learnable = keep tasks with 0 < measured success < 1; positive = success > 0 (only with --pool-file)")
    a("--pool-unmeasured", choices=["drop", "keep"], default="drop")
    a("--pool-oversample", type=int, default=1, help="repeat learnable tasks W times per epoch")
    a("--bank-share", type=float, default=0.0,
      help="share of every step's prompts drawn from bank tasks, the rest from the other sources (stratified order; "
           "0 = draw from the whole pool as one)")
    a("--pool-order", choices=["weighted", "frontier", "frontier_mix"], default="weighted",
      help="weighted = seeded epoch-wise shuffle (with --pool-oversample / --bank-share); frontier = frontier.py: each "
           "step's tasks drawn by their last group's outcome, mixed groups preferred; "
           "frontier_mix = mixorder: fresh bank + revisits of past frontier tasks + fresh other sources per step, "
           "in --mix-* shares")
    a("--mix-frontier-share", type=float, default=0.20, help="frontier_mix: share of each step's prompts revisited")
    a("--mix-rest-share", type=float, default=0.05, help="frontier_mix: share drawn fresh from the non-bank sources")
    a("--mix-min-gap", type=int, default=5, help="frontier_mix: steps before a frontier task can be revisited")
    a("--mix-band", type=int, nargs=2, default=[1, 6], metavar=("LO", "HI"),
      help="frontier_mix: a task is frontier while its latest group had LO..HI correct (of --group-size)")
    a("--mix-schedule", default="",
      help="frontier_mix (opt-in): STEP:FRONTIER:REST knots, comma-separated, e.g. 1:0:0.5,50:0:0.1,51:0.2:0.1,"
           "100:0.2:0; each step's shares are interpolated linearly between knots and replace --mix-frontier-share / "
           "--mix-rest-share")
    a("--mix-max-age", type=int, default=0,
      help="frontier_mix: revisit only queue entries drawn at most this many steps ago (0 = no cap)")
    a("--frontier-w-frontier", type=float, default=rf.DEFAULTS["w_frontier"], help="draw weight after a mixed group")
    a("--frontier-w-solved", type=float, default=rf.DEFAULTS["w_solved"], help="draw weight after G/G correct")
    a("--frontier-w-dead", type=float, default=rf.DEFAULTS["w_dead"], help="draw weight after 0/G, mostly truncated")
    a("--frontier-w-failed", type=float, default=rf.DEFAULTS["w_failed"], help="draw weight after 0/G, mostly finished")
    a("--frontier-trunc-hi", type=float, default=rf.DEFAULTS["trunc_hi"],
      help="a 0/G group with at least this share of its samples truncated counts as dead")
    a("--frontier-revisit-after", type=int, default=rf.DEFAULTS["revisit_after"],
      help="steps over which a solved / dead / failed task's weight returns to the unseen weight (1.0)")
    # rollouts
    a("--prompts-per-step", type=int, default=16)
    a("--group-size", type=int, default=8)
    a("--max-new-tokens", type=int, default=12288)
    a("--max-prompt-tokens", type=int, default=12288, help="skip prompts longer than this (student tokens)")
    a("--temperature", type=float, default=1.0)
    a("--max-steps", type=int, default=200)
    a("--seed", type=int, default=0)
    a("--per-request-seeds", action="store_true")
    # augmentation (augment.py): one random dihedral x colour permutation per rollout group, applied to every grid
    a("--augment", type=int, default=1, help="1 = augment every rollout group's task (eval stays unaugmented)")
    a("--aug-colors", choices=list(ra.COLOR_MODES), default="nonzero",
      help="nonzero = permute colours 1-9, keep 0; all = permute 0-9; none = no colour permutation")
    a("--aug-identity-prob", type=float, default=-1.0,
      help="probability of the identity dihedral (< 0 = uniform over the 8, i.e. 1/8)")
    a("--aug-original-prob", type=float, default=0.0,
      help="probability that a rollout group sees its task exactly as given (no dihedral, no colour permutation); "
           "the other groups are augmented as usual")
    a("--aug-max-tries", type=int, default=8,
      help="redraw the dihedral when the augmented prompt exceeds --max-prompt-tokens (then fall back to identity)")
    # rewards
    a("--w-outcome", type=float, default=1.0)
    a("--outcome-require-stop", type=int, default=1, help="1 = outcome only if the sample ended by EOS within budget")
    a("--outcome-all-tests", type=int, default=0,
      help="1 = outcome 1 only if every test output is right; 0 = mean over the task's test outputs (as the pool file)")
    a("--w-format", type=float, default=0.0)
    a("--format-shape", choices=["output", "none"], default="output")
    a("--format-allow-copy", action="store_true", help="give the format reward to a copied test input too")
    a("--beta-loop", type=float, default=0.3, help="per-token penalty on the loop region")
    a("--len-soft", type=int, default=0, help="overlong penalty starts at this completion token (0 = off)")
    a("--len-cap", type=int, default=0, help="overlong penalty reaches -w_len per token here (default: max-new-tokens)")
    a("--w-len", type=float, default=0.0, help="overlong per-token penalty at the cap (ramps up from --len-soft)")
    a("--len-abs-alpha", type=float, default=0.0,
      help="variant A length reward: a rewarded rollout's reward is scaled by 1 - alpha * min(L / budget, 1); "
           "0 = off (exclusive with --len-correct-alpha)")
    a("--len-abs-budget", type=int, default=0,
      help="the length budget B of variant A and --len-share-lambda (0 = --max-new-tokens)")
    a("--len-correct-alpha", type=float, default=0.0,
      help="Kimi-style: correct rollouts' reward *= 1 - alpha * group-normalised length (shorter right answers win)")
    a("--len-share-lambda", type=float, default=0.0,
      help="DRPO-style: a group's successes share their summed advantage with weights exp((1 - L / B) / lambda), "
           "B = --len-abs-budget (shorter right answers get more, none goes negative, the total is unchanged); "
           "0 = off, exclusive with the other length rewards")
    a("--len-allcorrect-beta", type=float, default=0.0,
      help="ConciseR-style: all-correct groups are kept and a rollout shorter than its group's mean length gets "
           "advantage beta * (mean - L) / max(std, --len-allcorrect-floor); longer ones 0")
    a("--len-allcorrect-floor", type=int, default=1024, help="the std floor (tokens) of --len-allcorrect-beta")
    a("--eval-budget-force", type=int, default=0,
      help="> 0: eval samples cut at the token budget get the reasoning closed (s1 budget forcing) and this many "
           "tokens to answer; logged as forced_* metrics next to the raw ones")
    a("--w-loop-seq", type=float, default=0.0, help="sequence-level penalty if the reasoning loops")
    a("--loop-detect", choices=["templated", "verbatim"], default="templated",
      help="templated = digit-masked paragraph comparison (also finds every verbatim loop)")
    a("--loop-onset", choices=["sustained", "first_repeat"], default="sustained",
      help="sustained = core.loop_kind onset (>= 80%% repeats after it); first_repeat = the first repeated paragraph")
    # advantages / loss
    a("--adv-norm", choices=["std", "mean", "maxrl", "maxrl_floor"], default="std",
      help="std = GRPO; mean = Dr. GRPO; maxrl = MaxRL (arXiv 2602.02710): (R - mean) / mean per group; "
           "maxrl_floor = (R - mean) / max(mean, --adv-floor)")
    a("--adv-eps", type=float, default=1e-4)
    a("--adv-floor", type=float, default=0.25,
      help="--adv-norm maxrl_floor: the fixed floor m0 of the denominator (a success rate, not 2 / group size)")
    a("--drop-zero-var", choices=["total", "seq", "outcome", "off"], default="total",
      help="drop groups with zero reward variance: total (R_seq + dense), seq (R_seq), outcome (the outcome reward "
           "before length shaping, so all-correct groups go too), off")
    a("--adv-clip", type=float, default=0.0, help="clip A_t to [-c, c] (0 = off)")
    a("--loss-agg", choices=["token", "seq"], default="token")
    a("--clip-eps", type=float, default=0.2)
    a("--clip-eps-high", type=float, default=None,
      help="upper PPO clip bound 1 + this (DAPO clip-higher, e.g. 0.28); default symmetric --clip-eps")
    # optimisation
    a("--lr", type=float, default=1e-6)
    a("--warmup-steps", type=int, default=10)
    a("--adam-eps", type=float, default=1e-8)
    a("--grad-clip", type=float, default=1.0)
    # eval / checkpoints / logging (eval set: data.build_eval_set)
    a("--eval-every", type=int, default=25, help="0 = never")
    a("--no-eval-at-start", action="store_true")
    a("--eval-bank", type=int, default=40)
    a("--eval-arc", type=int, default=40)
    a("--eval-n", type=int, default=4)
    a("--eval-max-tokens", type=int, default=16384)
    a("--eval-temperature", type=float, default=0.6)
    a("--eval-top-p", type=float, default=0.95)
    a("--eval-top-k", type=int, default=20)
    a("--save-every", type=int, default=50)
    a("--keep-optimizer-states", type=int, default=1, help="DCP optimizer states kept for the newest K checkpoints")
    a("--keep-model-dirs", type=int, default=0,
      help="> 0: after a save, delete the model/ dirs of all but the newest N checkpoints (disk; state.json stays)")
    a("--no-resume", action="store_true")
    a("--new-pool-on-resume", action="store_true",
      help="resume the model and optimizer but start a fresh prompt order over the (changed) pool: for phase switches "
           "such as training tasks -> mixed -> bank")
    a("--wandb-project", default=os.environ.get("WANDB_PROJECT", "arc-rl"),
      help="W&B project (W&B logs only when WANDB_API_KEY is set and --no-wandb is absent)")
    a("--no-wandb", action="store_true")
    a("--probe-every", type=int, default=1)
    a("--save-rollouts", type=int, default=1)
    # engines / trainer
    a("--sampler-max-model-len", type=int, default=0, help="0 = max(32768, max-prompt-tokens + max-new-tokens + 16)")
    a("--sampler-max-num-seqs", type=int, default=256)
    a("--sampler-max-num-batched-tokens", type=int, default=8192)
    a("--sampler-enforce-eager", action="store_true")
    a("--sync-dir", default=None, help="where each step's weights are written for the engines (default "
      "/dev/shm/rl_sync_<run-name>)")
    a("--sdpa-mode", choices=["off", "causal"], default="causal",
      help="causal = a verified purely causal SDPA mask becomes is_causal=True (flash SDPA kernel; hf_utils)")
    a("--logprob-chunk", type=int, default=2048)
    a("--dry-run", action="store_true", help="build the pool and eval set, print them, exit (CPU only)")
    a("--chat-template", default="", help="jinja chat template overriding the tokenizer's")
    a("--reasoning-effort", default="", choices=["", "xhigh", "medium", "low"],
      help="reasoning effort passed to the chat template ('' = the template's default)")
    a("--think-prefix", default="", help="prefilled after '<think>\\n' in every prompt (e.g. '<alpha>\\n' for the tagged "
      "render style); the completion (and its loss) starts after it")
    args = ap.parse_args(argv)
    args.think_prefix = args.think_prefix.replace("\\n", "\n")   # a literal backslash-n from the shell = newline
    if not args.student:
        ap.error("--student (or $ARC_RL_STUDENT) is required: the HF model dir the policy starts from")
    args.run_name = args.run_name or os.path.basename(os.path.normpath(args.out))
    args.sync_dir = args.sync_dir or f"/dev/shm/rl_sync_{args.run_name}"
    if not args.sampler_max_model_len:
        args.sampler_max_model_len = max(32768, args.max_prompt_tokens + args.max_new_tokens + 16)
    args.pool_budget = args.pool_budget or args.max_new_tokens
    if args.group_size < 2:
        ap.error("--group-size must be >= 2 (group normalisation)")
    if not 0.0 <= args.bank_share < 1.0:
        ap.error("--bank-share must be in [0, 1)")
    if args.pool_order == "frontier":
        if args.bank_share or args.pool_oversample != 1:
            ap.error("--pool-order frontier draws from the whole pool by outcome history: no --bank-share strata, and "
                     "it would ignore --pool-oversample counts")
        p = frontier_params(args)
        if (min(p["w_frontier"], p["w_solved"], p["w_dead"], p["w_failed"]) < 0 or not 0 < p["trunc_hi"] <= 1
                or p["revisit_after"] < 1):
            ap.error("--frontier-w-* must be >= 0, --frontier-trunc-hi in (0, 1], --frontier-revisit-after >= 1")
    if args.pool_order == "frontier_mix":
        lo, hi = args.mix_band
        if (args.mix_frontier_share < 0 or args.mix_rest_share < 0 or args.mix_frontier_share + args.mix_rest_share >= 1
                or args.mix_min_gap < 1 or not 1 <= lo <= hi < args.group_size):
            ap.error("--mix-*: shares >= 0 summing below 1, --mix-min-gap >= 1, 1 <= LO <= HI < --group-size")
        try:
            sched = parse_mix_schedule(args.mix_schedule)
            rmo.FrontierMixOrder([1, 1], [[0], [1]], 0, schedule=sched, max_age=args.mix_max_age)
        except ValueError as e:
            ap.error(f"--mix-schedule / --mix-max-age: {e}")
    if not 0.0 <= args.aug_original_prob <= 1.0:
        ap.error("--aug-original-prob must be in [0, 1]")
    if sum(bool(x) for x in (args.len_abs_alpha, args.len_correct_alpha, args.len_share_lambda)) > 1:
        ap.error("--len-abs-alpha, --len-correct-alpha and --len-share-lambda are exclusive")
    if args.len_share_lambda < 0:
        ap.error("--len-share-lambda must be >= 0")
    if args.len_allcorrect_beta < 0 or args.len_allcorrect_floor <= 0:
        ap.error("--len-allcorrect-beta must be >= 0 and --len-allcorrect-floor > 0")
    args.len_abs_budget = args.len_abs_budget or args.max_new_tokens
    return args


def reward_config(args) -> rr.RewardConfig:
    return rr.RewardConfig(
        w_outcome=args.w_outcome, outcome_require_stop=bool(args.outcome_require_stop),
        outcome_all_tests=bool(args.outcome_all_tests), w_format=args.w_format,
        format_shape=args.format_shape, format_reject_copy=not args.format_allow_copy, beta_loop=args.beta_loop,
        w_loop_seq=args.w_loop_seq, loop_detect=args.loop_detect, loop_onset=args.loop_onset,
        len_soft=args.len_soft, len_cap=args.len_cap or args.max_new_tokens, w_len=args.w_len)


def build_pool(args, tok, tasks, holdout, log):
    """Pool items (prompt ids rendered, <= max-prompt-tokens), filtered by the pool file; writes pool.json.
    Returns (items, per-item repeat counts per epoch)."""
    need = {"bank": ["bank_dir"], "arc_train": ["arc_train_dir"], "mixed": ["bank_dir", "arc_train_dir"]}[args.pool]
    missing = [k for k in need + ["arc_eval_dir"] if not getattr(args, k) or not os.path.isdir(getattr(args, k))]
    if missing:
        raise SystemExit(f"--pool {args.pool} needs existing directories for: "
                         + ", ".join("--" + k.replace("_", "-") for k in missing)
                         + " (--arc-eval-dir: the ARC-AGI-2 evaluation tasks, kept out of the pool)")
    # no SFT cache (vanilla model): nothing counts as seen
    trained = (oc.trained_task_ids(args.stage2_cache) if args.stage2_cache and os.path.exists(args.stage2_cache)
               else set())
    items, report = rp.build_rl_pool(args.pool, args.bank_dir, args.arc_train_dir, args.arc_eval_dir, trained,
                                     holdout_ids=holdout, load=tasks)
    rates, frep = {}, None
    if args.pool_file:
        rates = rp.load_pool_rates(args.pool_file, args.pool_budget)
        items, frep = rp.filter_pool(items, rates, args.pool_filter, args.pool_unmeasured, lo=args.pool_lo,
                                     hi=args.pool_hi)
    kept, too_long = [], []
    for it in items:
        ids = oc.render_ids(tok, oc.student_messages(tasks(it.path), args.arm))
        it.n_prompt_tokens = len(ids)
        if len(ids) > args.max_prompt_tokens:
            too_long.append(it.task_id)
            continue
        it.student_ids = ids
        kept.append(it)
    if not kept:
        raise SystemExit("empty prompt pool")
    oc.check_generation_prompt(tok, kept[0].student_ids, "student")
    oc.assert_no_eval(kept, args.arc_eval_dir, load=tasks)
    counts = rp.oversample_counts(kept, rates, args.pool_oversample) if args.pool_oversample > 1 else [1] * len(kept)
    rate_of = {it.task_id: rates.get(it.task_id) for it in kept}
    for it in kept:
        it.rate = rate_of[it.task_id]            # measured (unaugmented) success, for the per-step comparison
    ot.write_json(os.path.join(args.out, "pool.json"),
                  {"report": report, "filter": frep, "too_long": too_long, "pool_file": args.pool_file,
                   "pool_budget": args.pool_budget,
                   "items": [dict(it.meta(), rate=rate_of[it.task_id], copies=c) for it, c in zip(kept, counts)]})
    measured = [r for r in rate_of.values() if r is not None]
    log(f"prompt pool ({args.pool}): {len(kept)} prompts ({sum(it.seen for it in kept)} seen in SFT); "
        + (f"pool file {args.pool_file} @ budget {args.pool_budget}: filter {args.pool_filter} -> {json.dumps(frep)}, "
           f"mean measured success {sum(measured) / max(1, len(measured)):.3f}; " if args.pool_file else "no pool file; ")
        + f"skipped {len(too_long)} over {args.max_prompt_tokens} tokens; {len(report['held_out'])} held out for eval; "
        f"{sum(counts)} slots per epoch (oversample {args.pool_oversample})")
    return kept, counts


def frontier_params(args) -> dict:
    return {k: getattr(args, "frontier_" + k) for k in FRONTIER_PARAMS}


def parse_mix_schedule(text: str):
    """'1:0:0.5,50:0:0.1' -> [(1, 0.0, 0.5), (50, 0.0, 0.1)]; '' -> None (constant --mix-* shares)."""
    if not text.strip():
        return None
    knots = []
    for part in text.split(","):
        bits = part.strip().split(":")
        if len(bits) != 3:
            raise ValueError(f"knot {part!r} is not STEP:FRONTIER:REST")
        knots.append((int(bits[0]), float(bits[1]), float(bits[2])))
    return knots


def make_order(args, pool, counts):
    """The prompt order: one WeightedOrder over the pool, or (--bank-share) a StratifiedOrder that gives bank tasks
    that share of every step's prompts and the other sources the rest, or (--pool-order frontier) a FrontierOrder
    over the pool (counts unused; train.py feeds it every step's group outcomes)."""
    if getattr(args, "pool_order", "weighted") == "frontier":
        return rf.FrontierOrder(len(pool), args.seed, args.group_size, **frontier_params(args))
    if getattr(args, "pool_order", "weighted") == "frontier_mix":   # the same bank / other strata as below
        bank = [i for i, it in enumerate(pool) if it.source == "bank"]
        rest = [i for i, it in enumerate(pool) if it.source != "bank"]
        if not bank or not rest:
            raise SystemExit(f"--pool-order frontier_mix needs bank and non-bank prompts ({len(bank)} / {len(rest)})")
        return rmo.FrontierMixOrder(counts, [bank, rest], args.seed, args.mix_frontier_share, args.mix_rest_share,
                                    args.mix_min_gap, *args.mix_band,
                                    schedule=parse_mix_schedule(getattr(args, "mix_schedule", "")),
                                    max_age=getattr(args, "mix_max_age", 0),
                                    new_per_step=getattr(args, "pool_new_per_step", 0))
    if not args.bank_share:
        return rp.WeightedOrder(counts, args.seed)
    bank = [i for i, it in enumerate(pool) if it.source == "bank"]
    rest = [i for i, it in enumerate(pool) if it.source != "bank"]
    if not bank or not rest:
        raise SystemExit(f"--bank-share {args.bank_share} needs bank and non-bank prompts "
                         f"(pool: {len(bank)} bank, {len(rest)} other)")
    return rp.StratifiedOrder(counts, [bank, rest], [args.bank_share, 1.0 - args.bank_share], args.seed)


def split_test_view(args, task: dict, step: int, gi: int):
    """--train-split-tests: the group of a multi-test task sees ONE of its tests, as a single-test task (all training pairs kept), the test drawn by a
    (run seed, step, group) RNG; `_test_index` / `_n_test` record which. Single-test tasks, or the flag off: unchanged.
    Returns (task, test index or None)."""
    tests = task.get("test", [])
    if not getattr(args, "train_split_tests", False) or len(tests) < 2:
        return task, None
    j = random.Random(((args.seed * 1_000_003 + step) * 10_007 + gi) * 31 + 4_242_421).randrange(len(tests))
    sub = dict(task)
    sub["test"], sub["_test_index"], sub["_n_test"] = [tests[j]], j, len(tests)
    return sub, j


def augment_group(args, task, it, stok, step, gi):
    """One rollout group's view of its task -> (task, prompt ids, Aug, retries). With --aug-original-prob the group
    sees the task exactly as given; otherwise a random dihedral x colour permutation, redrawn while the prompt is
    over --max-prompt-tokens, then the identity dihedral keeping the colours (same token count). With
    --train-split-tests a multi-test task is first reduced to one test (split_test_view)."""
    task, j = split_test_view(args, task, step, gi)
    base_ids = it.student_ids if j is None else oc.render_ids(stok, oc.student_messages(task, args.arm))
    if not args.augment:
        return task, base_ids, ra.IDENTITY, 0
    rng = ra.group_rng(args.seed, step, gi)
    if args.aug_original_prob > 0 and rng.random() < args.aug_original_prob:
        return task, base_ids, ra.IDENTITY, 0
    retries = 0
    aug = ra.sample_aug(rng, args.aug_colors, args.aug_identity_prob)
    for _ in range(args.aug_max_tries):
        atask = aug.task(task)
        ids = oc.render_ids(stok, oc.student_messages(atask, args.arm))
        if len(ids) <= args.max_prompt_tokens:
            return atask, ids, aug, retries
        retries += 1
        aug = ra.sample_aug(rng, args.aug_colors, args.aug_identity_prob)
    aug = ra.Aug("id", aug.perm)
    atask = aug.task(task)
    return atask, oc.render_ids(stok, oc.student_messages(atask, args.arm)), aug, retries


UNSOLVED_RATE = 0.011     # pool files may mark never-solved tasks with rate 0.01; such rates are not compared


def slice_metrics(rollouts) -> dict:
    """Exact-match fraction per slice of the step (bank / other tasks, original / augmented views) and, over the
    slice's tasks with a measured unaugmented rate (unsolved-share tasks excluded), its ratio to that rate."""
    picks = {"bank": lambda r: r["item"].source == "bank", "arc": lambda r: r["item"].source != "bank",
             "orig": lambda r: r["aug"].is_identity(), "aug": lambda r: not r["aug"].is_identity()}
    out = {}
    for name, pick in picks.items():
        rs = [r for r in rollouts if pick(r)]
        if not rs:
            continue
        out[f"slice/{name}/rollouts"] = len(rs)
        out[f"slice/{name}/em_frac"] = sum(r["stats"]["em_frac"] for r in rs) / len(rs)
        rated = [r for r in rs if getattr(r["item"], "rate", None) is not None and r["item"].rate > UNSOLVED_RATE]
        if rated:
            out[f"slice/{name}/vs_screen"] = (sum(r["stats"]["em_frac"] for r in rated)
                                              / sum(r["item"].rate for r in rated))
    return out


def slice_line(m: dict) -> str:
    parts = []
    for name in ("bank", "arc", "orig", "aug"):
        if f"slice/{name}/em_frac" in m:
            ratio = m.get(f"slice/{name}/vs_screen")
            parts.append(f"{name} {100 * m[f'slice/{name}/em_frac']:.1f}%"
                         + (f" (x{ratio:.2f} screen)" if ratio is not None else "")
                         + f" n={m[f'slice/{name}/rollouts']}")
    return " | ".join(parts)
