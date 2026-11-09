#!/bin/bash
# 4B "floored MaxRL" recipe (run "rl4b4b"; "rl4b3a" differs as noted below): one node, 8 GPUs, FP8 vLLM rollouts
# time-sharing the GPUs with the FSDP2 trainer. See arcsolver/rl/README.md.
#   A = (R - m) / max(m, floor) per group (MaxRL, arXiv 2602.02710, with a fixed floor), sequence-level aggregation,
#   TIS + score centering (top-32, cap 2), no KL, LR 1.5e-6 after 10 warmup steps, 110 tasks x 8 rollouts,
#   rollouts up to 147,000 tokens, split tests, prefix continuations every 3rd step, pairwise length shaping.
#
#   RUN=rl4b OUT_ROOT=/data/rl MODEL=/models/sft4b SAMPLER_MODEL=/models/sft4b_fp8 \
#   BANK_DIR=/data/pool/bank ARC_TRAIN_DIR=/data/pool/arc_train ARC_EVAL_DIR=/data/ARC-AGI-2/data/evaluation \
#   bash scripts/rl/rl4b_maxrl_floor.sh
#   DRY=1 ...    print the resolved command only
#
# rl4b3a variant: FLOOR=0.25 (0.2 from its step 85), TRUNC_SCALE=0.3, FORCE_BUDGET=2048 (a right forced answer on a
# capped rollout earns 0.5 of a success), LR=1e-6, 96 prompts, a rollout cap that grew from 65,536 to 101,376 tokens.
set -u
: "${RUN:?RUN=<run name>}" "${OUT_ROOT:?OUT_ROOT=<run root>}"
: "${MODEL:?MODEL=<bf16 HF dir of the start checkpoint>}" "${SAMPLER_MODEL:?SAMPLER_MODEL=<its FP8 export, fp8_quant format>}"
: "${BANK_DIR:?BANK_DIR=<task pool dir>}" "${ARC_TRAIN_DIR:?ARC_TRAIN_DIR=<second pool source (frontier_mix needs one)>}"
: "${ARC_EVAL_DIR:?ARC_EVAL_DIR=<ARC-AGI-2 evaluation dir: kept out of the pool, evaluated every EVAL_EVERY steps>}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
O=$OUT_ROOT/$RUN
NPROC=${NPROC:-8}; BANK_EVAL_DIR=${BANK_EVAL_DIR:-}
PROMPTS=${PROMPTS:-110}; GROUP=${GROUP:-8}; B=${B:-147000}; MAX_PROMPT=${MAX_PROMPT:-49152}
LR=${LR:-1.5e-6}; WARMUP=${WARMUP:-10}; FINAL=${FINAL:-30}; SEED=${SEED:-0}
FLOOR=${FLOOR:-0.2}; TRUNC_SCALE=${TRUNC_SCALE:-1.0}; FORCE_BUDGET=${FORCE_BUDGET:-0}
EVAL_EVERY=${EVAL_EVERY:-10}; EVAL_N=${EVAL_N:-4}; EVAL_MAX=${EVAL_MAX:-$B}
KV=${KV:-fp8}; MTP=${MTP:-3}; STREAM=${STREAM:-32}; KV_HIGH=${KV_HIGH:-0.8}; UTIL=${UTIL:-0.87}
TRAINER_ATTN=${TRAINER_ATTN:-fa3}; TRAINER_SPEED=${TRAINER_SPEED:-sac_attn,fused_head}
VLLM_PYTHON=${VLLM_PYTHON:-${ARC_VLLM_PYTHON:-python}}; ENGINE_ENV=${ENGINE_ENV:-$REPO/scripts/rl/engine_env.sh}
TORCHRUN=${TORCHRUN:-torchrun}; TRIES=${TRIES:-3}
mkdir -p "$O"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARC_DRAFT_SAMPLE_METHOD=${ARC_DRAFT_SAMPLE_METHOD:-probabilistic}   # MTP drafts sampled from q (still exact)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--out "$O" --run-name "$RUN" --student "$MODEL" --sampler-model "$SAMPLER_MODEL" --sampler-fp8 1
  --vllm-python "$VLLM_PYTHON" --vllm-env-script "$ENGINE_ENV"
  --timeshare 1 --engine-sleep-level 1 --kv-cache-dtype "$KV" --mtp-tokens "$MTP" --engine-gpu-util "$UTIL"
  --engine-no-prefix-caching --engine-attention-backend FLASHINFER
  --sampler-dispatch-prompts 2 --sampler-stream-seqs "$STREAM" --sampler-kv-high "$KV_HIGH"
  --sampler-control-file "$O/sampler_control.json" --sampler-log-stats --per-request-seeds --seed "$SEED"
  --min-free-gb-trainer 40 --fsdp-keep-gathered 1
  --trainer-attn "$TRAINER_ATTN" --trainer-speed "$TRAINER_SPEED" --logprob-chunk 8192
  --balance lpt --balance-quad-tokens 115000 --reward-stream 4 --rollout-write-async
  --arm r2k_zb_max --stage2-cache "" --pool mixed --pool-filter none
  --bank-dir "$BANK_DIR" --arc-train-dir "$ARC_TRAIN_DIR" --arc-eval-dir "$ARC_EVAL_DIR" --bank200-dir "$BANK_EVAL_DIR"
  # one shuffled pass over the bank; the second source (ARC-AGI-2 training) gets a 0 share; no frontier revisits
  --pool-order frontier_mix --mix-schedule 1:0:0
  --augment 1 --aug-original-prob 0.33 --train-split-tests
  --prompts-per-step "$PROMPTS" --group-size "$GROUP" --temperature 1.0
  # objective: floored MaxRL, capped rollouts scored 0 (TRUNC_SCALE scales their negative advantage), seq aggregation
  --adv-norm maxrl_floor --adv-floor "$FLOOR" --adv-clip $((GROUP - 1)) --drop-zero-var outcome
  --loss-agg seq --minibatch-max 1 --minibatch-min-groups 8
  --pg-loss tis_sc --sc-topk 32 --tis-cap 2.0 --kl-coef 0
  --w-outcome 1 --outcome-require-stop 1 --w-format 0 --beta-loop 0 --w-loop-seq 0 --w-len 0
  --len-abs-alpha 0 --len-correct-alpha 0 --len-share-lambda 0 --len-allcorrect-beta 0
  --mask-truncated 0 --trunc-neg-scale "$TRUNC_SCALE"
  --train-budget-force "$FORCE_BUDGET" --forced-credit 0.5
  # pairwise length shaping in groups with >= 4/8 right, not in all-right groups
  --len-pair-lambda 0.075 --len-pair-delta 25000 --len-pair-min-correct 4 --len-pair-allright 0
  # prefix continuations of the previous step's own rollouts: 24 groups every 3rd step (+6 fresh prompts)
  --prefix-groups 24 --prefix-every 3 --prefix-from 3 --prefix-extra-prompts 6 --prefix-max-zero 4
  --prefix-zero-max-frac 0.5 --prefix-early-frac 0.6 --prefix-early-prob 0.8 --prefix-min-start 2500
  --prefix-end-margin 3000
  --max-new-tokens "$B" --max-prompt-tokens "$MAX_PROMPT" --sampler-max-model-len $((MAX_PROMPT + B))
  --lr "$LR" --warmup-steps "$WARMUP" --max-steps "$FINAL"
  # evals: the ARC-AGI-2 evaluation tasks, one prompt per test input, T 1.0, with budget-forced answers logged too
  --eval-every "$EVAL_EVERY" --no-eval-at-start --eval-arc 120 --eval-bank 0 --eval-n "$EVAL_N" --eval-tests split
  --eval-max-tokens "$EVAL_MAX" --eval-temperature 1.0 --eval-top-p 1.0 --eval-top-k 0 --eval-budget-force 1024
  --eval-partial-dir "$O/eval_partial" --rollout-store-dir "$O/rollout_store"
  --save-every 1 --dcp-split 1 --keep-optimizer-states 1 --keep-model-dirs 1 --bf16-export-every 10)
# optional: a focus set drawn at least N times per step, e.g.
#   --pool-focus-file focus.json --pool-focus-per-step 80 --pool-focus-order shuffle --pool-focus-shuffle-from 1
#   --pool-focus-cap-file never_solved.json --pool-focus-cap-schedule 2:10,8:0
read -ra EXTRA <<< "${EXTRA_ARGS:-}"
ARGS+=("${EXTRA[@]}")
printf '%q ' "${ARGS[@]}" > "$O/launch_args.txt"
[ "${DRY:-0}" = 1 ] && { echo "$TORCHRUN --standalone --nproc_per_node $NPROC -m arcsolver.rl.train ${ARGS[*]@Q}"; exit 0; }

last_step() { ls -d "$O"/checkpoints/step_* 2>/dev/null | while read -r d; do [ -f "$d/COMPLETE" ] && basename "$d"; done \
  | sort | tail -1 | sed 's/step_0*//'; }
# a crashed window resumes from the newest complete checkpoint, up to TRIES times in a row without progress
try=0
while true; do
  done_step=$(last_step); done_step=${done_step:-0}
  [ "$done_step" -ge "$FINAL" ] && { echo "done at step $done_step"; break; }
  try=$((try + 1))
  [ $try -gt "$TRIES" ] && { echo "$TRIES windows in a row without progress (last step $done_step)"; exit 1; }
  echo "$(date -u '+%F %T') window from step $done_step (try $try)"
  (cd "$REPO" && $TORCHRUN --standalone --nproc_per_node "$NPROC" -m arcsolver.rl.train "${ARGS[@]}" >> "$O/train.log" 2>&1)
  rc=$?
  new_step=$(last_step); new_step=${new_step:-0}
  echo "$(date -u '+%F %T') window exit $rc at step $new_step"
  [ "$new_step" -gt "$done_step" ] && try=0
done
