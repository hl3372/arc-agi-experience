#!/bin/bash
# 9B GRPO recipe ("grpo9b_v3"): GRPO with a binary exact-match reward, DAPO token-mean loss, TIS + score centering
# instead of a PPO ratio, reference KL k3 x 0.001, 272 tasks x 6 rollouts per step, FP8 vLLM rollouts (FP8 KV, MTP 3)
# time-sharing the GPUs with the FSDP2 trainer. See arcsolver/rl/README.md.
#
# Two phases, as in the original run:
#   PHASE=1  steps 1..22, contract grpo_dapo_kl_sc_v1 (no prefix groups)
#   PHASE=2  resume from step 22 with contract grpo_dapo_kl_sc_prefix_v1: on even steps 68 of the 272 groups continue
#            a frozen prefix of a previous-step rollout; runs until FINAL (the released checkpoint is step 34)
#
# Run on every node (same command, same paths); NNODES=1 runs one node. Rank 0 lives on NODE_RANK 0:
#   RUN=grpo9b OUT_ROOT=/data/rl MODEL=/models/sft9b SAMPLER_MODEL=/models/sft9b_fp8 \
#   BANK_DIR=/data/pool/bank ARC_TRAIN_DIR=/data/pool/arc_train ARC_EVAL_DIR=/data/ARC-AGI-2/data/evaluation \
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=$NODE0_ADDR SERVICE_ADDRS=$NODE0_ADDR,$NODE1_ADDR PHASE=1 bash scripts/rl/grpo9b.sh
#   DRY=1 ...    print the resolved command only
# With node-local OUT dirs, every node must hold every checkpoint file before a resume (the trainer checks and refuses
# otherwise): use a shared filesystem (not a FUSE object-store mount) or copy the newest checkpoint between nodes
# before relaunching.
set -u
: "${RUN:?RUN=<run name>}" "${OUT_ROOT:?OUT_ROOT=<run root>}"
: "${MODEL:?MODEL=<bf16 HF dir of the start checkpoint>}" "${SAMPLER_MODEL:?SAMPLER_MODEL=<its FP8 export, fp8_quant format>}"
: "${BANK_DIR:?BANK_DIR=<task pool dir>}" "${ARC_EVAL_DIR:?ARC_EVAL_DIR=<ARC-AGI-2 evaluation dir: kept out of the pool>}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
O=$OUT_ROOT/$RUN
PHASE=${PHASE:-1}
NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; NPN=${NPN:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}; MASTER_PORT=${MASTER_PORT:-29571}
SERVICE_ADDRS=${SERVICE_ADDRS:-$MASTER_ADDR}; SVC_PORT=${SVC_PORT:-29760}
ARC_TRAIN_DIR=${ARC_TRAIN_DIR:-}; BANK_EVAL_DIR=${BANK_EVAL_DIR:-}
PROMPTS=${PROMPTS:-272}; GROUP=${GROUP:-6}; B=${B:-147000}; MAX_PROMPT=${MAX_PROMPT:-49152}
LR=${LR:-2e-6}; WARMUP=${WARMUP:-10}; SEED=${SEED:-1}
if [ "$PHASE" = 1 ]; then FINAL=${FINAL:-22}; else FINAL=${FINAL:-34}; fi
KV=${KV:-fp8}; MTP=${MTP:-3}; STREAM=${STREAM:-84}; KV_HIGH=${KV_HIGH:-0.9}; UTIL=${UTIL:-0.92}
TRAINER_ATTN=${TRAINER_ATTN:-fa4}; TRAINER_SPEED=${TRAINER_SPEED:-sac_attn,fused_head}
VLLM_PYTHON=${VLLM_PYTHON:-${ARC_VLLM_PYTHON:-python}}; ENGINE_ENV=${ENGINE_ENV:-$REPO/scripts/rl/engine_env.sh}
TORCHRUN=${TORCHRUN:-torchrun}; TRIES=${TRIES:-3}
mkdir -p "$O"

# Trainer, node services and engines inherit this environment.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLA_TILELANG=${FLA_TILELANG:-0}                       # Blackwell: fla's TileLang kernels are much slower here
export VLLM_HAS_FLASHINFER_CUBIN=${VLLM_HAS_FLASHINFER_CUBIN:-1}
export ARC_DRAFT_SAMPLE_METHOD=${ARC_DRAFT_SAMPLE_METHOD:-probabilistic}   # MTP drafts sampled from q (still exact)
export ARC_GDN_PREFILL_BACKEND=${ARC_GDN_PREFILL_BACKEND:-triton}
# the budget-forcing suffix of these models (only used by forced answers, which this recipe does not train with)
export ARC_EVAL_FORCE_SUFFIX=${ARC_EVAL_FORCE_SUFFIX:-'\n\nNow final.\n</think>\n\n'}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--out "$O" --run-name "$RUN" --student "$MODEL" --sampler-model "$SAMPLER_MODEL" --sampler-fp8 1
  --vllm-python "$VLLM_PYTHON" --vllm-env-script "$ENGINE_ENV"
  --node-sampler-services 1 --sampler-service-addrs "$SERVICE_ADDRS" --sampler-service-port "$SVC_PORT"
  --timeshare 1 --engine-sleep-level 1 --kv-cache-dtype "$KV" --mtp-tokens "$MTP" --engine-gpu-util "$UTIL"
  --engine-no-prefix-caching --engine-attention-backend FLASHINFER
  --sampler-dispatch-prompts 2 --sampler-stream-seqs "$STREAM" --sampler-kv-high "$KV_HIGH"
  --sampler-control-file "$O/sampler_control.json" --sampler-log-stats --per-request-seeds --seed "$SEED"
  --trainer-attn "$TRAINER_ATTN" --trainer-speed "$TRAINER_SPEED" --probe-every 10
  --logprob-chunk 8192 --fsdp-keep-gathered 1 --balance lpt --balance-quad-tokens 115000
  --freeze-embed 1 --min-free-gb-trainer 100 --reward-stream 4 --rollout-write-async
  --arm r2k_zb_max --stage2-cache "" --pool bank --pool-filter none --pool-order weighted
  --bank-dir "$BANK_DIR" --arc-train-dir "$ARC_TRAIN_DIR" --arc-eval-dir "$ARC_EVAL_DIR" --bank200-dir "$BANK_EVAL_DIR"
  --augment 1 --aug-original-prob 0.33 --pool-drop-solved 1
  --prompts-per-step "$PROMPTS" --group-size "$GROUP" --train-split-tests
  --adv-norm std --adv-eps 1e-4 --adv-clip 0 --loss-agg token
  --pg-loss tis_sc --sc-topk 32 --tis-cap 2.0 --clip-eps 0.2 --clip-eps-high 0.2
  --drop-zero-var outcome --minibatch-max 1 --temperature 1
  --w-outcome 1 --outcome-require-stop 1 --w-format 0
  --beta-loop 0 --w-loop-seq 0 --w-len 0
  --len-abs-alpha 0 --len-correct-alpha 0 --len-share-lambda 0 --len-allcorrect-beta 0 --len-pair-lambda 0
  --mask-truncated 0 --trunc-neg-scale 1
  --train-budget-force 0 --train-budget-force-from 0 --forced-credit 0.5 --train-force-inline 0
  --kl-coef 0.001 --dyn-target-groups 0
  --max-new-tokens "$B" --max-prompt-tokens "$MAX_PROMPT" --sampler-max-model-len $((MAX_PROMPT + B))
  --eval-temperature 1.0 --eval-top-p 1.0 --eval-top-k 0
  --lr "$LR" --warmup-steps "$WARMUP" --max-steps "$FINAL"
  --eval-every 0 --no-eval-at-start --eval-arc 0 --eval-bank 0
  --save-every 1 --dcp-split 1 --keep-optimizer-states 1 --keep-model-dirs 2 --bf16-export-every 10)
if [ "$PHASE" = 1 ]; then
  ARGS+=(--objective-contract grpo_dapo_kl_sc_v1 --prefix-groups 0 --prefix-extra-prompts 0)
else
  # prefix groups on even steps; the resume accepts exactly the keys that change between the two contracts
  ARGS+=(--objective-contract grpo_dapo_kl_sc_prefix_v1
    --prefix-groups 68 --prefix-every 2 --prefix-from 0 --prefix-extra-prompts 0 --prefix-max-zero 11
    --prefix-zero-max-frac 0.5 --prefix-early-frac 0.6 --prefix-early-prob 0.8 --prefix-min-start 2500
    --prefix-end-margin 10000
    --allow-objective-change objective_contract,prefix_groups,prefix_every,prefix_max_zero,prefix_zero_max_frac,prefix_early_frac,prefix_end_margin)
fi
# optional extras, e.g. --pool-drop-solved-seed DIR:STEP of earlier runs, or --wandb-project NAME / --no-wandb
read -ra EXTRA <<< "${EXTRA_ARGS:-}"
ARGS+=("${EXTRA[@]}")
printf '%q ' "${ARGS[@]}" > "$O/launch_args.node$NODE_RANK.txt"
[ "${DRY:-0}" = 1 ] && { echo "$TORCHRUN --nnodes $NNODES --node-rank $NODE_RANK --nproc-per-node $NPN -m arcsolver.rl.train ${ARGS[*]@Q}"; exit 0; }

last_step() { ls -d "$O"/checkpoints/step_* 2>/dev/null | while read -r d; do [ -f "$d/COMPLETE" ] && basename "$d"; done \
  | sort | tail -1 | sed 's/step_0*//'; }
# a crashed window resumes from the newest complete checkpoint, up to TRIES times in a row without progress
try=0
while true; do
  done_step=$(last_step); done_step=${done_step:-0}
  [ "$done_step" -ge "$FINAL" ] && { echo "done at step $done_step"; break; }
  try=$((try + 1))
  [ $try -gt "$TRIES" ] && { echo "$TRIES windows in a row without progress (last step $done_step)"; exit 1; }
  echo "$(date -u '+%F %T') window from step $done_step (phase $PHASE, try $try)"
  (cd "$REPO" && $TORCHRUN --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$NPN" \
     --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" -m arcsolver.rl.train "${ARGS[@]}" \
     >> "$O/train.node$NODE_RANK.log" 2>&1)
  rc=$?
  new_step=$(last_step); new_step=${new_step:-0}
  echo "$(date -u '+%F %T') window exit $rc at step $new_step"
  [ "$new_step" -gt "$done_step" ] && try=0
done
