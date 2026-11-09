#!/bin/bash
# 4B ablation recipe (the scaling sweep's base arm): one cleaned teacher row per task (select_rows.py, held-out tasks
# excluded), 2 epochs, rows capped at 131,072 tokens, full fine-tuning of a vocab-pruned Qwen3.5-4B on 8x H200 (FA3):
#   peak LR 5e-5, 3% warmup, cosine to 2.5e-6 (--min-lr-ratio 0.05), 32 rows per optimizer step, AdamW (0.9, 0.999,
#   eps 1e-8, wd 0.001), grad clip 1.0. Speed: ~63k tok/s, ~33-35 s/step.
# Data (build_data.sh): SELECT_ARGS="--holdout holdout.json" (plus the arm's options: --frac / --n-tasks / --variants /
#   --all-rows / --answer-only) EPOCHS=2 MAX_SEQ=262144 SEED=20261106 CAP=131072 CAP_EPOCHS=2 CAP_SEED=20261106.
#   --max-row-tokens 131072 below repeats the cap inside the trainer (a no-op on capped data).
#
# Required env: DATA, INIT, OUT. Optional: FA3_DIR, NPROC (8), MILESTONES (HF exports at these step counts; default
#   every 0.5 epoch via fractions), STOP_FILE, WANDB_PROJECT / WANDB_RUN_NAME. Extra arguments go to the trainer.
set -euo pipefail
: "${DATA:?}" "${INIT:?}" "${OUT:?}"
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=${NPROC:-8}
TRAIN=(--data "$DATA" --base-model "$INIT" --out "$OUT" --global-rows 32 --epochs 2 --peak-lr 5e-5 --min-lr-ratio 0.05
       --warmup-frac 0.03 --weight-decay 0.001 --adam-eps 1e-8 --max-grad-norm 1.0 --save-every 50
       --attn fa3 --merge-tail --max-row-tokens 131072)
if [ -n "${MILESTONES:-}" ]; then read -ra M <<< "$MILESTONES"; TRAIN+=(--milestone-steps "${M[@]}")
else TRAIN+=(--milestone-fracs 0.25 0.5 0.75 1.0); fi
[ -n "${FA3_DIR:-}" ] && TRAIN+=(--fa3-dir "$FA3_DIR")
[ -n "${STOP_FILE:-}" ] && TRAIN+=(--stop-file "$STOP_FILE")
[ -n "${WANDB_PROJECT:-}" ] && TRAIN+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "${WANDB_RUN_NAME:-$(basename "$OUT")}")
python -m arcsolver.sft.train_fsdp "${TRAIN[@]}" "$@" --plan-only "$NPROC"
torchrun --nproc_per_node "$NPROC" -m arcsolver.sft.train_fsdp "${TRAIN[@]}" "$@"
