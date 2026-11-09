#!/bin/bash
# 4B recipe ("round 6"): full fine-tuning of a vocab-pruned Qwen3.5-4B on one node of 8 GPUs (H200 141 GB; FA3).
#   peak LR 1e-4, 3% linear warmup, cosine to 1e-6 (--min-lr-ratio 0.01) over 3 epochs, 64 rows per optimizer step,
#   AdamW (0.9, 0.999, eps 1e-8, wd 0.001 on matrices), grad clip 1.0, an HF export every 0.5 epoch.
# Data (build_data.sh): EPOCHS=3 MAX_SEQ=262144 SEED=20261019, no CAP (rows up to 262,144 tokens; ours peaked ~200k).
# Speed: ~73 s/step, ~63k tok/s on 8x H200 (1.76B tokens per epoch, 379 steps per epoch).
# Our run evaluated each milestone; the 2.5-epoch export was the best one (the final one was within noise on pass@2).
#
# Required env: DATA (tokenize_cache.py output), INIT (HF model dir to fine-tune), OUT (run dir; re-run to resume)
# Optional env: FA3_DIR (local FA3 build: python -m arcsolver.sft.attn_backends fetch-fa3 --dest DIR; else the
#   `kernels` package downloads it), NPROC (8), WANDB_PROJECT / WANDB_RUN_NAME (+ WANDB_API_KEY), STOP_FILE
# Extra arguments are passed to the trainer.
set -euo pipefail
: "${DATA:?}" "${INIT:?}" "${OUT:?}"
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=${NPROC:-8}
TRAIN=(--data "$DATA" --base-model "$INIT" --out "$OUT" --global-rows 64 --epochs 3 --peak-lr 1e-4
       --min-lr-ratio 0.01 --warmup-frac 0.03 --weight-decay 0.001 --adam-eps 1e-8 --max-grad-norm 1.0
       --milestone-fracs 0.16666666666666666 0.3333333333333333 0.5 0.6666666666666666 0.8333333333333334 1.0
       --save-every 20 --attn fa3 --merge-tail)
[ -n "${FA3_DIR:-}" ] && TRAIN+=(--fa3-dir "$FA3_DIR")
[ -n "${STOP_FILE:-}" ] && TRAIN+=(--stop-file "$STOP_FILE")
[ -n "${WANDB_PROJECT:-}" ] && TRAIN+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "${WANDB_RUN_NAME:-$(basename "$OUT")}")
python -m arcsolver.sft.train_fsdp "${TRAIN[@]}" "$@" --plan-only "$NPROC"      # writes $OUT/run_plan.json
torchrun --nproc_per_node "$NPROC" -m arcsolver.sft.train_fsdp "${TRAIN[@]}" "$@"
