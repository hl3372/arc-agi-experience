#!/bin/bash
# 9B recipe: full fine-tuning of a vocab-pruned Qwen3.5-9B on 2 nodes x 8 GPUs (B300; FlashAttention-4).
#   peak LR 8e-5, 40 warmup steps, cosine to 1e-6 (--min-lr-ratio 0.0125), 64 rows per optimizer step, AdamW (0.9,
#   0.999, eps 1e-8, wd 0.001 on matrices), grad clip 1.0, a 10-epoch plan with an HF export every 0.5 epoch.
# Data (build_data.sh): EPOCHS=3 MAX_SEQ=262144 SEED=20261019 CAP=163840 CAP_EPOCHS=10 CAP_SEED=20261026 (rows over
#   163,840 tokens dropped, 10 epoch orders).
# Stretched cosine: when new rows were added mid-run, we rebuilt the dataset with union_data.py (the trained steps
#   byte-identical, then new epochs over all rows) and resumed with
#     STRETCH="--lr-stretch-from 1250 --lr-stretch-total 5000"
#   (the rest of the original 5,000-step cosine stretched over the longer run; LR continuous at the restart), plus
#   LEVERS="--fused-kernels liger --recompute selective" (~1.14x; numerics matched the eager path on a resume A/B).
# Blackwell: FLA_TILELANG=0 is required, or the Gated DeltaNet (fla) layers run ~2.5x slower.
# Memory: --act-gpu-budget-gb 208 keeps every row's activations on the 268 GB B300 (no offload). On 141 GB H200s
#   we used --attn fa3 --act-gpu-budget-gb 28 --act-arena-gb 171 (pinned host arena per rank, ~20 GiB offloaded).
# Speed: ~30 s/step, ~145k tok/s on 2x8 B300 (after ~15 steps of Triton JIT warm-up).
#
# Run the same command on both nodes. Required env: DATA, INIT, OUT, NODE_RANK (0 / 1), MASTER_ADDR (node 0's address
#   on the inter-node network). Optional: NNODES (2), MASTER_PORT (29531), FA4_DIR (side dir with flash-attn-4, see
#   attn_backends.load_fa4), STRETCH, LEVERS, STOP_FILE, WANDB_PROJECT / WANDB_RUN_NAME.
# Checkpoints: every rank writes its own shards under $OUT/ckpt on its node; a resume needs every node to see the
#   newest checkpoint's complete file set (use a shared filesystem for $OUT, or sync ckpt/step_N between the nodes
#   before restarting). HF exports are written by node 0.
set -euo pipefail
: "${DATA:?}" "${INIT:?}" "${OUT:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}"
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLA_TILELANG=0
NNODES=${NNODES:-2}; NPN=${NPROC_PER_NODE:-8}
TRAIN=(--data "$DATA" --base-model "$INIT" --out "$OUT" --global-rows 64 --epochs 10 --peak-lr 8e-5
       --min-lr-ratio 0.0125 --warmup-steps 40 --weight-decay 0.001 --adam-beta2 0.999 --adam-eps 1e-8 --max-grad-norm 1.0
       --milestone-fracs 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
       --save-every 20 --attn fa4 --act-gpu-budget-gb 208 --merge-tail)
[ -n "${FA4_DIR:-}" ] && TRAIN+=(--fa4-dir "$FA4_DIR")
read -ra X <<< "${STRETCH:-} ${LEVERS:-}"; TRAIN+=("${X[@]}")
[ -n "${STOP_FILE:-}" ] && TRAIN+=(--stop-file "$STOP_FILE")
[ -n "${WANDB_PROJECT:-}" ] && TRAIN+=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "${WANDB_RUN_NAME:-$(basename "$OUT")}")
[ "$NODE_RANK" = 0 ] && python -m arcsolver.sft.train_fsdp "${TRAIN[@]}" "$@" --plan-only $((NNODES * NPN))
torchrun --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$NPN" --master-addr "$MASTER_ADDR" \
  --master-port "${MASTER_PORT:-29531}" -m arcsolver.sft.train_fsdp "${TRAIN[@]}" "$@"
