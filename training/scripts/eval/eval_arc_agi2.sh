#!/usr/bin/env bash
# P4 eval of one checkpoint on the ARC-AGI-2 public evaluation set (120 tasks, 167 test inputs), at the settings of
# the reference runs: 4 samples per test input, T 1.0 / top_p 1.0 / top-k off, cap 147,000 new tokens + forced answer
# 1,024, context 196,608, FP8 KV cache, the model's own MTP head (3 draft tokens), prefix caching off.
# Re-running the same command resumes (finished test inputs are skipped), then re-scores the samples file.
#
#   MODEL=/path/to/checkpoint TASKS=/path/to/ARC-AGI-2/data/evaluation OUT=runs/eval_mymodel \
#     [ITEMS=item_list.json] [DP=8] [TP=1] [MTP=3] [KV=fp8] bash scripts/eval/eval_arc_agi2.sh [extra run_eval flags]
#
# Extra flags go to run_eval, e.g. --no-think (answer-only models), --max-num-seqs 32, --gpus 0,1,2,3,
# --draft-sample-method probabilistic, --limit-tasks 2 (smoke test).
set -euo pipefail
MODEL=${MODEL:?checkpoint dir}
TASKS=${TASKS:?ARC task dir, e.g. ARC-AGI-2/data/evaluation}
OUT=${OUT:-runs/eval_$(basename "$MODEL")}
DP=${DP:-1}; TP=${TP:-1}; MTP=${MTP:-3}; KV=${KV:-fp8}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=${PYTHON:-python3}

ITEM_ARGS=()
[ -n "${ITEMS:-}" ] && ITEM_ARGS=(--items "$ITEMS")

"$PY" -m arcsolver.eval.run_eval --model "$MODEL" --tasks-dir "$TASKS" --out-dir "$OUT" ${ITEM_ARGS[@]+"${ITEM_ARGS[@]}"} \
  --n 4 --temperature 1.0 --top-p 1.0 --top-k 0 --max-new-tokens 147000 --max-model-len 196608 \
  --force-budget 1024 --kv-cache-dtype "$KV" --mtp "$MTP" --dp "$DP" --tp "$TP" "$@"

# independent re-check of the stored flags from the stored texts (no GPU needed)
"$PY" -m arcsolver.eval.score --samples "$OUT/samples.jsonl" --tasks-dir "$TASKS" ${ITEM_ARGS[@]+"${ITEM_ARGS[@]}"} \
  --out "$OUT/summary_rescored.json"
