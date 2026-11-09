#!/bin/bash
# Teacher attempts -> tokenized SFT dataset (README.md "Pipeline"):
#   build_teacher_rows (prompt match + re-score) -> clean_oov (pruned-vocab cleanup) -> [select_rows] ->
#   build_cache (eval leak check, epoch orders) -> tokenize_cache -> [subset_data: exact length cap, more epochs]
#
# Required env:
#   ATTEMPTS        teacher attempt files / dirs (space-separated; format in README.md)
#   TASK_ROOT       base dir of the attempts' relative task_file paths
#   ORIG_TOKENIZER  the original (unpruned) model / tokenizer dir
#   KEEP            the vocab keep set (prune_vocab.sh)
#   MODEL           the model to train (vocab-pruned): its tokenizer + chat template are used for lengths and ids
#   EVAL_DIR        ARC-AGI-2 evaluation tasks (leak check)
#   WORK            scratch dir;  OUT: the tokenized dataset dir (train_fsdp.py --data)
# Optional env:
#   EPOCHS (3) MAX_SEQ (262144) SEED (20261019) WORKERS (64)
#   SELECT_ARGS     select_rows.py options (e.g. "--holdout holdout.json"); unset = every cleaned row
#   CAP             exact token cap (subset_data.py; rows above it are dropped) with CAP_EPOCHS (= EPOCHS), CAP_SEED
#   EXCLUDE_IDS     build_cache --exclude-ids file(s)
set -euo pipefail
: "${ATTEMPTS:?}" "${TASK_ROOT:?}" "${ORIG_TOKENIZER:?}" "${KEEP:?}" "${MODEL:?}" "${EVAL_DIR:?}" "${WORK:?}" "${OUT:?}"
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
EPOCHS=${EPOCHS:-3}; MAX_SEQ=${MAX_SEQ:-262144}; SEED=${SEED:-20261019}; WORKERS=${WORKERS:-64}
read -ra ATT <<< "$ATTEMPTS"
mkdir -p "$WORK"
$PY -m arcsolver.sft.build_teacher_rows "$WORK/rows.jsonl" --attempts "${ATT[@]}" --task-root "$TASK_ROOT" \
  --workers "$WORKERS" | tee "$WORK/rows.log"
$PY -m arcsolver.sft.clean_oov --cot "$WORK/rows.jsonl" --orig "$ORIG_TOKENIZER" --keep "$KEEP" \
  --out "$WORK/rows_clean.jsonl" --regen "$WORK/regen_oov.jsonl" --edits "$WORK/oov_edits.tsv" --workers "$WORKERS"
ROWS=$WORK/rows_clean.jsonl
if [ -n "${SELECT_ARGS:-}" ]; then
  read -ra SA <<< "$SELECT_ARGS"
  $PY -m arcsolver.sft.select_rows --cot "$ROWS" --out "$WORK/rows_sel.jsonl" --manifest "$WORK/sel_manifest.json" "${SA[@]}"
  ROWS=$WORK/rows_sel.jsonl
fi
XI=(); [ -n "${EXCLUDE_IDS:-}" ] && { read -ra E <<< "$EXCLUDE_IDS"; XI=(--exclude-ids "${E[@]}"); }
$PY -m arcsolver.sft.build_cache --cot "$ROWS" --cot-repeat --aux-frac 0 --epochs "$EPOCHS" --max-seq-length "$MAX_SEQ" \
  --tokenizer "$MODEL" --seed "$SEED" --eval-dir "$EVAL_DIR" --workers "$WORKERS" "${XI[@]}" --output "$WORK/cache.json"
FULL=$OUT; [ -n "${CAP:-}" ] && FULL=$WORK/tok_full
rm -rf "$FULL.part"
$PY -m arcsolver.sft.tokenize_cache --cache "$WORK/cache.json" --tokenizer "$MODEL" --out "$FULL.part" --workers "$WORKERS"
rm -rf "$FULL" && mv "$FULL.part" "$FULL"
if [ -n "${CAP:-}" ]; then
  rm -rf "$OUT.part"
  $PY -m arcsolver.sft.subset_data --data "$FULL" --out "$OUT.part" --max-len "$CAP" --epochs "${CAP_EPOCHS:-$EPOCHS}" \
    --base-seed "${CAP_SEED:-20261025}"
  rm -rf "$OUT" && mv "$OUT.part" "$OUT"
fi
echo "dataset: $OUT"
