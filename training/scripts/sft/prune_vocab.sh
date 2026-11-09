#!/bin/bash
# Vocabulary pruning of a Qwen3.5-family base model (arcsolver.tools.vocab_prune): 248,320 -> K rows (K = 128,768 in our
# runs), exact tokenization for every text whose original encoding uses only kept tokens, exact logits at the kept ids.
#
# Required env:
#   BASE      original HF model dir (e.g. a local snapshot of Qwen/Qwen3.5-4B)
#   OUT       output dir for the pruned model
#   KEEP      keep-set JSON to write (or to reuse, if it exists)
# Corpus for the keep set (optional env; space-separated lists):
#   TASK_DIRS   ARC task dirs whose rendered prompts are added (e.g. ARC-AGI-2/data/training ARC-AGI-2/data/evaluation)
#   EXTRA       text / .json / .jsonl files or dirs added verbatim (e.g. your teacher CoT rows)
#   TRACE_BANK  trace-bank dir (layout: vocab_prune.iter_trace_texts)
# Our keep set used the arms r2k_zb_max r2k_zb_two_opt_max qwen38, --keep-chars symbols and --align 128 (below).
set -euo pipefail
: "${BASE:?}" "${OUT:?}" "${KEEP:?}"
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
read -ra TD <<< "${TASK_DIRS:-}"
read -ra EX <<< "${EXTRA:-}"
if [ ! -f "$KEEP" ]; then
  $PY -m arcsolver.tools.vocab_prune keep-set --tokenizer "$BASE" --out "$KEEP" --keep-chars symbols --align 128 \
    --arms ${ARMS:-r2k_zb_max r2k_zb_two_opt_max qwen38} --trace-bank "${TRACE_BANK:-}" --max-traces -1 --max-tasks -1 \
    --task-dir "${TD[@]}" --extra-file "${EX[@]}"
fi
$PY -m arcsolver.tools.vocab_prune prune --src "$BASE" --dst "$OUT" --keep "$KEEP"
# checks: tokenizer equivalence on the corpus + random texts, then exact logits (two sequential CPU loads)
$PY -m arcsolver.tools.vocab_prune validate-tokenizer --src "$BASE" --dst "$OUT" --trace-bank "${TRACE_BANK:-}" \
  --task-dir "${TD[@]}" --extra-file "${EX[@]}" --json "$OUT/validate_tokenizer.json"
$PY -m arcsolver.tools.vocab_prune validate-model --src "$BASE" --dst "$OUT" --device "${DEVICE:-cpu}"
