# ARC-AGI-2 evaluation (protocol P4)

A standalone vLLM evaluation of a checkpoint on ARC tasks, with the sampling, forced-answer and scoring settings
behind the numbers in our writeup.

**Status.** The reference numbers in the writeup came from our internal training/eval harness, run with the
same settings as here. This standalone tool has **not yet been GPU-validated** against that harness.
- The scoring path (`scoring.py`, `score.py`) has been checked. On 18 stored evals from the internal harness, it
  reproduces the internal scorer's natural and forced pass@1/2/4 exactly.
- The generation path has only been exercised on CPU, against a fake vLLM engine: queueing, forced answers,
  resume, sharding and merging.
- Its vLLM calls follow the ones the internal harness makes on vLLM 0.30.0 (`LLM.enqueue` plus
  `llm_engine.step()`, `TokensPrompt`, `speculative_config` method `"mtp"`).

| file | what |
|---|---|
| `run_eval.py` | generation (vLLM), forced answers, scoring, `samples.jsonl` + `summary.json` |
| `score.py` | re-scores an existing samples file against a task directory (no GPU) |
| `scoring.py` | the P4 rules (pure Python; shared by both) |
| `REQUIREMENTS.txt` | `vllm==0.30.0`, `transformers==5.2.0` (Python 3.12) |
| `../../scripts/eval/eval_arc_agi2.sh` | example launcher at the reference settings |

## Usage

Run these from the repository root, or put it on `PYTHONPATH`.

```bash
# ARC-AGI-2 public evaluation (120 tasks, 167 test inputs), reference settings, 8 GPUs
python -m arcsolver.eval.run_eval --model /path/to/checkpoint \
    --tasks-dir /path/to/ARC-AGI-2/data/evaluation --out-dir runs/eval_ckpt \
    --kv-cache-dtype fp8 --mtp 3 --dp 8

# the same through the launcher (it re-scores at the end)
MODEL=/path/to/checkpoint TASKS=/path/to/ARC-AGI-2/data/evaluation OUT=runs/eval_ckpt DP=8 \
    bash scripts/eval/eval_arc_agi2.sh

# re-check a finished (or partial) run without a GPU
python -m arcsolver.eval.score --samples runs/eval_ckpt/samples.jsonl \
    --tasks-dir /path/to/ARC-AGI-2/data/evaluation
```

**Selecting tasks.** `--items FILE` limits the tasks to those listed. The file can be:
- a JSON list of ids;
- a JSON list of `{"task_id": ...}` objects;
- a JSON object with such a list under `"items"`;
- plain text with one id per line.

`--limit-tasks N` keeps only the first N selected tasks, for smoke tests. Every selected task needs its test
outputs.

**Answer-only models.** `--no-think` renders prompts with `enable_thinking=False`. The chat template then closes
an empty reasoning block inside the prompt, so the whole reply is scored as the answer. There are no forced
answers in this mode.

**Resume.** Rows are appended and fsynced as each test input finishes. Rerunning the same command skips the test
inputs already in `samples.jsonl` and drops a line truncated by an interruption.

`config.json` records the sampling settings. A resume that changes any of them (model, n, temperature, top_p,
top_k, the caps, the force settings, thinking mode or the seed) is refused unless you pass
`--allow-config-change`. Engine-only settings (dp/tp, KV dtype, MTP) may change between resumes.

**Data / tensor parallel.**
- `--tp N` sets vLLM's tensor parallel size.
- `--dp N` starts N engine processes, each on its own `--tp` GPUs (taken from `--gpus`, else
  `$CUDA_VISIBLE_DEVICES`, else `0..dp*tp-1`).
- The test inputs are split round-robin across the engines. Each engine writes `samples.shard<r>.jsonl` and
  logs to `shard<r>.log`.
- Shard rows are merged into `samples.jsonl` when the shards end, or at the next start after an interruption.
- Sample seeds depend only on (`--seed`, task id, test index, sample index), so sharding and resuming do not change
  which seed a sample gets.

## The protocol (P4)

**Prompts.**
- Tests are split: a task with n test inputs becomes n single-test sub-tasks, each with all the training pairs.
- Each sub-task is rendered by `arcsolver.prompt.arc_render.build_messages(task, "r2k_zb_max")`.
- The messages go through the model's own chat template with `add_generation_prompt=True` and
  `enable_thinking=True`. The tool warns if the rendered prompt does not end in `<think>\n`.

**Sampling.**
- k = 4 samples per test input (`--n`). Each sample is its own request with its own seed.
- Temperature 1.0, top_p 1.0, top-k off (`--top-k 0`, which vLLM treats like -1).
- Up to 147,000 new tokens (`--max-new-tokens`), never past the context: max model length 196,608 (`--max-model-len`).
- Prefix caching is off.

**Forced answers.**
- Forcing applies to a sample that stopped at the token cap (`finish_reason == "length"`) while its reasoning was
  still open (no `</think>`).
- Such a sample gets the force suffix `"\n\nNow final.\n</think>\n\n"` (`--force-suffix`; backslash escapes are
  decoded) appended after its own tokens.
- It then gets up to 1,024 more tokens (`--force-budget`), sampled with the same settings as the samples
  (temperature 1.0, not greedy).
- No forced answer is generated when fewer than 64 tokens of context remain.
- A sample capped after it had already closed its reasoning keeps its natural text.
- The suffix is the closing line of the training traces. Some earlier internal evals used the bare
  `"\n</think>\n\n"`; pass `--force-suffix '\n</think>\n\n'` to reproduce those.

**Scoring.**
- A sample is correct when `arc_render.final_answer_text` reports the reasoning closed and the answer after the
  last `</think>` parses (`arc_render.answer_grids(answer, 1)`) to exactly one grid equal to the test output.
  Extra blocks after the grid, prose or code fences make it wrong.
- There are two views:
  - *natural*: the samples as generated (the headline);
  - *forced*: a forced sample is scored on text + suffix + forced answer, and every other sample keeps its
    natural score.

**Aggregation.**
- Per test input: the fraction correct of the k samples (pass@1) and the unbiased pass@2 / pass@4
  (`arc_render.pass_at_k`).
- These are averaged within a task over its test inputs, then over tasks.
- A task counts only once all its test inputs are scored. Partial tasks are listed as `incomplete_tasks`.

**Reported:**
- pass@1/2/4, natural and forced;
- closed share (samples whose reasoning closed);
- capped share;
- forced share and how many forced answers turned a wrong sample right;
- mean generated tokens of the natural samples;
- the tasks solved at least once;
- per-task scores.

## Outputs

`samples.jsonl` has one row per test input:

| field | content |
|---|---|
| `task_id`, `test_index`, `n_test`, `prompt_tokens`, `think` | which test input, its prompt length, thinking mode |
| `texts`, `gen_tokens`, `finish` | the k natural samples, their token counts, vLLM finish reasons |
| `closed` | reasoning closed, per sample |
| `forced`, `forced_tokens`, `forced_finish` | per sample: suffix + forced answer (or null), its token count and finish |
| `correct`, `correct_forced` | per-sample correctness, natural and forced views (`n_correct`, `n_correct_forced`: counts) |
| `seeds` | the per-sample request seeds |

`summary.json` holds the P4 summary (rates as fractions) plus the run configuration. `score.py` writes the same
summary, recomputed from the stored texts, and reports any test input whose stored counts disagree with the
recomputed ones. It also reads gzip files (`.jsonl.gz`) whose rows have `task_id`, `test_index`, `texts` and,
optionally, `forced`, so evals written by other tools in that format can be re-scored the same way.

## Engine notes

**Reference runs.** These used:
- block-wise FP8 weights (128x128 scales, the format of the official Qwen FP8 checkpoints);
- an FP8 KV cache (`--kv-cache-dtype fp8`);
- the checkpoint's own MTP head for speculative decoding: `--mtp 3`, which becomes
  `speculative_config={"method": "mtp", "num_speculative_tokens": 3}`, with
  `--draft-sample-method probabilistic`.

Speculative decoding is exact sampling with either draft method. FP8 weights and the FP8 KV cache change the
numerics slightly, so bf16 weights should give statistically equivalent but not identical numbers. Pass an FP8
checkpoint as `--model` to match the reference runs; `--quantization fp8` is vLLM's online FP8, which is a
different quantization.

**Long generations.** Samples run to 147k tokens, so the KV cache, not compute, limits how many sequences fit.
If vLLM logs preemptions, lower `--max-num-seqs` to roughly the KV capacity divided by about 100k tokens. The
engine must fit at least one 196,608-token sequence, so lower `--max-model-len` (and with it the cap) on small
GPUs, and note that the result is then no longer P4.

**Multimodal checkpoints.** When the checkpoint's `config.json` has a vision tower, the engine is started with
`limit_mm_per_prompt={"image": 0, "video": 0}` (text-only prompts).

**Noise.** Results carry sampling noise: with k = 4 over 120 tasks, expect run-to-run differences of a point or two
for the same model. The internal harness used different seeds, so texts differ sample by sample.
