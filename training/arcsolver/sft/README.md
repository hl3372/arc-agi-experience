# arcsolver.sft: supervised fine-tuning

Full fine-tuning of Qwen3.5-family models (4B, 9B) on teacher reasoning traces for ARC-AGI-2, with a data pipeline that
checks every trace against the exact prompt format and the reward's answer parser.

- `arcsolver/sft/`: the data pipeline and the FSDP2 trainer.
- `arcsolver/tools/`: vocabulary pruning (`vocab_prune.py`) and block-FP8 export (`make_fp8.py`, `fp8_quant.py`).
- `scripts/sft/`: the recipes we ran, as example launchers.

Every module runs as `python -m arcsolver.<pkg>.<module> -h`. The scripts set `PYTHONPATH` to the repo root, so they
work without installing the package. Dependencies are listed in `REQUIREMENTS.txt`.

## Pipeline

| step | module | what it does |
|---|---|---|
| 0 | `tools.vocab_prune` | Shrinks the base model's vocabulary (248,320 rows to K = 128,768) and proves that tokenization and logits are unchanged (`scripts/sft/prune_vocab.sh`). |
| 1 | `sft.build_teacher_rows` | Teacher attempts become CoT rows. It keeps an attempt only if (a) its request messages equal `arc_render.build_messages(task, "r2k_zb_max")` byte for byte and (b) `reasoning + "\n</think>\n\n" + answer` re-scores correct with `arc_render.score_task`. |
| 2 | `sft.clean_oov` | Removes out-of-vocabulary tokens from the reasoning, measured against the pruned vocabulary. Accents, check marks, full-width punctuation and a few Russian words are replaced. Rows with anything else (CJK words, Greek letters, box glyphs and so on) are dropped and listed for regeneration. `sft.oov_report` gives the same analysis as a report. |
| 3 | `sft.select_rows` | Optional, used for ablations. Keeps one row (or k rows) per base task, excludes held-out tasks, and supports nested task subsets. |
| 4 | `sft.build_cache` | Applies the chat format (`<think>\n{reasoning}\n</think>\n\n{answer}`) and the ARC-AGI-2 evaluation leak check (task id, or 2 or more eval grids in the prompt). Drops rows over `--max-seq-length` instead of truncating them. Writes per-epoch row orders (`--cot-repeat`: every row in every epoch, reshuffled). |
| 5 | `sft.tokenize_cache` | Pre-tokenizes the rows with the model's chat template. Writes `ids.u32`, `offsets.npy`, `label_start.npy` (the loss covers the assistant turn) and `source.npy`, plus `order.npy` and `meta.json`. |
| 6 | `sft.subset_data` | Optional. Applies an exact token cap (longer rows are dropped) and adds more epoch orders. |
| 6b | `sft.union_data` | Optional. Rebuilds the dataset when rows are added mid-run. The steps already trained stay byte-identical. Use it with the trainer's `--lr-stretch-from/--lr-stretch-total`. |
| 7 | `sft.fit_subset` | Optional memory check. Builds a dataset of the K longest rows, so one step with `--global-rows K` on K ranks hits the run's peak memory. |
| 8 | `sft.train_fsdp` | The trainer (see below). |
| 9 | `tools.make_fp8` | Optional. Converts an HF bf16 export to block-FP8 (128x128 e4m3 weights with bf16 `*_scale_inv`, the Qwen3.8-FP8 format) for vLLM serving. |

`scripts/sft/build_data.sh` runs steps 1 to 6 with the settings each recipe used.

## Input row format

**Teacher attempts** (input of `build_teacher_rows`): one JSON object per attempt, either one per line in `.jsonl`
files or one per `.json` file. A directory is searched recursively, so keep task files out of attempt directories.

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "reasoning": "the teacher's thinking text, without <think> tags",
 "answer": "the final reply: the test output grid(s) as digit rows, blank line between grids",
 "task_file": "variants/abcd1234__v1.json",
 "task_id": "abcd1234__v1", "unit": "abcd1234__v1", "trace_id": "...", "scored_correct": true,
 "model": "...", "source": "teacher", "cohort": "...", "bench": "...",
 "prompt_tokens": 9000, "completion_tokens": 60000}
```

- Required fields: `messages`, `reasoning`, `answer` and `task_file`. A relative `task_file` is resolved against
  `--task-root`. The task file uses the ARC JSON format, with outputs for the test inputs.
- `task_id` defaults to the task file's stem. The suffixes `__v<k>` (a variant: a color permutation or
  rotation of the base task) and `__t<k>` (a split by test input) are removed when `select_rows` groups rows by
  base task.
- An attempt with `"scored_correct": false` (the generation harness's own score) is skipped.
- The re-score checks the first test output (our attempts targeted single-test-input units). Pass
  `--require-all-tests` to check every test output.

**CoT rows** (output of `build_teacher_rows`/`clean_oov`/`select_rows`, input of `build_cache --cot`): one JSON
object per line.

```json
{"task_id", "trace_id", "unit", "bench", "cohort", "model", "source", "arm": "r2k_zb_max",
 "prompt_messages": [system, user], "reasoning": "...", "answer": "...", "prompt_tokens", "completion_tokens"}
```

`build_cache` also accepts `{"messages": [system?, user, assistant]}` rows whose assistant turn already holds
`<think>...</think>`. The tokenized dataset format is described in `tokenize_cache.py`. In `source.npy`, 0 marks
render traces, 1 aux rows, 2 CoT rows and 3 hinted CoT rows; the trainer logs `loss_<source>` for each.

## Vocabulary pruning

All our models were trained with a pruned vocabulary of K = 128,768 tokens. The keep set holds every added/special
token, all 256 single-byte tokens (so any text stays encodable), every ASCII-only token and every token that appears
in the corpus (our prompts and teacher traces). It also holds tokens made of ASCII plus common symbols
(`--keep-chars symbols`) and the merge ancestors of all of these, padded to a multiple of 128 with the most frequent
remaining tokens. With that set, any text whose original encoding uses only kept tokens tokenizes exactly as before,
and the logits at the kept ids are bit-identical (`validate-model`).

Our keep set file is not included: it is 1.2 MB and its corpus contains our trace data. To rebuild a keep set from
your own data:

```bash
BASE=/models/Qwen3.5-4B OUT=/models/qwen35_4b_k128768 KEEP=keep.json \
TASK_DIRS="ARC-AGI-2/data/training ARC-AGI-2/data/evaluation my_tasks/" EXTRA="teacher_rows.jsonl" \
  scripts/sft/prune_vocab.sh
```

The script runs `keep-set` with `--keep-chars symbols --align 128 --arms r2k_zb_max r2k_zb_two_opt_max qwen38`. A
different corpus gives a different K. A keep set built from one tokenizer can prune any model that shares it (we
used one set for the 2B, 4B and 9B models). Because `clean_oov` (step 2) needs the keep set, prune before building
the data.

## The trainer (`train_fsdp.py`)

- **Parallelism and precision.** Full fine-tuning with FSDP2. Every decoder layer and the root are sharded. The
  fp32 master weights and the AdamW state are sharded; compute runs in bf16 with fp32 gradient reduction. The vision
  tower is dropped for training and copied back from the base model at export.
- **Steps.** A step takes the next `--global-rows` rows of the epoch order. Rows go to ranks by
  longest-processing-time on an attention-aware cost. Each rank runs its rows back to back (batch 1, unpadded, no
  packing) with gradient checkpointing. The loss is the token mean over every label token of the step, with chunked
  cross-entropy so long rows never materialise full logits.
- **Schedule.** Linear warmup, then cosine to `--min-lr-ratio` x peak (HF `cosine_with_min_lr`). It is a function
  of the global step, so a resume reproduces it. `--lr-stretch-from S --lr-stretch-total T0` stretches the rest of
  a T0-step cosine over a longer run (for data added mid-run, see `union_data.py`).
- **Non-finite gradients.** An update with a non-finite gradient norm is skipped on all ranks. Three in a row stop
  the run.
- **Outputs** in `--out`:
  - `ckpt/step_N`: a resumable DCP checkpoint (the latest one is kept, plus `--save-at-milestones`);
  - `milestone-N`: an HF bf16 export at `--milestone-fracs`/`--milestone-steps`;
  - `train_log.jsonl`, `trainer_state.json` and `run_plan.json` (from `--plan-only WORLD`).
- **Resume and stop.** Re-run the same command to resume from the newest complete checkpoint. `--stop-file PATH`
  saves and exits after the step in which PATH appears; `--stop-at-step N` does the same at step N.
- **Multi-node.** Each rank writes its own checkpoint shards on its node, so no shared filesystem is needed for
  saving. To resume, every node needs the newest checkpoint's complete file set, so sync `ckpt/step_N` across
  nodes first.
- **Attention kernels.**
  - `--attn sdpa` (default): PyTorch SDPA, with a hook that turns transformers' explicit causal mask into
    `is_causal=True` so the flash backend is used.
  - `--attn fa3` (Hopper): FlashAttention-3. Fetch it once with
    `python -m arcsolver.sft.attn_backends fetch-fa3 --dest DIR` and pass `--fa3-dir DIR`, or let the `kernels`
    package download it. Check it with `python -m arcsolver.sft.attn_backends selftest --fa3-dir DIR`.
  - `--attn fa4` (Blackwell): FlashAttention-4. See `REQUIREMENTS.txt` for the side-directory install, then pass
    `--fa4-dir DIR`.
- **Memory levers** (none of them changes the math):
  - `--act-gpu-budget-gb G [--act-arena-gb A]` keeps up to G GiB of a row's saved activations on the GPU and puts the
    excess in a pinned host arena;
  - `--offload-activations-min-len N` is the older whole-row variant;
  - `--cpu-offload-optim` keeps the sharded fp32 state and AdamW on the CPU.
- **Speed levers.**
  - `--recompute selective` keeps the attention kernels' outputs through the recompute (exact);
  - `--fused-kernels compile_dynamic` gives an exact compiled RMSNorm and SwiGLU;
  - `--fused-kernels liger` uses Liger kernels (needs `liger-kernel`).
- **Experimental.** These are kept for completeness and none of our recipes used them:
  - `--zero3` and `--layerwise-load` (per-row reduce-scatter and layer-by-layer loading, prepared for a 27B model);
  - `--select-top-frac` and `--select-weight` (loss on the highest-loss tokens only: the pure variant collapsed in
    our test);
  - `--fp8` (torchao float8);
  - `--lr-schedule wsd`;
  - `--compile-norm-mlp` (static shapes, about 3x slower in SFT; use `--fused-kernels compile_dynamic`).
- **Testing.** `--cpu-test` runs on the CPU with gloo, for tiny models.

**Blackwell (B200/B300) note:** set `FLA_TILELANG=0`, or the Gated DeltaNet layers (fla) run about 2.5x slower.
`scripts/sft/train_9b_2node.sh` exports it.

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, as all recipes do: rows change length every micro-step.

## Recipes

These are the exact settings of our runs. Optimizer settings shared by all of them:

- AdamW with betas 0.9/0.999, eps 1e-8 and weight decay 0.001 (matrices only);
- grad clip 1.0;
- `--merge-tail`: an epoch's last chunk with fewer rows than ranks joins the previous step.

### 4B ("round 6"): `scripts/sft/train_4b_round6.sh`

- **Model and hardware.** Vocab-pruned Qwen3.5-4B; 1 node of 8 H200 GPUs with FA3.
- **Schedule.** Peak LR 1e-4, 3% warmup, cosine to 1e-6, 64 rows per step, 3 epochs. HF export every 0.5 epoch,
  checkpoint every 20 steps.
- **Data.** Built with `build_data.sh` using `EPOCHS=3 MAX_SEQ=262144 SEED=20261019` and no cap.
- **Throughput.** About 73 s/step (about 63k tokens/s) on 1.76B tokens per epoch.

```bash
ATTEMPTS=attempts/ TASK_ROOT=tasks/ ORIG_TOKENIZER=/models/Qwen3.5-4B KEEP=keep.json MODEL=/models/qwen35_4b_k128768 \
EVAL_DIR=ARC-AGI-2/data/evaluation WORK=work/r6 OUT=data/r6_tok EPOCHS=3 MAX_SEQ=262144 SEED=20261019 \
  scripts/sft/build_data.sh
DATA=data/r6_tok INIT=/models/qwen35_4b_k128768 OUT=runs/r6 FA3_DIR=kernels/flash-attn3 scripts/sft/train_4b_round6.sh
```

### 9B: `scripts/sft/train_9b_2node.sh`

- **Model and hardware.** Vocab-pruned Qwen3.5-9B; 2 nodes x 8 B300 GPUs with FA4 and `FLA_TILELANG=0`.
- **Schedule.** Peak LR 8e-5, 40 warmup steps, cosine to 1e-6 (`--min-lr-ratio 0.0125`), 64 rows per step, a
  10-epoch plan with an HF export every 0.5 epoch.
- **Data.** Built with `EPOCHS=3 MAX_SEQ=262144 SEED=20261019 CAP=163840 CAP_EPOCHS=10 CAP_SEED=20261026`.
- **Stretched cosine.** After 1,250 steps (2.5 epochs), new rows were added with `union_data.py`. The run resumed
  with `STRETCH="--lr-stretch-from 1250 --lr-stretch-total 5000"`, which stretches the rest of the original
  5,000-step cosine over the 5,390-step union plan, and with
  `LEVERS="--fused-kernels liger --recompute selective"` (about 1.14x faster).
- **Throughput.** About 30 s/step (about 145k tokens/s).

```bash
# on each node (NODE_RANK=0 and 1); MASTER_ADDR = node 0's address on the inter-node network
DATA=data/9b_tok INIT=/models/qwen35_9b_k128768 OUT=runs/9b NODE_RANK=0 MASTER_ADDR=node0 FA4_DIR=/opt/fa4_pkgs \
  scripts/sft/train_9b_2node.sh
# continuation on added rows:
python -m arcsolver.sft.union_data --old data/9b_tok --extra data/new_tok,2 --out data/9b_union --max-len 163840 \
  --keep-epochs 2 --partial-rows 16000 --new-epochs 5
DATA=data/9b_union ... STRETCH="--lr-stretch-from 1250 --lr-stretch-total 5000" \
  LEVERS="--fused-kernels liger --recompute selective" scripts/sft/train_9b_2node.sh
```

In the example, `--keep-epochs 2 --partial-rows 16000` keeps epochs 0 and 1 plus the first 250 steps x 64 rows of
epoch 2, which is exactly what had been trained by step 1,250.

On 141 GB H200 nodes we used `--attn fa3 --act-gpu-budget-gb 28 --act-arena-gb 171` in place of
`--attn fa4 --act-gpu-budget-gb 208`. To find a budget that fits, run a fit check first (see "Fit check" below).

### 4B ablation: `scripts/sft/train_4b_ablation.sh`

- **Model and hardware.** Vocab-pruned Qwen3.5-4B; 8 H200 GPUs with FA3.
- **Data.** One cleaned teacher row per base task, with held-out tasks excluded.
- **Schedule.** Peak LR 5e-5, 3% warmup, cosine to 2.5e-6 (`--min-lr-ratio 0.05`), 32 rows per step, 2 epochs.
- **Length cap.** 131,072 tokens.

```bash
... SELECT_ARGS="--holdout holdout.json" EPOCHS=2 MAX_SEQ=262144 SEED=20261106 CAP=131072 CAP_EPOCHS=2 \
  CAP_SEED=20261106 scripts/sft/build_data.sh
DATA=data/abl_tok INIT=/models/qwen35_4b_k128768 OUT=runs/abl scripts/sft/train_4b_ablation.sh
```

The sweep arms add `select_rows` options to `SELECT_ARGS`: `--n-tasks`/`--frac` (nested task subsets),
`--variants k`, `--all-rows`, `--orig-first` and `--answer-only`. The default `--seed` is the one the sweep used.

## Fit check

```bash
python -m arcsolver.sft.fit_subset --data DATA --out fit/data --rows 8
torchrun --nproc_per_node 8 -m arcsolver.sft.train_fsdp --data fit/data --base-model INIT --out fit/run \
  --global-rows 8 --epochs 1 --warmup-frac 0 --milestone-fracs 1.0 --save-every 1 <the run's kernel/memory flags>
```

It passes if `fit/run/milestone-1/milestone.json` and `fit/run/ckpt/step_000001/COMPLETE` exist.

## FP8 export

```bash
python -m arcsolver.tools.make_fp8 --src runs/r6/milestone-N --dst runs/r6/milestone-N-fp8 --device cuda
```

Block-FP8 weights and scales are bit-identical to what the official Qwen3.8-27B-FP8 pipeline produces from the same
bf16 weights (`fp8_quant.py`). The `quantization_config` written to config.json is the 27B checkpoint's
module list. It is a skip list that loaders only consult for Linear layers.
