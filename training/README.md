# training: supervised fine-tuning, GRPO and evaluation for long-reasoning ARC-AGI-2 solvers

This module holds the code we used to train and evaluate our ARC-AGI-2 models:
- the 4B (Qwen3.5-4B) and 9B (Qwen3.5-9B) solvers;
- the 4B data-scaling ablation.

The method has two stages:

1. **Supervised fine-tuning on long teacher reasoning.** A reasoning model (DeepSeek V4.1 Flash) solves tasks from our
   generated task bank (see [`../generator`](../generator) and the published bank). Only correct solutions are kept, with
   their full reasoning. A smaller model is fully fine-tuned on them with fp32 master weights.
2. **Short RL on top.** GRPO with a binary exact-match reward on bank tasks, sampled with FP8 vLLM engines that share
   the GPUs with the trainer.

| part | what it is | docs |
|---|---|---|
| [`arcsolver/prompt`](arcsolver/prompt) | The task prompt (`r2k_zb_max`: indexed grid rows plus run listings), answer parsing and exact-match scoring. Shared by every other part. | docstrings |
| [`arcsolver/sft`](arcsolver/sft) | Teacher attempts → checked rows → out-of-vocabulary cleanup → selection → stage-2 cache with an eval-leak check → tokenized cache → FSDP2 full fine-tuning (fp32 masters; FA3 on Hopper, FA4 on Blackwell). | [README](arcsolver/sft/README.md) |
| [`arcsolver/tools`](arcsolver/tools) | Vocabulary pruning (Qwen3.5's 248k vocabulary → 128,768 tokens, with a proof-backed check that tokenization is unchanged) and block-FP8 export of the sampler weights. | [README](arcsolver/sft/README.md#vocabulary-pruning) |
| [`arcsolver/rl`](arcsolver/rl) | GRPO trainer: vLLM sampler engines time-shared with an FSDP2 trainer on the same GPUs, one or two nodes. Covers TIS + score centering, a k3 KL to the start model, prefix-continuation groups, floored MaxRL (4B), and FP8 rollouts with MTP speculative decoding. | [README](arcsolver/rl/README.md) |
| [`arcsolver/eval`](arcsolver/eval) | Standalone ARC-AGI-2 evaluation with vLLM: split test inputs, k samples, 147k cap + forced answer, pass@1/2/4. Includes a re-scorer for saved outputs. | [README](arcsolver/eval/README.md) |
| [`scripts/`](scripts) | Example launchers with the exact recipes behind our reported models. | per-part READMEs |

## Install

Python 3.12, Linux, NVIDIA GPUs (H100/H200, or B200/B300 with FA4). The exact versions we ran are pinned in
[`requirements.txt`](requirements.txt). torch comes from the CUDA 12.8 index. The data pipeline's CPU steps and the
re-scorer need far less.

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # trainer, data pipeline, vLLM sampler/eval
pip install -e .                         # the arcsolver package
python -m pytest tests                   # offline tests (no GPU, no network)
```

**Blackwell (B200/B300):**
- Use `--attn fa4`. FlashAttention-3 is Hopper-only. Install FA4 into a side directory as described in the SFT README.
- Set `FLA_TILELANG=0`. Otherwise the model's Gated DeltaNet layers take FLA's TileLang path, which is very slow there:
  our 4B ran at about 40k tokens/s instead of about 105k on 8× B300.

## Quickstart

```bash
# 1. evaluate a checkpoint on the ARC-AGI-2 public evaluation set (4 samples per test input, 8 GPUs)
MODEL=/path/to/checkpoint TASKS=/path/to/ARC-AGI-2/data/evaluation OUT=runs/eval DP=8 bash scripts/eval/eval_arc_agi2.sh

# 2. SFT data: teacher attempts -> checked rows -> cache -> tokens (attempt format: arcsolver/sft/README.md)
ATTEMPTS=... TASK_ROOT=... ORIG_TOKENIZER=... KEEP=... MODEL=... EVAL_DIR=... WORK=... OUT=data/tok \
  bash scripts/sft/build_data.sh

# 3. full fine-tuning with the 4B ablation recipe (8 GPUs)
DATA=data/tok INIT=/path/to/pruned-Qwen3.5-4B OUT=runs/sft_4b bash scripts/sft/train_4b_ablation.sh

# 4. GRPO from an SFT checkpoint (task-pool format and variables: arcsolver/rl/README.md)
bash scripts/rl/grpo9b.sh
```

Every launcher reads its paths from environment variables or arguments; nothing points to a fixed location.

## What the reported numbers used

- **Scores in our writeup** come from our internal evaluation harness, with exactly the settings `arcsolver/eval` implements:
  - the ARC-AGI-2 public evaluation set, one prompt per test input;
  - 4 samples per input, T 1.0, top-p 1;
  - 147,000-token cap plus a 1,024-token forced answer;
  - FP8 weights and KV cache, MTP speculative decoding.
- **The standalone eval has not been run on GPUs against the harness.** Its scorer reproduces the harness counts exactly on our stored eval outputs.
- **Recipes:**

| model | recipe |
|---|---|
| 4B (best, "round 6") | full FT from an RL checkpoint, LR 1e-4 → 1e-6 cosine, 3% warmup, 64 rows/step, best at 2.5 of 3 epochs |
| 9B SFT | LR 8e-5 → 1e-6, 64 rows/step on 2 × 8 GPUs, stretched cosine when data was added mid-run; a uniform mean of 21 fp32 snapshots over the last two epochs started RL |
| 9B GRPO | 272 tasks × 6 rollouts, binary reward, TIS + score centering, KL 0.001, LR 2e-6; the best checkpoint came at step 34 |
| 4B ablation | LR 5e-5 → 2.5e-6, 32 rows/step, 2 epochs, 128k row cap |

## Status

This is a cleaned-up version of the research code we ran, prepared for release.
- Internal paths, infrastructure hooks and experiment notes were removed. Behavior was kept for every option the recipes use.
- The offline tests pass. The GPU paths of the cleaned-up code have not yet been run end to end.
- Please report problems by opening an issue.

## License

MIT (see [`LICENSE`](LICENSE)), copyright (c) 2026 Hazem Lashen. Model weights and data have their own licences.
