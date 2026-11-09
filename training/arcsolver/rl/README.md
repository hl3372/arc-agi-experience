# `arcsolver.rl`: GRPO-family RL for long-reasoning ARC solvers

This is the RL trainer behind the released 4B and 9B ARC-AGI-2 models. It runs on-policy RL with verifiable rewards
on Qwen3.5 models with very long rollouts (up to 147,000 generated tokens). vLLM sampler engines and an FSDP2 trainer
time-share the same GPUs, and the trainer re-syncs the sampler's weights after every step. The reward is binary exact
match on the ARC test outputs.

- Entry point: `torchrun ... -m arcsolver.rl.train`, as in [`scripts/rl/`](../../scripts/rl).
- Prompts and scoring come from `arcsolver.prompt.arc_render` (the r2k_zb_max rendering, answer parsing, `score_task`).
- Python 3.12, two environments: a training one (torch 2.10) and a vLLM 0.30 one. Pinned versions are in
  [`REQUIREMENTS.txt`](REQUIREMENTS.txt).

## Architecture

### One step

```
rank 0                                         all ranks (FSDP2 policy, fp32 masters sharded)
------                                         ---------------------------------------------
draw P tasks from the pool order
  -> per group: split test, augmentation
sample P x G rollouts on the vLLM engines      (idle; trainer state on CPU when --timeshare 1)
  (streaming dispatch, FP8 weights + FP8 KV,
   MTP speculative decoding, per-request seeds,
   top-k log-probs for score centering)
score rollouts (exact match), group advantages,
drop groups without signal
engines sleep  ------------------------------>  reload trainer state to GPU
scatter rollouts to ranks (cost-balanced)  -->  per rollout: forward + loss + backward (fp32 grad sum)
                                                clip, AdamW step
gather bf16 weights, quantize to the sampler's FP8 format -> --sync-dir (one copy per node)
engines wake, reload the new weights, acknowledge the new policy id
probe (every --probe-every steps): engine 0 vs the trainer on the same tokens
checkpoint (DCP shards + HF/FP8 exports), metrics, optional eval
```

- **Sampling.** Rank 0 owns one vLLM engine per GPU (`vllm_worker.py`, started with `CUDA_VISIBLE_DEVICES=g` in
  the vLLM environment, reached over a local socket). With `--sampler-stream-seqs N` every engine keeps N single
  samples decoding and pulls new ones from a shared queue as samples finish (continuous refill, KV guard
  `--sampler-kv-high`). The sampled token ids and the sampler's own log-probs are used for training as they are, never
  re-tokenised. With `--sc-topk k` the sampler's top-k ids and log-probs at every position go to `--topk-dir`.
- **Time-sharing (`--timeshare 1`).** The engines sleep while the trainer trains. `--engine-sleep-level 1` backs the
  weights up to host RAM, which keeps the MTP draft head intact. The trainer's fp32 masters and AdamW state move to CPU
  while the engines sample. Every GPU runs both, one at a time.
- **Training.** Every rank holds the FSDP2 policy (`fsdp.py`): fp32 master weights, bf16 compute, fp32 gradient
  reduce-scatter, gradient checkpointing, a chunked LM head (no full-vocab fp32 logits). Rank 0 assigns the rollouts
  that carry a gradient to the ranks. `--balance lpt` balances by an attention-aware cost `n + n²/q` and reduce-scatters
  once per minibatch; `equal` uses equal counts. Each rollout runs one forward/backward. A step is one optimizer update
  (`--minibatch-max 1`).
- **Weight sync.** With `--sampler-fp8 1` the trained weights are gathered one tensor at a time and re-quantized on the
  GPU with `fp8_quant` (128×128 blocks, e4m3) into exactly the format of the FP8 sampler checkpoint. Tensors the trainer
  does not hold (vision tower, MTP head) are copied once from that checkpoint. Every engine reloads. A policy id (a
  hash of the weights manifest and the serving settings) is stamped on every sample, and samples of any other policy
  are refused.
- **Checkpoints.** `--save-every` writes `checkpoints/step_N/` with DCP shards (`dcp_model/` + `dcp_optim/` with
  `--dcp-split 1`), `model_fp8/` (what the engines restart from), an optional bf16 HF `model/` export
  (`--bf16-export-every`), `state.json` (prompt order, objective snapshot, pool manifest hash) and `COMPLETE`. Rerunning
  the same command resumes from the newest complete checkpoint. The resume is refused if the objective flags or the
  prompt pool changed; `--allow-objective-change KEY,...` accepts named changes.
- **Objective contracts (`--objective-contract NAME`, `contract.py`).** A contract is the full set of objective and
  workload flags a run must have. The trainer refuses to start if any flag differs, so a run cannot silently inherit a
  parser default. Some defaults are legacy (`--beta-loop 0.3`, `--drop-zero-var total`): always pass the flags as the
  launchers do.

### Single-node and multi-node

- **Single node:** `torchrun --standalone --nproc_per_node 8 -m arcsolver.rl.train ...`. Rank 0 starts the 8 engines
  itself.
- **Multi-node (`--node-sampler-services 1 --sampler-service-addrs ADDR0,ADDR1,...`):** each node's local rank 0
  starts `node_service.py`, which owns that node's engines. Global rank 0 reaches every node's service over
  authenticated TCP (a random key per launch, port `--sampler-service-port + node`), so all GPUs of all nodes sample
  one step together.
  - Each sync goes to a fresh version directory `--sync-dir/v<step>`, one copy per node. The copies' manifests must
    agree before any engine reloads, and the previous version is removed only after every engine acknowledged the new
    one.
  - Top-k files stay on the node that sampled them; those rollouts are trained by a rank of the same node.
  - `--out` may be node-local (each node writes its own ranks' shards) or a coherent shared filesystem. It may not be a
    FUSE object-store mount. Before a resume every node needs the whole checkpoint, and the trainer checks this.
- **Both modes:** `scripts/rl/grpo9b.sh` uses the service path even on one node (`NNODES=1`).

## Objective

For a task with `G` rollouts `y_1..y_G` (one group):

**Reward.** `R_i = 1` if the attempt-1 grid(s) after `</think>` match the test output(s) exactly, else 0. A test may
list `alt_outputs`, which also count as right. With `--outcome-require-stop 1` a rollout that hits the token cap
(no EOS) scores 0 and stays in its group. Multi-test tasks: with `--train-split-tests` each group sees one seeded test
input as a single-test task.

**Group filtering.** With `--drop-zero-var outcome`, groups with no outcome variance (0/G or G/G right) are dropped.
Tasks whose group is all right are never drawn again with `--pool-drop-solved 1`.

**Advantages** (`rewards.group_advantages`; `m`, `s` = the group's mean and sample std, ddof 1), broadcast to every
completion token:

| `--adv-norm` | `A_i` |
|---|---|
| `std` (GRPO) | `(R_i − m) / (s + ε)`, `ε = --adv-eps` (1e-4) |
| `mean` (Dr. GRPO) | `R_i − m` |
| `maxrl` (MaxRL, arXiv 2602.02710) | `(R_i − m) / (m + ε)`, 0 for a group with `m = 0` |
| `maxrl_floor` | `(R_i − m) / max(m, f)`, `f = --adv-floor` |

Floored MaxRL weights rarely solved tasks up (≈ `1/p` instead of GRPO's `1/√(p(1−p))`) but caps the weight at `1/f`.
Truncated rollouts count as `R = 0` in `m`. `--trunc-neg-scale c` then multiplies a truncated rollout's (negative)
advantage by `c`.

**Policy-gradient term**, per token `t` of rollout `i`, with `p` the trainer (bf16) and `q` the sampler (FP8):

- `--pg-loss ppo`: `ℓ_t = −min(ρ_t A_i, clip(ρ_t, 1−ε, 1+ε_high) A_i)`, `ρ_t = p(y_t)/q(y_t)`.
  With `--ppo-old-logp trainer` the denominator is the trainer's own log-prob at the start of the step instead.
- `--pg-loss tis_sc` (both released recipes): truncated importance sampling + score centering (`sc.py`; Marek &
  Ryabinin, Eq. 11–12):

  ```
  g_t = w_y · log p(y_t) − Σ_{v ∈ top-k(q)} sg[q_v w_v − α p_v] · log p(v)
  w_v = min(p_v/q_v, C),   α = ρ · min(1/ρ, C),   ρ = (1 − Σ_head q) / (1 − Σ_head p)
  ℓ_t = −A_i · g_t
  ```

  `C = --tis-cap` (2) and `k = --sc-topk` (32). The centering term removes the drift that a mismatched sampler adds to
  the gradient (`E_q[A] · Σ_v q_v ∇log p_v ≠ 0` when `q ≠ p`). No PPO ratio is used: with one update per step the
  importance weight carries the trainer/sampler difference.

**Reference KL** (`--kl-coef β`, k3 estimator, gradient through `log p` only, `--ref-model` = the start checkpoint):

```
k3_t = exp(r_t) − r_t − 1,   r_t = log π_ref(y_t) − log p(y_t)   (clamped at 10)
```

**Loss aggregation:**

```
L = (1/N) Σ_i Σ_t (ℓ_{i,t} + β k3_{i,t})
--loss-agg token  (DAPO token mean):  N = all completion tokens of the step's kept rollouts
--loss-agg seq    (mean of per-rollout means):  rollout i is divided by n_kept · T_i
```

**Prefix groups** (`--prefix-groups N --prefix-every K`): on steps divisible by `K`, `N` groups continue a frozen
prefix of one of the previous step's rollouts.

- Sources come from groups that were not all right, at most `--prefix-max-zero` of them from 0-right groups.
- The cut lies in `[--prefix-min-start, R − --prefix-end-margin]` of the source's reasoning (`R` = its length),
  snapped back to a newline. With probability `--prefix-early-prob` it falls in the first `--prefix-early-frac` of `R`.
- The `G` continuations are rewarded on prefix + continuation and trained on the continuation tokens only.

**Optimizer.** AdamW (β1 0.9, β2 `--adam-beta2`, ε 1e-8), weight decay `--weight-decay` (0 = none), grad clip 1.0,
linear warmup over `--warmup-steps`, then a constant LR. Input embeddings are frozen with `--freeze-embed 1`.

## Released recipes

### 9B GRPO (`grpo9b_v3`, released checkpoint at step 34): [`scripts/rl/grpo9b.sh`](../../scripts/rl/grpo9b.sh)

| | |
|---|---|
| contract | `grpo_dapo_kl_sc_v1` for steps 1–22 (`PHASE=1`), then `grpo_dapo_kl_sc_prefix_v1` (`PHASE=2`) |
| batch | 272 tasks × 6 rollouts per step, one update per step; seed 1 |
| reward | binary exact match; capped rollouts score 0 and stay in their group; no forced answers |
| groups | 0/6 and 6/6 dropped; a task with a 6/6 group is never drawn again |
| advantage | group std (ddof 1, ε 1e-4), no clipping |
| loss | DAPO token mean; TIS + score centering (top-32, TIS cap 2), no PPO ratio |
| KL | k3 × 0.001 to the frozen start model |
| optimizer | AdamW, no weight decay, LR 2e-6 constant after a 10-step warmup, embeddings frozen |
| prefix groups | PHASE=2: even steps, 68 of 272 groups; ≤ 11 sources from 0-right groups (cut in their first half); cuts ≥ 2,500 tokens in and ≥ 10,000 before `</think>`; 80% of cuts in the first 60% |
| rollouts | FP8 vLLM (FP8 weights + FP8 KV), MTP 3 speculative tokens (probabilistic drafts, exact), T 1.0, top-p 1, max 147,000 new tokens, prompts ≤ 49,152 tokens |
| augmentation | ⅓ of groups see the task as given; the rest see a random dihedral × permutation of colours 1–9 |
| trainer | FA4 attention (Blackwell), selective attention recompute, fused LM head, LPT balance |
| hardware | 2 × 8 B300, ≈35 min per step |

### 4B floored MaxRL (`rl4b4b`; `rl4b3a` is a variant): [`scripts/rl/rl4b_maxrl_floor.sh`](../../scripts/rl/rl4b_maxrl_floor.sh)

| | rl4b4b (the launcher's defaults) | rl4b3a |
|---|---|---|
| advantage | `A = (R − m) / max(m, 0.2)` | floor 0.25 (0.2 from its step 85) |
| truncated rollouts | `R = 0`, in the loss at full weight (`--trunc-neg-scale 1`) | at 0.3 × A |
| forced answers | none | a right forced answer on a capped rollout earns 0.5 (`--train-budget-force 2048`) |
| batch | 110 × 8 | 96 × 8 |
| LR | 1.5e-6 | 1e-6 |
| rollout cap | 147,000 | 65,536, raised during the run to 101,376 |

Shared by both runs:

- Sequence-level aggregation (`--loss-agg seq`).
- TIS + score centering (top-32, cap 2); no KL; one update per step; 10-step warmup.
- Embeddings trained (tied with the LM head).
- Split tests and ⅓ unaugmented groups.
- Prefix continuations: 24 groups every 3rd step (+6 prompts).
- Pairwise length shaping (λ 0.075, δ 25,000 tokens, groups with ≥ 4/8 right, not all-right groups).
- Evals of the ARC-AGI-2 evaluation set every 10 steps.

The run-specific focus lists (`--pool-focus-*`) are optional and not shipped. The rl4b3a settings changed during that
run, so its row is a summary, not a launcher.

### Preparing a run

1. Install the two environments (`REQUIREMENTS.txt`).
2. Export the FP8 sampler checkpoint from the bf16 start model:
   `python -m arcsolver.rl.make_fp8 --src MODEL --dst MODEL_FP8`. This is the same quantizer the weight sync uses.
3. Lay out the task pool and point the launcher at it: `BANK_DIR`, `ARC_TRAIN_DIR`, and `ARC_EVAL_DIR`, which is
   required because it keeps the eval tasks out of the pool.
4. Set `VLLM_PYTHON`, plus `WANDB_API_KEY` for W&B (optional). Then run the launcher (`DRY=1` prints the command).

## Data formats (no data is shipped)

**Task file:** one ARC task per `*.json`; the file stem is the task id.

```json
{"train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}, ...],
 "test":  [{"input": [[...]], "output": [[...]], "alt_outputs": [[[...]]]}]}
```

`alt_outputs` is optional: other answers that count as correct.

**Pool directories:**

| flag | env default | content |
|---|---|---|
| `--bank-dir` | `$ARC_RL_BANK_DIR` | generated tasks, source `bank` (`--pool bank` or `mixed`) |
| `--arc-train-dir` | `$ARC_RL_ARC_TRAIN_DIR` | e.g. ARC-AGI-2 training, source `arc_train` (`--pool arc_train` or `mixed`) |
| `--arc-eval-dir` | `$ARC_RL_ARC_EVAL_DIR` | ARC-AGI-2 evaluation tasks; **required** |
| `--bank200-dir` | `$ARC_RL_BANK_EVAL_DIR` | optional bank holdout for in-run evals: `tasks/*.json` + `seen_split.json` = `{"seen": [ids], "unseen": [ids]}` |
| `--stage2-cache` | `$ARC_RL_SFT_CACHE` | optional SFT data cache (`{"examples": [{"prov": "trace\|cohort\|task_id\|..."}]}`): marks pool tasks as seen in SFT, for logging only |

- The pool drops a task that matches an evaluation task by id or by rendered content. It also drops holdout tasks and
  duplicates, and prompts longer than `--max-prompt-tokens`.
- The pool is written to `OUT/pool.json` and its manifest to `OUT/pools/`.

**Optional run files:**

| flag | format |
|---|---|
| `--pool-file` | measured success rates `{task_id: rate}` (or `{"tasks": {id: {"budgets": {"<B>": {"success": r}}}}}`) for `--pool-filter` / `--pool-oversample` |
| `--pool-focus-file`, `--pool-focus-cap-file` | JSON lists of task ids |
| `--eval-logit-bias` | `{"logit_bias": {token_id: bias}}` |
| `--sampler-control-file` | JSON `{stream_seqs, kv_high, minibatch_max}`, re-read every step to tune without a restart |

## Run directory (`--out`)

| path | content |
|---|---|
| `log.txt` | human-readable step lines |
| `metrics.jsonl` | every metric per step (W&B mirrors it) |
| `launches.jsonl` | each launch's argv and resolved objective |
| `pool.json`, `pools/`, `eval_set.json` | the prompt pool, its manifest and the eval set |
| `rollouts/step_N.jsonl.gz`, `rollouts/step_N.tokens.npz` | every rollout's text, scores and advantage, plus its exact prompt and completion token ids |
| `evals/step_N.{json,jsonl.gz,COMPLETE}` | eval results |
| `checkpoints/step_N/` | `dcp_model/`, `dcp_optim/`, `model_fp8/`, `model/`, `state.json`, `COMPLETE` |
| `engines/*.log` | engine logs |

Scratch:

- `--sync-dir` (default `/dev/shm/rl_sync_<run>`): the weights handed to the engines.
- `--topk-dir` (default `/dev/shm/rl_topk_<run>`): the top-k arrays.
- `--rollout-store-dir` and `--eval-partial-dir` (optional): every finished sample is saved as it arrives, so an
  interrupted step or eval resumes without redoing those samples.

## Environment variables

| variable | used by | effect |
|---|---|---|
| `WANDB_API_KEY`, `WANDB_PROJECT` | trainer | W&B logging is on only with a key (`--no-wandb` turns it off); `--wandb-project` defaults to `$WANDB_PROJECT` or `arc-rl` |
| `ARC_VLLM_PYTHON`, `ARC_VLLM_ENV_SCRIPT` | trainer | defaults of `--vllm-python` and `--vllm-env-script` |
| `ARC_RL_STUDENT`, `ARC_RL_*_DIR` | trainer | defaults of the path flags above |
| `ARC_EVAL_FORCE_SUFFIX` | trainer | budget-forcing suffix (backslash escapes decoded); default `"\n</think>\n\n"` |
| `ARC_DRAFT_SAMPLE_METHOD=probabilistic` | engines | MTP drafts sampled from the draft distribution (exact, more accepted tokens at T 1) |
| `ARC_REJECTION_SAMPLE_METHOD` | engines | vLLM's rejection-sampling variant |
| `ARC_GDN_PREFILL_BACKEND` | engines | vLLM's GatedDeltaNet prefill backend (`triton` in the 9B run) |
| `ARC_FA3_DIR`, `ARC_FA4_DIR` | trainer | FlashAttention-3 / -4 locations for `--trainer-attn fa3` / `fa4` |
| `FLA_TILELANG=0` | trainer | on Blackwell; with TileLang kernels fla trains far slower there |
| `ARC_ENGINE_GPUS` | `scripts/rl/engine_env.sh` | engine g uses the g-th entry of this GPU list |

## Modules

| module | role |
|---|---|
| `train.py` | entry point: distributed flags, the step loop, sampling orchestration, sync, checkpoints, evals |
| `config.py` | base flags, reward config, pool building and order, split tests, augmentation, slice metrics |
| `contract.py` | objective contracts and the resume check |
| `rewards.py` | exact-match outcome and optional shapings; group advantages; length shapings; forced-answer credit; step metrics |
| `loss.py` | chunked log-probs, PPO-clip loss, KL k3, normalisers |
| `sc.py` | TIS + score centering |
| `fsdp.py`, `fsdp_layout.py`, `hf_utils.py` | FSDP2 policy, offload/reload for time-sharing, weight sync writers, DCP checkpoints, HF/FP8 exports |
| `fp8_quant.py`, `make_fp8.py` | block-FP8 quantizer and the sampler-checkpoint exporter |
| `workers.py`, `vllm_worker.py`, `node_service.py`, `sampler_dispatch.py` | engine processes, the multi-node service, static/queue/streaming dispatch |
| `pool.py`, `mixorder.py`, `frontier.py`, `augment.py`, `core.py`, `data.py`, `identity.py` | pools and prompt orders, augmentation, prompt rendering, eval set, run identities |
| `attn_backends.py`, `trainer_speed.py`, `hostmem.py` | FA3/FA4 attention, opt-in trainer speedups, host-memory metrics |

## Experimental options

These are kept because removing them would touch the step loop, but neither released recipe uses them. Each is off by
default unless noted.

| option | what it does |
|---|---|
| `--dyn-target-groups` | dynamic sampling: rounds of prompts until N informative groups |
| `--len-abs-alpha`, `--len-correct-alpha`, `--len-share-lambda`, `--len-allcorrect-beta` | other length shapings |
| `--len-overlong-*` | DAPO soft overlong penalty (contract `grpo_dapo_kl_sc_prefix_v2`) |
| `--lr-ramp-*` | LR ramp (contract `grpo_dapo_kl_sc_prefix_v3`) |
| `--pg-loss ppo --ppo-old-logp trainer --minibatch-max 2` | PPO clip-higher (contract `grpo_dapo_clip_v1`; ran for one step only) |
| `--mask-truncated` | DAPO overlong filtering |
| `--beta-loop`, `--w-loop-seq` | loop penalties. **Note:** `--beta-loop` defaults to 0.3; the released recipes set it to 0 |
| `--w-format` | format reward |
| `--pool-order frontier` | revisit schedule (`frontier_mix` revisits too) |
| `--train-force-inline`, `--eval-force-inline` | forced answers queued on the engines |
| `--activation-offload-mb`, `--defer-optimizer-reload`, `--fsdp-keep-gathered 0` | memory levers for larger models |

Not included:

- the trained MTP draft heads and the MTP draft-window / XQA kernel patches used for extra sampling speed (they do not
  change the samples);
- the task pools, rollouts, checkpoints and eval sets.

## Tests

```
python -m pytest arcsolver/rl/tests
```

The CPU tests cover rewards and advantages, contracts, pool orders, prefix groups, forced answers, length shaping,
split tests, sampler dispatch and identities. Tests that need torch skip without it.
