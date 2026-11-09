# arc-agi-experience

Code for our ARC Prize 2026 work on ARC-AGI-2. We treat ARC-AGI largely as an *experience problem*: a model learns to
handle new tasks from enough broad, novel and verified tasks. This repository holds the pieces, one module per folder.

| Module | What it is | Status |
|---|---|---|
| [`generator/`](generator) | The ARC task generator. An *arm* (a kind of task, never a rule) seeds model-written ideas, which are checked against the nearest tasks already in the bank. A selector keeps the new and teachable ones. A planner fixes the rule and the rival rules the examples must exclude. A model writes the grids, and a second call re-derives the test answers. A blind solve, a judge and targeted repairs decide what enters the bank. | released: cleaned up, offline-tested |
| [`research-envs/`](research-envs) | The same method applied to two new domains: discovery worlds (probe a hidden device, predict withheld readouts) and construction problems (build graphs or set systems against a machine-checked spec). | released: cleaned up, offline-tested |
| [`aux-tasks/`](aux-tasks) | Procedural generators for auxiliary grid tasks that isolate one skill each (reading sizes and coordinates, counting, reflecting, recolouring, tracing paths, symmetry, cropping by row and column), with up to 10 extra prompt wordings per task. It also has the packing that turns them into SFT rows in varied layouts, encodings, answer styles and ARC-style bundles. Tasks 1-102 are verified and were our aux training data; 103-281 are experimental. | released: offline-tested; rebuilds our aux-pool rows exactly (spot-checked) |
| [`training/`](training) | The solver training and evaluation code. SFT on long, verified teacher reasoning: data pipeline, vocabulary pruning, and an FSDP2 full fine-tuning trainer with fp32 masters. GRPO with vLLM samplers time-shared with the trainer. A standalone ARC-AGI-2 evaluation. | released: cleaned up, offline-tested (GPU paths not yet re-run) |
| `inference/` | The inference stack used under Kaggle's limits. | coming |

Each module is self-contained: it has its own README, install instructions, requirements and tests.

**Data.** The task bank built by the generator, 17,470 verified tasks with the records of how each was designed and
checked, is published as [hl3372/arc-agi-task-bank](https://huggingface.co/datasets/hl3372/arc-agi-task-bank) on
Hugging Face (CC BY 4.0).

**Paper.** Link to come.

> **Release status.** Most of the code here is a cleaned-up version of the research code we ran, prepared for
> release. We didn't have the chance to test the cleaned-up version end to end before the deadline. Each module
> passes its offline test suite, and it should work. The code we ran locally is available on request. Please report
> any bugs you find, by opening an issue, so we can fix them.

## Licence

The code is released under the MIT No Attribution License (MIT-0; see [`LICENSE`](LICENSE)); copyright 2026 Hazem
Lashen. You may use, copy, modify and distribute it without keeping the notice. The dataset is CC BY 4.0.

**Model outputs.** The tasks, records and examples were produced with third-party models: OpenAI models through the
Codex CLI, and DeepSeek and Qwen models through OpenRouter. Our licences cover our rights. The model providers' terms
may also apply to model-generated content, for example terms on using outputs to develop competing models. Check them
for your use.

## Citation

To be added with the paper.
