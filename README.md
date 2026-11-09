# arc-agi-experience

Code for our ARC Prize 2026 work on ARC-AGI-2. We treat ARC-AGI largely as an *experience problem*: a model learns to
handle new tasks from enough broad, novel and verified tasks. This repository holds the pieces, one module per folder.

| Module | What it is | Status |
|---|---|---|
| [`generator/`](generator) | The ARC task generator. An *arm* (a kind of task, never a rule) seeds model-written ideas, which are checked against the nearest tasks already in the bank. A selector keeps the new and teachable ones. A planner fixes the rule and the rival rules the examples must exclude. A model writes the grids, and a second call re-derives the test answers. A blind solve, a judge and targeted repairs decide what enters the bank. | released: cleaned up, offline-tested |
| [`research-envs/`](research-envs) | The same method applied to two new domains: discovery worlds (probe a hidden device, predict withheld readouts) and construction problems (build graphs or set systems against a machine-checked spec). | released: cleaned up, offline-tested |
| `training/` | Supervised fine-tuning and RL code. | coming |
| `inference/` | The inference stack used under Kaggle's limits. | coming |

Each module is self-contained: it has its own README, install instructions, requirements and tests.

**Data.** The task bank built by the generator, 17,470 verified tasks with the records of how each was designed and
checked, is published as [hl3372/arc-agi-task-bank](https://huggingface.co/datasets/hl3372/arc-agi-task-bank) on
Hugging Face (CC BY 4.0).

**Paper.** Link to come.

> **Release status.** This is a cleaned-up version of our research code, prepared for release. Each module passes its
> offline test suite, but the cleaned-up code has not yet been run end to end against live models. If something
> breaks, please open an issue; we're happy to fix bugs.

## Licence

The code is released under the MIT License (see [`LICENSE`](LICENSE)); copyright (c) 2026 Hazem Lashen. The dataset is
CC BY 4.0.

**Model outputs.** The tasks, records and examples were produced with third-party models: OpenAI models through the
Codex CLI, and DeepSeek and Qwen models through OpenRouter. Our licences cover our rights. The model providers' terms
may also apply to model-generated content, for example terms on using outputs to develop competing models. Check them
for your use.

## Citation

To be added with the paper.
