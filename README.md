# research-envs

Two small demonstrations that a method for generating ARC tasks with language models carries over to domains that
have nothing in common with ARC grids:

- **Domain A, discovery worlds** (`research_envs/discover/`). A solver probes a hidden, deterministic simulated
  device through a fixed interface (abstract actions, integer readouts, a budget of 40 experiments), then predicts
  the readouts of 30 withheld experiments.
- **Domain B, construction problems** (`research_envs/construct/`). A solver must construct a finite graph or set
  system: optimize an objective under constraints, or find a counterexample to a claim. Every problem has a formal
  spec in a small, validated expression language, checked by a trusted evaluator.

The 20 tasks picked for the paper (10 worlds, 10 problems) are in [`examples/`](examples/README.md) with their full
definitions and answers. The design notes are in [`docs/`](docs/).

> **Release status.** This is a cleaned-up version of our research code, prepared for release. It passes its offline
> test suite (136 tests) and validates all 20 example tasks, but the cleaned-up code has not yet been run end to end
> against live models. If something breaks, please open an issue; we're happy to fix bugs.

**Related:** the ARC-AGI task bank these methods came from is published at
[hl3372/arc-agi-task-bank](https://huggingface.co/datasets/hl3372/arc-agi-task-bank) on Hugging Face; the ARC task generator
code is released alongside this repository.

## How the ARC method maps onto the two domains

| ARC generation step | Domain A: discovery worlds | Domain B: construction problems |
|---|---|---|
| Pin properties, never rules | Requests pin a demand (critical experiment, hidden variable, order, quantitative law) or a property lever (time-driven, cyclic, aliased sensing, coupled parts); matched controls drop it | Requests pin an object type and a demand (interacting constraints, exploitable structure, local optima, counterexample); matched controls drop it |
| Novel by construction | Behavioural fingerprint of the minimized machine against an archive of textbook mechanisms; a round registry blocks duplicate worlds | Behavioural fingerprint (exhaustive small sizes plus a probe battery) against textbook problems and a zoo of standard objects; a round registry |
| The plan names the shortcuts to defeat | Executable baseline experimenters and learners (sweep, random, symbolic law search, a hidden-counter learner) | Executable search baselines (greedy, hill climbing, kicks, annealing, symmetric search, relax-and-repair, margin search) at fixed budgets |
| Build by execution | The world is written in a bounded world language and compiled to transition tables; queries are sampled by code | The spec runs in our interpreter; the planner's witness program runs in a sandbox and is evaluated like any candidate |
| Blind check for a unique answer | Inverted: every published query must be decided by experiments an R-aware eliminator can afford (local identifiability), with no needle rivals | Replaced: a blind restatement of the statement is compared with the spec on probes and exhaustively on a small sibling instance |
| Adjudicate and repair | Pilot outcomes are scored mechanically per query, at budgets 0, 10 and 40 | Pilot outcomes are classed mechanically (invalid, loophole, below, matches or beats the reference) |
| Meta-review | Between rounds: one named generator fix per failure pattern | The same |

[`docs/DESIGN_A.md`](docs/DESIGN_A.md) and [`docs/DESIGN_B.md`](docs/DESIGN_B.md) give the full mapping, the
gates, the falsifiers and every revision.

## Requirements

- **Linux.** The program sandbox uses Landlock, which needs **Landlock ABI 6 or later (kernel 6.12+)**. Without it,
  domain B cannot build witnesses or run code pilots (it refuses rather than run model-written code unsandboxed),
  and domain A pilots are scored from their direct answers only. `research-envs info` reports the ABI.
- **Python 3.10+** (tested with 3.12) with numpy, networkx and OR-Tools. OR-Tools is not imported by the pipeline;
  generated witness and pilot programs are told they may use it inside the sandbox.
- **The Codex CLI, logged in to one account** (`codex login`), for every model call: idea, select, plan, blind
  restatement, naturalness rating, pilots, the reference improver. The account is `$CODEX_HOME`, or `~/.codex` if
  it is unset. The published runs used `gpt-6.1-sol`.
- **An OpenRouter key, only for the open-weight solver comparisons** (domain A pilots with `--solver or:MODEL`, as
  used for the DeepSeek and Qwen comparisons). Nothing else uses OpenRouter.

## Install

```bash
git clone <this repository> research-envs && cd research-envs
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # the tested versions; or let the next line resolve looser bounds
pip install -e '.[test]'
research-envs info                  # runs folder, Codex account, model, concurrency, sandbox status
```

## Quickstart

Offline, no model calls:

```bash
pytest                                                  # about 140 tests, under two minutes on 20 cores
research-envs validate examples/discover examples/construct     # re-derive the 20 demo tasks (a few minutes)
research-envs validate da_time_3 db_ctl_s1              # one world and one problem, by example id
```

Dry runs of whole rounds with a stub model, from the checkout (no model calls; the stub always proposes the same
task, so a second request in a round is rejected as a duplicate):

```bash
RE_LLM_STUB=tests/llm_stub_a.py research-envs discover round dry_a --requests A2:1 --budget 12
RE_LLM_STUB=tests/llm_stub.py research-envs construct round dry_b --requests control:graph:1 --budget 300
```

With model calls (these spend your Codex account's usage):

```bash
research-envs discover round my_worlds                  # 12 worlds with the final settings (preset demo)
research-envs construct round my_problems               # 12 problems with the final settings (preset demo)
research-envs discover pilot da_time_1 --solver codex:high          # a pilot on an example world
research-envs discover pilot my_worlds --solver codex:low --solver or:qwen/qwen3.8-flash   # solvers on a whole run
research-envs construct pilot db_ctl_s1                 # object and code pilots on an example problem
research-envs validate runs/my_worlds/tasks/da_time_1/dossier.json   # check a generated task offline
research-envs export my_worlds da_time_1                # write it as a self-contained task file
```

A round runs each request in its own process and records every model call, so rerunning the same command resumes
where it stopped (`--attempts N` retries automatically). The published demo rounds took about 2.5 hours (A) and
4.5 hours (B) for 12 requests each on a 20-core machine. Domain A makes about 10-12 model calls per task; the domain
B design budgeted roughly 300k tokens per accepted task.

## Commands and presets

| Command | What it does |
|---|---|
| `research-envs info` | Configuration, Codex account, sandbox status (no model calls) |
| `research-envs discover round NAME [--preset demo\|round2\|round1]` | Generate domain A worlds |
| `research-envs discover requery NAME --from RUN --ids a,b` | Rebuild a run's worlds with the current query rules, reusing their recorded design calls (how the demo's final query sets were made) |
| `research-envs discover pilot TARGET... --solver S` | Run solvers on worlds; `S` is `codex:EFFORT`, `codex:MODEL:EFFORT` or `or:OPENROUTER_MODEL` |
| `research-envs construct round NAME [--preset demo\|round2\|round1\|preview]` | Generate domain B problems |
| `research-envs construct pilot TARGET...` | The explicit-object and code pilots on problems |
| `research-envs validate TARGET... [--full] [--no-sandbox]` | Offline checks; `--full` reruns the whole build and audit (slow) |
| `research-envs export RUN ID...` | Task files from a run's dossiers |
| `research-envs check-isolation` | One model call: can a pilot read a file on this disk? (see below) |

A `TARGET` is a task file, an example id, a run's `dossier.json`, or (for `discover pilot`) a run name.
Presets reproduce the paper's rounds; `demo` is the final configuration and the default:

| Preset | Domain A | Domain B |
|---|---|---|
| `demo` (final) | 3 requests per lever; world-named interface, round registry, quality bar, 4 candidates per request, discriminating queries, answer cap 0.4, "nothing changes" cap 0.3 | 2 D2 graph, 3 D2 set system, 3 D4 graph, 2 + 2 controls; round registry, reference improver, quality bar |
| `round2` | 6 per lever + 8 controls; world-named interface, round registry | 32 requests; round registry, reference improver |
| `round1` | 4 per demand A1-A4 + 16 controls | 32 requests (D1-D4 x graph/set x 2, 16 controls) |
| `preview` | | 6 requests |

`--requests` sets the arms directly (A: `time:3,alias:2,A2:4,control:2`; B: `D2:graph:2,D4:graph:3,control:set:2`).
By default the 20 published examples go on the generator's avoid list (`--no-avoid-examples` turns that off;
`--avoid RUN` adds an earlier run's tasks). Two of the ten published worlds used answer cap 0.25 (`--answer-cap`).

## Configuration

Presets set these only where you have not set them yourself; command-line options override both.

| Variable | Default | Meaning |
|---|---|---|
| `CODEX_HOME` | `~/.codex` | The one Codex account used for every call |
| `RE_CODEX_MODEL` | `gpt-6.1-sol` | Model for every Codex call |
| `RE_CODEX_CONCURRENCY` | 8 | Codex calls at once, machine-wide (across all task processes sharing a runs folder) |
| `RE_CODEX_BIN` | `codex` | The Codex executable |
| `RE_ROOT` | the checkout | Outputs go to `$RE_ROOT/runs/` (git-ignored) |
| `RE_LLM_STUB` | | A Python file whose `call()` replaces the model, for dry runs |
| `OPENROUTER_API_KEY`, `OR_KEY_FILE` | | OpenRouter key, or a file holding it (only for `or:` solvers) |
| `RE_A_BUDGET` | 40 | Domain A experiment budget |
| `RE_A_V2` | demo: 1 | World-named settable values and clearer query wording (the published interface) |
| `RE_A_REGISTRY` | demo: 1 | No two worlds in a round share a fingerprint or structure |
| `RE_A_QUALITY` | demo: 1 | The reviewers' quality bar in the prompts, and the silent-readout gate |
| `RE_A_CANDIDATES` | 3 (demo: 4) | Candidate worlds per idea call |
| `RE_A_QUERY_MODE` | demo: `discriminating` | Every query refutes rival laws; adds "cap raised" rivals |
| `RE_A_ANSWER_CAP` | demo: 0.4 | Max share of queries with the same answer |
| `RE_A_TRIVIAL_CAP` | demo: 0.3 | Max share of "nothing changes" queries |
| `RE_A_COMPUTE_SLOTS` | 3 | Concurrent domain A builds |
| `RE_BUDGET` | 20000 | Domain B battery budget B (persistence is checked at 10 B) |
| `RE_ROUND_REGISTRY` | demo: 1 | No two problems in a round share a behavioural family |
| `RE_IMPROVE` | demo: 1 | A stronger code agent pushes the reference before the pilots |
| `RE_B_QUALITY` | demo: 1 | The reviewers' quality bar in the prompts |
| `RE_COMPUTE_SLOTS`, `RE_PROGRAM_SLOTS` | 5 on 20 cores, 3 | Concurrent domain B compute jobs and sandboxed programs |

## Outputs

`runs/NAME/` holds `requests/ID/` (the idea, select and plan calls), `tasks/ID/dossier.json` (everything about the
task, written last), `tasks/ID/build.json` (the cached build), `tasks/ID/calls/` (every model call: prompt, schema,
response, timing), plus `log.txt`, `progress.json` and, once every task has finished, `summary.json`. Dossiers hold
the private answers: do not publish `runs/` as it is; use `export` to write task files.

## Sandboxes and isolation: read before running pilots

Two kinds of model output are executed or given tools, and they are isolated differently.

**Model-written programs** (domain B witness programs, code pilots and the reference improver; domain A pilots'
predictor programs) run in `research_envs/construct/sandbox.py`, under Landlock, enforced by the kernel. The program
can read only system folders (`/usr`, `/etc`, `/lib`, `/lib64`), the Python installation and the folders holding
numpy, networkx and OR-Tools; it can write only its own fresh temporary folder; TCP bind and connect are denied;
signals and abstract unix sockets are scoped to the sandbox; it gets `no_new_privs`, a 2 GB address-space limit, a
CPU limit and a wall-clock timeout. `tests/test_sandbox_and_checks.py` checks that it cannot read a private file,
this checkout, `runs/` or your home folder, write outside its folder, connect over TCP, or signal another process.
What it does **not** do: Landlock has no rules for UDP, so a program can still send UDP datagrams (DNS included);
`/etc` is readable; it is not a container. Run rounds on a machine without network access if that matters.

**Model calls** (every generation stage and every pilot) run `codex exec --sandbox read-only` in an empty temporary
folder, and the prompt holds only the public task. Read-only mode does **not** stop the model from reading files:
wherever Codex's own sandbox works, a pilot can run shell commands that read anything your user can read, including
`runs/`, where the current round's queries and answers are already written when the pilots run. On the machine that
produced the published results, AppArmor blocked the user namespaces Codex's bubblewrap sandbox needs, so every
shell command a pilot tried failed and the pilots had no file access at all; their answers came from the model
alone. That isolation came from the machine, not from this code. Before trusting pilot scores elsewhere:

- run `research-envs check-isolation` (one model call: it plants a random token under `runs/` and asks the model to
  read it), and treat a leak as invalidating pilot scores on that machine; and/or
- run the pilots in a container or VM that holds nothing the model should not see, and confirm that Codex's shell
  cannot reach the runs folder there.

The code does not enforce pilot isolation, and nothing here claims it does.

## Limits

- **Model reviewers only.** The demo picks were chosen by a panel of two models (Opus 5.5 and GPT-6.1-sol, rubric in
  `docs/REVIEW_RUBRIC.md`). No human reviewed the tasks.
- **Construction problems are easy for frontier models.** On all ten published problems a GPT-6.1-sol pilot matched
  or beat the planner's reference. Domain B shows the method transfers, not that it yields hard problems.
- **No training results.** Nothing here shows the tasks are useful for training.
- **Fairness in domain A is relative to a rival set** centred on the true law (local identifiability), an
  Occam-style assumption stated in `docs/DESIGN_A.md`. Worlds are small, synthetic and noise-free; nothing here is
  scientific discovery about the real world.
- **Pilot scores are single draws** of one model family at the time of the runs.

## Analysis scripts

`scripts/analysis/` holds the scripts behind the design notes' numbers. They run on your own `runs/` (pass a run
name or folder); none ships with data.

| Script | Purpose | Model calls |
|---|---|---|
| `canaries.py` | Disguised textbook problems must match the B archive | no |
| `gate_a.py`, `gate_b.py` | The phase 1 gates on hand-built worlds and problems | no |
| `report.py RUN_DIR` | Domain B round report: classes, audit modes, bars and falsifiers | no |
| `recheck.py RUN` | Re-judge a B round with the current rules | only retries of unusable blind specs |
| `reaudit.py SRC_DIR OUT_DIR` | Rebuild a B round's audits with the current battery | no |
| `qtest_a.py RUN:ID,...` | Rebuild A worlds' queries with the discriminating chooser | no |
| `compare_a.py RUN`, `rescore_a.py RUN` | Compare and rescore domain A solvers (`runs/RUN/validation/`) | no |
| `frontier_fairness.py RUN` | Hard or unfair? Fit of each frontier pilot's own law | no |
| `review_packets.py OUT RUN...`, `review_gpt.py OUT`, `review_summary.py OUT` | The model review panel | `review_gpt.py` only |

## Licence and citation

The code is released under the MIT License (see `LICENSE`). Copyright (c) 2026 Hazem Lashen.

**Model outputs.** The example worlds and problems in `examples/` were written by a model (gpt-6.1-sol through the
Codex CLI) and checked by this code. Solver comparisons can also call DeepSeek and Qwen models through
OpenRouter. We release the examples under the same MIT terms, as far as our rights go. The model providers' terms
may also apply to model-generated content, for example terms on using outputs to develop competing models.
Check them for your use.

Citation: to be added with the paper.
