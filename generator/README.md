# ARC task generator

A pipeline that generates new ARC-style tasks (grid-to-grid reasoning puzzles) with language models, checks them,
and adds the verified ones to a task bank. Every model call goes through the [Codex CLI](https://github.com/openai/codex)
on one logged-in account.

**Data.** The bank our batches built, 17,470 verified tasks with the records of how each was designed and checked, is
published as the [ARC Task Bank](https://huggingface.co/datasets/hl3372/arc-agi-task-bank) dataset on Hugging Face
(CC BY 4.0). New batches can start from it instead of an empty bank (see
[Starting from our task bank](#starting-from-our-task-bank)).

> **Release status.** This is a cleaned-up version of our research code, prepared for release. It passes its offline
> test suite (340 tests, including an end-to-end batch with scripted model responses that commits into a bank built
> from the published dataset), but the cleaned-up code has not yet been run end to end against live models. If
> something breaks, please open an issue; we're happy to fix bugs.

## How it works

A batch is a fixed design of *idea calls* spread over *arms* (directions such as "two levels of organisation both
needed", "work back to what produced the input", "a controlled comparison rules out a plausible rival") and *property
slots* (for example a large constructed output, many separate things per picture, several colours that each do work).
For each idea it keeps, the pipeline develops one task:

1. **Ideas with nearest-bank novelty.** An idea call drafts 12 to 16 sketches. Each sketch is embedded and shown with
   its nearest existing bank tasks; the author drafts further sketches aimed beyond them (explore turn), drops those
   that need no fresh insight, and commits the rest as kernels.
2. **Selector.** One call per idea window ranks the window's kernels against their bank neighbours (and against the
   kernels earlier reference waves of the same batch selected) and keeps the useful, coherent, fairly realizable ones.
3. **Planner with rival rules.** A planner writes a concrete plan: the rule, the training and test pairs, and for each
   strong simpler rival rule the evidence in the pairs that defeats it.
4. **Model-written grids, independently re-derived.** A realizer writes the grids from the frozen plan. Deterministic
   checks validate them, and a technical checker re-derives the test outputs independently from the public grids and
   the plan.
5. **Blind solve.** A solver that sees only the training pairs and test inputs predicts the test outputs and names any
   serious alternative it sees.
6. **Judge.** When the blind solver misses or flags an alternative, a second solver and an adjudicator running on a
   different model decide whether the intended rule is supported or a comparable rival rule is equally supported.
7. **Repair.** A repairable ambiguity or mechanical defect gets a targeted repair (the rule and test answers stay
   frozen), followed by fresh checks and a fresh blind solve.
8. **Bank.** Tasks that pass, and that are not exact or colour-permutation duplicates of the bank or of each other,
   are committed to the bank by an explicit `commit`, with a dossier of every model call behind them.

Runs are resumable at every step: each logical call, attempt, frozen input and decision is recorded under the run
folder, and a resumed run re-verifies what it already has without repeating model calls.

## Models by role

The reference profile (`arcgen/batch/profiles/v1.json`) binds:

| Role | Model | Effort |
|---|---|---|
| Idea calls | `gpt-6.1-sol` (48 of 56 calls); `gpt-6-sol` (8 calls) | high, xhigh or max per call |
| Selector | `gpt-6.1-sol` | xhigh |
| Planner | `gpt-6.1-sol` | xhigh |
| Realizer (writes the grids) | `gpt-6.1-sol` | xhigh |
| Technical check (first) | `gpt-6.1-sol` | high |
| Blind solver (first, and after a repair) | `gpt-6.1-sol` | xhigh |
| Second blind solver | `gpt-6-astra` | xhigh |
| Technical re-check | `gpt-6-astra` | high |
| Ambiguity judge (adjudication) | `gpt-6-astra` | high |
| Ambiguity repair | `gpt-6-astra` | xhigh |
| Mechanical repair | `gpt-6.1-sol` | xhigh |

So the model that generates the tasks is also the first blind solver, while the judge that settles flagged ambiguity
(and the second solver it consults) is a different model. Change `science.models`, `science.roles` and
`science.call_overrides` in a profile to use other models.

## Requirements

- Linux or macOS with Python 3.12 (tested on 3.12.3).
- The Codex CLI installed and logged in to **one** account (`codex login`). The pipeline uses `$CODEX_HOME`, else
  `~/.codex`, or the folder set in `operations.codex_home`.
- An [OpenRouter](https://openrouter.ai) API key in `OPENROUTER_API_KEY`, for the embeddings used by novelty
  retrieval (`openai/text-embedding-3-large`). An empty bank needs no embedding at preparation; our published bank
  needs a one-off embedding of its tasks (see [Starting from our task bank](#starting-from-our-task-bank)).
- Memory: about 260 MB per concurrent Codex call (30 calls need about 8 GB free).

## Install

```sh
git clone https://github.com/hl3372/arc-agi-experience.git && cd arc-agi-experience/generator
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # offline suite, about 4 minutes; no network, no model calls
```

`bin/arcgen` runs with `$ARCGEN_PYTHON` if set, else `python3` on your `PATH` (activate the venv first).

## Quickstart

```sh
export OPENROUTER_API_KEY=...         # embeddings for novelty retrieval
bin/arcgen new demo                   # profile + allocation from the reference design (56 idea calls)
bin/arcgen start demo --note "first batch"
bin/arcgen progress demo --watch      # one status line a minute (Ctrl-C stops watching, not the run)
bin/arcgen summary demo               # results and tokens (also written when the run ends)
bin/arcgen commit demo --dry-run      # verify and stage only
bin/arcgen commit demo                # add the verified tasks to the bank
```

`start` runs the offline test suite if the code changed since it last passed, then prepares the run (bank snapshot,
reference embeddings, design), qualifies it, records your command line as the launch authorization with a usage
reading, and runs the batch under a background supervisor that survives the terminal closing. `stop demo` pauses (running
calls finish), `stop demo --now` stops at once, and `resume demo` continues from saved state. `bin/arcgen --help`
lists every command.

The first batch can start from an empty bank (one is created on first use), from our published task bank (next
section), or from another existing bank folder (see [The bank](#the-bank)).

## Starting from our task bank

The [ARC Task Bank](https://huggingface.co/datasets/hl3372/arc-agi-task-bank) dataset holds the 17,470 tasks our
batches committed. A converter turns it into a bank folder, so that new batches look up these tasks and aim beyond them:

```sh
pip install 'pyarrow==26.0.1' 'huggingface_hub==2.4.0'       # or: pip install '.[hf]'
python -m arcgen.batch.bank_from_hf --repo                    # download (about 260 MB) and write $ARCGEN_BANK
python -m arcgen.batch.bank_from_hf --local <copy> --out <folder>     # or: from a downloaded copy, into a folder
export ARCGEN_BANK=<folder>                                   # when --out is not the default bank
```

- **Where.** The bank is written to `ARCGEN_BANK` (default `workspace/bank`) unless `--out` names another folder. The
  converter refuses a folder that exists and is not empty. Run it before the first batch: an empty bank created by an
  earlier `start` must be moved away first.
- **What.** The task files are byte-identical to our bank's: each is checked against its original SHA-256, and its
  grids against the dataset's table. Each registry entry is the original one, with the dataset's `eval_near_twin` and
  `excluded_from_training` flags added. The dossiers are minimal: the task, its row of the dataset's `tasks` table
  without the grids, and provenance. The full design records stay in the dataset's `records` config; `--with-lifecycle`
  copies each task's lifecycle record into its dossier (about 1.3 GB more). The new bank is checked the way
  `bin/arcgen commit` checks a bank, and is moved into place only if it passes. The full bank takes about 2.5 minutes
  and 1.4 GB of disk.
- **Subsets.** `--limit N` keeps N tasks spread evenly over the commit order. `--exclude-eval-near-twins` leaves out the
  tasks flagged as near-twins, and `--exclude-flag` other flagged tasks (`excluded_from_training`,
  `admitted_via_right_twice`, `owner_override`, `overruled_credible_rival`; see the dataset card).
- **Near-twins of evaluation tasks.** By default the bank keeps the 39 tasks the dataset flags as a twin (2) or
  near-twin (37) of a public ARC-AGI-2 evaluation task, as the dataset does. Only the 7,909 tasks committed by
  mid-October were checked; the 9,561 later ones are `unchecked`, so the flag is a lower bound. If you train a system
  on tasks from this bank, or on tasks generated against it, and report a score on the public ARC-AGI-2 evaluation set,
  report it split by `eval_near_twin`.
- **One-off embedding.** The bank ships no embeddings. The first batch prepared on it embeds the bank's reference texts
  (each task's concept summary and rule) through OpenRouter (`OPENROUTER_API_KEY`, `openai/text-embedding-3-large`):
  17,466 distinct texts of about 2 million tokens in all, sent as 273 requests of 64 texts. At $0.13 per million tokens
  (the model's list price when we wrote this) that is about $0.25. At the 3.5-4.5 s per request our batches saw, it
  takes about 15-20 minutes, once. The vectors (about 1.1 GB) are cached in `workspace/cache/embeddings/` for every
  later batch, and each run folder keeps its own copy (another 1.1 GB per run).
- **First commit.** The dataset has no renders, so the first `bin/arcgen commit` into the bank draws one per bank task
  (about 30 ms each, in worker processes). Every commit also rebuilds the bank's contact sheet, which is about 3 GB for
  the full bank.
- **Licence.** The bank folder includes the dataset's licence. If you share the bank or tasks from it, credit the
  dataset as CC BY 4.0 requires.

## Configuration

`bin/arcgen new <name>` copies the reference profile to `workspace/batches/<name>.json`; edit it before `start`. A
profile has two sections:

- `science`: models and role bindings, idea efforts, planner, arms and property slots (`idea_modifiers`,
  `modifier_slots`, exclusions), selector depth and prompt budget, presentation overlays, colour and pose draws, the
  embedding provider (`embedding`, default OpenRouter `openai/text-embedding-3-large`), and the allocation (which arms,
  models and efforts each idea call uses; see `arcgen/batch/allocations/make_allocation.py`).
- `operations`: how it runs.

| `operations` key | Meaning |
|---|---|
| `workers` | Concurrent Codex calls (default 30). About 27% go to idea and selector calls, the rest to development; development also takes the idea slots once the idea work has ended. `bin/arcgen new --workers N` sets it; `bin/arcgen workers <name> N` changes it later (a running batch picks it up after `stop` and `resume`). |
| `codex_home` | The Codex account folder (default: `$CODEX_HOME`, else `~/.codex`). |
| `codex_executable`, `codex_sha256` | The Codex CLI to run (default `codex` on `PATH`) and an optional pin of its bytes. The launch binds the executable's path and hash; a changed executable stops a resume. |
| `timeout_seconds` | Per-call timeout (default 5400). |
| `usage_credits` | `true` if the account can run past its weekly limit; otherwise a batch pauses at the limit (status `paused_usage`) and continues with `bin/arcgen resume` after the reset. |
| `allow_concurrent_activity` | `true` to launch while other Codex calls run on the same account. |
| `group_size`, `continuous_supply`, `supply_windows`, `idle_supply_windows`, `development_windows`, `pool`, `ramp_up`, `slate_barriers` | Scheduling details; `new` sets sensible values. |

Paths are set by environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ARCGEN_WORKSPACE` | `workspace/` in the checkout | Where batches, runs, the shared embedding cache and the test gate are written. |
| `ARCGEN_BANK` | `$ARCGEN_WORKSPACE/bank` | The task bank folder. |
| `OPENROUTER_API_KEY` | (none) | Embedding requests. |
| `CODEX_HOME` | `~/.codex` | The Codex account, unless the profile sets `codex_home`. |

## Workspace layout

```
workspace/
  batches/<name>.json                profile (science + operations) made by `new`
  batches/<name>.spec.json           the compact allocation spec; <name>.allocation.json is built from it
  runs/<name>/                       the run: frozen config, inputs and design, every model call (prompt, response,
                                     tokens), task states, reports, gallery/index.html
  runs/<name>-ops/                   run.log, progress.log, usage readings, SUMMARY.md, READ_FIRST.json
  runs/<name>-task-commit-<date>/    commit receipts
  bank/                              the task bank (see below)
  cache/embeddings/                  embedding vectors shared by all runs
  gate/                              the code hash that last passed the offline suite
```

### The bank

A bank folder holds `registry.json` (every committed task), `active_registry.json` (the tasks used for novelty
retrieval), `quarantine_registry.json`, `basic_valid_registry.json`, the task files `tasks/<id>.json`, one dossier per
task under `dossiers/` (every model call behind it), renders and a catalog. A run reads only `active_registry.json` and
the task files: it copies (or hard-links) them into the run at preparation, so the bank can grow between batches. A
bank that ships `embeddings/<config hash>/` vectors seeds the shared cache, so its tasks need no new embedding requests.
To use another bank folder, set `ARCGEN_BANK` to it (committing into it also needs its dossiers); our published bank is
converted into one as described in [Starting from our task bank](#starting-from-our-task-bank).
Do not commit one batch while another prepared on the same bank is still to be resumed: a run stops if its bank's
registries changed since it was prepared.

## Costs and expectations

- A verified task took about 6 model calls and 220k-270k tokens in all in our batches (ideas, selection, planning,
  realization, checks, solving, judging and repairs together).
- Our batches of this design yielded about 3.2 verified tasks per idea call; yield varies with the bank, the design and
  the models. `new` prints estimates scaled from those batches: for the 56-call reference design about 180 verified
  tasks, about 1,100 model calls and 44M tokens, and about 7 hours at 30 concurrent calls (10 hours at 20, 19 at 10).
- A subscription account's usage limits, not the token count, decide how much one account can run per week; a batch
  pauses at the weekly limit unless `usage_credits` is set (see Configuration).
- Embedding costs are small: one vector per bank task (cached and reused) and one per drafted sketch or kernel.

Nothing is committed without `bin/arcgen commit`. `runs/<name>-ops/READ_FIRST.json` lists verified tasks worth a human
look before committing (a blind solver that missed, or a long inferred rule); hold any with `commit --hold <file>`.

## Tests

```sh
python -m pytest tests -q
```

The suite is offline: a fixture refuses network connections and Codex calls, model responses are scripted, and
embeddings use a local hashing provider. `tests/test_end_to_end.py` runs a two-window batch from `new` through
`prepare`, `ready`, `authorize`, `run` (idea loop, selector, both reference waves, planning, realization, checks, blind
solve, report and gallery) to `bin/arcgen commit` into a sample bank. `bin/arcgen test` runs the same suite and records
the gate that `start` checks.

## Code layout

| Path | Contents |
|---|---|
| `bin/arcgen` | Command-line entry point (`arcgen/batch/cli.py`). |
| `arcgen/batch/` | The batch pipeline: configuration, design and allocation, bank inputs and the converter from the published bank (`bank_from_hf.py`), idea loop (`idea_v1.py`), selector inputs (`supply.py`), retrieval, scheduler, development (`runtime.py`, `downstream.py`, `gate.py`, `repair*.py`, `lines.py`), reports, gallery and commit. |
| `arcgen/batch/prompts/v1/` | Idea, planner, slot, overlay and repair prompt texts of the reference design. |
| `arcgen/batch/profiles/v1.json`, `arcgen/batch/allocations/` | The reference profile and allocation spec. |
| `arcgen/lib/` | The shared library: model transport, contracts and schemas, the candidate DAG, evidence gates, palette, validation and bank tools. |
| `arcgen/astra_*`, `arcgen/sol_routine_*` | Earlier pipeline layers the batch pipeline builds on (contracts, adjudication, re-checks, repairs, palette allocation, selection); named by the date they were introduced. |
| `arcgen/prompts/`, `arcgen/config/` | Base prompt templates and the base factory configuration. |
| `tests/` | The offline suite and its fixtures. |

## License

The code is released under the MIT License (see `LICENSE`). Copyright (c) 2026 Hazem Lashen.

**Model outputs.** The pipeline drives third-party models: OpenAI models through the Codex CLI, and embeddings
through OpenRouter. The tasks, plans and records it produces are outputs of those models. The MIT License covers
this code only. When you generate or use such outputs, the model providers' terms may also apply, for example
terms on using outputs to develop competing models. Check them for your use.
