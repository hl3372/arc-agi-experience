# aux-tasks

Procedural generators for **auxiliary grid tasks**: short ARC-style exercises, each isolating one skill a solver
needs before it can induce rules. Examples are reading a grid's size or a cell's coordinates, counting objects of a
shape, reflecting, rotating or magnifying, finding the background, tracing paths, completing a symmetry, and cropping
rows and columns by number. Each task maps a `(task_id, seed)` pair to one example:

- an `input`: usually one grid (lists of ints 0-9, sides 1-30); a few tasks take text or several named grids;
- an `output`: a grid, a number, a colour label, a list of coordinates or a short text;
- `metadata`: every per-example fact the instruction needs.

The prompt (instruction) is rendered from the metadata. Generation is deterministic: the same `(task_id, seed)` always
gives the same example.

We used these tasks as auxiliary SFT data for our ARC-AGI-2 solver (see [`../training`](../training)). They teach
*grid literacy*: reading and writing grids and applying a stated operation. Few-shot rule induction is the job of the
main task bank. So almost every task states its rule in the prompt and takes a single grid; this is by design.

`packing/` turns the examples into training rows. It has two kinds of rows:
- the classic format;
- varied formats: galleries of grids, several questions per grid, comparisons, ARC-style bundles, and many grid
  encodings and answer styles.

See [`packing/README.md`](packing/README.md).

## Which tasks are verified

256 task ids are registered. [`TASKS.md`](TASKS.md) lists every id, what it asks and its status.

| ids | count | status | in our training data |
|---|---|---|---|
| **1-102** | 98 ids, plus 10 retired ids that raise an error | **verified** | yes: our aux pool uses only these |
| 103-281 | 158 ids | experimental: reviewed by automated passes only | no |

Ids 1-102 include letter-suffixed ids (`5a`, `15a`, `46a`, `68a`, `68b`).

**Tasks 1-102.** Every task in this range went through a review-and-rebuild pass in September 2026 (one family at
a time). The steps were:

1. **Review.** Each family was reviewed for:
   - correctness;
   - determinability: every example has exactly one answer that a careful reader of the prompt and the input
     reaches;
   - informativeness;
   - robustness to colour-symbol remapping;
   - diversity.
2. **Rebuild.** Where the review found problems, the family was rebuilt in its own module against a written design.
   An independent verifier then signed off each rebuild, using its own from-scratch solver over thousands of seeds
   per task.
3. **Human review.** The project lead reviewed examples in a review app throughout the pass, and the findings were
   fed back.
4. **Final sweep.** A sweep over all of 1-102 combined a mechanical scorecard (3,000 seeds per task: sizes, colours,
   answer balance, leak features), calibrated reviewers and a prompt auditor. The few remaining problems were fixed.

The standing rules the tasks meet:

- **One answer per example.** Variety is never bought with ambiguity. Anything a solver would need to break a tie is
  stated in the prompt.
- **Coverage close to ARC.** Grid sides run independently over most of 1-30, and all 10 colours are used. Backgrounds
  can be plain or textured, and object shapes and placements look drawn.
- **Symbol-safe prompts.** A prompt never contains a raw colour digit, and colours are named only through the palette
  helpers in `prompts.py`. So an example can be re-rendered under any symbol palette (`symbol_remap.py`).
- **No give-aways.** A prompt does not hand over what the solver should infer, for example the background colour
  when finding it is part of the skill.
- **Multiple wordings.** Every task has 6-10 extra approved wordings (628 in total, in `prompt_variants.json`). Each
  wording is checked against at least 2,000 coverage seeds; see `python prompt_variants.py validate`.

**Tasks 103-281** (the `core2_simple` and `core2_extra` modules) were written earlier. They had two automated passes:

1. a correctness and diversity review, whose bug fixes are applied;
2. a diversity-widening pass, in which an independent solver gated each change.

A blind-solve check covered every task: a strong model solved 3 examples per task from the rendered prompt and input
alone, and averaged 95%. The prompt gaps it found in this range were fixed.

These tasks were **not** rebuilt or reviewed task by task like 1-102. They have no extra wordings, and we never
trained on them. Treat them as a starting point.

**Known limits.**
- A generator can raise on a rare seed (task 25: 2 seeds in 10,000). The packing skips such seeds.
- Generation is fast: about 1 ms per example at the median, and ~40 ms for the slowest tasks.

## Install

Python 3.12. The generators need only numpy. The packing also needs `transformers` (for token lengths).

```bash
cd aux-tasks
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Run from this folder (the modules import each other by top-level name: `registry`, `prompts`, `generators`, `common`).

```python
import registry, prompts, prompt_variants

ex = registry.generate("22", seed=7)          # {"input", "output", "metadata", "task_id"}
print(prompts.get_prompt("22", ex["metadata"]))
print(ex["output"])

# an extra wording: always render it through prompt_variants.render (it expands compound tokens)
template = prompt_variants.variants_for("22")[0]["template"]
print(prompt_variants.render("22", template, ex["metadata"]))

# the same example under a symbol palette instead of digits
print(prompts.get_prompt("22", ex["metadata"], color_palette=list("ABCDEFGHIJ")))
```

A classic training row (instruction, input grid, target):

```python
import sft_wrapper
prompt, target = sft_wrapper.format_single("6", seed=0, fmt="compact")
items = [{"task_id": "6", "seed": 1}, {"task_id": "22", "seed": 2}, {"task_id": "77", "seed": 3}]
bundle_prompt, bundle_target = sft_wrapper.format_mixed_bundle(items, fmt="compact")
```

`python registry.py` prints three examples of every task. `python tools/make_task_table.py` regenerates `TASKS.md`.

## Our aux pool

The aux pool we trained on was built with `packing/` from exactly these files. It has 32 shards with seeds 101-132:
441,600 rows and 3.03B tokens. The settings were `--classic-share 0.3`, `--arc-questions all`, mixed encodings and
`--max-seq-length 32768`, with the Qwen3.5-4B tokenizer. These are the builder's defaults. The 4B solver's SFT used
seeds 101-121 (289,800 rows, 1.99B tokens).

The builder is deterministic, so the pool can be rebuilt instead of downloaded. The commands below rebuild seed 101.
Repeat them for each seed and merge with `python -m pv2.merge_caches`:

```bash
cd packing
python -m pv2.build_dataset --size 13800 --seed 101 --output out/shard_101.json --tokenizer Qwen/Qwen3.5-4B
```

We checked this copy against the shipped pool. The generator digest is the same (`bc8dc5e13031b305`, recorded in
each cache's metadata), and the first 40 rows of seed 101 are byte-identical to the pool's: messages, provenance and
lengths.

## Layout

```
registry.py            register(task_id), generate(task_id, seed), list_tasks(), RETIRED ids
prompts.py             base prompt per task; palette helpers (_ci, ...) so colours render through any palette
prompt_modules/        prompts of the families that keep them in their own module
prompt_variants.py     extra wordings: load, render, validate (CLI: list / add / validate)
prompt_variants.json   628 approved wordings for tasks 1-102
symbol_remap.py        symbol palettes (tokenizer_atoms/symbol_banks.json) and remapped rendering
sft_wrapper.py         the classic row format: single items, same-task and mixed-task bundles
generators/            one module per task family (core2_simple / core2_extra: tasks 103-281)
common/                shared grid, object, shape, topology and background libraries
packing/               the training-row builder (pv2) and its tests
tools/                 make_task_table.py
tests/                 generator tests (determinism, determinability oracles, coverage, prompts, pinned hashes)
```

## Adding or changing a task

- Register a generator with `@register("<id>")` in a module under `generators/`, and import that module in
  `registry.py`.
- The generator takes a `numpy.random.Generator` and returns `{"input", "output", "metadata"}`.
- Put every per-example fact the prompt uses into `metadata`, and render colours only through the palette helpers.
- Add a prompt to `PROMPT_TEMPLATES`, in `prompts.py` or in a `prompt_modules/` file.
- Add tests that cover:
  - determinism;
  - an independent solver that recovers the output from the prompt and the input alone;
  - coverage of sizes, colours and answers.
- Never alias a retired id silently: add it to `registry.RETIRED`.

Some tests pin hashes of fixed seeds (`tests/fixtures/`), so that a change to one family cannot silently alter
another. Refresh a pin only for the tasks you changed on purpose.

Code comments sometimes cite "the handbook" (section numbers) or "the user". The handbook is our internal review
manual and is not included; "the user" is the project lead, whose design decisions the comments record.

## Tests

```bash
python -m pytest tests -q                 # generators: 1,528 tests, ~15 min
cd packing && python -m pytest tests -q   # packing: 307 tests, ~15 min
```

Both suites are offline and pass (one skip each). Two generator tests time generation and can fail on a heavily
loaded machine; rerun them alone. A few packing tests build whole rows and need the tokenizer: set `PV2_TOKENIZER` to a local
Qwen3.5-4B folder, or they skip when the hub is unreachable. Some packing tests also cross-check rows against the
solver's prompt renderer and system prompts in `../training/arcsolver/prompt`; set `PV2_SOLVER_PROMPT_DIR` if you
move this folder.

## Licence

MIT (see [`LICENSE`](LICENSE)).
