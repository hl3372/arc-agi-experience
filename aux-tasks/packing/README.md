# packing (pv2)

The training-row builder for the auxiliary tasks. It turns the generators one folder up (`registry.generate`,
`prompts.get_prompt`, `prompt_variants.render`) into SFT rows. The generators are imported, never copied. The rows
come in varied layouts, grid encodings, answer styles and ARC-style bundles. This keeps a model from learning one
fixed scaffold instead of the skill. The design, with every decision and review outcome, is in `DESIGN.md`.

## What a row can look like

- **classic** (30% of rows by default): the classic formats, byte-identical to `../sft_wrapper.py`
  (single items, same-task bundles, mixed bundles with the long_tail_32k sizes).
- **v2 rows**: 1-12 *parts* whose families are drawn independently, so long rows mix families:
  - `reframed`: one registry item, one grid, the question before / after / around the grid;
  - `gallery`: 2-12 grids, labelled (Grid 1 / Grid A / Image / Panel / Map / Figure / #) or
    unlabelled and named by ordinal words ("the third grid", "the last grid"), items and library
    questions about some of them, the rest never asked about;
  - `multi_question`: one grid, 2-6 questions (library, or the item's own question plus reading
    questions);
  - `comparison`: cross-grid questions, verification of a proposed grid, multiple choice with
    misreading distractors;
  - `chain`: two-step dependent questions (the intermediate value is asked too);
  - `arc`: ARC-style bundles in the solver's own stage-2 format (or a plain encoding): the aux tasks
    themselves as rules (stated or hidden) and a transform library (71 families: 51 build bundles,
    20 only serve as rivals in the uniqueness check) plus scalar rules,
    with predict / identify-the-rule / verification / per-grid and per-group reading questions.
- **Answer styles**: bare, labelled (Answer 1: / Q1: / 1. / (a) ...), full sentence (one statement
  form per question phrasing), paragraph, JSON, key: value, chain line, bullets, restate-then-answer;
  always declared in the prompt, structured styles with their exact skeleton; half of the sentence /
  paragraph parts also state each question's exact, answer-neutral sentence.
- **Grid encodings**: compact, stage-2 (`r0 | ...` + runs), spaced, comma, JSON, bracketed, pipes,
  colour names, runs, sparse listing, row-labelled, indexed table -- with rules so an encoding never
  hands over an answer (sizes, coordinates, the background). Symbol palettes are not used in this version.

Every row is validated before it is written (`pv2/validate.py`, V1-V5): each grid parses back from
the prompt text, each reference resolves to exactly one grid (also across the whole row), each
answer equals the producer's gold value, the target parses back with its style's parser, and prompt
hygiene holds (no contradictory reply-shape phrases, no leftover placeholders, no background naming,
none of the perception probe's sentences).

## Build a cache

    cd packing
    python -m pv2.build_dataset --size 100000 --seed 1 --output out/aux_pv2.json
        [--max-seq-length 32768] [--classic-share 0.3] [--no-mix-encodings] [--tokenizer <path|hub id>]

Output: the stage-1 cache format (`{"metadata", "examples": [{"messages", "length", "prov"}]}`);
`length` is the real chat-templated token count;
`prov = "pv2|<version>|<kind>|<row seed>"` re-renders a row (`pv2.build_dataset.build_row`).
About 50 ms per row on one core (2000 rows: 106 s, 0 failures).

Large pools: build shards in parallel with DIFFERENT `--seed` values (row seeds are
`seed * 10_000_000 + i`, so shards never share a row), then merge (refuses overlapping shards or
mismatched settings):

    for s in 1 2 3 4 5 6 7 8; do python -m pv2.build_dataset --size 26000 --seed $s \
        --output out/shard_$s.json & done; wait
    python -m pv2.merge_caches --output out/aux_pv2.json out/shard_*.json

Options (defaults are today's mix): `--exclude-families arc,...`, `--arc-modes stated,...`,
`--arc-questions reading` (ARC-shaped bundles with no question that uses the rule), `--arc-tasks`,
`--arc-families`, `--grid-answer-encoding compact` (every grid ANSWER in one encoding, JSON kept out of
parts with a grid answer; answer styles and input encodings unaffected), `--no-mix-encodings` (every
v2 grid, input and answer, compact), `--classic-share`. The build stats report total and answer
(supervised) tokens per row kind.

The parity batches of 2026-09-30 (new: `--classic-share 0 --arc-questions reading
--grid-answer-encoding compact`; old: `--classic-share 1`) and the reviewers' reports are summarised in
DESIGN.md section 15.

The output is the cache format that the training module's builder reads as aux rows
(`../../training`, `arcsolver.sft.build_cache --aux-cache out/aux_pv2.json` with `--aux-frac` or `--aux-all`).

Knobs: `pv2.build_dataset.BuildConfig` (classic share, family weights, parts per row,
`mix_encodings`, `companion`: a row whose questions are all yes/no gets one reading question),
`pv2.assemble.AssembleConfig` (reference schemes, layouts, delimiters, answer styles, grid-answer encodings,
`declare_statements_p`: the share of sentence / paragraph parts that write every exact sentence into
the prompt, default 0.5), `pv2.encodings.DEFAULT_WEIGHTS`.

Grid-writing knobs (DESIGN.md section 15, addendum: ~530 grid answers per 1M tokens in the
reading/compact configuration, ~380 in the default one):
- `pv2.arc.bundle`:
  - `GRID_READ_P`: the share of ARC reading draws that write a window of a shown grid (a row, rows
    a..b, a column or a crop); 0.75 in reading-only parts, 0.5 in default parts.
  - `GRID_READ_KINDS`, `GRID_READ_MAX_CELLS` (600), `GRID_READ_SIZE_POWER`, `GRID_READ_BIG_CROP_P`,
    `GRID_READ_MONO_P`.
  - `READING_N`: reading questions per reading-only part (2-5).
- `pv2.library.single`:
  - `GRID_READ_WEIGHTS`: the weights of the row, column, row-range, column-range and crop questions.
  - `GRID_READ_SIZE_REF` / `GRID_READ_SIZE_CLIP`: how their weights grow with the grid's size.
- `pv2.families`: `GALLERY_LIB_SHARE` (0.15) and `ONE_GRID_ITEM_SHARE` (0.65), the shares of library
  and item questions.

## Evaluate

    python -m pv2.eval_v2 make  --n 600 --seed 424242 --out out/eval_v2.jsonl
    python -m pv2.eval_v2 score --eval out/eval_v2.jsonl --pred preds.jsonl

`preds.jsonl` holds `{"id", "text"}` (raw generations; the answer is taken after `</think>`).
Scores: strict (exact target), lenient per question (the row's own style parser), found (value
present).

## Tests

    python -m pytest tests -q -p no:cacheprovider      (307 tests, ~15 min)

The row-building tests need the tokenizer (`PV2_TOKENIZER`, a local Qwen3.5-4B folder, or the hub) and skip
without it. The checks against the solver's renderer and system prompts read `../../training/arcsolver/prompt`
(`PV2_SOLVER_PROMPT_DIR` overrides).

## Layout

    pv2/model.py         the producer / assembler contract (Slot, Query, Answer, Part, placeholders)
    pv2/encodings.py     grid encodings: canonical render + parse, reveal flags, weighted choice
    pv2/pictures.py      input grids borrowed from the reviewed generators
    pv2/wordings.py      every item wording, its flags, the answer-wrapper table
    pv2/items.py         registry items as queries
    pv2/classic.py       the old formats, byte-identical
    pv2/library/         68 grid-agnostic questions (245 phrasings) and the multi-question,
                         comparison and chain parts
    pv2/arc/             transforms, scalar rules, rival search, aux tasks as rules, bundles
    pv2/families.py      reframed / gallery / one-grid parts from the producers
    pv2/assemble/        references, layouts, answer styles, row assembly
    pv2/validate.py      V1-V5
    pv2/build_dataset.py the cache builder
    pv2/eval_v2.py       the eval split and scoring
