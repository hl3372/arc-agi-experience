"""Registry examples as v2 queries (DESIGN.md 3, 13.3, 13.6, 14).

An ITEM is one generated example of an in-scope task, shown as one ``Slot`` (its input grid) and
asked through one ``Query``: the task's own instruction in one of its wordings (canonical or an
approved stored paraphrase, uniformly among those the part allows), with the answer wrapper swapped
for the neutral notation clause (``pv2.wordings.neutral_text``).  The text is otherwise the
generator package's, unchanged, so the generator's guarantees and its answer stay valid; the
assembler prefixes the reference clause ("About the third grid: ...", ``lead_clause=True``) and
states the reply shape at part level.

The part tells the producer what it can host through ``Needs``; every wording whose flags clash is
skipped (per example: flags are measured on the text rendered for that example's metadata).

Encoding rules for items (``pv2.encodings`` flags):
* colour names never (item instructions say "color 5"; task 76 answers in names);
* background-revealing encodings never;
* coordinate tasks (``wordings.COORD_TASKS``): no coordinate-revealing encoding except ``stage2``
  (DESIGN 13.1: reading labelled rows is the transfer), unless ``Needs.allow_stage2`` is False;
* dimension tasks (task 1): nothing that reveals the size -- neither the sparse listing nor any
  row / column labels (the last label is the size minus one) -- and ``Slot.dims_header_ok=False``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from pv2 import encodings as enc
from pv2.pictures import try_generate
from pv2 import model
from pv2 import wordings as W
from pv2.paths import ensure_generator_package

ensure_generator_package()
from registry import generate  # noqa: E402

SEED_RANGE = 2 ** 31


@dataclass(frozen=True)
class Needs:
    """What the part hosting an item can take (every field narrows the eligible wordings).

    role_bearing          ARC part (grids are train/test inputs/outputs): no ``io_nouns`` wordings
    multi_answer          the part or row carries more than one answer: no ``nothing_else`` wordings
    question_after_grid   the question is not placed before its grid: no ``placement`` wordings
    multi_grid            more than one grid in the part: no ``placement`` wordings
    forbidden_label_words grid-label words the part uses ("image", "panel", "map", "figure", "#",
                          "grid_number" for numeric Grid N labels, a capital letter for Grid A..L
                          labels): no wording that contains them
    ordinal_refs          the part names grids by ordinal words: no ``inner_ordinals`` wordings
    forbidden_style_labels question / answer label words the part uses ("answer", "question", "task")
    block_capable         the answer style carries verbatim blocks; False: no ``structured`` answers
    short_only            restate-then-answer style: no ``long`` wordings
    forbidden_flags       any further flags to avoid (e.g. ``edit``, ``coords``)
    answer_kinds          allowed ``Answer.kind`` values (None = any)
    tasks                 restrict the task pool (None = every in-scope task)
    allow_stage2          coordinate tasks may use the stage-2 rendering
    encodings             restrict the slot's encoding to these names (None = any allowed)
    weights               encoding weights (None = ``encodings.DEFAULT_WEIGHTS``)
    max_cells             largest input grid (height x width) the part can hold
    """
    role_bearing: bool = False
    multi_answer: bool = False
    question_after_grid: bool = False
    multi_grid: bool = False
    forbidden_label_words: frozenset = frozenset()
    ordinal_refs: bool = False
    forbidden_style_labels: frozenset = frozenset()
    block_capable: bool = True
    short_only: bool = False
    forbidden_flags: frozenset = frozenset()
    answer_kinds: Optional[frozenset] = None
    tasks: Optional[tuple] = None
    allow_stage2: bool = True
    encodings: Optional[frozenset] = None
    weights: Optional[dict] = None
    max_cells: int = 900

    def forbidden(self) -> frozenset:
        """The flags a wording must not carry."""
        out = set(self.forbidden_flags)
        if self.role_bearing:
            out.add(model.FLAG_IO_NOUNS)
        if self.multi_answer:
            out.add(model.FLAG_NOTHING_ELSE)
        if self.question_after_grid or self.multi_grid:
            out.add(model.FLAG_PLACEMENT)
        if self.ordinal_refs:
            out.add(W.FLAG_INNER_ORDINALS)
        if not self.block_capable:
            out.add(model.FLAG_STRUCTURED)
        if self.short_only:
            out.add(model.FLAG_LONG)
        out.update(label_word_flag(w) for w in self.forbidden_label_words)
        out.update(W.FLAG_STYLE_LABEL_PREFIX + str(w).lower() for w in self.forbidden_style_labels)
        return frozenset(out)


def label_word_flag(word: str) -> str:
    """The flag for a grid-label word: a single letter keeps its capital ("Grid L" -> ``label_word:L``),
    words are lower case (``label_word:image``, ``label_word:grid_number`` for numeric "Grid 4")."""
    word = str(word)
    return model.LABEL_WORD_PREFIX + (word.upper() if len(word) == 1 and word.isalpha() else word.lower())


class ItemPick(tuple):
    """``(slot, query)`` -- unpacks like the plain pair -- with the provenance as ``.prov``
    (task id, seed, wording index / id, encoding): ``item_from_prov(pick.prov)`` rebuilds it."""

    def __new__(cls, slot: model.Slot, query: model.Query, prov: dict):
        obj = super().__new__(cls, (slot, query))
        obj.prov = prov
        return obj

    def __reduce__(self):                      # picklable across worker processes
        return (ItemPick, (self[0], self[1], self.prov))

    @property
    def slot(self) -> model.Slot:
        return self[0]

    @property
    def query(self) -> model.Query:
        return self[1]


# --- encodings --------------------------------------------------------------------------------------

# Coordinate tasks whose answer is a list of positions or runs: the stage-2 runs listing prints
# those almost verbatim ("3@c4-c7"), so stage2 would hand the answer over (items review, 2026-09-29).
STAGE2_GIVES_AWAY = frozenset({"2", "60", "62", "94"})


def encoding_weights(tid: str, grid, needs: Needs = Needs()) -> dict[str, float]:
    """The encodings (with weights) an item of this task may use for this grid."""
    weights = dict(enc.DEFAULT_WEIGHTS if needs.weights is None else needs.weights)
    coords, dims = tid in W.COORD_TASKS, tid in W.DIMS_TASKS
    out = {}
    for name, w in weights.items():
        if w <= 0 or (needs.encodings is not None and name not in needs.encodings):
            continue
        e = enc.ENCODINGS[name]
        if e.uses_names or e.reveals_background:
            continue
        if dims and (e.reveals_dims or e.reveals_coords):
            continue
        if coords and e.reveals_coords and not (name == "stage2" and needs.allow_stage2
                                                and tid not in STAGE2_GIVES_AWAY):
            continue
        if not enc.allowed(grid, name, names_ok=False, coords_ok=True, dims_ok=not dims, background_ok=False):
            continue
        out[name] = float(w)
    return out


# --- task pool ----------------------------------------------------------------------------------------

_PROBE_SEEDS = (0, 1, 2)


@lru_cache(maxsize=None)
def _always_flags(tid: str) -> tuple[frozenset, ...]:
    """Per wording: the flags it carried on every probe example (a cheap servability test)."""
    per = None
    for seed in _PROBE_SEEDS:
        rendered = W.render_all(tid, generate(tid, seed).get("metadata", {}))
        flags = [r.flags for r in rendered]
        per = flags if per is None else [a & b for a, b in zip(per, flags)]
    return tuple(per)


def can_serve(tid: str, needs: Needs = Needs()) -> bool:
    """Whether some wording of the task can meet ``needs``: task-level flags first, then the flags
    each wording carried on every probe example.  Only a pre-filter that saves wasted draws; the
    per-example check in ``item_query`` is what decides."""
    if tid not in W.TASKS:
        return False
    if needs.tasks is not None and tid not in needs.tasks:
        return False
    if needs.answer_kinds is not None and W.answer_kind(tid) not in needs.answer_kinds:
        return False
    forbidden = needs.forbidden()
    if W.task_flags(tid) & forbidden:
        return False
    return any(not (flags & forbidden) for flags in _always_flags(tid))


def task_pool(needs: Needs = Needs()) -> list[str]:
    return [t for t in W.TASKS if can_serve(t, needs)]


def sample_item_task(rng, needs: Optional[Needs] = None) -> str:
    """A task id drawn uniformly among the in-scope tasks that can serve ``needs``."""
    pool = task_pool(needs or Needs())
    if not pool:
        raise ValueError("no in-scope task can serve these needs")
    return pool[int(rng.integers(len(pool)))]


# --- items ----------------------------------------------------------------------------------------

def _build(tid: str, seed: int, example: dict, rendered: W.Rendered, encoding: str) -> ItemPick:
    grid = [list(map(int, row)) for row in example["input"]]
    slot = model.Slot(grid, encoding=encoding, role="grid", dims_header_ok=tid not in W.DIMS_TASKS)
    w = rendered.wording
    qid = f"item:{tid}:" + ("canon" if w.index == 0 else w.variant_id.rsplit("_", 1)[1])
    query = model.Query(source="item", qid=qid, reads=(0,), text=rendered.neutral,
                        answer=W.make_answer(tid, example["output"]), statement=None,
                        flags=rendered.flags, lead_clause=True)
    prov = {"source": "item", "task_id": tid, "seed": int(seed), "wording": w.wid,
            "wording_index": w.index, "encoding": encoding}
    return ItemPick(slot, query, prov)


def item_query(rng, task_id: Optional[str] = None, *, needs: Needs = Needs(),
               tries: int = 8) -> Optional[ItemPick]:
    """One registry example as ``(Slot, Query)`` meeting ``needs``, or None.

    Draws a task (uniformly among those that can serve, unless ``task_id`` is given), a seed, one
    wording uniformly among the example's eligible wordings, and an encoding (weighted, under the
    item rules).  Gives up after ``tries`` examples without an eligible wording / encoding."""
    if task_id is not None:
        task_id = str(task_id)
        if not can_serve(task_id, needs):
            return None
        pool = [task_id]
    else:
        pool = task_pool(needs)
        if not pool:
            return None
    forbidden = needs.forbidden()
    for _ in range(tries):
        tid = pool[int(rng.integers(len(pool)))]
        seed = int(rng.integers(0, SEED_RANGE))
        example = try_generate(tid, seed)
        if example is None:
            continue
        grid = example["input"]
        if len(grid) * len(grid[0]) > needs.max_cells:
            continue
        weights = encoding_weights(tid, grid, needs)
        if not weights:
            continue
        eligible = [r for r in W.render_all(tid, example.get("metadata", {})) if not (r.flags & forbidden)]
        if not eligible:
            continue
        rendered = eligible[int(rng.integers(len(eligible)))]
        encoding = enc.choose(rng, grid, names_ok=False, coords_ok=True, dims_ok=tid not in W.DIMS_TASKS,
                              background_ok=False, weights=weights)
        if encoding not in weights:          # choose() falls back to compact when nothing is allowed
            raise AssertionError(f"encoding {encoding!r} outside the allowed set {sorted(weights)}")
        return _build(tid, seed, example, rendered, encoding)
    return None


def item_from_prov(prov: dict) -> ItemPick:
    """Rebuild an item exactly from its provenance (C4: re-rendering is byte-identical)."""
    tid, seed = str(prov["task_id"]), int(prov["seed"])
    example = generate(tid, seed)
    rendered = W.render_all(tid, example.get("metadata", {}))[int(prov["wording_index"])]
    if rendered.wording.wid != prov["wording"]:
        raise ValueError(f"wording {prov['wording']!r} is no longer at index {prov['wording_index']}")
    return _build(tid, seed, example, rendered, str(prov["encoding"]))


def item_part(rng, needs: Needs = Needs(), task_id: Optional[str] = None) -> Optional[model.Part]:
    """A "reframed single": one grid, one item question (None when nothing can serve ``needs``)."""
    pick = item_query(rng, task_id, needs=needs)
    if pick is None:
        return None
    slot, query = pick
    part = model.Part(family="reframed", slots=[slot], queries=[query],
                      single_answer=model.FLAG_NOTHING_ELSE in query.flags, prov=dict(pick.prov))
    part.validate_shape()
    return part
