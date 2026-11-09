"""Part families built from the producers (DESIGN.md section 4).

    reframed_part(rng, *, multi_answer)        one registry item, one grid, one question
    gallery_part(rng, *, n_grids, n_asked)     several grids; items asked about some of them, library
                                               questions about others, some never asked about
    one_grid_part(rng)                         one grid, several questions: library only, or the item's
                                               own question plus pure reading questions
    comparison_part / chain_part               from pv2.library
    arc_part                                   from pv2.arc (when present)

Presentation (references, labels, placement, answer style) is the assembler's; families only set
the structural needs (several grids, several answers, ARC roles) and the assembler adapts to every
query's flags.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pv2 import encodings as enc
from pv2 import items as I
from pv2 import library as L
from pv2.library.core import AS_SHOWN
from pv2.library.parts import _conflicts
from pv2.model import FLAG_EDIT, Part, Slot
from pv2.pictures import sample_picture

# Registry tasks whose answer IS a literal reading of the grid (a size, a count of a colour, a row,
# a crop, a rotation, a histogram, the centre cell, the background ...): a library question on the same
# grid could hand their answer over, so they never share a grid with one.
LITERAL_ITEM_TASKS = frozenset({
    "1", "2", "3", "6", "10", "11", "25", "29", "30", "44", "45", "54", "55", "65", "66", "70", "75",
    "76", "77", "79", "80", "86",
})
# Library questions that only read cells, lines or runs: safe next to an item that is not literal.
READING_QIDS = frozenset({"cell", "first_cell", "last_cell", "corner", "row", "col", "runs_row",
                          "run_length", "run_end"})


def _lib_exclude_except(keep) -> set[str]:
    return {"lib:" + q.qid[4:] if q.qid.startswith("lib:") else "lib:" + q.qid
            for q in L.QUESTIONS.values() if (q.qid[4:] if q.qid.startswith("lib:") else q.qid) not in keep}


def reframed_part(rng, *, multi_answer: bool = False, max_cells: int = 900) -> Part:
    for _ in range(20):
        part = I.item_part(rng, I.Needs(multi_answer=multi_answer, max_cells=max_cells))
        if part is not None:
            return part
    raise RuntimeError("reframed_part: no eligible item")


def _picture_slot(rng, max_cells: int) -> tuple[Slot, dict]:
    grid, tid, seed = sample_picture(rng, max_cells=max_cells)
    encoding = enc.choose(rng, grid, names_ok=False, coords_ok=True, dims_ok=True, background_ok=False)
    return Slot(grid, encoding), {"picture": [tid, seed]}


GALLERY_CELL_BUDGET = 5400      # total cells of a gallery (six 30x30 grids); grids keep their natural size


def _grid_cap(budget_left: int, grids_left: int, max_cells: int) -> int:
    """The largest grid this slot may take so every later slot still has room for a small grid."""
    return int(max(25, min(max_cells, budget_left - 25 * (grids_left - 1))))


# Library-question share of a gallery's asked grids, and item share of one-grid parts (were 0.35 and 0.4):
# items write grids 62% of the time, library questions 21% (grid-writing rate, DESIGN 15); more items
# also means more item questions per task
GALLERY_LIB_SHARE = 0.15
ONE_GRID_ITEM_SHARE = 0.65


def gallery_part(rng, *, n_grids: int, n_asked: int, max_cells: int = 900, lib_share: float = GALLERY_LIB_SHARE,
                 cell_budget: int = GALLERY_CELL_BUDGET) -> Part:
    """``n_grids`` grids, ``n_asked`` of them asked about (by an item, or by a library question on a
    borrowed picture), the rest shown and never asked about.

    Grids follow the generators' natural sizes (28% have 400+ cells) under a TOTAL cell budget rather
    than a per-grid cap: the training side's audit (2026-09-30) found per-grid caps of 225-400 left
    2.4% of gallery grids at 400+ cells, far below the classic data (27%) and ARC-AGI-2 (43%)."""
    if not 2 <= n_grids <= 12 or not 1 <= n_asked <= n_grids:
        raise ValueError("gallery_part: 2 <= n_grids <= 12 and 1 <= n_asked <= n_grids")
    slots, queries, prov = [], [], {"slots": []}
    asked = set(int(i) for i in rng.choice(n_grids, size=n_asked, replace=False))
    used = 0
    for i in range(n_grids):
        cap = _grid_cap(cell_budget - used, n_grids - i, max_cells)
        needs = I.Needs(multi_grid=True, multi_answer=True, question_after_grid=True, max_cells=cap)
        if i in asked and rng.random() >= lib_share:
            pick = None
            for _ in range(20):
                pick = I.item_query(rng, needs=needs)
                if pick is not None:
                    break
            if pick is None:
                raise RuntimeError("gallery_part: no eligible item")
            slot, query = pick
            query = _reread(query, len(slots))
            slots.append(slot)
            used += len(slot.grid) * len(slot.grid[0])
            queries.append(query)
            prov["slots"].append({"item": pick.prov})
            continue
        slot, pic = _picture_slot(rng, cap)
        slots.append(slot)
        used += len(slot.grid) * len(slot.grid[0])
        prov["slots"].append(pic)
        if i in asked:
            q = L.pose(rng, slots, (len(slots) - 1,))
            if q is not None:
                queries.append(q)
    if not queries:
        return gallery_part(rng, n_grids=n_grids, n_asked=n_asked, max_cells=max_cells, lib_share=lib_share,
                            cell_budget=cell_budget)
    # questions in grid order is the common case; a shuffled order is part of the variety
    if rng.random() < 0.3:
        order = list(rng.permutation(len(queries)))
        queries = [queries[i] for i in order]
    for i, s in enumerate(slots):
        s.dims_header_ok = s.dims_header_ok and not any("dims" in q.flags and i in q.reads for q in queries)
    return Part("gallery", slots, queries, prov=prov)


def _reread(query, slot_index: int):
    """The same query, reading slot ``slot_index`` of the part instead of slot 0."""
    import copy
    q = copy.copy(query)
    object.__setattr__(q, "reads", (slot_index,))
    return q


def one_grid_part(rng, *, n_questions: int, item_share: float = ONE_GRID_ITEM_SHARE, max_cells: int = 900) -> Part:
    """One grid, several questions: library questions only, or the item's own question plus pure
    reading questions (never for literal item tasks, whose answer a reading question could give)."""
    if rng.random() >= item_share:
        return L.library_part(rng, n_questions=n_questions)
    needs = I.Needs(multi_answer=True, max_cells=max_cells,
                    tasks=tuple(t for t in I.task_pool() if t not in LITERAL_ITEM_TASKS))
    pick = None
    for _ in range(20):
        pick = I.item_query(rng, needs=needs)
        if pick is not None:
            break
    if pick is None:
        return L.library_part(rng, n_questions=n_questions)
    slot, query = pick
    slots, queries = [slot], [query]
    exclude = _lib_exclude_except(READING_QIDS)
    for _ in range(n_questions * 4):
        if len(queries) >= n_questions:
            break
        q = L.pose(rng, slots, (0,), exclude=exclude | {q.qid for q in queries})
        if q is None:
            break
        if not _conflicts(q, queries):
            queries.append(q)
    order = list(rng.permutation(len(queries)))
    queries = [queries[i] for i in order]
    context = [AS_SHOWN] if any(FLAG_EDIT in q.flags for q in queries) and len(queries) > 1 else []
    return Part("multi_question", slots, queries, context=context, prov={"item": pick.prov})


def comparison_part(rng) -> Part:
    return L.comparison_part(rng)


def chain_part(rng) -> Part:
    return L.chain_part(rng)


def arc_part(rng, *, modes=None, questions: str = "all", aux_tasks=None, families=None) -> Optional[Part]:
    """An ARC part.  With no options this is exactly ``pv2.arc.bundle.arc_part(rng)``.

    ``modes``: draw the mode among these only (weights renormalised); ``questions="reading"``: no
    question that uses the rule; ``aux_tasks`` / ``families``: restrict the rule sources."""
    try:
        from pv2.arc import bundle
    except ImportError:
        return None
    kw = {}
    if modes:
        unknown = sorted(set(modes) - set(bundle.MODES))
        if unknown:
            raise ValueError(f"unknown ARC modes {unknown}; choose from {bundle.MODES}")
        w = np.asarray([bundle.MODE_WEIGHTS[bundle.MODES.index(m)] for m in modes], dtype=float)
        kw["mode"] = modes[int(rng.choice(len(modes), p=w / w.sum()))]
    if questions != "all":
        kw["questions"] = questions
    if aux_tasks:
        kw["aux_tasks"] = list(aux_tasks)
    if families:
        kw["families"] = list(families)
    return bundle.arc_part(rng, **kw)
