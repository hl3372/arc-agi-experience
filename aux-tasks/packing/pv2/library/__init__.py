"""The grid-agnostic question library (DESIGN.md section 7) and the parts built only from it.

    pose(rng, slots, reads, qid=None, *, exclude=())  -> LibQuery | None
        one library question about ``slots[i] for i in reads`` (1 grid, 2 grids, or several),
        applicable to those grids and their already chosen encodings
    library_part(rng, *, n_questions, grid=None)      -> Part  (family "multi_question")
    comparison_part(rng)                              -> Part  (family "comparison")
    chain_part(rng)                                   -> Part  (family "chain")

``QUESTIONS`` maps every question id to its ``Spec`` (arity, answer kind, phrasings).
"""
from __future__ import annotations

from pv2.library import multi, single
from pv2.library.core import (AS_SHOWN, REGISTERS, ZERO_MARKERS, Ctx, LibQuery, Spec, arity_ok, build_query,
                              encoding_ok, weighted_order)

QUESTIONS: dict[str, Spec] = {}
for _spec in single.SPECS + multi.SPECS:
    if _spec.qid in QUESTIONS:
        raise RuntimeError(f"duplicate library question id {_spec.qid}")
    QUESTIONS[_spec.qid] = _spec


def _bare(qid: str) -> str:
    return qid[4:] if qid.startswith("lib:") else qid


def applicable_specs(slots, reads, *, exclude=()) -> list[Spec]:
    """Questions whose arity and encoding restrictions fit these slots (the grid-dependent guards
    are only known once a question is drawn)."""
    excluded = {_bare(q) for q in exclude}
    encodings = [slots[i].encoding for i in reads]
    return [s for s in QUESTIONS.values()
            if arity_ok(s, len(reads)) and s.qid not in excluded and encoding_ok(s, encodings)]


def pose(rng, slots, reads, qid=None, *, exclude=(), phrasing=None):
    """One library question about the grids ``slots[i]`` for ``i`` in ``reads``.

    ``qid`` ("lib:row" or "row") asks for that question; otherwise one is drawn (weighted) among
    those that apply.  ``exclude``: question ids not to use (e.g. those already in the part).
    ``phrasing`` forces a phrasing index (tests; None is returned when it does not fit the drawn
    case, e.g. a capped listing's wording for an uncapped draw).  Returns None when nothing
    applies.  A size
    question switches off the dims header of the slots it reads (``Slot.dims_header_ok``).
    """
    reads = tuple(int(i) for i in reads)
    if not reads or len(set(reads)) != len(reads) or any(not 0 <= i < len(slots) for i in reads):
        raise ValueError(f"bad reads {reads!r} for {len(slots)} slots")
    if qid is not None:
        spec = QUESTIONS.get(_bare(qid))
        if spec is None:
            raise KeyError(f"unknown library question {qid!r}")
        candidates = [s for s in applicable_specs(slots, reads, exclude=exclude) if s.qid == spec.qid]
    else:
        candidates = applicable_specs(slots, reads, exclude=exclude)
    if not candidates:
        return None
    ctx = Ctx([slots[i].grid for i in reads], [slots[i].encoding for i in reads], reads)
    if len(reads) == 1:
        candidates = single.size_weighted(candidates, ctx.g)     # long rows to write on big grids
    for spec in weighted_order(rng, candidates):
        draw = spec.sample(rng, ctx)
        if draw is not None:
            q = build_query(rng, spec, draw, reads, slots, phrasing)
            if q is not None or phrasing is not None:
                return q
    return None


from pv2.library.parts import chain_part, comparison_part, library_part  # noqa: E402

__all__ = ["AS_SHOWN", "QUESTIONS", "REGISTERS", "ZERO_MARKERS", "LibQuery", "Spec", "applicable_specs", "pose",
           "library_part", "comparison_part", "chain_part"]
