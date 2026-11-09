"""Parts built only from the question library.

* ``library_part``    one grid, several questions (family "multi_question")
* ``comparison_part`` several grids: the same question over each, cross-grid questions, VERIFICATION
                      of a proposed grid (slot role "proposed") and MULTIPLE CHOICE among candidate
                      grids (role "candidate") whose distractors are the known misreadings
* ``chain_part``      two-hop chains whose intermediate value is also asked on its own first

Guards shared by all parts: at most one size question per grid, no question whose text takes a
colour to be present next to a question asking whether it is, a bounded number of grid answers,
and the "as shown" sentence (``AS_SHOWN``) whenever an edit shares a grid with another question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pv2 import encodings as enc
from pv2.library import QUESTIONS, multi, pose, single
from pv2.library.core import (AS_SHOWN, Draw, Phrasing as P, Spec, choose_encoding, colour_counts, grid_list,
                              inflect, pick, ref_list, build_query)
from pv2.model import FLAG_DIMS, FLAG_EDIT, Part, Slot
from pv2.pictures import sample_picture

NAMES_P = 0.06          # share of parts whose grids are written in colour names

# Size mix.  Borrowed pictures are ~16% under 100 cells, ~57% 100-399 and ~27% at 400+ (ARC-AGI-2
# evaluation grids: ~43% at 400+).  Library pictures are drawn to ~15% / ~45% / ~40% by thinning the
# smaller bands, never by capping the size; parts with many grids stay within the token budget by
# showing fewer grids or candidates, or the compact encoding, instead.
BAND_ACCEPT = {"small": 0.63, "medium": 0.53, "large": 1.0}
PART_TOKEN_BUDGET = 10000   # grids shown + grid answers, estimated from above (the part's text is extra)
TOKENS_PER_CELL = {          # measured upper ends with the training tokenizer (30x30-ish pictures)
    "compact": 1.1, "spaced": 2.1, "comma": 2.1, "json_flat": 2.1, "json_rows": 2.2, "brackets": 2.2,
    "pipes": 2.2, "names": 2.1, "sparse": 1.8, "row_labelled": 1.5, "runs": 3.8, "indexed_table": 3.7,
    "stage2": 5.0}
ANSWER_TOKENS_PER_CELL = 2.2

# Symmetric pictures (yes answers for the symmetry questions) and near misses (a symmetric picture
# with one cell or one short run changed: the "no" answers that need a careful check).
SYMMETRIC_P, NEAR_MISS_P = 0.25, 0.15

GRID_ANSWER_QIDS = frozenset(q for q, s in QUESTIONS.items() if s.grid_answer)
DIMS_QIDS = frozenset(q for q, s in QUESTIONS.items() if s.dims)
BOUNDS_QIDS = frozenset(q for q, s in QUESTIONS.items() if s.bounds_size)
EACH_QIDS = frozenset(s.qid for s in multi.EACH_SPECS)
WHICH_QIDS = frozenset(s.qid for s in multi.WHICH_SPECS)
PAIR_QIDS = frozenset(s.qid for s in multi.PAIR_SPECS)
ALL_QIDS = frozenset(QUESTIONS)


# --- helpers ---------------------------------------------------------------------------------------

def size_band(cells: int) -> str:
    return "small" if cells < 100 else "medium" if cells < 400 else "large"


def _picture(rng, max_cells=900):
    """A borrowed picture, the size bands reweighted by ``BAND_ACCEPT``."""
    for _ in range(200):
        grid, tid, seed = sample_picture(rng, max_cells=max_cells)
        if rng.random() < BAND_ACCEPT[size_band(len(grid) * len(grid[0]))]:
            break
    return grid, [tid, seed]


def _grid_tokens(grid, encoding) -> int:
    return int(len(grid) * len(grid[0]) * TOKENS_PER_CELL[encoding]) + len(grid) + 12


def part_tokens(slots, queries=()) -> int:
    """An upper estimate of the tokens a part's grids and grid answers take."""
    t = sum(_grid_tokens(s.grid, s.encoding) for s in slots)
    for q in queries:
        if q.answer.kind == "grid":
            t += int(len(q.answer.value) * len(q.answer.value[0]) * ANSWER_TOKENS_PER_CELL) + len(q.answer.value)
    return t


def _squeeze(slots, queries):
    """Over the budget, rewrite the costliest grids in the compact encoding (it hands over nothing any
    question depends on; colour names stay, the texts name colours in words)."""
    while part_tokens(slots, queries) > PART_TOKEN_BUDGET:
        costly = [i for i, s in enumerate(slots) if s.encoding not in ("compact", "names")]
        if not costly:
            return
        i = max(costly, key=lambda k: _grid_tokens(slots[k].grid, slots[k].encoding))
        slots[i].encoding = "compact"


def near_miss(rng, grid, axis):
    """A symmetric grid with one cell or one short run (2-4 cells) changed on one side of the axis:
    it differs from its mirror image in 2-8 cells.  None when the grid is too narrow."""
    g = np.array(grid)
    t = g if axis == "lr" else g.T                  # work on rows; the axis runs down the middle
    h, w = t.shape
    if w < 2:
        return None
    half = w // 2
    r = int(rng.integers(h))
    length = 1 if rng.random() < 0.6 else int(rng.integers(2, 5))
    length = min(length, half)
    c = int(rng.integers(0, half - length + 1))
    old = set(t[r, c:c + length].tolist())
    palette = [v for v in sorted(colour_counts(g)) if v not in old] or [v for v in range(10) if v not in old]
    t[r, c:c + length] = pick(rng, palette)
    return grid_list(t if axis == "lr" else t.T)


def symmetry_variant(rng, grid, p_sym=SYMMETRIC_P, p_near=NEAR_MISS_P, axis=None):
    """The picture as drawn, made mirror-symmetric (``p_sym``), or made a near miss (``p_near``)."""
    u = rng.random()
    if u >= p_sym + p_near:
        return grid, None
    sym, axis = symmetrise(rng, grid, axis)
    if axis is None:
        return grid, None
    if u < p_sym:
        return sym, "symmetric:" + axis
    return near_miss(rng, sym, axis), "near_miss:" + axis


def symmetrise(rng, grid, axis=None):
    """Mirror one half of the grid onto the other (left-right or top-bottom; ``axis`` or either)."""
    g = np.array(grid)
    h, w = g.shape
    axes = [a for a, n in (("lr", w), ("tb", h)) if n >= 2 and axis in (None, a)]
    if not axes:
        return grid_list(g), None
    axis = pick(rng, axes)
    if axis == "lr":
        g[:, w - w // 2:] = g[:, :w // 2][:, ::-1]
    else:
        g[h - h // 2:] = g[:h // 2][::-1]
    return grid_list(g), axis


# Questions asked only when their grids have the same size (the first of each pair would always be
# "yes" next to them) or only when the grids differ (the first would always be "no").
IMPLIED_PAIRS = frozenset(
    [frozenset(("lib:same_size", q)) for q in ("lib:n_diff", "lib:diff_cells", "lib:diff_rows", "lib:first_diff")]
    + [frozenset(("lib:identical", q)) for q in ("lib:diff_cells", "lib:more_colour", "lib:larger",
                                                 # asked only of grids that differ, so their presence
                                                 # answered "identical?" (no in 12 of 12, recheck 2026-09-30)
                                                 "lib:n_diff", "lib:diff_rows", "lib:first_diff")])


def _reflect_axis(q):
    """The axis of a reflection the question names (a reflect question, or a verification / choice
    about a reflection); an edit is never the identity, so it implies "not symmetric" on that axis."""
    p = getattr(q, "params", {})
    if q.qid == "lib:reflect" or (q.qid in ("lib:verify", "lib:verify_diff", "lib:verify_redo", "lib:choice")
                                  and p.get("op") == "reflect"):
        return p.get("axis")
    return None


def _conflicts(q, others) -> bool:
    """Questions on a shared grid that would hand each other an answer: a text that takes colour X
    to be present next to one asking whether X is present (or how often); a text naming a row or
    column index, a half or a scale factor (it bounds the size) next to a size question; a question
    asked only of same-size (or of differing) grids next to "same size?" (or "identical?"); a
    reflection next to "symmetric?" on the same axis; two questions reading the same window (cell,
    line, range, crop, half) of the same grid, e.g. ``last_cell`` and the bottom-right ``corner``."""
    for o in others:
        if not set(q.reads) & set(o.reads):
            continue
        if frozenset((q.qid, o.qid)) in IMPLIED_PAIRS and set(q.reads) == set(o.reads):
            return True
        for a, b in ((q, o), (o, q)):
            axis = _reflect_axis(a)
            if axis and b.qid == "lib:symmetric_" + axis and b.reads[0] == a.reads[0]:
                return True
        qp, op_ = getattr(q, "params", {}), getattr(o, "params", {})
        if qp.get("window") is not None and qp.get("window") == op_.get("window") and q.reads[0] == o.reads[0]:
            return True             # the same cell / line / range / crop of the same grid, asked twice
        if (FLAG_DIMS in q.flags and op_.get("bounds_size")) or (FLAG_DIMS in o.flags and qp.get("bounds_size")):
            return True
        a = set(getattr(q, "params", {}).get("presupposes", ()))
        b = set(getattr(o, "params", {}).get("asks_presence", ()))
        c = set(getattr(q, "params", {}).get("asks_presence", ()))
        d = set(getattr(o, "params", {}).get("presupposes", ()))
        if a & b or c & d:
            return True
    return False


def _context(queries) -> list[str]:
    for q in queries:
        if FLAG_EDIT in q.flags and any(o is not q and set(o.reads) & set(q.reads) for o in queries):
            return [AS_SHOWN]
    return []


def _refresh_dims(slots, queries):
    for i, s in enumerate(slots):
        s.dims_header_ok = not any(FLAG_DIMS in q.flags and i in q.reads for q in queries)


def _prov(queries, **extra):
    out = dict(extra)
    out["queries"] = [{"qid": q.qid, "reads": list(q.reads), "params": getattr(q, "params", {})} for q in queries]
    return out


def _adhoc(rng, qid, kind, phrasings, draw, reads, slots, *, coords=False, dims=False, edit=False,
           bounds_size=False):
    spec = Spec(qid, len(reads), kind, None, tuple(phrasings), coords=coords, dims=dims, edit=edit,
                grid_answer=kind == "grid", bounds_size=bounds_size)
    return build_query(rng, spec, draw, reads, slots)


def _pose_into(rng, slots, reads, queries, *, exclude=(), tries=6, **kw):
    """Pose a question on ``reads`` that does not conflict with ``queries``; append it.  A question
    id that conflicts is not drawn again for this slot set (so a retry tries another question)."""
    exclude = set(exclude)
    for _ in range(tries):
        q = pose(rng, slots, reads, exclude=exclude, **kw)
        if q is None:
            return None
        if not _conflicts(q, queries):
            queries.append(q)
            return q
        exclude.add(q.qid)
    return None


# --- one grid, many questions ---------------------------------------------------------------------------

def library_part(rng, *, n_questions, grid=None) -> Part:
    """One grid (a borrowed picture unless ``grid`` is given) and ``n_questions`` (2-6 intended)
    distinct library questions about it."""
    if not 1 <= n_questions <= 8:
        raise ValueError("n_questions must be 1..8")
    prov = {}
    if grid is None:
        grid, prov["picture"] = _picture(rng)
        grid, prov["symmetry"] = symmetry_variant(rng, grid)
    grid = enc.check_grid([list(map(int, row)) for row in grid])
    encoding = enc.choose(rng, grid, names_ok=True, coords_ok=True, dims_ok=True, background_ok=True)
    slot = Slot(grid, encoding)
    cells = len(grid) * len(grid[0])
    max_grid_answers = 2 if cells <= 300 else 1
    queries = []
    for _ in range(n_questions * 6):
        if len(queries) >= n_questions:
            break
        exclude = {q.qid for q in queries}
        if sum(q.answer.kind == "grid" for q in queries) >= max_grid_answers:
            exclude |= {"lib:" + q for q in GRID_ANSWER_QIDS}
        if any(FLAG_DIMS in q.flags for q in queries):
            exclude |= {"lib:" + q for q in DIMS_QIDS | BOUNDS_QIDS}
        if any(q.params.get("bounds_size") for q in queries):
            exclude |= {"lib:" + q for q in DIMS_QIDS}
        q = pose(rng, [slot], (0,), exclude=exclude)
        if q is None:
            break
        if not _conflicts(q, queries):
            queries.append(q)
    if not queries:
        raise RuntimeError("library_part: no question applies")
    _refresh_dims([slot], queries)
    return Part("multi_question", [slot], queries, context=_context(queries),
                prov=_prov(queries, encoding=encoding, **prov))


# --- operations for verification and multiple choice ------------------------------------------------------

@dataclass
class Op:
    name: str
    params: dict
    noun: str                 # "⟪g0⟫ rotated 90 degrees clockwise"
    Noun: str                 # capitalised
    coords: bool
    edit: bool
    result: np.ndarray
    mistakes: list = field(default_factory=list)      # [(name, grid)]


def _cap(s: str) -> str:
    return "⟪G0⟫" + s[4:] if s.startswith("⟪g0⟫") else s[:1].upper() + s[1:]


def _one_cell_off(rng, res):
    r, c = int(rng.integers(res.shape[0])), int(rng.integers(res.shape[1]))
    palette = [v for v in sorted(colour_counts(res)) if v != res[r, c]] or [v for v in range(10) if v != res[r, c]]
    out = res.copy()
    out[r, c] = pick(rng, palette)
    return out


def _window(g, r, c, kh, kw):
    h, w = g.shape
    if r < 0 or c < 0 or r + kh > h or c + kw > w:
        return None
    return g[r:r + kh, c:c + kw]


def make_op(rng, g: np.ndarray, kind: str, cw) -> Op | None:
    """An operation on ``g`` with its correct result and the realistic mistakes made doing it."""
    h, w = g.shape
    if kind == "rotate":
        d = pick(rng, ("cw", "ccw", "180"))
        rot, turn, k = single.ROTATIONS[d]
        rot = pick(rng, (rot, turn))          # "90 degrees clockwise" / "a quarter turn clockwise"
        res = np.rot90(g, k)
        mistakes = {"cw": [("wrong_direction", np.rot90(g, 1)), ("transposed", g.T), ("half_turn", np.rot90(g, 2))],
                    "ccw": [("wrong_direction", np.rot90(g, -1)), ("transposed", g.T), ("half_turn", np.rot90(g, 2))],
                    "180": [("mirrored_left_right", g[:, ::-1]), ("flipped_upside_down", g[::-1]),
                            ("quarter_turn", np.rot90(g, -1))]}[d]
        op = Op("rotate", {"dir": d}, f"⟪g0⟫ rotated {rot}", f"⟪G0⟫ rotated {rot}", False, True, res, mistakes)
    elif kind == "reflect":
        a = pick(rng, ("lr", "tb"))
        how, _effect, detail = single.REFLECTIONS[a]
        res = g[:, ::-1] if a == "lr" else g[::-1]
        other = g[::-1] if a == "lr" else g[:, ::-1]
        op = Op("reflect", {"axis": a}, f"⟪g0⟫ mirrored {how} {detail}", f"⟪G0⟫ mirrored {how} {detail}", False, True,
                res, [("wrong_axis", other), ("half_turn", np.rot90(g, 2))])
    elif kind == "transpose":
        op = Op("transpose", {}, "⟪g0⟫ transposed (column i becomes row i)",
                "⟪G0⟫ transposed (column i becomes row i)", False, True, g.T,
                [("rotated_clockwise", np.rot90(g, -1)), ("rotated_counterclockwise", np.rot90(g, 1)),
                 ("anti_transposed", np.rot90(g, 2).T)])
    elif kind in ("row", "col"):
        lines = g if kind == "row" else g.T
        n = lines.shape[0]
        if n < 2:
            return None
        i = int(rng.integers(n))
        shape = (lambda v: v.reshape(1, -1)) if kind == "row" else (lambda v: v.reshape(-1, 1))
        rivals = [("one_indexed", i - 1), ("far_edge", n - 1 - i), ("far_edge_one_indexed", n - i),
                  ("next_line", i + 1)]
        mistakes = [(name, shape(lines[j])) for name, j in rivals if 0 <= j < n and j != i]
        other = g.T if kind == "row" else g
        if i < other.shape[0]:
            mistakes.append(("other_axis", shape(other[i])))
        if kind == "row":
            op = Op("row", {"r": i}, f"row {i} of ⟪g0⟫", f"Row {i} of ⟪g0⟫", True, False, shape(lines[i]), mistakes)
        else:
            op = Op("col", {"c": i}, f"column {i} of ⟪g0⟫ (as a one-column grid)",
                    f"Column {i} of ⟪g0⟫ (as a one-column grid)", True, False, shape(lines[i]), mistakes)
    elif kind == "crop":
        if h < 2 or w < 2:
            return None
        for _ in range(20):
            a, b = sorted(int(v) for v in rng.integers(0, h, size=2))
            c0, c1 = sorted(int(v) for v in rng.integers(0, w, size=2))
            if not ((a, b, c0, c1) == (0, h - 1, 0, w - 1) or (a == b and c0 == c1)):
                break
        else:
            return None
        kh, kw = b - a + 1, c1 - c0 + 1
        res = g[a:b + 1, c0:c1 + 1]
        cands = [("one_indexed", _window(g, a - 1, c0 - 1, kh, kw)), ("rows_one_indexed", _window(g, a - 1, c0, kh, kw)),
                 ("columns_one_indexed", _window(g, a, c0 - 1, kh, kw)), ("shifted_down", _window(g, a + 1, c0, kh, kw)),
                 ("transposed_ranges", _window(g, c0, a, kw, kh))]
        noun = f"the block of ⟪g0⟫ at rows {a}-{b} and columns {c0}-{c1} (ranges inclusive)"
        op = Op("crop", {"a": a, "b": b, "c0": c0, "c1": c1}, noun, _cap(noun), True, False, res,
                [(n_, m) for n_, m in cands if m is not None])
    elif kind == "recolour":
        counts = colour_counts(g)
        x = pick(rng, sorted(counts))
        y = pick(rng, [v for v in range(10) if v != x])
        res = np.where(g == x, y, g)
        mistakes = [("swapped", np.where(g == x, y, np.where(g == y, x, g)))]
        missed = res.copy()
        cells = np.argwhere(g == x)
        r, c = cells[int(rng.integers(len(cells)))]
        missed[r, c] = x
        mistakes.append(("missed_cell", missed))
        others = [z for z in counts if z not in (x, y)]
        if others:
            z = pick(rng, others)
            mistakes.append(("other_colour", np.where(g == z, y, g)))
        noun = f"⟪g0⟫ with every {cw(x)} cell changed to {cw(y)}"
        op = Op("recolour", {"X": x, "Y": y}, noun, _cap(noun), False, True, res, mistakes)
    elif kind == "half":
        halves = {}
        if h % 2 == 0:
            halves.update(top=g[:h // 2], bottom=g[h // 2:])
        if w % 2 == 0:
            halves.update(left=g[:, :w // 2], right=g[:, w // 2:])
        if not halves:
            return None
        name = pick(rng, sorted(halves))
        opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[name]
        mistakes = [("opposite_half", halves[opposite])]
        mistakes += [("other_axis_half", v) for k, v in halves.items() if k not in (name, opposite)]
        op = Op("half", {"half": name}, f"the {name} half of ⟪g0⟫", f"The {name} half of ⟪g0⟫", False, False,
                halves[name], mistakes)
    else:
        raise ValueError(kind)
    if np.array_equal(op.result, g) and op.edit:
        return None            # an identity edit tells nothing
    op.mistakes.append(("one_cell_off", _one_cell_off(rng, op.result)))
    return op


OP_KINDS = ("rotate", "reflect", "transpose", "row", "col", "crop", "recolour", "half")
OP_WEIGHTS = (1.2, 1.0, 0.7, 1.0, 0.8, 1.0, 1.0, 0.6)


def _distinct(grids):
    seen, out = set(), []
    for name, m in grids:
        key = (m.shape, m.tobytes())
        if key not in seen:
            seen.add(key)
            out.append((name, m))
    return out


def _zs(op):
    return " {ZS}" if op.coords else ""


VERIFY_IS = lambda op: (  # noqa: E731
    P("interrogative", "Is ⟪g1⟫ exactly " + op.noun + "?" + _zs(op),
      single.YN("⟪G1⟫ is exactly " + op.noun, "⟪G1⟫ is not exactly " + op.noun),
      declared="Asked whether ⟪g1⟫ is exactly " + op.noun + ", the answer is ⟪answer⟫."),
    P("interrogative", "⟪G1⟫ is offered as " + op.noun + ". Is it right?" + _zs(op),
      single.YN("⟪G1⟫ is right: it is exactly " + op.noun, "⟪G1⟫ is wrong: it is not exactly " + op.noun),
      declared="Asked whether ⟪g1⟫, offered as " + op.noun + ", is right, the answer is ⟪answer⟫."),
    P("imperative", "Check ⟪g1⟫ against " + op.noun + " and say whether the two match exactly." + _zs(op),
      single.YN("⟪G1⟫ matches " + op.noun + " exactly", "⟪G1⟫ does not match " + op.noun + " exactly"),
      declared="On whether ⟪g1⟫ matches " + op.noun + " exactly, the answer is ⟪answer⟫."),
    P("terse", "⟪G1⟫: is it " + op.noun + "?" + _zs(op),
      single.YN("⟪G1⟫ is " + op.noun, "⟪G1⟫ is not " + op.noun),
      declared="The answer to whether ⟪g1⟫ is " + op.noun + " is ⟪answer⟫."),
)
VERIFY_DIFF = lambda op: (  # noqa: E731
    P("interrogative", "In how many cells does ⟪g1⟫ differ from " + op.noun + "?" + _zs(op),
      "⟪G1⟫ differs from " + op.noun + " in ⟪answer⟫ {cells}.",
      declared="The number of cells in which ⟪g1⟫ differs from " + op.noun + " is ⟪answer⟫."),
    P("imperative", "Count the cells where ⟪g1⟫ disagrees with " + op.noun + "." + _zs(op),
      "⟪G1⟫ disagrees with " + op.noun + " in ⟪answer⟫ {cells}.",
      declared="The count of cells where ⟪g1⟫ disagrees with " + op.noun + " is ⟪answer⟫."),
    P("terse", "⟪G1⟫ against " + op.noun + ": number of mismatched cells?" + _zs(op),
      "Between ⟪g1⟫ and " + op.noun + ", ⟪answer⟫ {cells} {is_are} mismatched.",
      declared="Between ⟪g1⟫ and " + op.noun + ", the number of mismatched cells is ⟪answer⟫."),
)
VERIFY_REDO = lambda op: (  # noqa: E731
    P("imperative", "Write out " + op.noun + "." + _zs(op), op.Noun + " is:"),
    P("interrogative", "What is " + op.noun + "?" + _zs(op), op.Noun + " is:"),
)
CHOICE = lambda op, cands, cands_or: (  # noqa: E731
    P("interrogative", "Which of " + cands + " is " + op.noun + "?" + _zs(op),
      "Of " + cands + ", the one that is " + op.noun + " is ⟪answer⟫."),
    P("interrogative", "Exactly one of " + cands + " is " + op.noun + ". Which one is it?" + _zs(op),
      "The one of " + cands + " that is " + op.noun + " is ⟪answer⟫."),
    P("imperative", "Pick the one among " + cands + " that equals " + op.noun + "." + _zs(op),
      "The one among " + cands + " that equals " + op.noun + " is ⟪answer⟫."),
    P("terse", op.Noun + ": " + cands_or + "?" + _zs(op), "Of " + cands + ", " + op.noun + " is ⟪answer⟫."),
)


def _names_part(rng):
    return rng.random() < NAMES_P


def _cw(names):
    return lambda v: enc.colour_word(int(v), "names" if names else "compact")


def _draw_op(rng, g, cw, *, need, max_tries=30):
    """An op on g with at least ``need`` usable mistakes (``need`` gets the filtered list)."""
    for _ in range(max_tries):
        kind = pick(rng, OP_KINDS, OP_WEIGHTS)
        op = make_op(rng, g, kind, cw)
        if op is None:
            continue
        usable = need(op)
        if usable is not None:
            return op, usable
    return None, None


def _op_window(op, g) -> list | None:
    """The window of ``g`` an op reads (row / column / crop / half), as reading questions record it."""
    h, w = g.shape
    p = op.params
    if op.name == "row":
        return [p["r"], p["r"], 0, w - 1]
    if op.name == "col":
        return [0, h - 1, p["c"], p["c"]]
    if op.name == "crop":
        return [p["a"], p["b"], p["c0"], p["c1"]]
    if op.name == "half":
        return {"top": [0, h // 2 - 1, 0, w - 1], "bottom": [h // 2, h - 1, 0, w - 1],
                "left": [0, h - 1, 0, w // 2 - 1], "right": [0, h - 1, w // 2, w - 1]}[p["half"]]
    return None


def _verification(rng, names):
    g, pic = _picture(rng)
    g = np.asarray(g)
    cw = _cw(names)

    def need(op):
        wrong = [(n, m) for n, m in op.mistakes if m.shape == op.result.shape and not np.array_equal(m, op.result)]
        return _distinct(wrong)
    op, wrong = _draw_op(rng, g, cw, need=need)
    if op is None:
        return None
    correct = not wrong or rng.random() < 0.5
    if correct:
        proposal, mistake = op.result, None
    else:
        weights = [0.5 if n == "one_cell_off" else 1.0 for n, _ in wrong]
        mistake, proposal = pick(rng, wrong, weights)
    slots = [Slot(grid_list(g), choose_encoding(rng, g, coords=op.coords, names=names)),
             Slot(grid_list(proposal), choose_encoding(rng, proposal, coords=op.coords, names=names), role="proposed")]
    base = {"op": op.name, **op.params, "mistake": mistake, "window": _op_window(op, g)}
    yes = bool(np.array_equal(proposal, op.result))
    text, case = single._yn(yes)
    queries = [_adhoc(rng, "verify", "bool", VERIFY_IS(op), Draw(yes, text, case=case, params=dict(base)),
                      (0, 1), slots, coords=op.coords, bounds_size=op.coords or op.name == "half")]
    n = int((proposal != op.result).sum())
    queries.append(_adhoc(rng, "verify_diff", "number", VERIFY_DIFF(op),
                          Draw(n, str(n), fmt=inflect(n), params=dict(base)), (0, 1), slots, coords=op.coords,
                          bounds_size=op.coords or op.name == "half"))
    if rng.random() < 0.3:
        queries.append(_adhoc(rng, "verify_redo", "grid", VERIFY_REDO(op),
                              Draw(grid_list(op.result), None, params=dict(base)), (0,), slots,
                              coords=op.coords, edit=op.edit, bounds_size=op.coords or op.name == "half"))
    return slots, queries, {"mode": "verify", "pictures": [pic]}


def _choice(rng, names):
    g, pic = _picture(rng)
    g = np.asarray(g)
    cw = _cw(names)

    def need(op):
        pool = [(n, m) for n, m in _distinct([("correct", op.result)] + op.mistakes) if n != "correct"]
        main = [(n, m) for n, m in pool if n != "one_cell_off"]
        return pool if len(pool) >= 2 and main else None
    op, pool = _draw_op(rng, g, cw, need=need)
    if op is None:
        return None
    main = [(n, m) for n, m in pool if n != "one_cell_off"]
    k = min(len(pool), pick(rng, (2, 3)))
    chosen = list(main)
    rng.shuffle(chosen)
    chosen = chosen[:k]
    if len(chosen) < k:
        chosen += [(n, m) for n, m in pool if n == "one_cell_off"][:k - len(chosen)]
    source = Slot(grid_list(g), choose_encoding(rng, g, coords=op.coords, names=names))
    cands = [("correct", op.result, choose_encoding(rng, op.result, coords=op.coords, names=names))]
    cands += [(n, m, choose_encoding(rng, m, coords=op.coords, names=names)) for n, m in chosen]
    # big grids: fewer distractors (never fewer than one) to stay within the part's token budget
    dropped = 0
    while len(cands) > 2 and part_tokens([source] + [Slot(grid_list(m), e) for _, m, e in cands]) > PART_TOKEN_BUDGET:
        cands.pop()
        dropped += 1
    order = [int(i) for i in rng.permutation(len(cands))]
    cands = [cands[i] for i in order]
    slots = [source] + [Slot(grid_list(m), e, role="candidate") for _, m, e in cands]
    right = 1 + [n for n, _, _ in cands].index("correct")
    reads = tuple(range(len(slots)))
    cand_refs = ref_list(range(1, len(slots)))
    cand_or = ref_list(range(1, len(slots)), "or")
    draw = Draw([right], f"⟪g{right}⟫",
                params={"op": op.name, **op.params, "candidates": [n for n, _, _ in cands],
                        "window": _op_window(op, g), "dropped_for_budget": dropped})
    queries = [_adhoc(rng, "choice", "token", CHOICE(op, cand_refs, cand_or), draw, reads, slots, coords=op.coords,
                      bounds_size=op.coords or op.name == "half")]
    if rng.random() < 0.3:
        _pose_into(rng, slots, (0,), queries, exclude={"lib:" + q for q in GRID_ANSWER_QIDS})
    return slots, queries, {"mode": "choice", "pictures": [pic]}


# --- comparisons -------------------------------------------------------------------------------------------

def _slots_for(rng, grids, names, *, dims=False):
    return [Slot(grid_list(np.asarray(g)), choose_encoding(rng, np.asarray(g), dims=dims, names=names))
            for g in grids]


def _edited(rng, g, k):
    """g with k distinct cells recoloured (each to a colour it did not have, from the grid's palette)."""
    h, w = g.shape
    palette = sorted(colour_counts(g))
    out = g.copy()
    for idx in rng.choice(h * w, size=min(k, h * w), replace=False):
        r, c = divmod(int(idx), w)
        options = [v for v in palette if v != g[r, c]] or [v for v in range(10) if v != g[r, c]]
        out[r, c] = pick(rng, options)
    return out


def _pair_second(rng, g, kinds=("edited", "patch", "transformed", "other"), weights=(0.40, 0.17, 0.15, 0.28)):
    """A second grid to compare with g, never identical to it: an edited copy, a patched copy, a
    transformed copy or an unrelated picture."""
    g = np.asarray(g)
    h, w = g.shape
    for _ in range(20):
        kind = pick(rng, kinds, weights)
        pic = None
        if kind == "edited":
            out = _edited(rng, g, int(pick(rng, range(1, 13), [1 / i for i in range(1, 13)])))
        elif kind == "near_edit":                 # a near miss: one to three cells
            out = _edited(rng, g, int(pick(rng, (1, 2, 3), (0.6, 0.25, 0.15))))
        elif kind == "patch":
            a, b = sorted(int(v) for v in rng.integers(0, h, size=2))
            c0, c1 = sorted(int(v) for v in rng.integers(0, w, size=2))
            out = g.copy()
            out[a:b + 1, c0:c1 + 1] = pick(rng, range(10))
        elif kind == "transformed":
            options = [("reflect_lr", g[:, ::-1]), ("reflect_tb", g[::-1]), ("rotate_180", np.rot90(g, 2))]
            if h == w:
                options.append(("rotate_cw", np.rot90(g, -1)))
            name, out = pick(rng, options)
            kind = "transformed:" + name
        else:
            other, pic = _picture(rng)
            out = np.asarray(other)
        if not np.array_equal(out, g):
            return out, kind, pic
    return _edited(rng, g, 1), "edited", None


def _other_shape(rng, g, same):
    """For "same size?": a second grid of g's shape (an edited copy, or a window of another picture)
    or, for no, of another shape (a near miss: one row or column more or fewer; or another picture)."""
    g = np.asarray(g)
    h, w = g.shape
    if same:
        if rng.random() < 0.5:
            for _ in range(30):
                other, pic = _picture(rng)
                o = np.asarray(other)
                if o.shape[0] >= h and o.shape[1] >= w and not (o.shape == g.shape and np.array_equal(o, g)):
                    r = int(rng.integers(0, o.shape[0] - h + 1))
                    c = int(rng.integers(0, o.shape[1] - w + 1))
                    return o[r:r + h, c:c + w].copy(), "window_of_other", pic
        return _edited(rng, g, int(pick(rng, range(1, 13), [1 / i for i in range(1, 13)]))), "edited", None
    if rng.random() < 0.5:
        options = []
        if h >= 2:
            options += [("minus_row", g[:-1]), ("minus_first_row", g[1:])]
        if w >= 2:
            options += [("minus_col", g[:, :-1]), ("minus_first_col", g[:, 1:])]
        if h < 30 and (h + 1) * w <= 900:
            options += [("plus_row", np.vstack([g, g[-1:]]))]
        if w < 30 and h * (w + 1) <= 900:
            options += [("plus_col", np.hstack([g, g[:, -1:]]))]
        if options:
            name, out = pick(rng, options)
            return out.copy(), "near_shape:" + name, None
    for _ in range(30):
        other, pic = _picture(rng)
        if np.asarray(other).shape != g.shape:
            return np.asarray(other), "other_shape", pic
    return g[:-1].copy() if h >= 2 else g[:, :-1].copy(), "near_shape", None


# How a cross part is planned: "identical?" and "same size?" are decided answer-first (half yes,
# half a near miss), so their answers are balanced; every other cross question is drawn freely.
CROSS_PLANS = {"identical": 0.33, "same_size": 0.14, "free": 0.53}
PLANNED = ("identical", "same_size")


def _cross(rng, names):
    plan = pick(rng, list(CROSS_PLANS), list(CROSS_PLANS.values()))
    g0, pic = _picture(rng)
    g0 = np.asarray(g0)
    yes = rng.random() < 0.5
    if plan == "identical":
        if yes:
            g1, how, pic1 = g0.copy(), "same", None
        else:
            g1, how, pic1 = _pair_second(rng, g0, ("near_edit", "patch", "transformed", "other"),
                                         (0.45, 0.2, 0.15, 0.2))
    elif plan == "same_size":
        g1, how, pic1 = _other_shape(rng, g0, yes)
    else:
        g1, how, pic1 = _pair_second(rng, g0)
    if plan == "same_size" and not 1 <= g1.size <= 900:
        return None
    slots = _slots_for(rng, [g0, g1], names, dims=plan == "same_size")
    queries = []
    n_free = int(pick(rng, (1, 2, 3), (0.4, 0.4, 0.2)))
    if plan != "free":
        if _pose_into(rng, slots, (0, 1), queries, exclude={"lib:" + q for q in ALL_QIDS if q != plan}) is None:
            return None
        n_free -= 1
    for _ in range(n_free):
        exclude = ({"lib:" + q for q in ALL_QIDS - PAIR_QIDS} | {"lib:" + q for q in PLANNED}
                   | {q.qid for q in queries})
        if any(FLAG_DIMS in q.flags for q in queries):
            exclude |= {"lib:" + q for q in DIMS_QIDS}
        if np.array_equal(g0, g1):          # an exact copy: differences would all be 0 / none
            exclude |= {"lib:n_diff", "lib:diff_cells", "lib:diff_rows", "lib:first_diff"}
        if _pose_into(rng, slots, (0, 1), queries, exclude=exclude) is None:
            break
    if not queries:
        return None
    return slots, queries, {"mode": "cross", "plan": plan, "pair": how, "pictures": [pic, pic1]}


def _fit(slots, min_n):
    """Drop trailing grids while the part is over the token budget (never below ``min_n``)."""
    while len(slots) > min_n and part_tokens(slots) > PART_TOKEN_BUDGET:
        slots.pop()
    return slots


def _several(rng, n, names, p_sym, p_near):
    """n pictures (fewer when over the token budget); one symmetry axis per part, so questions about
    that axis over all the grids meet symmetric pictures and near misses often enough."""
    pics = [_picture(rng) for _ in range(n)]
    axis = pick(rng, ("lr", "tb"))
    grids = [symmetry_variant(rng, g, p_sym, p_near, axis)[0] for g, _ in pics]
    slots = _fit(_slots_for(rng, grids, names), 2)
    return slots, [p for _, p in pics[:len(slots)]]


def _each(rng, names):
    n = int(pick(rng, (2, 3, 4, 5), (0.3, 0.35, 0.2, 0.15)))
    slots, pics = _several(rng, n, names, 0.5, 0.15)
    n = len(slots)
    queries = []
    if rng.random() < 0.5:
        _pose_into(rng, slots, tuple(range(n)), queries, exclude={"lib:" + q for q in ALL_QIDS - EACH_QIDS})
    else:
        singles = [q for q, s in QUESTIONS.items() if s.arity == 1 and (n <= 3 or not s.grid_answer)]
        for _ in range(10):
            qid = pick(rng, singles, [QUESTIONS[q].weight for q in singles])
            posed = [pose(rng, slots, (i,), qid) for i in range(n)]
            if all(p is not None for p in posed):
                queries = posed
                break
    if not queries:
        return None
    return slots, queries, {"mode": "each", "pictures": pics}


def _which(rng, names):
    n = int(pick(rng, (3, 4, 5), (0.45, 0.35, 0.2)))
    slots, pics = _several(rng, n, names, 0.5, 0.15)
    n = len(slots)
    queries = []
    _pose_into(rng, slots, tuple(range(n)), queries, exclude={"lib:" + q for q in ALL_QIDS - WHICH_QIDS})
    if queries and rng.random() < 0.5:
        exclude = {"lib:" + q for q in ALL_QIDS - EACH_QIDS}
        if any(FLAG_DIMS in q.flags for q in queries):
            exclude |= {"lib:" + q for q in DIMS_QIDS}
        _pose_into(rng, slots, tuple(range(n)), queries, exclude=exclude)
    if not queries:
        return None
    return slots, queries, {"mode": "which", "pictures": pics}


COMPARISON_MODES = {"each": (_each, 0.20), "cross": (_cross, 0.25), "which": (_which, 0.10),
                    "verify": (_verification, 0.27), "choice": (_choice, 0.18)}


def comparison_part(rng, mode=None) -> Part:
    """Several grids: the same question over each, cross-grid questions, verification of a proposed
    grid, or multiple choice among candidates.  ``mode`` forces one of ``COMPARISON_MODES``."""
    names_list = list(COMPARISON_MODES)
    for _ in range(40):
        m = mode or pick(rng, names_list, [COMPARISON_MODES[k][1] for k in names_list])
        built = COMPARISON_MODES[m][0](rng, _names_part(rng))
        if built is None:
            continue
        slots, queries, prov = built
        _refresh_dims(slots, queries)
        _squeeze(slots, queries)
        return Part("comparison", slots, queries, context=_context(queries), prov=_prov(queries, **prov))
    raise RuntimeError("comparison_part: no comparison could be built")


# --- two-hop chains ---------------------------------------------------------------------------------------------

# hop 1: a library question whose answer becomes k (a number) or X (a colour); its definition
# phrase for the second question (question form, statement form) from the posed query's params
HOP1_NUMBER = {
    "rows": lambda p, cw: ("the number of rows of ⟪g0⟫",) * 2,
    "cols": lambda p, cw: ("the number of columns of ⟪g0⟫",) * 2,
    "n_colours": lambda p, cw: ("the number of different colors in ⟪g0⟫",) * 2,
    "count": lambda p, cw: (f"the number of {cw(p['X'])} cells in ⟪g0⟫",) * 2,
    "n_rows_containing": lambda p, cw: (f"the number of rows of ⟪g0⟫ that contain {cw(p['X'])}",) * 2,
    "n_distinct_rows": lambda p, cw: ("the number of distinct rows of ⟪g0⟫",) * 2,
    "first_row_containing": lambda p, cw: (f"the number of the first row of ⟪g0⟫ that contains {cw(p['X'])}",) * 2,
}
HOP1_COLOUR = {
    "first_cell": lambda p, cw: ("the color of the top-left cell of ⟪g0⟫",) * 2,
    "last_cell": lambda p, cw: ("the color of the bottom-right cell of ⟪g0⟫",) * 2,
    "most_frequent": lambda p, cw: ("the most frequent color of ⟪g0⟫ (exactly one color is most frequent)",
                                    "the most frequent color of ⟪g0⟫"),
    "cell": lambda p, cw: (f"the color of the cell at row {p['r']}, column {p['c']} of ⟪g0⟫",) * 2,
}
HOP1_COORDS = {"first_row_containing", "cell"}
HOP1_DIMS = {"rows", "cols"}


def _cap_def(s):
    return s[:1].upper() + s[1:]


CHAIN_LINE = {
    "row": (
        P("imperative", "Let k be {def}. Write out row {expr} of ⟪g1⟫. {ZS}", "{Def} is {k}, so row {idx} of ⟪g1⟫ is:",
          declared="With k as {def}, row {expr} of ⟪g1⟫ is:"),
        P("interrogative", "Taking k as {def}, what does row {expr} of ⟪g1⟫ look like? {ZS}",
          "Since {defs} is {k}, row {expr} of ⟪g1⟫ is row {idx}, which looks like this:",
          declared="Taking k as {def}, row {expr} of ⟪g1⟫ looks like this:"),
        P("terse", "With k = {def}: ⟪g1⟫, row {expr} {ZP}?", "{Def} is {k}, so ⟪g1s⟫ row {idx} is:",
          declared="With k = {def}, ⟪g1s⟫ row {expr} is:"),
    ),
    "col": (
        P("imperative", "Let k be {def}. Write out column {expr} of ⟪g1⟫ as a one-column grid. {ZS}",
          "{Def} is {k}, so column {idx} of ⟪g1⟫ is:", declared="With k as {def}, column {expr} of ⟪g1⟫ is:"),
        P("interrogative", "Taking k as {def}, what does column {expr} of ⟪g1⟫ look like, read from top to bottom? {ZS}",
          "Since {defs} is {k}, column {expr} of ⟪g1⟫ is column {idx}, which reads from top to bottom:",
          declared="Taking k as {def}, column {expr} of ⟪g1⟫, read from top to bottom, is:"),
    ),
    "top_rows": (
        P("imperative", "Let k be {def}. Give the first k rows of ⟪g1⟫, counting from the top.",
          "{Def} is {k}, so the first {k} rows of ⟪g1⟫ are:", declared="With k as {def}, the first k rows of ⟪g1⟫ are:"),
        P("interrogative", "If k is {def}, what are the top k rows of ⟪g1⟫?", "{Def} is {k}, so the top {k} rows of ⟪g1⟫ are:",
          declared="If k is {def}, the top k rows of ⟪g1⟫ are:"),
    ),
}
CHAIN_COLOUR = {
    "count": (
        P("interrogative", "Let X be {def}. How many X cells does ⟪g1⟫ contain?",
          "{Def} is {Xw}, and ⟪g1⟫ contains ⟪answer⟫ {Xw} {cells}.",
          declared="With X as {def}, the number of X cells in ⟪g1⟫ is ⟪answer⟫."),
        P("imperative", "Let X be {def}. Count the cells of ⟪g1⟫ that have color X.",
          "{Def} is {Xw}, and ⟪g1⟫ has ⟪answer⟫ {cells} of that color.",
          declared="With X as {def}, the count of cells of ⟪g1⟫ that have color X is ⟪answer⟫."),
        P("terse", "X = {def}; number of X cells in ⟪g1⟫?", "{Def} is {Xw}, so ⟪g1⟫ has ⟪answer⟫ {Xw} {cells}.",
          declared="With X = {def}, the number of X cells in ⟪g1⟫ is ⟪answer⟫."),
    ),
    "has": (
        P("interrogative", "Let X be {def}. Does X occur anywhere in ⟪g1⟫?",
          single.YN("{Def} is {Xw}, and it occurs in ⟪g1⟫", "{Def} is {Xw}, and it does not occur in ⟪g1⟫"),
          declared="With X as {def}, the answer to whether X occurs anywhere in ⟪g1⟫ is ⟪answer⟫."),
        P("terse", "X = {def}; any X cell in ⟪g1⟫?",
          single.YN("{Def} is {Xw}, and ⟪g1⟫ has {Xw} cells", "{Def} is {Xw}, and ⟪g1⟫ has no {Xw} cells"),
          declared="With X = {def}, the answer to whether ⟪g1⟫ has any X cell is ⟪answer⟫."),
    ),
    "recolour": (
        P("imperative", "Let X be {def}. Change every X cell of ⟪g1⟫ to {Y}, leaving every other cell unchanged.",
          "{Def} is {Xw}, so ⟪g1⟫ with every {Xw} cell changed to {Y} is:",
          declared="With X as {def}, ⟪g1⟫ with every X cell changed to {Y} is:"),
        P("interrogative", "Taking X as {def}, what does ⟪g1⟫ look like once every X cell is recolored {Y} and every "
                           "other cell stays unchanged?",
          "Since {defs} is {Xw}, ⟪g1⟫ with every {Xw} cell recolored {Y} looks like this:",
          declared="Taking X as {def}, ⟪g1⟫ with every X cell recolored {Y} looks like this:"),
    ),
    "rows_containing": (
        P("interrogative", "Let X be {def}. Which rows of ⟪g1⟫ contain at least one X cell? List the row numbers in "
                           "increasing order, separated by commas. {ZS}",
          {"many": "{Def} is {Xw}, and the rows of ⟪g1⟫ that contain it are rows ⟪answer⟫.",
           "one": "{Def} is {Xw}, and the only row of ⟪g1⟫ that contains it is row ⟪answer⟫."},
          declared="With X as {def}, the rows of ⟪g1⟫ that contain at least one X cell are: ⟪answer⟫."),
    ),
}


def _same_grid(phrasings):
    def sub(t):
        return t.replace("⟪g1⟫", "⟪g0⟫").replace("⟪G1⟫", "⟪G0⟫").replace("⟪g1s⟫", "⟪g0s⟫")
    out = []
    for p in phrasings:
        st = sub(p.statement) if isinstance(p.statement, str) else {k: sub(v) for k, v in p.statement.items()}
        out.append(P(p.register, sub(p.text), st, p.tag, p.only, sub(p.declared) if p.declared else None))
    return out


def _with_zero(phrasings):
    """Coordinate chains state the 0-indexed convention in every wording."""
    return [p if ("{ZS}" in p.text or "{ZP}" in p.text) else
            P(p.register, p.text + " {ZS}", p.statement, p.tag, p.only, p.declared) for p in phrasings]


def _chain_once(rng):
    names = _names_part(rng)
    cw = _cw(names)
    same = rng.random() < 0.3
    colour_hop = rng.random() < 0.45
    hop1 = pick(rng, sorted(HOP1_COLOUR if colour_hop else HOP1_NUMBER))
    hop2 = pick(rng, sorted(CHAIN_COLOUR if colour_hop else CHAIN_LINE))
    if same and hop2 == "has":
        return None                     # a colour of the grid trivially occurs in it
    g0, pic0 = _picture(rng)
    g1, pic1 = (g0, None) if same else _picture(rng)
    coords = hop1 in HOP1_COORDS or hop2 in ("row", "col", "rows_containing")
    dims = hop1 in HOP1_DIMS
    slots = [Slot(g0, choose_encoding(rng, np.asarray(g0), coords=coords, dims=dims, names=names))]
    if not same:
        slots.append(Slot(g1, choose_encoding(rng, np.asarray(g1), coords=coords, dims=dims, names=names)))
    q1 = pose(rng, slots, (0,), hop1)
    if q1 is None or q1.answer.value is None:
        return None
    d_q, d_s = (HOP1_COLOUR if colour_hop else HOP1_NUMBER)[hop1](q1.params, cw)
    b = np.asarray(g1)
    fmt = {"def": d_q, "Def": _cap_def(d_s), "defs": d_s}
    params = {"hop1": hop1, "hop2": hop2, "hop1_params": {k: v for k, v in q1.params.items() if k in ("X", "r", "c")}}
    if colour_hop:
        x = int(q1.answer.value)
        fmt.update(Xw=cw(x))
        params["X"] = x
        present = bool((b == x).any())
        if hop2 == "count":
            n = int((b == x).sum())
            fmt.update(inflect(n))
            draw, kind = Draw(n, str(n), fmt=fmt, params=params), "number"
        elif hop2 == "has":
            text, case = single._yn(present)
            draw, kind = Draw(present, text, case=case, fmt=fmt, params=params), "bool"
        elif hop2 == "recolour":
            if not present:
                return None
            y = pick(rng, [v for v in range(10) if v != x])
            fmt.update(Y=cw(y))
            params["Y"] = y
            draw, kind = Draw(grid_list(np.where(b == x, y, b)), None, fmt=fmt, params=params), "grid"
        else:
            if not present or (b == x).mean() > single.DOMINANT_SHARE:
                return None             # absent, or so dominant that every row contains it
            rows = [r for r in range(b.shape[0]) if (b[r] == x).any()]
            draw, kind = Draw(rows, ", ".join(map(str, rows)), case="one" if len(rows) == 1 else "many",
                              fmt=fmt, params=params), "list"
        phrasings = CHAIN_COLOUR[hop2]
    else:
        k = int(q1.answer.value)
        fmt["k"] = k
        params["k"] = k
        if hop2 == "top_rows":
            if not 2 <= k < b.shape[0]:
                return None
            draw, kind = Draw(grid_list(b[:k]), None, fmt=fmt, params=params), "grid"
        else:
            lines = b if hop2 == "row" else b.T
            n = lines.shape[0]
            exprs = [("k", k)] if hop1 == "first_row_containing" else [("k-1", k - 1), ("k", k)]
            exprs = [(e, i) for e, i in exprs if 0 <= i < n]
            if n < 2 or not exprs:
                return None
            expr, idx = pick(rng, exprs)
            # an off-by-one in either hop lands on a neighbouring line: it must look different
            if any(0 <= j < n and np.array_equal(lines[j], lines[idx]) for j in (idx - 1, idx + 1)):
                return None
            fmt.update(expr=expr, idx=idx)
            params.update(expr=expr, idx=idx)
            ans = b[idx:idx + 1] if hop2 == "row" else b[:, idx:idx + 1]
            draw, kind = Draw(grid_list(ans), None, fmt=fmt, params=params), "grid"
        phrasings = CHAIN_LINE[hop2]
    reads = (0,) if same else (0, 1)
    if coords:
        phrasings = _with_zero(phrasings)
    if same:
        phrasings = _same_grid(phrasings)
    q2 = _adhoc(rng, f"chain:{hop1}>{hop2}", kind, phrasings, draw, reads, slots, coords=coords, dims=dims,
                edit=hop2 == "recolour")
    queries = [q1, q2]
    _refresh_dims(slots, queries)
    _squeeze(slots, queries)
    return Part("chain", slots, queries, context=_context(queries),
                prov=_prov(queries, mode="chain", same_grid=same, pictures=[pic0, pic1]))


def chain_part(rng) -> Part:
    """Two questions: an intermediate value (a number k or a colour X) asked on its own, then a
    question that uses it ("Let k be the number of rows of ⟪g0⟫. Write out row k-1 of ⟪g1⟫.").
    Every index is range-guarded; never more than two hops."""
    for _ in range(60):
        part = _chain_once(rng)
        if part is not None:
            return part
    raise RuntimeError("chain_part: no chain could be built")
