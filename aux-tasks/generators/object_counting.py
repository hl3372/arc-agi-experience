"""Object counting generators: tasks 4, 5, 68a, 69a, 68b, 69b.

Tasks 4 and 5 (shape counting) use one object rule -- every 8-connected group
of same-coloured non-background cells is one object -- and a closed shape
taxonomy (``classify_cells``): solid rectangle (squares and 1-cell-wide bars
included), L, plus, X, hollow frame, explicit ``#``/``.`` pattern, or other.
Scenes are built answer-first: the target count is sampled, exactly that many
matches are placed, then near-miss distractors, then an optional background
texture from ``common.backgrounds`` (sparse isolated cells, which the prompt
does not name -- the object rule says isolated single cells are background and
the solver infers them -- or solid regions / structured fills, which the prompt
names).  Object sizes, the
object budget and the answer range all scale with the canvas area, and the
object palette reaches 9 colours (10 with the background).  Every example is
verified by an independent recount (``recount_objects``) on the final
displayed grid.

Tasks 68a/68b delegate to component_count_68 and draw connectivity per example.
The former 69a/69b registrations are retired; legacy helper bodies below are
retained without changes to the passed shape-counting generators.
"""
from __future__ import annotations

import numpy as np

from common import backgrounds as bglib
from common import shapes as shape_lib
from common.grid import (
    NUM_COLORS,
    generate_scene,
    grid_to_list,
    make_grid,
    pick_colors,
    random_background,
    random_grid_size,
)
from common.objects import (
    bounding_box,
    connected_components,
    random_shape,
    stamp_rectangle,
)
from registry import register

MAX_RETRIES = 20

# ---------------------------------------------------------------------------
# Tasks 4/5 -- shape taxonomy
# ---------------------------------------------------------------------------

QUERY_CATEGORIES = (
    "rectangle",
    "non_square_rectangle",
    "square",
    "line",
    "L",
    "plus",
    "X",
    "frame",
    "pattern",
)
_CATEGORY_WEIGHTS = {
    "rectangle": 0.16,
    "non_square_rectangle": 0.12,
    "square": 0.12,
    "line": 0.10,
    "L": 0.12,
    "plus": 0.08,
    "X": 0.08,
    "frame": 0.07,
    "pattern": 0.15,
}

# Explicit query patterns ("#" = cell, "." = empty, rows separated by "//").
# None of them is a rectangle, L, plus, X or frame (asserted at import time),
# and all are distinct up to rotation and reflection.
PATTERN_TEXTS = (
    "###//.#.",          # T tetromino
    ".##//##.",          # S tetromino
    "##//##//#.",        # P pentomino (2x2 block with a tail)
    ".##//##.//.#.",     # F pentomino
    ".#//.#//##//#.",    # N pentomino
    "###//.#.//.#.",     # T pentomino
    "#.#//###",          # U pentomino
    "#..//##.//.##",     # W pentomino
    ".#//##//.#//.#",    # Y pentomino
    "##.//.#.//.##",     # Z pentomino
    "#.#//###//#.#",     # H heptomino
    "#..#//####",        # wide U hexomino
    ".#.//###//#.#",     # arch hexomino
    "####//#...//####",  # C (9 cells)
    "#####//..#..//..#..",  # wide T (9 cells)
    "#...#//#...#//#####",  # wide U (9 cells)
    "##...//.##..//..##.//...##",  # domino staircase (8 cells)
    "#.#.#//#####",      # comb (8 cells)
    "###//###//#.#",     # block with two legs (8 cells)
)

_CATEGORY_PHRASES = {
    "rectangle": (
        "solid rectangles (a solid block of at least 2 cells that completely fills "
        "its bounding box; squares and 1-cell-wide bars count)"
    ),
    "non_square_rectangle": (
        "solid non-square rectangles (a solid block of at least 2 cells that "
        "completely fills its bounding box and whose height and width differ; "
        "1-cell-wide bars count, squares do not)"
    ),
    "square": (
        "solid squares (a solid block that completely fills its bounding box, "
        "with equal height and width of at least 2)"
    ),
    "line": (
        "lines (a straight horizontal or vertical bar exactly 1 cell wide and at "
        "least 2 cells long; a line is also a rectangle, but count only lines here)"
    ),
    "L": (
        "L-shapes (two straight arms, each exactly 1 cell wide, meeting at a right "
        "angle and sharing one corner cell; each arm, counting the shared corner "
        "cell, is at least 2 cells long; any orientation or mirror image counts)"
    ),
    "plus": (
        "plus signs (a horizontal bar and a vertical bar, each exactly 1 cell wide "
        "and of the same odd length of at least 3, crossing at their center cells)"
    ),
    "X": (
        "X-shapes (both diagonals of an odd-sided square of side at least 3, each "
        "1 cell wide, crossing at the center cell; the diagonal cells touch only at "
        "their corners)"
    ),
    "frame": (
        "hollow rectangles (a rectangular ring exactly 1 cell thick whose bounding "
        "box is at least 3 by 3, with no cell of the object inside the ring; hollow "
        "squares count; other objects may lie inside the ring and do not disqualify it)"
    ),
}


def _norm(cells) -> tuple[tuple[int, int], ...]:
    """Translate cells so the bounding box starts at (0, 0); sorted tuple."""
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


_TRANSFORMS = (
    lambda r, c: (r, c),
    lambda r, c: (c, -r),
    lambda r, c: (-r, -c),
    lambda r, c: (-c, r),
    lambda r, c: (r, -c),
    lambda r, c: (c, r),
    lambda r, c: (-r, c),
    lambda r, c: (-c, -r),
)


def _orientations(cells) -> list[tuple[tuple[int, int], ...]]:
    """All 8 rotations/reflections of a shape, each normalized."""
    return [_norm([t(r, c) for r, c in cells]) for t in _TRANSFORMS]


def _canonical(cells) -> tuple[tuple[int, int], ...]:
    """Canonical form up to rotation and reflection."""
    return min(_orientations(cells))


def pattern_cells(text: str) -> frozenset:
    """Parse a ``#``/``.`` pattern (rows separated by ``//``) into cells."""
    return frozenset(
        (r, c)
        for r, row in enumerate(text.split("//"))
        for c, ch in enumerate(row)
        if ch == "#"
    )


def pattern_text(cells) -> str:
    """Render cells as a ``#``/``.`` pattern with ``//`` row separators."""
    norm = _norm(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    cell_set = set(norm)
    return "//".join(
        "".join("#" if (r, c) in cell_set else "." for c in range(w)) for r in range(h)
    )


_PATTERN_CELLS = {text: pattern_cells(text) for text in PATTERN_TEXTS}
_PATTERN_BY_CANON = {_canonical(cells): text for text, cells in _PATTERN_CELLS.items()}


def classify_cells(cells) -> dict:
    """Classify one object under the task 4/5 taxonomy.

    Returns ``{"label", "n", "h", "w"}``; ``label`` is one of ``single``,
    ``rectangle`` (solid, includes squares and 1-wide bars), ``L``, ``plus``,
    ``X``, ``frame``, ``pattern:<text>`` or ``other``.
    """
    norm = _norm(cells)
    n = len(norm)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    info = {"n": n, "h": h, "w": w, "label": "other"}
    if n < 2:
        info["label"] = "single"
        return info
    if n == h * w:
        info["label"] = "rectangle"
        return info
    cell_set = set(norm)
    if h >= 2 and w >= 2 and n == h + w - 1:
        for rr in (0, h - 1):
            for cc in (0, w - 1):
                if cell_set == {(rr, c) for c in range(w)} | {(r, cc) for r in range(h)}:
                    info["label"] = "L"
                    return info
    if h == w and h % 2 == 1 and h >= 3 and n == 2 * h - 1:
        mid = h // 2
        if all(r == mid or c == mid for r, c in norm):
            info["label"] = "plus"
            return info
        if all(r == c or r + c == h - 1 for r, c in norm):
            info["label"] = "X"
            return info
    if (
        h >= 3
        and w >= 3
        and n == 2 * (h + w) - 4
        and all(r in (0, h - 1) or c in (0, w - 1) for r, c in norm)
    ):
        info["label"] = "frame"
        return info
    text = _PATTERN_BY_CANON.get(_canonical(norm))
    if text is not None:
        info["label"] = "pattern:" + text
    return info


def fine_label(info: dict) -> str:
    """Diagnostic sub-label: line / square / rectangle / L / ... / pattern / other."""
    label = info["label"]
    if label == "rectangle":
        if info["h"] == 1 or info["w"] == 1:
            return "line"
        return "square" if info["h"] == info["w"] else "rectangle"
    if label.startswith("pattern:"):
        return "pattern"
    return label


def query_matches(info: dict, query: str, size_filter=None) -> bool:
    """Does an object (``classify_cells`` output) count for a query?"""
    label = info["label"]
    h, w = info["h"], info["w"]
    if query == "rectangle":
        ok = label == "rectangle"
    elif query == "non_square_rectangle":
        ok = label == "rectangle" and h != w
    elif query == "square":
        ok = label == "rectangle" and h == w
    elif query == "line":
        ok = label == "rectangle" and (h == 1 or w == 1)
    else:
        ok = label == query
    if ok and size_filter is not None:
        a, b = int(size_filter[0]), int(size_filter[1])
        ok = (h, w) == (a, b) or (h, w) == (b, a)
    return ok


for _text, _cells in _PATTERN_CELLS.items():
    assert classify_cells(_cells)["label"] == "pattern:" + _text, _text
assert len(_PATTERN_BY_CANON) == len(PATTERN_TEXTS)
del _text, _cells


def recount_objects(grid: np.ndarray, background_colors) -> list[dict]:
    """Independent recount: 8-connected same-colour components of non-background colours.

    Returns ``[{"color", "cells", "info"}]`` sorted by (min row, min col).
    """
    rows = grid.tolist()
    H = len(rows)
    W = len(rows[0]) if H else 0
    bg = set(int(c) for c in background_colors)
    seen: set[tuple[int, int]] = set()
    objects = []
    for r in range(H):
        row = rows[r]
        for c in range(W):
            color = int(row[c])
            if color in bg or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr in (cr - 1, cr, cr + 1):
                    if nr < 0 or nr >= H:
                        continue
                    nrow = rows[nr]
                    for nc in (cc - 1, cc, cc + 1):
                        if nc < 0 or nc >= W or (nr, nc) in seen:
                            continue
                        if int(nrow[nc]) == color:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            fcells = frozenset(cells)
            objects.append({"color": color, "cells": fcells, "info": classify_cells(fcells)})
    objects.sort(key=lambda o: min(o["cells"]))
    return objects


def count_matches(objects, query, size_filter=None, color=None) -> int:
    return sum(
        1
        for o in objects
        if (color is None or o["color"] == color) and query_matches(o["info"], query, size_filter)
    )


# ---------------------------------------------------------------------------
# Tasks 4/5 -- shape builders
# ---------------------------------------------------------------------------


def _rect(h: int, w: int) -> frozenset:
    return frozenset((r, c) for r in range(h) for c in range(w))


def _frame(h: int, w: int, thickness: int = 1) -> frozenset:
    return shape_lib.frame(None, h, w, thickness)


def _plus(n: int) -> frozenset:
    return shape_lib.plus(None, (n - 1) // 2, 1)


def _x(n: int) -> frozenset:
    return frozenset({(i, i) for i in range(n)} | {(i, n - 1 - i) for i in range(n)})


def _orient(rng: np.random.Generator, cells) -> frozenset:
    return shape_lib.random_orientation(rng, frozenset(cells))


def _is_connected4(cells) -> bool:
    cells = set(cells)
    if not cells:
        return False
    start = min(cells)
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _perturb_pattern(rng: np.random.Generator, cells, mode: str):
    """Pattern with one cell moved / added / removed (still 4-connected, different shape)."""
    base = set(_norm(cells))
    base_canon = _canonical(base)
    for _ in range(30):
        cur = set(base)
        if mode in ("moved", "missing"):
            removable = [x for x in sorted(cur) if _is_connected4(cur - {x})]
            if not removable:
                return None
            cur.remove(removable[int(rng.integers(0, len(removable)))])
        if mode in ("moved", "extra"):
            cand = sorted(
                {(r + dr, c + dc) for r, c in cur for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
                - cur
            )
            if mode == "moved":
                cand = [x for x in cand if x not in base]
            if not cand:
                continue
            cur.add(cand[int(rng.integers(0, len(cand)))])
        if len(cur) >= 3 and _is_connected4(cur) and _canonical(cur) != base_canon:
            return frozenset(_norm(cur))
    return None


def _size_cap(H: int, W: int) -> int:
    """Largest bounding-box side for compact objects: half the short side, 3..14."""
    return max(3, min(14, min(H, W) // 2))


def _max_line_len(H: int, W: int) -> int:
    """Longest bar: two thirds of the long side, 3..20."""
    return max(3, min(20, (2 * max(H, W)) // 3))


def _rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Uniform integer in the inclusive range [lo, hi]."""
    if hi <= lo:
        return int(lo)
    return int(rng.integers(lo, hi + 1))


def _odd_side(rng: np.random.Generator, cap: int) -> int:
    sizes = [n for n in (3, 5, 7, 9, 11, 13) if n <= cap]
    weights = np.array([0.30, 0.28, 0.20, 0.13, 0.06, 0.03][: len(sizes)], dtype=float)
    return int(sizes[int(rng.choice(len(sizes), p=weights / weights.sum()))])


def _sample_category_shape(rng, query: str, cap: int, max_len: int, size_filter):
    """One object that matches ``query`` (and ``size_filter`` when given)."""
    if size_filter is not None:
        a, b = int(size_filter[0]), int(size_filter[1])
        h, w = (a, b) if rng.random() < 0.5 else (b, a)
    if query == "rectangle":
        if size_filter is None:
            u = rng.random()
            if u < 0.22:
                h = w = _rint(rng, 2, cap)
            elif u < 0.45:
                return _orient(rng, _rect(1, _rint(rng, 2, max_len)))
            else:
                h, w = _rint(rng, 1, cap), _rint(rng, 1, cap)
                if h == 1 and w == 1:
                    w = 2
        return _rect(h, w)
    if query == "non_square_rectangle":
        if size_filter is None:
            if rng.random() < 0.30:
                return _orient(rng, _rect(1, _rint(rng, 2, max_len)))
            h, w = _rint(rng, 1, cap), _rint(rng, 1, cap)
            if h == w:
                w = w + 1 if w < cap else w - 1
            if h == 1 and w == 1:
                w = 2
        return _rect(h, w)
    if query == "square":
        s = h if size_filter is not None else _rint(rng, 2, cap)
        return _rect(s, s)
    if query == "line":
        n = max(h, w) if size_filter is not None else _rint(rng, 2, max_len)
        return _orient(rng, _rect(1, n))
    if query == "L":
        if size_filter is None:
            h, w = _rint(rng, 2, cap), _rint(rng, 2, cap)
            if max(h, w) < 3 and rng.random() < 0.7:
                w = _rint(rng, 3, cap)
        return shape_lib.letter_L(rng, h, w, 1)
    if query == "plus":
        return _plus(h if size_filter is not None else _odd_side(rng, cap))
    if query == "X":
        return _x(h if size_filter is not None else _odd_side(rng, cap))
    if query == "frame":
        if size_filter is None:
            h, w = _rint(rng, 3, cap), _rint(rng, 3, cap)
        return _frame(h, w)
    if query.startswith("pattern:"):
        return _orient(rng, _PATTERN_CELLS[query.split(":", 1)[1]])
    raise ValueError(query)


# Near-miss distractor kinds per query category (first entries are the most
# confusable and are listed more than once to weight them).
_NEAR_MISS = {
    "rectangle": ["L", "frame", "notched_rect", "bumped_rect", "plus", "staircase",
                  "T", "U", "diag", "pattern_any", "polyomino"],
    "non_square_rectangle": ["square", "square", "square", "frame", "L", "notched_rect",
                             "bumped_rect", "pattern_any", "staircase"],
    "square": ["near_square_rect", "near_square_rect", "near_square_rect", "frame_square",
               "notched_rect", "bumped_rect", "L", "plus", "pattern_any"],
    "line": ["thick_bar", "thick_bar", "diag", "diag", "L", "T", "rect_nonsq", "pattern_any"],
    "L": ["frame", "T", "thick_L", "staircase", "U", "bar", "V", "L_extra", "pattern_any",
          "rect_nonsq"],
    "plus": ["X", "X", "T", "asym_plus", "thick_plus", "frame", "pattern_any", "polyomino"],
    "X": ["plus", "plus", "diag", "V", "asym_X", "thick_X", "pattern_any", "polyomino"],
    "frame": ["rect_solid", "rect_solid", "broken_frame", "broken_frame", "thick_frame", "thick_frame",
              "thick_frame", "L", "U", "frame_with_tail", "pattern_any"],
    "pattern": ["pattern_moved", "pattern_moved", "pattern_extra", "pattern_missing",
                "pattern_other", "polyomino", "L", "rect_nonsq"],
}
_FILLER_KINDS = ("polyomino", "blob", "diag", "staircase")


def _build_distractor(rng, kind: str, query: str, cap: int, max_len: int, size_filter, H: int, W: int):
    """Build one distractor shape of the given kind (may return None)."""
    sf = None if size_filter is None else (int(size_filter[0]), int(size_filter[1]))
    if kind == "same_cat_other_size":
        for _ in range(8):
            cells = _sample_category_shape(rng, query, cap, max_len, None)
            info = classify_cells(cells)
            if sf is not None and (info["h"], info["w"]) not in (sf, sf[::-1]):
                return cells
        return None
    if kind == "square":
        s = sf[0] if (sf is not None and sf[0] == sf[1] and rng.random() < 0.5) else _rint(rng, 2, cap)
        return _rect(s, s)
    if kind == "rect_nonsq":
        h, w = _rint(rng, 1, cap), _rint(rng, 1, cap)
        if h == w:
            w = w + 1 if w < cap else w - 1
        if h == 1 and w == 1:
            w = 2
        return _rect(h, w)
    if kind == "rect_solid":
        if sf is not None and rng.random() < 0.6:
            return _orient(rng, _rect(sf[0], sf[1]))
        return _rect(_rint(rng, 2, cap), _rint(rng, 2, cap))
    if kind == "near_square_rect":
        s = sf[0] if (sf is not None and rng.random() < 0.5) else _rint(rng, 2, cap)
        t = s + 1 if s < cap else s - 1
        if t < 1 or s == t:
            return None
        return _orient(rng, _rect(s, t))
    if kind == "bar":
        return _orient(rng, _rect(1, _rint(rng, 2, max_len)))
    if kind == "thick_bar":
        n = max(sf) if (sf is not None and rng.random() < 0.5) else _rint(rng, 3, max(3, max_len))
        return _orient(rng, _rect(2, max(2, n)))
    if kind == "L":
        if sf is not None and min(sf) >= 2 and rng.random() < 0.5:
            return shape_lib.letter_L(rng, sf[0], sf[1], 1)
        return shape_lib.letter_L(rng, _rint(rng, 2, cap), _rint(rng, 2, cap), 1)
    if kind == "thick_L":
        if cap < 4:
            return None
        return shape_lib.letter_L(rng, _rint(rng, 4, cap), _rint(rng, 4, cap), 2)
    if kind == "L_extra":
        a, b = _rint(rng, 3, cap), _rint(rng, 3, cap)
        cells = {(r, 0) for r in range(a)} | {(a - 1, c) for c in range(b)} | {(a - 2, 1)}
        return _orient(rng, cells)
    if kind == "frame":
        if sf is not None and min(sf) >= 3 and rng.random() < 0.5:
            return _orient(rng, _frame(sf[0], sf[1]))
        return _frame(_rint(rng, 3, cap), _rint(rng, 3, cap))
    if kind == "frame_square":
        s = sf[0] if (sf is not None and sf[0] >= 3 and rng.random() < 0.6) else _rint(rng, 3, cap)
        return _frame(s, s)
    if kind == "thick_frame":
        if cap < 5:
            return None
        return _frame(_rint(rng, 5, cap), _rint(rng, 5, cap), 2)
    if kind == "broken_frame":
        if sf is not None and min(sf) >= 3 and rng.random() < 0.5:
            h, w = sf
        else:
            h, w = _rint(rng, 3, cap), _rint(rng, 3, cap)
        ring = set(_frame(h, w))
        edge = sorted(x for x in ring if x not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)})
        ring.remove(edge[int(rng.integers(0, len(edge)))])
        return _orient(rng, ring)
    if kind == "frame_with_tail":
        h, w = _rint(rng, 3, cap), _rint(rng, 3, cap)
        cells = set(_frame(h, w)) | {(h, _rint(rng, 0, w - 1))}
        return _orient(rng, cells)
    if kind == "notched_rect":
        if sf is not None and min(sf) >= 3 and rng.random() < 0.5:
            h, w = sf
        else:
            h, w = _rint(rng, 3, cap), _rint(rng, 3, cap)
        return _orient(rng, set(_rect(h, w)) - {(0, 0)})
    if kind == "bumped_rect":
        h, w = _rint(rng, 2, cap), _rint(rng, 2, cap)
        if w >= 3:
            bump = (h, _rint(rng, 1, w - 2))
        elif h >= 3:
            bump = (_rint(rng, 1, h - 2), w)
        else:
            bump = (h, 0)
        return _orient(rng, set(_rect(h, w)) | {bump})
    if kind == "plus":
        return _plus(_odd_side(rng, cap))
    if kind == "X":
        return _x(_odd_side(rng, cap))
    if kind == "asym_plus":
        top = (cap - 1) // 2
        for _ in range(10):
            arms = [_rint(rng, 1, top) for _ in range(4)]
            if len(set(arms)) > 1:
                up, down, left, right = arms
                cells = {(r, left) for r in range(up + down + 1)} | {(up, c) for c in range(left + right + 1)}
                return frozenset(cells)
        return None
    if kind == "thick_plus":
        if cap < 4:
            return None
        return shape_lib.plus(None, _rint(rng, 1, (cap - 2) // 2), 2)
    if kind == "asym_X":
        n = _odd_side(rng, cap)
        return _orient(rng, set(_x(n)) - {(0, 0)})
    if kind == "thick_X":
        sizes = [n for n in (5, 7) if n <= cap]
        if not sizes:
            return None
        n = sizes[int(rng.integers(0, len(sizes)))]
        return frozenset(
            (r, c) for r in range(n) for c in range(n)
            if min(abs(r - c), abs(r + c - (n - 1))) <= 1
        )
    if kind == "V":
        n = _rint(rng, 3, cap)
        cells = {(i, i) for i in range(n)} | {(i, n - 1 - i) for i in range(n // 2 + 1)}
        return _orient(rng, cells)
    if kind == "diag":
        return shape_lib.diagonal(rng, _rint(rng, 2, min(cap, 8)))
    if kind == "T":
        return shape_lib.letter_T(rng, _rint(rng, 3, cap), _rint(rng, 1, max(1, cap - 1)), 1)
    if kind == "U":
        return shape_lib.letter_U(rng, _rint(rng, 2, cap), _rint(rng, 3, cap), 1)
    if kind == "staircase":
        return shape_lib.staircase(rng, _rint(rng, 3, min(cap, 5)), 1)
    if kind == "polyomino":
        return shape_lib.polyomino(rng, _rint(rng, 3, 5))
    if kind == "blob":
        return shape_lib.blob(rng, _rint(rng, 5, max(5, min(30, cap * cap // 2))))
    if kind in ("pattern_any", "pattern_other"):
        own = query.split(":", 1)[1] if query.startswith("pattern:") else None
        texts = [t for t in PATTERN_TEXTS if t != own]
        cells = _PATTERN_CELLS[texts[int(rng.integers(0, len(texts)))]]
        return _orient(rng, cells)
    if kind in ("pattern_moved", "pattern_extra", "pattern_missing"):
        if not query.startswith("pattern:"):
            return None
        base = _PATTERN_CELLS[query.split(":", 1)[1]]
        cells = _perturb_pattern(rng, base, kind.split("_", 1)[1])
        return None if cells is None else _orient(rng, cells)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Tasks 4/5 -- scene placement, background texture
# ---------------------------------------------------------------------------


class _Scene:
    """Grid plus occupancy masks for moat-respecting placement.

    ``free`` marks cells whose whole 8-neighbourhood is still background (any
    object cell may go there); ``near[color]`` marks cells within Chebyshev
    distance 1 of an object of that colour (used when different colours are
    allowed to touch).
    """

    def __init__(self, H: int, W: int, bg: int):
        self.H, self.W, self.bg = H, W, bg
        self.grid = np.full((H, W), bg, dtype=np.int64)
        self.occupied = np.zeros((H, W), dtype=bool)
        self.free = np.ones((H, W), dtype=bool)
        self.near: dict[int, np.ndarray] = {}

    def place(self, rng: np.random.Generator, cells, color: int, touching: bool) -> bool:
        offs = sorted(cells)
        hs = max(r for r, _ in offs) + 1
        ws = max(c for _, c in offs) + 1
        H, W = self.H, self.W
        if hs > H or ws > W:
            return False
        if touching:
            allowed = ~self.occupied
            near = self.near.get(color)
            if near is not None:
                allowed = allowed & ~near
        else:
            allowed = self.free
        nh, nw = H - hs + 1, W - ws + 1
        valid = np.ones((nh, nw), dtype=bool)
        for dr, dc in offs:
            valid &= allowed[dr:dr + nh, dc:dc + nw]
        anchors = np.argwhere(valid)
        if len(anchors) == 0:
            return False
        r0, c0 = (int(v) for v in anchors[int(rng.integers(0, len(anchors)))])
        rows = np.fromiter((r0 + dr for dr, _ in offs), dtype=int, count=len(offs))
        cols = np.fromiter((c0 + dc for _, dc in offs), dtype=int, count=len(offs))
        self.grid[rows, cols] = color
        self.occupied[rows, cols] = True
        # Dilate the shape by one cell (8-neighbourhood) inside a padded box.
        m = np.zeros((hs + 2, ws + 2), dtype=bool)
        m[rows - r0 + 1, cols - c0 + 1] = True
        dil = m.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                dil |= np.roll(np.roll(m, dr, axis=0), dc, axis=1)
        rl, rh = r0 - 1, r0 + hs + 1
        cl, ch = c0 - 1, c0 + ws + 1
        top = -rl if rl < 0 else 0
        left = -cl if cl < 0 else 0
        rl, cl = max(rl, 0), max(cl, 0)
        rh, ch = min(rh, H), min(ch, W)
        sub = dil[top:top + (rh - rl), left:left + (ch - cl)]
        self.free[rl:rh, cl:ch] &= ~sub
        near = self.near.get(color)
        if near is None:
            near = np.zeros((H, W), dtype=bool)
            self.near[color] = near
        near[rl:rh, cl:ch] |= sub
        return True


# Background textures come from the shared library (common/backgrounds.py):
# sparse isolated cells (inferred by the solver through the isolated-cell rule
# in the prompt), solid regions and structured fills (named in the prompt).
_TEXTURE_P = (0.20, 0.12, 0.03)   # sparse / solid / structured shares of all examples


def _object_view(display: np.ndarray, bg: int, bg_colors) -> np.ndarray:
    """The displayed grid reduced to what the prompt calls objects: every background /
    texture colour in ``bg_colors`` becomes ``bg``, then every isolated single cell (no
    8-neighbour of its own colour) becomes ``bg`` -- the isolated-cell rule."""
    view = np.where(np.isin(display, list(bg_colors)), bg, display).astype(display.dtype)
    iso = bglib.isolated_cell_mask(view) & (view != bg)
    return np.where(iso, bg, view).astype(display.dtype)


# ---------------------------------------------------------------------------
# Tasks 4/5 -- scene design
# ---------------------------------------------------------------------------


def _sample_hw(rng: np.random.Generator) -> tuple[int, int]:
    """H, W independent; bands [6,12]/[13,20]/[21,30] at 31/32/37% (≈ width-
    proportional with a slight small lean, so the per-value density is nearly
    flat across the whole range)."""

    def side() -> int:
        u = rng.random()
        if u < 0.31:
            return _rint(rng, 6, 12)
        if u < 0.63:
            return _rint(rng, 13, 20)
        return _rint(rng, 21, 30)

    return side(), side()


def _sample_query(rng: np.random.Generator, H: int, W: int) -> str:
    names = list(_CATEGORY_WEIGHTS)
    probs = np.array([_CATEGORY_WEIGHTS[n] for n in names], dtype=float)
    cat = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    if cat != "pattern":
        return cat
    fits = []
    for text in PATTERN_TEXTS:
        info = classify_cells(_PATTERN_CELLS[text])
        if max(info["h"], info["w"]) <= max(H, W) and min(info["h"], info["w"]) <= min(H, W):
            fits.append(text)
    return "pattern:" + fits[int(rng.integers(0, len(fits)))]


def _sample_size_filter(rng: np.random.Generator, category: str, cap: int, max_len: int):
    """Bounding-box filter (a, b) with a <= b; None when only one size is feasible
    (a size filter needs a same-category, other-size distractor to be meaningful)."""
    if category == "line":
        return (1, _rint(rng, 2, max_len))
    if category == "square":
        s = _rint(rng, 2, cap)
        return (s, s)
    if category in ("plus", "X"):
        if cap < 5:
            return None
        n = _odd_side(rng, cap)
        return (n, n)
    if category == "frame":
        if cap < 4:
            return None
        a, b = _rint(rng, 3, cap), _rint(rng, 3, cap)
        return (min(a, b), max(a, b))
    if category == "L":
        a, b = _rint(rng, 2, cap), _rint(rng, 2, cap)
        return (min(a, b), max(a, b))
    if category == "rectangle":
        a, b = _rint(rng, 1, cap), _rint(rng, 1, cap)
        if a == 1 and b == 1:
            b = 2
        return (min(a, b), max(a, b))
    if category == "non_square_rectangle":
        a, b = _rint(rng, 1, cap), _rint(rng, 1, cap)
        if a == b:
            b = b + 1 if b < cap else b - 1
        return (min(a, b), max(a, b))
    raise ValueError(category)


MAX_COUNT = 12                 # answer ceiling
COUNT_FILL = 0.55            # share of the canvas the matches may claim (150+ cells)
COUNT_FILL_SMALL = 0.55      # same, under 150 cells (the user does not want busy small grids)
DISTRACTOR_RESERVE = 12        # cells kept for the mandatory near miss (a small shape plus its moat)


def _min_match_bbox(query: str, size_filter) -> tuple[int, int]:
    """Bounding box of the smallest object the match sampler can produce for ``query``
    (the filtered box when a size filter is set)."""
    if size_filter is not None:
        return int(size_filter[0]), int(size_filter[1])
    if query in ("rectangle", "non_square_rectangle", "line"):
        return 1, 2
    if query in ("square", "L"):
        return 2, 2
    if query in ("plus", "X", "frame"):
        return 3, 3
    if query.startswith("pattern:"):
        info = classify_cells(_PATTERN_CELLS[query.split(":", 1)[1]])
        return int(info["h"]), int(info["w"])
    raise ValueError(query)


def _count_cap(area: int, query: str, size_filter) -> int:
    """Largest answer for one example: how many copies of the smallest matching object
    (bounding box plus its shared one-cell moat, (h + 1) x (w + 1) cells) fit in the
    share of the canvas reserved for matches once the mandatory near miss has its
    DISTRACTOR_RESERVE cells; clamped to 1..MAX_COUNT.  A 9x9 canvas allows 5 bars,
    3 squares or 2 plus signs; a 20x20 canvas allows 12 of anything up to 3x3."""
    h, w = _min_match_bbox(query, size_filter)
    footprint = (h + 1) * (w + 1)
    fill = COUNT_FILL_SMALL if area < 150 else COUNT_FILL
    return max(1, min(MAX_COUNT, int((fill * area - DISTRACTOR_RESERVE) / footprint)))


_COUNT_BASE = [0.12, 0.15, 0.17, 0.14, 0.11, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02]


def _sample_count(rng: np.random.Generator, K: int, size_specific: bool) -> int:
    """Answer 0..K with a decaying tail; the zero share is 15% with a size filter, 12% without."""
    zero = 0.15 if size_specific else 0.12
    rest = np.array(_COUNT_BASE[1:K + 1], dtype=float)
    probs = np.concatenate(([zero], rest * (1.0 - zero) / rest.sum()))
    return int(rng.choice(K + 1, p=probs))


def _sample_n_colors(rng: np.random.Generator, colored: bool) -> int:
    """Object palette size (the background is extra): small palettes are common, but the
    full 9 non-background colours are reachable so busy scenes show all 10 colours."""
    # Near-flat with a mild lean toward fewer colours (the user's rule: every count 2..10
    # reachable, none overwhelmingly dominant); the realised count still leans low because
    # small scenes cannot use a large palette.
    if colored:
        return int(rng.choice([2, 3, 4, 5, 6, 7, 8, 9], p=[0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09]))
    return int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.12, 0.13, 0.13, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09]))


def _query_desc(query: str) -> str:
    if query.startswith("pattern:"):
        text = query.split(":", 1)[1]
        return (
            f"shaped exactly like the pattern {text} (# = object cell, . = empty, rows "
            "separated by //, and rotated or mirrored copies count)"
        )
    return _CATEGORY_PHRASES[query]


_DEBUG_FAIL = None  # set to a collections.Counter to record why builds are rejected


def _fail(reason: str):
    if _DEBUG_FAIL is not None:
        _DEBUG_FAIL[reason] += 1
    return None


def _sample_spec(rng: np.random.Generator, H: int, W: int, colored: bool) -> dict:
    """The answer-defining part of an example: background, palette, query, size filter,
    the target count k (from the footprint cap) and the touching flag."""
    bg = int(rng.integers(0, NUM_COLORS))
    palette = [int(c) for c in pick_colors(rng, _sample_n_colors(rng, colored), exclude=[bg])]
    cap, max_len = _size_cap(H, W), _max_line_len(H, W)
    query = _sample_query(rng, H, W)
    category = "pattern" if query.startswith("pattern:") else query
    size_filter = None
    if category != "pattern" and rng.random() < 0.30:
        size_filter = _sample_size_filter(rng, category, cap, max_len)
    k_max = _count_cap(H * W, query, size_filter)
    k = _sample_count(rng, k_max, size_filter is not None)
    touching = rng.random() < (0.25 if colored else 0.15)
    return {"bg": bg, "palette": palette, "query": query, "category": category,
            "size_filter": size_filter, "k": k, "k_max": k_max, "touching": touching}


def _build_example(rng: np.random.Generator, H: int, W: int, colored: bool, spec: dict):
    """One attempt at building the scene for ``spec`` on an H x W grid; None when it fails."""
    area = H * W
    bg = spec["bg"]
    palette = spec["palette"]
    query_color = palette[0] if colored else None
    other_colors = palette[1:] if colored else palette
    cap, max_len = _size_cap(H, W), _max_line_len(H, W)
    query, category, size_filter = spec["query"], spec["category"], spec["size_filter"]
    k, touching = spec["k"], spec["touching"]

    queues: dict[int, list[int]] = {}

    def rand_color(pool):
        """Cycle through a shuffled copy of the pool so every palette colour is used
        once there are enough objects (uniform draws left most of a big palette unused)."""
        q = queues.get(id(pool))
        if not q:
            q = [pool[i] for i in rng.permutation(len(pool)).tolist()]
            queues[id(pool)] = q
        return q.pop()

    # Many matches on one canvas: shrink the match shapes so they still fit.
    match_cap = cap if k <= 6 else max(3, int(round(cap * (6.0 / k) ** 0.8)))
    match_len = max_len if k <= 6 else max(3, int(round(max_len * (6.0 / k) ** 0.8)))

    def match_shape():
        return _sample_category_shape(rng, query, match_cap, match_len, size_filter)

    kinds = list(_NEAR_MISS[category])
    if size_filter is not None:
        kinds = ["same_cat_other_size"] * 3 + kinds

    def distractor(kind=None):
        for _ in range(12):
            kd = kind
            if kd is None:
                kd = kinds[int(rng.integers(0, len(kinds)))]
            elif kd == "filler":
                kd = _FILLER_KINDS[int(rng.integers(0, len(_FILLER_KINDS)))]
            cells = _build_distractor(rng, kd, query, cap, max_len, size_filter, H, W)
            if cells is None:
                continue
            info = classify_cells(cells)
            if info["n"] < 2 or info["h"] > H or info["w"] > W:
                continue
            if query_matches(info, query, size_filter):
                continue
            return cells, kd
        return None, None

    plan: list[dict] = []

    def add(cells, color, role, kind):
        plan.append({"cells": cells, "color": int(color), "role": role, "kind": kind})

    for _ in range(k):
        add(match_shape(), query_color if colored else rand_color(palette), "match", "match")

    if colored:
        extra = 0 if area < 100 else 1 if area < 400 else 2
        n_color_d = 1 + _rint(rng, 0, extra)
        n_shape_d = 1 + _rint(rng, 0, extra)
        # Other-coloured objects are the only route to many colours on task 5 (every match
        # shares the query colour), so the budget grows with the canvas.
        n_other_d = _rint(rng, 0, 0 if area < 100 else 1 if area < 200 else 3 if area < 400 else 6 if area < 700 else 8)
        for i in range(n_color_d):
            add(match_shape(), rand_color(other_colors), "must" if i == 0 else "opt", "color_distractor")
        for i in range(n_shape_d):
            # With a size filter the first query-colour distractor is the same
            # category at another size (the hard case); otherwise a near miss.
            cells, kd = (None, None)
            if i == 0 and size_filter is not None:
                cells, kd = distractor("same_cat_other_size")
            if cells is None:
                cells, kd = distractor()
            if cells is None:
                if i == 0:
                    return _fail("no shape distractor (task 5)")
                continue
            add(cells, query_color, "must" if i == 0 else "opt", kd)
        for _ in range(n_other_d):
            cells, kd = distractor()
            if cells is not None:
                add(cells, rand_color(other_colors), "opt", kd)
    else:
        hi = 1 if area < 60 else 2 if area < 120 else 3 if area < 200 else 5 if area < 400 else 7 if area < 700 else 8
        for i in range(_rint(rng, 1, hi)):
            cells, kd = distractor()
            if cells is None:
                if i == 0:
                    return _fail("no distractor (task 4)")
                continue
            add(cells, rand_color(palette), "must" if i == 0 else "opt", kd)
        if size_filter is not None:
            cells, kd = distractor("same_cat_other_size")
            if cells is None:
                return _fail("no same-category other-size distractor")
            add(cells, rand_color(palette), "must", kd)

    for _ in range(_rint(rng, 0, 0 if area < 150 else 1 if area < 300 else 2 if area < 500 else 4 if area < 700 else 6)):
        cells, kd = distractor("filler")
        if cells is not None:
            add(cells, rand_color(palette), "opt", kd)

    order = rng.permutation(len(plan)).tolist()
    plan = [plan[i] for i in order]
    prio = {"match": 0, "must": 1, "opt": 2}
    plan.sort(key=lambda e: (prio[e["role"]], -len(e["cells"])))

    scene = _Scene(H, W, bg)
    for entry in plan:
        placed = scene.place(rng, entry["cells"], entry["color"], touching)
        tries = 0
        while not placed and tries < 3:
            tries += 1
            if entry["kind"] in ("match", "color_distractor"):
                cells = match_shape()
            else:
                cells, _kd = distractor(entry["kind"])
            if cells is None:
                break
            entry["cells"] = cells
            placed = scene.place(rng, cells, entry["color"], touching)
        entry["placed"] = placed
        if not placed and entry["role"] != "opt":
            return _fail(f"placement failed: {entry['role']}/{entry['kind']}")
    placed_entries = [e for e in plan if e["placed"]]

    # Optional background texture from the shared library in colours no object uses:
    # sparse isolated cells (20%, inferred through the isolated-cell rule), solid regions
    # (12%, named) or structured fills (3%, named, short side >= 12); never on a short
    # side under 8.
    display = scene.grid.copy()
    style = "uniform"
    texture_group = None
    texture_kind = None
    texture_colors: list[int] = []
    texture_named = False
    texture_coverage = 0.0
    u = rng.random()
    p_sparse, p_solid, p_struct = _TEXTURE_P
    if min(H, W) >= 8 and u < p_sparse + p_solid + p_struct:
        if u < p_sparse:
            group = "sparse"
        elif u < p_sparse + p_solid or min(H, W) < 12:
            group = "solid"
        else:
            group = "structured"
        # Structured fills are limited to the regular mosaic tiling: the mirrored random fill
        # is not inferable as background without naming it (nothing is named in the prompt).
        painted = bglib.apply_background(rng, scene.grid, bg, scene.occupied, group=group,
                                         kind=("mosaic" if group == "structured" else None), exclude=palette)
        if painted is not None:
            display, info = painted
            texture_group, texture_kind, style = info["group"], info["kind"], info["kind"]
            texture_colors = [int(c) for c in info["texture_colors"]]
            texture_named = bool(info["named"])
            texture_coverage = float(info["coverage"])

    # Independent verification on the final displayed grid, two ways that must agree:
    # (a) colour-based, every texture colour treated as background; (b) structural, as the
    # solver reads the prompt: only the named colours are background, then isolated single
    # cells of any colour are dropped.  Both views then drop isolated cells (the rule).
    named_colors = set(texture_colors) if texture_named else set()
    view_a = _object_view(display, bg, {bg} | set(texture_colors))
    view_b = _object_view(display, bg, {bg} | named_colors)
    if not np.array_equal(view_a, view_b):
        raise RuntimeError("tasks 4/5: colour-based and structural object views disagree")
    # (c) the prompt never names the background: a solver must be able to infer it as the
    # most frequent colour once the named texture colours and isolated cells are set aside,
    # by a clear margin and as the largest 4-connected region.  Otherwise the scene is
    # rejected (same spec, new content).
    objects = recount_objects(view_a, {bg})
    if len(objects) != len(placed_entries):
        return _fail("object count != placed shapes")
    largest_object = max(o["info"]["n"] for o in objects)
    inferred = bglib.infer_background(display, named_colors, beat_size=largest_object)
    if not inferred["identifiable"] or inferred["color"] != bg:
        return _fail("background not identifiable")
    view_c = _object_view(display, inferred["color"], {inferred["color"]} | named_colors)
    if not np.array_equal(view_a, view_c):
        raise RuntimeError("tasks 4/5: inferred-background object view disagrees")
    if count_matches(objects, query, size_filter, query_color) != k:
        return _fail("recount != target")
    if any(o["info"]["n"] < 2 for o in objects):
        return _fail("single-cell object")
    if set(texture_colors) & ({bg} | set(palette)):
        raise RuntimeError("tasks 4/5: texture colour collides with the background or an object colour")
    # Hardness: the mandatory near-miss objects really are on the grid.
    if not any(not query_matches(o["info"], query, size_filter) for o in objects):
        return _fail("no non-matching object")
    if size_filter is not None and not any(
        query_matches(o["info"], query, None) and not query_matches(o["info"], query, size_filter)
        for o in objects
    ):
        return _fail("no same-category other-size object")
    if colored:
        if not any(o["color"] == query_color and not query_matches(o["info"], query, size_filter) for o in objects):
            return _fail("no query-color shape distractor")
        if not any(o["color"] != query_color and query_matches(o["info"], query, size_filter) for o in objects):
            return _fail("no other-color match")

    class_counts: dict[str, int] = {}
    for o in objects:
        fl = fine_label(o["info"])
        class_counts[fl] = class_counts.get(fl, 0) + 1
    used_colors = sorted({o["color"] for o in objects})
    metadata = {
        "query_shape": query,
        "query_category": category,
        "query_shape_desc": _query_desc(query),
        "size_filter": None if size_filter is None else [int(size_filter[0]), int(size_filter[1])],
        "background": int(bg),
        "background_margin": round(inferred["count"] / max(1, inferred["runner_up_count"]), 3),
        "background_style": style,
        "texture_group": texture_group,
        "texture_kind": texture_kind,
        "texture_colors": texture_colors,
        "texture_named": texture_named,
        "texture_coverage": texture_coverage,
        "connectivity": 8,
        "grid_h": int(H),
        "grid_w": int(W),
        "n_objects": len(objects),
        "n_matches": int(k),
        "answer": int(k),
        "k_max": int(spec["k_max"]),
        "k_first_sampled": int(spec.get("k_first", k)),
        "zero_case": k == 0,
        "class_counts": dict(sorted(class_counts.items())),
        "object_colors": used_colors,
        "n_colors": len(used_colors),
        "touching_allowed": bool(touching),
        "distractor_kinds": sorted({e["kind"] for e in placed_entries if e["role"] != "match"}),
    }
    if colored:
        metadata["query_color"] = int(query_color)
        metadata["n_query_color_objects"] = sum(1 for o in objects if o["color"] == query_color)
    return {"input": grid_to_list(display), "output": str(k), "metadata": metadata}


def _generate_shape_count(rng: np.random.Generator, colored: bool) -> dict:
    """Sizes, then specs, then scenes: a scene that fails to place is retried for the SAME
    spec (same k) several times before a new spec is drawn, so placement failures on
    crowded counts do not silently thin the answer tail."""
    k_first = None
    for _size_attempt in range(40):
        H, W = _sample_hw(rng)
        for _spec_attempt in range(3):
            spec = _sample_spec(rng, H, W, colored)
            if k_first is None:
                k_first = spec["k"]
            spec["k_first"] = k_first
            for _scene_attempt in range(6):
                example = _build_example(rng, H, W, colored, spec)
                if example is not None:
                    return example
    raise RuntimeError("tasks 4/5: could not build a verified shape-counting scene")


# ---------------------------------------------------------------------------
# Task 4 -- Count shapes (grid -> text)
# ---------------------------------------------------------------------------


@register("4")
def generate_count_shapes(rng: np.random.Generator) -> dict:
    """Task 4: count the objects whose shape belongs to a named category."""
    return _generate_shape_count(rng, colored=False)


# ---------------------------------------------------------------------------
# Task 5 -- Count colored shapes (grid -> text)
# ---------------------------------------------------------------------------


@register("5")
def generate_count_colored_shapes(rng: np.random.Generator) -> dict:
    """Task 5: count the objects of a given colour whose shape belongs to a named category."""
    return _generate_shape_count(rng, colored=True)


# ---------------------------------------------------------------------------
# Shared helpers for tasks 68a/69a/68b/69b (unchanged)
# ---------------------------------------------------------------------------


def _count_4_components_in_cells(cells: set[tuple[int, int]]) -> int:
    """Count how many 4-connected components exist inside a cell set."""
    pending = set(cells)
    count = 0
    while pending:
        count += 1
        stack = [pending.pop()]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in pending:
                    pending.remove((nr, nc))
                    stack.append((nr, nc))
    return count


def _place_diagonal_pair(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    bg: int,
    row_lo: int = 0,
    row_hi: int | None = None,
    col_lo: int = 0,
    col_hi: int | None = None,
) -> bool:
    """Place an isolated same-color diagonal pair so 8-connectivity matters."""
    h, w = grid.shape
    row_hi = h if row_hi is None else row_hi
    col_hi = w if col_hi is None else col_hi
    if row_hi - row_lo < 2 or col_hi - col_lo < 2:
        return False

    for _ in range(50):
        cells = [(0, 0), (1, 1)] if rng.random() < 0.5 else [(0, 1), (1, 0)]
        r0 = int(rng.integers(row_lo, row_hi - 1))
        c0 = int(rng.integers(col_lo, col_hi - 1))
        placed = [(r0 + dr, c0 + dc) for dr, dc in cells]
        r_min = max(row_lo, min(r for r, _ in placed) - 1)
        r_max = min(row_hi, max(r for r, _ in placed) + 2)
        c_min = max(col_lo, min(c for _, c in placed) - 1)
        c_max = min(col_hi, max(c for _, c in placed) + 2)
        if np.all(grid[r_min:r_max, c_min:c_max] == bg):
            for r, c in placed:
                grid[r, c] = color
            return True
    return False


# ---------------------------------------------------------------------------
# Task 68a -- Count 4-connected objects in whole grid (grid -> text)
# ---------------------------------------------------------------------------


@register("68a")
def generate_count_objects_4(rng: np.random.Generator) -> dict:
    from generators.component_count_68 import generate
    return generate(rng, framed=False)


# ---------------------------------------------------------------------------
# Task 69a -- Count 8-connected objects in whole grid (grid -> text)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ROI helpers for tasks 68b and 69b
# ---------------------------------------------------------------------------


def _build_roi_scene(rng: np.random.Generator) -> tuple[np.ndarray, int, int, tuple[int, int, int, int]] | None:
    """Build a grid with a marked ROI rectangle and interior objects.

    Returns (grid, bg_color, roi_color, (roi_r, roi_c, roi_h, roi_w)) or None on failure.
    The ROI frame is a hollow rectangle drawn with roi_color.
    Interior objects are placed strictly inside the frame.
    Uses diverse polyomino shapes on a semantic single-color background.
    """
    h, w = random_grid_size(rng, min_dim=12, max_dim=28)
    grid, bg = random_background(rng, h, w, style="uniform")

    # Reserve one color for the ROI frame
    available = [c for c in range(NUM_COLORS) if c != bg]
    if len(available) < 2:
        return None
    roi_color = int(rng.choice(available))
    scene_colors = [c for c in available if c != roi_color]
    if not scene_colors:
        return None

    # Decide ROI rectangle position and size (interior must be at least 4x4)
    min_interior = 4
    roi_h = int(rng.integers(min_interior + 2, min(h - 1, 16) + 1))  # +2 for frame
    roi_w = int(rng.integers(min_interior + 2, min(w - 1, 16) + 1))
    if roi_h > h or roi_w > w:
        return None
    roi_r = int(rng.integers(0, h - roi_h + 1))
    roi_c = int(rng.integers(0, w - roi_w + 1))

    # Place some diverse shapes outside the ROI area
    n_outside = int(rng.integers(1, 4))
    for _ in range(n_outside):
        color = int(rng.choice(scene_colors))
        shape = random_shape(rng, size_range=(2, 5), connectivity=4)
        # Try to place outside the ROI region
        for _attempt in range(30):
            cells_ok = True
            trial_r = int(rng.integers(0, h))
            trial_c = int(rng.integers(0, w))
            for sr, sc in shape:
                nr, nc = trial_r + sr, trial_c + sc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    cells_ok = False
                    break
                if roi_r <= nr < roi_r + roi_h and roi_c <= nc < roi_c + roi_w:
                    cells_ok = False
                    break
                if grid[nr, nc] != bg:
                    cells_ok = False
                    break
            if cells_ok:
                for sr, sc in shape:
                    grid[trial_r + sr, trial_c + sc] = color
                break

    # Place objects inside the ROI interior (strictly inside the frame)
    interior_r0 = roi_r + 1
    interior_c0 = roi_c + 1
    interior_h = roi_h - 2
    interior_w = roi_w - 2

    # Create a sub-grid view for interior placement using diverse shapes
    n_inside = int(rng.integers(1, 6))
    for _ in range(n_inside):
        color = int(rng.choice(scene_colors))
        max_shape_size = min(4, interior_h, interior_w)
        if max_shape_size < 2:
            max_shape_size = 2
        shape = random_shape(rng, size_range=(1, max_shape_size), connectivity=4)
        # Try to place inside the interior
        for _attempt in range(30):
            cells_ok = True
            trial_r = int(rng.integers(interior_r0, interior_r0 + interior_h))
            trial_c = int(rng.integers(interior_c0, interior_c0 + interior_w))
            placed_cells = []
            for sr, sc in shape:
                nr, nc = trial_r + sr, trial_c + sc
                if nr < interior_r0 or nr >= interior_r0 + interior_h:
                    cells_ok = False
                    break
                if nc < interior_c0 or nc >= interior_c0 + interior_w:
                    cells_ok = False
                    break
                if grid[nr, nc] != bg:
                    cells_ok = False
                    break
                placed_cells.append((nr, nc))
            if cells_ok and placed_cells:
                for nr, nc in placed_cells:
                    grid[nr, nc] = color
                break

    # Now draw the ROI frame (overwriting whatever is at the border)
    stamp_rectangle(grid, roi_r, roi_c, roi_h, roi_w, roi_color, filled=False)

    return grid, bg, roi_color, (roi_r, roi_c, roi_h, roi_w)


def _normalize_shape_cells(
    shape: frozenset[tuple[int, int]] | set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, int]:
    """Normalize a shape to a 0-based bounding box."""
    cells = set(shape)
    r0, c0, r1, c1 = bounding_box(cells)
    normalized = [(r - r0, c - c0) for r, c in cells]
    return normalized, r1 - r0 + 1, c1 - c0 + 1


def _region_has_clearance(
    grid: np.ndarray,
    bg: int,
    cells: list[tuple[int, int]],
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    padding: int = 1,
) -> bool:
    """Check that candidate cells fit in a region with a moat from existing objects."""
    cell_set = set(cells)
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    rr0 = max(row_lo, min_r - padding)
    rr1 = min(row_hi, max_r + padding + 1)
    cc0 = max(col_lo, min_c - padding)
    cc1 = min(col_hi, max_c + padding + 1)
    for r in range(rr0, rr1):
        for c in range(cc0, cc1):
            if (r, c) in cell_set:
                if grid[r, c] != bg:
                    return False
            elif grid[r, c] != bg:
                return False
    return True


def _try_place_shape_in_region(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    bg: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    size_range: tuple[int, int] = (1, 4),
    padding: int = 1,
) -> bool:
    """Place one 4-connected shape strictly inside a rectangular region."""
    shape = random_shape(rng, size_range=size_range, connectivity=4)
    normalized, sh, sw = _normalize_shape_cells(shape)
    if row_hi - row_lo < sh or col_hi - col_lo < sw:
        return False
    for _ in range(120):
        base_r = int(rng.integers(row_lo, row_hi - sh + 1))
        base_c = int(rng.integers(col_lo, col_hi - sw + 1))
        cells = [(base_r + dr, base_c + dc) for dr, dc in normalized]
        if _region_has_clearance(grid, bg, cells, row_lo, row_hi, col_lo, col_hi, padding=padding):
            for r, c in cells:
                grid[r, c] = color
            return True
    return False


def _try_place_diagonal_object_in_region(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    bg: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    padding: int = 1,
) -> bool:
    """Place one same-color diagonal chain so 8-connectivity differs from 4-connectivity."""
    if row_hi - row_lo < 2 or col_hi - col_lo < 2:
        return False
    patterns = [
        [(0, 0), (1, 1)],
        [(0, 1), (1, 0)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    for _ in range(120):
        normalized = patterns[int(rng.integers(0, len(patterns)))]
        sh = max(r for r, _ in normalized) + 1
        sw = max(c for _, c in normalized) + 1
        if row_hi - row_lo < sh or col_hi - col_lo < sw:
            continue
        base_r = int(rng.integers(row_lo, row_hi - sh + 1))
        base_c = int(rng.integers(col_lo, col_hi - sw + 1))
        cells = [(base_r + dr, base_c + dc) for dr, dc in normalized]
        if _region_has_clearance(grid, bg, cells, row_lo, row_hi, col_lo, col_hi, padding=padding):
            for r, c in cells:
                grid[r, c] = color
            return True
    return False


def _build_targeted_roi_scene(
    rng: np.random.Generator,
    target_count: int,
    connectivity: int,
) -> tuple[np.ndarray, int, int, tuple[int, int, int, int]] | None:
    """Build a marked-ROI scene whose cropped interior realizes a chosen object count."""
    min_dim = 12 if target_count <= 2 else 14 if target_count <= 4 else 17
    h, w = random_grid_size(rng, min_dim=min_dim, max_dim=28)
    grid, bg = random_background(rng, h, w, style="uniform")

    available = [c for c in range(NUM_COLORS) if c != bg]
    if len(available) < 3:
        return None
    roi_color = int(rng.choice(available))
    scene_colors = [c for c in available if c != roi_color]

    min_interior = 5 if target_count <= 2 else 7 if target_count <= 4 else 9
    max_roi_h = min(h - 1, 18)
    max_roi_w = min(w - 1, 18)
    if max_roi_h < min_interior + 2 or max_roi_w < min_interior + 2:
        return None

    roi_h = int(rng.integers(min_interior + 2, max_roi_h + 1))
    roi_w = int(rng.integers(min_interior + 2, max_roi_w + 1))
    roi_r = int(rng.integers(0, h - roi_h + 1))
    roi_c = int(rng.integers(0, w - roi_w + 1))

    interior_r0 = roi_r + 1
    interior_c0 = roi_c + 1
    interior_r1 = roi_r + roi_h - 1
    interior_c1 = roi_c + roi_w - 1

    placed_ok = True
    used_diagonal = False
    for idx in range(target_count):
        color = int(rng.choice(scene_colors))
        if connectivity == 8 and (idx == 0 or (not used_diagonal and idx == target_count - 1)):
            placed = _try_place_diagonal_object_in_region(
                grid, rng, color, bg,
                interior_r0, interior_r1, interior_c0, interior_c1,
            )
            used_diagonal = used_diagonal or placed
        else:
            max_shape = 4 if connectivity == 4 else 3
            placed = _try_place_shape_in_region(
                grid, rng, color, bg,
                interior_r0, interior_r1, interior_c0, interior_c1,
                size_range=(1, max_shape),
            )
        if not placed:
            placed_ok = False
            break
    if not placed_ok:
        return None

    n_outside = int(rng.integers(1, 5))
    for _ in range(n_outside):
        color = int(rng.choice(scene_colors))
        _try_place_shape_in_region(
            grid, rng, color, bg,
            0, h, 0, w,
            size_range=(1, 4),
            padding=1,
        )

    stamp_rectangle(grid, roi_r, roi_c, roi_h, roi_w, roi_color, filled=False)

    interior = grid[interior_r0:interior_r1, interior_c0:interior_c1].copy()
    comps = connected_components(interior, connectivity=connectivity, background=bg)
    comps = [comp for comp in comps if comp["color"] != roi_color]
    if len(comps) != target_count:
        return None
    if connectivity == 8 and not any(_count_4_components_in_cells(comp["cells"]) > 1 for comp in comps):
        return None

    return grid, bg, roi_color, (roi_r, roi_c, roi_h, roi_w)


# ---------------------------------------------------------------------------
# Task 68b -- Count 4-connected objects inside rectangle (grid -> text)
# ---------------------------------------------------------------------------


@register("68b")
def generate_count_objects_4_roi(rng: np.random.Generator) -> dict:
    from generators.component_count_68 import generate
    return generate(rng, framed=True)


# ---------------------------------------------------------------------------
# Task 69b -- Count 8-connected objects inside rectangle (grid -> text)
# ---------------------------------------------------------------------------
