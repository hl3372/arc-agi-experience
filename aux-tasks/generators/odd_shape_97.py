"""Task 97 -- recolour the one color X object that is odd among its siblings.

The September 2026 rebuild replaced the inline version in
``generators/core2_extra/batch_object_a.py``; this is the follow-up pass that
closes what the first rebuild left open.

**Why the two criteria can never disagree.**  Two objects with the same
normalised shape always have the same number of cells, so an object whose cell
count is unique also has a unique shape.  Write ``U_c`` for the objects with a
unique count and ``U_s`` for those with a unique shape: ``U_c`` is a subset of
``U_s``.  The cell-count reading answers exactly when ``|U_c| = 1`` and the
shape reading exactly when ``|U_s| = 1``; if both answer, the one object in
``U_s`` is also the one in ``U_c``, so they name the SAME object.  A scene where
the two criteria single out different objects does not exist, and neither does
one where the shape reading answers while two objects carry unique counts.
That is a fact about the two rules, not about this builder: no construction can
make "apply whichever criterion singles out exactly one object" wrong, and no
construction can make "recolour the uniquely largest or smallest object" wrong
either (a uniquely extreme count IS a unique count, so it is in ``U_c``, and
when the stated reading answers, ``U_c`` is either empty or exactly the answer).

What is left to design is therefore not disagreement but **balance**, and the
identity ``criterion classifier = 0.5 + (share the criterion decides) / 2``
says exactly how much of each is available:

* ``agree`` (share ``AGREE_P``): every non-odd shape repeats AND the odd cell
  count is unique, so BOTH readings answer and name the same object.  The scene
  is drawn by one code path in both variants, so nothing in the picture says
  which criterion the prompt will state.
* ``shape only`` (the rest of the shape examples): the non-odd shapes still
  repeat, but the odd figure carries the SAME cell count as one of its
  siblings, so the count reading finds no unique object.
* ``count only`` (the rest of the cell-count examples): the odd count is unique
  but one sibling group has pairwise different shapes, so the shape reading
  finds three or more unique shapes and no answer.

``AGREE_P = 0.72`` realises 26% and 64% over 3000 seeds (the classifier was
99-100% before this pass).  The same identity prices the other reader: a solver
that always applies ONE criterion is right wherever that reading answers, which
is 85% (shape) and 90% (count) here, and

    fixed-reader accuracy + criterion classifier = 1.5

exactly, so the two cannot both be pushed below 0.75.  Lowering ``AGREE_P``
walks back along that line (0.5 puts both at 75% and the criterion decides
50%); the review asked for the classifier, so the classifier is what this
setting buys.

The other standing properties:

* **The scene is one archetype for both criteria.**  The non-odd figures are
  split into groups of at least two that share a cell count; each group either
  echoes one body (every member a translation of it) or scatters pairwise
  different bodies of that count.  Group sizes, group counts, bodies, canvas,
  palette and texture are drawn the same way whichever criterion is stated;
  only "all groups echo" and "the odd count is unique" follow it, and those are
  the two facts the identity above already prices in.
* **The odd count sits between its siblings' where the scene allows it.**  With
  two or more groups the odd figure's count is drawn strictly inside their
  range, which is what stops "recolour the biggest (or smallest) object" from
  firing.  It cannot be made to fire and be WRONG (see the theorem), so the
  lever is how often it fires at all: 50% before this pass, 23% after, and
  what is left is the three- and four-figure scenes, where the non-odd
  figures form ONE group and the odd count has nothing to sit between.
* **The connectivity is load-bearing and unpredictable.**  On a planned share
  the odd figure is a sibling's body with a second clean piece joined at a
  corner: under 8-connectivity it is one object whose shape (and count) is
  unique, under 4-connectivity the extra piece is the object that stands alone,
  so the two readings recolour different cells.  The whole scene is drawn
  before the connectivity exists and every check runs on both readings, so the
  coin is fair.
* **Coverage**: independent 5..30 sides (no ``square_prob``, no token-budget
  cap), palettes of 2..9 input colours (the recolour colour is reserved),
  bodies from the shared vocabulary through ``border_objects_90``'s clean-body
  builder, scaled with the canvas, and sparse textures from
  ``common/backgrounds.py`` that the prompt's isolated-cell clause disposes of.
  The background is never named.
* **The generator proves the answer twice.**  ``validate_example`` re-reads the
  emitted grid with the literal reader AND with a second, independent one
  (union-find labelling, string shape signatures), compares both to the output
  and to the construction plan, and raises on any disagreement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import apply_background, infer_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

# module import, attributes looked up lazily: a direct ``from ... import`` would
# create a registry-time import cycle (registry imports 90 before this module)
import generators.border_objects_90 as _b90


MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 40
TEXTURE_MIN_SIDE = 8
TEXTURE_P = 0.34
SPECKLE_P = 0.40
BG_MARGIN = 1.25
MIN_PIECE = 3                 # the smallest 4-connected piece of a target figure
MAX_FIGURE_CELLS = 64
MAX_OBJECTS = 8

# both readings answer and name the same object; the remainder is split between
# the two one-sided cells, which is what the criterion clause decides
AGREE_P = 0.72
# the odd cell count is drawn strictly between two group counts when there are
# two or more groups: the only honest lever against the size-rank shortcut
MIDDLE_P = 0.88
# the odd figure is a sibling's body with a clean piece joined at a corner
JOIN_P = 0.42
# a whole echo group is built from corner-joined bodies (variety, not an answer)
GROUP_JOIN_P = 0.14
# in the shape-only cell the odd figure is a rotation or reflection of the body
# it shares its cell count with, which is what "do not rotate or reflect" decides
ROTATED_P = 0.95

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# near-flat over 2..10 with a mild lean to the top; the recolour colour is
# reserved, so the INPUT reaches 9 (the standing trade-off on recolour tasks)
_PALETTE_SIZES = np.arange(2, 11)
_PALETTE_P = np.array([.09, .10, .11, .11, .11, .11, .12, .12, .13])


# ---------------------------------------------------------------------------
# Literal readers (pure Python, no numpy, no access to the plan)
# ---------------------------------------------------------------------------

def _normalize(cells) -> frozenset:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _canonical(cells) -> frozenset:
    """The shape up to the eight rotations and reflections (the reading the
    prompt forbids with "do not rotate or reflect")."""
    best = None
    current = set(cells)
    for _ in range(4):
        current = {(c, -r) for r, c in current}
        for variant in (current, {(r, -c) for r, c in current}):
            key = _normalize(variant)
            ordered = tuple(sorted(key))
            if best is None or ordered < best[0]:
                best = (ordered, key)
    return best[1]


def color_objects(grid, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    """The color X objects as the prompt defines them: same-colour groups of at
    least two cells under the stated connectivity (a lone cell is texture)."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] == color}
    out: list[list[tuple[int, int]]] = []
    while pending:
        start = min(pending)
        pending.remove(start)
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) >= 2:
            out.append(comp)
    return out


def _key(comp, criterion: str, invariant: bool = False):
    if criterion == "cell_count":
        return len(comp)
    if criterion != "shape":
        raise ValueError(f"unknown criterion: {criterion}")
    return _canonical(comp) if invariant else _normalize(comp)


def solve(grid, color: int, connectivity: int, criterion: str,
          invariant: bool = False) -> Optional[list[tuple[int, int]]]:
    """The cells of the object the prompt asks for, or None when the reading
    does not single out exactly one object."""
    comps = color_objects(grid, color, connectivity)
    keys = [_key(comp, criterion, invariant) for comp in comps]
    odd = [comp for comp, key in zip(comps, keys) if keys.count(key) == 1]
    if len(odd) != 1:
        return None
    return sorted(odd[0])


# ---------------------------------------------------------------------------
# A second, independent reader: union-find labelling and string signatures.
# Nothing here shares a line with ``color_objects`` / ``solve``; the validator
# runs both on the emitted grid and raises when they disagree.
# ---------------------------------------------------------------------------

def _oracle_objects(grid, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    rows = [list(row) for row in grid]
    height, width = len(rows), len(rows[0])
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(cell):
        root = cell
        while parent[root] != root:
            root = parent[root]
        while parent[cell] != root:
            parent[cell], cell = root, parent[cell]
        return root

    back = ((-1, 0), (0, -1)) if connectivity == 4 else ((-1, 0), (0, -1), (-1, -1), (-1, 1))
    for r in range(height):
        for c in range(width):
            if rows[r][c] != color:
                continue
            parent[(r, c)] = (r, c)
            for dr, dc in back:
                q = (r + dr, c + dc)
                if q in parent:
                    a, b = find((r, c)), find(q)
                    if a != b:
                        parent[b] = a
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in list(parent):
        groups.setdefault(find(cell), []).append(cell)
    return [sorted(cells) for cells in groups.values() if len(cells) >= 2]


def _oracle_signature(cells, criterion: str) -> str:
    if criterion == "cell_count":
        return f"n={len(cells)}"
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    height = max(r for r, _ in cells) - top + 1
    width = max(c for _, c in cells) - left + 1
    bitmap = ["."] * (height * width)
    for r, c in cells:
        bitmap[(r - top) * width + (c - left)] = "#"
    return f"{height}x{width}:" + "".join(bitmap)


def oracle_solve(grid, color: int, connectivity: int,
                 criterion: str) -> Optional[list[tuple[int, int]]]:
    comps = _oracle_objects(grid, color, connectivity)
    signatures = [_oracle_signature(comp, criterion) for comp in comps]
    alone = [comp for comp, sig in zip(comps, signatures) if signatures.count(sig) == 1]
    if len(alone) != 1:
        return None
    return sorted(alone[0])


def _is_extreme(value: int, sizes: list[int]) -> bool:
    """True when ``value`` is the ONE largest or the ONE smallest size: what a
    reader guessing "the odd size out" would pick without measuring."""
    return ((sizes.count(max(sizes)) == 1 and value == max(sizes))
            or (sizes.count(min(sizes)) == 1 and value == min(sizes)))


def recolor(grid, cells, target_color: int) -> list[list[int]]:
    out = [list(row) for row in grid]
    for r, c in cells:
        out[r][c] = int(target_color)
    return out


def read_ground(grid) -> int:
    """The colour a reader told only "ignoring the background" settles on: the
    most frequent colour once every isolated single cell is set aside."""
    return _b90.read_ground(grid)


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------

def _sized_body(rng: np.random.Generator, cells: float, lim_h: int, lim_w: int,
                *, min_cells: int = MIN_PIECE, fill: float = 0.35,
                exact: Optional[int] = None) -> Optional[tuple[np.ndarray, str]]:
    """A clean body of about ``cells`` cells whose box fits ``lim_h x lim_w``.

    ``fill`` is a floor on how much of its box the body covers, so the library's
    bar family cannot answer a roomy box with a one-cell-wide stick.  ``exact``
    asks for a body with exactly that many cells (that is how a group comes to
    share one cell count, and how the odd figure comes to share its siblings');
    both are rejections, never a fallback shape.
    """
    lim_h, lim_w = max(1, int(lim_h)), max(1, int(lim_w))
    floor = max(2, int(min_cells))
    if exact is not None:
        floor = min(floor, int(exact))
        cells = float(exact)
    if lim_h * lim_w < floor:
        return None
    side = float(np.sqrt(max(1.0, float(cells))))
    tries = 26 if exact is not None else 10
    for _ in range(tries):
        bh = int(np.clip(round(side * float(rng.uniform(0.75, 1.6))), 1, lim_h))
        bw = int(np.clip(round(side * float(rng.uniform(0.75, 1.6))), 1, lim_w))
        for _ in range(10):
            if bh * bw >= floor:
                break
            if bh < lim_h and (bh <= bw or bw >= lim_w):
                bh += 1
            elif bw < lim_w:
                bw += 1
            else:
                break
        want = floor if exact is not None else max(floor, int(round(fill * bh * bw)))
        built = _b90._body(rng, bh, bw, min_cells=want)
        if built is None:
            continue
        mask, fam = built
        if exact is not None and int(mask.sum()) != int(exact):
            continue
        return np.ascontiguousarray(mask), fam
    return None


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    current = mask
    for _ in range(4):
        current = np.ascontiguousarray(np.rot90(current))
        out.append(current)
        out.append(np.ascontiguousarray(current[:, ::-1]))
    return out


def _mask_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _shape_of(mask: np.ndarray) -> frozenset:
    return _normalize(_mask_cells(mask))


def _rotated_twin(rng: np.random.Generator, mask: np.ndarray, lim_h: int,
                  lim_w: int) -> Optional[np.ndarray]:
    """A rotation or reflection of ``mask`` that is not a translation of it."""
    base = _shape_of(mask)
    cands = [m for m in _orientations(mask)
             if _shape_of(m) != base and m.shape[0] <= lim_h and m.shape[1] <= lim_w]
    if not cands:
        return None
    return np.ascontiguousarray(cands[int(rng.integers(0, len(cands)))])


def _join(rng: np.random.Generator, masks: list[np.ndarray], lim_h: int,
          lim_w: int) -> Optional[np.ndarray]:
    joined = _b90._corner_join(rng, masks)
    if joined is None:
        return None
    fig = joined[0]
    if fig.shape[0] > lim_h or fig.shape[1] > lim_w:
        return None
    return np.ascontiguousarray(fig)


def _joined_body(rng: np.random.Generator, cells: int, lim_h: int, lim_w: int,
                 floor: int) -> Optional[tuple[np.ndarray, str]]:
    """A figure of exactly ``cells`` cells made of two clean pieces joined only
    at a corner (one object under 8-connectivity, two under 4)."""
    cells = int(cells)
    if cells < 2 * floor:
        return None
    for _ in range(6):
        first_cells = int(rng.integers(floor, cells - floor + 1))
        rest = cells - first_cells
        first = _sized_body(rng, first_cells, max(1, lim_h - 1), max(1, lim_w - 1),
                            min_cells=min(floor, first_cells), exact=first_cells)
        second = _sized_body(rng, rest, max(1, lim_h - 1), max(1, lim_w - 1),
                             min_cells=min(floor, rest), exact=rest)
        if first is None or second is None:
            continue
        fig = _join(rng, [first[0], second[0]], lim_h, lim_w)
        if fig is None or int(fig.sum()) != cells:
            continue
        return fig, first[1] + "+" + second[1]
    return None


def _bodies_with_count(rng: np.random.Generator, count: int, how_many: int,
                       lim_h: int, lim_w: int, floor: int,
                       taken: set) -> Optional[tuple[list[np.ndarray], list[str]]]:
    """``how_many`` bodies of exactly ``count`` cells with pairwise new shapes.

    This is the SCATTER group: every member's shape occurs once, so the shape
    reading finds three or more lone objects and no answer, while the count
    reading is untouched.  Orientations of one body are different shapes under
    the translation-only rule, so a drawn family fills a group naturally.
    """
    masks: list[np.ndarray] = []
    kinds: list[str] = []
    for _ in range(120):
        if len(masks) == how_many:
            break
        built = _sized_body(rng, count, lim_h, lim_w,
                            min_cells=min(floor, count), exact=count)
        if built is None:
            continue
        mask, fam = built
        variants = [mask] + _orientations(mask)
        keep = 1 if rng.random() < 0.5 else 2
        added = 0
        for cand in variants:
            if len(masks) == how_many or added == keep:
                break
            if cand.shape[0] > lim_h or cand.shape[1] > lim_w:
                continue
            shape = _shape_of(cand)
            if shape in taken:
                continue
            taken.add(shape)
            masks.append(np.ascontiguousarray(cand))
            kinds.append(fam)
            added += 1
    if len(masks) != how_many:
        return None
    return masks, kinds


# ---------------------------------------------------------------------------
# The figure set: ONE archetype, drawn the same way for both criteria
# ---------------------------------------------------------------------------

def _group_sizes(rng: np.random.Generator, n_common: int) -> Optional[list[int]]:
    """Split the non-odd figures into groups of at least two.

    Two or more groups are what let the odd figure's cell count sit BETWEEN its
    siblings' instead of at one end, so they are preferred whenever four or
    more non-odd figures are available.
    """
    if n_common < 2:
        return None
    if n_common < 4:
        return [n_common]
    for _ in range(8):
        sizes, left = [], n_common
        while left >= 2:
            size = int(rng.integers(2, 4))
            if left - size < 2:
                size = left
            sizes.append(size)
            left -= size
        if left == 0 and len(sizes) >= 2:
            return sizes
    return [n_common]


def _counts(rng: np.random.Generator, plan: dict,
            groups: list[int]) -> Optional[dict]:
    """One cell count per group plus the odd figure's.

    ``count_unique`` says whether the odd count occurs once (the count reading
    answers) or equals a group's (it does not).  When the odd figure is
    corner-joined its count is the sum of the sibling body it copies and the
    extra piece, so the extra piece's size is drawn here too: under
    4-connectivity that piece is the object standing alone, and its count must
    be unique among the group counts for the cell-count reading to survive the
    other half of the coin.
    """
    floor = max(2, int(plan["piece_floor"]))
    k = len(groups)
    base = int(np.clip(round(plan["per_figure"]), floor + 2, MAX_FIGURE_CELLS))
    spread = max(3, int(round(base * 0.7)))
    lo = max(floor, base - spread)
    hi = int(min(MAX_FIGURE_CELLS, base + spread))
    pool = list(range(lo, hi + 1))
    if len(pool) < k + 2:
        return None
    unique = bool(plan["count_unique"])
    for _ in range(24):
        if unique:
            # draw the k + 1 counts together and give the odd figure an INTERIOR
            # one: that is what keeps "recolour the biggest (or the smallest)"
            # from answering, and it is only available with two or more groups
            values = sorted(int(v) for v in rng.choice(pool, size=k + 1, replace=False))
            if k >= 2 and rng.random() < MIDDLE_P:
                index = int(rng.integers(1, k))
            else:
                index = int(rng.integers(0, k + 1))
            odd_count = values[index]
            group_counts = [v for i, v in enumerate(values) if i != index]
        else:
            group_counts = sorted(int(v) for v in rng.choice(pool, size=k, replace=False))
            odd_count = int(group_counts[int(rng.integers(0, k))])
        if plan["join"] != "odd":
            return {"group_counts": group_counts, "odd_count": odd_count,
                    "extra": None, "anchor": None}
        # The odd figure is a sibling's body plus an extra piece.  Under
        # 4-connectivity that extra piece is the object standing alone, so its
        # cell count has to sit between the group counts as well, or the
        # size-rank shortcut answers that half of the coin.
        cands = []
        for index, anchor in enumerate(group_counts):
            extra = odd_count - anchor
            if extra < floor or extra in group_counts:
                continue
            inner = group_counts[0] < extra < group_counts[-1]
            cands.append((1 if inner else 0, index, extra))
        if not cands:
            continue
        best = max(score for score, _, _ in cands)
        pick = [c for c in cands if c[0] == best] if rng.random() < MIDDLE_P else cands
        _, index, extra = pick[int(rng.integers(0, len(pick)))]
        return {"group_counts": group_counts, "odd_count": odd_count,
                "extra": int(extra), "anchor": int(index)}
    return None


def _figures(rng: np.random.Generator, plan: dict, lim_h: int,
             lim_w: int) -> Optional[dict]:
    """Every figure in the scene, built from the archetype in ``plan``."""
    groups = _group_sizes(rng, plan["n_objects"] - 1)
    if groups is None:
        return None
    spec = _counts(rng, plan, groups)
    if spec is None:
        return None
    floor = max(2, int(plan["piece_floor"]))
    count = len(groups)
    members: list[list[np.ndarray]] = [[] for _ in range(count)]
    kinds_of: list[list[str]] = [[] for _ in range(count)]
    bodies: list[Optional[np.ndarray]] = [None] * count
    shapes: set = set()
    joined_groups = 0

    # The group the odd figure shares its cell count with is built LAST, as a
    # rotation of the odd figure: that is what makes "do not rotate or reflect"
    # decide the answer, and building it from the odd (rather than the odd from
    # it) keeps the corner-join draw independent of this cell, so the joined
    # figures a reader can see say nothing about which criterion is stated.
    rotate_at = -1
    if not plan["count_unique"] and rng.random() < ROTATED_P:
        twins = [i for i, value in enumerate(spec["group_counts"])
                 if value == spec["odd_count"]]
        if twins:
            rotate_at = int(twins[0])

    for index, (size, cells) in enumerate(zip(groups, spec["group_counts"])):
        if index == rotate_at:
            continue
        if plan["all_echo"]:
            built = None
            if index == plan["group_join"]:
                built = _joined_body(rng, cells, lim_h, lim_w, floor)
                if built is not None:
                    joined_groups += 1
            if built is None:
                built = _sized_body(rng, cells, lim_h, lim_w,
                                    min_cells=min(floor, cells), exact=cells)
            if built is None:
                return None
            mask, fam = built
            shape = _shape_of(mask)
            if shape in shapes:
                return None
            shapes.add(shape)
            bodies[index] = mask
            members[index] = [mask] * size
            kinds_of[index] = [fam] * size
        else:
            built = _bodies_with_count(rng, cells, size, lim_h, lim_w, floor, shapes)
            if built is None:
                return None
            bodies[index] = built[0][0]
            members[index] = built[0]
            kinds_of[index] = built[1]

    odd: Optional[np.ndarray] = None
    odd_kind = "odd"
    if spec["extra"] is not None:
        # the odd figure is a sibling's body with a clean piece joined at a
        # corner: one object under 8-connectivity, two under 4
        anchor = bodies[spec["anchor"]]
        if anchor is None:
            return None
        for _ in range(8):
            extra = _sized_body(rng, spec["extra"], max(1, lim_h - 1), max(1, lim_w - 1),
                                min_cells=min(floor, spec["extra"]), exact=spec["extra"])
            if extra is None or _shape_of(extra[0]) in shapes:
                continue
            fig = _join(rng, [np.array(anchor), extra[0]], lim_h, lim_w)
            if fig is None or _shape_of(fig) in shapes:
                continue
            odd, odd_kind = fig, extra[1] + "_joined"
            break
    if odd is None:
        for _ in range(10):
            built = _sized_body(rng, spec["odd_count"], lim_h, lim_w,
                                min_cells=min(floor, spec["odd_count"]),
                                exact=spec["odd_count"])
            if built is None or _shape_of(built[0]) in shapes:
                continue
            odd, odd_kind = built[0], built[1]
            break
    if odd is None:
        return None

    if rotate_at >= 0:
        twin = _rotated_twin(rng, odd, lim_h, lim_w)
        if twin is None or _shape_of(twin) in shapes:
            return None
        shapes.add(_shape_of(twin))
        bodies[rotate_at] = twin
        members[rotate_at] = [twin] * groups[rotate_at]
        kinds_of[rotate_at] = ["rotated"] * groups[rotate_at]

    masks = [mask for group in members for mask in group]
    kinds = [kind for group in kinds_of for kind in group] + [odd_kind]
    if len(masks) != plan["n_objects"] - 1:
        return None
    order = list(rng.permutation(len(masks) + 1))
    everything = masks + [odd]
    odd_at = order.index(len(masks))
    return {"masks": [everything[i] for i in order], "odd_index": odd_at,
            "kinds": kinds, "groups": groups, "rotated_group": rotate_at >= 0,
            "group_counts": spec["group_counts"], "odd_count": spec["odd_count"],
            "extra_cells": spec["extra"], "joined_groups": joined_groups}


# ---------------------------------------------------------------------------
# The plan (everything except the connectivity)
# ---------------------------------------------------------------------------

def sample_plan(rng: np.random.Generator) -> dict:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    density = float(rng.uniform(0.10, 0.26))
    criterion = "shape" if rng.random() < 0.5 else "cell_count"
    # the two switches the criterion is allowed to touch, and nothing else
    agree = bool(rng.random() < AGREE_P)
    all_echo = bool(agree or criterion == "shape")
    count_unique = bool(agree or criterion == "cell_count")
    join = "odd" if rng.random() < JOIN_P else "none"
    # a figure plus the one-cell moat that keeps the target objects apart costs
    # about twenty cells; three objects is the floor for an odd one out, and
    # five is the floor for two sibling groups, which is what lets the odd cell
    # count sit between theirs
    cap = int(np.clip(area // 20, 3, MAX_OBJECTS))
    low = 5 if cap >= 5 else 3
    n_objects = int(rng.integers(low, cap + 1))
    group_join = -1
    if all_echo and n_objects >= 5 and rng.random() < GROUP_JOIN_P:
        group_join = 0
    piece_floor = MIN_PIECE if area >= 120 else 2
    per_figure = float(np.clip(density * area / n_objects,
                               float(piece_floor + 1), float(MAX_FIGURE_CELLS)))

    bg = int(rng.integers(0, NUM_COLORS))
    pool = [c for c in range(NUM_COLORS) if c != bg]
    object_color = int(rng.choice(pool))
    target_color = int(rng.choice([c for c in pool if c != object_color]))
    palette = int(rng.choice(_PALETTE_SIZES, p=_PALETTE_P))
    texture = bool(min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P)
    # the recolour colour is reserved for the output, so the input palette is
    # ground + object colour + distractors (+ texture)
    budget = int(np.clip(area // 30, 0, 7))
    n_distractors = int(np.clip(palette - 2 - (1 if texture else 0), 0, budget))
    if n_distractors and rng.random() < 0.35:
        n_distractors = min(budget, n_distractors + int(rng.integers(0, 2)))
    distractor_colors = [int(c) for c in rng.choice(
        [c for c in pool if c != object_color and c != target_color],
        size=min(n_distractors, 8), replace=False)] if n_distractors else []
    return {
        "h": h, "w": w, "area": area, "density": density,
        "criterion": criterion, "agree": agree,
        "all_echo": all_echo, "count_unique": count_unique,
        "join": join, "group_join": group_join,
        "n_objects": n_objects, "per_figure": per_figure, "piece_floor": piece_floor,
        "bg": bg, "object_color": object_color, "target_color": target_color,
        "distractor_colors": distractor_colors, "palette_target": palette,
        "plan_texture": texture,
        "per_distractor": float(np.clip(density * area * 0.45 / max(1, n_distractors),
                                        4.0, 70.0)),
    }


def structure_label(plan: dict) -> str:
    """A compact name for the archetype (metadata only; never in the prompt)."""
    return ("echo" if plan["all_echo"] else "scatter") \
        + ("_unique" if plan["count_unique"] else "_shared") \
        + ("_join" if plan["join"] == "odd" else "")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _place(rng: np.random.Generator, grid: np.ndarray, occ: np.ndarray,
           mask: np.ndarray, color: int, free: np.ndarray) -> Optional[np.ndarray]:
    if mask.shape[0] > grid.shape[0] or mask.shape[1] > grid.shape[1]:
        return None
    ok = placement_map(mask, free)
    if ok is None:
        return None
    spots = np.argwhere(ok)
    if not len(spots):
        return None
    r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
    cells = np.zeros(grid.shape, dtype=bool)
    cells[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
    grid[cells] = int(color)
    occ |= cells
    return cells


def build_scene(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    h, w, bg = plan["h"], plan["w"], plan["bg"]
    lim_h = max(1, min(h - 1, 13))
    lim_w = max(1, min(w - 1, 13))
    built = _figures(rng, plan, lim_h, lim_w)
    if built is None:
        return None
    grid = np.full((h, w), bg, dtype=np.int16)
    occ = np.zeros((h, w), dtype=bool)
    target_occ = np.zeros((h, w), dtype=bool)
    odd_cells = None
    figures: list[list[tuple[int, int]]] = []
    sizes = []
    for index, mask in enumerate(built["masks"]):
        # target objects never come within one cell of each other: they must not
        # merge under 8-connectivity, which would rewrite the shape multiset
        cells = _place(rng, grid, occ, mask, plan["object_color"], free=~halo(occ))
        if cells is None:
            return None
        target_occ |= cells
        sizes.append(int(mask.sum()))
        figures.append(sorted((int(r), int(c)) for r, c in np.argwhere(cells)))
        if index == built["odd_index"]:
            odd_cells = cells
    if odd_cells is None:
        return None

    placed, touching = 0, 0
    distractor_kinds: list[str] = []
    for color in plan["distractor_colors"]:
        for shrink in (1.0, 0.6, 0.35):
            body = _sized_body(rng, plan["per_distractor"] * shrink, lim_h, lim_w,
                               min_cells=max(2, plan["piece_floor"]))
            if body is None:
                continue
            # a third of the distractors may touch a target object: different
            # colours never merge under the rule, and the contact is what a
            # reader who dropped the colour clause would fall for
            contact = rng.random() < 0.34
            cells = _place(rng, grid, occ, body[0], int(color),
                           free=(~occ) if contact else (~halo(occ)))
            if cells is None:
                cells = _place(rng, grid, occ, body[0], int(color), free=~halo(occ))
            if cells is None:
                continue
            placed += 1
            distractor_kinds.append(body[1])
            if bool((halo(cells) & target_occ).any()):
                touching += 1
            break
    return {"grid": grid, "occ": occ, "target_occ": target_occ,
            "odd_cells": odd_cells, "object_sizes": sizes, "figures": figures,
            "odd_index": built["odd_index"],
            "groups": built["groups"], "group_counts": built["group_counts"],
            "odd_count": built["odd_count"], "extra_cells": built["extra_cells"],
            "joined_groups": built["joined_groups"],
            "rotated_group": built["rotated_group"],
            "kinds": built["kinds"], "distractor_kinds": distractor_kinds,
            "n_distractors": placed, "distractors_touching": touching}


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(rng: np.random.Generator, grid: np.ndarray, bg: int, occ: np.ndarray,
             *, exclude, enabled: bool) -> tuple[np.ndarray, dict]:
    """A sparse library texture, clear of every object by one cell.

    Sparse only: its cells are isolated single cells, which the prompt's object
    rule already disposes of.  A solid band would read as another object.
    """
    if not enabled:
        return grid, dict(_NO_TEXTURE)
    kind = "multi_speckle" if rng.random() < SPECKLE_P else None
    painted = apply_background(rng, grid, bg, halo(occ), group="sparse", kind=kind,
                               exclude=[int(c) for c in exclude])
    if painted is None:
        return grid, dict(_NO_TEXTURE)
    display, info = painted
    return display, {"texture_group": info["group"], "texture_kind": info["kind"],
                     "texture_colors": [int(c) for c in info["texture_colors"]],
                     "texture_named": False,
                     "texture_coverage": round(float(info["coverage"]), 4)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(example: dict, plan: Optional[dict] = None) -> None:
    """Re-derive the answer twice from the emitted grid and raise on any
    disagreement: once with the literal reader, once with the independent
    union-find oracle, and (when the builder passes it) against the plan that
    drew the picture.
    """
    grid, meta = example["input"], example["metadata"]
    color = int(meta["object_color"])
    target = int(meta["target_color"])
    conn = int(meta["connectivity"])
    criterion = meta["criterion"]
    other_conn = 4 if conn == 8 else 8

    cells = solve(grid, color, conn, criterion)
    if cells is None:
        raise ValueError("task 97: the stated reading does not single out one object")
    if recolor(grid, cells, target) != example["output"]:
        raise ValueError("task 97: the output disagrees with the literal reading")
    if len(cells) != meta["odd_cells"]:
        raise ValueError("task 97: the odd object's cell count disagrees with the metadata")

    # the second implementation of the same rule
    twin = oracle_solve(grid, color, conn, criterion)
    if twin is None or twin != cells:
        raise ValueError("task 97: the independent oracle disagrees with the reader")
    for other in (4, 8):
        for rule in ("shape", "cell_count"):
            if oracle_solve(grid, color, other, rule) != solve(grid, color, other, rule):
                raise ValueError(
                    f"task 97: the two readers disagree ({other}-connected, {rule})")

    comps = color_objects(grid, color, conn)
    if len(comps) != meta["n_objects"]:
        raise ValueError("task 97: the object count disagrees with the metadata")
    if sum(len(comp) for comp in comps) != sum(
            1 for row in grid for value in row if int(value) == color):
        raise ValueError("task 97: a color X cell is not part of an object")
    if any(len(comp) < 2 for comp in color_objects(grid, color, 4)):
        raise ValueError("task 97: a 4-connected piece of the target colour is a single cell")

    if plan is not None:
        # the picture shows the figures the builder drew, and the figure it
        # meant to be odd is the one the 8-connected reading singles out
        drawn = {frozenset(cells_) for cells_ in plan["figures"]}
        seen = {frozenset((int(r), int(c)) for r, c in comp)
                for comp in color_objects(grid, color, 8)}
        if drawn != seen:
            raise ValueError("task 97: the drawn figures are not the objects on the grid")
        planned_odd = sorted(plan["figures"][plan["odd_index"]])
        if solve(grid, color, 8, criterion) != planned_odd:
            raise ValueError("task 97: the odd figure is not the one the plan drew")
        if sorted(len(cells_) for cells_ in plan["figures"]) != sorted(
                len(comp) for comp in color_objects(grid, color, 8)):
            raise ValueError("task 97: the drawn cell counts are not the visible ones")

    alt = solve(grid, color, other_conn, criterion)
    if (alt is not None and set(alt) != set(cells)) != meta["differs_under_other_connectivity"]:
        raise ValueError("task 97: the connectivity flag is wrong")
    if alt is None:
        raise ValueError("task 97: the other connectivity has no unique object")

    other_criterion = "cell_count" if criterion == "shape" else "shape"
    swapped = solve(grid, color, conn, other_criterion)
    if (swapped is not None) != meta["other_criterion_has_answer"]:
        raise ValueError("task 97: the other-criterion flag is wrong")
    if (swapped is None or set(swapped) != set(cells)) != meta["differs_under_other_criterion"]:
        raise ValueError("task 97: the criterion flag is wrong")
    if swapped is not None and set(swapped) != set(cells):
        # the theorem in the module docstring: this cannot happen
        raise ValueError("task 97: the two criteria disagree, which is impossible")

    loose = solve(grid, color, conn, criterion, invariant=True)
    if (loose is None or set(loose) != set(cells)) != meta["differs_under_rotation_invariant"]:
        raise ValueError("task 97: the rotation flag is wrong")

    sizes = [len(comp) for comp in comps]
    if _is_extreme(len(cells), sizes) != meta["odd_is_extreme_size"]:
        raise ValueError("task 97: the extreme-size flag is wrong")

    if any(int(value) == target for row in grid for value in row):
        raise ValueError("task 97: the recolour colour already occurs in the input")

    arr = np.asarray(grid)
    info = infer_background(arr, margin=BG_MARGIN)
    if not info["identifiable"] or int(info["color"]) != int(meta["bg_color"]):
        raise ValueError("task 97: the background is not identifiable")
    if read_ground(grid) != int(meta["bg_color"]):
        raise ValueError("task 97: the visible ground disagrees with the plan")
    if meta["texture_colors"]:
        if color in meta["texture_colors"] or target in meta["texture_colors"]:
            raise ValueError("task 97: a texture colour is an object colour")
        tex = np.isin(arr, meta["texture_colors"])
        if not bool(isolated_cell_mask(arr)[tex].all()):
            raise ValueError("task 97: a texture cell is not isolated")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = sample_plan(rng)
        scene = None
        for _ in range(3):
            scene = build_scene(rng, plan)
            if scene is not None:
                break
        if scene is None:
            continue
        display, tex = _texture(rng, scene["grid"], plan["bg"], scene["occ"],
                                exclude=[plan["object_color"], plan["target_color"],
                                         *plan["distractor_colors"]],
                                enabled=plan["plan_texture"])
        rows = grid_to_list(display.astype(np.int16))
        color, criterion = plan["object_color"], plan["criterion"]

        # Nothing above this line has looked at the connectivity, and every
        # check below runs on BOTH readings before the coin is flipped.
        odd8 = solve(rows, color, 8, criterion)
        odd4 = solve(rows, color, 4, criterion)
        if odd8 is None or odd4 is None:
            continue
        if set(odd8) != {(int(r), int(c)) for r, c in np.argwhere(scene["odd_cells"])}:
            continue
        if any(len(comp) < 2 for comp in color_objects(rows, color, 4)):
            continue
        if read_ground(rows) != plan["bg"]:
            continue
        info = infer_background(display, margin=BG_MARGIN)
        if not info["identifiable"] or int(info["color"]) != plan["bg"]:
            continue

        conn = 4 if rng.random() < 0.5 else 8
        cells = odd4 if conn == 4 else odd8
        alt = odd8 if conn == 4 else odd4
        other_criterion = "cell_count" if criterion == "shape" else "shape"
        swapped = solve(rows, color, conn, other_criterion)
        loose = solve(rows, color, conn, criterion, invariant=True)
        comps = color_objects(rows, color, conn)
        comps4 = color_objects(rows, color, 4)
        out = recolor(rows, cells, plan["target_color"])
        present = sorted({int(v) for v in np.unique(display)})
        occ_cells = int(scene["occ"].sum())
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"], "n_colors": len(present),
            "connectivity": conn,
            "criterion": criterion,
            # kept for continuity with the pre-rebuild metadata contract
            "equivalence_rule": ("same_normalized_shape" if criterion == "shape"
                                 else "same_cell_count"),
            "structure": structure_label(plan),
            "non_odd_shapes_repeat": bool(plan["all_echo"]),
            "odd_count_is_unique": bool(plan["count_unique"]),
            "readings_agree": bool(plan["agree"]),
            "n_sibling_groups": len(scene["groups"]),
            "sibling_group_sizes": list(scene["groups"]),
            "sibling_group_counts": [int(v) for v in scene["group_counts"]],
            "object_color": int(plan["object_color"]),
            "target_color": int(plan["target_color"]),
            "background_color": int(plan["bg"]),
            "bg_color": int(plan["bg"]),
            "n_objects": len(comps),
            "n_objects_other_connectivity": len(color_objects(
                rows, color, 4 if conn == 8 else 8)),
            "n_figures": plan["n_objects"],
            "n_pieces": len(comps4),
            "odd_cells": len(cells),
            # is the answer the one biggest or the one smallest object, the way
            # a "recolour the odd size out" reader would guess?  The grouped
            # cell counts are what let the answer sit in the middle instead
            "odd_is_extreme_size": bool(
                _is_extreme(len(cells), [len(comp) for comp in comps])),
            "object_sizes": sorted(len(comp) for comp in comps),
            "piece_sizes": sorted(len(comp) for comp in comps4),
            "min_piece_cells": min((len(comp) for comp in comps4), default=0),
            # figures that read as one object under 8 and several under 4
            "corner_joined_figures": sum(
                1 for comp in color_objects(rows, color, 8)
                if sum(1 for piece in comps4 if set(piece) <= set(comp)) > 1),
            "differs_under_other_connectivity": bool(set(alt) != set(cells)),
            "other_criterion_has_answer": bool(swapped is not None),
            "differs_under_other_criterion": bool(
                swapped is None or set(swapped) != set(cells)),
            "differs_under_rotation_invariant": bool(
                loose is None or set(loose) != set(cells)),
            "n_distractors": scene["n_distractors"],
            "distractors_touching_target": scene["distractors_touching"],
            "distractor_colors": [int(c) for c in plan["distractor_colors"]],
            "object_kinds": scene["kinds"],
            "distractor_kinds": scene["distractor_kinds"],
            "background_margin": round(float(info["count"])
                                       / max(1, int(info["runner_up_count"])), 3),
            "occupancy": round(occ_cells / plan["area"], 4),
            "object_cells": occ_cells,
            "planned_density": round(plan["density"], 4),
            "palette_target": plan["palette_target"],
            "tie_rule": "exactly one object stands alone under the stated criterion",
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows, "output": out, "metadata": meta}
        validate_example(example, plan={"figures": scene["figures"],
                                        "odd_index": scene["odd_index"]})
        return example
    raise RuntimeError(f"task 97: no valid scene after {MAX_ATTEMPTS} attempts")


@register("97")
def generate_odd_one_out_recolor(rng: np.random.Generator) -> dict:
    """Task 97: recolour the color X object that is odd under the stated criterion."""
    return _example(rng)
