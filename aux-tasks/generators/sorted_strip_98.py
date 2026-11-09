"""Task 98 -- sort the objects by a stated key and lay them out in a strip.

The September 2026 rebuild replaced the inline version in
``generators/core2_extra/batch_object_a.py``; this is the follow-up pass that
closes what the first rebuild left open.

* **The scene is drawn before the rule, and the rule is a coin.**  The first
  rebuild drew the strip axis FIRST and then sized the objects along it, so the
  objects came out tall and narrow for a horizontal strip and short and wide for
  a vertical one: an axis classifier from the shapes alone scored 92%.  Here the
  whole picture -- canvas, start cell, gap, every object's bounding box, every
  body, the colours and the texture -- is drawn without an axis, and the object
  set is sized so that BOTH strips fit (the heights and the widths each sum,
  with the gaps, inside the canvas).  Only then is the axis flipped, so it is
  independent of the picture by construction, not by tuning.  The sort key is
  drawn the same way: nothing in the builder ever looks at it, so no scene
  statistic can predict it.  The tie structure is what used to give it away --
  the key's values were forced pairwise distinct while the other two measures
  were left to collide -- so the bounding boxes now carry pairwise distinct
  heights AND widths AND cell counts, and a planned tie (``TIE_P``) is a pair
  that shares all three.  Ties then happen at the same rate whichever key is
  stated (21%, against 22% before).  The cost is the object count: n distinct
  extents cost n(n+3)/2 cells on EACH axis once every body is two cells thick,
  so a strip carries at most six objects, and the palette -- one ground colour,
  one per object, at most three texture colours -- is capped at 1 + n + 3.  The
  palette target is drawn inside that ceiling and leans to it, which puts eight
  to ten colours on 5% of grids overall and 24% of those with 500 cells or more
  (3% and 7% before).  Raising ``MAX_OBJECTS`` past six would not help: the
  extents, not the constant, are what the canvas cannot pay for, and dropping
  the distinctness to fit more objects hands the tie structure back to the sort
  key.  Only a strip too tight for three two-cell-thick bodies falls back to
  one-cell bars (17% of objects, nearly all on canvases with a short side under
  twelve).
* **No layout is rejected for coinciding with the reading order.**  Rejecting
  it (what the first rebuild did) is what made "lay them out in reverse reading
  order" right on 34% of examples and on 82% of the two-object scenes: with two
  objects the sorted order is the reading order or its reverse, so forbidding
  one hands the solver the other.  Both orders are now left to chance, and the
  builder draws three or more objects wherever the canvas can hold them, which
  is what actually makes either reader rare: 25% forward and 21% backward,
  which is the rate chance gives for two- and three-object strips.  Neither can
  go lower without tilting the other, and the strip's capacity is what bounds
  the object count.
* **The output was underspecified.**  "Leave all other cells empty" was never
  defined: the prompt now carries the standard object rule, states that the
  output has the same dimensions as the input, says what fills the cells no
  object covers, and words the gap as the number of empty columns (rows)
  BETWEEN the two bounding boxes, which is the only reading that survives.
* **Coverage**: independent 5..30 sides, palettes of 2..10 colours (object
  colours plus up to three texture colours), bodies from the shared vocabulary
  through ``border_objects_90``'s clean-body builder scaled with the canvas, and
  sparse textures from ``common/backgrounds.py``, which the isolated-cell clause
  disposes of and the output drops.  The background is never named; the
  generator guarantees a plurality margin of 1.25 (``background_margin``).
* **The connectivity is drawn per example**, after the whole scene exists: a
  same-colour figure may be two clean pieces joined only at corners, which is
  one object under 8-connectivity and two under 4, and the two readings then
  sort and lay out different objects.  Every check that can reject a scene (both
  strips fit, the order is total) runs on both readings before the coin.
* **The generator proves the answer twice.**  ``validate_example`` rebuilds the
  output with the literal reader AND with a second, independent implementation
  (union-find labelling, an explicit insertion sort, cumulative offsets),
  compares both to the emitted output and to the construction plan, and raises
  on any disagreement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import (apply_background, infer_background, isolated_cell_mask,
                                pick_texture_colors)
from common.grid import NUM_COLORS, grid_to_list
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

# module import, attributes looked up lazily: a direct ``from ... import`` would
# create a registry-time import cycle (registry imports 90 before this module)
import generators.border_objects_90 as _b90


MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 40
TEXTURE_MIN_SIDE = 8
TEXTURE_P = 0.72
BG_MARGIN = 1.25
MIN_BODY = 2
MAX_OBJECTS = 6
TIE_P = 0.17                  # examples built with two objects of equal key
JOIN_P = 0.62                 # examples with a corner-joined same-colour figure

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))

_SORT_KEYS = ("cell_count", "bbox_height", "bbox_width")
_AXES = ("horizontal", "vertical")
# sparse kinds that paint one colour; multi_speckle is the two-or-three one
_SINGLE_KINDS = ("random_dots", "lattice", "dotted_stripes", "sparse_checker",
                 "dotted_border", "dotted_diagonal")
_PALETTE_SIZES = np.arange(2, 11)
_PALETTE_P = np.array([.07, .08, .09, .10, .11, .12, .13, .14, .16])


# ---------------------------------------------------------------------------
# Literal readers (pure Python, no numpy, no access to the plan)
# ---------------------------------------------------------------------------

def read_ground(grid) -> int:
    """The colour a reader told only "ignoring the background" settles on: the
    most frequent colour once every isolated single cell is set aside."""
    return _b90.read_ground(grid)


def objects(grid, connectivity: int, ground: Optional[int] = None) -> list[dict]:
    """The objects the prompt defines: same-colour groups of at least two cells
    under the stated connectivity, once the background is set aside."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    if ground is None:
        ground = read_ground(rows)
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] != ground}
    out: list[dict] = []
    while pending:
        start = min(pending)
        pending.remove(start)
        color = rows[start[0]][start[1]]
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending and rows[q[0]][q[1]] == color:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) < 2:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        out.append({"cells": sorted(comp), "color": int(color),
                    "box": (min(rs), min(cs), max(rs), max(cs))})
    return out


def key_value(obj: dict, sort_key: str) -> int:
    r0, c0, r1, c1 = obj["box"]
    if sort_key == "cell_count":
        return len(obj["cells"])
    if sort_key == "bbox_height":
        return r1 - r0 + 1
    if sort_key == "bbox_width":
        return c1 - c0 + 1
    raise ValueError(f"unknown sort key: {sort_key}")


def sorted_objects(objs: list[dict], sort_key: str) -> list[dict]:
    return sorted(objs, key=lambda obj: (key_value(obj, sort_key),
                                         obj["box"][0], obj["box"][1]))


def order_is_total(objs: list[dict], sort_key: str) -> bool:
    keys = [(key_value(obj, sort_key), obj["box"][0], obj["box"][1]) for obj in objs]
    return len(set(keys)) == len(keys)


def strip_extent(objs: list[dict], axis: str, gap: int) -> tuple[int, int]:
    """(length along the strip, extent across it) for the ordered objects."""
    along, across = 0, 0
    for index, obj in enumerate(objs):
        r0, c0, r1, c1 = obj["box"]
        height, width = r1 - r0 + 1, c1 - c0 + 1
        if axis == "horizontal":
            along += width + (gap if index else 0)
            across = max(across, height)
        else:
            along += height + (gap if index else 0)
            across = max(across, width)
    return along, across


def fits(objs: list[dict], h: int, w: int, axis: str, gap: int,
         start_r: int, start_c: int) -> bool:
    along, across = strip_extent(objs, axis, gap)
    if axis == "horizontal":
        return start_c + along <= w and start_r + across <= h
    return start_r + along <= h and start_c + across <= w


def layout(grid, objs: list[dict], ground: int, axis: str, gap: int,
           start_r: int, start_c: int) -> list[list[int]]:
    """Paint the ordered objects into a fresh canvas of the input's size."""
    h, w = len(grid), len(grid[0])
    out = [[int(ground)] * w for _ in range(h)]
    cur_r, cur_c = start_r, start_c
    for obj in objs:
        r0, c0, r1, c1 = obj["box"]
        for r, c in obj["cells"]:
            out[cur_r + r - r0][cur_c + c - c0] = obj["color"]
        if axis == "horizontal":
            cur_c += (c1 - c0 + 1) + gap
        else:
            cur_r += (r1 - r0 + 1) + gap
    return out


def solve(grid, connectivity: int, sort_key: str, axis: str, gap: int,
          start_r: int, start_c: int) -> Optional[list[list[int]]]:
    """The whole rule, read literally off the visible grid."""
    ground = read_ground(grid)
    objs = objects(grid, connectivity, ground)
    if not objs or not order_is_total(objs, sort_key):
        return None
    ordered = sorted_objects(objs, sort_key)
    h, w = len(grid), len(grid[0])
    if not fits(ordered, h, w, axis, gap, start_r, start_c):
        return None
    return layout(grid, ordered, ground, axis, gap, start_r, start_c)


def reading_order(objs: list[dict]) -> list[tuple[int, int]]:
    return sorted((obj["box"][0], obj["box"][1]) for obj in objs)


# ---------------------------------------------------------------------------
# A second, independent implementation of the whole rule.  Different labelling
# (union-find over a row-major scan), different ground tally, an explicit
# insertion sort with a written-out comparator and cumulative offsets instead of
# the running cursor.  The validator runs both and raises when they disagree.
# ---------------------------------------------------------------------------

def oracle_ground(grid) -> int:
    rows = [list(row) for row in grid]
    height, width = len(rows), len(rows[0])
    tally: dict[int, int] = {}
    for r in range(height):
        for c in range(width):
            color = rows[r][c]
            touching = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and rows[rr][cc] == color:
                        touching = True
                        break
                if touching:
                    break
            if touching:
                tally[color] = tally.get(color, 0) + 1
    best = None
    for color in sorted(tally):
        if best is None or tally[color] > tally[best]:
            best = color
    return int(best)


def oracle_objects(grid, connectivity: int, ground: int) -> list[dict]:
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
            if rows[r][c] == ground:
                continue
            parent[(r, c)] = (r, c)
            for dr, dc in back:
                q = (r + dr, c + dc)
                if q in parent and rows[q[0]][q[1]] == rows[r][c]:
                    a, b = find((r, c)), find(q)
                    if a != b:
                        parent[b] = a
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in list(parent):
        groups.setdefault(find(cell), []).append(cell)
    out = []
    for cells in groups.values():
        if len(cells) < 2:
            continue
        cells = sorted(cells)
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        bottom = max(r for r, _ in cells)
        right = max(c for _, c in cells)
        out.append({"cells": cells, "color": int(rows[cells[0][0]][cells[0][1]]),
                    "box": (top, left, bottom, right),
                    "height": bottom - top + 1, "width": right - left + 1})
    return out


def oracle_output(grid, connectivity: int, sort_key: str, axis: str, gap: int,
                  start_r: int, start_c: int) -> Optional[list[list[int]]]:
    ground = oracle_ground(grid)
    objs = oracle_objects(grid, connectivity, ground)
    if not objs:
        return None

    def rank(obj):
        if sort_key == "cell_count":
            measure = len(obj["cells"])
        elif sort_key == "bbox_height":
            measure = obj["height"]
        elif sort_key == "bbox_width":
            measure = obj["width"]
        else:
            raise ValueError(f"unknown sort key: {sort_key}")
        return (measure, obj["box"][0], obj["box"][1])

    if len({rank(obj) for obj in objs}) != len(objs):
        return None
    ordered: list[dict] = []
    for obj in objs:                       # insertion sort, written out
        place = len(ordered)
        for index, seen in enumerate(ordered):
            if rank(obj) < rank(seen):
                place = index
                break
        ordered.insert(place, obj)

    height, width = len(grid), len(grid[0])
    offset = start_c if axis == "horizontal" else start_r
    placements = []
    for obj in ordered:
        placements.append(offset)
        offset += (obj["width"] if axis == "horizontal" else obj["height"]) + gap
    out = [[int(ground)] * width for _ in range(height)]
    for obj, where in zip(ordered, placements):
        top, left = obj["box"][0], obj["box"][1]
        for r, c in obj["cells"]:
            if axis == "horizontal":
                rr, cc = start_r + (r - top), where + (c - left)
            else:
                rr, cc = where + (r - top), start_c + (c - left)
            if not (0 <= rr < height and 0 <= cc < width):
                return None
            out[rr][cc] = obj["color"]
    return out


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------

def _tight(mask: np.ndarray) -> np.ndarray:
    """The mask cropped to its own bounding box.

    The library hands back an array of the box it was asked for, which may carry
    an empty edge row or column; the object's VISIBLE bounding box is the tight
    one, and this task sorts by that box, so every mask is cropped before its
    extents are read.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(cols):
        return mask
    return np.ascontiguousarray(mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])


def _body(rng: np.random.Generator, bh: int, bw: int, *, min_cells: int = MIN_BODY,
          exact_cells: Optional[int] = None, exact_box: bool = True,
          tries: int = 1, fill: float = 0.45,
          avoid: tuple = ()) -> Optional[tuple[np.ndarray, str]]:
    """A clean body whose box is exactly ``bh x bw``.

    ``fill`` is a floor on how much of the box the body covers: without it the
    library's bar family answers a 4x10 box with a one-cell-wide stick, and a
    strip of sticks is what the review called a row of bars rather than a scene.
    ``exact_cells`` and ``avoid`` are rejections (retry another body), never a
    canned shape: the first is how two objects come to share a cell count, the
    second is how the others come to differ in one.
    """
    bh, bw = max(1, int(bh)), max(1, int(bw))
    floor = max(2, int(min_cells), int(round(fill * bh * bw)))
    if exact_cells is not None:
        floor = min(floor, int(exact_cells))
    if bh * bw < floor:
        return None
    for _ in range(max(1, tries)):
        built = _b90._body(rng, bh, bw, min_cells=floor)
        if built is None:
            continue
        mask, fam = built
        mask = _tight(mask)
        if exact_box and (mask.shape[0] != bh or mask.shape[1] != bw):
            continue
        if exact_cells is not None and int(mask.sum()) != int(exact_cells):
            continue
        if avoid and int(mask.sum()) in avoid:
            continue
        return np.ascontiguousarray(mask), fam
    return None


def _with_corner(mask: np.ndarray, corner: tuple[int, int]) -> Optional[np.ndarray]:
    """``mask``, flipped if need be, with the given corner cell filled.

    The flips keep the bounding box, the cell count and the family; they only
    move which corner of the body is solid, which is what the corner join needs.
    """
    row, col = corner
    for flip_rows in (False, True):
        for flip_cols in (False, True):
            cand = mask
            if flip_rows:
                cand = cand[::-1]
            if flip_cols:
                cand = cand[:, ::-1]
            if bool(cand[row, col]):
                return np.ascontiguousarray(cand)
    return None


def _joined_body(rng: np.random.Generator, bh: int, bw: int,
                 avoid_h: tuple = (), avoid_w: tuple = (),
                 avoid_cells: tuple = (),
                 tries: int = 24) -> Optional[tuple[np.ndarray, str]]:
    """Two clean pieces joined only at a corner, in a box of exactly bh x bw.

    One object under 8-connectivity and two under 4, so the two readings sort
    and lay out different objects.  The box has to come out exactly as planned,
    or the figure would carry an extent the plan did not draw; and under
    4-connectivity the PIECES are two more objects, so their extents and their
    cell counts have to miss the siblings' as well -- otherwise that reading
    carries a tie no sort key asked for, which is exactly what would tell a
    solver which key the prompt states.
    """
    if bh < 2 or bw < 2 or bh + bw < 6:
        return None
    for _ in range(tries):
        h1 = int(rng.integers(1, bh))
        w1 = int(rng.integers(1, bw))
        h2, w2 = bh - h1, bw - w1
        if h1 * w1 < MIN_BODY or h2 * w2 < MIN_BODY:
            continue
        if h1 == h2 or w1 == w2 or h1 in avoid_h or h2 in avoid_h \
                or w1 in avoid_w or w2 in avoid_w:
            continue
        first = _body(rng, h1, w1, tries=8, avoid=avoid_cells)
        if first is None:
            continue
        second = _body(rng, h2, w2, tries=8,
                       avoid=tuple(avoid_cells) + (int(first[0].sum()),))
        if second is None:
            continue
        if int(first[0].sum()) + int(second[0].sum()) in avoid_cells:
            continue
        if first[0].shape == second[0].shape and np.array_equal(first[0], second[0]):
            # under 4-connectivity the pieces are two objects of the SAME
            # colour: identical ones would swap invisibly and the tie-break
            # clause would decide nothing on that half of the coin
            continue
        # The two pieces are laid out diagonally, so the union's box is exactly
        # the two boxes stacked: one piece needs a filled cell in the corner
        # that meets the other, which a flip supplies when the drawn body is
        # missing it.  Their boxes are disjoint on both axes, so the union is
        # one 8-connected object and exactly two 4-connected ones.
        down = rng.random() < 0.5
        top = _with_corner(first[0], (h1 - 1, w1 - 1) if down else (h1 - 1, 0))
        bottom = _with_corner(second[0], (0, 0) if down else (0, w2 - 1))
        if top is None or bottom is None:
            continue
        fig = np.zeros((bh, bw), dtype=bool)
        if down:
            fig[:h1, :w1] = top
            fig[h1:, w1:] = bottom
        else:
            fig[:h1, w2:] = top
            fig[h1:, :w2] = bottom
        return np.ascontiguousarray(fig), first[1] + "+" + second[1]
    return None


# ---------------------------------------------------------------------------
# The plan: the whole picture, drawn without an axis and without a sort key
# ---------------------------------------------------------------------------

def _distinct_extents(rng: np.random.Generator, n: int, budget: int,
                      floor: int = 1) -> Optional[list[int]]:
    """``n`` pairwise distinct extents of at least ``floor`` cells summing to at
    most ``budget``, spread over it so the objects scale with the canvas.

    Distinctness is what keeps the tie rate off the sort key: with heights,
    widths and cell counts all pairwise distinct, a tie happens only where the
    plan asked for one, whichever measure the prompt ends up naming.  ``floor``
    is 2 wherever the canvas can pay for it, so a strip is a row of bodies and
    not a row of one-cell bars.
    """
    if n * floor + n * (n - 1) // 2 > budget:
        return None
    # start as large as the budget affords, so the objects scale with the
    # canvas instead of hugging the floor
    low = max(floor, (budget - n * (n - 1) // 2) // n)
    values = [low + i for i in range(n)]
    for _ in range(80):
        slack = budget - sum(values)
        if slack <= 0:
            break
        # adding one to a suffix keeps the extents pairwise distinct
        index = 0 if rng.random() < 0.4 else int(rng.integers(0, n))
        if n - index <= slack:
            for j in range(index, n):
                values[j] += 1
    return values


def sample_plan(rng: np.random.Generator) -> Optional[dict]:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    start_r = int(rng.choice([0, 1, 2], p=[0.22, 0.56, 0.22]))
    start_c = int(rng.choice([0, 1, 2], p=[0.22, 0.56, 0.22]))
    start_r = min(start_r, max(0, h - 4))
    start_c = min(start_c, max(0, w - 4))
    budget_h, budget_w = h - start_r, w - start_c
    if budget_h < 3 or budget_w < 3:
        return None
    room = min(budget_h, budget_w)

    # A corner-joined figure reads as two objects under 4-connectivity, so that
    # strip carries one more gap than the 8-connected one: the budget pays for
    # it up front rather than rejecting the scene afterwards.
    join = bool(rng.random() < JOIN_P)

    # How many objects fit with PAIRWISE DISTINCT extents on both axes: n
    # distinct extents cost at least n(n+1)/2, plus the gaps and the reserve.
    # The gap is drawn among the values that leave room for three objects,
    # because a strip of two is the one case where the sorted order is the
    # reading order or exactly its reverse.
    def _capacity(gap_value: int, reserve: int, floor: int) -> int:
        best = 0
        for cand in range(2, MAX_OBJECTS + 1):
            cost = cand * floor + cand * (cand - 1) // 2 + (cand - 1) * gap_value
            if cost + reserve <= room:
                best = cand
            else:
                break
        return best

    # Bodies are at least two cells thick on both axes wherever the canvas can
    # pay for the extra n cells per axis; only a strip too tight for three such
    # objects falls back to one-cell bars.
    min_extent = 2
    caps = {g: _capacity(g, g if join else 0, min_extent) for g in (1, 2, 3)}
    if max(caps.values()) < 3:
        min_extent = 1
        caps = {g: _capacity(g, g if join else 0, min_extent) for g in (1, 2, 3)}
    if max(caps.values()) < 2 and join:
        join = False
        caps = {g: _capacity(g, 0, min_extent) for g in (1, 2, 3)}
    best = max(caps.values())
    if best < 2:
        return None
    roomy = [g for g, c in caps.items() if c >= min(3, best)]
    gap = int(roomy[int(rng.integers(0, len(roomy)))])
    reserve = gap if join else 0
    cap = caps[gap]
    roll = rng.random()
    n_objects = cap if roll < 0.80 else (cap - 1 if roll < 0.95 else cap - 2)
    # Three objects wherever the canvas can hold them: with two, the sorted
    # order is the reading order or exactly its reverse, so one of those two
    # readers is right more than half the time whatever the builder does.
    n_objects = int(np.clip(n_objects, 3 if cap >= 3 else 2, cap))

    spare_h = budget_h - reserve - gap * (n_objects - 1)
    spare_w = budget_w - reserve - gap * (n_objects - 1)
    fill_h = float(rng.uniform(0.85, 1.0))
    fill_w = float(rng.uniform(0.85, 1.0))
    floor = n_objects * min_extent + n_objects * (n_objects - 1) // 2
    heights = _distinct_extents(rng, n_objects,
                                int(np.clip(round(spare_h * fill_h), floor, spare_h)),
                                floor=min_extent)
    widths = _distinct_extents(rng, n_objects,
                               int(np.clip(round(spare_w * fill_w), floor, spare_w)),
                               floor=min_extent)
    if heights is None or widths is None:
        return None
    order = rng.permutation(n_objects)
    # a tenth of the scenes pair the extents co-monotonically (the tallest body
    # is also the widest); the rest pair them independently, which is what makes
    # the three sort keys order the objects differently
    other = order if rng.random() < 0.10 else rng.permutation(n_objects)
    heights = [int(heights[int(i)]) for i in order]
    widths = [int(widths[int(i)]) for i in other]
    # a one-cell box is not an object; swap the offender into a taller slot
    for index in range(n_objects):
        if heights[index] * widths[index] >= MIN_BODY:
            continue
        other = max(range(n_objects), key=lambda i: widths[i])
        if other == index:
            return None
        widths[index], widths[other] = widths[other], widths[index]
        if heights[index] * widths[index] < MIN_BODY:
            return None

    join_at = -1
    if join:
        seats = [i for i in range(n_objects)
                 if heights[i] >= 2 and widths[i] >= 2 and heights[i] + widths[i] >= 6]
        if seats:
            join_at = int(seats[int(rng.integers(0, len(seats)))])
        else:
            join = False

    # The planned tie: one pair shares its height, its width AND its cell count,
    # so a tie is a tie whichever key the prompt ends up naming.  Copying the
    # smaller extents can only free budget, never overrun it.
    tie_at = None
    free = [i for i in range(n_objects) if i != join_at]
    if len(free) >= 2 and rng.random() < TIE_P:
        pair = [int(free[int(v)]) for v in rng.choice(len(free), size=2, replace=False)]
        first, second = pair
        heights[first] = heights[second] = min(heights[first], heights[second])
        widths[first] = widths[second] = min(widths[first], widths[second])
        if heights[first] * widths[first] >= MIN_BODY:
            tie_at = (first, second)

    # The palette a scene can carry is one ground colour, one colour per object
    # and at most three texture colours, and the object count is what the strip
    # capacity allows, so the target is drawn inside that ceiling instead of
    # being asked for and quietly missed.
    bg = int(rng.integers(0, NUM_COLORS))
    pool = [c for c in range(NUM_COLORS) if c != bg]
    texture = bool(min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P)
    ceiling = min(10, 1 + n_objects + (3 if texture else 0))
    sizes = _PALETTE_SIZES
    weights = _PALETTE_P
    if ceiling < 2:
        return None
    # a third of the scenes take the ceiling outright: the strip's capacity is
    # what caps the palette, so leaning to the top of what this scene can carry
    # is the only way the eight-to-ten band is reachable at all
    palette = ceiling if rng.random() < 0.35 else min(
        int(rng.choice(sizes, p=weights / weights.sum())), ceiling)
    n_texture_colors = int(np.clip(palette - 1 - n_objects, 0, 3)) if texture else 0
    n_object_colors = int(np.clip(palette - 1 - n_texture_colors, 1, n_objects))
    if tie_at is not None:
        # a tie between two objects of the same colour AND shape is a dead
        # clause: swapping them leaves the picture alone
        n_object_colors = max(n_object_colors, 2)
    chosen = [int(c) for c in rng.choice(pool, size=n_object_colors, replace=False)]
    colors = list(chosen)
    while len(colors) < n_objects:
        colors.append(int(chosen[int(rng.integers(0, len(chosen)))]))
    colors = [int(colors[int(i)]) for i in rng.permutation(len(colors))]
    if tie_at is not None and n_object_colors >= 2 and colors[tie_at[0]] == colors[tie_at[1]]:
        other = [c for c in chosen if c != colors[tie_at[0]]]
        colors[tie_at[1]] = int(other[int(rng.integers(0, len(other)))])

    return {
        "h": h, "w": w, "area": area, "gap": gap,
        "start_r": start_r, "start_c": start_c,
        "budget_h": budget_h, "budget_w": budget_w,
        "n_objects": n_objects, "heights": heights, "widths": widths,
        "tie_at": tie_at, "join_at": join_at,
        "bg": bg, "colors": colors, "palette_target": palette,
        "plan_texture": texture, "n_texture_colors": n_texture_colors,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _masks(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    """One mask per planned object, each in its planned bounding box.

    The boxes are pairwise distinct on both axes by construction; the cell
    counts are made distinct here, with exact-size rebuilds, because two clean
    bodies of similar area collide often enough that the tie clause would
    otherwise decide a third of the examples -- and an unplanned tie can be a
    dead one (two objects of the same colour and shape swap invisibly).
    """
    n = plan["n_objects"]
    heights, widths = plan["heights"], plan["widths"]
    slot = plan["join_at"]
    tie = plan["tie_at"]
    masks: list[Optional[np.ndarray]] = [None] * n
    kinds: list[Optional[str]] = [None] * n
    for index in range(n):
        if index == slot:
            continue
        built = _body(rng, heights[index], widths[index], tries=24)
        if built is None:
            return None
        masks[index], kinds[index] = built

    if tie is not None:
        first, second = tie
        want = int(masks[first].sum())
        twin = _body(rng, heights[second], widths[second], exact_cells=want, tries=48)
        if twin is None:
            # a mirrored copy always shares every key; the colours differ, so
            # the tie-break still decides the output
            copy = masks[first][:, ::-1] if rng.random() < 0.5 else masks[first][::-1]
            twin = (np.ascontiguousarray(copy), "twin")
        masks[second], kinds[second] = twin

    used: set = set()
    if tie is not None:
        used.add(int(masks[tie[0]].sum()))
    for index in range(n):
        if index == slot or (tie is not None and index in tie):
            continue
        if int(masks[index].sum()) in used:
            fresh = _body(rng, heights[index], widths[index], tries=48,
                          avoid=tuple(sorted(used)))
            if fresh is None:
                return None
            masks[index], kinds[index] = fresh
        used.add(int(masks[index].sum()))

    joined = False
    if slot >= 0:
        built = _joined_body(rng, heights[slot], widths[slot],
                             avoid_h=tuple(heights), avoid_w=tuple(widths),
                             avoid_cells=tuple(sorted(used)))
        if built is None:
            # the pieces could not miss every sibling's extent and cell count;
            # a join that repeats one only adds a tie, which the tie-break
            # clause already covers
            built = _joined_body(rng, heights[slot], widths[slot])
        joined = built is not None
        if built is None:
            # a corner join the box cannot carry is answered with a plain body,
            # not with a lost scene: the connectivity coin is fair either way
            built = _body(rng, heights[slot], widths[slot], tries=36,
                          avoid=tuple(sorted(used)))
            if built is None:
                return None
        masks[slot], kinds[slot] = built
    return {"masks": masks, "kinds": kinds, "join_at": slot if joined else -1}


def build_scene(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    built = _masks(rng, plan)
    if built is None:
        return None
    h, w, bg = plan["h"], plan["w"], plan["bg"]
    grid = np.full((h, w), bg, dtype=np.int16)
    occ = np.zeros((h, w), dtype=bool)
    masks = built["masks"]
    boxes: list[tuple[int, int]] = []
    for index, mask in enumerate(masks):
        if mask.shape[0] > h or mask.shape[1] > w:
            return None
        ok = placement_map(mask, ~halo(occ))
        if ok is None:
            return None
        spots = np.argwhere(ok)
        if not len(spots):
            return None
        r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
        cells = np.zeros((h, w), dtype=bool)
        cells[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
        grid[cells] = int(plan["colors"][index])
        occ |= cells
        boxes.append((r, c))
    return {"grid": grid, "occ": occ, "kinds": built["kinds"],
            "masks": masks, "boxes": boxes, "join_at": built["join_at"]}


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(rng: np.random.Generator, grid: np.ndarray, bg: int, occ: np.ndarray,
             *, exclude, want: int) -> tuple[np.ndarray, dict]:
    """A sparse library texture of exactly ``want`` colours, clear of every
    object by one cell.

    Sparse only: its cells are isolated single cells, which the object rule
    disposes of, and the plain background the output fills with stays readable.
    The colour count is chosen here rather than left to the library, because it
    is what carries the palette to eight, nine and ten on a roomy canvas.
    """
    if want <= 0:
        return grid, dict(_NO_TEXTURE)
    present = {int(v) for v in np.unique(grid)}
    blocked = sorted(present | {int(bg)} | {int(c) for c in exclude})
    colors = pick_texture_colors(rng, want, blocked)
    if len(colors) < want:
        want = len(colors)
        colors = colors[:want]
    if want <= 0:
        return grid, dict(_NO_TEXTURE)
    if want >= 2:
        kind = "multi_speckle"
    else:
        kind = _SINGLE_KINDS[int(rng.integers(0, len(_SINGLE_KINDS)))]
    painted = apply_background(rng, grid, bg, halo(occ), kind=kind,
                               colors=[int(c) for c in colors])
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
    """Rebuild the output twice from the emitted grid -- once with the literal
    reader, once with the independent oracle -- and raise on any disagreement,
    with the construction plan as a third witness when the builder passes it.
    """
    grid, meta = example["input"], example["metadata"]
    conn = int(meta["connectivity"])
    key = meta["sort_key"]
    axis = meta["strip_axis"]
    gap = int(meta["gap"])
    start_r, start_c = int(meta["strip_start_row"]), int(meta["strip_start_col"])

    ground = read_ground(grid)
    if ground != int(meta["bg_color"]):
        raise ValueError("task 98: the visible ground disagrees with the plan")
    if oracle_ground(grid) != ground:
        raise ValueError("task 98: the two ground readings disagree")
    objs = objects(grid, conn, ground)
    if len(objs) != meta["n_objects"]:
        raise ValueError("task 98: the object count disagrees with the metadata")
    if len(objs) < 2:
        raise ValueError("task 98: fewer than two objects")
    if not order_is_total(objs, key):
        raise ValueError("task 98: the stated order does not decide between two objects")
    ordered = sorted_objects(objs, key)
    h, w = len(grid), len(grid[0])
    if not fits(ordered, h, w, axis, gap, start_r, start_c):
        raise ValueError("task 98: an object does not fit in the strip")
    out = layout(grid, ordered, ground, axis, gap, start_r, start_c)
    if out != example["output"]:
        raise ValueError("task 98: the output disagrees with the literal reading")

    twin = oracle_output(grid, conn, key, axis, gap, start_r, start_c)
    if twin is None or twin != example["output"]:
        raise ValueError("task 98: the independent oracle disagrees with the output")
    for other in (4, 8):
        for other_key in _SORT_KEYS:
            for other_axis in _AXES:
                if (oracle_output(grid, other, other_key, other_axis, gap, start_r, start_c)
                        != solve(grid, other, other_key, other_axis, gap, start_r, start_c)):
                    raise ValueError(
                        f"task 98: the two readers disagree ({other}-connected, "
                        f"{other_key}, {other_axis})")

    if plan is not None:
        # the picture shows the boxes the builder drew, and every object the
        # 8-connected reading finds is one of them
        drawn = sorted((int(bh), int(bw)) for bh, bw in plan["boxes"])
        seen = sorted((obj["box"][2] - obj["box"][0] + 1, obj["box"][3] - obj["box"][1] + 1)
                      for obj in objects(grid, 8, ground))
        if drawn != seen:
            raise ValueError("task 98: the drawn bounding boxes are not the visible ones")
        if sorted(plan["cells"]) != sorted(len(obj["cells"]) for obj in
                                           objects(grid, 8, ground)):
            raise ValueError("task 98: the drawn cell counts are not the visible ones")

    values = [key_value(obj, key) for obj in objs]
    if (len(set(values)) != len(values)) != meta["has_ties"]:
        raise ValueError("task 98: the tie flag is wrong")
    order = [(obj["box"][0], obj["box"][1]) for obj in ordered]
    if (order == reading_order(objs)) != meta["order_equals_reading_order"]:
        raise ValueError("task 98: the reading-order flag is wrong")

    other = 4 if conn == 8 else 8
    alt = solve(grid, other, key, axis, gap, start_r, start_c)
    if alt is None:
        raise ValueError("task 98: the other connectivity has no answer")
    if (alt != out) != meta["differs_under_other_connectivity"]:
        raise ValueError("task 98: the connectivity flag is wrong")

    changed = 0
    for other_key in _SORT_KEYS:
        if other_key == key:
            continue
        swapped = solve(grid, conn, other_key, axis, gap, start_r, start_c)
        if swapped is None or swapped != out:
            changed += 1
    if (changed > 0) != meta["differs_under_other_key"]:
        raise ValueError("task 98: the sort-key flag is wrong")

    flipped = solve(grid, conn, key, "vertical" if axis == "horizontal" else "horizontal",
                    gap, start_r, start_c)
    if flipped is None:
        raise ValueError("task 98: the other strip axis does not fit")
    if (flipped != out) != meta["differs_under_other_axis"]:
        raise ValueError("task 98: the axis flag is wrong")

    arr = np.asarray(grid)
    info = infer_background(arr, margin=BG_MARGIN)
    if not info["identifiable"] or int(info["color"]) != ground:
        raise ValueError("task 98: the background is not identifiable")
    if meta["texture_colors"]:
        tex = np.isin(arr, meta["texture_colors"])
        if not bool(isolated_cell_mask(arr)[tex].all()):
            raise ValueError("task 98: a texture cell is not isolated")
        if ground in meta["texture_colors"]:
            raise ValueError("task 98: a texture colour is the ground colour")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = sample_plan(rng)
        if plan is None:
            continue
        scene = None
        for _ in range(3):
            candidate = build_scene(rng, plan)
            if candidate is not None:
                scene = candidate
                break
        if scene is None:
            continue
        display, tex = _texture(rng, scene["grid"], plan["bg"], scene["occ"],
                                exclude=[int(c) for c in plan["colors"]],
                                want=plan["n_texture_colors"] if plan["plan_texture"] else 0)
        rows = grid_to_list(display.astype(np.int16))
        gap = plan["gap"]
        start_r, start_c = plan["start_r"], plan["start_c"]

        if read_ground(rows) != plan["bg"]:
            continue
        info = infer_background(display, margin=BG_MARGIN)
        if not info["identifiable"] or int(info["color"]) != plan["bg"]:
            continue

        # Nothing above this line has looked at the axis, the sort key or the
        # connectivity.  All four (axis, connectivity) strips must fit and all
        # three keys must order the objects before any of the three is drawn,
        # so each of them is independent of the picture.
        objs4 = objects(rows, 4, plan["bg"])
        objs8 = objects(rows, 8, plan["bg"])
        if len(objs8) != plan["n_objects"] or len(objs4) != plan["n_objects"] + (
                1 if scene["join_at"] >= 0 else 0):
            continue
        ok = True
        for objs in (objs4, objs8):
            if len(objs) < 2:
                ok = False
                break
            for key in _SORT_KEYS:
                if not order_is_total(objs, key):
                    ok = False
                    break
                ordered = sorted_objects(objs, key)
                for axis in _AXES:
                    if not fits(ordered, plan["h"], plan["w"], axis, gap, start_r, start_c):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        conn = 4 if rng.random() < 0.5 else 8
        axis = _AXES[int(rng.integers(0, 2))]
        key = _SORT_KEYS[int(rng.integers(0, 3))]
        other_axis = "vertical" if axis == "horizontal" else "horizontal"
        out = solve(rows, conn, key, axis, gap, start_r, start_c)
        alt = solve(rows, 8 if conn == 4 else 4, key, axis, gap, start_r, start_c)
        flipped = solve(rows, conn, key, other_axis, gap, start_r, start_c)
        if out is None or alt is None or flipped is None:
            continue
        objs = objs4 if conn == 4 else objs8
        ordered = sorted_objects(objs, key)
        values = [key_value(obj, key) for obj in objs]
        changed_keys = 0
        for other_key in _SORT_KEYS:
            if other_key == key:
                continue
            swapped = solve(rows, conn, other_key, axis, gap, start_r, start_c)
            if swapped is None or swapped != out:
                changed_keys += 1
        # the reading a solver who ignored "at least two cells" would take
        singleton_cells = sum(1 for r, row in enumerate(rows) for c, value in enumerate(row)
                              if value != plan["bg"]
                              and all(not (0 <= r + dr < plan["h"] and 0 <= c + dc < plan["w"]
                                           and rows[r + dr][c + dc] == value)
                                      for dr, dc in _ORTH + _DIAG))
        along, across = strip_extent(ordered, axis, gap)
        present = sorted({int(v) for v in np.unique(display)})
        occ_cells = int(scene["occ"].sum())
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"], "n_colors": len(present),
            "connectivity": conn,
            "sort_key": key, "sort_order": "ascending",
            "strip_axis": axis, "gap": gap,
            "strip_start_row": start_r, "strip_start_col": start_c,
            "n_objects": len(objs),
            "n_objects_other_connectivity": len(objs8 if conn == 4 else objs4),
            "object_sizes": sorted(len(obj["cells"]) for obj in objs),
            "object_colors": sorted({obj["color"] for obj in objs}),
            "object_kinds": scene["kinds"],
            "object_boxes": [[int(obj["box"][2] - obj["box"][0] + 1),
                              int(obj["box"][3] - obj["box"][1] + 1)] for obj in objs],
            "sort_values": [key_value(obj, key) for obj in ordered],
            "has_ties": bool(len(set(values)) != len(values)),
            "n_tied_pairs": sum(values.count(v) - 1 for v in set(values)),
            "planned_tie": plan["tie_at"] is not None,
            "order_equals_reading_order": bool(
                [(obj["box"][0], obj["box"][1]) for obj in ordered] == reading_order(objs)),
            "order_equals_reverse_reading_order": bool(
                [(obj["box"][0], obj["box"][1]) for obj in ordered]
                == list(reversed(reading_order(objs)))),
            "differs_under_other_connectivity": bool(alt != out),
            "differs_under_other_key": bool(changed_keys > 0),
            "differs_under_other_axis": bool(flipped != out),
            "n_keys_that_change_the_output": changed_keys,
            "corner_joined_figures": max(0, len(objs4) - len(objs8)),
            "singleton_cells": singleton_cells,
            "strip_length": along, "strip_across": across,
            "strip_capacity": (plan["w"] - start_c) if axis == "horizontal"
                              else (plan["h"] - start_r),
            "background_color": int(plan["bg"]),
            "bg_color": int(plan["bg"]),
            "background_margin": round(float(info["count"])
                                       / max(1, int(info["runner_up_count"])), 3),
            "occupancy": round(occ_cells / plan["area"], 4),
            "object_cells": occ_cells,
            "palette_target": plan["palette_target"],
            "placement_rule": "each object keeps its shape and color; its bounding box is "
                              "aligned to the current strip position",
            "tie_rule": "equal sort values are ordered by top bounding-box row, then left "
                        "bounding-box column",
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows, "output": out, "metadata": meta}
        validate_example(example, plan={
            "boxes": [(int(mask.shape[0]), int(mask.shape[1])) for mask in scene["masks"]],
            "cells": [int(mask.sum()) for mask in scene["masks"]]})
        return example
    raise RuntimeError(f"task 98: no valid scene after {MAX_ATTEMPTS} attempts")


@register("98")
def generate_sort_objects_into_strip(rng: np.random.Generator) -> dict:
    """Task 98: sort the objects by the stated key and lay them out in a strip."""
    return _example(rng)
