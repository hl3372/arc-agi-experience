"""Tasks 93 and 94 -- the two colour-defined object tasks.

* **93** tight-crops the unique color X object: the output is the smallest
  rectangle that contains the object, copied cell for cell out of the input,
  non-object cells included.
* **94** lists the tight bounding box of every color X object, ordered by top
  row then left column, with the connectivity drawn per example.

Both rules are defined by COLOUR, not by "everything that is not the
background": an object is a connected group of color X cells and nothing else.
That is why neither prompt names or defines a background colour (93 never
mentions one at all; 94 carries only the standing "Ignore the background,
whether it is a solid color or a pattern." sentence), and why textures are free
to touch objects here -- nothing is "ignored" by a structural rule that a
texture could fool.

The September 2026 rebuild replaces the inline versions in
``generators/core2_simple.py``.  What it changes:

* **Sizes**: independent 5..30 sides, flat, no ``square_prob`` and no tilt
  toward small grids (the old builders drew 9..20 / 10..19 with a 4.8x lean to
  the small end and forced a square on a quarter of the grids).  Bodies scale
  with the canvas from a density drawn per example, so occupancy stays flat
  across the area bands instead of falling from 16% to 9%.
* **Bodies** come from the shared vocabulary through ``border_objects_90``'s
  clean-body builder (rectangles, bars, frames, thick letters, staircases,
  blobs assembled from rectangles): no random walks, no carving.  An
  8-connected target is a group of clean pieces joined only at corners, not a
  random diagonal walk.
* **Backgrounds** come from ``common/backgrounds.py`` (sparse and solid kinds)
  instead of a private texture painter.  A texture colour is never the target
  colour -- a texture cell of color X would create a second object out of thin
  air -- and the background is never named.
* **93's crop is contested**: other-colour objects sit INSIDE the target's
  bounding box on about a third of the grids (bodies in the concavity of an L,
  U or staircase target, a body inside a frame target, a piece in the gap of a
  corner-joined target), so "copying every input cell in that rectangle" is
  load-bearing instead of decorative.  The old builder placed every object with
  a two-cell moat, so no other object ever reached the box (0 of 5000 grids).
* **94's prompt no longer lies**: the old sentence "cells of any other color,
  including isolated single cells of patterned background detail, are not
  objects" described the background against policy AND was false (only 74% of
  the texture components were single cells).  The rule is now the colour
  sentence plus the standing background sentence, and it states that a lone
  color X cell IS an object.
* **Validators** recompute the answer from the visible grid with a literal
  pure-Python reader and raise on any disagreement (93 used to emit the planned
  crop with no check at all; 94's loop ``continue``d instead of raising).

Load-bearing structure of the connectivity:

* On 94 the whole scene, including the corner-joined target groups, is drawn
  BEFORE the connectivity, which is then a fair coin: no input feature can
  predict the variant, and the two readings disagree on almost every grid (a
  corner-joined group is one box under 8 and one box per piece under 4).
* On 93 the direction is one-way: a target drawn as one 4-connected body reads
  the same under both rules, while a corner-joined target has no unique object
  under 4-connectivity at all.  So the scene is drawn first and the
  connectivity is drawn among the readings the drawing admits -- 8 for a
  corner-joined target, a coin for a solid one.  A corner-joined target is
  exactly the set of grids on which the other reading has no unique object; the crop itself is the box of every colour-X cell under either reading.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import apply_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list
import generators.border_objects_90 as _b90  # module import: the helpers are looked up lazily to avoid a registry-time import cycle


def _body(*args, **kwargs):
    return _b90._body(*args, **kwargs)


def _build_family(*args, **kwargs):
    return _b90._build_family(*args, **kwargs)


def _capacity(*args, **kwargs):
    return _b90._capacity(*args, **kwargs)


def _corner_join(*args, **kwargs):
    return _b90._corner_join(*args, **kwargs)


def _to_mask(*args, **kwargs):
    return _b90._to_mask(*args, **kwargs)
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 40
TEXTURE_MIN_SIDE = 8          # the library's floor for sparse / solid kinds
TEXTURE_P = 0.34              # attempt rate on eligible grids
SOLID_SHARE = 0.40            # of the attempted textures, the solid-group share
MAX_BOXES = 12                # 94: the answer cap, footprint-derived below
TARGET_MAX_CELLS = 120        # 93: a target body never swallows a large canvas

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# families whose bounding box keeps free cells another object can sit in
_HOLLOW_FAMILIES = ("frame", "U", "L", "H", "T", "plus", "staircase")

# near-flat over 2..10 with a mild lean to the top (the palette is truncated by
# the number of objects a scene can hold, so the tail needs the extra weight)
_PALETTE_SIZES = np.arange(2, 11)
_PALETTE_P = np.array([.09, .10, .11, .11, .11, .11, .12, .12, .13])


# ---------------------------------------------------------------------------
# Literal readers (pure Python, no plan, no numpy): the prompts' rule verbatim
# ---------------------------------------------------------------------------

def color_components(grid, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    """Every maximal group of color ``color`` cells under the given connectivity.

    This is the prompt's object definition read literally: cells of any other
    colour are not part of an object, and a lone color X cell is a group of one.
    """
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
        out.append(comp)
    return out


def bbox(cells) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def solve_boxes(grid, color: int, connectivity: int) -> list[tuple[int, int, int, int]]:
    """Task 94's answer: one box per object, by top row then left column."""
    boxes = [bbox(comp) for comp in color_components(grid, color, connectivity)]
    return sorted(boxes, key=lambda b: (b[0], b[1]))


def solve_crop(grid, color: int, connectivity: int) -> list[list[int]]:
    """Task 93's answer: the input cells inside the unique object's box."""
    comps = color_components(grid, color, connectivity)
    if len(comps) != 1:
        raise ValueError(f"task 93: {len(comps)} objects of the target colour, not one")
    r0, c0, r1, c1 = bbox(comps[0])
    return [list(row[c0:c1 + 1]) for row in grid[r0:r1 + 1]]


def format_boxes(boxes) -> str:
    return "[" + ",".join(f"({r0},{c0},{r1},{c1})" for r0, c0, r1, c1 in boxes) + "]"


def _ordering_is_total(boxes) -> bool:
    """No two boxes may share a top row AND a left column: the prompt's order
    would not decide between them."""
    keys = [(b[0], b[1]) for b in boxes]
    return len(set(keys)) == len(keys)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _hollow_body(rng: np.random.Generator, bh: int, bw: int,
                 floor: int) -> Optional[tuple[np.ndarray, str]]:
    """A clean body with free cells left inside its own bounding box."""
    if min(bh, bw) < 2:
        return None
    fams = [f for f in _HOLLOW_FAMILIES
            if _capacity(f, bh, bw) >= min(floor, 3)
            and not (f in ("frame", "T", "U", "plus", "H") and min(bh, bw) < 3)]
    if not fams:
        return None
    fam = fams[int(rng.integers(0, len(fams)))]
    mask = _to_mask(_build_family(rng, fam, bh, bw, max(2, floor)))
    if mask.shape[0] > bh or mask.shape[1] > bw:
        return None
    if rng.random() < 0.5:
        mask = mask[::-1]
    if rng.random() < 0.5:
        mask = mask[:, ::-1]
    return np.ascontiguousarray(mask), fam


def _sized_body(
    rng: np.random.Generator,
    cells: float,
    lim_h: int,
    lim_w: int,
    *,
    min_cells: int = 3,
    hollow: bool = False,
) -> Optional[tuple[np.ndarray, str]]:
    """A clean body of about ``cells`` cells whose box fits ``lim_h x lim_w``.

    ``hollow`` asks for a family (frame, U, L, H, T, plus, staircase) whose
    bounding box keeps free cells, which is what lets task 93 put another
    object inside the target's crop.
    """
    lim_h, lim_w = max(1, int(lim_h)), max(1, int(lim_w))
    floor = max(1, int(min_cells))
    if lim_h * lim_w < floor:
        floor = lim_h * lim_w
    side = float(np.sqrt(max(1.0, float(cells))))
    for _ in range(14):
        bh = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_h))
        bw = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_w))
        for _ in range(10):
            if bh * bw >= floor:
                break
            if bh < lim_h and (bh <= bw or bw >= lim_w):
                bh += 1
            elif bw < lim_w:
                bw += 1
            else:
                break
        if hollow:
            bh = max(bh, min(3, lim_h))
            bw = max(bw, min(3, lim_w))
            built = _hollow_body(rng, bh, bw, floor)
            if built is None or int(built[0].size - built[0].sum()) < 2:
                continue
            return built
        built = _body(rng, bh, bw, min_cells=floor)
        if built is None:
            continue
        mask, fam = built
        return np.ascontiguousarray(mask), fam
    return None


def _place(
    rng: np.random.Generator,
    grid: np.ndarray,
    occ: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    free: np.ndarray,
    window: Optional[tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Drop ``mask`` on a free spot (optionally inside ``window``) and paint it."""
    if mask.shape[0] > grid.shape[0] or mask.shape[1] > grid.shape[1]:
        return None
    ok = placement_map(mask, free)
    if ok is None:
        return None
    fh, fw = mask.shape
    if window is not None:
        r0, c0, r1, c1 = window
        keep = np.zeros(ok.shape, dtype=bool)
        rlo, rhi = max(0, r0), min(ok.shape[0] - 1, r1 - fh + 1)
        clo, chi = max(0, c0), min(ok.shape[1] - 1, c1 - fw + 1)
        if rhi < rlo or chi < clo:
            return None
        keep[rlo:rhi + 1, clo:chi + 1] = True
        ok = ok & keep
    spots = np.argwhere(ok)
    if not len(spots):
        return None
    r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
    cells = np.zeros(grid.shape, dtype=bool)
    cells[r:r + fh, c:c + fw] = mask
    grid[cells] = color
    occ |= cells
    return cells


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(
    rng: np.random.Generator,
    grid: np.ndarray,
    bg: int,
    occupied: np.ndarray,
    *,
    exclude,
    probability: float = TEXTURE_P,
) -> tuple[np.ndarray, dict]:
    """A sparse or solid library texture on the free cells.

    The object rule is defined by colour, so a texture may touch an object and
    a solid region is as safe as a scatter of dots: neither can become a color X
    object, because ``apply_background`` only ever draws colours that are not
    already in the grid (the target colour is, by construction) and the caller
    excludes the rest.  Nothing about it is named in the prompt.
    """
    h, w = grid.shape
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= probability:
        return grid, dict(_NO_TEXTURE)
    group = "solid" if rng.random() < SOLID_SHARE else "sparse"
    painted = apply_background(rng, grid, bg, occupied, group=group,
                               exclude=[int(c) for c in exclude])
    if painted is None:
        return grid, dict(_NO_TEXTURE)
    display, info = painted
    return display, {
        "texture_group": info["group"],
        "texture_kind": info["kind"],
        "texture_colors": [int(c) for c in info["texture_colors"]],
        "texture_named": False,
        "texture_coverage": round(float(info["coverage"]), 4),
    }


def _palette_target(rng: np.random.Generator) -> int:
    return int(rng.choice(_PALETTE_SIZES, p=_PALETTE_P))


def _colors(rng: np.random.Generator, n: int, exclude) -> list[int]:
    pool = [c for c in range(NUM_COLORS) if c not in {int(x) for x in exclude}]
    n = max(0, min(int(n), len(pool)))
    if n == 0:
        return []
    return [int(c) for c in rng.choice(pool, size=n, replace=False)]


def _assign(rng: np.random.Generator, colors: list[int], n: int) -> list[int]:
    """One colour per planned object, every planned colour used at least once.

    Drawing each object's colour independently from the palette is what used
    to strand the 8-10 colour tail: with five colours and five objects, two of
    them went unused about half the time.
    """
    if not colors or n <= 0:
        return []
    out = list(colors)[:n]
    while len(out) < n:
        out.append(int(colors[int(rng.integers(0, len(colors)))]))
    return [int(out[int(i)]) for i in rng.permutation(len(out))]


def _foreign_objects(rows, target_color: int, object_mask: np.ndarray):
    """The non-target OBJECT components of the scene (texture cells excluded)."""
    h, w = len(rows), len(rows[0])
    pending = {(r, c) for r in range(h) for c in range(w)
               if bool(object_mask[r, c]) and rows[r][c] != target_color}
    out = []
    while pending:
        start = min(pending)
        pending.remove(start)
        color = rows[start[0]][start[1]]
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH + _DIAG:
                q = (r + dr, c + dc)
                if q in pending and rows[q[0]][q[1]] == color:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        out.append(comp)
    return out


# ---------------------------------------------------------------------------
# Task 93 -- tight-crop the unique color X object
# ---------------------------------------------------------------------------

def _plan_93(rng: np.random.Generator) -> dict:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    density = float(rng.uniform(0.15, 0.33))
    ink = density * area
    # A corner-joined target is the only drawing on which the stated
    # connectivity leaves the 4-reading without a unique object, so it is drawn at a solid rate and the
    # connectivity of a solid target is then leaned the other way to keep the
    # 4 / 8 shares near even.
    corner = bool(rng.random() < 0.42)
    pieces = 1 if not corner else (2 if rng.random() < 0.72 else 3)
    target_budget = float(np.clip(ink * float(rng.uniform(0.16, 0.60)), 3.0, TARGET_MAX_CELLS))
    # On a solid share of the grids another object is BIGGER than the target:
    # otherwise "crop the largest body" solves the task without ever reading
    # the colour clause (it scored 75% before this was added).
    big_distractor = bool(rng.random() < 0.45)
    big_budget = float(np.clip(target_budget * float(rng.uniform(1.2, 2.4)), 6.0, 170.0))
    in_box = bool(rng.random() < 0.36)
    # The palette is drawn near-flat over 2..10 and the object count follows
    # from it: every planned colour has to reach the grid on a body of its own,
    # so a rich palette buys more distractors instead of confetti in one colour.
    budget = int(np.clip(area // 22, 1, 12))
    n_distractors = 0 if rng.random() < 0.08 else int(rng.integers(1, budget + 1))
    palette = _palette_target(rng)
    texture = bool(min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P)
    n_colors_objects = int(np.clip(palette - 2 - (1 if texture else 0), 0, budget))
    if n_distractors < n_colors_objects:
        n_distractors = n_colors_objects
    bg = int(rng.integers(0, NUM_COLORS))
    target_color = int(rng.choice([c for c in range(NUM_COLORS) if c != bg]))
    distractor_colors = _colors(rng, n_colors_objects, [bg, target_color])
    return {
        "h": h, "w": w, "area": area, "density": density,
        "corner": corner, "target_pieces": pieces, "target_budget": target_budget,
        "in_box": in_box, "n_distractors": n_distractors,
        "big_distractor": big_distractor, "big_budget": big_budget,
        "bg": bg, "target_color": target_color, "distractor_colors": distractor_colors,
        "colors": _assign(rng, distractor_colors, n_distractors),
        "palette_target": palette, "plan_texture": texture,
        "per_distractor": float(np.clip((ink - target_budget) / max(1, n_distractors), 4.0, 90.0)),
    }


def _build_93(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    h, w, bg = plan["h"], plan["w"], plan["bg"]
    grid = np.full((h, w), bg, dtype=np.int16)
    occ = np.zeros((h, w), dtype=bool)
    lim_h = max(1, min(h - 1, 20))
    lim_w = max(1, min(w - 1, 20))

    # --- the target -------------------------------------------------------
    families: list[str] = []
    if plan["target_pieces"] == 1:
        built = _sized_body(rng, plan["target_budget"], lim_h, lim_w,
                            min_cells=3, hollow=plan["in_box"])
        if built is None:
            return None
        mask, fam = built
        families.append(fam)
    else:
        masks = []
        per = max(2.0, plan["target_budget"] / plan["target_pieces"])
        for _ in range(plan["target_pieces"]):
            built = _sized_body(rng, per, max(1, lim_h - 1), max(1, lim_w - 1), min_cells=2)
            if built is None:
                return None
            masks.append(built[0])
            families.append(built[1])
        joined = _corner_join(rng, masks)
        if joined is None:
            return None
        mask = joined[0]
    target_cells = _place(rng, grid, occ, mask, plan["target_color"],
                          free=np.ones((h, w), dtype=bool))
    if target_cells is None:
        return None
    box = _mask_bbox(target_cells)

    # --- objects inside the target's bounding box -------------------------
    # The crop copies every cell of the box, so a body in the concavity of an
    # L or U target, inside a frame target or in the gap of a corner-joined
    # one is what makes "including any non-object cells" load-bearing.
    in_box_objects = 0
    distractor_families: list[str] = []
    palette = list(plan["distractor_colors"])
    if plan["in_box"] and palette:
        r0, c0, r1, c1 = box
        room = int((~occ)[r0:r1 + 1, c0:c1 + 1].sum())
        for _ in range(2 if room >= 8 else 1):
            if room < 1:
                break
            size = int(rng.integers(1, min(7, max(2, room)) + 1))
            built = _sized_body(rng, size, min(r1 - r0 + 1, 4), min(c1 - c0 + 1, 4), min_cells=1)
            if built is None:
                break
            color = int(palette[int(rng.integers(0, len(palette)))])
            cells = _place(rng, grid, occ, built[0], color, free=~occ, window=box)
            if cells is None:
                break
            in_box_objects += 1
            if rng.random() < 0.6:
                break

    # --- the rest of the scene --------------------------------------------
    placed = 0
    for index, color in enumerate(plan["colors"]):
        want = plan["big_budget"] if (index == 0 and plan["big_distractor"]) else plan["per_distractor"]
        for shrink in (1.0, 0.55, 0.3):
            built = _sized_body(rng, want * shrink, lim_h, lim_w, min_cells=3)
            if built is None:
                continue
            # a third of the bodies may touch what is already drawn: different
            # colours never merge under the rule, and the contact is what a
            # reader who ignored the colour clause would fall for
            free = (~occ) if rng.random() < 0.34 else (~halo(occ))
            cells = _place(rng, grid, occ, built[0], int(color), free=free)
            if cells is None:
                cells = _place(rng, grid, occ, built[0], int(color), free=~halo(occ))
            if cells is not None:
                placed += 1
                distractor_families.append(built[1])
                break
    return {"grid": grid, "occ": occ, "target": target_cells, "box": box,
            "families": families, "distractor_families": distractor_families,
            "n_distractors": placed, "in_box_objects": in_box_objects}


def validate_93(example: dict) -> None:
    """Re-read the emitted grid and re-derive the crop, literally."""
    grid, meta = example["input"], example["metadata"]
    color, conn = int(meta["target_color"]), int(meta["connectivity"])
    comps = color_components(grid, color, conn)
    if len(comps) != 1:
        raise ValueError(f"task 93: the target colour forms {len(comps)} objects, not one")
    if len(comps[0]) != meta["target_cells"]:
        raise ValueError("task 93: the target cell count disagrees with the metadata")
    if list(bbox(comps[0])) != list(meta["target_box"]):
        raise ValueError("task 93: the target box disagrees with the metadata")
    crop = solve_crop(grid, color, conn)
    if crop != example["output"]:
        raise ValueError("task 93: the crop disagrees with the literal reading")
    other = 4 if conn == 8 else 8
    if len(color_components(grid, color, other)) != meta["n_target_components_other"]:
        raise ValueError("task 93: the other-connectivity component count is wrong")
    for row in grid:
        for value in row:
            if not 0 <= int(value) <= 9:
                raise ValueError("task 93: a cell is outside the palette")
    if meta["texture_colors"]:
        if color in meta["texture_colors"]:
            raise ValueError("task 93: a texture colour is the target colour")
        arr = np.asarray(grid)
        if meta["texture_group"] == "sparse":
            tex = np.isin(arr, meta["texture_colors"])
            if not bool(isolated_cell_mask(arr)[tex].all()):
                raise ValueError("task 93: a sparse texture cell is not isolated")


def _example_93(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = _plan_93(rng)
        scene = None
        for _ in range(3):
            scene = _build_93(rng, plan)
            if scene is not None:
                break
        if scene is None:
            continue
        grid, occ = scene["grid"], scene["occ"]
        display, tex = _texture(rng, grid, plan["bg"], occ,
                                exclude=[plan["target_color"], *plan["distractor_colors"]],
                                probability=1.0 if plan["plan_texture"] else 0.0)
        rows = grid_to_list(display.astype(np.int16))

        # Nothing above this line looked at the connectivity.  A corner-joined
        # target has no unique object under 4-connectivity, so only a solid one
        # can take either reading.
        pieces4 = len(color_components(rows, plan["target_color"], 4))
        if pieces4 > 1:
            conn = 8
        else:
            conn = 8 if rng.random() < 0.22 else 4
        comps = color_components(rows, plan["target_color"], conn)
        if len(comps) != 1:
            continue
        other = 4 if conn == 8 else 8
        n_other = len(color_components(rows, plan["target_color"], other))
        r0, c0, r1, c1 = bbox(comps[0])
        out = [list(row[c0:c1 + 1]) for row in rows[r0:r1 + 1]]

        target_set = set(comps[0])
        in_box_objects = sum(
            1 for comp in _foreign_objects(rows, plan["target_color"], occ)
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp)
        )
        foreign_in_box = sum(
            1
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in target_set and bool(occ[r, c])
        )
        present = sorted({int(v) for v in np.unique(display)})
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"], "n_colors": len(present),
            "connectivity": conn, "target_color": int(plan["target_color"]),
            "n_targets": 1,
            "target_cells": len(comps[0]),
            "target_pieces": pieces4,
            "target_box": [r0, c0, r1, c1],
            "output_h": r1 - r0 + 1, "output_w": c1 - c0 + 1,
            "n_target_components_other": n_other,
            "differs_under_other_connectivity": bool(n_other != 1),
            "n_distractors": scene["n_distractors"] + scene["in_box_objects"],
            "distractors_in_box": in_box_objects,
            "foreign_cells_in_box": foreign_in_box,
            "non_target_cells_in_box": (r1 - r0 + 1) * (c1 - c0 + 1) - len(comps[0]),
            "object_kinds": scene["families"],
            "distractor_kinds": scene["distractor_families"],
            "background_color": int(plan["bg"]),
            "occupancy": round(float(occ.sum()) / plan["area"], 4),
            "object_cells": int(occ.sum()),
            "planned_density": round(plan["density"], 4),
            "palette_target": plan["palette_target"],
            "selection_rule": "the unique object of the target color",
            "crop_rule": "smallest rectangle containing all cells of the selected object",
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows, "output": out, "metadata": meta}
        validate_93(example)
        return example
    raise RuntimeError(f"task 93: no valid scene after {MAX_ATTEMPTS} attempts")


@register("93")
def generate_tight_crop_selected_object(rng: np.random.Generator) -> dict:
    """Task 93: crop the smallest rectangle containing the unique color X object."""
    return _example_93(rng)


# ---------------------------------------------------------------------------
# Task 94 -- list the bounding box of every color X object
# ---------------------------------------------------------------------------

def _box_cap(h: int, w: int) -> int:
    """How many target objects the canvas can carry.

    Footprint-derived: a target figure plus the one-cell moat that keeps it
    from merging with its neighbours under 8-connectivity costs about sixteen
    cells, so the cap is the area divided by that, never more than twelve.
    """
    return int(np.clip((h * w) // 16, 1, MAX_BOXES))


def _plan_94(rng: np.random.Generator) -> dict:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    density = float(rng.uniform(0.12, 0.30))
    cap = _box_cap(h, w)
    # Under 4-connectivity the answer counts PIECES, under 8-connectivity it
    # counts corner-joined GROUPS.  A plan-level coin (the connectivity does
    # not exist yet) decides which of the two is drawn near-flat over 1..cap,
    # so neither reading ends up with a lopsided answer histogram.
    anchor = "pieces" if rng.random() < 0.5 else "groups"
    figures: list[int] = []
    if anchor == "pieces":
        left = int(rng.integers(1, cap + 1))
        while left > 0:
            if left >= 2 and rng.random() < 0.45:
                k = 3 if (left >= 3 and rng.random() < 0.25) else 2
            else:
                k = 1
            figures.append(k)
            left -= k
    else:
        left = cap
        for _ in range(int(rng.integers(1, max(1, (cap * 2) // 3) + 1))):
            if left <= 0:
                break
            if left >= 2 and rng.random() < 0.45:
                k = 3 if (left >= 3 and rng.random() < 0.25) else 2
            else:
                k = 1
            figures.append(k)
            left -= k
    n_pieces = sum(figures)
    ink = density * area
    per_piece = float(np.clip(ink * 0.62 / max(1, n_pieces), 2.0, 90.0))
    # The palette is drawn near-flat over 2..10 and the distractor count
    # follows from it: every planned colour needs a body of its own.
    budget = int(np.clip(area // 26, 0, 10))
    n_distractors = 0 if rng.random() < 0.12 else int(rng.integers(1, max(1, budget) + 1))
    palette = _palette_target(rng)
    texture = bool(min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P)
    n_colors_objects = int(np.clip(palette - 2 - (1 if texture else 0), 0, budget))
    if n_distractors < n_colors_objects:
        n_distractors = n_colors_objects
    bg = int(rng.integers(0, NUM_COLORS))
    target_color = int(rng.choice([c for c in range(NUM_COLORS) if c != bg]))
    distractor_colors = _colors(rng, n_colors_objects, [bg, target_color])
    return {
        "h": h, "w": w, "area": area, "density": density, "cap": cap,
        "anchor": anchor, "figures": figures, "n_pieces": n_pieces,
        "per_piece": per_piece, "single_p": 0.17,
        "n_distractors": n_distractors, "bg": bg, "target_color": target_color,
        "distractor_colors": distractor_colors,
        "colors": _assign(rng, distractor_colors, n_distractors),
        "palette_target": palette, "plan_texture": texture,
        "per_distractor": float(np.clip(ink * 0.45 / max(1, n_distractors), 4.0, 80.0)),
    }


def _build_94(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    h, w, bg = plan["h"], plan["w"], plan["bg"]
    grid = np.full((h, w), bg, dtype=np.int16)
    occ = np.zeros((h, w), dtype=bool)
    target_occ = np.zeros((h, w), dtype=bool)
    lim_h = max(1, min(h - 1, 16))
    lim_w = max(1, min(w - 1, 16))
    singles = 0
    families: list[str] = []
    figure_sizes: list[int] = []
    for k in plan["figures"]:
        masks = []
        for _ in range(k):
            if rng.random() < plan["single_p"]:
                masks.append(np.ones((1, 1), dtype=bool))
                families.append("dot")
                singles += 1
                continue
            built = _sized_body(rng, plan["per_piece"], lim_h, lim_w, min_cells=2)
            if built is None:
                return None
            masks.append(built[0])
            families.append(built[1])
        if k == 1:
            fig = masks[0]
        else:
            joined = _corner_join(rng, masks)
            if joined is None:
                return None
            fig = joined[0]
        # a different target figure must never come within one cell of this
        # one, or the two would merge under 8-connectivity
        cells = _place(rng, grid, occ, fig, plan["target_color"],
                       free=~halo(target_occ) & ~occ)
        if cells is None:
            return None
        target_occ |= cells
        figure_sizes.append(int(fig.sum()))

    placed = 0
    touching = 0
    distractor_families: list[str] = []
    for color in plan["colors"]:
        for shrink in (1.0, 0.55, 0.3):
            built = _sized_body(rng, plan["per_distractor"] * shrink, lim_h, lim_w, min_cells=3)
            if built is None:
                continue
            # half of the distractors may touch a target: a different colour is
            # never part of a color X object, and the contact is what a reader
            # who dropped the colour clause would merge
            contact = rng.random() < 0.5
            cells = _place(rng, grid, occ, built[0], int(color),
                           free=(~occ) if contact else (~halo(occ)))
            if cells is None:
                cells = _place(rng, grid, occ, built[0], int(color), free=~halo(occ))
            if cells is None:
                continue
            placed += 1
            distractor_families.append(built[1])
            if bool((halo(cells) & target_occ).any()):
                touching += 1
            break
    return {"grid": grid, "occ": occ, "target_occ": target_occ,
            "n_distractors": placed, "touching": touching, "singles": singles,
            "families": families, "distractor_families": distractor_families,
            "figure_sizes": figure_sizes}


def validate_94(example: dict) -> None:
    """Re-read the emitted grid and re-derive every box, literally."""
    grid, meta = example["input"], example["metadata"]
    color, conn = int(meta["target_color"]), int(meta["connectivity"])
    comps = color_components(grid, color, conn)
    if not comps:
        raise ValueError("task 94: no object of the target colour")
    boxes = sorted((bbox(comp) for comp in comps), key=lambda b: (b[0], b[1]))
    if not _ordering_is_total(boxes):
        raise ValueError("task 94: two boxes share a top row and a left column")
    if format_boxes(boxes) != example["output"]:
        raise ValueError("task 94: the boxes disagree with the literal reading")
    if len(boxes) != meta["n_targets"]:
        raise ValueError("task 94: the object count disagrees with the metadata")
    if sum(len(comp) for comp in comps) != meta["target_cells"]:
        raise ValueError("task 94: the target cell count disagrees with the metadata")
    other = 4 if conn == 8 else 8
    alt = solve_boxes(grid, color, other)
    if format_boxes(alt) != meta["answer_other_connectivity"]:
        raise ValueError("task 94: the other-connectivity answer is not the recorded one")
    if (format_boxes(alt) != example["output"]) != meta["differs_under_other_connectivity"]:
        raise ValueError("task 94: the load-bearing flag is wrong")
    if meta["texture_colors"]:
        if color in meta["texture_colors"]:
            raise ValueError("task 94: a texture colour is the target colour")
        arr = np.asarray(grid)
        if meta["texture_group"] == "sparse":
            tex = np.isin(arr, meta["texture_colors"])
            if not bool(isolated_cell_mask(arr)[tex].all()):
                raise ValueError("task 94: a sparse texture cell is not isolated")


def _example_94(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = _plan_94(rng)
        scene = None
        for _ in range(3):
            scene = _build_94(rng, plan)
            if scene is not None:
                break
        if scene is None:
            continue
        grid, occ = scene["grid"], scene["occ"]
        display, tex = _texture(rng, grid, plan["bg"], occ,
                                exclude=[plan["target_color"], *plan["distractor_colors"]],
                                probability=1.0 if plan["plan_texture"] else 0.0)
        rows = grid_to_list(display.astype(np.int16))

        # Every check that could reject the scene runs on BOTH readings, before
        # the connectivity exists: a rejection that looked at one reading only
        # would correlate the variant with the picture.
        boxes4 = solve_boxes(rows, plan["target_color"], 4)
        boxes8 = solve_boxes(rows, plan["target_color"], 8)
        if not boxes4 or not boxes8:
            continue
        if not _ordering_is_total(boxes4) or not _ordering_is_total(boxes8):
            continue
        if len(boxes4) > MAX_BOXES or len(boxes8) > MAX_BOXES:
            continue
        if len(boxes8) != len(plan["figures"]) or len(boxes4) != plan["n_pieces"]:
            continue

        conn = 4 if rng.random() < 0.5 else 8
        boxes = boxes4 if conn == 4 else boxes8
        alt = boxes8 if conn == 4 else boxes4
        comps = color_components(rows, plan["target_color"], conn)
        in_box = 0
        for comp in _foreign_objects(rows, plan["target_color"], occ):
            if any(any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes)
                   for r, c in comp):
                in_box += 1
        present = sorted({int(v) for v in np.unique(display)})
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"], "n_colors": len(present),
            "connectivity": conn, "target_color": int(plan["target_color"]),
            "n_targets": len(boxes),
            "target_cells": int(scene["target_occ"].sum()),
            "target_object_sizes": sorted(len(comp) for comp in comps),
            "single_cell_targets": sum(1 for comp in comps if len(comp) == 1),
            "n_pieces": plan["n_pieces"],
            "n_figures": len(plan["figures"]),
            "multi_piece_figures": sum(1 for k in plan["figures"] if k > 1),
            "answer_cap": plan["cap"],
            "answer_anchor": plan["anchor"],
            "answer_other_connectivity": format_boxes(alt),
            "n_targets_other_connectivity": len(alt),
            "differs_under_other_connectivity": bool(alt != boxes),
            "n_distractors": scene["n_distractors"],
            "distractors_touching_target": scene["touching"],
            "distractors_in_box": in_box,
            "object_kinds": scene["families"],
            "distractor_kinds": scene["distractor_families"],
            "background_color": int(plan["bg"]),
            "occupancy": round(float(occ.sum()) / plan["area"], 4),
            "object_cells": int(occ.sum()),
            "planned_density": round(plan["density"], 4),
            "palette_target": plan["palette_target"],
            "box_rule": "smallest rectangle enclosing all cells of that object",
            "ordering": "by top row ascending, then by left column ascending",
            "format": "[(top_row,left_column,bottom_row,right_column),...]",
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows, "output": format_boxes(boxes), "metadata": meta}
        validate_94(example)
        return example
    raise RuntimeError(f"task 94: no valid scene after {MAX_ATTEMPTS} attempts")


@register("94")
def generate_target_object_bounding_boxes(rng: np.random.Generator) -> dict:
    """Task 94: the bounding box of every color X object, 4- or 8-connected."""
    return _example_94(rng)
