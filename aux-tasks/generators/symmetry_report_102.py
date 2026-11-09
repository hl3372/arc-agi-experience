"""Task 102 -- report which of five symmetries a grid has.

The September 2026 rebuild replaces the inline version in
``generators/core2_extra/batch_symmetry_f.py``.  What the review found (5000
seeds, its own oracle 5000/5000) and what this module does about it:

* **A forced square for no reason.**  ``square_only = mode in {"main_diagonal",
  "anti_diagonal", "hv_combo"}`` made the horizontal-plus-vertical answer square
  as well, which nothing about it requires; squares were 47% of grids and
  corr(H, W) was 0.43.  Only the two diagonal reflections need H == W, so only
  the four answer classes that contain one draw a square canvas; every other
  class draws the two sides independently over 5..30 (they were 5..16).
* **A "no" was always loud.**  The grid was per-orbit colour noise, so the
  smallest violating fraction on a "no" label was 29% and a reader that checks
  five random cells per label scored 97.5% exact.  Scenes are now drawn -- bodies
  from ``common.shapes`` carried around the orbits of a symmetry GROUP on a
  ground that can carry a folded library texture -- and on a measured share of
  the comparable "no" labels the answer turns on a handful of cells: the scene is
  drawn symmetric under a bigger group and then broken by one small patch or one
  displaced body, so the grid misses that symmetry by 1-4 cells.
* **Two answer classes were unreachable.**  The set of symmetries a grid has is
  always a group, and nine such groups are possible here; the old modes reached
  seven, never {main diagonal, anti-diagonal, 180} and (except by a freak
  accident) never all five.  The class is now drawn first from all nine, near
  balanced, with the all-no class about an eighth of grids.
* **One shared predicate and no validator.**  The answer, the test and the
  generator all called the same ``_is_symmetric``.  The five predicates now have
  two independent implementations -- cell-by-cell coordinate arithmetic and
  whole-array comparisons -- and ``validate_example`` rebuilds the answer string
  from both and raises when they disagree with each other, with the emitted
  answer or with any metadata field.
* **Three metadata keys.**  The example now records the canvas, the palette, how
  it was generated, which labels are present, how small the smallest violation
  is, whether the grid is a near miss and the attempt count.

The follow-up pass of the same review closed two residuals.

* **The answer class could be read off the ink.**  Folding duplicates a body, so
  a scene drawn under a big group came out denser and with fewer colours than a
  freely drawn one: mean occupancy 0.365 on the all-yes class against 0.287 on
  the all-no class, mean colours 4.14 against 5.36, and a depth-4 tree on cheap
  input features scored 25.9% exact against a 12.2% constant.  The cause was
  that the budget followed the construction: the cell budget was a
  stop-when-exceeded loop whose last body overran it by a whole orbit, and the
  palette ceiling was literally ``2 + area // (5 * group_size)``.  The picture is
  now planned before the class is allowed to matter -- an occupancy target, a
  colour-region size and a colour count, all drawn from the canvas alone -- and
  every class is built to meet those numbers: a body's base is sized so that its
  copies land inside the remaining budget, it is cut into bands of the planned
  region size so that one colour region looks the same whatever the group is,
  and a class whose fundamental domain is too small to carry the planned colours
  seats a body across the axes, where it is its own mirror image and costs one
  copy instead of eight.
* **A mode whose feasibility depended on the canvas.**  A near miss used to be
  attempted only when ``area >= 60``, and every failed attempt resampled the
  whole plan, canvas included, so any rejection could tilt the side marginals.
  The canvas and the answer class are now drawn once and never resampled; only
  the drawing is retried, and the near miss falls back to a directly drawn scene
  instead of throwing the canvas away.

Nothing in the prompt names a colour: the task is about the picture's geometry,
so the ground and the texture are never mentioned.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import COLOR_COUNT, SOLID_KINDS, SPARSE_KINDS
from common.grid import NUM_COLORS, grid_to_list
from common.shapes import sample_shape, shape_size
# module import, with the drawing helpers looked up lazily inside _example:
# tasks 99, 101 and 102 all import each other through the registry, and a
# from-import here fails whenever one of them is the entry point (handbook 12)
import generators.symmetry_fill_99 as _sym99
from registry import register

Cell = tuple[int, int]

MIN_SIDE, MAX_SIDE = 5, 30
MAX_AREA = 900
MAX_ATTEMPTS = 60
BODY_TRIES = 90
EDIT_TRIES = 40

LABELS = (
    "horizontal_reflection",
    "vertical_reflection",
    "main_diagonal_reflection",
    "anti_diagonal_reflection",
    "rotational_180",
)
DIAGONAL_LABELS = frozenset({"main_diagonal_reflection", "anti_diagonal_reflection"})

HORIZONTAL, VERTICAL, MAIN, ANTI, HALF_TURN = LABELS

# Every set of symmetries a grid can have is a group, and these nine are the
# ones this label set admits.  The four that contain a diagonal reflection need
# a square canvas; the other five draw their sides independently.
CLASSES: dict[str, frozenset] = {
    "none": frozenset(),
    "horizontal": frozenset({HORIZONTAL}),
    "vertical": frozenset({VERTICAL}),
    "rotation": frozenset({HALF_TURN}),
    "hv": frozenset({HORIZONTAL, VERTICAL, HALF_TURN}),
    "main_diagonal": frozenset({MAIN}),
    "anti_diagonal": frozenset({ANTI}),
    "diagonals": frozenset({MAIN, ANTI, HALF_TURN}),
    "full": frozenset(LABELS),
}
CLASS_WEIGHTS = {
    "none": 0.12, "horizontal": 0.12, "vertical": 0.12, "rotation": 0.11,
    "hv": 0.11, "main_diagonal": 0.11, "anti_diagonal": 0.11,
    "diagonals": 0.10, "full": 0.10,
}
GROUP_SIZE = {"none": 1, "horizontal": 2, "vertical": 2, "rotation": 2,
              "hv": 4, "main_diagonal": 2, "anti_diagonal": 2,
              "diagonals": 4, "full": 8}

NEAR_MISS_P = 0.85        # of the grids that could be drawn from a bigger group
NEAR_MISS_MAX_SHARE = 0.08   # a near miss turns on at most this share of cells
NEAR_MISS_ATTEMPTS = 24   # after this many failures, draw the class directly
RELAX_AFTER = 30          # attempts before the cell budget is allowed to grow
RELAX_FACTOR = 2.5
MOVE_BODY_P = 0.35        # near misses made by displacing a body instead of a patch
TEXTURE_P = 0.30          # attempt rate on grids with a short side of 8+
TEXTURE_SPARSE_P = 0.70

OCCUPANCY_RANGE = (0.16, 0.44)   # body cells over area, drawn before the class
REGION_MIN = 3            # the smallest colour region a body may carry
MAX_BANDS = 6             # colour regions one body is cut into
STRADDLE_P = 0.30         # bodies seated across the axes when the budget allows
SEAT_MISSES = 5           # failures before a centred seat is given up on
# what one colour region costs in cells; the ceiling below is deliberately the
# same arithmetic for every class, so the colour count never reports the group
REGION_COST = 6.0
# the directions a body can be striped in; a band runs along the level sets of
# one of these, and only the ones the scene's group leaves alone are used
BAND_KEYS = (
    lambda r, c, h, w: r,                                   # rows
    lambda r, c, h, w: c,                                   # columns
    lambda r, c, h, w: r + c,                               # main-diagonal stripes
    lambda r, c, h, w: r - c,                               # anti-diagonal stripes
    lambda r, c, h, w: abs(2 * r - h + 1),                  # away from the mid row
    lambda r, c, h, w: abs(2 * c - w + 1),                  # away from the mid column
    lambda r, c, h, w: abs(r - c),                          # away from the diagonal
    lambda r, c, h, w: abs(r + c - h + 1),                  # away from the anti-diagonal
    lambda r, c, h, w: abs(2 * r - h + 1) + abs(2 * c - w + 1),   # diamonds
    lambda r, c, h, w: max(abs(2 * r - h + 1), abs(2 * c - w + 1)),  # rings
)
PALETTE_VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
PALETTE_WEIGHTS = (0.115, 0.125, 0.125, 0.120, 0.115, 0.110, 0.105, 0.095, 0.090)


def square_required(labels) -> bool:
    return bool(set(labels) & DIAGONAL_LABELS)


def supergroups(name: str, square: bool) -> list[str]:
    """The classes that contain this one, on a canvas of this shape."""
    own = CLASSES[name]
    out = []
    for other, labels in CLASSES.items():
        if other == name or not own < labels:
            continue
        if square_required(labels) and not square:
            continue
        out.append(other)
    return out


# ---------------------------------------------------------------------------
# The five predicates, twice
# ---------------------------------------------------------------------------

def partner(r: int, c: int, h: int, w: int, label: str) -> Cell:
    """The cell the prompt's sentence for ``label`` pairs (r, c) with."""
    if label == HORIZONTAL:
        return (h - 1 - r, c)
    if label == VERTICAL:
        return (r, w - 1 - c)
    if label == MAIN:
        return (c, r)
    if label == ANTI:
        return (w - 1 - c, h - 1 - r)
    if label == HALF_TURN:
        return (h - 1 - r, w - 1 - c)
    raise ValueError(f"task 102: unknown label {label!r}")


def present(rows: list[list[int]], label: str) -> bool:
    """Cell by cell, through the coordinate rule the prompt states."""
    h = len(rows)
    w = len(rows[0])
    if label in DIAGONAL_LABELS and h != w:
        return False
    for r in range(h):
        for c in range(w):
            rr, cc = partner(r, c, h, w, label)
            if rows[rr][cc] != rows[r][c]:
                return False
    return True


def literal_present(rows: list[list[int]], label: str) -> bool:
    """The same question as one whole-array comparison: the second reading."""
    grid = np.asarray(rows)
    h, w = grid.shape
    if label in DIAGONAL_LABELS and h != w:
        return False
    if label == HORIZONTAL:
        return bool(np.array_equal(grid, grid[::-1, :]))
    if label == VERTICAL:
        return bool(np.array_equal(grid, grid[:, ::-1]))
    if label == MAIN:
        return bool(np.array_equal(grid, grid.T))
    if label == ANTI:
        return bool(np.array_equal(grid, grid[::-1, ::-1].T))
    if label == HALF_TURN:
        return bool(np.array_equal(grid, grid[::-1, ::-1]))
    raise ValueError(f"task 102: unknown label {label!r}")


def violating_cells(rows: list[list[int]], label: str) -> Optional[int]:
    """How many cells disagree with their partner, or None on a shape mismatch."""
    h = len(rows)
    w = len(rows[0])
    if label in DIAGONAL_LABELS and h != w:
        return None
    bad = 0
    for r in range(h):
        for c in range(w):
            rr, cc = partner(r, c, h, w, label)
            if rows[rr][cc] != rows[r][c]:
                bad += 1
    return bad


def report(rows: list[list[int]]) -> str:
    """The answer: one line per label, in the fixed order."""
    return "\n".join(
        f"{label}: {'yes' if present(rows, label) else 'no'}" for label in LABELS
    )


def solve(rows: list[list[int]]) -> str:
    return report(rows)


# ---------------------------------------------------------------------------
# Orbits of a whole group
# ---------------------------------------------------------------------------

def group_orbits(shape: tuple[int, int], labels):
    """Every cell grouped with the cells the given symmetries tie it to."""
    h, w = shape
    names = list(labels)
    index = [[-1] * w for _ in range(h)]
    orbits: list[tuple[Cell, ...]] = []
    for r in range(h):
        for c in range(w):
            if index[r][c] >= 0:
                continue
            cells: set[Cell] = set()
            stack = [(r, c)]
            while stack:
                cur = stack.pop()
                if cur in cells:
                    continue
                cells.add(cur)
                for name in names:
                    stack.append(partner(cur[0], cur[1], h, w, name))
            for rr, cc in cells:
                index[rr][cc] = len(orbits)
            orbits.append(tuple(sorted(cells)))
    return orbits, index


def components(cells) -> list[set]:
    """The 4-connected pieces of a set of cells."""
    rest = set(cells)
    out: list[set] = []
    while rest:
        seed = rest.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in rest:
                    rest.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def colour_regions(rows: list[list[int]], colors) -> list[int]:
    """Sizes of the same-colour pieces drawn in ``colors`` (the body palette)."""
    wanted = {int(c) for c in colors}
    h, w = len(rows), len(rows[0])
    seen = [[False] * w for _ in range(h)]
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or rows[r][c] not in wanted:
                continue
            color = rows[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] \
                            and rows[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


# ---------------------------------------------------------------------------
# Plan: the picture is budgeted before the class is allowed to matter
# ---------------------------------------------------------------------------

def region_ceiling(area: int) -> int:
    """The biggest colour region a canvas of this size carries."""
    return int(min(24, max(4, area // 20)))


def palette_ceiling(budget: int, region: int, n_texture: int) -> int:
    """How many colours the planned ink can carry, whatever the class is.

    One colour region costs about ``REGION_COST * region`` cells: on a class
    with a big group a region is either copied -- eight cells for every one
    drawn -- or seated across the axes, where it is its own image and costs what
    it covers, and measured over the nine classes the mean lands near four and a
    half.  The same arithmetic is used for every class, so the colour count
    cannot report the group.
    """
    return int(min(10, max(2, 1 + n_texture + int(budget / (REGION_COST * region)))))


def palette_cost(palette: int, region: int, n_texture: int) -> int:
    """The ink a palette of this size needs before any class can show it."""
    return int(REGION_COST * region * max(0, palette - 1 - n_texture))


def _sample_plan(rng: np.random.Generator) -> dict:
    """The answer class and canvas first, then a budget that ignores both."""
    names = list(CLASSES)
    probs = np.array([CLASS_WEIGHTS[n] for n in names], dtype=float)
    answer = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    if square_required(CLASSES[answer]):
        h = w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    else:
        h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
        w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    ground = int(rng.integers(0, NUM_COLORS))

    kind: Optional[str] = None
    n_texture_colors = 0
    if min(h, w) >= 8 and rng.random() < TEXTURE_P:
        kinds = SPARSE_KINDS if rng.random() < TEXTURE_SPARSE_P else SOLID_KINDS
        kind = str(kinds[int(rng.integers(0, len(kinds)))])
        lo, hi = COLOR_COUNT[kind]
        n_texture_colors = int(rng.integers(lo, hi + 1))

    # The palette is drawn first, near-flat over what the canvas could hold at
    # its finest, and the picture is then granular enough to show it: a colour
    # is a region, so ten of them need ten regions' worth of ink, and a picture
    # of two colours is free to draw them big.  Both numbers come from the
    # canvas, never from the class.
    ceiling = palette_ceiling(int(OCCUPANCY_RANGE[1] * area), REGION_MIN,
                              n_texture_colors)
    values = [v for v in PALETTE_VALUES if v <= ceiling]
    weights = np.array(PALETTE_WEIGHTS[:len(values)], dtype=float)
    palette = int(rng.choice(values, p=weights / weights.sum()))
    top = region_ceiling(area)
    needed = palette - 1 - n_texture_colors
    if needed > 0:
        top = min(top, int(OCCUPANCY_RANGE[1] * area / (REGION_COST * needed)))
    region = int(rng.integers(REGION_MIN, max(REGION_MIN, top) + 1))
    occupancy = float(rng.uniform(*OCCUPANCY_RANGE))
    budget = max(REGION_MIN, int(round(occupancy * area)),
                 palette_cost(palette, region, n_texture_colors))
    occupancy = budget / area
    return {
        "answer": answer,
        "h": h,
        "w": w,
        "area": area,
        "ground": ground,
        "occupancy_target": round(occupancy, 4),
        "body_budget": budget,
        "region_cells": region,
        "palette_target": palette,
        "texture_kind": kind,
        "n_texture_colors": n_texture_colors,
        "move_body": bool(rng.random() < MOVE_BODY_P),
    }


# ---------------------------------------------------------------------------
# Drawing: bodies sized so that the copies land inside the budget
# ---------------------------------------------------------------------------

def seat_modes(labels, multiplier: int) -> list[str]:
    """The ways a body can be seated under this group, cheapest copy last.

    A body seated across an axis of the group is its own image in that axis, so
    the fold makes half as many copies of it; seated across all of them it is
    the whole picture's centre piece and the fold makes none.  That is how a
    class with a big group pays for the colours the plan asked for without
    spending more ink than a freely drawn one.
    """
    modes = ["none"]
    if multiplier > 1:
        if HORIZONTAL in labels:
            modes.append("rows")
        if VERTICAL in labels:
            modes.append("cols")
        modes.append("both")
    return modes


def copies_under(mode: str, multiplier: int) -> int:
    """How many cells a body of one cell costs when seated this way."""
    if mode == "both":
        return 1
    if mode in ("rows", "cols"):
        return max(1, multiplier // 2)
    return multiplier


def _seat(rng: np.random.Generator, plan: dict, mode: str, sh: int, sw: int,
          free) -> Cell:
    """Where to put a shape of ``sh`` by ``sw``.

    A centred seat straddles the axes named by ``mode``; every other coordinate
    is drawn from the cells no body has claimed yet, which keeps a crowded
    canvas from spending all its tries on positions that are already taken.
    """
    h, w = plan["h"], plan["w"]
    if len(free):
        ar, ac = free[int(rng.integers(0, len(free)))]
        loose_r = int(min(max(int(ar) - int(rng.integers(0, sh)), 0), h - sh))
        loose_c = int(min(max(int(ac) - int(rng.integers(0, sw)), 0), w - sw))
    else:
        loose_r = int(rng.integers(0, h - sh + 1))
        loose_c = int(rng.integers(0, w - sw + 1))
    r0 = _sym99._centered_seat(rng, h, sh) if mode in ("rows", "both") else loose_r
    c0 = _sym99._centered_seat(rng, w, sw) if mode in ("cols", "both") else loose_c
    return r0, c0


def _split_regions(rng: np.random.Generator, base: set, cells: set, orbits, index,
                   region: int) -> Optional[list[frozenset]]:
    """Cut a body into bands one region big, then fold each band.

    The cut runs along the orbits, so every band is closed under the scene's
    group and the picture stays exactly symmetric.  How many bands to cut is
    read off the body as it will be SEEN: a body whose copies land apart shows
    pieces the size of the drawn shape, one whose copies land on the diagonal
    shows a single piece twice that size, and the cut is made finer in exactly
    that proportion, so one colour region looks the same under every group.
    """
    pieces = components(cells)
    bands = int(max(1, min(MAX_BANDS, round(len(cells) / len(pieces) / region))))
    if bands == 1:
        return [frozenset(cells)] if min(len(p) for p in pieces) >= REGION_MIN else None
    per = max(1, len(base) / bands)

    weight: dict[int, int] = {}
    first: dict[int, Cell] = {}
    touched: list[int] = []
    for r, c in sorted(base):
        oid = index[r][c]
        if oid not in weight:
            touched.append(oid)
        weight[oid] = weight.get(oid, 0) + 1
        first.setdefault(oid, (r, c))
    h, w = len(index), len(index[0])
    # a band has to be a union of orbits, so the direction it runs in must be
    # constant on them: stripes across a mirrored body, rings around a rotated
    # one.  Cutting a folded body along a direction its group does not preserve
    # is what used to leave odd little pieces, and those pieces are visible in
    # a component count -- which is a reader of the answer class.
    live = [line for line in BAND_KEYS
            if all(len({line(r, c, h, w) for r, c in orbits[oid]}) == 1
                   for oid in touched)]
    line = live[int(rng.integers(0, len(live)))] if live else BAND_KEYS[-1]
    oids = sorted(weight, key=lambda oid: (line(*first[oid], h, w), first[oid]))

    groups: list[list[int]] = []
    current: list[int] = []
    carried = 0
    for oid in oids:
        current.append(oid)
        carried += weight[oid]
        if carried >= per and len(groups) + 1 < bands:
            groups.append(current)
            current = []
            carried = 0
    if current:
        groups.append(current)

    out: list[frozenset] = []
    for group in groups:
        band: set[Cell] = set()
        for oid in group:
            band.update(orbits[oid])
        if any(len(comp) < REGION_MIN for comp in components(band)):
            return None
        out.append(frozenset(band))
    return out or None


def _place_bodies(rng: np.random.Generator, plan: dict, labels, orbits, index,
                  multiplier: int, slots_needed: int):
    """Seat bodies until the planned cell budget is spent.

    Every candidate is measured on the cells it would really add -- the union of
    the orbits it touches -- and dropped when those overrun the budget, so the
    ink on the canvas is the number the plan drew and not a consequence of how
    many copies the group makes.  When the colours still to place need more room
    than the copies would leave, the body is seated across the axes, where it is
    its own image.
    """
    h, w, area = plan["h"], plan["w"], plan["area"]
    budget, region = plan["body_budget"], plan["region_cells"]
    slack = max(2, int(round(0.01 * area)))
    weights = _sym99.LARGE_BODY_WEIGHTS if region >= 5 else _sym99.BODY_WEIGHTS
    blocked = np.zeros((h, w), dtype=bool)
    free = np.argwhere(~blocked)
    cheap_modes = [m for m in seat_modes(labels, multiplier)
                   if copies_under(m, multiplier) < multiplier]
    misses: dict[str, int] = {}
    bodies: list[dict] = []
    covered = 0
    slots = 0
    topped_up = False
    for _ in range(BODY_TRIES):
        room = budget + slack - covered
        if covered >= budget or room < REGION_MIN:
            break
        short = max(0, slots_needed - slots)
        per_slot = room / short if short else float(room)
        live = [m for m in cheap_modes if misses.get(m, 0) < SEAT_MISSES]
        want_cheap = bool(live) and (
            multiplier * REGION_MIN > room
            or per_slot < multiplier * region
            or rng.random() < STRADDLE_P)
        mode = live[int(rng.integers(0, len(live)))] if want_cheap else "none"
        copies = copies_under(mode, multiplier)
        if mode == "none":
            max_h = max(2, min(h, int(h * 0.6) + 1))
            max_w = max(2, min(w, int(w * 0.6) + 1))
        else:
            max_h = max(2, min(h, int(round(h * 0.8))))
            max_w = max(2, min(w, int(round(w * 0.8))))
        cap = room // copies
        allowance = room
        if cap < REGION_MIN:
            if mode == "none" and live:
                continue           # too little room left for a copied body
            # the leftover is smaller than the smallest body this group can
            # make: overrunning it by a few cells lands nearer the plan than
            # leaving a whole orbit of it undrawn, so one body is allowed to
            if topped_up or room * 2 < copies * REGION_MIN:
                break
            topped_up = True
            cap = REGION_MIN
            allowance = copies * REGION_MIN
        # a body carries as many colour regions as the palette still needs,
        # so a class that can only afford a few bodies still reaches the count
        bands = int(min(MAX_BANDS, max(1, short))) if want_cheap and short \
            else int(rng.integers(1, MAX_BANDS + 1))
        want = int(round(bands * region * float(rng.uniform(0.85, 1.15))))
        want = max(REGION_MIN, min(want, cap))
        # the budget is spent, not approached: when what would be left over is
        # too small for another body to use, this body takes all of it, so the
        # ink lands on the plan's number instead of a whole orbit short of it
        if cap - want < REGION_MIN:
            want = cap
        shape = None
        for _ in range(3):
            shape, family = sample_shape(rng, want, max_h, max_w, weights=weights)
            if len(shape) >= REGION_MIN:
                break
        if shape is None or len(shape) < REGION_MIN:
            continue
        sh, sw = shape_size(shape)
        if sh > h or sw > w:
            continue
        r0, c0 = _seat(rng, plan, mode, sh, sw, free)
        base = {(r0 + dr, c0 + dc) for dr, dc in shape}
        cells: set[Cell] = set()
        for r, c in base:
            cells.update(orbits[index[r][c]])
        if len(cells) > allowance or any(blocked[r, c] for r, c in cells):
            if mode != "none":
                misses[mode] = misses.get(mode, 0) + 1
            continue
        if not _sym99._body_is_clean(cells, REGION_MIN):
            continue
        regions = _split_regions(rng, base, cells, orbits, index, region)
        if regions is None:
            continue
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True
        free = np.argwhere(~blocked)
        bodies.append({
            "cells": frozenset(cells),
            "base": frozenset(base),
            "family": family,
            "regions": regions,
            "seat": mode,
        })
        covered += len(cells)
        slots += len(regions)
    return bodies


def _neighbour_regions(regions: list[frozenset]) -> list[set]:
    """Which bands of one body touch which, so two of them never merge."""
    owner: dict[Cell, int] = {}
    for i, cells in enumerate(regions):
        for cell in cells:
            owner[cell] = i
    out: list[set] = [set() for _ in regions]
    for (r, c), i in owner.items():
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            j = owner.get(nb)
            if j is not None and j != i:
                out[i].add(j)
                out[j].add(i)
    return out


def _paint(rng: np.random.Generator, grid: np.ndarray, bodies: list[dict],
           ground: int, forbidden: set, want_colors: int) -> list[int]:
    """Give every band a colour, using exactly ``want_colors`` of them if it can."""
    slots = [(bi, ri) for bi, body in enumerate(bodies)
             for ri in range(len(body["regions"]))]
    pool = [c for c in range(NUM_COLORS) if c != ground and c not in forbidden]
    n = max(1, min(want_colors, len(slots), len(pool)))
    colors = [int(v) for v in rng.choice(pool, size=n, replace=False)]
    order = list(range(len(slots)))
    rng.shuffle(order)
    touching = [_neighbour_regions(body["regions"]) for body in bodies]
    chosen: dict[tuple[int, int], int] = {}
    for rank, pos in enumerate(order):
        bi, ri = slots[pos]
        if rank < n:
            color = colors[rank]
        else:
            taken = {chosen[(bi, rj)] for rj in touching[bi][ri] if (bi, rj) in chosen}
            free = [c for c in colors if c not in taken] or colors
            color = int(free[int(rng.integers(0, len(free)))])
        chosen[(bi, ri)] = color
    for (bi, ri), color in chosen.items():
        bodies[bi].setdefault("colors", {})[ri] = color
        for r, c in bodies[bi]["regions"][ri]:
            grid[r, c] = color
    for body in bodies:
        used = set(body["colors"].values())
        body["color"] = int(next(iter(used))) if len(used) == 1 else None
    return sorted(set(colors))


# ---------------------------------------------------------------------------
# Breaking a symmetry by a hair
# ---------------------------------------------------------------------------

def _closure(cells, orbits, index) -> set[Cell]:
    out: set[Cell] = set()
    for r, c in cells:
        out.update(orbits[index[r][c]])
    return out


def _small_patch(rng: np.random.Generator, shape: tuple[int, int], seed: Cell,
                 want: int) -> set[Cell]:
    """A connected block of ``want`` cells grown from ``seed``."""
    h, w = shape
    patch = {seed}
    while len(patch) < want:
        options = [
            (r + dr, c + dc)
            for r, c in patch
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in patch
        ]
        if not options:
            break
        patch.add(options[int(rng.integers(0, len(options)))])
    return patch


def _break_to_answer(rng: np.random.Generator, grid: np.ndarray, plan: dict,
                     bodies: list[dict]):
    """Knock the scene's symmetry down to the answer's, by as little as possible.

    The edit is closed under the ANSWER's group, so that group survives it, and
    it is small, so the symmetries the scene had and the answer does not are
    missed by a few cells rather than by half the canvas.  Every candidate edit
    is measured on the grid it produces: it is kept only when the answer class
    comes out exactly right, when the smallest violation is still a near miss
    AND when the palette is the one the plan drew -- a patch that swallowed a
    whole colour region would report the class through the colour count.
    """
    h, w = plan["h"], plan["w"]
    shape = (h, w)
    wanted = CLASSES[plan["answer"]]
    orbits, index = group_orbits(shape, wanted)
    # the break is a handful of cells, never a share of a big canvas: a near miss
    # has to stay a near miss on a 30x30 grid as much as on a 8x8 one
    limit = int(min(max(2, NEAR_MISS_MAX_SHARE * plan["area"]), 12))
    move_limit = int(min(max(4, NEAR_MISS_MAX_SHARE * plan["area"]), 26))
    palette = {int(v) for v in np.unique(grid)}
    pool = sorted(palette)
    body_cells = {cell for body in bodies for cell in body["cells"]}
    seeds = sorted(body_cells | {
        (r + dr, c + dc)
        for r, c in body_cells
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= r + dr < h and 0 <= c + dc < w
    })

    def accepted(candidate: np.ndarray) -> bool:
        """The class is exactly the answer's and the miss is still a near one."""
        if {int(v) for v in np.unique(candidate)} != palette:
            return False
        rows = grid_to_list(candidate)
        labels = frozenset(label for label in LABELS if present(rows, label))
        if labels != wanted:
            return False
        absent = [violating_cells(rows, label) for label in LABELS if label not in labels]
        comparable = [v for v in absent if v is not None]
        if not comparable or min(comparable) == 0:
            return False
        return min(comparable) / (h * w) <= NEAR_MISS_MAX_SHARE

    movable = [body for body in bodies if body.get("color") is not None]
    if plan["move_body"] and movable:
        for _ in range(EDIT_TRIES):
            body = movable[int(rng.integers(0, len(movable)))]
            cells = set(body["cells"])
            if len(cells) > move_limit:
                continue
            dr = int(rng.integers(-2, 3))
            dc = int(rng.integers(-2, 3))
            if dr == 0 and dc == 0:
                continue
            moved = {(r + dr, c + dc) for r, c in sorted(body["base"])}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
                continue
            target = _closure(moved, orbits, index)
            if len(target | cells) > move_limit:
                continue
            candidate = grid.copy()
            for r, c in cells:
                candidate[r, c] = plan["ground"]
            color = int(body["color"])
            for r, c in target:
                candidate[r, c] = color
            if accepted(candidate):
                return candidate, "move_body", len(target | cells)

    for _ in range(EDIT_TRIES):
        want = int(rng.choice([1, 2, 3, 4], p=[0.40, 0.35, 0.17, 0.08]))
        if seeds and rng.random() < 0.75:
            seed = seeds[int(rng.integers(0, len(seeds)))]
        else:
            seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        patch = _small_patch(rng, shape, seed, want)
        cells = _closure(patch, orbits, index)
        if len(cells) > limit:
            continue
        color = int(pool[int(rng.integers(0, len(pool)))])
        if all(int(grid[r, c]) == color for r, c in cells):
            continue
        candidate = grid.copy()
        for r, c in cells:
            candidate[r, c] = color
        if accepted(candidate):
            return candidate, "patch", len(cells)
    return None, "", 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> None:
    """Rebuild the answer from two independent readings and check the metadata."""
    rows = example["input"]
    answer = example["output"]
    meta = example["metadata"]
    h, w = len(rows), len(rows[0])
    if (h, w) != (int(meta["grid_h"]), int(meta["grid_w"])):
        raise ValueError("task 102: metadata size does not match the grid")
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE and h * w <= MAX_AREA):
        raise ValueError(f"task 102: canvas {h}x{w} outside the envelope")
    if list(meta["allowed_symmetries"]) != list(LABELS):
        raise ValueError("task 102: the label order changed")

    lines = answer.split("\n")
    if len(lines) != len(LABELS):
        raise ValueError("task 102: the answer does not have one line per label")
    for label, line in zip(LABELS, lines):
        if line not in (f"{label}: yes", f"{label}: no"):
            raise ValueError(f"task 102: malformed answer line {line!r}")
    cell_reading = "\n".join(
        f"{label}: {'yes' if present(rows, label) else 'no'}" for label in LABELS)
    array_reading = "\n".join(
        f"{label}: {'yes' if literal_present(rows, label) else 'no'}" for label in LABELS)
    if cell_reading != array_reading:
        raise ValueError("task 102: the two readings of the predicates disagree")
    if cell_reading != answer:
        raise ValueError("task 102: the re-solved report differs from the answer")

    labels = [label for label in LABELS if present(rows, label)]
    if labels != list(meta["present_labels"]):
        raise ValueError("task 102: present_labels does not match the grid")
    if int(meta["n_present"]) != len(labels):
        raise ValueError("task 102: n_present does not match the grid")
    # the set of symmetries a grid has is always a group
    if frozenset(labels) not in set(CLASSES.values()):
        raise ValueError(f"task 102: {labels} is not one of the nine possible classes")
    if h != w and (set(labels) & DIAGONAL_LABELS):
        raise ValueError("task 102: a diagonal reflection on a rectangle")
    if int(meta["n_colors"]) != len({v for row in rows for v in row}):
        raise ValueError("task 102: n_colors does not match the grid")

    # the ink: the budget the plan drew has to be the budget on the canvas
    body_colors = [int(c) for c in meta["body_colors"]]
    if not body_colors or len(set(body_colors)) != len(body_colors):
        raise ValueError("task 102: the body palette is not a set of drawn colours")
    if int(meta["bg_color"]) in body_colors or set(body_colors) & set(meta["texture_colors"]):
        raise ValueError("task 102: the body palette overlaps the ground or the texture")
    if int(meta["n_body_colors"]) != len(body_colors):
        raise ValueError("task 102: n_body_colors does not match the palette")
    sizes = colour_regions(rows, body_colors)
    if not sizes:
        raise ValueError("task 102: the body palette is not on the canvas")
    if sum(sizes) != int(meta["body_cells"]):
        raise ValueError("task 102: body_cells does not match the grid")
    if len(sizes) != int(meta["n_regions"]):
        raise ValueError("task 102: n_regions does not match the grid")
    if max(sizes) != int(meta["largest_region"]):
        raise ValueError("task 102: largest_region does not match the grid")
    if abs(float(meta["occupancy"]) - round(sum(sizes) / (h * w), 4)) > 1e-9:
        raise ValueError("task 102: occupancy does not match the grid")
    # a directly drawn scene is all bodies and bands; only the near miss is
    # allowed to leave a mark of a cell or two, because that mark is the answer
    if meta["generation_mode"] in ("folded", "free") and min(sizes) < REGION_MIN:
        raise ValueError("task 102: a drawn scene with a fragment for a region")

    absent = [violating_cells(rows, label) for label in LABELS if label not in labels]
    comparable = [v for v in absent if v is not None]
    smallest = min(comparable) if comparable else 0
    fraction = smallest / (h * w) if comparable else 1.0
    if abs(float(meta["min_violating_fraction"]) - round(fraction, 4)) > 1e-9:
        raise ValueError("task 102: min_violating_fraction does not match the grid")
    if bool(meta["near_miss"]) != bool(comparable and round(fraction, 4) <= NEAR_MISS_MAX_SHARE):
        raise ValueError("task 102: near_miss does not match the grid")
    if comparable and smallest == 0:
        raise ValueError("task 102: a label reported absent is not violated anywhere")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    apply_symmetric_texture = _sym99._apply_symmetric_texture
    plan = _sample_plan(rng)
    h, w, area = plan["h"], plan["w"], plan["area"]
    answer = plan["answer"]
    bigger = supergroups(answer, h == w)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        scene = answer
        if bigger and attempt <= NEAR_MISS_ATTEMPTS and rng.random() < NEAR_MISS_P:
            scene = bigger[int(rng.integers(0, len(bigger)))]
        scene_labels = CLASSES[scene]
        orbits, index = group_orbits((h, w), scene_labels)

        # a canvas of 36 cells has orbits of eight under the full group, so a
        # budget of nine cells cannot be met by any drawn body; after a run of
        # failures the budget, and only the budget, is allowed to grow
        drawn = plan
        if attempt > RELAX_AFTER:
            grown = int(min(0.5 * area, plan["body_budget"] * RELAX_FACTOR))
            drawn = dict(plan, body_budget=max(plan["body_budget"], grown))
        want_colors = max(1, plan["palette_target"] - 1 - plan["n_texture_colors"])
        bodies = _place_bodies(
            rng, drawn, scene_labels, orbits, index, GROUP_SIZE[scene], want_colors)
        if not bodies:
            continue

        grid = np.full((h, w), plan["ground"], dtype=np.int16)
        occupied = np.zeros((h, w), dtype=bool)
        for body in bodies:
            for r, c in body["cells"]:
                occupied[r, c] = True
        # the texture goes down first, on the bare ground, so the colours the
        # bodies may use are known exactly and the palette lands on its target
        grid, tex_info = apply_symmetric_texture(
            rng, grid, plan, occupied, orbits, index, exclude=[plan["ground"]],
        )
        tex_colors = [int(v) for v in tex_info["texture_colors"]] if tex_info else []
        want_colors = max(1, plan["palette_target"] - 1 - len(tex_colors))
        body_colors = _paint(rng, grid, bodies, plan["ground"], set(tex_colors),
                             want_colors)

        mode = "folded" if scene_labels else "free"
        edit_cells = 0
        if scene != answer:
            broken, kind, edit_cells = _break_to_answer(rng, grid, plan, bodies)
            if broken is None:
                continue
            grid = broken
            mode = f"near_miss_{kind}"

        rows = grid_to_list(grid)
        labels = [label for label in LABELS if present(rows, label)]
        if frozenset(labels) != CLASSES[answer]:
            continue                      # the drawing did not land on the planned class
        absent = [violating_cells(rows, label) for label in LABELS if label not in labels]
        comparable = [v for v in absent if v is not None]
        if comparable and min(comparable) == 0:
            continue
        fraction = round(min(comparable) / area, 4) if comparable else 1.0

        sizes = colour_regions(rows, body_colors)
        if not sizes:
            continue
        meta = {
            "grid_h": h,
            "grid_w": w,
            "square": h == w,
            "requires_square": square_required(CLASSES[answer]),
            "n_colors": len({v for row in rows for v in row}),
            "bg_color": plan["ground"],
            "allowed_symmetries": list(LABELS),
            "selection_rule": "report_each_allowed_symmetry_as_yes_or_no",
            "output_format_rule":
                "one_label_per_line_in_allowed_symmetries_order_with_yes_no_values",
            "generation_mode": mode,
            "answer_class": answer,
            "scene_class": scene,
            "n_present": len(labels),
            "present_labels": labels,
            "min_violating_cells": min(comparable) if comparable else 0,
            "min_violating_fraction": fraction,
            "near_miss": bool(comparable and fraction <= NEAR_MISS_MAX_SHARE),
            "edited_cells": edit_cells,
            "n_bodies": len(bodies),
            "body_families": sorted({body["family"] for body in bodies}),
            "body_cells": sum(sizes),
            "occupancy": round(sum(sizes) / area, 4),
            "occupancy_target": plan["occupancy_target"],
            "region_cells": plan["region_cells"],
            "n_regions": len(sizes),
            "largest_region": max(sizes),
            "palette_target": plan["palette_target"],
            "n_body_colors": len(body_colors),
            "body_colors": list(body_colors),
            "texture_group": tex_info["group"] if tex_info else "none",
            "texture_kind": tex_info["kind"] if tex_info else "none",
            "texture_colors": list(tex_colors),
            "texture_coverage": tex_info["coverage"] if tex_info else 0.0,
            "texture_named": False,
            "attempts": attempt,
        }
        example = {"input": rows, "output": report(rows), "metadata": meta}
        validate_example(example)
        return example
    raise RuntimeError(f"task 102: no valid scene after {MAX_ATTEMPTS} attempts")


@register("102")
def generate_identify_grid_symmetry(rng: np.random.Generator) -> dict:
    """Task 102: report, label by label, which symmetries the grid has."""
    return _example(rng)
