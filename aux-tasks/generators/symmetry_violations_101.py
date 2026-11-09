"""Task 101 -- erase the cells of one colour that break that colour's symmetry.

The September 2026 rebuild replaces the inline version in
``generators/core2_extra/batch_symmetry_f.py``.  What the review found (5000
seeds, its own oracle 5000/5000) and what this module does about it:

* **The task id carried the rule.**  Only the two axis mirrors were ever drawn.
  The transform is now drawn per example -- horizontal mirror, vertical mirror,
  main diagonal, anti-diagonal, half turn -- and the prompt states it with its
  coordinate rule and the standing 0-indexed sentence.  Only the two diagonal
  mirrors need a square canvas; the other three draw the two sides independently.
* **Sides were 6..16 and nothing reached 700 cells.**  Sides are now 5..30, drawn
  independently and near-flat, so about 4-5% of grids pass 700 cells.
* **The target colour was confetti.**  The old grid scattered mirrored single
  cells (14.6 components per grid, 59% of them one cell).  The target colour is
  now drawn as BODIES from ``common.shapes``: a body and the image the transform
  makes of it, or one body seated across the axis, on a ground that may carry a
  library texture folded onto the same orbits, so the picture is a symmetric
  drawing before anything breaks it.
* **The violations were dots.**  They are now whole small bodies with no partner,
  or a readable limb growing out of one copy of a paired body; erasing them
  restores the drawing's symmetry.
* **Every grid had the same signature.**  9-14 mirrored pairs, 2-6 violating
  cells and EXACTLY one violation facing a coloured cell, on every single
  example.  The number of violating cells now follows the bodies that carry
  them, violations facing a coloured cell vary 0..k (about three quarters of
  grids carry at least one, never by construction), and about 9% of examples
  have nothing to erase at all.
* **"Look only at cells of color X" decided nothing.**  In the old scenes the
  target colour was the only colour that broke the symmetry, so "recolour every
  cell that breaks the grid's symmetry" was right on 100% of grids.  Distractor
  bodies in other colours now break it too, on almost every grid.
* **No validator and no per-example metadata.**  The output used to be built by
  construction, and the test read the answer's parameters back from metadata.
  Every example is now re-solved twice from the visible grid -- once cell by cell
  through the coordinate rule, once as a whole-array flip -- and the validator
  raises when either disagrees with the answer, when a cell of another colour
  moved, when the erased grid still breaks the symmetry in the target colour, or
  when any metadata count is wrong.

Nothing about the ground or the texture is named: the target colour and the
replacement colour are the two per-example givens, and they are part of the task
statement.  The replacement colour is never the ground colour, so no wording ever
hands the solver the background.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import COLOR_COUNT, SOLID_KINDS, SPARSE_KINDS
from common.grid import NUM_COLORS, grid_to_list
from common.shapes import sample_shape, shape_size
# module import, with the helpers looked up lazily inside the functions: task 99,
# task 101 and task 102 all import each other through the registry, and a
# from-import here fails whenever one of them is the entry point (handbook 12)
import generators.symmetry_fill_99 as _sym99
from registry import register

Cell = tuple[int, int]

MIN_SIDE, MAX_SIDE = 5, 30
MAX_AREA = 900
MAX_ATTEMPTS = 60
BODY_TRIES = 70

TRANSFORMS = (
    "horizontal_mirror",
    "vertical_mirror",
    "main_diagonal",
    "anti_diagonal",
    "rotation_180",
)
# the two transforms that map a canvas onto itself only when it is square
SQUARE_TRANSFORMS = frozenset({"main_diagonal", "anti_diagonal"})

PALETTE_VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
# near-flat with a mild lean to fewer colours (handbook 4.2)
PALETTE_WEIGHTS = (0.130, 0.130, 0.125, 0.120, 0.110, 0.105, 0.100, 0.095, 0.085)

TEXTURE_P = 0.32          # attempt rate on grids with a short side of 8+
TEXTURE_SPARSE_P = 0.68   # of those, the share drawn from the sparse group
IDENTITY_P = 0.09         # examples whose target colour is already symmetric
SHADOW_P = 0.85           # per violating body: plant its image in another colour
LIMB_P = 0.50             # violations that grow out of a paired body
AXIS_BODY_P = 0.45        # grids whose first paired body is seated across the axis

# Bodies a person would draw (the task 99 vocabulary); the free polyominoes stay
# in for the small canvases and are dropped once there is room for real shapes.
BODY_WEIGHTS = {
    "polyomino": 0.12, "bar": 0.07, "rect": 0.14, "square": 0.06, "frame": 0.10,
    "L": 0.08, "T": 0.06, "U": 0.06, "plus": 0.06, "H": 0.04, "staircase": 0.06,
    "blob": 0.11, "big_blob": 0.07, "scaled": 0.03,
}
LARGE_BODY_WEIGHTS = {k: v for k, v in BODY_WEIGHTS.items() if k != "polyomino"}
VIOLATION_WEIGHTS = {
    "polyomino": 0.22, "bar": 0.10, "rect": 0.20, "square": 0.10, "L": 0.12,
    "T": 0.08, "staircase": 0.06, "blob": 0.12,
}


# ---------------------------------------------------------------------------
# The rule (both readings work from the visible grid alone)
# ---------------------------------------------------------------------------

def solve(rows: list[list[int]], transform: str, target: int, replacement: int) -> list[list[int]]:
    """The prompt read literally, cell by cell, through the coordinate rule."""
    map_cell = _sym99.map_cell
    h = len(rows)
    w = len(rows[0])
    out = [list(row) for row in rows]
    for r in range(h):
        for c in range(w):
            if rows[r][c] != target:
                continue
            rr, cc = map_cell(r, c, (h, w), transform)
            if rows[rr][cc] != target:
                out[r][c] = replacement
    return out


_FLIP = {
    "horizontal_mirror": lambda g: g[::-1, :],
    "vertical_mirror": lambda g: g[:, ::-1],
    "main_diagonal": lambda g: g.T,
    "anti_diagonal": lambda g: g[::-1, ::-1].T,
    "rotation_180": lambda g: g[::-1, ::-1],
}


def literal_solve(rows: list[list[int]], transform: str, target: int,
                  replacement: int) -> list[list[int]]:
    """The same rule as one whole-array flip: the validator's second reading.

    ``_FLIP[transform](g)`` holds, at every position, the colour of that
    position's partner, so the answer is one masked assignment.  The two
    implementations share nothing but the answer they must agree on.
    """
    grid = np.asarray(rows, dtype=np.int16)
    if transform in SQUARE_TRANSFORMS and grid.shape[0] != grid.shape[1]:
        raise ValueError(f"task 101: {transform} needs a square canvas, got {grid.shape}")
    partner = _FLIP[transform](grid)
    out = np.where((grid == target) & (partner != target), replacement, grid)
    return [[int(v) for v in row] for row in out]


def violating_cells(rows: list[list[int]], transform: str, target: int) -> list[Cell]:
    """Every cell of ``target`` whose partner in the grid is not ``target``."""
    map_cell = _sym99.map_cell
    h = len(rows)
    w = len(rows[0])
    bad = []
    for r in range(h):
        for c in range(w):
            if rows[r][c] != target:
                continue
            rr, cc = map_cell(r, c, (h, w), transform)
            if rows[rr][cc] != target:
                bad.append((r, c))
    return bad


def changed_cells(rows_in: list[list[int]], rows_out: list[list[int]]) -> list[Cell]:
    return [
        (r, c)
        for r in range(len(rows_in))
        for c in range(len(rows_in[0]))
        if rows_in[r][c] != rows_out[r][c]
    ]


def components(cells) -> list[set[Cell]]:
    """4-connected groups of a cell set."""
    cells = set(cells)
    seen: set[Cell] = set()
    groups = []
    for cell in cells:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        group = {cell}
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    group.add(nb)
                    stack.append(nb)
        groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def _sample_plan(rng: np.random.Generator) -> dict:
    """Transform first, canvas second: a square is a consequence of the rule."""
    transform = TRANSFORMS[int(rng.integers(0, len(TRANSFORMS)))]
    if transform in SQUARE_TRANSFORMS:
        h = w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    else:
        h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
        w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    target = int(rng.integers(0, NUM_COLORS))
    ground = int(rng.choice([c for c in range(NUM_COLORS) if c != target]))
    palette = int(rng.choice(PALETTE_VALUES, p=np.array(PALETTE_WEIGHTS) / sum(PALETTE_WEIGHTS)))
    kind: Optional[str] = None
    n_texture_colors = 0
    if min(h, w) >= 8 and rng.random() < TEXTURE_P:
        kinds = SPARSE_KINDS if rng.random() < TEXTURE_SPARSE_P else SOLID_KINDS
        kind = str(kinds[int(rng.integers(0, len(kinds)))])
        lo, hi = COLOR_COUNT[kind]
        n_texture_colors = int(rng.integers(lo, hi + 1))
    identity = bool(rng.random() < IDENTITY_P)
    n_violation_bodies = 0 if identity else int(rng.choice([1, 2, 3], p=[0.46, 0.34, 0.20]))
    # body counts scale with the canvas, so a 30x30 grid is not two small bodies
    # in a sea of ground (handbook 3.10)
    max_bodies = int(min(7, max(1, 1 + area // 130)))
    n_paired = int(rng.integers(1, max_bodies + 1))
    if area < 120:
        n_violation_bodies = min(n_violation_bodies, 2)
    return {
        "transform": transform,
        "h": h,
        "w": w,
        "area": area,
        "target": target,
        "ground": ground,
        "palette_target": palette,
        "texture_kind": kind,
        "n_texture_colors": n_texture_colors,
        "identity": identity,
        "n_paired": n_paired,
        "n_violation_bodies": n_violation_bodies,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _fold(cells, shape: tuple[int, int], transform: str) -> set[Cell]:
    map_cell = _sym99.map_cell
    return {map_cell(r, c, shape, transform) for r, c in cells}


def _dilate(mask: np.ndarray) -> np.ndarray:
    """The cells of ``mask`` plus their 8-neighbours (the one-cell moat)."""
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out |= np.roll(np.roll(mask, dr, axis=0), dc, axis=1) & _shift_valid(mask, dr, dc)
    return out


def _shift_valid(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """The positions a roll by (dr, dc) filled from inside the canvas."""
    keep = np.ones_like(mask, dtype=bool)
    if dr == 1:
        keep[0, :] = False
    elif dr == -1:
        keep[-1, :] = False
    if dc == 1:
        keep[:, 0] = False
    elif dc == -1:
        keep[:, -1] = False
    return keep


class _Canvas:
    """Seats bodies from the shared vocabulary, keeping a one-cell moat."""

    def __init__(self, rng: np.random.Generator, plan: dict):
        self.rng = rng
        self.plan = plan
        self.h, self.w = plan["h"], plan["w"]
        self.shape = (self.h, self.w)
        self.transform = plan["transform"]
        self.painted = np.zeros(self.shape, dtype=bool)
        self.taken = np.zeros(self.shape, dtype=bool)     # painted plus the moat
        self.target_cells: set[Cell] = set()
        self.reserved: set[Cell] = set()                  # partners that must stay off-target
        area = plan["area"]
        self.floor = int(min(18, max(3, area // 60)))
        self.cap = int(min(42, max(self.floor + 3, area * 0.085)))
        self.max_h = max(2, min(self.h, int(self.h * 0.55) + 1))
        self.max_w = max(2, min(self.w, int(self.w * 0.55) + 1))
        self.weights = BODY_WEIGHTS if area < 220 else LARGE_BODY_WEIGHTS

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    def free(self, cells) -> bool:
        return all(not self.taken[r, c] for r, c in cells)

    def unpainted(self, cells) -> bool:
        return all(not self.painted[r, c] for r, c in cells)

    def mark(self, cells) -> None:
        for r, c in cells:
            self.painted[r, c] = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h and 0 <= nc < self.w:
                        self.taken[nr, nc] = True

    def blocked_by_others(self, own: set[Cell]) -> np.ndarray:
        """The moat of every painted cell that does not belong to ``own``."""
        others = self.painted.copy()
        for r, c in own:
            others[r, c] = False
        return _dilate(others)

    def seat(self, lo: int, hi: int, centered: bool, weights: dict, min_cells: int = 3):
        """Draw one shape from the vocabulary and place it on the canvas."""
        want = int(self.rng.integers(lo, max(lo, hi) + 1))
        shape, family = sample_shape(self.rng, want, self.max_h, self.max_w, weights=weights)
        if len(shape) < min(want, min_cells):
            return None            # a body a person would not bother to draw
        sh, sw = shape_size(shape)
        if sh > self.h or sw > self.w:
            return None
        if centered:
            r0 = _sym99._centered_seat(self.rng, self.h, sh)
            c0 = _sym99._centered_seat(self.rng, self.w, sw)
            if self.transform == "horizontal_mirror":
                c0 = int(self.rng.integers(0, self.w - sw + 1))
            elif self.transform == "vertical_mirror":
                r0 = int(self.rng.integers(0, self.h - sh + 1))
        else:
            r0 = int(self.rng.integers(0, self.h - sh + 1))
            c0 = int(self.rng.integers(0, self.w - sw + 1))
        return {(r0 + dr, c0 + dc) for dr, dc in shape}, family


def _grow_limb(canvas: _Canvas, body: set[Cell], want: int) -> Optional[set[Cell]]:
    """A small readable block growing out of one copy of a paired body.

    The limb is what breaks the symmetry, so every one of its cells must face a
    cell that is not the target colour: its image may touch neither the body nor
    the limb itself, and it may not sit on another body's moat.
    """
    rng = canvas.rng
    shape, transform = canvas.shape, canvas.transform
    blocked = canvas.blocked_by_others(body)

    def usable(cell: Cell) -> bool:
        return (canvas.inside(cell) and cell not in body
                and not canvas.painted[cell] and not blocked[cell])

    seeds = [
        nb
        for cell in body
        for nb in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]),
                   (cell[0], cell[1] - 1), (cell[0], cell[1] + 1))
        if usable(nb)
    ]
    if not seeds:
        return None
    limb = {seeds[int(rng.integers(0, len(seeds)))]}
    while len(limb) < want:
        options = [
            nb
            for cell in limb
            for nb in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]),
                       (cell[0], cell[1] - 1), (cell[0], cell[1] + 1))
            if usable(nb) and nb not in limb
        ]
        if not options:
            break
        limb.add(options[int(rng.integers(0, len(options)))])
    if len(limb) < 2:
        return None
    image = _fold(limb, shape, transform)
    if image & (limb | body) or image & canvas.target_cells:
        return None
    if limb & canvas.reserved:
        return None
    return limb


def _draw_target(rng: np.random.Generator, plan: dict):
    """Paired bodies in the target colour, plus the bodies that break the pairing."""
    canvas = _Canvas(rng, plan)
    shape, transform = canvas.shape, canvas.transform
    bodies: list[dict] = []

    axis_first = bool(rng.random() < AXIS_BODY_P)
    for _ in range(BODY_TRIES):
        if len(bodies) >= plan["n_paired"]:
            break
        centered = (not bodies and axis_first) or (bool(bodies) and rng.random() < 0.15)
        seated = canvas.seat(canvas.floor, canvas.cap, centered, canvas.weights,
                             min_cells=max(3, canvas.floor // 2))
        if seated is None:
            continue
        base, family = seated
        image = _fold(base, shape, transform)
        cells = base | image
        if not canvas.free(cells):
            continue
        if not _sym99._body_is_clean(cells, min(3, len(base))):
            continue
        canvas.mark(cells)
        canvas.target_cells |= cells
        bodies.append({
            "cells": frozenset(cells),
            "violating": frozenset(),
            "family": family,
            "kind": "axis" if base & image else "paired",
        })
    if not bodies:
        return None

    small_lo = 2 if plan["area"] < 150 else 3
    small_hi = max(small_lo + 1, min(10, canvas.cap // 2))
    wanted = plan["n_violation_bodies"]
    made = 0
    for _ in range(BODY_TRIES):
        if made >= wanted:
            break
        hosts = [b for b in bodies if not b["violating"] and b["kind"] != "lone"]
        if hosts and rng.random() < LIMB_P:
            host = hosts[int(rng.integers(0, len(hosts)))]
            limb = _grow_limb(canvas, set(host["cells"]), int(rng.integers(2, 5)))
            if limb is None:
                continue
            merged = set(host["cells"]) | limb
            if not _sym99._body_is_clean(merged, 2):
                continue
            canvas.mark(limb)
            canvas.target_cells |= limb
            canvas.reserved |= _fold(limb, shape, transform)
            host["cells"] = frozenset(merged)
            host["violating"] = frozenset(limb)
            host["kind"] = "paired_with_limb"
            made += 1
            continue
        seated = canvas.seat(small_lo, small_hi, False, VIOLATION_WEIGHTS,
                             min_cells=small_lo)
        if seated is None:
            continue
        base, family = seated
        if len(base) < 2:
            continue
        image = _fold(base, shape, transform)
        if base & image or image & canvas.target_cells:
            continue
        if not canvas.free(base) or base & canvas.reserved:
            continue
        if not _sym99._body_is_clean(base, min(2, len(base))):
            continue
        canvas.mark(base)
        canvas.target_cells |= base
        canvas.reserved |= image
        bodies.append({
            "cells": frozenset(base),
            "violating": frozenset(base),
            "family": family,
            "kind": "lone",
        })
        made += 1
    if made < wanted:
        return None
    return canvas, bodies


def _place_distractors(canvas: _Canvas, bodies: list[dict], colors: list[int],
                       grid: np.ndarray) -> list[dict]:
    """Shadows facing the violations, plus free bodies in the other colours.

    A shadow is the image of a violating body painted in another colour, so the
    violation faces a drawn cell rather than empty ground -- that is what makes
    "if the mirrored partner is also that color" say more than "if the partner is
    empty".  The free bodies are not folded: they break the picture's symmetry in
    colours the rule does not mention, so "look only at cells of that color" has
    to be read.
    """
    rng = canvas.rng
    shape, transform = canvas.shape, canvas.transform
    placed: list[dict] = []
    if not colors:
        return placed
    pending = list(colors)
    rng.shuffle(pending)

    def take_color() -> int:
        if pending:
            return int(pending.pop())
        return int(colors[int(rng.integers(0, len(colors)))])

    for body in bodies:
        if not body["violating"] or rng.random() >= SHADOW_P:
            continue
        image = _fold(body["violating"], shape, transform)
        if not canvas.unpainted(image):
            continue
        color = take_color()
        for r, c in image:
            grid[r, c] = color
        canvas.mark(image)
        placed.append({"cells": frozenset(image), "color": color, "kind": "shadow"})

    want = len(colors) + int(canvas.plan["area"] // 260)
    lo = max(3, canvas.floor // 2)
    hi = max(lo + 2, canvas.cap)
    for _ in range(BODY_TRIES):
        if len(placed) >= want and not pending:
            break
        seated = canvas.seat(lo, hi, False, canvas.weights, min_cells=3)
        if seated is None:
            continue
        cells, _family = seated
        if not canvas.free(cells) or not _sym99._body_is_clean(cells, 3):
            continue
        color = take_color()
        for r, c in cells:
            grid[r, c] = color
        canvas.mark(cells)
        placed.append({"cells": frozenset(cells), "color": color, "kind": "free"})
    return placed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> None:
    """Re-solve the visible grid twice and check every guarantee."""
    map_cell = _sym99.map_cell
    rows_in = example["input"]
    rows_out = example["output"]
    meta = example["metadata"]
    transform = meta["transform"]
    target = int(meta["target_color"])
    replacement = int(meta["replacement_color"])
    ground = int(meta["ground_color"])
    h, w = len(rows_in), len(rows_in[0])
    if (h, w) != (int(meta["grid_h"]), int(meta["grid_w"])):
        raise ValueError("task 101: metadata size does not match the grid")
    if len(rows_out) != h or any(len(row) != w for row in rows_out):
        raise ValueError("task 101: the output is not the same size as the input")
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE and h * w <= MAX_AREA):
        raise ValueError(f"task 101: canvas {h}x{w} outside the envelope")
    if bool(meta["requires_square"]) != (transform in SQUARE_TRANSFORMS):
        raise ValueError("task 101: requires_square does not match the transform")
    if transform in SQUARE_TRANSFORMS and h != w:
        raise ValueError(f"task 101: {transform} on a {h}x{w} canvas")
    if replacement in (target, ground):
        raise ValueError("task 101: the replacement colour must differ from the target and the ground")

    if solve(rows_in, transform, target, replacement) != rows_out:
        raise ValueError("task 101: the cell-by-cell re-solve differs from the answer")
    if literal_solve(rows_in, transform, target, replacement) != rows_out:
        raise ValueError("task 101: the whole-array re-solve differs from the answer")

    changed = changed_cells(rows_in, rows_out)
    for r, c in changed:
        if rows_in[r][c] != target or rows_out[r][c] != replacement:
            raise ValueError(f"task 101: cell ({r},{c}) changed outside the rule")
    bad = violating_cells(rows_in, transform, target)
    if sorted(changed) != sorted(bad):
        raise ValueError("task 101: the changed cells are not the violating cells")
    if violating_cells(rows_out, transform, target):
        raise ValueError("task 101: the answer still breaks the symmetry of the target colour")
    if not any(v == target for row in rows_in for v in row):
        raise ValueError("task 101: the target colour is missing from the input")

    if int(meta["n_violations"]) != len(bad):
        raise ValueError("task 101: n_violations does not match the grid")
    if int(meta["answer_changed_cells"]) != len(changed):
        raise ValueError("task 101: answer_changed_cells does not match the grids")
    coloured = sum(
        1 for r, c in bad
        if rows_in[map_cell(r, c, (h, w), transform)[0]][map_cell(r, c, (h, w), transform)[1]] != ground
    )
    if int(meta["n_coloured_partner_violations"]) != coloured:
        raise ValueError("task 101: n_coloured_partner_violations does not match the grid")
    if bool(meta["identity_output"]) != (rows_in == rows_out):
        raise ValueError("task 101: identity_output does not match the grids")
    if int(meta["n_colors"]) != len({v for row in rows_in for v in row}):
        raise ValueError("task 101: n_colors does not match the input")
    groups = components(bad)
    if int(meta["n_violation_bodies"]) != len(groups):
        raise ValueError("task 101: n_violation_bodies does not match the grid")
    if groups and min(len(g) for g in groups) < 2:
        raise ValueError("task 101: a violation is a single cell")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    map_cell = _sym99.map_cell
    orbit_partition = _sym99.orbit_partition
    _apply_symmetric_texture = _sym99._apply_symmetric_texture
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = _sample_plan(rng)
        h, w, area = plan["h"], plan["w"], plan["area"]
        transform, target, ground = plan["transform"], plan["target"], plan["ground"]
        drawn = _draw_target(rng, plan)
        if drawn is None:
            continue
        canvas, bodies = drawn

        grid = np.full((h, w), ground, dtype=np.int16)
        for body in bodies:
            for r, c in body["cells"]:
                grid[r, c] = target

        n_tex = plan["n_texture_colors"] if plan["texture_kind"] else 0
        n_distractor_colors = max(0, plan["palette_target"] - 2 - n_tex)
        pool = [c for c in range(NUM_COLORS) if c not in (ground, target)]
        rng.shuffle(pool)
        distractor_colors = [int(c) for c in pool[:n_distractor_colors]]
        placed = _place_distractors(canvas, bodies, distractor_colors, grid)

        # the texture is folded onto the transform's orbits, so the ground is as
        # symmetric as the drawing; it lands neither on a body nor on a body's
        # image, so no violation ever faces a texture cell
        orbits, index = orbit_partition((h, w), transform)
        mirrored = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if canvas.painted[r, c]:
                    mirrored[map_cell(r, c, (h, w), transform)] = True
        grid, tex_info = _apply_symmetric_texture(
            rng, grid, plan, canvas.painted | mirrored, orbits, index,
            exclude=[ground, target] + distractor_colors,
        )

        rows_in = grid_to_list(grid)
        present = {v for row in rows_in for v in row}
        if target not in present:
            continue
        bad = violating_cells(rows_in, transform, target)
        planned = {cell for body in bodies for cell in body["violating"]}
        if set(bad) != planned:
            continue                       # a later body changed a partner: redraw
        if plan["identity"] != (not bad):
            continue

        # the replacement colour is a per-example given: never the ground and
        # never the target, drawn either from the colours already on the canvas
        # or from the ones it does not use
        texture_colors = [int(v) for v in (tex_info["texture_colors"] if tex_info else [])]
        scene_pool = [c for c in sorted({int(b["color"]) for b in placed})
                      if c not in texture_colors and c not in (ground, target)]
        reserved_pool = [c for c in range(NUM_COLORS) if c not in present and c != target]
        if scene_pool and (not reserved_pool or rng.random() < 0.5):
            replacement = int(scene_pool[int(rng.integers(0, len(scene_pool)))])
        elif reserved_pool:
            replacement = int(reserved_pool[int(rng.integers(0, len(reserved_pool)))])
        elif scene_pool:
            replacement = int(scene_pool[int(rng.integers(0, len(scene_pool)))])
        else:
            continue

        rows_out = solve(rows_in, transform, target, replacement)
        changed = changed_cells(rows_in, rows_out)
        groups = components(bad)

        def partner_color(cell: Cell) -> int:
            rr, cc = map_cell(cell[0], cell[1], (h, w), transform)
            return rows_in[rr][cc]

        coloured = sum(1 for cell in bad if partner_color(cell) != ground)
        coloured_bodies = sum(1 for g in groups if any(partner_color(cell) != ground for cell in g))
        other_asymmetric = 0
        for r in range(h):
            for c in range(w):
                rr, cc = map_cell(r, c, (h, w), transform)
                if rows_in[r][c] != rows_in[rr][cc] and target not in (rows_in[r][c], rows_in[rr][cc]):
                    other_asymmetric += 1
        target_cells = [(r, c) for r in range(h) for c in range(w) if rows_in[r][c] == target]
        axis_cells = sum(1 for cell in target_cells
                         if map_cell(cell[0], cell[1], (h, w), transform) == cell)
        target_groups = components(target_cells)

        meta = {
            "grid_h": h,
            "grid_w": w,
            "square": h == w,
            "n_colors": len(present),
            "n_output_colors": len({v for row in rows_out for v in row}),
            "transform": transform,
            "requires_square": transform in SQUARE_TRANSFORMS,
            "target_color": target,
            "replacement_color": replacement,
            "ground_color": ground,
            "replacement_reserved": replacement not in present,
            "n_violations": len(bad),
            "n_violation_bodies": len(groups),
            "min_violation_body_cells": min((len(g) for g in groups), default=0),
            "n_coloured_partner_violations": coloured,
            "n_coloured_partner_bodies": coloured_bodies,
            "identity_output": not bad,
            "answer_changed_cells": len(changed),
            "n_target_cells": len(target_cells),
            "n_target_components": len(target_groups),
            "n_single_cell_components": sum(1 for g in target_groups if len(g) == 1),
            "n_axis_target_cells": axis_cells,
            "n_bodies": len(bodies),
            "body_kinds": sorted({body["kind"] for body in bodies}),
            "body_families": sorted({body["family"] for body in bodies}),
            "n_distractor_bodies": len(placed),
            "n_shadow_bodies": sum(1 for b in placed if b["kind"] == "shadow"),
            "distractor_colors": sorted({int(b["color"]) for b in placed}),
            "other_color_asymmetric_cells": other_asymmetric,
            "occupancy": round(len(target_cells) / area, 4),
            "palette_target": plan["palette_target"],
            "texture_group": tex_info["group"] if tex_info else "none",
            "texture_kind": tex_info["kind"] if tex_info else "none",
            "texture_colors": texture_colors,
            "texture_coverage": tex_info["coverage"] if tex_info else 0.0,
            "texture_named": False,
            "attempts": attempt,
        }
        example = {"input": rows_in, "output": rows_out, "metadata": meta}
        validate_example(example)
        return example
    raise RuntimeError(f"task 101: no valid scene after {MAX_ATTEMPTS} attempts")


@register("101")
def generate_remove_symmetry_violations(rng: np.random.Generator) -> dict:
    """Task 101: erase the cells of one colour that break that colour's symmetry."""
    return _example(rng)
