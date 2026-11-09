"""Task 99 -- complete a symmetry: only cells of the unknown colour may change.

The September 2026 rebuild replaces the inline versions of tasks 99 and 100 in
``generators/core2_extra/batch_symmetry_f.py`` and merges 100 into 99.  What the
review found and what this module does about it:

* **The two ids were the same construction.**  99 drew a mirror axis, 100 drew a
  rotation angle; everything else was identical, so the task id handed the model
  the rule.  One task now draws the transform per example -- horizontal mirror,
  vertical mirror, main diagonal, anti-diagonal, rotation by 90 degrees,
  rotation by 180 degrees -- and the prompt states it with its coordinate rule.
  The transform is drawn *first* and the canvas afterwards, so the square
  canvases the diagonals and the quarter turn need are a consequence of the
  rule, never a hint dropped into the scene after the fact.
* **Every grid was per-orbit random colour noise.**  ``_fill_orbits`` gave each
  orbit an independent random colour out of three to five; there was no ground,
  no object, nothing drawn.  Scenes are now pictures: a ground colour, bodies
  from ``common.shapes`` placed once and carried around their orbit by the
  transform (so an object appears with its mirror image or its three rotated
  copies, and a body seated across the axis becomes one self-symmetric body),
  and -- on a bounded share -- a library texture from ``common.backgrounds``
  folded onto its own orbits so that it is symmetric too.  The whole picture is
  exactly symmetric before anything is masked.
* **Coverage.**  Sides were 5..15 x 5..17 and colours 3..5.  Sides are now
  5..30, independent for the two axis mirrors and the half turn, square 5..30
  for the diagonals and the quarter turn; the picture palette is drawn near-flat
  over 2..9 (the unknown colour is reserved and never occurs in the picture, so
  nine is the structural ceiling and the *input* reaches ten colours), and body
  sizes and counts scale with the canvas.
* **Decoy orbits on 100% of grids.**  An orbit that is unknown on every cell
  stays unknown; it used to be present in every single example, which made
  "leave something unknown" the rule rather than a decision.  Decoy orbits are
  now planned on about 30% of grids, cells on the axis (whose orbit is the cell
  itself) on about 45%, and about 7% of examples have nothing to fill at all.
* **No validator.**  The generator emitted whatever it drew.  Every example is
  now re-solved by ``solve``, a literal reading of the prompt that sees only the
  visible input; because ``_example`` also produces the answer with ``solve``, that
  comparison alone is not an independent proof -- the proof is the check of the
  literal fill against the DRAWN picture on every determined orbit, plus the
  symmetry, known-cells-unchanged and orbit-accounting checks. The validator
  raises when any of those disagree, when a known cell changed, when the output
  is not exactly symmetric, or when any metadata count is wrong.
* **The coordinate rule had no origin.**  The prompt wrote ``height - 1 - r``
  without ever saying where row 0 is.  It now carries the standing sentence
  "Rows and columns are 0-indexed from the top-left corner." and the standing
  "Leave every other cell unchanged."

Nothing about the ground or the texture is named: the unknown colour is the one
per-example given the prompt states, because it is part of the task statement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import COLOR_COUNT, SOLID_KINDS, SPARSE_KINDS, apply_background
from common.grid import NUM_COLORS, grid_to_list
from common.shapes import sample_shape, shape_size
from registry import register

Cell = tuple[int, int]

MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 60
BODY_TRIES = 110

TRANSFORMS = (
    "horizontal_mirror",
    "vertical_mirror",
    "main_diagonal",
    "anti_diagonal",
    "rotation_90",
    "rotation_180",
)
# the three transforms that map a canvas onto itself only when it is square
SQUARE_TRANSFORMS = frozenset({"main_diagonal", "anti_diagonal", "rotation_90"})
# the largest orbit a transform can produce (the quarter turn cycles four cells)
MAX_ORBIT = {"rotation_90": 4}

PALETTE_VALUES = (2, 3, 4, 5, 6, 7, 8, 9)
# near-flat with a mild lean to fewer colours (handbook 4.2)
PALETTE_WEIGHTS = (0.155, 0.150, 0.140, 0.130, 0.120, 0.110, 0.100, 0.095)

TEXTURE_P = 0.34          # attempt rate on grids with a short side of 8+
TEXTURE_SPARSE_P = 0.70   # of those, the share drawn from the sparse group
TEXTURE_MAX_COVERAGE = {"sparse": 0.15, "solid": 0.50}  # of the free cells, after folding

DECOY_P = 0.30            # grids that carry at least one all-unknown orbit
AXIS_P = 0.45             # grids that mask at least one cell that maps to itself
IDENTITY_P = 0.07         # examples with nothing to fill
AXIS_BODY_P = 0.55        # grids whose first body is seated across the axis / centre
FIGURE_ORBIT_P = 0.65     # masked cells steered onto the drawing rather than the ground
MAX_UNKNOWN_SHARE = 0.25
BODY_CELL_FRACTION = 0.11  # per-body cell budget, before the transform copies it

# Bodies a person would draw: solid blocks, frames, thick letters, staircases and
# compact blobs lead; the small free polyominoes stay in for the small canvases.
BODY_WEIGHTS = {
    "polyomino": 0.12, "bar": 0.07, "rect": 0.14, "square": 0.06, "frame": 0.10,
    "L": 0.08, "T": 0.06, "U": 0.06, "plus": 0.06, "H": 0.04, "staircase": 0.06,
    "blob": 0.11, "big_blob": 0.07, "scaled": 0.03,
}
LARGE_BODY_WEIGHTS = {k: v for k, v in BODY_WEIGHTS.items() if k != "polyomino"}


# ---------------------------------------------------------------------------
# The rule (nothing here looks at the plan; cells come from the picture)
# ---------------------------------------------------------------------------

def map_cell(r: int, c: int, shape: tuple[int, int], transform: str) -> Cell:
    """Where the stated transform sends row ``r``, column ``c``."""
    h, w = shape
    if transform in SQUARE_TRANSFORMS and h != w:
        raise ValueError(f"task 99: {transform} needs a square canvas, got {h}x{w}")
    if transform == "horizontal_mirror":
        return (h - 1 - r, c)
    if transform == "vertical_mirror":
        return (r, w - 1 - c)
    if transform == "main_diagonal":
        return (c, r)
    if transform == "anti_diagonal":
        return (w - 1 - c, h - 1 - r)
    if transform == "rotation_90":
        return (c, h - 1 - r)
    if transform == "rotation_180":
        return (h - 1 - r, w - 1 - c)
    raise ValueError(f"task 99: unknown transform {transform!r}")


def orbit_partition(shape: tuple[int, int], transform: str):
    """Every cell of the canvas grouped with the cells the rule ties it to.

    Returns ``(orbits, index)``: a list of cell tuples and an ``h x w`` table of
    orbit indices.  An orbit of one cell is a cell the transform fixes (the
    mirror axis, the diagonal, the centre of a rotation).
    """
    h, w = shape
    index = [[-1] * w for _ in range(h)]
    orbits: list[tuple[Cell, ...]] = []
    for r in range(h):
        for c in range(w):
            if index[r][c] >= 0:
                continue
            cells: list[Cell] = []
            cur = (r, c)
            while index[cur[0]][cur[1]] < 0:
                index[cur[0]][cur[1]] = len(orbits)
                cells.append(cur)
                cur = map_cell(cur[0], cur[1], shape, transform)
            orbits.append(tuple(cells))
    return orbits, index


def solve(rows: list[list[int]], transform: str, unknown: int) -> list[list[int]]:
    """The prompt read literally, from the visible grid alone.

    Every cell of an orbit must end up with the same colour.  When some cell of
    the orbit is known, that colour is copied onto the unknown ones; when the
    whole orbit is unknown, it stays unknown.  Two different known colours in
    one orbit would make the example unanswerable, so that raises.
    """
    h = len(rows)
    w = len(rows[0])
    out = [list(row) for row in rows]
    orbits, _ = orbit_partition((h, w), transform)
    for orbit in orbits:
        known = {rows[r][c] for r, c in orbit}
        known.discard(unknown)
        if len(known) > 1:
            raise ValueError(f"task 99: orbit {orbit} carries two known colors {sorted(known)}")
        if not known:
            continue
        color = known.pop()
        for r, c in orbit:
            out[r][c] = color
    return out


def is_symmetric(rows: list[list[int]], transform: str) -> bool:
    h = len(rows)
    w = len(rows[0])
    for r in range(h):
        for c in range(w):
            rr, cc = map_cell(r, c, (h, w), transform)
            if rows[rr][cc] != rows[r][c]:
                return False
    return True


def _components(rows: list[list[int]], mask: list[list[bool]]) -> int:
    """4-connected same-colour components among the cells ``mask`` selects."""
    h = len(rows)
    w = len(rows[0])
    seen = [[False] * w for _ in range(h)]
    total = 0
    for r in range(h):
        for c in range(w):
            if seen[r][c] or not mask[r][c]:
                continue
            total += 1
            color = rows[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                a, b = stack.pop()
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < h and 0 <= nb < w and not seen[na][nb] and mask[na][nb] \
                            and rows[na][nb] == color:
                        seen[na][nb] = True
                        stack.append((na, nb))
    return total


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def _sample_plan(rng: np.random.Generator) -> dict:
    transform = TRANSFORMS[int(rng.integers(0, len(TRANSFORMS)))]
    if transform in SQUARE_TRANSFORMS:
        side = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
        h = w = side
    else:
        h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
        w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    unknown = int(rng.integers(0, NUM_COLORS))
    ground = int(rng.choice([c for c in range(NUM_COLORS) if c != unknown]))
    palette = int(rng.choice(PALETTE_VALUES, p=np.array(PALETTE_WEIGHTS) / sum(PALETTE_WEIGHTS)))
    kind: Optional[str] = None
    n_texture_colors = 0
    if min(h, w) >= 8 and rng.random() < TEXTURE_P:
        kinds = SPARSE_KINDS if rng.random() < TEXTURE_SPARSE_P else SOLID_KINDS
        kind = str(kinds[int(rng.integers(0, len(kinds)))])
        lo, hi = COLOR_COUNT[kind]
        n_texture_colors = int(rng.integers(lo, hi + 1))
    identity = bool(rng.random() < IDENTITY_P)
    want_decoys = bool(rng.random() < DECOY_P)
    want_axis = bool(rng.random() < AXIS_P)
    if identity and not (want_decoys or want_axis):
        # nothing to fill only makes an example when something stays unknown
        want_decoys = True
    return {
        "transform": transform,
        "h": h,
        "w": w,
        "area": h * w,
        "unknown": unknown,
        "ground": ground,
        "palette_target": palette,
        "texture_kind": kind,
        "n_texture_colors": n_texture_colors,
        "min_bodies": max(1, palette - 1 - n_texture_colors),
        "density": float(rng.uniform(0.16, 0.46)),
        "mask_fraction": float(rng.uniform(0.04, 0.20)),
        "want_decoys": want_decoys,
        "want_axis": want_axis,
        "identity": identity,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _body_is_clean(cells: set[Cell], min_component: int) -> bool:
    """A body and its images must read as shapes, not as a speckle lattice.

    The copies the transform makes are accepted only when each is a block of at
    least ``min_component`` cells and the copies either merge into one shape or
    stay a full cell apart.  Copies that only touch at corners -- a thin cross
    or a one-cell bar turned into an interleaved diagonal lattice by a quarter
    turn -- are exactly the per-cell noise the review rejected.
    """
    comp: dict[Cell, int] = {}
    n_comp = 0
    for cell in cells:
        if cell in comp:
            continue
        cid = n_comp
        n_comp += 1
        stack = [cell]
        comp[cell] = cid
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in cells and nb not in comp:
                    comp[nb] = cid
                    stack.append(nb)
        if size < min_component:
            return False
    for (r, c), cid in comp.items():
        for nb in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if comp.get(nb, cid) != cid:
                return False
    return True


def _centered_seat(rng: np.random.Generator, span: int, size: int) -> int:
    """A position that puts a body of ``size`` across the middle of ``span``."""
    base = (span - size) // 2
    jitter = int(rng.integers(-1, 2))
    return int(min(max(base + jitter, 0), span - size))


def _place_bodies(rng: np.random.Generator, plan: dict, orbits, index,
                  multiplier: Optional[int] = None) -> list[dict]:
    """Seat bodies from the shared vocabulary and carry each around its orbit.

    A body is drawn once and then painted on every cell of every orbit it
    touches, so the transform supplies its mirror image or its three rotated
    copies for free and a body seated across the axis comes back as one
    self-symmetric body.  Bodies of different draws keep a one-cell moat, so the
    picture stays a set of readable objects instead of one sprawling mass.

    ``orbits`` / ``index`` are any partition of the canvas, so task 102 passes
    the orbits of a whole symmetry group and states its size in ``multiplier``
    (the number of copies one body turns into); left at None it is the orbit
    size of this task's single transform.
    """
    h, w, area = plan["h"], plan["w"], plan["area"]
    transform = plan["transform"]
    if multiplier is None:
        multiplier = MAX_ORBIT.get(transform, 2)
    budget = int(plan["density"] * area)
    floor = int(min(22, max(4, area // 45)))
    cap = int(min(48, max(floor + 2, area * BODY_CELL_FRACTION / multiplier)))
    min_cells = max(3, floor // 2)
    body_limit = max(0.40 * area, 3 * multiplier)
    weights = BODY_WEIGHTS if floor <= 5 else LARGE_BODY_WEIGHTS
    max_h = max(2, min(h, int(h * 0.6) + 1))
    max_w = max(2, min(w, int(w * 0.6) + 1))
    blocked = np.zeros((h, w), dtype=bool)
    bodies: list[dict] = []
    covered = 0

    def seat(lo: int, hi: int, centered: bool) -> int:
        """Draw one body of lo..hi cells and seat it; returns the cells added."""
        want = int(rng.integers(lo, hi + 1))
        shape, family = sample_shape(rng, want, max_h, max_w, weights=weights)
        if len(shape) < min(want, min_cells):
            return 0                       # a body a person would not bother to draw
        sh, sw = shape_size(shape)
        if sh > h or sw > w:
            return 0
        if centered:
            r0 = _centered_seat(rng, h, sh)
            c0 = _centered_seat(rng, w, sw)
            if transform == "horizontal_mirror":
                c0 = int(rng.integers(0, w - sw + 1))
            elif transform == "vertical_mirror":
                r0 = int(rng.integers(0, h - sh + 1))
        else:
            r0 = int(rng.integers(0, h - sh + 1))
            c0 = int(rng.integers(0, w - sw + 1))
        base = {(r0 + dr, c0 + dc) for dr, dc in shape}
        cells: set[Cell] = set()
        for r, c in base:
            cells.update(orbits[index[r][c]])
        if len(cells) > body_limit:
            return 0
        if any(blocked[r, c] for r, c in cells):
            return 0
        if not _body_is_clean(cells, min(3, len(shape))):
            return 0
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True
        bodies.append({
            "cells": frozenset(cells),
            "family": family,
            "base": frozenset(base),
            "self_symmetric": len(cells) < multiplier * len(base),
        })
        return len(cells)

    axis_first = bool(rng.random() < AXIS_BODY_P)
    for _ in range(BODY_TRIES):
        if covered >= budget:
            break
        centered = (not bodies and axis_first) or (bool(bodies) and rng.random() < 0.12)
        covered += seat(min(floor, cap), cap, centered)
    # the palette is a plan, not an afterthought: when the scene came out with
    # fewer bodies than the planned colour count needs, top it up with smaller
    # ones rather than quietly clamping the palette
    small = max(3, cap // 3)
    for _ in range(80):
        if len(bodies) >= plan["min_bodies"]:
            break
        covered += seat(3, small, False)
    return bodies


def _apply_symmetric_texture(
    rng: np.random.Generator,
    grid: np.ndarray,
    plan: dict,
    occupied: np.ndarray,
    orbits,
    index,
    exclude: list[int],
):
    """Paint a library texture on the free cells and fold it onto its orbits.

    ``apply_background`` draws the texture on a fundamental region's worth of
    free cells like any other task; every textured cell then repaints its whole
    orbit with its own colour, so the ground and its texture are symmetric under
    the transform before anything is masked.  The occupied mask is symmetric by
    construction, so an orbit is either entirely free or entirely occupied and
    the fold never lands on a body.
    """
    kind = plan["texture_kind"]
    if kind is None or plan["n_texture_colors"] <= 0:
        return grid, None
    from common.backgrounds import pick_texture_colors

    colors = pick_texture_colors(rng, plan["n_texture_colors"], sorted(set(exclude)))
    if len(colors) < plan["n_texture_colors"]:
        return grid, None
    painted = apply_background(
        rng, grid, plan["ground"], occupied, kind=kind, colors=colors,
    )
    if painted is None:
        return grid, None
    display, info = painted
    hit: dict[int, int] = {}
    for r, c in np.argwhere(display != grid):
        hit.setdefault(index[int(r)][int(c)], int(display[int(r), int(c)]))
    n_free = int((~occupied).sum())
    # folding multiplies a texture's coverage by the orbit size, so whole orbits
    # are dropped at random until the library's own budget for the group holds
    limit = int(TEXTURE_MAX_COVERAGE[info["group"]] * n_free)
    ids = list(hit)
    rng.shuffle(ids)
    folded = grid.copy()
    used = 0
    for oid in ids:
        size = len(orbits[oid])
        if used + size > limit:
            continue
        for rr, cc in orbits[oid]:
            folded[rr, cc] = hit[oid]
        used += size
    if used == 0:
        return grid, None
    coverage = used / max(1, n_free)
    info = dict(info)
    info["coverage"] = round(coverage, 4)
    info["texture_colors"] = sorted({int(v) for v in folded[folded != grid]})
    return folded, info


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def _mask(rng: np.random.Generator, full: np.ndarray, plan: dict, orbits, index):
    """Replace a bounded share of cells with the unknown colour.

    Determined masks always leave a known partner in the orbit.  Decoy orbits
    (unknown on every cell) and masked cells that map to themselves are the
    cases the prompt tells the solver to leave unknown; both are planned, not
    incidental.
    """
    h, w, area = plan["h"], plan["w"], plan["area"]
    unknown = plan["unknown"]
    ground = plan["ground"]
    grid = full.copy()
    limit = max(1, int(MAX_UNKNOWN_SHARE * area))
    used = 0

    multi = [i for i, orbit in enumerate(orbits) if len(orbit) > 1]
    singles = [i for i, orbit in enumerate(orbits) if len(orbit) == 1]
    rng.shuffle(multi)
    rng.shuffle(singles)

    decoys: list[int] = []
    if plan["want_decoys"] and multi:
        want = int(rng.integers(1, 4))
        for oid in list(multi):
            if len(decoys) >= want:
                break
            if used + len(orbits[oid]) > limit:
                continue
            decoys.append(oid)
            multi.remove(oid)
            for r, c in orbits[oid]:
                grid[r, c] = unknown
            used += len(orbits[oid])

    axis: list[int] = []
    if plan["want_axis"] and singles:
        want = int(rng.integers(1, min(3, len(singles)) + 1))
        for oid in singles[:want]:
            if used + 1 > limit:
                break
            axis.append(oid)
            r, c = orbits[oid][0]
            grid[r, c] = unknown
            used += 1

    filled: list[Cell] = []
    figure_fills = 0
    if not plan["identity"]:
        target = max(1, int(round(plan["mask_fraction"] * area)))
        figure = [oid for oid in multi if any(full[r, c] != ground for r, c in orbits[oid])]
        on_figure_ids = set(figure)
        plain = [oid for oid in multi if oid not in on_figure_ids]
        while len(filled) < target and (figure or plain):
            pool = figure if (figure and (not plain or rng.random() < FIGURE_ORBIT_P)) else plain
            if not pool:
                break
            oid = pool.pop()
            orbit = orbits[oid]
            k = int(rng.integers(1, len(orbit)))
            picks = [orbit[i] for i in rng.choice(len(orbit), size=k, replace=False)]
            if used + k > limit:
                continue
            on_figure = any(full[r, c] != ground for r, c in orbit)
            for r, c in picks:
                grid[r, c] = unknown
                filled.append((r, c))
            used += k
            if on_figure:
                figure_fills += 1

    return grid, decoys, axis, filled, figure_fills


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> None:
    """Re-solve the visible grid and check every guarantee the design relies on."""
    rows_in = example["input"]
    rows_out = example["output"]
    meta = example["metadata"]
    transform = meta["transform"]
    unknown = int(meta["unknown_color"])
    h, w = len(rows_in), len(rows_in[0])
    if (h, w) != (int(meta["grid_h"]), int(meta["grid_w"])):
        raise ValueError("task 99: metadata size does not match the grid")
    if len(rows_out) != h or any(len(row) != w for row in rows_out):
        raise ValueError("task 99: the output is not the same size as the input")

    solved = solve(rows_in, transform, unknown)
    if solved != rows_out:
        raise ValueError("task 99: the re-solved grid differs from the answer")

    orbits, _ = orbit_partition((h, w), transform)
    for r in range(h):
        for c in range(w):
            if rows_in[r][c] != unknown and rows_out[r][c] != rows_in[r][c]:
                raise ValueError(f"task 99: known cell ({r},{c}) changed")
    if not is_symmetric(rows_out, transform):
        raise ValueError("task 99: the answer is not symmetric under the stated transform")

    n_unknown = sum(row.count(unknown) for row in rows_in)
    n_filled = sum(1 for r in range(h) for c in range(w) if rows_in[r][c] != rows_out[r][c])
    n_decoy = 0
    n_axis = 0
    for orbit in orbits:
        if all(rows_in[r][c] == unknown for r, c in orbit):
            if len(orbit) == 1:
                n_axis += 1
            else:
                n_decoy += 1
        else:
            for r, c in orbit:
                if rows_in[r][c] == unknown and rows_out[r][c] == unknown:
                    raise ValueError(f"task 99: cell ({r},{c}) has a known partner and stayed unknown")
    if n_unknown != int(meta["n_unknown"]):
        raise ValueError("task 99: n_unknown does not match the grid")
    if n_filled != int(meta["n_filled"]):
        raise ValueError("task 99: n_filled does not match the grid")
    if n_decoy != int(meta["n_decoy_orbits"]):
        raise ValueError("task 99: n_decoy_orbits does not match the grid")
    if n_axis != int(meta["n_axis_unknowns"]):
        raise ValueError("task 99: n_axis_unknowns does not match the grid")
    if bool(meta["identity_output"]) != (rows_in == rows_out):
        raise ValueError("task 99: identity_output does not match the grids")
    if n_unknown == 0:
        raise ValueError("task 99: the example has no unknown cell")
    if n_unknown > MAX_UNKNOWN_SHARE * h * w + 1:
        raise ValueError("task 99: too much of the grid is unknown")
    picture = {v for row in rows_out for v in row}
    if unknown in picture and n_decoy + n_axis == 0:
        raise ValueError("task 99: the unknown colour survives without an all-unknown orbit")
    if int(meta["n_colors"]) != len(picture - {unknown}):
        raise ValueError("task 99: n_colors does not match the answer")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = _sample_plan(rng)
        h, w = plan["h"], plan["w"]
        orbits, index = orbit_partition((h, w), plan["transform"])
        bodies = _place_bodies(rng, plan, orbits, index)
        if not bodies:
            continue

        n_tex = plan["n_texture_colors"] if plan["texture_kind"] else 0
        n_body_colors = plan["palette_target"] - 1 - n_tex
        if n_body_colors < 1:
            plan["texture_kind"], plan["n_texture_colors"], n_tex = None, 0, 0
            n_body_colors = plan["palette_target"] - 1
        n_body_colors = max(1, min(n_body_colors, len(bodies)))
        pool = [c for c in range(NUM_COLORS) if c not in (plan["ground"], plan["unknown"])]
        body_colors = [int(v) for v in rng.choice(pool, size=n_body_colors, replace=False)]

        grid = np.full((h, w), plan["ground"], dtype=np.int16)
        order = list(range(len(bodies)))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            color = body_colors[rank] if rank < len(body_colors) else \
                body_colors[int(rng.integers(0, len(body_colors)))]
            bodies[idx]["color"] = color
            for r, c in bodies[idx]["cells"]:
                grid[r, c] = color

        occupied = np.zeros((h, w), dtype=bool)
        for body in bodies:
            for r, c in body["cells"]:
                occupied[r, c] = True
        grid, tex_info = _apply_symmetric_texture(
            rng, grid, plan, occupied, orbits, index,
            exclude=[plan["ground"], plan["unknown"]] + body_colors,
        )

        picture = grid_to_list(grid)
        if not is_symmetric(picture, plan["transform"]):
            continue
        if plan["unknown"] in {v for row in picture for v in row}:
            continue

        masked, decoys, axis, filled, figure_fills = _mask(rng, grid, plan, orbits, index)
        rows_in = grid_to_list(masked)
        n_unknown = sum(row.count(plan["unknown"]) for row in rows_in)
        if n_unknown == 0:
            continue                      # nothing masked at all: not an example
        if not plan["identity"] and (not filled or figure_fills == 0):
            continue

        # The answer is the prompt read literally: everything a known partner
        # determines, and the unknown colour left where a whole orbit is unknown.
        rows_out = solve(rows_in, plan["transform"], plan["unknown"])
        determined = [
            (r, c)
            for orbit in orbits if not all(masked[r, c] == plan["unknown"] for r, c in orbit)
            for r, c in orbit
        ]
        if any(rows_out[r][c] != picture[r][c] for r, c in determined):
            continue
        if bool(plan["identity"]) != (rows_in == rows_out):
            continue

        # naturalness is measured on the picture the solver sees, through the
        # body colours alone (the ground and the texture colours are disjoint
        # from them), so a reviewer can recompute it from the example
        figure_colors = {int(body["color"]) for body in bodies}
        figure_mask = [
            [rows_out[r][c] in figure_colors for c in range(w)]
            for r in range(h)
        ]
        scene_mask = [
            [picture[r][c] in figure_colors for c in range(w)]
            for r in range(h)
        ]
        body_cells = int(occupied.sum())
        identity = rows_in == rows_out
        meta = {
            "grid_h": h,
            "grid_w": w,
            "square": h == w,
            "n_colors": len({v for row in rows_out for v in row} - {plan["unknown"]}),
            "n_scene_colors": len({v for row in picture for v in row}),
            "n_input_colors": len({v for row in rows_in for v in row}),
            "palette_target": plan["palette_target"],
            "transform": plan["transform"],
            "unknown_color": plan["unknown"],
            "ground_color": plan["ground"],
            "max_orbit": MAX_ORBIT.get(plan["transform"], 2),
            "n_unknown": n_unknown,
            "n_filled": sum(1 for r in range(h) for c in range(w) if rows_in[r][c] != rows_out[r][c]),
            "n_decoy_orbits": len(decoys),
            "n_axis_unknowns": len(axis),
            "n_fixed_cells": sum(1 for orbit in orbits if len(orbit) == 1),
            "identity_output": bool(identity),
            "unknown_share": round(n_unknown / plan["area"], 4),
            "mask_fraction_planned": round(plan["mask_fraction"], 4),
            "n_bodies": len(bodies),
            "body_families": sorted({body["family"] for body in bodies}),
            "body_colors": sorted({int(body["color"]) for body in bodies}),
            "body_cells": body_cells,
            "n_self_symmetric_bodies": sum(1 for body in bodies if body["self_symmetric"]),
            "n_figure_components": _components(rows_out, figure_mask),
            "n_scene_components": _components(picture, scene_mask),
            "occupancy": round(body_cells / plan["area"], 4),
            "planned_density": round(plan["density"], 4),
            "texture_group": tex_info["group"] if tex_info else "none",
            "texture_kind": tex_info["kind"] if tex_info else "none",
            "texture_colors": list(tex_info["texture_colors"]) if tex_info else [],
            "texture_coverage": tex_info["coverage"] if tex_info else 0.0,
            "texture_named": False,
            "attempts": attempt,
        }
        example = {"input": rows_in, "output": rows_out, "metadata": meta}
        validate_example(example)
        return example
    raise RuntimeError(f"task 99: no valid scene after {MAX_ATTEMPTS} attempts")


@register("99")
def generate_complete_symmetry(rng: np.random.Generator) -> dict:
    """Task 99: fill the unknown cells so the whole grid obeys the stated symmetry."""
    return _example(rng)
