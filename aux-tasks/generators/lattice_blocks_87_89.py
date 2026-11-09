"""Lattice-block tasks 87, 88 and 89 (September 2026 rebuild).

The three tasks share one lattice family, broadened from the task 14 work.

Lattice
-------
Straight lines of a single colour run across complete rows and complete columns of
the canvas; the rectangular areas between them are the *blocks*.

* Row spacing and column spacing are drawn INDEPENDENTLY (2..8), so blocks are
  usually not square.
* Each axis has its own PHASE: the first line sits at offset 0..spacing-1 (offset 0
  in ~40% of axes), so the lattice can be cut short on ANY side.  A positive phase
  leaves a clipped block row above the first line / a clipped block column left of
  it, exactly like the clipped band the far edge leaves.  Clipped edge blocks COUNT
  as blocks and every prompt says so.
* A minority of axes (~18% of draws) use IRREGULAR spacing: the gap between
  consecutive lines is redrawn per line, so the lattice is not periodic.  The prompt
  never promises periodicity and the block decomposition (maximal runs of non-line
  rows and columns) does not use it.
* Lines are 1..3 cells thick (1 usually, 3 rarely, task 14's ``thick`` kind) and
  always thinner than every gap, so two line groups never merge and no block is
  empty.  At least one complete row line and one complete column line always lie
  inside the canvas.
* No dotted lines and no overlaid lattices: both make "block" ill-defined.
* Canvas sides are drawn FIRST, independently and uniformly over 5..30 (task 88
  additionally rejects the single 5x5 pair, see ``generate_fill_between``), so the
  per-side density is flat, squares only happen by coincidence and H*W reaches 900.

Recovery (what the validators do, and what a solver can do)
-----------------------------------------------------------
The lattice is recovered WITHOUT an anchor: the line rows are exactly the monochrome
rows of the grid and the line columns exactly the monochrome columns.  No block
content ever uses the line colour, so no line row/column can be missed; and a
monochrome row of a NON-line colour cannot exist because at least one complete
column line crosses every row (symmetrically for columns), so no forged line can be
found either.  ``lattice_is_clean`` re-checks that belt-and-braces on every grid.
The blocks are the maximal runs of non-line rows crossed with the maximal runs of
non-line columns.  ``recover_lattice`` implements this and every validator re-solves
the task from the FINAL visible grid through it, raising ``ValueError`` on any
mismatch.

Tasks
-----
87  Paint the block at a given 0-based (block row, block column) index.  The target
    block is decorated and textured like every other block, so the index is the only
    route to it; the target is never the unique all-background block.
88  Every marked block (a block filled completely with one colour that is not the
    background) propagates its colour to the blocks strictly between it and another
    marked block of the same colour on the same block row or block column.  Row and
    column groups mix inside one grid, groups that fill nothing (adjacent markers,
    unaligned markers, a lone marker) occur, and ~10% of grids are identities.  Each
    group owns its own colour and a block set disjoint from every other group, so
    the rule never cascades and two fills never compete for one block.  The COLOUR
    clause is made load-bearing on purpose: groups prefer lanes another group's
    markers already use, and ~40% of grids also park a lone off-span marker of a
    second colour in such a lane, so a reader that fills between marked blocks
    regardless of colour fills a span the rule leaves alone (it reproduces the
    output on ~31% of grids; ``color_clause_matters`` records this per example).
    The group count and the markers per group follow the block count, so a 50-block
    lattice is as busy as a 12-block one.  The background plurality is certified
    three times: over block cells, over completely filled blocks, and over block
    cells once 8-isolated single cells are set aside (``background_margin_isolated``,
    the suite's standing texture convention, which a ground living in 1x1 and 1x2
    edge blocks can otherwise lose).
89  Count the blocks that are completely filled with a named colour.  Blocks that
    contain that colour without being filled with it occur in ~63% of grids and
    their number follows the block count (up to n_blocks / 3), which is what makes
    "completely" load-bearing: counting every block that CONTAINS the colour is
    right on ~37% of grids.

Backgrounds follow the shared policy: sparse textures from ``common.backgrounds``
inside background blocks only (never on the lattice lines, never the line / marker /
target / paint colours), and no prompt names or defines the background or a texture.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from common.backgrounds import apply_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list, make_grid, pick_colors

SIDE_MIN, SIDE_MAX = 5, 30
SPACINGS = (2, 3, 4, 5, 6, 7, 8)
SPACING_WEIGHTS = (0.12, 0.20, 0.20, 0.16, 0.13, 0.10, 0.09)
THICKNESSES = (1, 2, 3)
THICKNESS_WEIGHTS = (0.72, 0.22, 0.06)
PHASE_ZERO_P = 0.40     # share of axes whose first line sits at offset 0
IRREGULAR_P = 0.18      # share of lattices that try non-periodic line positions
PALETTE_SIZES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
PALETTE_WEIGHTS = (0.13, 0.13, 0.12, 0.12, 0.11, 0.10, 0.10, 0.10, 0.09)
TEXTURE_P = 0.35
MAX_ATTEMPTS = 60
_SIZE_RUN = 20          # task 88 keeps one canvas for this many scene attempts


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


def _weighted(rng: np.random.Generator, values, weights):
    p = np.asarray(weights, dtype=float)
    return values[int(rng.choice(len(values), p=p / p.sum()))]


def _palette_size(rng: np.random.Generator, floor: int = 2) -> int:
    """Near-flat draw over ``floor``..10 (8-10 together ~29%).

    Tasks with a structural colour floor (88 needs a line, a background and a marker;
    89 needs a line, a background and the target) sample from their floor upwards
    instead of piling up on it.
    """
    keep = [(v, p) for v, p in zip(PALETTE_SIZES, PALETTE_WEIGHTS) if v >= floor]
    return _weighted(rng, [v for v, _ in keep], [p for _, p in keep])


# ---------------------------------------------------------------------------
# lattice geometry
# ---------------------------------------------------------------------------

def bands_from_starts(n: int, starts, thickness: int):
    """Line indices, block bands and per-band clipping for one axis of length ``n``.

    ``starts`` is the increasing list of line START indices; its LAST entry is a
    sentinel that may lie at or beyond ``n`` (it says where the next line would have
    gone, which is what decides whether the final band is cut short).  A line covers
    ``thickness`` indices; a band is the half-open run between one line and the next.
    A band is *clipped* when an edge of the canvas cuts it short -- the near edge for
    the band above/left of the first line, the far edge for a band whose next line
    would have fallen outside the canvas.
    """
    lines: list[int] = []
    bands: list[tuple[int, int]] = []
    clipped: list[bool] = []
    real = [int(s) for s in starts if 0 <= int(s) < n]
    if not real:
        return lines, bands, clipped
    if real[0] > 0:
        bands.append((0, real[0]))
        clipped.append(True)
    for i, start in enumerate(real):
        lines.extend(range(start, min(start + thickness, n)))
        nxt = int(starts[i + 1]) if i + 1 < len(starts) else n + 1
        lo, hi = start + thickness, min(nxt, n)
        if lo < hi:
            bands.append((lo, hi))
            clipped.append(nxt > n)
    return lines, bands, clipped


def axis_starts(n: int, spacing: int, phase: int) -> list[int]:
    """Periodic line starts from ``phase``, plus the out-of-canvas sentinel."""
    starts = list(range(int(phase), n, int(spacing)))
    starts.append((starts[-1] if starts else int(phase)) + int(spacing))
    return starts


def axis_bands(n: int, spacing: int, thickness: int, phase: int = 0):
    """The periodic form of :func:`bands_from_starts`."""
    return bands_from_starts(n, axis_starts(n, spacing, phase), thickness)


def _phase_weight(phase: int, spacing: int) -> float:
    if spacing <= 1:
        return 1.0
    return PHASE_ZERO_P if phase == 0 else (1.0 - PHASE_ZERO_P) / (spacing - 1)


@lru_cache(maxsize=None)
def _axis_options(n: int, thickness: int) -> tuple:
    """Every feasible (spacing, phase) of one axis, with its nominal weight.

    Cached: the geometry of an axis depends only on its length and the line
    thickness, so the enumeration costs nothing after the first few seeds.
    """
    out: list[tuple] = []
    for spacing, w_s in zip(SPACINGS, SPACING_WEIGHTS):
        if thickness >= spacing:
            continue
        for phase in range(spacing):
            if phase >= n:
                continue
            starts = axis_starts(n, spacing, phase)
            lines, bands, clipped = bands_from_starts(n, starts, thickness)
            if not lines or not bands:
                continue
            out.append((int(spacing), int(phase), tuple(starts), tuple(bands),
                        tuple(clipped), w_s * _phase_weight(phase, spacing)))
    return tuple(out)


@lru_cache(maxsize=None)
def _axis_groups(n: int, thickness: int) -> tuple:
    """The axis options of ``n`` grouped by how many blocks they produce."""
    groups: dict = {}
    for option in _axis_options(n, thickness):
        entry = groups.setdefault(len(option[3]), [0.0, []])
        entry[0] += option[5]
        entry[1].append(option)
    return tuple(sorted((k, v[0], tuple(v[1])) for k, v in groups.items()))


def _pick(rng: np.random.Generator, items, weights):
    p = np.asarray(weights, dtype=float)
    return items[int(rng.choice(len(items), p=p / p.sum()))]


def _axis_spec(option: tuple, regular: bool) -> dict:
    spacing, phase, starts, bands, clipped, _ = option
    near = bands[0][0] == 0 and phase > 0
    far = bool(clipped) and clipped[-1] and len(bands) > (1 if near else 0)
    return {
        "spacing": int(spacing),
        "phase": int(phase),
        "starts": [int(s) for s in starts[:-1]],
        "bands": [tuple(b) for b in bands],
        "clipped": [bool(c) for c in clipped],
        "regular": bool(regular),
        "near_clipped": bool(near),
        "far_clipped": bool(far),
    }


def _sample_irregular_axis(rng: np.random.Generator, n: int, thickness: int) -> Optional[tuple]:
    """One axis whose gaps are redrawn per line (non-periodic line positions)."""
    spacings = [s for s in SPACINGS if s > thickness]
    weights = [w for s, w in zip(SPACINGS, SPACING_WEIGHTS) if s > thickness]
    if not spacings:
        return None
    first = int(_pick(rng, spacings, weights))
    phase = 0 if rng.random() < PHASE_ZERO_P else _rint(rng, 1, first - 1)
    if phase >= n:
        return None
    starts = [phase]
    pos = phase
    while pos < n:
        pos += int(_pick(rng, spacings, weights))
        starts.append(pos)
    visible = [s for s in starts if s < n]
    gaps = [b - a for a, b in zip(visible, visible[1:])]
    if len(visible) < 3 or len(set(gaps)) < 2:
        return None      # the part inside the canvas came out periodic after all
    lines, bands, clipped = bands_from_starts(n, starts, thickness)
    if not lines or not bands:
        return None
    return (min(gaps), phase, tuple(starts), tuple(bands), tuple(clipped), 1.0)


def _feasible_thicknesses(h: int, w: int):
    items, weights = [], []
    for thickness, w_t in zip(THICKNESSES, THICKNESS_WEIGHTS):
        if _axis_options(h, thickness) and _axis_options(w, thickness):
            items.append(thickness)
            weights.append(w_t)
    return items, weights


def _pick_regular_axis(rng: np.random.Generator, n: int, thickness: int) -> Optional[tuple]:
    options = _axis_options(n, thickness)
    if not options:
        return None
    return _pick(rng, options, [o[5] for o in options])


def _irregular_lattice(rng: np.random.Generator, h: int, w: int,
                       min_blocks: int, min_line_blocks: int) -> Optional[dict]:
    items, weights = _feasible_thicknesses(h, w)
    if not items:
        return None
    for _ in range(8):
        thickness = int(_pick(rng, items, weights))
        row_irregular = rng.random() < 0.7
        col_irregular = (not row_irregular) or rng.random() < 0.5
        row = (_sample_irregular_axis(rng, h, thickness) if row_irregular
               else _pick_regular_axis(rng, h, thickness))
        col = (_sample_irregular_axis(rng, w, thickness) if col_irregular
               else _pick_regular_axis(rng, w, thickness))
        if row is None or col is None:
            continue
        if len(row[3]) * len(col[3]) < min_blocks:
            continue
        if max(len(row[3]), len(col[3])) < min_line_blocks:
            continue
        return _lattice_spec(h, w, thickness, row, col, not row_irregular, not col_irregular)
    return None


def _lattice_spec(h: int, w: int, thickness: int, row: tuple, col: tuple,
                  row_regular: bool, col_regular: bool) -> dict:
    rows = _axis_spec(row, row_regular)
    cols = _axis_spec(col, col_regular)
    return {
        "grid_h": int(h),
        "grid_w": int(w),
        "thickness": int(thickness),
        "rows": rows,
        "cols": cols,
        "row_spacing": rows["spacing"],
        "col_spacing": cols["spacing"],
        "row_phase": rows["phase"],
        "col_phase": cols["phase"],
        "row_bands": rows["bands"],
        "col_bands": cols["bands"],
        "row_clipped": rows["clipped"],
        "col_clipped": cols["clipped"],
    }


@lru_cache(maxsize=None)
def _lattice_combos(h: int, w: int, min_blocks: int, min_line_blocks: int) -> tuple:
    """Every feasible (thickness, row option group, column option group) with weights.

    The joint constraint only looks at how many blocks each axis produces, and the
    weights factorise, so the axis options are grouped by block count first: the cross
    product stays in the hundreds instead of the tens of thousands a full (thickness,
    spacing, phase, spacing, phase) enumeration would need.  Cached per canvas.
    """
    combos: list[tuple] = []
    weights: list[float] = []
    for thickness, w_t in zip(THICKNESSES, THICKNESS_WEIGHTS):
        for n_row, w_r, row_options in _axis_groups(h, thickness):
            for n_col, w_c, col_options in _axis_groups(w, thickness):
                if n_row * n_col < min_blocks:
                    continue
                if max(n_row, n_col) < min_line_blocks:
                    continue
                combos.append((thickness, row_options, col_options))
                weights.append(w_t * w_r * w_c)
    return tuple(combos), tuple(weights)


def sample_lattice(
    rng: np.random.Generator,
    h: int,
    w: int,
    *,
    min_blocks: int = 1,
    min_line_blocks: int = 1,
) -> Optional[dict]:
    """Draw line positions and a thickness for a canvas that is already fixed.

    The periodic path enumerates every feasible (thickness, spacing, phase) pair per
    axis -- cached, and grouped by block count so only the joint block-count pairs
    have to be weighed -- and draws one with the nominal weights renormalised over
    what fits.  Rejection sampling would have thrown the canvas away whenever the
    first few draws missed, which put a visible dip on the small sides; the size
    stays exactly as drawn here.  A minority of calls try a non-periodic lattice
    first and fall back to the periodic enumeration when it does not fit.
    """
    if rng.random() < IRREGULAR_P:
        spec = _irregular_lattice(rng, h, w, min_blocks, min_line_blocks)
        if spec is not None:
            return spec
    combos, weights = _lattice_combos(h, w, min_blocks, min_line_blocks)
    if not combos:
        return None
    thickness, row_options, col_options = _pick(rng, combos, weights)
    row = _pick(rng, row_options, [o[5] for o in row_options])
    col = _pick(rng, col_options, [o[5] for o in col_options])
    return _lattice_spec(h, w, thickness, row, col, True, True)


def block_box(spec: dict, br: int, bc: int) -> tuple[int, int, int, int]:
    r0, r1 = spec["row_bands"][br]
    c0, c1 = spec["col_bands"][bc]
    return r0, r1, c0, c1


def block_area(spec: dict, br: int, bc: int) -> int:
    r0, r1, c0, c1 = block_box(spec, br, bc)
    return (r1 - r0) * (c1 - c0)


def all_blocks(spec: dict) -> list[tuple[int, int]]:
    return [(r, c) for r in range(len(spec["row_bands"])) for c in range(len(spec["col_bands"]))]


def build_lattice(spec: dict, line_color: int, bg_color: int) -> np.ndarray:
    """Line colour everywhere, then every block set to the background colour."""
    grid = make_grid(spec["grid_h"], spec["grid_w"], fill=int(line_color))
    for r0, r1 in spec["row_bands"]:
        for c0, c1 in spec["col_bands"]:
            grid[r0:r1, c0:c1] = int(bg_color)
    return grid


def clipped_block_count(spec: dict) -> int:
    return sum(
        1
        for br in range(len(spec["row_bands"]))
        for bc in range(len(spec["col_bands"]))
        if spec["row_clipped"][br] or spec["col_clipped"][bc]
    )


def _runs(n: int, drop: set) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i in range(n):
        if i in drop:
            if start is not None:
                runs.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        runs.append((start, n))
    return runs


def monochrome_lines(grid) -> tuple[list[int], list[int], set]:
    """The monochrome rows, the monochrome columns and the colours they use."""
    g = np.asarray(grid)
    mono_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == g[r, 0]))]
    mono_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == g[0, c]))]
    colors = {int(g[r, 0]) for r in mono_rows} | {int(g[0, c]) for c in mono_cols}
    return mono_rows, mono_cols, colors


def recover_lattice(grid) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Recover (line colour, row bands, column bands) from a visible grid alone.

    No anchor: the lattice lines are exactly the monochrome rows and columns.  Block
    content never uses the line colour, so every line is found; and every row meets a
    complete column line (and every column a complete row line), so a monochrome row
    or column of any other colour cannot exist -- which is re-checked here.
    """
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("lattice tasks: the grid is not a rectangle")
    h, w = g.shape
    mono_rows, mono_cols, colors = monochrome_lines(g)
    if not mono_rows or not mono_cols:
        raise ValueError("lattice tasks: the grid has no complete row line "
                         "or no complete column line")
    if len(colors) != 1:
        raise ValueError("lattice tasks: the complete rows and columns are not all one colour")
    line_color = int(next(iter(colors)))
    row_bands = _runs(h, set(mono_rows))
    col_bands = _runs(w, set(mono_cols))
    if not row_bands or not col_bands:
        raise ValueError("lattice tasks: the recovered lattice has no blocks")
    return line_color, row_bands, col_bands


def lattice_is_clean(grid, line_color: int) -> bool:
    """No forged line: no monochrome row or column of a colour other than the line's.

    True by construction (at least one complete line crosses every row and every
    column), and checked anyway on every generated grid.
    """
    _, _, colors = monochrome_lines(grid)
    return colors == {int(line_color)}


def certify_lattice(grid, spec: dict, line_color: int) -> None:
    """The visible grid must hand back exactly the lattice that was drawn."""
    found, rows, cols = recover_lattice(grid)
    if found != int(line_color):
        raise ValueError("lattice tasks: the line colour is not recoverable")
    if rows != [tuple(b) for b in spec["row_bands"]] or cols != [tuple(b) for b in spec["col_bands"]]:
        raise ValueError("lattice tasks: the visible blocks differ from the drawn lattice")
    if not lattice_is_clean(grid, line_color):
        raise ValueError("lattice tasks: a complete row or column forges a lattice line")


# ---------------------------------------------------------------------------
# block content
# ---------------------------------------------------------------------------

def partial_mask(rng: np.random.Generator, bh: int, bw: int) -> Optional[np.ndarray]:
    """A clean, proper, non-empty rectangular subset of a bh x bw block."""
    if bh * bw < 2:
        return None
    for _ in range(12):
        kind = _weighted(rng, ("subrect", "half", "bar"), (0.5, 0.3, 0.2))
        mask = np.zeros((bh, bw), dtype=bool)
        if kind == "subrect":
            hh, ww = _rint(rng, 1, bh), _rint(rng, 1, bw)
            r0, c0 = _rint(rng, 0, bh - hh), _rint(rng, 0, bw - ww)
            mask[r0:r0 + hh, c0:c0 + ww] = True
        elif kind == "half":
            if bh >= 2 and (bw < 2 or rng.random() < 0.5):
                mask[:bh // 2, :] = True
            elif bw >= 2:
                mask[:, :bw // 2] = True
            if mask.any() and rng.random() < 0.5:
                mask = ~mask
        else:
            if bh >= 2 and (bw < 2 or rng.random() < 0.5):
                mask[_rint(rng, 0, bh - 1), :] = True
            elif bw >= 2:
                mask[:, _rint(rng, 0, bw - 1)] = True
        if mask.any() and not mask.all():
            return mask
    return None


def fill_block(grid: np.ndarray, spec: dict, br: int, bc: int, color: int) -> None:
    r0, r1, c0, c1 = block_box(spec, br, bc)
    grid[r0:r1, c0:c1] = int(color)


def paint_partial(rng: np.random.Generator, grid: np.ndarray, spec: dict, br: int, bc: int,
                  color: int, base: Optional[int] = None) -> bool:
    """Paint a proper part of a block; ``base`` repaints the rest of it first."""
    r0, r1, c0, c1 = block_box(spec, br, bc)
    mask = partial_mask(rng, r1 - r0, c1 - c0)
    if mask is None:
        return False
    if base is not None:
        grid[r0:r1, c0:c1] = int(base)
    view = grid[r0:r1, c0:c1]
    view[mask] = int(color)
    return True


def _uniform_block_colors(grid: np.ndarray, rows, cols) -> dict:
    """Per block: its single colour when it is filled completely, else None."""
    out: dict = {}
    for i, (r0, r1) in enumerate(rows):
        for j, (c0, c1) in enumerate(cols):
            view = grid[r0:r1, c0:c1]
            first = int(view.flat[0])
            out[(i, j)] = first if bool(np.all(view == first)) else None
    return out


def block_colors(grid: np.ndarray, spec: dict) -> dict:
    return _uniform_block_colors(grid, spec["row_bands"], spec["col_bands"])


def block_plurality(grid: np.ndarray, spec: dict, bg_color: int) -> float:
    """Background blocks divided by the commonest other completely filled colour."""
    counts: dict = {}
    for color in block_colors(grid, spec).values():
        if color is None:
            continue
        counts[color] = counts.get(color, 0) + 1
    ground = counts.get(int(bg_color), 0)
    rival = max((n for c, n in counts.items() if c != int(bg_color)), default=0)
    return 99.0 if rival == 0 else round(ground / rival, 4)


def block_cell_mask(shape, rows, cols) -> np.ndarray:
    """True on every cell that belongs to a block (the lattice lines are False)."""
    row_mask = np.zeros(int(shape[0]), dtype=bool)
    col_mask = np.zeros(int(shape[1]), dtype=bool)
    for r0, r1 in rows:
        row_mask[r0:r1] = True
    for c0, c1 in cols:
        col_mask[c0:c1] = True
    return row_mask[:, None] & col_mask[None, :]


def _plurality(values, bg_color: int) -> float:
    counts = np.bincount(np.asarray(values, dtype=np.int64).ravel(), minlength=NUM_COLORS)
    ground = int(counts[int(bg_color)]) if int(bg_color) < counts.size else 0
    if not ground:
        return 0.0
    counts[int(bg_color)] = 0
    rival = int(counts.max())
    return 99.0 if rival == 0 else round(ground / rival, 4)


def cell_plurality(grid: np.ndarray, spec: dict, bg_color: int) -> float:
    """Background cells divided by the commonest other colour, over block cells only."""
    g = np.asarray(grid)
    return _plurality(g[block_cell_mask(g.shape, spec["row_bands"], spec["col_bands"])],
                      bg_color)


def isolated_cell_plurality(grid, rows, cols, bg_color: int) -> float:
    """Block-cell plurality for a reader that first sets aside isolated single cells.

    The suite's standing convention ("isolated single cells, whatever their colour,
    are background texture") removes every cell with no 8-neighbour of its own
    colour.  A ground that survives only in 1x1 and 1x2 edge blocks loses those cells
    to that convention, so the margin is certified after the removal as well.
    """
    g = np.asarray(grid)
    keep = block_cell_mask(g.shape, rows, cols) & ~isolated_cell_mask(g)
    return _plurality(g[keep], bg_color)


def add_texture(
    rng: np.random.Generator,
    grid: np.ndarray,
    spec: dict,
    bg_color: int,
    exclude,
    free_blocks,
    wanted: bool = True,
) -> tuple[np.ndarray, dict]:
    """Sparse texture inside the given background blocks only (never on the lines).

    Blocks of a single cell are left out: one isolated texture cell would fill them
    and turn a background block into a completely filled one.
    """
    empty = {"group": None, "kind": None, "colors": [], "coverage": 0.0}
    if not wanted:
        return grid, empty
    free = np.zeros(grid.shape, dtype=bool)
    for br, bc in free_blocks:
        if block_area(spec, br, bc) < 2:
            continue
        r0, r1, c0, c1 = block_box(spec, br, bc)
        free[r0:r1, c0:c1] = True
    if not free.any():
        return grid, empty
    result = apply_background(
        rng, grid, int(bg_color), ~free, group="sparse", exclude=sorted({int(c) for c in exclude})
    )
    if result is None:
        return grid, empty
    display, info = result
    for br, bc in free_blocks:
        r0, r1, c0, c1 = block_box(spec, br, bc)
        view = display[r0:r1, c0:c1]
        if bool(np.all(view == view.flat[0])) and int(view.flat[0]) != int(bg_color):
            return grid, empty
    return display, {
        "group": info["group"],
        "kind": info["kind"],
        "colors": [int(c) for c in info["texture_colors"]],
        "coverage": round(float(info["coverage"]), 4),
    }


def clipped_sides(spec: dict) -> list[str]:
    sides = []
    if spec["rows"]["near_clipped"]:
        sides.append("top")
    if spec["rows"]["far_clipped"]:
        sides.append("bottom")
    if spec["cols"]["near_clipped"]:
        sides.append("left")
    if spec["cols"]["far_clipped"]:
        sides.append("right")
    return sides


def _common_meta(spec: dict, grid: np.ndarray, line_color: int, bg_color: int,
                 texture: dict, attempts: int) -> dict:
    return {
        "grid_h": int(spec["grid_h"]),
        "grid_w": int(spec["grid_w"]),
        "n_colors": int(len(np.unique(grid))),
        "row_spacing": int(spec["row_spacing"]),
        "col_spacing": int(spec["col_spacing"]),
        "row_phase": int(spec["row_phase"]),
        "col_phase": int(spec["col_phase"]),
        "row_regular": bool(spec["rows"]["regular"]),
        "col_regular": bool(spec["cols"]["regular"]),
        "row_line_starts": [int(s) for s in spec["rows"]["starts"]],
        "col_line_starts": [int(s) for s in spec["cols"]["starts"]],
        "clipped_sides": clipped_sides(spec),
        "line_thickness": int(spec["thickness"]),
        "block_rows": int(len(spec["row_bands"])),
        "block_cols": int(len(spec["col_bands"])),
        "n_blocks": int(len(spec["row_bands"]) * len(spec["col_bands"])),
        "clipped_blocks": int(clipped_block_count(spec)),
        "line_color": int(line_color),
        "bg_color": int(bg_color),
        "background_margin": float(block_plurality(grid, spec, bg_color)),
        "background_cell_margin": float(cell_plurality(grid, spec, bg_color)),
        "texture_group": texture["group"],
        "texture_kind": texture["kind"],
        "texture_colors": [int(c) for c in texture["colors"]],
        "texture_coverage": float(texture["coverage"]),
        "texture_named": False,
        "attempts": int(attempts),
    }


# ---------------------------------------------------------------------------
# Task 87 — paint one block by its 0-based (block row, block column) index
# ---------------------------------------------------------------------------

def generate_paint_block(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        h, w = _rint(rng, SIDE_MIN, SIDE_MAX), _rint(rng, SIDE_MIN, SIDE_MAX)
        spec = sample_lattice(rng, h, w, min_blocks=4)
        if spec is None:
            continue
        blocks = all_blocks(spec)
        n_blocks = len(blocks)

        use_texture = min(h, w) >= 8 and rng.random() < TEXTURE_P
        want = _palette_size(rng)
        reuse = rng.random() < 0.4          # the paint colour is already on the canvas
        n_fill = max(0, want - 2 - (1 if use_texture else 0))
        n_fill = max(0, min(n_fill, n_blocks - 2, 8 if reuse else 7))
        palette = pick_colors(rng, 2 + n_fill + (0 if (reuse and n_fill) else 1))
        line_color, bg_color = int(palette[0]), int(palette[1])
        fill_colors = [int(c) for c in palette[2:2 + n_fill]]
        paint_color = (
            int(fill_colors[_rint(rng, 0, n_fill - 1)])
            if (reuse and n_fill)
            else int(palette[-1])
        )

        target = blocks[int(rng.integers(0, n_blocks))]
        n_decorated = 0 if n_fill == 0 else _rint(rng, n_fill, max(n_fill, int(0.55 * n_blocks)))
        n_decorated = max(0, min(n_decorated, n_blocks - 2))
        order = list(rng.permutation(n_blocks))
        decorated = [blocks[int(i)] for i in order[:n_decorated]]
        ground = [b for b in blocks if b not in set(decorated)]

        grid = build_lattice(spec, line_color, bg_color)
        assigned: list[int] = []
        if n_fill:
            assigned = list(fill_colors)
            while len(assigned) < n_decorated:
                assigned.append(int(fill_colors[int(rng.integers(0, n_fill))]))
            rng.shuffle(assigned)  # type: ignore[arg-type]
        for block, color in zip(decorated, assigned):
            if block == target and color == paint_color and n_fill > 1:
                color = int(next(c for c in fill_colors if c != paint_color))
            if rng.random() < 0.3 and paint_partial(rng, grid, spec, block[0], block[1], color):
                continue
            if block == target and color == paint_color:
                # a solid pre-painted target would make the edit invisible
                paint_partial(rng, grid, spec, block[0], block[1], color)
                continue
            fill_block(grid, spec, block[0], block[1], color)

        texture_exclude = {line_color, bg_color, paint_color, *fill_colors}
        grid, texture = add_texture(rng, grid, spec, bg_color, texture_exclude, ground, use_texture)

        colors = block_colors(grid, spec)
        plain = [b for b, c in colors.items() if c == bg_color]
        if target in plain and len(plain) == 1:
            others = [b for b in ground if b != target]
            if not others:
                continue
            other = others[int(rng.integers(0, len(others)))]
            fill_block(grid, spec, other[0], other[1], bg_color)
            colors = block_colors(grid, spec)
            plain = [b for b, c in colors.items() if c == bg_color]
            texture = dict(texture, colors=[c for c in texture["colors"] if bool((grid == c).any())])
            if not texture["colors"]:
                texture = {"group": None, "kind": None, "colors": [], "coverage": 0.0}
        if colors[target] == paint_color:
            continue

        out = grid.copy()
        fill_block(out, spec, target[0], target[1], paint_color)

        meta = _common_meta(spec, grid, line_color, bg_color, texture, attempt)
        r0, r1, c0, c1 = block_box(spec, target[0], target[1])
        meta.update({
            "paint_color": int(paint_color),
            "target_row": int(target[0]),
            "target_col": int(target[1]),
            "target_is_clipped": bool(spec["row_clipped"][target[0]] or spec["col_clipped"][target[1]]),
            "target_block_h": int(r1 - r0),
            "target_block_w": int(c1 - c0),
            "n_plain_blocks": int(len(plain)),
            "n_decorated_blocks": int(n_decorated),
        })
        example = {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}
        validate_paint_block(example)
        return example
    raise RuntimeError("task 87: no lattice after the retry budget")


def solve_paint_block(grid, target_row: int, target_col: int, paint_color: int) -> np.ndarray:
    """Independent reading of task 87's rule from a visible grid."""
    g = np.asarray(grid)
    _, rows, cols = recover_lattice(g)
    if not (0 <= target_row < len(rows)) or not (0 <= target_col < len(cols)):
        raise ValueError("task 87: the target block index is outside the recovered lattice")
    r0, r1 = rows[target_row]
    c0, c1 = cols[target_col]
    out = g.copy()
    out[r0:r1, c0:c1] = int(paint_color)
    return out


def _validated_lattice(example: dict, task: str):
    """Anchor-free recovery plus the lattice guarantees, shared by all three tasks."""
    meta = example["metadata"]
    inp = np.asarray(example["input"])
    line_color, rows, cols = recover_lattice(inp)
    if line_color != int(meta["line_color"]):
        raise ValueError(f"task {task}: the line colour is not recoverable")
    if len(rows) != int(meta["block_rows"]) or len(cols) != int(meta["block_cols"]):
        raise ValueError(f"task {task}: the recovered block grid differs from the metadata")
    if not lattice_is_clean(inp, line_color):
        raise ValueError(f"task {task}: a complete row or column forges a lattice line")
    thickness = int(meta["line_thickness"])
    for n, starts, bands, phase, name in (
        (inp.shape[0], meta["row_line_starts"], rows, meta["row_phase"], "row"),
        (inp.shape[1], meta["col_line_starts"], cols, meta["col_phase"], "column"),
    ):
        drawn: set = set()
        for start in starts:
            drawn.update(range(int(start), min(int(start) + thickness, n)))
        if not drawn or _runs(n, drawn) != bands:
            raise ValueError(f"task {task}: the {name} lines in the metadata "
                             "do not give the visible blocks")
        if int(phase) != min(int(s) for s in starts):
            raise ValueError(f"task {task}: the {name} phase differs from the first line")
    return inp, line_color, rows, cols


def validate_paint_block(example: dict) -> None:
    meta = example["metadata"]
    out = np.asarray(example["output"])
    inp, line_color, rows, cols = _validated_lattice(example, "87")
    expected = solve_paint_block(inp, int(meta["target_row"]), int(meta["target_col"]),
                                 int(meta["paint_color"]))
    if not np.array_equal(expected, out):
        raise ValueError("task 87: the output is not the input with the indexed block painted")
    if np.array_equal(inp, out):
        raise ValueError("task 87: the painted block was already that colour")
    r0, r1 = rows[int(meta["target_row"])]
    c0, c1 = cols[int(meta["target_col"])]
    if int(meta["target_block_h"]) != r1 - r0 or int(meta["target_block_w"]) != c1 - c0:
        raise ValueError("task 87: the target block shape differs from the metadata")
    plain = sum(1 for color in _uniform_block_colors(inp, rows, cols).values()
                if color == int(meta["bg_color"]))
    if plain != int(meta["n_plain_blocks"]):
        raise ValueError("task 87: the visible all-background blocks differ from the metadata")
    if plain == 1 and bool(np.all(inp[r0:r1, c0:c1] == int(meta["bg_color"]))):
        raise ValueError("task 87: the target is the only all-background block")
    changed = np.argwhere(inp != out)
    if changed.size and (changed[:, 0].min() < r0 or changed[:, 0].max() >= r1
                         or changed[:, 1].min() < c0 or changed[:, 1].max() >= c1):
        raise ValueError("task 87: cells outside the target block changed")


# ---------------------------------------------------------------------------
# Task 88 — propagate marker colours between marked blocks
# ---------------------------------------------------------------------------

_NOOP_KINDS = ("adjacent", "unaligned", "single")
_NOOP_WEIGHTS = (0.40, 0.35, 0.25)
# one colour per group, plus the line and the background colour: 8 groups is the cap
_MAX_GROUPS = 8
_NOOP_COUNT_WEIGHTS = (0.70, 0.24, 0.06)
_DECOY_P = 0.40         # a second colour deliberately parked in a lane already in use
_CROSS_P = 0.75         # how often a group prefers a placement the colour clause decides
_MIN_GROUND_BLOCKS = 3


def _max_markers(n_blocks: int) -> int:
    """2..4 markers per group, following the block count of the lattice."""
    return 2 + (1 if n_blocks >= 15 else 0) + (1 if n_blocks >= 28 else 0)


def _sample_group(rng: np.random.Generator, n_rows: int, n_cols: int, kind: str,
                  max_markers: int = 3, long_spans: bool = False) -> Optional[dict]:
    if kind in ("row", "col"):
        along = n_cols if kind == "row" else n_rows
        across = n_rows if kind == "row" else n_cols
        if along < 3:
            return None
        line = _rint(rng, 0, across - 1)
        # a span covers 3..along blocks; on a lattice with many blocks the long half
        # is favoured, so a big canvas carries long spans rather than more short ones
        span = _rint(rng, 3, along)
        if long_spans and along >= 5 and rng.random() < 0.55:
            span = _rint(rng, (along + 3) // 2, along)
        first = _rint(rng, 0, along - span)
        last = first + span - 1
        interior = list(range(first + 1, last))
        extra: list[int] = []
        while (len(extra) + 2 < max_markers and len(interior) - len(extra) >= 2
               and rng.random() < 0.45):
            free = [p for p in interior if p not in extra]
            extra.append(int(free[int(rng.integers(0, len(free)))]))
        markers = sorted({first, last, *extra})
        fills = [p for p in interior if p not in extra]
        if not fills:
            return None
        to_cell = (lambda p: (line, p)) if kind == "row" else (lambda p: (p, line))
        return {
            "kind": kind,
            "orientation": "row" if kind == "row" else "column",
            "markers": [to_cell(p) for p in markers],
            "fills": [to_cell(p) for p in fills],
            "blocks": {to_cell(p) for p in range(first, last + 1)},
        }
    if kind == "adjacent":
        if n_cols >= 2 and (n_rows < 2 or rng.random() < 0.5):
            r, c = _rint(rng, 0, n_rows - 1), _rint(rng, 0, n_cols - 2)
            markers = [(r, c), (r, c + 1)]
        elif n_rows >= 2:
            r, c = _rint(rng, 0, n_rows - 2), _rint(rng, 0, n_cols - 1)
            markers = [(r, c), (r + 1, c)]
        else:
            return None
    elif kind == "unaligned":
        if n_rows < 2 or n_cols < 2:
            return None
        r1, r2 = rng.choice(n_rows, size=2, replace=False)
        c1, c2 = rng.choice(n_cols, size=2, replace=False)
        markers = [(int(r1), int(c1)), (int(r2), int(c2))]
    else:
        markers = [(_rint(rng, 0, n_rows - 1), _rint(rng, 0, n_cols - 1))]
    return {
        "kind": kind,
        "orientation": "none",
        "markers": markers,
        "fills": [],
        "blocks": set(markers),
    }


def _crosses(markers, others) -> bool:
    """A marker of this group shares a block row or column with a marker of another
    group and the two are at least one block apart -- the placement that makes a
    colour-blind reader fill a span the rule leaves alone."""
    for br, bc in markers:
        for orow, ocol in others:
            if br == orow and abs(bc - ocol) >= 2:
                return True
            if bc == ocol and abs(br - orow) >= 2:
                return True
    return False


def _sample_decoy(rng: np.random.Generator, n_rows: int, n_cols: int,
                  groups: list, used: set) -> Optional[dict]:
    """One off-span marker parked in a block row / column another group already uses.

    The blocks between it and that group's marker stay free, so the rule leaves them
    as background while a reader that ignores the colours fills them.
    """
    cands: list[tuple] = []
    markers = [m for g in groups for m in g["markers"]]
    if not markers:
        return None
    for i in rng.permutation(len(markers)):
        br, bc = markers[int(i)]
        for pos in range(n_cols):
            if abs(pos - bc) < 2 or (br, pos) in used:
                continue
            lo, hi = (pos, bc) if pos < bc else (bc, pos)
            if all((br, p) not in used for p in range(lo + 1, hi)):
                cands.append((br, pos))
        for pos in range(n_rows):
            if abs(pos - br) < 2 or (pos, bc) in used:
                continue
            lo, hi = (pos, br) if pos < br else (br, pos)
            if all((p, bc) not in used for p in range(lo + 1, hi)):
                cands.append((pos, bc))
        if len(cands) >= 24:
            break
    if not cands:
        return None
    cell = cands[int(rng.integers(0, len(cands)))]
    return {"kind": "decoy", "orientation": "none", "markers": [cell],
            "fills": [], "blocks": {cell}}


def _plan_groups(rng: np.random.Generator, n_rows: int, n_cols: int) -> Optional[tuple]:
    # a real group claims three or more blocks and the background must keep a 1.25x
    # plurality, so the group count and the markers per group follow the block
    # capacity of this lattice: a 50-block lattice carries as many spans, in
    # proportion, as a 10-block one.
    total = n_rows * n_cols
    capacity = max(1, min(_MAX_GROUPS, total // 4))
    # the group count tracks the block count (one group per ~5.5 blocks, jittered), so
    # a 50-block lattice is as busy as a 12-block one instead of holding two spans
    n_groups = int(round(total / 5.5 * (0.55 + 0.75 * rng.random())))
    n_groups = max(1, min(capacity, n_groups))
    max_markers = _max_markers(total)
    if rng.random() < 0.055:
        # an all-noop plan gives an identity output; it is drawn below the target
        # share because a scene with no fills is never rejected by the margins while
        # a busy one sometimes is
        n_groups = min(n_groups, 3)
        n_noop = n_groups
    else:
        n_noop = min(n_groups - 1, _weighted(rng, (0, 1, 2), _NOOP_COUNT_WEIGHTS))
    kinds: list[str] = []
    for _ in range(n_groups - n_noop):
        row_ok, col_ok = n_cols >= 3, n_rows >= 3
        if row_ok and col_ok:
            kinds.append("row" if rng.random() < 0.5 else "col")
        elif row_ok:
            kinds.append("row")
        elif col_ok:
            kinds.append("col")
        else:
            return None
    kinds.extend(_weighted(rng, _NOOP_KINDS, _NOOP_WEIGHTS) for _ in range(n_noop))
    rng.shuffle(kinds)  # type: ignore[arg-type]
    n_decoy = 0
    if len(kinds) < _MAX_GROUPS and rng.random() < _DECOY_P:
        n_decoy = 1
        if total >= 24 and len(kinds) + 1 < _MAX_GROUPS and rng.random() < 0.45:
            n_decoy = 2

    used: set = set()
    groups: list[dict] = []
    biggest = 0
    prefer_cross = rng.random() < _CROSS_P
    for kind in kinds + ["decoy"] * n_decoy:
        placed = None
        spare = 6           # tries spent looking for a crossing placement once one fits
        for _ in range(2 if kind == "decoy" else 40):
            cand = (_sample_decoy(rng, n_rows, n_cols, groups, used) if kind == "decoy"
                    else _sample_group(rng, n_rows, n_cols, kind, max_markers,
                                       long_spans=total >= 24))
            if cand is None or used & cand["blocks"]:
                continue
            size = max(biggest, len(cand["blocks"]))
            # the background must keep its 1.25x plurality over the biggest group,
            # with one block of slack for a decorated or textured ground block
            if total - len(used | cand["blocks"]) < max(_MIN_GROUND_BLOCKS,
                                                        int(1.25 * size) + 2):
                continue
            if kind == "decoy" or (prefer_cross and _crosses(
                    cand["markers"], [m for g in groups for m in g["markers"]])):
                placed = cand           # prefer a placement the colour clause decides
                break
            if placed is None:
                placed = cand
            spare -= 1
            if spare <= 0:
                break
        if placed is None:
            if kind == "decoy":
                continue                # a decoy is a bonus, never a reason to fail
            return None
        used |= placed["blocks"]
        biggest = max(biggest, len(placed["blocks"]))
        groups.append(placed)
    return groups, used


def generate_fill_between(rng: np.random.Generator) -> dict:
    h = w = None
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        if h is None or attempt % _SIZE_RUN == 1:
            # the canvas is drawn FIRST and kept for a run of attempts, so a lattice or
            # a scene that does not seat never costs the size its share of the histogram
            h, w = _rint(rng, SIDE_MIN, SIDE_MAX), _rint(rng, SIDE_MIN, SIDE_MAX)
            if h == SIDE_MIN and w == SIDE_MIN:
                # structural floor: a group needs three blocks along one direction and
                # the background keeps a plurality, so 5x5 (at most 2x2 blocks) is out.
                # It is the only excluded pair.
                h = None
                continue
        spec = sample_lattice(rng, h, w, min_blocks=6, min_line_blocks=3)
        if spec is None:
            h = None
            continue
        n_rows, n_cols = len(spec["row_bands"]), len(spec["col_bands"])
        blocks = all_blocks(spec)
        planned = _plan_groups(rng, n_rows, n_cols)
        if planned is None:
            continue
        groups, claimed = planned
        free = [b for b in blocks if b not in claimed]
        if not free:
            continue

        use_texture = min(h, w) >= 8 and rng.random() < TEXTURE_P
        roomy_free = [b for b in free if block_area(spec, b[0], b[1]) >= 2]
        want = _palette_size(rng, floor=3)
        extra = max(0, want - 2 - len(groups) - (1 if use_texture else 0))
        n_decor_colors = max(0, min(extra, 5, max(0, len(roomy_free) - 1),
                                    NUM_COLORS - 2 - len(groups)))
        palette = pick_colors(rng, 2 + len(groups) + n_decor_colors)
        line_color, bg_color = int(palette[0]), int(palette[1])
        marker_colors = [int(c) for c in palette[2:2 + len(groups)]]
        decor_colors = [int(c) for c in palette[2 + len(groups):]]

        grid = build_lattice(spec, line_color, bg_color)
        for group, color in zip(groups, marker_colors):
            group["color"] = color
            for br, bc in group["markers"]:
                fill_block(grid, spec, br, bc, color)

        decorated: list = []
        if decor_colors:
            order = list(rng.permutation(len(roomy_free)))
            wanted = min(len(roomy_free) - 1, _rint(rng, len(decor_colors), len(decor_colors) + 3))
            for i in range(max(0, wanted)):
                block = roomy_free[int(order[i])]
                color = decor_colors[i % len(decor_colors)]
                if paint_partial(rng, grid, spec, block[0], block[1], color):
                    decorated.append(block)
        ground = [b for b in free if b not in set(decorated)]
        if not ground:
            continue

        texture_exclude = {line_color, bg_color, *marker_colors, *decor_colors}
        grid, texture = add_texture(rng, grid, spec, bg_color, texture_exclude, ground, use_texture)

        if block_plurality(grid, spec, bg_color) < 1.25:
            continue
        if cell_plurality(grid, spec, bg_color) < 1.25:
            continue
        isolated_margin = isolated_cell_plurality(grid, spec["row_bands"],
                                                  spec["col_bands"], bg_color)
        if isolated_margin < 1.25:
            continue

        out = solve_fill_between(grid)
        identity = bool(np.array_equal(out, grid))
        n_noop = sum(1 for g in groups if not g["fills"])
        if identity != (n_noop == len(groups)):
            continue
        blind = apply_color_blind_rule(grid, spec["row_bands"], spec["col_bands"],
                                       line_color, bg_color)
        clause_matters = not bool(np.array_equal(blind, out))

        meta = _common_meta(spec, grid, line_color, bg_color, texture, attempt)
        meta.update({
            "background_margin_isolated": float(isolated_margin),
            "color_clause_matters": bool(clause_matters),
            "n_groups": int(len(groups)),
            "group_colors": [int(g["color"]) for g in groups],
            "group_kinds": [str(g["kind"]) for g in groups],
            "orientations": [str(g["orientation"]) for g in groups],
            "markers_per_group": [int(len(g["markers"])) for g in groups],
            "gap_blocks": [int(len(g["fills"])) for g in groups],
            "n_filled_blocks": int(sum(len(g["fills"]) for g in groups)),
            "n_noop_groups": int(n_noop),
            "identity_output": bool(identity),
            "n_marked_blocks": int(sum(len(g["markers"]) for g in groups)),
            "n_decorated_blocks": int(len(decorated)),
        })
        example = {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}
        validate_fill_between(example)
        return example
    raise RuntimeError("task 88: no scene after the retry budget")


def infer_ground(grid, rows, cols, margin: float = 1.25) -> int:
    """The background a solver told only "ignore the background" can read off.

    Block cells only (the lattice lines are not part of any block); the commonest
    colour must beat the runner-up by ``margin``.
    """
    g = np.asarray(grid)
    counts: dict = {}
    for r0, r1 in rows:
        for c0, c1 in cols:
            values, n = np.unique(g[r0:r1, c0:c1], return_counts=True)
            for v, k in zip(values, n):
                counts[int(v)] = counts.get(int(v), 0) + int(k)
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(order) > 1 and order[1][1] and order[0][1] < margin * order[1][1]:
        raise ValueError("task 88: the background is not identifiable in the grid")
    return int(order[0][0])


def apply_fill_rule(grid, rows, cols, line_color: int, ground: int) -> np.ndarray:
    """Task 88's rule, applied literally to a visible grid with a given background."""
    g = np.asarray(grid)
    marked: dict = {}
    for block, color in _uniform_block_colors(g, rows, cols).items():
        if color is None or color == ground or color == line_color:
            continue
        marked.setdefault(color, []).append(block)
    out = g.copy()
    for color, cells in marked.items():
        by_row: dict = {}
        by_col: dict = {}
        for br, bc in cells:
            by_row.setdefault(br, []).append(bc)
            by_col.setdefault(bc, []).append(br)
        for br, cs in by_row.items():
            if len(cs) < 2:
                continue
            r0, r1 = rows[br]
            for bc in range(min(cs) + 1, max(cs)):
                c0, c1 = cols[bc]
                out[r0:r1, c0:c1] = color
        for bc, rs in by_col.items():
            if len(rs) < 2:
                continue
            c0, c1 = cols[bc]
            for br in range(min(rs) + 1, max(rs)):
                r0, r1 = rows[br]
                out[r0:r1, c0:c1] = color
    return out


def apply_color_blind_rule(grid, rows, cols, line_color: int, ground: int) -> np.ndarray:
    """The misreading task 88's colour clause forbids, kept here as a counter-reading.

    It drops "For each color separately" and fills between CONSECUTIVE marked blocks
    of a block row / block column whatever their colours, using the colour of the
    first of the pair.  The generator records whether it reaches the same output
    (``color_clause_matters``); the tests bound how often it does.
    """
    g = np.asarray(grid)
    marked = [(block, color) for block, color in _uniform_block_colors(g, rows, cols).items()
              if color is not None and color not in (int(ground), int(line_color))]
    out = g.copy()
    for axis in (0, 1):
        lanes: dict = {}
        for block, color in marked:
            lanes.setdefault(block[axis], []).append((block[1 - axis], color))
        for lane, items in lanes.items():
            if len(items) < 2:
                continue
            items.sort()
            for (pos, color), (nxt, _) in zip(items, items[1:]):
                for k in range(pos + 1, nxt):
                    br, bc = (lane, k) if axis == 0 else (k, lane)
                    r0, r1 = rows[br]
                    c0, c1 = cols[bc]
                    out[r0:r1, c0:c1] = color
    return out


def solve_fill_between(grid) -> np.ndarray:
    """Independent reading of task 88's rule from a visible grid alone."""
    line_color, rows, cols = recover_lattice(grid)
    ground = infer_ground(grid, rows, cols)
    return apply_fill_rule(grid, rows, cols, line_color, ground)


def validate_fill_between(example: dict) -> None:
    meta = example["metadata"]
    out = np.asarray(example["output"])
    inp, line_color, rows, cols = _validated_lattice(example, "88")
    ground = infer_ground(inp, rows, cols)
    if ground != int(meta["bg_color"]):
        raise ValueError("task 88: the visible background differs from the drawn one")
    expected = apply_fill_rule(inp, rows, cols, line_color, ground)
    if not np.array_equal(expected, out):
        raise ValueError("task 88: the output is not the rule applied to the visible input")
    if not np.array_equal(apply_fill_rule(out, rows, cols, line_color, ground), out):
        raise ValueError("task 88: the rule is not stable on its own output")
    if bool(np.array_equal(inp, out)) != bool(meta["identity_output"]):
        raise ValueError("task 88: the identity flag does not match the grids")
    if float(meta["background_margin"]) < 1.25:
        raise ValueError("task 88: the background blocks do not hold a 1.25x plurality")
    if float(meta["background_cell_margin"]) < 1.25:
        raise ValueError("task 88: the background cells do not hold a 1.25x plurality")
    isolated = isolated_cell_plurality(inp, rows, cols, int(meta["bg_color"]))
    if isolated < 1.25 or abs(isolated - float(meta["background_margin_isolated"])) > 1e-6:
        raise ValueError("task 88: the background does not hold a 1.25x plurality once "
                         "isolated single cells are set aside")
    blind = apply_color_blind_rule(inp, rows, cols, line_color, ground)
    if bool(not np.array_equal(blind, out)) != bool(meta["color_clause_matters"]):
        raise ValueError("task 88: the colour-clause flag does not match the grids")
    marked = sum(1 for color in _uniform_block_colors(inp, rows, cols).values()
                 if color is not None and color not in (line_color, int(meta["bg_color"])))
    if marked != int(meta["n_marked_blocks"]):
        raise ValueError("task 88: the visible marked blocks differ from the plan")


# ---------------------------------------------------------------------------
# Task 89 — count the blocks completely filled with the target colour
# ---------------------------------------------------------------------------

def generate_count_filled(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        h, w = _rint(rng, SIDE_MIN, SIDE_MAX), _rint(rng, SIDE_MIN, SIDE_MAX)
        spec = sample_lattice(rng, h, w, min_blocks=4)
        if spec is None:
            continue
        blocks = all_blocks(spec)
        n_blocks = len(blocks)

        cap = min(n_blocks - 1, 12)
        if n_blocks - 1 > 12 and rng.random() < 0.35:
            cap = min(n_blocks - 1, 24)
        answer = 0 if rng.random() < 0.10 else _rint(rng, 1, cap)

        order = list(rng.permutation(n_blocks))
        pool = [blocks[int(i)] for i in order]
        full = pool[:answer]
        rest = pool[answer:]
        roomy = [b for b in rest if block_area(spec, b[0], b[1]) >= 2]

        # blocks that CONTAIN the target colour without being filled with it are what
        # makes "completely" load-bearing, so they are the rule rather than a garnish
        # and their number follows the block count of the lattice
        partial_cap = max(1, n_blocks // 3)
        n_partial = 0
        if roomy and rng.random() < (0.78 if answer == 0 else 0.62):
            n_partial = _rint(rng, 1, min(partial_cap, len(roomy)))
        partial = roomy[:n_partial]
        rest = [b for b in rest if b not in set(partial)]

        use_texture = min(h, w) >= 8 and rng.random() < TEXTURE_P
        want = _palette_size(rng, floor=3)
        n_distractor_colors = max(0, min(want - 3, 7, len(rest)))
        palette = pick_colors(rng, 3 + n_distractor_colors)
        line_color, bg_color, target_color = int(palette[0]), int(palette[1]), int(palette[2])
        distractor_colors = [int(c) for c in palette[3:]]

        distractors: list = []
        other_partial: list = []
        if distractor_colors:
            n_distractor = _rint(
                rng, len(distractor_colors),
                max(len(distractor_colors), min(len(rest), int(0.45 * n_blocks))),
            )
            distractors = rest[:min(n_distractor, len(rest))]
            leftover = [b for b in rest[len(distractors):] if block_area(spec, b[0], b[1]) >= 2]
            if leftover and rng.random() < 0.3:
                other_partial = leftover[:_rint(rng, 1, min(2, len(leftover)))]

        grid = build_lattice(spec, line_color, bg_color)
        for br, bc in full:
            fill_block(grid, spec, br, bc, target_color)
        drawn_partial: list = []
        for br, bc in partial:
            base = None
            if distractor_colors and rng.random() < 0.4:
                base = distractor_colors[int(rng.integers(0, len(distractor_colors)))]
            if paint_partial(rng, grid, spec, br, bc, target_color, base=base):
                drawn_partial.append((br, bc))
        for i, (br, bc) in enumerate(distractors):
            fill_block(grid, spec, br, bc, distractor_colors[i % len(distractor_colors)])
        for br, bc in other_partial:
            color = distractor_colors[int(rng.integers(0, len(distractor_colors)))]
            paint_partial(rng, grid, spec, br, bc, color)

        touched = set(full) | set(drawn_partial) | set(distractors) | set(other_partial)
        ground = [b for b in blocks if b not in touched]
        texture_exclude = {line_color, bg_color, target_color, *distractor_colors}
        grid, texture = add_texture(rng, grid, spec, bg_color, texture_exclude, ground, use_texture)

        if solve_count_filled(grid, target_color) != answer:
            continue

        contains = 0
        for br, bc in blocks:
            r0, r1, c0, c1 = block_box(spec, br, bc)
            if bool(np.any(grid[r0:r1, c0:c1] == target_color)):
                contains += 1

        meta = _common_meta(spec, grid, line_color, bg_color, texture, attempt)
        meta.update({
            "target_color": int(target_color),
            "answer": int(answer),
            "answer_cap": int(cap),
            "n_full_blocks": int(answer),
            "n_partial_blocks": int(len(drawn_partial)),
            "n_distractor_blocks": int(len(distractors) + len(other_partial)),
            "n_blocks_containing_target": int(contains),
            "distractor_colors": [int(c) for c in distractor_colors],
        })
        example = {"input": grid_to_list(grid), "output": str(answer), "metadata": meta}
        validate_count_filled(example)
        return example
    raise RuntimeError("task 89: no scene after the retry budget")


def solve_count_filled(grid, target_color: int) -> int:
    """Independent reading of task 89's rule from a visible grid alone."""
    g = np.asarray(grid)
    _, rows, cols = recover_lattice(g)
    total = 0
    for r0, r1 in rows:
        for c0, c1 in cols:
            if bool(np.all(g[r0:r1, c0:c1] == int(target_color))):
                total += 1
    return total


def validate_count_filled(example: dict) -> None:
    meta = example["metadata"]
    inp, line_color, rows, cols = _validated_lattice(example, "89")
    answer = solve_count_filled(inp, int(meta["target_color"]))
    if str(answer) != str(example["output"]):
        raise ValueError("task 89: the answer is not the number of completely filled blocks")
    if answer != int(meta["n_full_blocks"]):
        raise ValueError("task 89: the metadata count differs from the visible count")
    contains = 0
    for r0, r1 in rows:
        for c0, c1 in cols:
            if bool(np.any(inp[r0:r1, c0:c1] == int(meta["target_color"]))):
                contains += 1
    if contains != int(meta["n_blocks_containing_target"]):
        raise ValueError("task 89: the blocks containing the target colour differ from the metadata")
    if contains - answer != int(meta["n_partial_blocks"]):
        raise ValueError("task 89: the partially filled block count is inconsistent")
