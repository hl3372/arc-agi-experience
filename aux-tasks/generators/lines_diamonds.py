"""Lines and diamonds family generators: tasks 15, 15a, 16, 71, 72, 73, 74."""
from __future__ import annotations

import numpy as np

from common.grid import (
    NUM_COLORS,
    grid_to_list,
    make_grid,
    pick_color,
    pick_colors,
    random_background,
    random_grid_size,
)
from registry import register
from common import backgrounds as BG

MAX_RETRIES = 20

# ---------------------------------------------------------------------------
# Line-drawing helpers
# ---------------------------------------------------------------------------


def _draw_hline(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    """Draw a horizontal line on *grid* from (row, c0) to (row, c1) inclusive."""
    if c0 > c1:
        c0, c1 = c1, c0
    grid[row, c0 : c1 + 1] = color


def _draw_vline(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    """Draw a vertical line on *grid* from (r0, col) to (r1, col) inclusive."""
    if r0 > r1:
        r0, r1 = r1, r0
    grid[r0 : r1 + 1, col] = color


def _draw_line(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> list[tuple[int, int]]:
    """Draw a general line using Bresenham's algorithm.  Returns list of cells."""
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = color
            cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def _line_cells(
    H: int, W: int, r0: int, c0: int, r1: int, c1: int
) -> list[tuple[int, int]]:
    """Return the list of cells a Bresenham line would paint (no grid needed)."""
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        if 0 <= r < H and 0 <= c < W:
            cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def _ordered_line_cells(
    H: int, W: int, line: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    """Return cells along a line in endpoint order."""
    r0, c0, r1, c1 = line
    return _line_cells(H, W, r0, c0, r1, c1)


def _diagonal_band_cells(
    H: int,
    W: int,
    kind: str,
    band: int,
) -> list[tuple[int, int]]:
    """Return full-span 45-degree diagonal line cells for a fixed band."""
    cells: list[tuple[int, int]] = []
    if kind == "diag_down":
        for r in range(H):
            c = r - band
            if 0 <= c < W:
                cells.append((r, c))
    else:
        for r in range(H):
            c = band - r
            if 0 <= c < W:
                cells.append((r, c))
    return cells


def _valid_diagonal_bands(H: int, W: int, kind: str) -> list[int]:
    """Return all diagonal bands that produce a readable line."""
    if kind == "diag_down":
        candidates = range(-(W - 1), H)
    else:
        candidates = range(0, H + W - 1)
    return [band for band in candidates if len(_diagonal_band_cells(H, W, kind, band)) >= 2]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    """Color a list of cells on a grid."""
    for r, c in cells:
        grid[r, c] = color


def _pick_colors(rng: np.random.Generator, n: int, exclude: list[int]) -> list[int]:
    """Pick *n* distinct colors not in *exclude*."""
    available = [c for c in range(NUM_COLORS) if c not in exclude]
    chosen = rng.choice(available, size=n, replace=False)
    return [int(c) for c in chosen]


def _make_textured_bg(
    rng: np.random.Generator,
    H: int,
    W: int,
    bg_color: int,
    accent_palette: list[int],
) -> np.ndarray:
    """Return a background grid that is either solid or carries a patterned mix.

    The pattern only ever uses *bg_color* plus colors drawn from *accent_palette*
    (which the caller keeps DISJOINT from the task's semantic colors), so the
    semantic foreground stays unambiguous and the output can be computed from the
    textured grid. Solid is sampled ~half the time to keep clean examples too.
    """
    grid = make_grid(H, W, fill=bg_color)
    if not accent_palette or rng.random() < 0.5:
        return grid  # solid
    style = str(rng.choice(["noisy", "checkered", "striped"]))
    if style == "noisy":
        rate = float(rng.uniform(0.04, 0.13))
        n_noise = max(1, int(H * W * rate))
        for _ in range(n_noise):
            r = int(rng.integers(0, H))
            c = int(rng.integers(0, W))
            grid[r, c] = int(rng.choice(accent_palette))
    elif style == "checkered":
        cs = int(rng.integers(1, 3))
        alt = int(rng.choice(accent_palette))
        for r in range(H):
            for c in range(W):
                if ((r // cs) + (c // cs)) % 2 == 1:
                    grid[r, c] = alt
    else:  # striped
        n_stripe = min(len(accent_palette), int(rng.integers(1, 3)))
        stripe_colors = [
            int(x) for x in rng.choice(accent_palette, size=n_stripe, replace=False)
        ]
        bands = [bg_color] + stripe_colors
        horizontal = bool(rng.random() < 0.5)
        width = int(rng.integers(1, 4))
        dim = H if horizontal else W
        for i in range(dim):
            color = bands[(i // width) % len(bands)]
            if horizontal:
                grid[i, :] = color
            else:
                grid[:, i] = color
    return grid


def _random_hv_line(
    rng: np.random.Generator, H: int, W: int, orientation: str
) -> tuple[int, int, int, int]:
    """Return (r0, c0, r1, c1) for a full-span H or V line."""
    if orientation == "h":
        row = int(rng.integers(0, H))
        return (row, 0, row, W - 1)
    else:
        col = int(rng.integers(0, W))
        return (0, col, H - 1, col)


# ---------------------------------------------------------------------------
# Shared helpers for Tasks 15 & 16 — strict H/V/45-deg lines with clear
# intersections (lines extend >=2 cells past every crossing).
# ---------------------------------------------------------------------------


def _generate_strict_line(
    rng: np.random.Generator, H: int, W: int, min_len: int = 5,
    kinds: tuple[int, ...] = (0, 1, 2),
) -> tuple[int, int, int, int] | None:
    """Generate a horizontal, vertical, or 45-degree diagonal line.

    Returns (r0, c0, r1, c1) or None if the grid is too small.
    Lines are guaranteed to be at least *min_len* cells long. *kinds* restricts
    the orientation: 0=H, 1=V, 2=diag45.
    """
    kind = int(rng.choice(kinds))  # 0=H, 1=V, 2=diag45

    if kind == 0:
        # Horizontal
        if W < min_len:
            return None
        row = int(rng.integers(0, H))
        c0 = int(rng.integers(0, W - min_len + 1))
        c1 = int(rng.integers(c0 + min_len - 1, W))
        return (row, c0, row, c1)

    elif kind == 1:
        # Vertical
        if H < min_len:
            return None
        col = int(rng.integers(0, W))
        r0 = int(rng.integers(0, H - min_len + 1))
        r1 = int(rng.integers(r0 + min_len - 1, H))
        return (r0, col, r1, col)

    else:
        # 45-degree diagonal (|dr| == |dc|)
        max_diag = min(H, W)
        if max_diag < min_len:
            return None
        length = int(rng.integers(min_len, max_diag + 1))
        dr_sign = int(rng.choice([-1, 1]))
        dc_sign = int(rng.choice([-1, 1]))
        # Pick starting point so the line fits
        if dr_sign == 1:
            r0 = int(rng.integers(0, H - length + 1))
        else:
            r0 = int(rng.integers(length - 1, H))
        if dc_sign == 1:
            c0 = int(rng.integers(0, W - length + 1))
        else:
            c0 = int(rng.integers(length - 1, W))
        r1 = r0 + dr_sign * (length - 1)
        c1 = c0 + dc_sign * (length - 1)
        return (r0, c0, r1, c1)


def _lines_extend_past_intersections(
    lines: list[tuple[int, int, int, int]],
    line_cell_sets: list[set[tuple[int, int]]],
    H: int,
    W: int,
    min_extension: int = 2,
) -> bool:
    """Check that every line extends at least *min_extension* cells past
    every intersection point (no line ending exactly at an intersection).
    """
    n = len(lines)
    for i in range(n):
        cells_i = line_cell_sets[i]
        # Collect all intersection points for line i
        ix_points: set[tuple[int, int]] = set()
        for j in range(n):
            if j == i:
                continue
            ix_points |= (cells_i & line_cell_sets[j])
        if not ix_points:
            continue
        # For each intersection point, check distance to both endpoints
        ordered = _ordered_line_cells(H, W, lines[i])
        cell_indices = {cell: idx for idx, cell in enumerate(ordered)}
        for pt in ix_points:
            idx = cell_indices.get(pt)
            if idx is None:
                return False
            dist_to_start = idx
            dist_to_end = len(ordered) - 1 - idx
            if dist_to_start < min_extension or dist_to_end < min_extension:
                return False
    return True


def _generate_lines_with_intersections(
    rng: np.random.Generator,
    H: int,
    W: int,
    n_lines: int,
    min_intersections: int = 1,
    max_inner_retries: int = 40,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """Try to generate n_lines strict lines with clear intersections.

    Returns (lines, line_cell_sets, n_intersections) or None.
    """
    for _ in range(max_inner_retries):
        lines: list[tuple[int, int, int, int]] = []
        for _ in range(n_lines):
            line = _generate_strict_line(rng, H, W, min_len=5)
            if line is None:
                break
            lines.append(line)
        if len(lines) < n_lines:
            continue

        # Compute cell sets
        line_cell_sets: list[set[tuple[int, int]]] = []
        for r0, c0, r1, c1 in lines:
            cells = _line_cells(H, W, r0, c0, r1, c1)
            line_cell_sets.append(set(cells))

        # Reject collinear overlaps (>2 shared cells between any pair)
        collinear = False
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                if len(line_cell_sets[i] & line_cell_sets[j]) > 2:
                    collinear = True
                    break
            if collinear:
                break
        if collinear:
            continue

        # Count intersections
        count_grid = np.zeros((H, W), dtype=int)
        for cs in line_cell_sets:
            for r, c in cs:
                count_grid[r, c] += 1
        n_ix = int(np.sum(count_grid >= 2))
        if n_ix < min_intersections:
            continue

        # Verify lines extend >= 2 cells past every intersection
        if not _lines_extend_past_intersections(
            lines, line_cell_sets, H, W, min_extension=2
        ):
            continue

        return lines, line_cell_sets, n_ix

    return None


def _count_intersection_cells(
    line_cell_sets: list[set[tuple[int, int]]],
    H: int,
    W: int,
) -> tuple[int, np.ndarray]:
    """Return the number of cells occupied by at least two lines."""
    count_grid = np.zeros((H, W), dtype=int)
    for cs in line_cell_sets:
        for r, c in cs:
            count_grid[r, c] += 1
    return int(np.sum(count_grid >= 2)), count_grid


def _line_sets_from_lines(
    H: int,
    W: int,
    lines: list[tuple[int, int, int, int]],
) -> list[set[tuple[int, int]]]:
    """Compute cell sets for a concrete line layout."""
    return [set(_line_cells(H, W, r0, c0, r1, c1)) for r0, c0, r1, c1 in lines]


def _expanded_cells(
    cells: set[tuple[int, int]],
    H: int,
    W: int,
    radius: int = 1,
) -> set[tuple[int, int]]:
    """Return cells plus a Chebyshev-radius buffer clipped to the grid."""
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    expanded.add((rr, cc))
    return expanded


def _line_layout_is_valid(
    lines: list[tuple[int, int, int, int]],
    line_cell_sets: list[set[tuple[int, int]]],
    H: int,
    W: int,
    target_intersections: int,
) -> bool:
    """Validate a candidate layout for tasks 15/16."""
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if len(line_cell_sets[i] & line_cell_sets[j]) > 2:
                return False
    n_ix, _ = _count_intersection_cells(line_cell_sets, H, W)
    if n_ix != target_intersections:
        return False
    return _lines_extend_past_intersections(
        lines, line_cell_sets, H, W, min_extension=2
    )


def _maybe_add_isolated_line(
    rng: np.random.Generator,
    lines: list[tuple[int, int, int, int]],
    line_cell_sets: list[set[tuple[int, int]]],
    H: int,
    W: int,
    orientation: str | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]]:
    """Optionally add a buffered non-intersecting distractor line.

    *orientation* keeps the distractor consistent with a pure layout: "hv" adds
    an H/V line, "diag" adds a diagonal, None allows any.
    """
    kinds = {"hv": (0, 1), "diag": (2,)}.get(orientation, (0, 1, 2))
    occupied = set().union(*line_cell_sets) if line_cell_sets else set()
    blocked = _expanded_cells(occupied, H, W, radius=1)
    for _ in range(80):
        extra = _generate_strict_line(rng, H, W, min_len=5, kinds=kinds)
        if extra is None:
            break
        extra_cells = set(_line_cells(H, W, *extra))
        if not extra_cells:
            continue
        if extra_cells & blocked:
            continue
        return lines + [extra], line_cell_sets + [extra_cells]
    return lines, line_cell_sets


def _render_line_scene(
    H: int,
    W: int,
    bg: int,
    lines: list[tuple[int, int, int, int]],
    line_cell_sets: list[set[tuple[int, int]]],
    line_colors: list[int],
    junction_color: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the line scene.

    Normal mode (*junction_color* is None): each line has its own color and is
    drawn in order, so a shared crossing cell takes the color of whichever line
    is drawn last. Because every line extends >=2 cells past every crossing, the
    crossing remains unambiguous and each line is still visible on both sides.

    Legacy mode (*junction_color* given, no longer used by tasks 15/16): every
    cell occupied by 2+ lines is painted with that color after line bodies.
    """
    assert len(lines) == len(line_cell_sets) == len(line_colors)

    grid = make_grid(H, W, fill=bg)
    _, count_grid = _count_intersection_cells(line_cell_sets, H, W)
    ix_mask = count_grid >= 2
    for cells, color in zip(line_cell_sets, line_colors):
        for r, c in cells:
            if junction_color is not None and ix_mask[r, c]:
                continue
            grid[r, c] = color
    if junction_color is not None:
        grid[ix_mask] = junction_color
    return grid, count_grid


def _rendered_lines_are_clear(
    grid: np.ndarray,
    line_cell_sets: list[set[tuple[int, int]]],
    line_colors: list[int],
    count_grid: np.ndarray,
    junction_color: int | None,
) -> bool:
    """Check that every line stays readable in the rendered grid.

    With distinct per-line colors and no junction color, a non-crossing line
    cell must show its own line color, and a crossing cell must show one of the
    colors of the lines passing through it (never the background or a distractor
    color). Combined with the >=2-cell extension past every crossing, this means
    every line is visible on both sides of every junction.
    """
    ix_mask = count_grid >= 2
    if junction_color is None:
        colors_through: dict[tuple[int, int], set[int]] = {}
        for cells, color in zip(line_cell_sets, line_colors):
            for cell in cells:
                colors_through.setdefault(cell, set()).add(int(color))
        for cells, color in zip(line_cell_sets, line_colors):
            for r, c in cells:
                if ix_mask[r, c]:
                    if int(grid[r, c]) not in colors_through[(r, c)]:
                        return False
                elif int(grid[r, c]) != int(color):
                    return False
        return True
    # Legacy junction-color path (kept for compatibility; unused by 15/16).
    for cells, color in zip(line_cell_sets, line_colors):
        for r, c in cells:
            expected = junction_color if ix_mask[r, c] else color
            if int(grid[r, c]) != int(expected):
                return False
    return True


def _shared_render_has_clear_separation(
    line_cell_sets: list[set[tuple[int, int]]],
    H: int,
    W: int,
) -> bool:
    """Return whether same-color lines avoid extra visual merging."""
    for i in range(len(line_cell_sets)):
        expanded = _expanded_cells(line_cell_sets[i], H, W, radius=1)
        for j in range(i + 1, len(line_cell_sets)):
            if line_cell_sets[i] & line_cell_sets[j]:
                continue
            if expanded & line_cell_sets[j]:
                return False
    return True


_DISTRACTOR_OBJECT_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
]


def _maybe_paint_distractor_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    line_cell_sets: list[set[tuple[int, int]]],
    exclude_colors: set[int],
) -> tuple[int, list[int]]:
    """Paint small non-line objects away from all line cells, if safe."""
    if rng.random() >= 0.35:
        return 0, []

    H, W = grid.shape
    available = [c for c in range(NUM_COLORS) if c not in exclude_colors]
    if not available:
        return 0, []

    line_cells = set().union(*line_cell_sets) if line_cell_sets else set()
    blocked = _expanded_cells(line_cells, H, W, radius=1)
    target_count = int(rng.choice([1, 2], p=[0.78, 0.22]))
    painted_colors: list[int] = []

    for _ in range(target_count):
        placed = False
        shape_order = rng.permutation(len(_DISTRACTOR_OBJECT_SHAPES))
        for shape_idx in shape_order:
            shape = _DISTRACTOR_OBJECT_SHAPES[int(shape_idx)]
            max_dr = max(dr for dr, _ in shape)
            max_dc = max(dc for _, dc in shape)
            if H <= max_dr or W <= max_dc:
                continue
            for _try in range(80):
                r0 = int(rng.integers(0, H - max_dr))
                c0 = int(rng.integers(0, W - max_dc))
                cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
                if cells & blocked:
                    continue
                color = int(rng.choice(available))
                for r, c in cells:
                    grid[r, c] = color
                blocked |= _expanded_cells(cells, H, W, radius=1)
                painted_colors.append(color)
                placed = True
                break
            if placed:
                break

    return len(painted_colors), painted_colors


def _choose_line_render_palette(
    rng: np.random.Generator,
    n_lines: int,
    bg: int,
    include_highlight: bool,
    forced_mode: str | None = None,
) -> tuple[str, list[int], int | None, int | None]:
    """Choose one distinct color per line (plus an optional highlight).

    Crossings are NOT painted in a separate junction color. Each line has its
    own color and every line extends at least two cells past every crossing, so
    a shared crossing cell can take either crossing line's color and still be
    unambiguous: each line stays visible on both sides of the junction. This
    replaces the previous junction-color rendering, which the review flagged as
    wrong ("the junction should NOT be colored differently").
    """
    del forced_mode
    render_mode = "distinct"
    required = n_lines + int(include_highlight)
    if required > NUM_COLORS - 1:
        raise RuntimeError(
            f"Cannot render {n_lines} lines with distinct line and highlight colors"
        )

    colors = _pick_colors(rng, required, [bg])
    line_colors = colors[:n_lines]
    junction_color = None
    highlight = colors[n_lines] if include_highlight else None
    return render_mode, line_colors, junction_color, highlight


def _build_hv_intersection_family(
    rng: np.random.Generator,
    H: int,
    W: int,
    h_count: int,
    v_count: int,
    min_gap: int = 1,
    max_lines: int = 7,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """Construct an H/V line family with exactly h_count * v_count intersections.

    ``min_gap`` > 1 (task 15 on large canvases) keeps parallel lines at least that far
    apart; ``max_lines`` bounds the optional isolated extra line.  The defaults reproduce
    the original behaviour (task 16).
    """
    if H < 7 or W < 7 or H - 4 < h_count or W - 4 < v_count:
        return None
    rows = sorted(int(v) for v in rng.choice(np.arange(2, H - 2), size=h_count, replace=False))
    cols = sorted(int(v) for v in rng.choice(np.arange(2, W - 2), size=v_count, replace=False))
    if min_gap > 1:
        if any(b - a < min_gap for a, b in zip(rows, rows[1:])) or any(b - a < min_gap for a, b in zip(cols, cols[1:])):
            return None
    # Longer, more varied lines: bias toward full-span lines, and give partial lines a larger random
    # margin so they aren't always minimal stubs around the intersection cluster.
    if rng.random() < 0.6:
        h_start, h_end = 0, W - 1
    else:
        h_margin = int(rng.integers(2, 7))
        h_start = max(0, cols[0] - h_margin)
        h_end = min(W - 1, cols[-1] + h_margin)
    if rng.random() < 0.6:
        v_start, v_end = 0, H - 1
    else:
        v_margin = int(rng.integers(2, 7))
        v_start = max(0, rows[0] - v_margin)
        v_end = min(H - 1, rows[-1] + v_margin)
    lines = [(row, h_start, row, h_end) for row in rows]
    lines.extend((v_start, col, v_end, col) for col in cols)
    line_cell_sets = _line_sets_from_lines(H, W, lines)
    target = h_count * v_count
    if not _line_layout_is_valid(lines, line_cell_sets, H, W, target):
        return None
    if len(lines) < max_lines and rng.random() < 0.5:
        lines, line_cell_sets = _maybe_add_isolated_line(
            rng, lines, line_cell_sets, H, W, orientation="hv"
        )
    return lines, line_cell_sets, target


def _build_target5_family(
    rng: np.random.Generator,
    H: int,
    W: int,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """Construct a mixed H/V/diagonal family with exactly five intersections."""
    if H < 11 or W < 11:
        return None
    for _ in range(80):
        cols = sorted(int(v) for v in rng.choice(np.arange(2, W - 2), size=3, replace=False))
        c_left, c_mid, c_right = cols
        if c_mid - c_left < 2 or c_right - c_mid < 2:
            continue
        row_min = (c_mid - c_left) + 2
        row_max = H - 3 - (c_right - c_mid)
        if row_min > row_max:
            continue
        h_row = int(rng.integers(row_min, row_max + 1))
        upper_ix_row = h_row - (c_mid - c_left)
        lower_ix_row = h_row + (c_right - c_mid)
        h_line = (h_row, max(0, c_left - 2), h_row, min(W - 1, c_right + 2))
        v_top = max(0, upper_ix_row - 2)
        v_bottom = min(H - 1, lower_ix_row + 2)
        left_v = (v_top, c_left, v_bottom, c_left)
        right_v = (v_top, c_right, v_bottom, c_right)
        diag = (upper_ix_row - 2, c_left - 2, lower_ix_row + 2, c_right + 2)
        lines = [h_line, left_v, right_v, diag]
        line_cell_sets = _line_sets_from_lines(H, W, lines)
        if not _line_layout_is_valid(lines, line_cell_sets, H, W, 5):
            continue
        if rng.random() < 0.5:
            lines, line_cell_sets = _maybe_add_isolated_line(
                rng, lines, line_cell_sets, H, W
            )
        return lines, line_cell_sets, 5
    return None


def _diag_line_endpoints(
    H: int, W: int, kind: str, band: int
) -> tuple[int, int, int, int] | None:
    """Endpoints (r0,c0,r1,c1) of the full-span 45-degree diagonal at *band*."""
    cells = _diagonal_band_cells(H, W, kind, band)
    if len(cells) < 2:
        return None
    return (cells[0][0], cells[0][1], cells[-1][0], cells[-1][1])


def _build_diagonal_lattice_family(
    rng: np.random.Generator,
    H: int,
    W: int,
    n_down: int,
    n_up: int,
    max_lines: int = 7,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """Lattice of *n_down* down-diagonals and *n_up* up-diagonals with exactly
    n_down*n_up intersections, so a single grid carries MULTIPLE diagonal lines.

    A down-diagonal has r-c constant, an up-diagonal r+c constant; the two cross
    at an integer cell only when their constants share parity, which the
    construction guarantees (down band = rc0-cc0+2*sd*i, up band = rc0+cc0+2*su*j,
    so every sum is even). All crossings are kept >=2 cells from the border, so
    every line stays visible on both sides of every junction.
    """
    if H < 11 or W < 11 or n_down < 1 or n_up < 1:
        return None
    for _ in range(50):
        # Band step (half) for the two slopes; bias wider so the X-crossings are
        # clearly separated, falling back to 1 only where the grid is too tight.
        sd = int(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
        su = int(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
        span = sd * (n_down - 1) + su * (n_up - 1)
        if span > H - 5 or span > W - 5:
            continue
        rc0 = int(rng.integers(2, H - 3 - span + 1))
        cc0 = int(rng.integers(2 + sd * (n_down - 1), W - 3 - su * (n_up - 1) + 1))
        down_bands = [(rc0 - cc0) + 2 * sd * i for i in range(n_down)]
        up_bands = [(rc0 + cc0) + 2 * su * j for j in range(n_up)]
        lines: list[tuple[int, int, int, int]] = []
        ok = True
        for kind, bands in (("diag_down", down_bands), ("diag_up", up_bands)):
            for band in bands:
                ep = _diag_line_endpoints(H, W, kind, band)
                if ep is None:
                    ok = False
                    break
                lines.append(ep)
            if not ok:
                break
        if not ok:
            continue
        line_cell_sets = _line_sets_from_lines(H, W, lines)
        target = n_down * n_up
        if not _line_layout_is_valid(lines, line_cell_sets, H, W, target):
            continue
        if len(lines) < max_lines and rng.random() < 0.4:
            lines, line_cell_sets = _maybe_add_isolated_line(
                rng, lines, line_cell_sets, H, W, orientation="diag"
            )
        return lines, line_cell_sets, target
    return None


def _build_mixed_lines_family(
    rng: np.random.Generator,
    H: int,
    W: int,
    max_lines: int = 7,
    per_kind_max: int = 2,
    min_gap: int = 1,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """A layout that MIXES H/V straight lines and 45-degree diagonals in one grid,
    returning the natural intersection count. Review: examples should vary between
    straight-only, diagonal-only, and a mix of both. The straight and diagonal
    lines are full-span so they genuinely cross each other.

    ``min_gap`` > 1 (task 15) keeps parallel lines at least that far apart in index
    (rows, columns or diagonal bands); the default reproduces task 16 exactly."""
    if H < 12 or W < 12:
        return None
    for _ in range(50):
        n_h = int(rng.integers(0, per_kind_max + 1))
        n_v = int(rng.integers(0, per_kind_max + 1))
        n_down = int(rng.integers(0, per_kind_max + 1))
        n_up = int(rng.integers(0, per_kind_max + 1))
        if n_h + n_v == 0:  # guarantee at least one straight line
            n_h, n_v = (1, 0) if rng.random() < 0.5 else (0, 1)
        if n_down + n_up == 0:  # guarantee at least one diagonal
            if rng.random() < 0.5:
                n_down = 1
            else:
                n_up = 1
        if n_h + n_v + n_down + n_up > max_lines:  # keep the line count readable
            continue
        if max_lines > 7 and n_h + n_v + n_down + n_up < 5:  # large canvases carry full scenes
            continue
        lines: list[tuple[int, int, int, int]] = []
        if n_h:
            picked = rng.choice(np.arange(3, H - 3), size=n_h, replace=False)
            if min_gap > 1 and _t15_indices_too_close(picked, min_gap):
                continue
            for row in picked:
                lines.append((int(row), 0, int(row), W - 1))
        if n_v:
            picked = rng.choice(np.arange(3, W - 3), size=n_v, replace=False)
            if min_gap > 1 and _t15_indices_too_close(picked, min_gap):
                continue
            for col in picked:
                lines.append((0, int(col), H - 1, int(col)))
        ok = True
        for kind, n in (("diag_down", n_down), ("diag_up", n_up)):
            if not n:
                continue
            long_bands = [b for b in _valid_diagonal_bands(H, W, kind)
                          if len(_diagonal_band_cells(H, W, kind, b)) >= min(H, W) - 2]
            pool = long_bands if len(long_bands) >= n else _valid_diagonal_bands(H, W, kind)
            if len(pool) < n:
                ok = False
                break
            picked = rng.choice(pool, size=n, replace=False)
            if min_gap > 1 and _t15_indices_too_close(picked, min_gap):
                ok = False
                break
            for b in picked:
                ep = _diag_line_endpoints(H, W, kind, int(b))
                if ep is None:
                    ok = False
                    break
                lines.append(ep)
            if not ok:
                break
        if not ok or len(lines) < 2:
            continue
        line_cell_sets = _line_sets_from_lines(H, W, lines)
        count, _ = _count_intersection_cells(line_cell_sets, H, W)
        if not 2 <= count <= (12 if max_lines <= 7 else 20):
            continue
        has_hv = any(r0 == r1 or c0 == c1 for (r0, c0, r1, c1) in lines)
        has_diag = any(r0 != r1 and c0 != c1 for (r0, c0, r1, c1) in lines)
        if not (has_hv and has_diag):
            continue
        if not _line_layout_is_valid(lines, line_cell_sets, H, W, count):
            continue
        return lines, line_cell_sets, count
    return None


def _generate_targeted_intersection_layout(
    rng: np.random.Generator,
    H: int,
    W: int,
    target_intersections: int,
    scaled: bool = False,
    max_lines: int = 9,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """Build a layout with an intended exact intersection count (straight-only or
    diagonal-only lattice).  ``scaled`` (task 15) adds the large targets 15 / 16 / 20,
    keeps parallel lines two cells apart and allows up to nine lines; ``max_lines``
    (task 16, eight) lowers that ceiling without touching the default stream."""
    max_lines = min(9 if scaled else 7, max_lines)
    min_gap = 2 if scaled else 1
    pair_options: dict[int, list[tuple[int, int]]] = {
        2: [(1, 2), (2, 1)],
        3: [(1, 3), (3, 1)],
        4: [(2, 2), (1, 4), (4, 1)],
        6: [(2, 3), (3, 2)],
        8: [(2, 4), (4, 2)],
        9: [(3, 3)],
        10: [(2, 5), (5, 2)],
        12: [(3, 4), (4, 3)],
        15: [(3, 5), (5, 3)],
        16: [(4, 4)],
        20: [(4, 5), (5, 4)],
    }
    diag_options: dict[int, list[tuple[int, int]]] = {
        2: [(1, 2), (2, 1)],
        3: [(1, 3), (3, 1)],
        4: [(2, 2)],
        5: [(1, 5), (5, 1)],
        6: [(2, 3), (3, 2)],
        8: [(2, 4), (4, 2)],
        9: [(3, 3)],
        10: [(2, 5), (5, 2)],
        12: [(3, 4), (4, 3)],
        15: [(3, 5), (5, 3)],
        16: [(4, 4)],
        20: [(4, 5), (5, 4)],
    }
    # ~50/50 between a diagonal-only lattice and a straight-only H/V lattice.
    if H >= 11 and W >= 11 and rng.random() < 0.5:
        opts = diag_options.get(target_intersections, [])
        for k in rng.permutation(len(opts)):
            res = _build_diagonal_lattice_family(rng, H, W, *opts[int(k)], max_lines=max_lines)
            if res is not None:
                return res
    if target_intersections == 5:
        return _build_target5_family(rng, H, W)
    for _ in range(20):
        options = pair_options.get(target_intersections, [])
        if not options:
            return None
        h_count, v_count = options[int(rng.integers(0, len(options)))]
        result = _build_hv_intersection_family(rng, H, W, h_count, v_count, min_gap=min_gap, max_lines=max_lines)
        if result is not None:
            return result
    return None


def _t15_indices_too_close(picked, min_gap: int) -> bool:
    srt = sorted(int(v) for v in picked)
    return any(b - a < min_gap for a, b in zip(srt, srt[1:]))


def _t15_line_kind(line: tuple[int, int, int, int]) -> tuple[str, int]:
    """('h', row) / ('v', col) / ('d', r - c) for a down-diagonal / ('u', r + c) for an up-diagonal."""
    r0, c0, r1, c1 = line
    if r0 == r1:
        return "h", r0
    if c0 == c1:
        return "v", c0
    if (r1 - r0) * (c1 - c0) > 0:
        return "d", r0 - c0
    return "u", r0 + c0


def _t15_layout_flaws(lines, line_cell_sets, H: int, W: int) -> str | None:
    """Task-15 geometry guard (task 16 keeps the shared families untouched).

    Returns the first flaw found, else None:
    - "phantom crossing": two opposite-slope diagonals whose real intersection falls
      between cells (odd band parity) inside both segments -- they visibly cross but share
      no cell, so the count would not match what a reader sees;
    - "adjacent parallel lines": two lines of the same orientation less than two rows /
      columns / bands apart where their extents overlap;
    - "line end touches another line": an endpoint 8-adjacent to a cell of another line it
      does not share (a T-junction near miss, which contradicts "continues straight
      through and at least two cells past each point where it meets another line").
    """
    n = len(lines)
    kinds = [_t15_line_kind(ln) for ln in lines]
    rows = [(min(ln[0], ln[2]), max(ln[0], ln[2])) for ln in lines]
    cols = [(min(ln[1], ln[3]), max(ln[1], ln[3])) for ln in lines]

    def overlap(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    for i in range(n):
        ki, bi = kinds[i]
        for j in range(i + 1, n):
            kj, bj = kinds[j]
            if ki == kj:
                if abs(bi - bj) < 2:
                    along = overlap(cols[i], cols[j]) if ki == "h" else overlap(rows[i], rows[j])
                    if along:
                        return "adjacent parallel lines"
            elif {ki, kj} == {"d", "u"}:
                bd, bu = (bi, bj) if ki == "d" else (bj, bi)
                if (bd + bu) % 2 != 0:
                    r2, c2 = bd + bu, bu - bd          # twice the real crossing point
                    if all(2 * rows[x][0] < r2 < 2 * rows[x][1] and 2 * cols[x][0] < c2 < 2 * cols[x][1]
                           for x in (i, j)):
                        return "phantom crossing"
    for i in range(n):
        r0, c0, r1, c1 = lines[i]
        for p in ((r0, c0), (r1, c1)):
            for j in range(n):
                if j == i or p in line_cell_sets[j]:
                    continue
                if any((p[0] + dr, p[1] + dc) in line_cell_sets[j]
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                    return "line end touches another line"
    return None


_T15_RARE_TARGETS = (7, 11, 13, 14, 17, 18, 19)
_T15_RARE_W = (0.06, 0.10, 0.14, 0.14, 0.16, 0.18, 0.22)   # the high counts need the most help
_T15_RARE_P = 0.33
_T15_THIN_TWO_P = 0.35         # small canvases mostly yield two crossings; some are redrawn at the same size


def _t15_spaced_sample(rng: np.random.Generator, lo: int, hi: int, n: int, gap: int):
    """``n`` sorted integers in [lo, hi) at least ``gap`` apart, or None."""
    if n <= 0:
        return []
    if hi - lo < (n - 1) * gap + 1:
        return None
    for _ in range(12):
        picked = sorted(int(v) for v in rng.choice(np.arange(lo, hi), size=n, replace=False))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _t15_diagonal_crossing_exactly(rng: np.random.Generator, H: int, W: int, blocked, k: int, kinds):
    """A 45-degree segment that crosses exactly ``k`` cells of ``blocked`` (the cells of the
    lines already placed), never through a cell shared by two of them, with both ends at
    least two cells clear of every placed line and at least two cells past every crossing.
    ``blocked`` maps cell -> number of placed lines through it."""
    clear = _expanded_cells(set(blocked), H, W, 1)
    candidates = []
    for kind in kinds:
        for band in _valid_diagonal_bands(H, W, kind):
            cells = _diagonal_band_cells(H, W, kind, band)
            if len(cells) < k + 4:
                continue
            hits = [i for i, cell in enumerate(cells) if cell in blocked]
            free = [i for i, cell in enumerate(cells) if cell not in clear]
            if len(hits) < k:
                continue
            for m in range(len(hits) - k + 1):
                window = hits[m:m + k]
                if any(blocked[cells[i]] >= 2 for i in window):
                    continue
                prev_hit = hits[m - 1] if m > 0 else -1
                next_hit = hits[m + k] if m + k < len(hits) else len(cells)
                starts = [i for i in free if prev_hit < i <= window[0] - 2]
                ends = [i for i in free if window[-1] + 2 <= i < next_hit]
                if starts and ends:
                    candidates.append((kind, band, cells, starts, ends))
    if not candidates:
        return None
    kind, band, cells, starts, ends = candidates[int(rng.integers(0, len(candidates)))]
    i = starts[int(rng.integers(0, len(starts)))]
    j = ends[int(rng.integers(0, len(ends)))]
    return kind, band, (cells[i][0], cells[i][1], cells[j][0], cells[j][1])


def _build_rare_count_layout(rng: np.random.Generator, H: int, W: int, target: int, max_lines: int = 9):
    """A straight lattice of a x b lines (parallel lines >= 3 apart) plus one or two
    45-degree segments that each cross an exact number of lattice lines away from the
    nodes, so the total is exactly ``target`` -- the counts (7, 11, 13, 14, 17, 18, 19)
    that no plain lattice produces.  The result is re-counted and passes the geometry
    guard before it is returned.  ``max_lines`` (default nine; eight for task 16, whose
    palette must also hold the highlight) restricts the constructions considered."""
    options = []
    for a in range(1, 5):
        for b in range(1, 6):
            rest = target - a * b
            if 1 <= rest <= a + b and a + b + 1 <= max_lines:  # at most max_lines lines on a canvas
                options.append((a, b, rest, 0))
            if a + b + 2 > max_lines:
                continue
            for k1 in range(1, a + b + 1):
                k2 = rest - k1
                if 1 <= k2 <= k1:
                    options.append((a, b, k1, k2))
    if not options:
        return None
    for _ in range(12):
        a, b, k1, k2 = options[int(rng.integers(0, len(options)))]
        if rng.random() < 0.5:
            a, b = b, a
        rows = _t15_spaced_sample(rng, 2, H - 2, a, 3)
        cols = _t15_spaced_sample(rng, 2, W - 2, b, 3)
        if rows is None or cols is None:
            continue
        if rng.random() < 0.6:
            h_start, h_end = 0, W - 1
        else:
            margin = int(rng.integers(2, 7))
            h_start, h_end = max(0, cols[0] - margin), min(W - 1, cols[-1] + margin)
        if rng.random() < 0.6:
            v_start, v_end = 0, H - 1
        else:
            margin = int(rng.integers(2, 7))
            v_start, v_end = max(0, rows[0] - margin), min(H - 1, rows[-1] + margin)
        lines = [(row, h_start, row, h_end) for row in rows]
        lines.extend((v_start, col, v_end, col) for col in cols)
        sets = _line_sets_from_lines(H, W, lines)
        blocked: dict[tuple[int, int], int] = {}
        for cs in sets:
            for cell in cs:
                blocked[cell] = blocked.get(cell, 0) + 1
        kinds = ["diag_down", "diag_up"]
        rng.shuffle(kinds)
        first = _t15_diagonal_crossing_exactly(rng, H, W, blocked, k1, kinds)
        if first is None:
            continue
        kind, band, seg = first
        lines.append(seg)
        if k2:
            # a second, parallel segment two or more bands away (same kind: parallel
            # segments never cross, so the total stays exact and no phantom can arise)
            seg_cells = set(_line_cells(H, W, *seg))
            blocked2 = dict(blocked)
            for cell in seg_cells:
                blocked2[cell] = blocked2.get(cell, 0) + 1
            second = None
            for _try in range(4):
                cand = _t15_diagonal_crossing_exactly(rng, H, W, blocked2, k2, [kind])
                if cand is None:
                    break
                if abs(cand[1] - band) >= 2 and not (set(_line_cells(H, W, *cand[2])) & seg_cells):
                    second = cand
                    break
            if second is None:
                continue
            lines.append(second[2])
        sets = _line_sets_from_lines(H, W, lines)
        count, _ = _count_intersection_cells(sets, H, W)
        if count != target:
            continue
        if not _line_layout_is_valid(lines, sets, H, W, target):
            continue
        if _t15_layout_flaws(lines, sets, H, W) is not None:
            continue
        return lines, sets, target
    return None


def _choose_intersection_layout(
    rng: np.random.Generator,
    H: int,
    W: int,
    scaled: bool = False,
    max_lines: int = 9,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]], int] | None:
    """Pick a layout for tasks 15/16 with variety: a MIX of straight + diagonal,
    a diagonal-only lattice, or a straight-only H/V lattice. Returns
    (lines, line_cell_sets, count) or None.

    ``scaled`` (tasks 15, 15a, 16): the line and crossing budget grows with the canvas --
    from 300 cells the large targets 15 / 16 / 20 join with weights rising with the
    area, mixed layouts may hold up to nine lines, and the free layout draws 4..9 lines.
    ``max_lines`` (task 16, eight: a distinct colour per line plus the reserved
    highlight) lowers that ceiling -- target 20 (a 4 x 5 lattice) is then left out --
    without touching the default stream.  The unscaled default reproduces the
    original behaviour.
    """
    area = H * W
    big = scaled and area >= 300
    # Task 15: the counts a lattice cannot produce are targeted directly on roomy canvases
    # so every answer 2..20 is well represented.
    if scaled and area >= 100 and rng.random() < _T15_RARE_P:
        target = int(rng.choice(_T15_RARE_TARGETS, p=_T15_RARE_W))
        res = _build_rare_count_layout(rng, H, W, target, max_lines=max_lines)
        if res is not None:
            return res
    # A good share of roomy grids are a genuine mix; the targeted dispatcher
    # otherwise yields diagonal-only or straight-only lattices.
    if H >= 12 and W >= 12 and rng.random() < 0.4:
        res = _build_mixed_lines_family(rng, H, W, max_lines=min(9, max_lines), per_kind_max=3, min_gap=2) if big else \
            (_build_mixed_lines_family(rng, H, W, min_gap=2) if scaled else _build_mixed_lines_family(rng, H, W))
        if res is not None:
            return res
    if big:
        grow = min(1.0, (area - 300) / 600.0)   # 0 at 300 cells, 1 at 900
        target_options = np.array([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20], dtype=int)
        target_weights = np.array([0.8, 0.8, 1.0, 0.9, 1.0, 0.7, 0.65, 0.55, 0.5,
                                   0.25 + 0.9 * grow, 0.25 + 0.9 * grow, 0.2 + 0.9 * grow], dtype=float)
        if max_lines < 9:
            target_options, target_weights = target_options[:-1], target_weights[:-1]
    else:
        target_options = np.array([2, 3, 4, 5, 6, 8, 9, 10, 12], dtype=int)
        target_weights = np.array([1.0, 1.0, 1.15, 1.0, 1.05, 0.5, 0.42, 0.3, 0.22], dtype=float)
    target = int(rng.choice(target_options, p=target_weights / target_weights.sum()))
    res = _generate_targeted_intersection_layout(rng, H, W, target, scaled=big, max_lines=max_lines)
    if res is not None:
        return res
    n_lines = int(rng.integers(6, min(10, max_lines + 1))) if big else int(rng.integers(4, 8))
    return _generate_lines_with_intersections(
        rng, H, W, n_lines, min_intersections=max(2, min(4, target))
    )


# ---------------------------------------------------------------------------
# Task 15 — Count line intersections (grid -> text)
# ---------------------------------------------------------------------------

_T15_TEXTURE_P = 0.30
_T15_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
                   "texture_named": False, "texture_coverage": 0.0}


def _solve_intersections_from_grid(grid: np.ndarray, bg: int) -> int:
    """Independent solver for task 15 from the visible grid alone.

    A line is a colour whose cells all lie on one horizontal, vertical or 45-degree
    segment of at least five cells, where every gap along that segment is a cell of
    another (non-background) colour -- the crossing cells that show the other line's
    colour.  Isolated cells, small distractor objects and dotted textures have background
    gaps or are not collinear, so they are not lines.  Returns the number of cells that
    lie on two or more lines.
    """
    from collections import Counter

    segments = []
    for color in np.unique(grid):
        if int(color) == int(bg):
            continue
        cells = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
        if len(cells) < 3:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 == r1:
            line = [(r0, c) for c in range(c0, c1 + 1)]
        elif c0 == c1:
            line = [(r, c0) for r in range(r0, r1 + 1)]
        elif (r1 - r0) == (c1 - c0):
            if all((r - r0) == (c - c0) for r, c in cells):
                line = [(r0 + i, c0 + i) for i in range(r1 - r0 + 1)]
            elif all((r - r0) == (c1 - c) for r, c in cells):
                line = [(r0 + i, c1 - i) for i in range(r1 - r0 + 1)]
            else:
                continue
        else:
            continue
        if len(line) < 5:
            continue
        own = set(cells)
        lset = set(line)
        if not own <= lset:
            continue
        if any(int(grid[r, c]) == int(bg) for r, c in lset - own):
            continue
        segments.append(lset)
    count = Counter(cell for s in segments for cell in s)
    return sum(1 for v in count.values() if v >= 2)


@register("15")
def generate_count_intersections(rng: np.random.Generator) -> dict:
    """Task 15: count the cells that lie on two or more lines (H / V / 45-degree).

    Sizes 7..30 per side, independent, and content is retried at the same size so
    rejections do not thin the small sides.  Layouts scale with the canvas: straight or
    diagonal lattices, genuine mixes of both, free layouts, and (30% from 150 cells)
    layouts targeted at the counts a lattice cannot produce (7, 11, 13, 14, 17, 18, 19)
    -- a straight lattice plus one or two 45-degree segments crossing an exact number
    of its lines -- from 100 cells,
    with up to 9 lines and up to 20 crossings on large grids.  Every layout passes a
    geometry guard: no two opposite-slope diagonals may cross between cells (a
    "phantom" crossing that shares no cell), parallel lines whose extents overlap are
    at least two rows / columns / bands apart, no line ends 8-adjacent to another
    line it does not share, and every line runs >= 2 cells past every crossing.  Each
    line has its own colour; the background varies; 30% of canvases with a short side
    >= 8 carry a sparse background-library texture (isolated cells, never adjacent to
    a line) that the prompt's "ignore isolated non-line distractor objects" sentence
    already covers -- nothing about the background is named.  Every answer is
    recomputed by an independent solver from the visible grid; the generator raises
    on disagreement.
    """
    for _size_attempt in range(12):
        H, W = random_grid_size(rng, min_dim=7, max_dim=30)
        for _content_attempt in range(25):
            bg = pick_color(rng)
            result = _choose_intersection_layout(rng, H, W, scaled=True)
            if result is None:
                continue
            lines, line_cell_sets, intersections = result
            intersections, _ = _count_intersection_cells(line_cell_sets, H, W)
            if intersections < 2:
                continue
            if intersections == 2 and rng.random() < _T15_THIN_TWO_P:
                continue                                   # keeps the answer histogram flatter on small canvases
            if _t15_layout_flaws(lines, line_cell_sets, H, W) is not None:
                continue
            example = _t15_finish(rng, H, W, bg, lines, line_cell_sets, intersections)
            if example is not None:
                return example
    raise RuntimeError("Task 15: failed to generate valid example")


def _t15_finish(rng, H, W, bg, lines, line_cell_sets, intersections):
    """Render, decorate and validate one task-15 layout; None when a render check fails."""
    if True:

        render_mode, line_colors, junction_color, _ = _choose_line_render_palette(
            rng, len(lines), bg, include_highlight=False
        )
        grid, count_grid = _render_line_scene(
            H, W, bg, lines, line_cell_sets, line_colors, junction_color
        )
        if int(np.sum(count_grid >= 2)) != intersections:
            return None
        if not _rendered_lines_are_clear(
            grid, line_cell_sets, line_colors, count_grid, junction_color
        ):
            return None

        exclude_colors = {int(bg), *[int(c) for c in line_colors]}
        distractor_count, distractor_colors = _maybe_paint_distractor_objects(
            rng, grid, line_cell_sets, exclude_colors
        )
        if not _rendered_lines_are_clear(
            grid, line_cell_sets, line_colors, count_grid, junction_color
        ):
            return None

        # Sparse background texture, one cell clear of every line and distractor cell.
        texture = dict(_T15_NO_TEXTURE)
        if min(H, W) >= 8 and rng.random() < _T15_TEXTURE_P:
            occupied = np.zeros((H, W), dtype=bool)
            for r, c in _expanded_cells({(int(r), int(c)) for r, c in np.argwhere(grid != bg)}, H, W, radius=1):
                occupied[r, c] = True
            painted = BG.apply_background(
                rng, grid, int(bg), occupied, group="sparse",
                exclude=[int(c) for c in line_colors] + [int(c) for c in distractor_colors],
            )
            if painted is not None:
                grid, info = painted
                texture = {"texture_group": info["group"], "texture_kind": info["kind"],
                           "texture_colors": [int(c) for c in info["texture_colors"]],
                           "texture_named": False, "texture_coverage": float(info["coverage"])}

        # Ensure all line colors are visible
        visible_colors = set(int(v) for v in np.unique(grid) if v != bg)
        if not all(c in visible_colors for c in set(line_colors)):
            return None

        solved = _solve_intersections_from_grid(grid, int(bg))
        if solved != intersections:
            raise RuntimeError(f"task 15: independent solver counted {solved}, generator {intersections}")

        answer = str(intersections)
        return {
            "input": grid_to_list(grid),
            "output": answer,
            "metadata": {
                "bg_color": int(bg),
                "line_colors": [int(c) for c in line_colors],
                "line_count": len(lines),
                "diagonal_line_count": sum(1 for (lr0, lc0, lr1, lc1) in lines if lr0 != lr1 and lc0 != lc1),
                "intersection_count": intersections,
                "render_mode": render_mode,
                "junction_color": None,
                "distractor_object_count": distractor_count,
                "distractor_object_colors": [int(c) for c in distractor_colors],
                "n_colors": int(len(np.unique(grid))),
                "grid_h": int(H),
                "grid_w": int(W),
                "min_parallel_gap": 2,
                **texture,
            },
        }


# ---------------------------------------------------------------------------
# Task 15a — Count pairwise line intersections (grid -> text)
# ---------------------------------------------------------------------------


_T15A_MULTI_P = 0.20          # layouts built around a cell shared by three or four lines (from 100 cells)
_T15A_THIN_TWO_P = 0.35       # small canvases mostly yield two crossings; some are redrawn at the same size
_T15A_TEXTURE_P = 0.30


def _t15a_hub_line(rng: np.random.Generator, H: int, W: int, kind: str, r: int, c: int):
    """One line of ``kind`` through the hub cell (r, c) with at least two cells on each side."""
    if kind == "h":
        if c < 2 or W - 1 - c < 2:
            return None
        return (r, c - int(rng.integers(2, c + 1)), r, c + int(rng.integers(2, W - c)))
    if kind == "v":
        if r < 2 or H - 1 - r < 2:
            return None
        return (r - int(rng.integers(2, r + 1)), c, r + int(rng.integers(2, H - r)), c)
    if kind == "d":                       # down-diagonal: (r-a, c-a) .. (r+b, c+b)
        amax, bmax = min(r, c), min(H - 1 - r, W - 1 - c)
        if amax < 2 or bmax < 2:
            return None
        a, b = int(rng.integers(2, amax + 1)), int(rng.integers(2, bmax + 1))
        return (r - a, c - a, r + b, c + b)
    amax, bmax = min(r, W - 1 - c), min(H - 1 - r, c)   # up-diagonal: (r-a, c+a) .. (r+b, c-b)
    if amax < 2 or bmax < 2:
        return None
    a, b = int(rng.integers(2, amax + 1)), int(rng.integers(2, bmax + 1))
    return (r - a, c + a, r + b, c - b)


def _t15a_multiplicity_layout(rng: np.random.Generator, H: int, W: int):
    """A hub cell crossed by three or four lines (a subset of horizontal, vertical, the two
    45-degree diagonals) plus up to three further lines that each cross the scene once or
    more, every line running >= 2 cells past every crossing and the geometry guard passing.
    This is what makes task 15a differ from task 15: the hub counts C(k, 2) pairs.
    Returns (lines, line_cell_sets, pairwise) or None."""
    if min(H, W) < 7:
        return None
    for _ in range(20):
        k = 4 if (min(H, W) >= 9 and rng.random() < 0.35) else 3
        kinds = [str(x) for x in rng.choice(np.array(["h", "v", "d", "u"]), size=k, replace=False)]
        r = int(rng.integers(2, H - 2))
        c = int(rng.integers(2, W - 2))
        lines = []
        for kind in kinds:
            ln = _t15a_hub_line(rng, H, W, kind, r, c)
            if ln is None:
                break
            lines.append(ln)
        if len(lines) < k:
            continue
        sets = _line_sets_from_lines(H, W, lines)
        if _t15_layout_flaws(lines, sets, H, W) is not None:
            continue
        n_extra = int(rng.choice([0, 1, 2, 3], p=[0.12, 0.33, 0.33, 0.22])) if H * W >= 150 else int(rng.integers(0, 2))
        for _e in range(n_extra):
            for _t in range(30):
                extra = _generate_strict_line(rng, H, W, min_len=5)
                if extra is None:
                    break
                cells = set(_line_cells(H, W, *extra))
                if (r, c) in cells or any(len(cells & sset) > 1 for sset in sets):
                    continue
                if not any(cells & sset for sset in sets):
                    continue                      # every extra line must cross something
                cand_lines, cand_sets = lines + [extra], sets + [cells]
                if not _lines_extend_past_intersections(cand_lines, cand_sets, H, W, min_extension=2):
                    continue
                if _t15_layout_flaws(cand_lines, cand_sets, H, W) is not None:
                    continue
                lines, sets = cand_lines, cand_sets
                break
        if not _lines_extend_past_intersections(lines, sets, H, W, min_extension=2):
            continue
        if _t15_layout_flaws(lines, sets, H, W) is not None:
            continue
        _n_cells, count_grid = _count_intersection_cells(sets, H, W)
        pairwise = int(np.sum(count_grid * (count_grid - 1) // 2))
        return lines, sets, pairwise
    return None


def _t15_segments_from_grid(grid: np.ndarray, bg: int) -> list[set[tuple[int, int]]]:
    """The straight one-colour segments a solver reads off the visible grid (see
    ``_solve_intersections_from_grid`` for the rule; this is the same reading, kept apart so
    task 15's solver stays byte-for-byte as it was)."""
    segments = []
    for color in np.unique(grid):
        if int(color) == int(bg):
            continue
        cells = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
        if len(cells) < 3:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 == r1:
            line = [(r0, c) for c in range(c0, c1 + 1)]
        elif c0 == c1:
            line = [(r, c0) for r in range(r0, r1 + 1)]
        elif (r1 - r0) == (c1 - c0):
            if all((r - r0) == (c - c0) for r, c in cells):
                line = [(r0 + i, c0 + i) for i in range(r1 - r0 + 1)]
            elif all((r - r0) == (c1 - c) for r, c in cells):
                line = [(r0 + i, c1 - i) for i in range(r1 - r0 + 1)]
            else:
                continue
        else:
            continue
        if len(line) < 5:
            continue
        own = set(cells)
        lset = set(line)
        if not own <= lset:
            continue
        if any(int(grid[r, c]) == int(bg) for r, c in lset - own):
            continue
        segments.append(lset)
    return segments


def _solve_pairwise_from_grid(grid: np.ndarray, bg: int) -> int:
    """Independent solver for task 15a from the visible grid alone: the number of pairs of
    lines that share a cell, i.e. the sum of C(k, 2) over the cells crossed by k lines."""
    from collections import Counter

    count = Counter(cell for s in _t15_segments_from_grid(grid, bg) for cell in s)
    return sum(v * (v - 1) // 2 for v in count.values())


@register("15a")
def generate_count_pairwise_intersections(rng: np.random.Generator) -> dict:
    """Task 15a: count the line crossings counting multiplicity.

    Unlike task 15 (the CELLS on two or more lines), the answer is the number of pairs of
    lines that share a cell: a cell where k lines meet contributes C(k, 2).  The prompt
    states that rule.  Layouts are task 15's (sizes 7..30 with content retried at the same
    size; lattices, mixes, free layouts and constructive rare-count layouts that scale with
    the canvas; the geometry guard: no phantom diagonal crossing, parallel lines >= 2 apart,
    no line end touching another line, >= 2 cells past every crossing) plus, on 20% of
    canvases from 100 cells, a layout built around a hub cell crossed by three or four
    lines so the multiplicity rule is exercised deliberately.  Sparse background-library
    textures on 30% of canvases with a short side >= 8, isolated cells never adjacent to a
    line, covered by the prompt's "ignore isolated non-line distractor objects" sentence;
    nothing about the background is named.  Every answer is recomputed by an independent
    solver from the visible grid; the generator raises on disagreement.
    """
    for _size_attempt in range(12):
        H, W = random_grid_size(rng, min_dim=7, max_dim=30)
        for _content_attempt in range(25):
            bg = pick_color(rng)
            result = None
            if H * W >= 100 and rng.random() < _T15A_MULTI_P:
                result = _t15a_multiplicity_layout(rng, H, W)
            if result is None:
                result = _choose_intersection_layout(rng, H, W, scaled=True)
            if result is None:
                continue
            lines, line_cell_sets, _ = result
            # Each line pair meets in at most one cell, so the pairwise count is exactly the
            # number of pairs of lines that cross.
            if any(len(line_cell_sets[i] & line_cell_sets[j]) > 1
                   for i in range(len(lines)) for j in range(i + 1, len(lines))):
                continue
            n_cells, count_grid = _count_intersection_cells(line_cell_sets, H, W)
            pairwise = int(np.sum(count_grid * (count_grid - 1) // 2))
            if pairwise < 2:
                continue
            if pairwise == 2 and rng.random() < _T15A_THIN_TWO_P:
                continue
            if _t15_layout_flaws(lines, line_cell_sets, H, W) is not None:
                continue
            example = _t15a_finish(rng, H, W, bg, lines, line_cell_sets, n_cells, pairwise, count_grid)
            if example is not None:
                return example
    raise RuntimeError("Task 15a: failed to generate valid example")


def _t15a_finish(rng, H, W, bg, lines, line_cell_sets, n_cells, pairwise, count_grid):
    """Render, decorate and validate one task-15a layout; None when a render check fails."""
    render_mode, line_colors, junction_color, _ = _choose_line_render_palette(
        rng, len(lines), bg, include_highlight=False
    )
    grid, render_count = _render_line_scene(
        H, W, bg, lines, line_cell_sets, line_colors, junction_color
    )
    if int(np.sum(render_count >= 2)) != n_cells:
        return None
    if not _rendered_lines_are_clear(
        grid, line_cell_sets, line_colors, render_count, junction_color
    ):
        return None

    exclude_colors = {int(bg), *[int(c) for c in line_colors]}
    distractor_count, distractor_colors = _maybe_paint_distractor_objects(
        rng, grid, line_cell_sets, exclude_colors
    )
    if not _rendered_lines_are_clear(
        grid, line_cell_sets, line_colors, render_count, junction_color
    ):
        return None

    # Sparse background texture, one cell clear of every line and distractor cell.
    texture = dict(_T15_NO_TEXTURE)
    if min(H, W) >= 8 and rng.random() < _T15A_TEXTURE_P:
        occupied = np.zeros((H, W), dtype=bool)
        for r, c in _expanded_cells({(int(r), int(c)) for r, c in np.argwhere(grid != bg)}, H, W, radius=1):
            occupied[r, c] = True
        painted = BG.apply_background(
            rng, grid, int(bg), occupied, group="sparse",
            exclude=[int(c) for c in line_colors] + [int(c) for c in distractor_colors],
        )
        if painted is not None:
            grid, info = painted
            texture = {"texture_group": info["group"], "texture_kind": info["kind"],
                       "texture_colors": [int(c) for c in info["texture_colors"]],
                       "texture_named": False, "texture_coverage": float(info["coverage"])}

    visible_colors = set(int(v) for v in np.unique(grid) if v != bg)
    if not all(c in visible_colors for c in set(line_colors)):
        return None

    solved = _solve_pairwise_from_grid(grid, int(bg))
    if solved != pairwise:
        raise RuntimeError(f"task 15a: independent solver counted {solved}, generator {pairwise}")

    max_through = int(count_grid.max()) if count_grid.size else 0
    return {
        "input": grid_to_list(grid),
        "output": str(pairwise),
        "metadata": {
            "bg_color": int(bg),
            "line_colors": [int(c) for c in line_colors],
            "line_count": len(lines),
            "diagonal_line_count": sum(1 for (lr0, lc0, lr1, lc1) in lines if lr0 != lr1 and lc0 != lc1),
            "intersection_cell_count": int(n_cells),
            "pairwise_intersection_count": int(pairwise),
            "max_lines_through_a_cell": max_through,
            "cells_with_three_or_more_lines": int(np.sum(count_grid >= 3)),
            "render_mode": render_mode,
            "junction_color": None,
            "distractor_object_count": distractor_count,
            "distractor_object_colors": [int(c) for c in distractor_colors],
            "n_colors": int(len(np.unique(grid))),
            "grid_h": int(H),
            "grid_w": int(W),
            "min_parallel_gap": 2,
            **texture,
        },
    }


# ---------------------------------------------------------------------------
# Task 16 — Color all intersections (grid -> grid)
# ---------------------------------------------------------------------------

_T16_TEXTURE_P = 0.30
_T16_THIN_TWO_P = 0.35         # small canvases mostly yield two crossings; some are redrawn at the same size
_T16_MAX_LINES = NUM_COLORS - 2  # a distinct colour per line, the reserved highlight and the background


def _solve_paint_crossings_from_grid(grid: np.ndarray, bg: int, highlight: int) -> tuple[np.ndarray, int]:
    """Independent re-solver for task 16 from the visible input alone.

    Rebuilds every line from its own colour (the task-15 segment rule: all cells of a
    colour on one straight H / V / 45-degree segment of at least five cells, every gap
    along it a non-background cell), then paints every cell lying on two or more lines
    with ``highlight``.  Returns (expected output, number of painted cells).
    """
    from collections import Counter

    segments = _t15_segments_from_grid(grid, bg)
    count = Counter(cell for seg in segments for cell in seg)
    out = grid.copy()
    painted = 0
    for (r, c), v in count.items():
        if v >= 2:
            out[r, c] = highlight
            painted += 1
    return out, painted


@register("16")
def generate_color_intersections(rng: np.random.Generator) -> dict:
    """Task 16: paint every cell that lies on two or more lines with a named colour.

    The task-15 treatment: sizes 7..30 per side, independent, with content retried at
    the same size; layouts from task 15's scaled chooser (straight / diagonal lattices,
    genuine mixes, free layouts and the constructive layouts that reach the counts a
    lattice cannot produce), so the line budget and the crossing count grow with the
    canvas.  The budget is eight lines: a distinct colour per line plus the reserved
    highlight and the background fill the palette, so the 4 x 5 lattice (twenty
    crossings) is the one task-15 layout left out.  Every layout passes the task-15
    geometry guard (no phantom diagonal crossing between cells, parallel lines at least
    two apart, no line end touching a line it does not share, >= 2 cells past every
    crossing).  Each line has its own
    colour and a crossing cell shows one of its lines' colours; the highlight colour is
    reserved and never appears in the input; 30% of canvases with a short side >= 8
    carry a sparse background-library texture (isolated cells one cell clear of every
    line and distractor, identical in input and output) covered by the prompt's
    "ignore isolated non-line distractor objects" sentence.  Nothing about the
    background is named.  Every output is recomputed by an independent solver from the
    visible input; the generator raises on disagreement.
    """
    for _size_attempt in range(12):
        H, W = random_grid_size(rng, min_dim=7, max_dim=30)
        for _content_attempt in range(25):
            bg = pick_color(rng)
            result = _choose_intersection_layout(rng, H, W, scaled=True, max_lines=_T16_MAX_LINES)
            if result is None:
                continue
            lines, line_cell_sets, intersections = result
            if len(lines) > _T16_MAX_LINES:
                continue                                   # defensive: a ninth line would need a tenth colour
            intersections, _ = _count_intersection_cells(line_cell_sets, H, W)
            if intersections < 2:
                continue
            if intersections == 2 and rng.random() < _T16_THIN_TWO_P:
                continue                                   # keeps the count histogram flatter on small canvases
            if _t15_layout_flaws(lines, line_cell_sets, H, W) is not None:
                continue
            example = _t16_finish(rng, H, W, bg, lines, line_cell_sets, intersections)
            if example is not None:
                return example
    raise RuntimeError("Task 16: failed to generate valid example")


def _t16_finish(rng, H, W, bg, lines, line_cell_sets, intersections):
    """Render, decorate, paint and validate one task-16 layout; None when a render check fails."""
    render_mode, line_colors, junction_color, highlight = _choose_line_render_palette(
        rng, len(lines), bg, include_highlight=True
    )
    if highlight is None:
        return None
    grid, count_grid = _render_line_scene(
        H, W, bg, lines, line_cell_sets, line_colors, junction_color
    )
    if int(np.sum(count_grid >= 2)) != intersections:
        return None
    if not _rendered_lines_are_clear(
        grid, line_cell_sets, line_colors, count_grid, junction_color
    ):
        return None

    exclude_colors = {int(bg), int(highlight), *[int(c) for c in line_colors]}
    distractor_count, distractor_colors = _maybe_paint_distractor_objects(
        rng, grid, line_cell_sets, exclude_colors
    )
    if not _rendered_lines_are_clear(
        grid, line_cell_sets, line_colors, count_grid, junction_color
    ):
        return None

    # Sparse background texture, one cell clear of every line and distractor cell; the
    # texture colour is never a line, distractor or highlight colour, so it survives
    # unchanged in the output (painting touches crossing cells only).
    texture = dict(_T15_NO_TEXTURE)
    if min(H, W) >= 8 and rng.random() < _T16_TEXTURE_P:
        occupied = np.zeros((H, W), dtype=bool)
        for r, c in _expanded_cells({(int(r), int(c)) for r, c in np.argwhere(grid != bg)}, H, W, radius=1):
            occupied[r, c] = True
        painted = BG.apply_background(
            rng, grid, int(bg), occupied, group="sparse",
            exclude=[int(c) for c in line_colors] + [int(c) for c in distractor_colors] + [int(highlight)],
        )
        if painted is not None:
            grid, info = painted
            texture = {"texture_group": info["group"], "texture_kind": info["kind"],
                       "texture_colors": [int(c) for c in info["texture_colors"]],
                       "texture_named": False, "texture_coverage": float(info["coverage"])}

    visible_colors = set(int(v) for v in np.unique(grid) if v != bg)
    if not all(c in visible_colors for c in set(line_colors)):
        return None
    if int(highlight) in set(int(v) for v in np.unique(grid)):
        return None

    out = grid.copy()
    out[count_grid >= 2] = highlight
    solved_out, solved_n = _solve_paint_crossings_from_grid(grid, int(bg), int(highlight))
    if solved_n != intersections or not np.array_equal(solved_out, out):
        raise RuntimeError(f"task 16: independent solver painted {solved_n} cells, generator {intersections}")
    if np.array_equal(grid, out):
        return None

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "highlight_color": int(highlight),
            "bg_color": int(bg),
            "line_colors": [int(c) for c in line_colors],
            "line_count": len(lines),
            "diagonal_line_count": sum(1 for (lr0, lc0, lr1, lc1) in lines if lr0 != lr1 and lc0 != lc1),
            "intersection_count": int(intersections),
            "render_mode": render_mode,
            "junction_color": None,
            "distractor_object_count": distractor_count,
            "distractor_object_colors": [int(c) for c in distractor_colors],
            "n_colors": int(len(np.unique(grid))),
            "n_colors_output": int(len(np.unique(out))),
            "grid_h": int(H),
            "grid_w": int(W),
            "min_parallel_gap": 2,
            **texture,
        },
    }


# ---------------------------------------------------------------------------
# Task 73 — Distance between parallel lines (grid -> text)
# ---------------------------------------------------------------------------


@register("73")
def generate_parallel_distance(rng: np.random.Generator) -> dict:
    from generators.parallel_geometry_73_74 import generate_measure_parallel as build

    return build(rng)


# ---------------------------------------------------------------------------
# Task 74 — Draw offset parallel line (grid -> grid)
# ---------------------------------------------------------------------------


@register("74")
def generate_draw_parallel(rng: np.random.Generator) -> dict:
    from generators.parallel_geometry_73_74 import generate_draw_parallel as build

    return build(rng)


# ---------------------------------------------------------------------------
# Task 71 — Draw diamond of given corner-to-corner length (grid -> grid)
# ---------------------------------------------------------------------------


def _draw_diamond(grid: np.ndarray, cr: int, cc: int, r: int, color: int) -> None:
    """Draw a filled diamond: all cells where |row-cr| + |col-cc| <= r."""
    H, W = grid.shape
    for dr in range(-r, r + 1):
        row = cr + dr
        if row < 0 or row >= H:
            continue
        spread = r - abs(dr)
        for dc in range(-spread, spread + 1):
            col = cc + dc
            if 0 <= col < W:
                grid[row, col] = color


def _diamond_cells(cr: int, cc: int, r: int) -> set[tuple[int, int]]:
    """Return the (possibly off-grid) cells of a filled diamond, before clipping."""
    cells: set[tuple[int, int]] = set()
    for dr in range(-r, r + 1):
        spread = r - abs(dr)
        for dc in range(-spread, spread + 1):
            cells.add((cr + dr, cc + dc))
    return cells


@register("71")
def generate_draw_diamond(rng: np.random.Generator) -> dict:
    from generators.diamond_geometry_71_72 import generate_draw_diamonds as build

    return build(rng)


# ---------------------------------------------------------------------------
# Task 72 — Infer diamond corner-to-corner length (grid -> text)
# ---------------------------------------------------------------------------


@register("72")
def generate_infer_diamond_length(rng: np.random.Generator) -> dict:
    from generators.diamond_geometry_71_72 import generate_measure_diamond as build

    return build(rng)
