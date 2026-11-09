"""Miscellaneous task generators: tasks 26, 27, 28, 31, 35, 56, 57, 58, 59, 80, 82, 83."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import NUM_COLORS, make_grid
from common.objects import DELTAS_4
from registry import register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_colors(rng: np.random.Generator, n: int, exclude: list[int] | None = None) -> list[int]:
    """Pick *n* distinct colors not in *exclude*."""
    if exclude is None:
        exclude = []
    available = [c for c in range(NUM_COLORS) if c not in exclude]
    chosen = rng.choice(available, size=min(n, len(available)), replace=False)
    return [int(c) for c in chosen]


def _rects_overlap(r1: tuple, r2: tuple, spacing: int = 0) -> bool:
    """Check overlap between two (row, col, h, w) rects with spacing."""
    r1r, r1c, r1h, r1w = r1
    r2r, r2c, r2h, r2w = r2
    if r1r + r1h + spacing <= r2r or r2r + r2h + spacing <= r1r:
        return False
    if r1c + r1w + spacing <= r2c or r2c + r2w + spacing <= r1c:
        return False
    return True


def _pick_unused_color_from_grid(
    grid: np.ndarray,
    exclude: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> int | None:
    """Pick a color that does not currently appear in *grid*."""
    excluded = set(exclude or [])
    used = set(int(v) for v in np.unique(grid))
    candidates = [color for color in range(NUM_COLORS) if color not in used and color not in excluded]
    if not candidates:
        return None
    if rng is not None:
        return int(rng.choice(candidates))
    return candidates[0]
    return None


def _decorate_background_canvas(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    exclude_colors: set[int],
    *,
    style: str = "simple",
) -> np.ndarray:
    """Create a simple repeating canvas using colors that are not task objects."""
    canvas = make_grid(h, w, fill=bg)
    if style == "plain":
        return canvas

    accent_candidates = [
        c for c in range(NUM_COLORS)
        if c != bg and c not in exclude_colors
    ]
    if not accent_candidates or rng.random() < 0.6:
        return canvas

    accents = [int(c) for c in rng.choice(accent_candidates, size=min(2, len(accent_candidates)), replace=False)]
    primary = accents[0]
    secondary = accents[-1]
    mode = str(rng.choice(["checker", "stripes"]))

    if mode == "checker":
        block = int(rng.integers(1, 3))
        for r in range(h):
            for c in range(w):
                if ((r // block) + (c // block)) % 2 == 0:
                    canvas[r, c] = primary
                elif style == "regular" and secondary != primary:
                    canvas[r, c] = secondary
    else:
        vertical = bool(rng.integers(0, 2))
        period = int(rng.integers(2, 4))
        for r in range(h):
            for c in range(w):
                idx = c if vertical else r
                if idx % period == 0:
                    canvas[r, c] = primary if (idx // period) % 2 == 0 else secondary

    return canvas


def _reachable_background_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    """Return background cells reachable from the border via 4-neighbors."""
    h, w = grid.shape
    reached = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if grid[r, c] != bg or reached[r, c]:
                continue
            reached[r, c] = True
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DELTAS_4:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not reached[nr, nc]
                and grid[nr, nc] == bg
            ):
                reached[nr, nc] = True
                queue.append((nr, nc))
    return reached


def _count_enclosed_regions(grid: np.ndarray, bg: int) -> int:
    """Count 4-connected background regions that are not border-reachable."""
    h, w = grid.shape
    reached = _reachable_background_mask(grid, bg)
    interior = (grid == bg) & (~reached)
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not interior[r, c] or seen[r, c]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and interior[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _closure_candidates(grid: np.ndarray, obj_color: int, bg: int) -> list[tuple[int, int]]:
    """Return background cells whose addition would close the open enclosure."""
    h, w = grid.shape
    candidates: set[tuple[int, int]] = set()
    obj_cells = set(zip(*np.where(grid == obj_color)))
    for r, c in obj_cells:
        for dr, dc in DELTAS_4:
            nr, nc = r + dr, c + dc
            if 0 < nr < h - 1 and 0 < nc < w - 1 and grid[nr, nc] == bg:
                candidates.add((nr, nc))

    valid: list[tuple[int, int]] = []
    for nr, nc in sorted(candidates):
        trial = grid.copy()
        trial[nr, nc] = obj_color
        if _count_enclosed_regions(trial, bg) > 0:
            valid.append((nr, nc))
    return valid


def _dilate_cells(
    cells: set[tuple[int, int]],
    steps: int,
    *,
    connectivity: int = 4,
) -> set[tuple[int, int]]:
    """Dilate a cell set by repeated 4- or 8-neighbor expansion."""
    deltas = DELTAS_4 if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]
    dilated = set(cells)
    frontier = set(cells)
    for _ in range(steps):
        new_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in deltas:
                new_cell = (r + dr, c + dc)
                if new_cell not in dilated:
                    dilated.add(new_cell)
                    new_frontier.add(new_cell)
        frontier = new_frontier
    return dilated


def _erode_cells(cells: set[tuple[int, int]], thickness: int) -> set[tuple[int, int]]:
    """Remove *thickness* many 4-neighbor boundary layers from a cell set."""
    interior = set(cells)
    for _ in range(thickness):
        boundary = {
            (r, c)
            for (r, c) in interior
            if any((r + dr, c + dc) not in interior for dr, dc in DELTAS_4)
        }
        interior -= boundary
        if not interior:
            break
    return interior


# ---------------------------------------------------------------------------
# Task 26 — Extract the interior of a unique rectangular frame
# ---------------------------------------------------------------------------

from generators.crop_frame_26 import generate_extract_subgrid

register("26")(generate_extract_subgrid)


# ---------------------------------------------------------------------------
# Task 27 — Pack connected shapes by cell area against a specified edge
# ---------------------------------------------------------------------------



from generators.packing_27 import generate_pack_shapes as generate_pack_rects_bottom

register("27")(generate_pack_rects_bottom)


# ---------------------------------------------------------------------------
# Task 28 — Fit objects into matching holes
# ---------------------------------------------------------------------------

from generators.fit_holes_28 import generate_fit_holes as generate_fit_into_holes

register("28")(generate_fit_into_holes)


# ---------------------------------------------------------------------------
# Task 31 — Move objects to border by gravity (nearest border)
# ---------------------------------------------------------------------------

@register("31")
def generate_move_to_nearest_border(rng: np.random.Generator) -> dict:
    from generators.gravity_31 import generate_move_to_nearest_border as generate_impl
    return generate_impl(rng)


# ---------------------------------------------------------------------------
# Task 35 — Count target-color cells inside rectangle
# ---------------------------------------------------------------------------

@register("35")
def generate_count_color_in_roi(rng: np.random.Generator) -> dict:
    from generators.count_frame_35 import generate_count_color_in_roi as generate_impl
    return generate_impl(rng)


# ---------------------------------------------------------------------------
# Task 56 — Does an object touch a target cell?
# ---------------------------------------------------------------------------

@register("56")
def generate_object_touches_target(rng: np.random.Generator) -> dict:
    from generators.contact_56 import generate
    return generate(rng)


@register("57")
def generate_most_frequent_in_enclosure(rng: np.random.Generator) -> dict:
    from generators.enclosure_frequency_57 import generate
    return generate(rng)


@register("58")
def generate_highlight_shape_in_noise(rng: np.random.Generator) -> dict:
    from generators.shape_search_58_59 import generate
    return generate(rng, counting=False)


@register("59")
def generate_count_shape_in_noise(rng: np.random.Generator) -> dict:
    from generators.shape_search_58_59 import generate
    return generate(rng, counting=True)


# ---------------------------------------------------------------------------
# Task 80 — Return one half of a grid
# ---------------------------------------------------------------------------

@register("80")
def generate_return_half(rng: np.random.Generator) -> dict:
    """Extract a feasible requested half after finishing the input drawing."""
    from generators.grid_reading_76_80 import generate_half
    return generate_half(rng)


# ---------------------------------------------------------------------------
# Task 82 — Draw thick outline around object
# ---------------------------------------------------------------------------

@register("82")
def generate_thick_outline(rng: np.random.Generator) -> dict:
    """Draw a requested outline around a finished, freely placed scene."""
    from generators.outline_82 import generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 83 — Fill object leaving outer shell
# ---------------------------------------------------------------------------

@register("83")
def generate_fill_leaving_shell(rng: np.random.Generator) -> dict:
    """Task 83: Fill interiors under the requested edge/corner boundary rule."""
    from generators.interior_83 import generate_fill_leaving_shell as generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 91 — Pack objects against one edge
#
# Rebuilt on 2026-09-15 and moved to generators/pack_edge_91.py, which
# registers the task itself (registry.py imports it).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 92 — Count objects that touch another object
#
# Rebuilt on 2026-09-15 and moved to generators/touching_92.py, which
# registers the task itself (registry.py imports it).  Task 92b is merged
# into it: the connectivity is drawn per example.
# ---------------------------------------------------------------------------
