"""Region and boundary generators for selected second-batch tasks."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import grid_to_list, make_grid, pick_color
from registry import register

from .utils import MAX_RETRIES, add_patterned_background, unused_colors


Cell = tuple[int, int]
DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DELTAS_8 = DELTAS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _deltas(connectivity: int) -> list[Cell]:
    return DELTAS_8 if int(connectivity) == 8 else DELTAS_4


def _in_bounds(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(grid: np.ndarray, cell: Cell, connectivity: int = 4) -> list[Cell]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr, dc in _deltas(connectivity)
        if _in_bounds(grid, r + dr, c + dc)
    ]


def _reachable(
    grid: np.ndarray,
    starts: set[Cell],
    passable,
    connectivity: int = 4,
    max_distance: int | None = None,
) -> set[Cell]:
    seen: set[Cell] = set()
    q: deque[tuple[Cell, int]] = deque()
    for cell in sorted(starts):
        if passable(cell):
            seen.add(cell)
            q.append((cell, 0))
    while q:
        cell, dist = q.popleft()
        if max_distance is not None and dist >= max_distance:
            continue
        for nb in _neighbors(grid, cell, connectivity):
            if nb not in seen and passable(nb):
                seen.add(nb)
                q.append((nb, dist + 1))
    return seen


def _components_of_color(grid: np.ndarray, color: int, connectivity: int = 4) -> list[set[Cell]]:
    seen: set[Cell] = set()
    comps: list[set[Cell]] = []
    for r, c in np.argwhere(grid == color):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        comp = _reachable(grid, {cell}, lambda p: int(grid[p]) == int(color), connectivity)
        seen |= comp
        comps.append(comp)
    return comps


def _background_components(
    grid: np.ndarray,
    background_color: int,
    connectivity: int = 4,
) -> list[set[Cell]]:
    return _components_of_color(grid, background_color, connectivity)


def _border_background_reachable(
    grid: np.ndarray,
    background_color: int,
    barrier_colors: set[int],
    connectivity: int = 4,
) -> set[Cell]:
    h, w = grid.shape
    starts = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1))
    }
    return _reachable(
        grid,
        starts,
        lambda p: int(grid[p]) == background_color and int(grid[p]) not in barrier_colors,
        connectivity,
    )


def _fill_reachable_region(
    grid: np.ndarray,
    seed_color: int,
    background_color: int,
    wall_color: int,
    fill_color: int,
    connectivity: int,
) -> np.ndarray:
    seed_cells = {tuple(int(v) for v in p) for p in np.argwhere(grid == seed_color)}
    fillable = {background_color, seed_color}
    reachable = _reachable(
        grid,
        seed_cells,
        lambda p: int(grid[p]) in fillable and int(grid[p]) != wall_color,
        connectivity,
    )
    out = grid.copy()
    for cell in reachable:
        out[cell] = fill_color
    return out


def _expand_target_objects(
    grid: np.ndarray,
    target_color: int,
    protected_colors: set[int],
    distance: int,
    connectivity: int,
) -> np.ndarray:
    starts = {tuple(int(v) for v in p) for p in np.argwhere(grid == target_color)}
    expanded = _reachable(
        grid,
        starts,
        lambda p: int(grid[p]) not in protected_colors,
        connectivity,
        max_distance=distance,
    )
    out = grid.copy()
    for cell in expanded:
        if int(grid[cell]) not in protected_colors:
            out[cell] = target_color
    return out


def _hole_fill_threshold(
    grid: np.ndarray,
    background_color: int,
    target_color: int,
    fill_color: int,
    threshold: int,
    condition: str,
    connectivity: int,
) -> np.ndarray:
    outside = _border_background_reachable(grid, background_color, set(), connectivity)
    out = grid.copy()
    for comp in _background_components(grid, background_color, connectivity):
        if comp & outside:
            continue
        touching = {
            int(grid[nb])
            for cell in comp
            for nb in _neighbors(grid, cell, 4)
            if int(grid[nb]) != background_color
        }
        if touching != {int(target_color)}:
            continue
        area = len(comp)
        ok = area <= threshold if condition == "at_most" else area >= threshold
        if ok:
            for cell in comp:
                out[cell] = fill_color
    return out


def _articulation_cells(grid: np.ndarray, target_color: int, connectivity: int) -> set[Cell]:
    comps = _components_of_color(grid, target_color, connectivity)
    if not comps:
        return set()
    object_cells = max(comps, key=len)
    connectors: set[Cell] = set()
    for removed in object_cells:
        remaining = object_cells - {removed}
        if len(remaining) < 2:
            continue
        start = min(remaining)
        seen = {start}
        q = deque([start])
        while q:
            cell = q.popleft()
            for dr, dc in _deltas(connectivity):
                nb = (cell[0] + dr, cell[1] + dc)
                if nb in remaining and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if seen != remaining:
            connectors.add(removed)
    return connectors


def _mark_connectors(
    grid: np.ndarray,
    target_color: int,
    connector_color: int,
    connectivity: int,
) -> np.ndarray:
    out = grid.copy()
    for cell in _articulation_cells(grid, target_color, connectivity):
        out[cell] = connector_color
    return out


def _mark_corners(
    grid: np.ndarray,
    target_color: int,
    outer_corner_color: int,
    inner_corner_color: int,
) -> np.ndarray:
    object_cells = {tuple(int(v) for v in p) for p in np.argwhere(grid == target_color)}
    pairs = [
        ((-1, 0), (0, -1), (-1, -1)),
        ((-1, 0), (0, 1), (-1, 1)),
        ((1, 0), (0, -1), (1, -1)),
        ((1, 0), (0, 1), (1, 1)),
    ]
    out = grid.copy()
    for r, c in sorted(object_cells):
        outer = False
        inner = False
        for (dr1, dc1), (dr2, dc2), (ddr, ddc) in pairs:
            side1 = (r + dr1, c + dc1) in object_cells
            side2 = (r + dr2, c + dc2) in object_cells
            diag = (r + ddr, c + ddc) in object_cells
            outer = outer or (not side1 and not side2)
            inner = inner or (side1 and side2 and not diag)
        if inner:
            out[r, c] = inner_corner_color
        elif outer:
            out[r, c] = outer_corner_color
    return out


def _classify_background(
    grid: np.ndarray,
    background_color: int,
    barrier_colors: set[int],
    outside_color: int,
    inside_color: int,
) -> np.ndarray:
    outside = _border_background_reachable(grid, background_color, barrier_colors)
    out = grid.copy()
    for r, c in np.argwhere(grid == background_color):
        cell = (int(r), int(c))
        out[cell] = outside_color if cell in outside else inside_color
    return out


def _fill_holes_with_surrounding_color(grid: np.ndarray, background_color: int) -> np.ndarray:
    outside = _border_background_reachable(grid, background_color, set())
    out = grid.copy()
    for comp in _background_components(grid, background_color, 4):
        if comp & outside:
            continue
        touching = {
            int(grid[nb])
            for cell in comp
            for nb in _neighbors(grid, cell, 4)
            if int(grid[nb]) != background_color
        }
        if len(touching) == 1:
            fill_color = next(iter(touching))
            for cell in comp:
                out[cell] = fill_color
    return out


def _boundary_layer(
    grid: np.ndarray,
    target_color: int,
    layer_index: int,
    connectivity: int,
) -> set[Cell]:
    remaining = {tuple(int(v) for v in p) for p in np.argwhere(grid == target_color)}
    layer: set[Cell] = set()
    for _ in range(layer_index):
        layer = {
            (r, c)
            for r, c in remaining
            if any((r + dr, c + dc) not in remaining for dr, dc in _deltas(connectivity))
        }
        remaining -= layer
        if not layer:
            break
    return layer


def _mark_boundary_layer(
    grid: np.ndarray,
    target_color: int,
    layer_color: int,
    layer_index: int,
    connectivity: int,
) -> np.ndarray:
    out = grid.copy()
    for cell in _boundary_layer(grid, target_color, layer_index, connectivity):
        out[cell] = layer_color
    return out


def _neighbor_count_transform(
    grid: np.ndarray,
    eligible_color: int,
    neighbor_color: int,
    marker_color: int,
    connectivity: int,
    condition: str,
    count: int,
) -> np.ndarray:
    out = grid.copy()
    for r, c in np.argwhere(grid == eligible_color):
        cell = (int(r), int(c))
        n = sum(1 for nb in _neighbors(grid, cell, connectivity) if int(grid[nb]) == neighbor_color)
        ok = n == count if condition == "exactly" else n >= count
        if ok:
            out[cell] = marker_color
    return out


def _draw_rect_outline(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color


@register("111")
def generate_reachable_seed_fill(rng: np.random.Generator) -> dict:
    # Independent H/W. Lower bound 8 is the structural minimum: the gate needs
    # gate_r=integers(4, h-3) to be valid (h >= 8) and a non-empty chamber.
    # Upper bound 22 keeps a single grid->grid example inside the token budget.
    h = int(rng.integers(8, 23))
    w = int(rng.integers(8, 23))
    bg = int(rng.integers(0, 10))
    wall, seed_color, fill, distractor = unused_colors(rng, 4, bg)
    connectivity = int(rng.choice([4, 8]))
    grid = make_grid(h, w, fill=bg)

    gate_r = int(rng.integers(4, h - 3))
    gate_c = int(rng.integers(4, w - 3))
    grid[:gate_r, gate_c] = wall
    grid[gate_r, :gate_c] = wall
    seed_row = int(rng.integers(1, gate_r))
    seed_col = int(rng.integers(1, gate_c))
    grid[seed_row, seed_col] = seed_color

    diagonal_gate = {(gate_r - 1, gate_c - 1), (gate_r, gate_c)}
    chamber_cells = [
        (r, c)
        for r in range(gate_r)
        for c in range(gate_c)
        if int(grid[r, c]) == bg and (r, c) not in diagonal_gate
    ]
    outside_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == bg and (r >= gate_r or c >= gate_c) and (r, c) not in diagonal_gate
    ]
    for pool in (chamber_cells, outside_cells):
        for _ in range(min(int(rng.integers(2, 5)), len(pool))):
            idx = int(rng.integers(0, len(pool)))
            cell = pool.pop(idx)
            grid[cell] = distractor

    grid = np.rot90(grid, int(rng.integers(0, 4))).copy()
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(_fill_reachable_region(grid, seed_color, bg, wall, fill, connectivity)),
        "metadata": {
            "background_color": bg,
            "wall_color": int(wall),
            "seed_color": int(seed_color),
            "fill_color": int(fill),
            "connectivity": connectivity,
            "fillable_colors": ["background_color", "seed_color"],
            "distractor_colors": [int(distractor)],
            "seed_rule": "the unique seed-color cell",
        },
    }


@register("113")
def generate_expand_targets_protected(rng: np.random.Generator) -> dict:
    # Independent H/W. Lower bound 8 keeps the two target seed blocks
    # (rows 2:4 and h-4:h-2) and both protected colors placeable; both the
    # reached and unreached background-decoy pools stay non-empty down there.
    # Upper bound 22 keeps a single grid->grid example inside the token budget.
    h = int(rng.integers(8, 23))
    w = int(rng.integers(8, 23))
    bg = int(rng.integers(0, 10))
    target, protected_a, protected_b, distractor = unused_colors(rng, 4, bg)
    distance = int(rng.integers(1, 4))
    connectivity = int(rng.choice([4, 8]))
    grid = make_grid(h, w, fill=bg)
    grid[2:4, 2:4] = target
    grid[h - 4:h - 2, w - 4:w - 2] = target
    if rng.random() < 0.5:
        grid[1:h - 1, w // 2] = protected_a
    else:
        grid[h // 2, 1:w - 1] = protected_a
    grid[h // 2 - 1:h // 2 + 2, w // 2 - 1:w // 2 + 2] = protected_b
    grid[4, 2] = protected_a
    grid[h - 5, w - 3] = protected_b

    protected_colors = {int(protected_a), int(protected_b)}
    starts = {tuple(int(v) for v in p) for p in np.argwhere(grid == target)}
    reached = _reachable(
        grid,
        starts,
        lambda p: int(grid[p]) not in protected_colors,
        connectivity,
        max_distance=distance,
    )
    reached_candidates = sorted(cell for cell in reached if int(grid[cell]) == bg)
    unreached_candidates = sorted(
        (int(r), int(c))
        for r, c in np.argwhere(grid == bg)
        if (int(r), int(c)) not in reached
    )
    for pool in (reached_candidates, unreached_candidates):
        for _ in range(min(2, len(pool))):
            idx = int(rng.integers(0, len(pool)))
            cell = pool.pop(idx)
            grid[cell] = distractor

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(_expand_target_objects(grid, target, protected_colors, distance, connectivity)),
        "metadata": {
            "background_color": bg,
            "target_color": int(target),
            "protected_colors": [int(protected_a), int(protected_b)],
            "distractor_colors": [int(distractor)],
            "distance": distance,
            "connectivity": connectivity,
            "overwrite_rule": "only non-protected cells may be recolored",
        },
    }


def _place_simple_frame(
    grid: np.ndarray,
    rng: np.random.Generator,
    side: int,
    color: int,
    bg: int,
    placed_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Place a hollow square frame of the given side on free background space."""
    h, w = grid.shape
    if side + 1 > h or side + 1 > w:
        return None
    for _ in range(120):
        r0 = int(rng.integers(0, h - side + 1))
        c0 = int(rng.integers(0, w - side + 1))
        box = (r0 - 1, c0 - 1, r0 + side, c0 + side)  # 1-cell margin
        if any(
            not (box[2] < o[0] or o[2] < box[0] or box[3] < o[1] or o[3] < box[1])
            for o in placed_boxes
        ):
            continue
        block = grid[r0:r0 + side, c0:c0 + side]
        if not np.all(block == bg):
            continue
        _draw_rect_outline(grid, r0, c0, side, side, color)
        placed_boxes.append((r0, c0, r0 + side - 1, c0 + side - 1))
        return (r0, c0, r0 + side - 1, c0 + side - 1)
    return None


@register("114")
def generate_fill_threshold_holes(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        target, fill, distractor = unused_colors(rng, 3, bg)
        connectivity = int(rng.choice([4, 8]))
        condition = str(rng.choice(["at_most", "at_least"]))

        # The complex (subdivided) frame goes top-left; its size drives the
        # large target hole.  The matching distractor decoy must satisfy the
        # area condition, so the "at_least" case needs a large decoy hole and
        # hence a larger grid, while "at_most" fits in a smaller grid.  Size
        # the grid to the chosen frames (independent H/W) so the envelope
        # widens both down and up instead of being fixed in the high 20s.
        fh = int(rng.integers(9, 13))
        fw = int(rng.integers(9, 13))
        simple_frame = 5
        distractor_frame = 5 if condition == "at_most" else int(rng.integers(9, 12))

        # Vertical stack: complex frame, then a row holding the simple + decoy.
        bottom_band = max(simple_frame, distractor_frame)
        min_h = fh + bottom_band + 4
        min_w = max(fw, simple_frame + distractor_frame + 3) + 2
        h_cap = 26
        w_cap = 26
        if min_h > h_cap or min_w > w_cap:
            continue
        h = int(rng.integers(min_h, min(h_cap, min_h + 8) + 1))
        w = int(rng.integers(min_w, min(w_cap, min_w + 8) + 1))
        grid = make_grid(h, w, fill=bg)
        placed_boxes: list[tuple[int, int, int, int]] = []

        r0 = 1
        c0 = 1
        _draw_rect_outline(grid, r0, c0, fh, fw, target)
        placed_boxes.append((r0, c0, r0 + fh - 1, c0 + fw - 1))

        interior_h = fh - 2
        interior_w = fw - 2
        top_h = int(rng.integers(2, interior_h - 2))
        left_w = int(rng.integers(2, interior_w - 2))
        interior_top = r0 + 1
        interior_left = c0 + 1
        grid[interior_top:interior_top + top_h, interior_left + left_w] = target
        grid[interior_top + top_h, interior_left:interior_left + left_w] = target

        corner_hole_area = top_h * left_w
        merged_main_area = interior_h * interior_w - top_h - left_w
        rest_hole_area = merged_main_area - corner_hole_area
        if rest_hole_area <= 0:
            continue

        # Simple target frame (a positive case under either connectivity).
        simple_box = _place_simple_frame(grid, rng, simple_frame, target, bg, placed_boxes)
        simple_hole_area = (simple_frame - 2) * (simple_frame - 2)
        if simple_box is None:
            continue

        # Distractor frame whose enclosed hole touches only the distractor
        # color: it satisfies the area condition but must be ignored.
        distractor_box = _place_simple_frame(grid, rng, distractor_frame, distractor, bg, placed_boxes)
        distractor_hole_area = (distractor_frame - 2) * (distractor_frame - 2)
        if distractor_box is None:
            continue

        if connectivity == 8:
            small = simple_hole_area
            large = merged_main_area
        else:
            small = min(corner_hole_area, rest_hole_area)
            large = max(corner_hole_area, rest_hole_area)
        threshold = int((small + large) // 2)
        if not small < threshold < large:
            continue
        if condition == "at_most" and distractor_hole_area > threshold:
            continue
        if condition == "at_least" and distractor_hole_area < threshold:
            continue

        # Vary the number of objects: optionally add extra distractor decoy
        # frames in free space.  They never touch the target color, so they are
        # ignored holes and do not change the target/decoy classification.
        n_extra = int(rng.integers(0, 3))
        for _ in range(n_extra):
            extra_side = int(rng.choice([4, 5, 6]))
            _place_simple_frame(grid, rng, extra_side, distractor, bg, placed_boxes)

        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(_hole_fill_threshold(grid, bg, target, fill, threshold, condition, connectivity)),
            "metadata": {
                "background_color": bg,
                "target_color": int(target),
                "fill_color": int(fill),
                "distractor_colors": [int(distractor)],
                "connectivity": connectivity,
                "area_condition": condition,
                "area_threshold": threshold,
                "hole_rule": "background components not border-reachable and touching only the target color by edge",
            },
        }
    raise RuntimeError("Task 114: failed to generate valid example")


def _scatter_solid_blobs(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    bg: int,
    n: int,
    forbidden: set[Cell],
) -> int:
    """Place up to n small solid rectangles on bg cells, avoiding `forbidden`
    cells and a 1-cell halo around them.  Solid blobs enclose no background, so
    they never create a spurious negative-space region."""
    h, w = grid.shape
    halo = {
        (r + dr, c + dc)
        for r, c in forbidden
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    placed = 0
    for _ in range(n):
        for _ in range(60):
            bh = int(rng.integers(1, 4))
            bw = int(rng.integers(1, 4))
            r0 = int(rng.integers(0, h - bh + 1))
            c0 = int(rng.integers(0, w - bw + 1))
            cells = {(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)}
            ring = {
                (r + dr, c + dc)
                for r, c in cells
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }
            if cells & halo or ring & forbidden:
                continue
            if not all(grid[r, c] == bg for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = color
            placed += 1
            break
    return placed


@register("124")
def generate_extract_negative_space(rng: np.random.Generator) -> dict:
    # Independent H/W widened down to the structural minimum (a 6-tall frame /
    # 5-tall bars need h,w >= 8) and up to a grid->grid token-safe cap.
    h = int(rng.integers(8, 23))
    w = int(rng.integers(8, 23))
    bg = int(rng.integers(0, 10))
    fg, out_color, distractor = unused_colors(rng, 3, bg)
    mode = str(rng.choice(["inside_frame", "between_objects"]))
    grid = make_grid(h, w, fill=bg)
    mask: set[Cell] = set()
    # Cells the negative-space rule depends on; distractors must avoid these.
    forbidden: set[Cell] = set()
    if mode == "inside_frame":
        fh = int(rng.integers(6, h - 1))
        fw = int(rng.integers(6, w - 1))
        r0 = int(rng.integers(1, h - fh))
        c0 = int(rng.integers(1, w - fw))
        _draw_rect_outline(grid, r0, c0, fh, fw, fg)
        mask = {
            (r, c)
            for r in range(r0 + 1, r0 + fh - 1)
            for c in range(c0 + 1, c0 + fw - 1)
        }
        # Forbid the whole frame box (walls + interior) for distractors.
        forbidden = {
            (r, c)
            for r in range(r0, r0 + fh)
            for c in range(c0, c0 + fw)
        }
    else:
        top = int(rng.integers(1, max(2, min(5, h - 4))))
        height = int(rng.integers(3, h - top - 1))
        left_c = int(rng.integers(1, max(2, min(4, w - 5))))
        max_gap = (w - 3) - (left_c + 2)
        gap = int(rng.integers(1, max(2, min(6, max_gap + 1))))
        right_c = left_c + 2 + gap
        grid[top:top + height, left_c:left_c + 2] = fg
        grid[top:top + height, right_c:right_c + 2] = fg
        mask = {
            (r, c)
            for r in range(top, top + height)
            for c in range(left_c + 2, right_c)
        }
        # Forbid every row that holds a bar (so distractors cannot add fg-row
        # columns) and the whole gap column band, plus the bar columns.
        forbidden = {
            (r, c)
            for r in range(top, top + height)
            for c in range(left_c, right_c + 2)
        }
    # Vary the number of objects with ignored solid distractor blobs that the
    # output never shows (output is background everywhere except the mask).
    n_distractor = _scatter_solid_blobs(
        grid, rng, int(distractor), bg, int(rng.integers(0, 3)), forbidden
    )
    out = make_grid(h, w, fill=bg)
    for cell in mask:
        out[cell] = out_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "foreground_color": int(fg),
            "output_color": int(out_color),
            "distractor_colors": [int(distractor)] if n_distractor else [],
            "negative_space_mode": mode,
            "output_rule": "same-size mask on background color",
        },
    }


def _draw_corridor(grid: np.ndarray, a: Cell, b: Cell, color: int) -> None:
    """Draw a width-1 L-shaped (horizontal-then-vertical) corridor between a and b."""
    (r0, c0), (r1, c1) = a, b
    lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
    grid[r0, lo:hi + 1] = color
    lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
    grid[lo:hi + 1, c1] = color


@register("125")
def generate_mark_connector_cells(rng: np.random.Generator) -> dict:
    connectivity = 4
    for _ in range(MAX_RETRIES):
        # Independent H/W; lower bound 9 still fits >=2 spaced blobs plus
        # corridors, upper bound 22 keeps a single grid->grid example in budget.
        h = int(rng.integers(9, 23))
        w = int(rng.integers(9, 23))
        bg = int(rng.integers(0, 10))
        target, connector, distractor = unused_colors(rng, 3, bg)
        grid = make_grid(h, w, fill=bg)

        num_blobs = int(rng.integers(2, 5))
        blobs = []  # each blob: (top, left, height, width, center_cell)
        attempts = 0
        while len(blobs) < num_blobs and attempts < 60:
            attempts += 1
            bh = int(rng.integers(2, 5))
            bw = int(rng.integers(2, 5))
            r0 = int(rng.integers(1, h - bh - 1))
            c0 = int(rng.integers(1, w - bw - 1))
            # keep a 2-cell margin so corridors do not accidentally fuse blobs
            overlaps = any(
                r0 - 2 <= br0 + brh - 1 and br0 - 2 <= r0 + bh - 1
                and c0 - 2 <= bc0 + bcw - 1 and bc0 - 2 <= c0 + bw - 1
                for (br0, bc0, brh, bcw, _) in blobs
            )
            if overlaps:
                continue
            center = (r0 + bh // 2, c0 + bw // 2)
            blobs.append((r0, c0, bh, bw, center))
        if len(blobs) < 2:
            continue
        for (r0, c0, bh, bw, _) in blobs:
            grid[r0:r0 + bh, c0:c0 + bw] = target

        # Spanning tree of corridors: connect each blob to a previously placed one.
        centers = [b[4] for b in blobs]
        order = list(range(len(blobs)))
        rng.shuffle(order)
        for i in range(1, len(order)):
            cur = order[i]
            prev = order[int(rng.integers(0, i))]
            _draw_corridor(grid, centers[cur], centers[prev], target)

        # Optionally add an extra corridor to create a cycle, so some bridge
        # cells are NOT articulation points.
        if len(blobs) >= 3 and rng.random() < 0.5:
            a, b = rng.choice(len(blobs), size=2, replace=False)
            _draw_corridor(grid, centers[int(a)], centers[int(b)], target)

        comps = _components_of_color(grid, target, connectivity)
        if len(comps) != 1:
            continue
        connectors = _articulation_cells(grid, target, connectivity)
        if not connectors:
            continue

        # Vary the object count with ignored solid distractor blobs of a
        # separate color, placed (with a 1-cell halo) clear of the target so
        # the target stays a single component and only its articulation cells
        # are recolored.
        target_cells = {tuple(int(v) for v in p) for p in np.argwhere(grid == target)}
        n_distractor = _scatter_solid_blobs(
            grid, rng, int(distractor), bg, int(rng.integers(0, 3)), target_cells
        )

        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(_mark_connectors(grid, target, connector, connectivity)),
            "metadata": {
                "background_color": bg,
                "target_color": int(target),
                "connector_color": int(connector),
                "connectivity": connectivity,
                "num_blobs": len(blobs),
                "distractor_colors": [int(distractor)] if n_distractor else [],
                "split_rule": "removal creates more than one connected component",
            },
        }
    raise RuntimeError("Task 125: failed to generate valid example")


@register("126")
def generate_mark_boundary_corners(rng: np.random.Generator) -> dict:
    # Independent H/W widened down to ~10 (smallest that fits a 7-wide notched
    # rectangle with margin) and up to a grid->grid token-safe cap.  The target
    # rectangle itself stays a modest 7..11 and simply floats with more
    # background margin on larger grids.
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    bg = int(rng.integers(0, 10))
    target, outer, inner = unused_colors(rng, 3, bg)
    grid = make_grid(h, w, fill=bg)
    r0 = int(rng.integers(1, max(2, min(4, h - 8))))
    c0 = int(rng.integers(1, max(2, min(4, w - 8))))
    rh = int(rng.integers(7, min(12, h - r0) + 1))
    rw = int(rng.integers(7, min(12, w - c0) + 1))
    grid[r0:r0 + rh, c0:c0 + rw] = target
    sides = list(rng.choice(["top", "bottom", "left", "right"], size=int(rng.integers(1, 4)), replace=False))
    notch_sides = []
    for side in sides:
        if side in ("top", "bottom"):
            notch_width = int(rng.integers(2, min(5, rw - 3)))
            notch_depth = int(rng.integers(1, min(4, rh // 2)))
            nc = int(rng.integers(c0 + 2, c0 + rw - notch_width - 1))
            rows = slice(r0, r0 + notch_depth) if side == "top" else slice(r0 + rh - notch_depth, r0 + rh)
            grid[rows, nc:nc + notch_width] = bg
        else:
            notch_height = int(rng.integers(2, min(5, rh - 3)))
            notch_depth = int(rng.integers(1, min(4, rw // 2)))
            nr = int(rng.integers(r0 + 2, r0 + rh - notch_height - 1))
            cols = slice(c0, c0 + notch_depth) if side == "left" else slice(c0 + rw - notch_depth, c0 + rw)
            grid[nr:nr + notch_height, cols] = bg
        notch_sides.append(side)
    # Optional sparse background texture. The corner rule reads only the target
    # cells' edge-adjacency, so distinct-colored texture cells (recorded below)
    # never change which target cells are corners and are left unchanged.
    bg_info = add_patterned_background(
        grid, rng, bg=bg,
        exclude_colors=[int(target), int(outer), int(inner)],
        protected_boxes=[(r0, c0, r0 + rh - 1, c0 + rw - 1)],
    )
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(_mark_corners(grid, target, outer, inner)),
        "metadata": {
            "background_color": bg,
            "target_color": int(target),
            "outer_corner_color": int(outer),
            "inner_corner_color": int(inner),
            "connectivity": 4,
            "notch_sides": notch_sides,
            "background_style": bg_info["style"],
            "background_pattern_colors": bg_info["colors"],
            "corner_rule": "cell-edge 90-degree corners; inner corners take priority",
        },
    }


@register("151")
def generate_classify_inside_outside_background(rng: np.random.Generator) -> dict:
    # Independent H/W widened down to 9 (smallest that fits a 6-tall enclosing
    # frame with margin) and up to a grid->grid token-safe cap.
    # Background is left solid on purpose: the rule partitions background cells
    # by border-reachability, so any added texture would fragment the
    # background into spurious pockets and change the inside/outside split.
    h = int(rng.integers(9, 23))
    w = int(rng.integers(9, 23))
    bg = int(rng.integers(0, 10))
    wall, obj, outside, inside = unused_colors(rng, 4, bg)
    grid = make_grid(h, w, fill=bg)
    fh = int(rng.integers(6, h - 2))
    fw = int(rng.integers(6, w - 2))
    r0 = int(rng.integers(2, h - fh))
    c0 = int(rng.integers(2, w - fw))
    _draw_rect_outline(grid, r0, c0, fh, fw, wall)
    grid[1:3, 1:3] = obj
    if rng.random() < 0.5:
        grid[h - 3:h - 1, w - 3:w - 1] = obj
    barrier_colors = {int(wall), int(obj)}
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(_classify_background(grid, bg, barrier_colors, outside, inside)),
        "metadata": {
            "background_color": bg,
            "barrier_colors": [int(wall), int(obj)],
            "wall_color": int(wall),
            "object_color": int(obj),
            "outside_color": int(outside),
            "inside_color": int(inside),
            "connectivity": 4,
        },
    }


def _draw_two_color_rect_outline(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    color_a: int,
    color_b: int,
) -> None:
    """Outline whose top+left edges are color_a and bottom+right edges are color_b.

    The single enclosed hole then edge-touches both colors, so it is a
    'multiple surrounding colors' negative case.
    """
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0:c1 + 1] = color_a          # top
    grid[r0:r1 + 1, c0] = color_a          # left
    grid[r1, c0:c1 + 1] = color_b          # bottom
    grid[r0:r1 + 1, c1] = color_b          # right
    # ensure both colors actually border the interior on multiple sides
    grid[r0, c1] = color_b
    grid[r1, c0] = color_a


@register("156")
def generate_fill_holes_surrounding_color(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W widened both ways around the original mid-teens band.
        # Background is left solid on purpose: the rule fills a hole only when
        # every surrounding object cell shares one color, so any texture inside
        # a hole would add a colour and destroy the single-colour positive case.
        h = int(rng.integers(12, 23))
        w = int(rng.integers(12, 23))
        bg = int(rng.integers(0, 10))
        c1, c2, c3 = unused_colors(rng, 3, bg)
        grid = make_grid(h, w, fill=bg)

        # Top-left region frame (always a single-color positive case).
        _draw_rect_outline(grid, 2, 2, int(rng.integers(5, 8)), int(rng.integers(5, 8)), c1)

        # Bottom-right region frame: with probability 0.5 make it a two-color
        # frame whose hole touches two object colors (the discriminative
        # negative case the prompt promises but rarely exercises).
        fh2 = int(rng.integers(6, 8))
        fw2 = int(rng.integers(6, 8))
        two_color = bool(rng.random() < 0.5)
        if two_color:
            _draw_two_color_rect_outline(grid, h - fh2 - 1, w - fw2 - 1, fh2, fw2, c2, c3)
        else:
            _draw_rect_outline(grid, h - fh2 - 1, w - fw2 - 1, fh2, fw2, c2)

        # Optionally add a third small frame in the top-right region for variety.
        used_colors = {int(c1), int(c2)}
        if two_color:
            used_colors.add(int(c3))
        if rng.random() < 0.5:
            fh3 = int(rng.integers(5, 7))
            fw3 = int(rng.integers(5, 7))
            r3 = 2
            c3start = w - fw3 - 1
            # keep clear of the top-left frame
            if c3start > 2 + 7 + 1:
                _draw_rect_outline(grid, r3, c3start, fh3, fw3, c3)
                used_colors.add(int(c3))

        out = _fill_holes_with_surrounding_color(grid, bg)
        # Require at least one hole actually filled (a positive case present).
        if np.array_equal(out, grid):
            continue

        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "object_colors": sorted(used_colors),
                "connectivity": 4,
                "hole_rule": "enclosed background component touching exactly one surrounding object color",
            },
        }
    raise RuntimeError("Task 156: failed to generate valid example")


@register("167")
def generate_mark_nth_boundary_layer(rng: np.random.Generator) -> dict:
    # Independent H/W widened down to 11 (smallest that fits the thickest
    # layer_index=3 rectangle, floor 2*li+3=9, with a margin) and up to a
    # grid->grid token-safe cap.
    h = int(rng.integers(11, 23))
    w = int(rng.integers(11, 23))
    bg = int(rng.integers(0, 10))
    target, layer_color, distractor = unused_colors(rng, 3, bg)
    layer_index = int(rng.integers(1, 4))
    connectivity = 4
    grid = make_grid(h, w, fill=bg)
    rh = int(rng.integers(max(7, 2 * layer_index + 3), h - 1))
    rw = int(rng.integers(max(7, 2 * layer_index + 3), w - 1))
    r0 = int(rng.integers(1, h - rh))
    c0 = int(rng.integers(1, w - rw))
    grid[r0:r0 + rh, c0:c0 + rw] = target
    if rng.random() < 0.5 and rh > 8 and rw > 8:
        grid[r0 + 2:r0 + 4, c0 + rw - 2:c0 + rw] = bg
    rect_box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
    # Vary the object count with ignored solid distractor blobs (a separate
    # color, kept clear of the target) and optionally add sparse background
    # texture.  The boundary-layer rule reads only the target object's cells, so
    # both kinds of extra cells are non-target and are left unchanged.
    target_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == target)}
    n_distractor = _scatter_solid_blobs(
        grid, rng, int(distractor), bg, int(rng.integers(0, 3)), target_cells
    )
    bg_info = add_patterned_background(
        grid, rng, bg=bg,
        exclude_colors=[int(target), int(layer_color), int(distractor)],
        protected_boxes=[rect_box],
    )
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(_mark_boundary_layer(grid, target, layer_color, layer_index, connectivity)),
        "metadata": {
            "background_color": bg,
            "target_color": int(target),
            "layer_color": int(layer_color),
            "layer_index": layer_index,
            "connectivity": connectivity,
            "distractor_colors": [int(distractor)] if n_distractor else [],
            "background_style": bg_info["style"],
            "background_pattern_colors": bg_info["colors"],
            "layer_rule": "iteratively remove boundary cells; mark the requested removal layer",
        },
    }


@register("173")
def generate_neighbor_count_recolor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W, widened down to the structural minimum (interior
        # ring needs h,w >= 3) and up to a grid->grid token-safe cap.
        h = int(rng.integers(5, 23))
        w = int(rng.integers(5, 23))
        bg = int(rng.integers(0, 10))
        neighbor, marker = unused_colors(rng, 2, bg)
        grid = make_grid(h, w, fill=bg)
        density = float(rng.uniform(0.16, 0.30))
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if rng.random() < density:
                    grid[r, c] = neighbor
        # Optional sparse background texture. Eligible cells are background
        # only and the rule counts neighbor-color neighbors, so texture cells
        # (a distinct recorded color) are neither eligible nor counted and are
        # left unchanged by the simultaneous recolor.
        bg_info = add_patterned_background(
            grid, rng, bg=bg, exclude_colors=[int(neighbor), int(marker)],
        )
        connectivity = int(rng.choice([4, 8]))
        condition = str(rng.choice(["exactly", "at_least"]))
        count = int(rng.integers(1, 4 if connectivity == 4 else 6))
        out = _neighbor_count_transform(grid, bg, neighbor, marker, connectivity, condition, count)
        changed = int(np.sum(out == marker))
        if 1 <= changed <= (h * w) // 2:
            return {
                "input": grid_to_list(grid),
                "output": grid_to_list(out),
                "metadata": {
                    "background_color": bg,
                    "eligible_color": bg,
                    "neighbor_color": int(neighbor),
                    "marker_color": int(marker),
                    "connectivity": connectivity,
                    "condition": condition,
                    "neighbor_count": count,
                    "background_style": bg_info["style"],
                    "background_pattern_colors": bg_info["colors"],
                    "application_rule": "simultaneous",
                },
            }
    raise RuntimeError("Task 173: failed to generate valid example")
