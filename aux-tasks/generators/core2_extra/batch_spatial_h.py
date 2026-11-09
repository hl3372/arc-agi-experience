"""Spatial reasoning generators for core2 batch spatial H."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from common.objects import bounding_box, random_shape
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    boxes_overlap,
    unused_colors,
)
from registry import register


Coord = tuple[int, int]

DIRECTIONS: dict[str, Coord] = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
    "northwest": (-1, -1),
    "northeast": (-1, 1),
    "southwest": (1, -1),
    "southeast": (1, 1),
}

ORTHO_DIRS: list[Coord] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _normalize(cells: set[Coord] | frozenset[Coord]) -> frozenset[Coord]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate_cw(cells: set[Coord] | frozenset[Coord]) -> frozenset[Coord]:
    return _normalize({(c, -r) for r, c in cells})


def _transform_shape(shape: frozenset[Coord], transform: str) -> frozenset[Coord]:
    out = shape
    turns = {"rot0": 0, "rot90": 1, "rot180": 2, "rot270": 3}[transform]
    for _ in range(turns):
        out = _rotate_cw(out)
    return out


def _shape_grid(shape: frozenset[Coord], fg: int, bg: int) -> list[list[int]]:
    r0, c0, r1, c1 = bounding_box(shape)
    rows = []
    for r in range(r0, r1 + 1):
        row = []
        for c in range(c0, c1 + 1):
            row.append(int(fg if (r, c) in shape else bg))
        rows.append(row)
    return rows


def _stamp(grid: np.ndarray, top: int, left: int, shape: frozenset[Coord], color: int) -> set[Coord]:
    cells = {(top + r, left + c) for r, c in shape}
    for r, c in cells:
        grid[r, c] = color
    return cells


def _try_place_shape(
    grid: np.ndarray,
    rng: np.random.Generator,
    shape: frozenset[Coord],
    color: int,
    bg: int,
    boxes: list[tuple[int, int, int, int]],
    *,
    row_min: int = 0,
    gap: int = 1,
) -> set[Coord] | None:
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = bounding_box(shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    if row_min + sh > h or sw > w:
        return None
    for _ in range(120):
        top = int(rng.integers(row_min, h - sh + 1))
        left = int(rng.integers(0, w - sw + 1))
        box = (top, left, top + sh - 1, left + sw - 1)
        if any(boxes_overlap(box, other, gap=gap) for other in boxes):
            continue
        cells = {(top + r - sr0, left + c - sc0) for r, c in shape}
        if all(grid[r, c] == bg for r, c in cells):
            for r, c in cells:
                grid[r, c] = color
            boxes.append(box)
            return cells
    return None


def _path_from_waypoints(points: list[Coord]) -> list[Coord]:
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        dr = int(np.sign(r1 - r0))
        dc = int(np.sign(c1 - c0))
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            if (r, c) != path[-1]:
                path.append((r, c))
    if len(path) != len(set(path)):
        raise ValueError("path repeats a cell")
    return path


def _turn_cells(path: list[Coord]) -> set[Coord]:
    turns = set()
    for a, b, c in zip(path, path[1:], path[2:]):
        prev_step = (b[0] - a[0], b[1] - a[1])
        next_step = (c[0] - b[0], c[1] - b[1])
        if prev_step != next_step:
            turns.add(b)
    return turns


def _components_of_color(grid: np.ndarray, color: int) -> list[set[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    comps.sort(key=lambda cells: bounding_box(cells))
    return comps


def _distance(a: Coord, b: Coord, metric: str) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc if metric == "manhattan" else max(dr, dc)


def _line_network(grid: np.ndarray, center: Coord, arms: list[tuple[Coord, int]], color: int) -> set[Coord]:
    cells = {center}
    for (dr, dc), length in arms:
        for step in range(1, length + 1):
            cells.add((center[0] + dr * step, center[1] + dc * step))
    for r, c in cells:
        grid[r, c] = color
    return cells


def _segment_cells(a: Coord, b: Coord) -> set[Coord]:
    (r0, c0), (r1, c1) = a, b
    dr = int(np.sign(r1 - r0))
    dc = int(np.sign(c1 - c0))
    cells = {(r0, c0)}
    r, c = r0, c0
    while (r, c) != (r1, c1):
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _draw_line_segments(grid: np.ndarray, segments: list[tuple[Coord, Coord]], color: int) -> set[Coord]:
    cells: set[Coord] = set()
    for a, b in segments:
        cells |= _segment_cells(a, b)
    for r, c in cells:
        grid[r, c] = color
    return cells


def _region_components_not_wall(grid: np.ndarray, wall_color: int) -> list[set[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == wall_color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != wall_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


@register("178")
def generate_orientation_to_color_from_prototype(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    prototype_color, object_color, *orientation_colors = unused_colors(rng, 6, bg)
    prototype = frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)})
    transforms = ["rot0", "rot90", "rot180", "rot270"]
    color_map = dict(zip(transforms, orientation_colors))
    # Decouple H and W and shift the envelope down toward the ARC median; the
    # prototype/targets are 3x3 so even an 8-side leaves room for two objects.
    h = int(rng.integers(8, 18))
    w = int(rng.integers(8, 18))
    for _ in range(MAX_RETRIES):
        grid = make_grid(h, w, fill=bg)
        # Place the prototype at a random position rather than a fixed corner.
        prow = int(rng.integers(0, h - 2))
        pcol = int(rng.integers(0, w - 2))
        _stamp(grid, prow, pcol, prototype, prototype_color)
        boxes = [(prow, pcol, prow + 2, pcol + 2)]
        targets = []
        # Vary the object count, capping it to what the grid area can hold so
        # placement stays reliable as the grid shrinks.
        cap = max(2, min(6, (h * w) // 22))
        n_targets = int(rng.integers(2, cap + 1))
        chosen = list(rng.choice(transforms, size=n_targets, replace=True))
        if len(set(chosen)) < 2:
            chosen[0], chosen[1] = "rot0", "rot90"
        for transform in chosen:
            shape = _transform_shape(prototype, transform)
            cells = _try_place_shape(grid, rng, shape, object_color, bg, boxes, row_min=0, gap=1)
            if cells is None:
                break
            targets.append({"transform": transform, "bbox": list(bounding_box(cells))})
        else:
            # Optional sparse/patterned background using neither the prototype,
            # the target color, nor any orientation color, so it never forms a
            # spurious target component and never collides with a recolor color.
            bg_info = add_patterned_background(
                grid,
                rng,
                bg=bg,
                exclude_colors=[prototype_color, object_color, *orientation_colors],
                protected_boxes=boxes,
                solid_probability=0.4,
            )
            out = grid.copy()
            for target in targets:
                r0, c0, r1, c1 = target["bbox"]
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if grid[r, c] == object_color:
                            out[r, c] = color_map[target["transform"]]
            return {
                "input": grid_to_list(grid),
                "output": grid_to_list(out),
                "metadata": {
                    "background_color": bg,
                    "prototype_color": int(prototype_color),
                    "target_object_color": int(object_color),
                    "connectivity": 4,
                    "allowed_transforms": transforms,
                    "prototype_pattern": _shape_grid(prototype, prototype_color, bg),
                    "orientation_color_map": [
                        {"transform": name, "color": int(color_map[name])}
                        for name in transforms
                    ],
                    "target_rule": "each 4-connected target object is one allowed rotation of the prototype",
                    "tie_rule": "the prototype is deliberately asymmetric, so exactly one listed transform matches each target",
                    "background_texture": bg_info,
                },
            }
    raise RuntimeError("Task 178: failed to place target shapes")


@register("183")
def generate_mark_turn_cells(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 19))
    w = int(rng.integers(6, 19))
    bg = int(rng.integers(0, NUM_COLORS))
    path_color, turn_color = unused_colors(rng, 2, bg)
    rows = sorted(rng.choice(np.arange(1, h - 1), size=3, replace=False).tolist())
    cols = sorted(rng.choice(np.arange(1, w - 1), size=3, replace=False).tolist())
    variant = str(rng.choice(["zig_horizontal", "zig_vertical", "single_turn"]))
    if variant == "zig_horizontal":
        points = [(rows[0], cols[0]), (rows[0], cols[1]), (rows[2], cols[1]), (rows[2], cols[2])]
    elif variant == "zig_vertical":
        points = [(rows[0], cols[0]), (rows[1], cols[0]), (rows[1], cols[2]), (rows[2], cols[2])]
    else:
        points = [(rows[0], cols[0]), (rows[0], cols[2]), (rows[2], cols[2])]
    path = _path_from_waypoints(points)
    grid = make_grid(h, w, fill=bg)
    for r, c in path:
        grid[r, c] = path_color
    # Scatter a few other-color decoy cells (varying the object count) that are
    # kept one cell clear of the path so the path stays a clean single component
    # and the recolor rule, which only touches path-color cells, is unaffected.
    forbidden = {
        (r + dr, c + dc)
        for r, c in path
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    n_decoys = int(rng.integers(0, 4))
    decoy_colors = unused_colors(rng, max(1, n_decoys), bg, path_color, turn_color)
    placed_decoys: list[int] = []
    for color in decoy_colors[:n_decoys]:
        for _ in range(40):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if (r, c) not in forbidden and grid[r, c] == bg:
                grid[r, c] = color
                forbidden |= {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                placed_decoys.append(int(color))
                break
    out = grid.copy()
    turns = _turn_cells(path)
    for r, c in turns:
        out[r, c] = turn_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "decoy_colors": placed_decoys,
            "path_color": int(path_color),
            "turn_color": int(turn_color),
            "connectivity": 4,
            "path_width": 1,
            "path_shape": variant,
            "turn_rule": "a path cell is a turn when its two path neighbors lie in perpendicular directions",
        },
    }


@register("184")
def generate_fill_object_bbox_gaps(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 19))
    w = int(rng.integers(6, 19))
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, gap_color = unused_colors(rng, 2, bg)
    shapes = [
        frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)}),
        frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)}),
        frozenset({(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (3, 2)}),
        frozenset({(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)}),
    ]
    shape = shapes[int(rng.integers(0, len(shapes)))]
    grid = make_grid(h, w, fill=bg)
    sr0, sc0, sr1, sc1 = bounding_box(shape)
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    top = int(rng.integers(1, h - sh))
    left = int(rng.integers(1, w - sw))
    cells = _stamp(grid, top, left, shape, object_color)
    box = bounding_box(cells)
    boxes = [box]
    n_distractors = int(rng.integers(0, 4))
    distractor_colors = (
        unused_colors(rng, n_distractors, bg, object_color, gap_color) if n_distractors else []
    )
    placed_distractors: list[int] = []
    distractor_shapes = [
        frozenset({(0, 0), (0, 1), (1, 0)}),
        frozenset({(0, 0), (1, 0), (1, 1), (2, 1)}),
        frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    ]
    for color in distractor_colors:
        candidates = distractor_shapes.copy()
        rng.shuffle(candidates)
        for candidate in candidates:
            if _try_place_shape(grid, rng, candidate, int(color), bg, boxes, gap=1) is not None:
                placed_distractors.append(int(color))
                break
    out = grid.copy()
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r, c] == bg:
                out[r, c] = gap_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "target_object_color": int(object_color),
            "gap_color": int(gap_color),
            "distractor_colors": placed_distractors,
            "connectivity": 4,
            "target_bbox": list(box),
            "fill_rule": "inside the target object's bounding box only, recolor background cells that are not object cells",
        },
    }


@register("185")
def generate_nearest_marker_background_coloring(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 19))
    w = int(rng.integers(6, 19))
    bg = int(rng.integers(0, NUM_COLORS))
    # Vary the source count, capping it so the markers stay separable on small grids.
    max_sources = max(2, min(5, (h * w) // 12))
    n_sources = int(rng.integers(2, max_sources + 1))
    source_colors = unused_colors(rng, n_sources, bg)
    metric = str(rng.choice(["manhattan", "chebyshev"]))
    grid = make_grid(h, w, fill=bg)
    positions: list[Coord] = []
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    min_sep = 3
    chosen: list[Coord] = []
    for _ in range(MAX_RETRIES):
        candidate = interior[int(rng.integers(0, len(interior)))]
        if all(_distance(candidate, p, "chebyshev") >= min_sep for p in chosen):
            chosen.append(candidate)
            if len(chosen) == n_sources:
                break
    # Fall back to remaining interior cells if separation could not be met (small grids).
    if len(chosen) < n_sources:
        remaining = [p for p in interior if p not in chosen]
        rng.shuffle(remaining)
        for p in remaining:
            chosen.append(p)
            if len(chosen) == n_sources:
                break
    for color, pos in zip(source_colors, chosen):
        grid[pos] = color
        positions.append(pos)
    # A few other-color decoy objects (not sources) vary the shape/object count.
    # They are non-background, so the nearest-source rule never recolors them and
    # they never act as sources; geometric distance ignores them entirely.
    source_boxes = [(r, c, r, c) for r, c in positions]
    n_decoys = int(rng.integers(0, 3))
    decoy_colors = (
        unused_colors(rng, n_decoys, bg, *source_colors) if n_decoys else []
    )
    placed_decoys: list[int] = []
    for color in decoy_colors:
        shape = random_shape(rng, (2, 4), connectivity=4)
        if _try_place_shape(grid, rng, shape, int(color), bg, source_boxes, gap=1) is not None:
            placed_decoys.append(int(color))
    out = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg:
                continue
            distances = [_distance((r, c), src, metric) for src in positions]
            best = min(distances)
            if distances.count(best) == 1:
                out[r, c] = source_colors[distances.index(best)]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "source_colors": [int(c) for c in source_colors],
            "decoy_colors": placed_decoys,
            "metric": metric,
            "eligible_color": bg,
            "tie_rule": "leave tied background cells unchanged",
            "source_policy": "source cells keep their original colors; other-color cells are left unchanged",
            "sources": [
                {"row": int(pos[0]), "col": int(pos[1]), "color": int(color)}
                for pos, color in zip(positions, source_colors)
            ],
        },
    }


@register("186")
def generate_mark_first_ray_hits(rng: np.random.Generator) -> dict:
    h = int(rng.integers(8, 20))
    w = int(rng.integers(8, 20))
    bg = int(rng.integers(0, NUM_COLORS))
    source_color, hit_color, mark_color = unused_colors(rng, 3, bg)
    grid = make_grid(h, w, fill=bg)
    source = (int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3)))
    grid[source] = source_color
    direction_sets = [
        ["north", "south", "east", "west"],
        ["northwest", "northeast", "southwest", "southeast"],
        ["north", "east", "southwest", "southeast"],
        ["south", "west", "northeast", "northwest"],
    ]
    directions = list(direction_sets[int(rng.integers(0, len(direction_sets)))])
    guaranteed = False
    for name in directions:
        dr, dc = DIRECTIONS[name]
        ray = []
        r, c = source[0] + dr, source[1] + dc
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r += dr
            c += dc
        if len(ray) < 2:
            continue
        if rng.random() < 0.75 or not guaranteed:
            idx = int(rng.integers(1, len(ray)))
            grid[ray[idx]] = hit_color
            guaranteed = True
            if idx + 2 < len(ray) and rng.random() < 0.5:
                grid[ray[int(rng.integers(idx + 1, len(ray)))]] = hit_color
    for _ in range(3):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if grid[r, c] == bg:
            grid[r, c] = hit_color
    out = grid.copy()
    for name in directions:
        dr, dc = DIRECTIONS[name]
        r, c = source[0] + dr, source[1] + dc
        while 0 <= r < h and 0 <= c < w:
            if grid[r, c] == hit_color:
                out[r, c] = mark_color
                break
            r += dr
            c += dc
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "source_color": int(source_color),
            "hit_color": int(hit_color),
            "mark_color": int(mark_color),
            "directions": directions,
            "hit_condition": "first cell of hit_color along the ray",
            "ray_rule": "only the first hit cell in each direction is recolored; empty rays do nothing",
        },
    }


@register("187")
def generate_mark_line_network_degrees(rng: np.random.Generator) -> dict:
    # Decouple H and W; the network has a bounded extent (<=4 cells from the
    # centre on any side), so any side >= 9 fits a centred network. The upper
    # bound is held at 18 so a single example stays inside the bundling budget.
    h = int(rng.integers(9, 19))
    w = int(rng.integers(9, 19))
    bg = int(rng.integers(0, NUM_COLORS))
    line_color, endpoint_color, branch_color = unused_colors(rng, 3, bg)
    grid = make_grid(h, w, fill=bg)
    center = (h // 2, w // 2)
    variant = str(
        rng.choice(
            [
                "tee",
                "plus",
                "three_way_bent",
                "four_way_uneven",
                "h_shape",
                "double_tee",
                "ladder",
            ]
        )
    )
    single_star_defs = {
        "tee": [((0, -1), 3), ((0, 1), 3), ((-1, 0), 3)],
        "plus": [((-1, 0), 3), ((1, 0), 3), ((0, -1), 3), ((0, 1), 3)],
        "three_way_bent": [((-1, 0), 3), ((1, 0), 2), ((0, 1), 4)],
        "four_way_uneven": [((-1, 0), 2), ((1, 0), 4), ((0, -1), 2), ((0, 1), 3)],
    }
    if variant in single_star_defs:
        cells = _line_network(grid, center, single_star_defs[variant], line_color)
    else:
        cr, cc = center
        if variant == "h_shape":
            # Two vertical bars joined by a horizontal rung -> 2 branchpoints, 4 endpoints.
            left_c, right_c = cc - 2, cc + 2
            segments = [
                ((cr - 3, left_c), (cr + 3, left_c)),
                ((cr - 3, right_c), (cr + 3, right_c)),
                ((cr, left_c), (cr, right_c)),
            ]
        elif variant == "double_tee":
            # Horizontal spine with two perpendicular arms -> 2 branchpoints.
            arm_a, arm_b = cc - 2, cc + 2
            segments = [
                ((cr, cc - 3), (cr, cc + 3)),
                ((cr, arm_a), (cr + 3, arm_a)),
                ((cr, arm_b), (cr + 3, arm_b)),
            ]
        else:  # ladder: two parallel rails joined by two interior rungs -> 4 branchpoints.
            left_c, right_c = cc - 3, cc + 3
            rung_a, rung_b = cr - 1, cr + 1
            segments = [
                ((cr - 3, left_c), (cr + 3, left_c)),
                ((cr - 3, right_c), (cr + 3, right_c)),
                ((rung_a, left_c), (rung_a, right_c)),
                ((rung_b, left_c), (rung_b, right_c)),
            ]
        cells = _draw_line_segments(grid, segments, line_color)
    # Scatter a few other-color decoy objects, kept one cell clear of the network,
    # to vary the object count. Degree is computed only over line-color cells, so
    # decoys never change any endpoint/branchpoint and are left unchanged.
    network_box = bounding_box(cells)
    n_decoys = int(rng.integers(0, 4))
    decoy_colors = (
        unused_colors(rng, n_decoys, bg, line_color, endpoint_color, branch_color)
        if n_decoys
        else []
    )
    placed_decoys: list[int] = []
    for color in decoy_colors:
        shape = random_shape(rng, (2, 4), connectivity=4)
        if _try_place_shape(grid, rng, shape, int(color), bg, [network_box], gap=1) is not None:
            placed_decoys.append(int(color))
    out = grid.copy()
    for r, c in cells:
        degree = sum((r + dr, c + dc) in cells for dr, dc in ORTHO_DIRS)
        if degree == 1:
            out[r, c] = endpoint_color
        elif degree >= 3:
            out[r, c] = branch_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "line_color": int(line_color),
            "endpoint_marker_color": int(endpoint_color),
            "branchpoint_marker_color": int(branch_color),
            "decoy_colors": placed_decoys,
            "connectivity": 4,
            "path_width": 1,
            "network_shape": variant,
            "degree_rule": "endpoint degree is exactly 1; branchpoint degree is 3 or more",
        },
    }


@register("190")
def generate_motif_blocks_to_legend_colors(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    motif_color, *legend_colors = unused_colors(rng, 5, bg)
    size = int(rng.choice([2, 3]))
    patterns_by_size = {
        2: [
            ((1, 0), (0, 1)),
            ((1, 1), (0, 0)),
            ((1, 0), (1, 0)),
            ((0, 1), (0, 1)),
        ],
        3: [
            ((1, 0, 0), (1, 1, 0), (0, 1, 1)),
            ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
            ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
            ((1, 0, 1), (0, 1, 0), (1, 0, 1)),
        ],
    }
    n_motifs = int(rng.integers(3, 5))
    pattern_bits = patterns_by_size[size][:n_motifs]
    motif_grids = [
        [[int(motif_color if bit else bg) for bit in row] for row in pattern]
        for pattern in pattern_bits
    ]
    # Vary the block-grid dimensions independently; with size in {2,3} this
    # spans roughly 4x4 up to 18x18, decoupling H from W.
    block_rows = int(rng.integers(2, 7))
    block_cols = int(rng.integers(2, 7))
    grid = make_grid(block_rows * size, block_cols * size, fill=bg)
    out = make_grid(block_rows * size, block_cols * size, fill=bg)
    for br in range(block_rows):
        for bc in range(block_cols):
            idx = int(rng.integers(0, n_motifs))
            motif = motif_grids[idx]
            color = legend_colors[idx]
            for rr in range(size):
                for cc in range(size):
                    grid[br * size + rr, bc * size + cc] = motif[rr][cc]
                    out[br * size + rr, bc * size + cc] = color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "motif_color": int(motif_color),
            "block_height": size,
            "block_width": size,
            "partition_rule": "split the whole input into non-overlapping equal-size blocks from the top-left",
            "motif_color_map": [
                {"motif": motif_grids[i], "color": int(legend_colors[i])}
                for i in range(n_motifs)
            ],
        },
    }


@register("191")
def generate_recolor_by_containment_depth(rng: np.random.Generator) -> dict:
    # Decouple H and W: the nested frames become concentric rectangles (not just
    # squares), which keeps the bounding-box containment-depth rule intact.
    # Both sides need >= 8 to nest two frames; cap the area so a single example
    # stays within the bundling token budget (grid->grid caps lower).
    area_cap = 270
    h = int(rng.integers(8, 22))
    w = int(rng.integers(8, min(21, area_cap // h) + 1))
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, *depth_colors = unused_colors(rng, 5, bg)
    # Each frame is inset by 2 on every side; an innermost frame of depth d needs
    # both sides >= 5 + 4*(d-1) to keep a one-cell-thick rectangle with interior.
    max_depth_count = min(4, ((min(h, w) - 4) // 4) + 1)
    depth_count = int(rng.integers(2, max_depth_count + 1))
    grid = make_grid(h, w, fill=bg)
    boxes = []
    margin = 1
    for depth in range(depth_count):
        top = margin + depth * 2
        left = margin + depth * 2
        bottom = h - margin - depth * 2 - 1
        right = w - margin - depth * 2 - 1
        boxes.append((top, left, bottom, right))
        grid[top, left : right + 1] = object_color
        grid[bottom, left : right + 1] = object_color
        grid[top : bottom + 1, left] = object_color
        grid[top : bottom + 1, right] = object_color
    out = grid.copy()
    for depth, (top, left, bottom, right) in enumerate(boxes):
        color = depth_colors[depth]
        out[top, left : right + 1] = color
        out[bottom, left : right + 1] = color
        out[top : bottom + 1, left] = color
        out[top : bottom + 1, right] = color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "object_color": int(object_color),
            "connectivity": 4,
            "object_kind": "nested one-cell-thick rectangular frames",
            "depth_rule": "depth is the number of other object bounding boxes that strictly contain this object's bounding box",
            "depth_color_map": [
                {"depth": int(depth), "color": int(depth_colors[depth])}
                for depth in range(depth_count)
            ],
        },
    }


@register("192")
def generate_fill_regions_with_single_marker(rng: np.random.Generator) -> dict:
    # Vary the compartment grid and the cell size independently so the overall
    # grid spans a wider, decoupled H/W envelope. Cap each side near 17 so a
    # single example stays within the bundling token budget (grid->grid).
    max_side = 17

    def _pick_cells(n_comp: int) -> int:
        # Largest cell size keeping n_comp*cell + n_comp + 1 <= max_side.
        hi = max(2, (max_side - n_comp - 1) // n_comp)
        return int(rng.integers(2, hi + 1))

    rows = int(rng.integers(2, 5))
    cols = int(rng.integers(2, 5))
    cell_h = _pick_cells(rows)
    cell_w = _pick_cells(cols)
    h = rows * cell_h + rows + 1
    w = cols * cell_w + cols + 1
    bg = int(rng.integers(0, NUM_COLORS))
    wall_color, *marker_colors = unused_colors(rng, 5, bg)
    grid = make_grid(h, w, fill=bg)
    grid[0, :] = wall_color
    grid[-1, :] = wall_color
    grid[:, 0] = wall_color
    grid[:, -1] = wall_color
    for rr in range(1, rows):
        grid[rr * (cell_h + 1), :] = wall_color
    for cc in range(1, cols):
        grid[:, cc * (cell_w + 1)] = wall_color
    region_specs = []
    counts = [0, 1, 2]
    while len(counts) < rows * cols:
        counts.append(int(rng.choice([0, 1, 1, 2])))
    rng.shuffle(counts)
    marker_idx = 0
    for idx, count in enumerate(counts[: rows * cols]):
        rr, cc = divmod(idx, cols)
        r0 = rr * (cell_h + 1) + 1
        c0 = cc * (cell_w + 1) + 1
        positions = [(r0, c0), (r0 + cell_h - 1, c0 + cell_w - 1)]
        colors = []
        for i in range(count):
            color = marker_colors[marker_idx % len(marker_colors)]
            marker_idx += 1
            # Guarantee the two markers in a 2-marker region never share a color,
            # so the count cannot be misread as a single marker color.
            if int(color) in colors:
                color = marker_colors[marker_idx % len(marker_colors)]
                marker_idx += 1
            pos = positions[i]
            grid[pos] = color
            colors.append(int(color))
        region_specs.append({"region_row": rr, "region_col": cc, "marker_count": count, "marker_colors": colors})
    marker_set = set(marker_colors)
    out = grid.copy()
    for comp in _region_components_not_wall(grid, wall_color):
        markers = [(r, c) for r, c in comp if grid[r, c] in marker_set]
        if len(markers) == 1:
            fill_color = int(grid[markers[0]])
            for r, c in comp:
                if grid[r, c] == bg:
                    out[r, c] = fill_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "wall_color": int(wall_color),
            "marker_colors": [int(c) for c in marker_colors],
            "connectivity": 4,
            "region_rule": "regions are 4-connected non-wall compartments; markers belong to their containing compartment",
            "fill_rule": "fill only background cells in regions with exactly one marker, using that marker's color",
            "regions": region_specs,
        },
    }


@register("196")
def generate_recolor_one_side_of_divider(rng: np.random.Generator) -> dict:
    h = int(rng.integers(7, 20))
    w = int(rng.integers(7, 20))
    bg = int(rng.integers(0, NUM_COLORS))
    divider_color, side_color, obstacle_color = unused_colors(rng, 3, bg)
    grid = make_grid(h, w, fill=bg)
    orientation = str(rng.choice(["horizontal", "vertical", "main_diagonal", "anti_diagonal"]))
    if orientation == "horizontal":
        index = int(rng.integers(2, h - 2))
        grid[index, :] = divider_color
        side = str(rng.choice(["above", "below"]))
        line_parameter = index
    elif orientation == "vertical":
        index = int(rng.integers(2, w - 2))
        grid[:, index] = divider_color
        side = str(rng.choice(["left", "right"]))
        line_parameter = index
    elif orientation == "main_diagonal":
        max_offset = min(2, h - 3, w - 3)
        # Retry the offset/side draw until the named side covers enough cells so
        # the divider does not merely clip a corner (a degenerate, near-trivial instance).
        for _ in range(MAX_RETRIES):
            offset = int(rng.integers(-max_offset, max_offset + 1))
            side = str(rng.choice(["below_left", "above_right"]))
            if side == "below_left":
                side_cells = sum(1 for r in range(h) for c in range(w) if c - r < offset)
            else:
                side_cells = sum(1 for r in range(h) for c in range(w) if c - r > offset)
            # 6 minimum recolored cells plus headroom for up to 4 obstacles landing on the side
            if side_cells >= 10:
                break
        for r in range(h):
            c = r + offset
            if 0 <= c < w:
                grid[r, c] = divider_color
        line_parameter = offset
    else:
        offsets = [s for s in range(2, h + w - 3) if 0 < s < h + w - 2]
        for _ in range(MAX_RETRIES):
            offset = int(offsets[int(rng.integers(0, len(offsets)))])
            side = str(rng.choice(["above_left", "below_right"]))
            if side == "above_left":
                side_cells = sum(1 for r in range(h) for c in range(w) if r + c < offset)
            else:
                side_cells = sum(1 for r in range(h) for c in range(w) if r + c > offset)
            # 6 minimum recolored cells plus headroom for up to 4 obstacles landing on the side
            if side_cells >= 10:
                break
        for r in range(h):
            c = offset - r
            if 0 <= c < w:
                grid[r, c] = divider_color
        line_parameter = offset
    for _ in range(int(rng.integers(2, 5))):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if grid[r, c] == bg:
            grid[r, c] = obstacle_color
    out = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg:
                continue
            on_side = (
                (orientation == "horizontal" and side == "above" and r < line_parameter)
                or (orientation == "horizontal" and side == "below" and r > line_parameter)
                or (orientation == "vertical" and side == "left" and c < line_parameter)
                or (orientation == "vertical" and side == "right" and c > line_parameter)
                or (orientation == "main_diagonal" and side == "below_left" and c - r < line_parameter)
                or (orientation == "main_diagonal" and side == "above_right" and c - r > line_parameter)
                or (orientation == "anti_diagonal" and side == "above_left" and r + c < line_parameter)
                or (orientation == "anti_diagonal" and side == "below_right" and r + c > line_parameter)
            )
            if on_side:
                out[r, c] = side_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "divider_color": int(divider_color),
            "side_color": int(side_color),
            "obstacle_color": int(obstacle_color),
            "divider_orientation": orientation,
            "divider_index": int(line_parameter),
            "side": side,
            "side_rule": "only background cells strictly on the named side of the divider are recolored",
        },
    }
