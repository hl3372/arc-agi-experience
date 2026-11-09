"""Graph, object-relation, and count-expression task generators."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import grid_to_list, make_grid
from common.objects import bounding_box, connected_components, normalize_shape, random_shape
from generators.core2_extra.utils import (
    MAX_RETRIES,
    available_colors,
    boxes_overlap,
    place_shape,
    unused_colors,
)
from registry import register


DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _add_background_texture(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    exclude_colors: tuple[int, ...] | list[int],
    protected_cells: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    present_probability: float = 0.6,
) -> dict:
    """Optionally scatter sparse, isolated single-cell background dots.

    The dots use a single colour disjoint from every semantically meaningful
    colour, sit only on free background cells (never on a protected/object cell),
    and are laid on a 2-spaced lattice so no two dots are 4-adjacent — they can
    never coalesce into something an object detector would treat as an object.
    Background stays the dominant colour. Returns a record describing the
    texture so the prompt can tell the solver to ignore it; ``{"present": False}``
    means the example was left solid (a genuine solid+textured mix over seeds).
    """
    if rng.random() >= present_probability:
        return {"present": False, "color": None}
    palette = available_colors(bg, *exclude_colors)
    if not palette:
        return {"present": False, "color": None}
    color = int(palette[int(rng.integers(0, len(palette)))])
    protected = set(protected_cells or set())
    h, w = grid.shape
    # 2-spaced lattice with a random phase keeps every dot isolated.
    pr = int(rng.integers(0, 2))
    pc = int(rng.integers(0, 2))
    density = float(rng.uniform(0.25, 0.7))  # fraction of lattice slots used
    painted = 0
    for r in range(pr, h, 2):
        for c in range(pc, w, 2):
            if grid[r, c] != bg or (r, c) in protected:
                continue
            if rng.random() < density:
                grid[r, c] = color
                painted += 1
    if painted == 0:
        return {"present": False, "color": None}
    return {"present": True, "color": color}


def _scatter_decoration_shapes(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    exclude_colors: tuple[int, ...] | list[int],
    protected_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
    present_probability: float = 0.6,
    max_shapes: int = 4,
) -> dict:
    """Optionally scatter a few small, mutually isolated decoration *shapes*.

    Used by tasks whose rule is keyed strictly by colour (so a decoration in a
    disjoint colour can never be confused with a semantic object): the shapes
    use one colour disjoint from every meaningful colour, each carries a 1-cell
    halo (so no two decorations touch and none touches a protected/semantic
    cell), and they sit only on free background cells outside ``protected_cells``.
    Returns ``{"present": True, "color": k}`` when shapes were laid, else
    ``{"present": False, "color": None}`` — yielding a genuine plain+decorated
    mix over seeds. Unlike :func:`_add_background_texture` (single dots), this
    widens the shape vocabulary, so it must only be used where the task records
    and ignores the decoration colour AND finds objects by an exact colour match.
    """
    if rng.random() >= present_probability:
        return {"present": False, "color": None}
    palette = available_colors(bg, *exclude_colors)
    if not palette:
        return {"present": False, "color": None}
    color = int(palette[int(rng.integers(0, len(palette)))])
    protected = set(protected_cells)
    h, w = grid.shape
    placed = 0
    n_target = int(rng.integers(1, max_shapes + 1))
    for _ in range(40):
        if placed >= n_target:
            break
        # A small varied polyomino (1-4 cells): single cells, dominoes, and
        # little L/T/square shapes all appear, broadening the shape vocabulary.
        size = int(rng.integers(1, 5))
        shape = random_shape(rng, (size, size), connectivity=4)
        sh, sw = _shape_size(shape)
        if sh > h or sw > w:
            continue
        top = int(rng.integers(0, h - sh + 1))
        left = int(rng.integers(0, w - sw + 1))
        cells = _translated_shape(shape, top, left)
        # Build a 1-cell halo around the shape; reject if the halo hits any
        # protected cell, a non-bg cell, or the grid edge wrap (out of bounds is
        # simply skipped — interior halo cells alone must all be free).
        halo = set()
        for cr, cc in cells:
            for hr in range(cr - 1, cr + 2):
                for hc in range(cc - 1, cc + 2):
                    halo.add((hr, hc))
        bad = False
        for hr, hc in halo:
            if not (0 <= hr < h and 0 <= hc < w):
                continue
            if (hr, hc) in protected or grid[hr, hc] != bg:
                bad = True
                break
        if bad:
            continue
        for cell in cells:
            grid[cell] = color
        protected |= halo
        placed += 1
    if placed == 0:
        return {"present": False, "color": None}
    return {"present": True, "color": color}


def _shape_size(shape: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = bounding_box(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _translated_shape(shape: set[tuple[int, int]] | frozenset[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    sr0, sc0, _, _ = bounding_box(shape)
    return {(r0 + r - sr0, c0 + c - sc0) for r, c in shape}


def _stamp(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _components_of(grid: np.ndarray, bg: int, color: int | None = None) -> list[dict]:
    comps = connected_components(grid, connectivity=4, background=bg)
    if color is not None:
        comps = [comp for comp in comps if comp["color"] == color]
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))


def _place_shape(
    grid: np.ndarray,
    shape: set[tuple[int, int]] | frozenset[tuple[int, int]],
    top: int,
    left: int,
    color: int,
    boxes: list[tuple[int, int, int, int]],
    gap: int = 1,
) -> dict:
    cells = _translated_shape(shape, top, left)
    box = bounding_box(cells)
    if any(boxes_overlap(box, other, gap=gap) for other in boxes):
        raise ValueError("shape placement overlaps another object")
    _stamp(grid, cells, color)
    boxes.append(box)
    return {"color": int(color), "cells": cells, "bbox": box}


def _mapping_list(values: list[int], colors: list[int], value_key: str) -> list[dict]:
    return [{value_key: int(value), "color": int(color)} for value, color in zip(values, colors)]


def _mapping_lookup(mapping: list[dict], value_key: str) -> dict[int, int]:
    return {int(entry[value_key]): int(entry["color"]) for entry in mapping}


@register("197")
def generate_panel_grid_intersection_crop(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, 10))
    divider, row_marker, col_marker = unused_colors(rng, 3, bg)
    block_h = int(rng.integers(2, 5))
    block_w = int(rng.integers(2, 5))
    n_rows = int(rng.integers(2, 5))
    n_cols = int(rng.integers(2, 5))
    panel_h = n_rows * block_h + (n_rows - 1)
    panel_w = n_cols * block_w + (n_cols - 1)
    grid = make_grid(panel_h + 1, panel_w + 1, bg)
    grid[1:, 1:] = bg
    for k in range(1, n_rows):
        grid[1 + k * block_h + (k - 1), 1:] = divider
    for k in range(1, n_cols):
        grid[1:, 1 + k * block_w + (k - 1)] = divider

    fill_colors = unused_colors(rng, min(6, 10), bg, divider, row_marker, col_marker)
    for br in range(n_rows):
        r0 = 1 + br * (block_h + 1)
        for bc in range(n_cols):
            c0 = 1 + bc * (block_w + 1)
            block = np.array(rng.choice(fill_colors, size=(block_h, block_w), replace=True), dtype=np.uint8)
            grid[r0:r0 + block_h, c0:c0 + block_w] = block

    selected_row = int(rng.integers(0, n_rows))
    selected_col = int(rng.integers(0, n_cols))
    marker_r = 1 + selected_row * (block_h + 1) + block_h // 2
    marker_c = 1 + selected_col * (block_w + 1) + block_w // 2
    grid[marker_r, 0] = row_marker
    grid[0, marker_c] = col_marker
    r0 = 1 + selected_row * (block_h + 1)
    c0 = 1 + selected_col * (block_w + 1)
    output = grid[r0:r0 + block_h, c0:c0 + block_w]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(output),
        "metadata": {
            "background_color": bg,
            "divider_color": divider,
            "row_marker_color": row_marker,
            "column_marker_color": col_marker,
            "block_height": block_h,
            "block_width": block_w,
            "selection_rule": "intersection_of_unique_row_and_column_markers",
        },
    }


@register("198")
def generate_reorder_marked_axes(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, 10))
    axis = str(rng.choice(["rows", "columns"]))
    # Markers occupy the header cell of each marked line; the perpendicular data
    # extent is sampled independently across a wide span so the grid is not stuck
    # in the narrow few-marker square. Palette: markers + data + bg must fit 10.
    marker_count = int(rng.integers(3, 7))
    n_data = int(rng.integers(2, min(5, 10 - marker_count)))
    marker_colors = unused_colors(rng, marker_count, bg)
    data_colors = unused_colors(rng, n_data, bg, *marker_colors)
    order = list(marker_colors)
    rng.shuffle(order)
    original = list(order)
    while original == order:
        rng.shuffle(original)

    data_extent = int(rng.integers(5, 21))
    if axis == "rows":
        width = data_extent
        grid = make_grid(marker_count, width, bg)
        for r, color in enumerate(original):
            grid[r, 0] = color
            grid[r, 1:] = rng.choice(data_colors, size=width - 1, replace=True)
        index_by_marker = {int(grid[r, 0]): r for r in range(marker_count)}
        output = grid[[index_by_marker[color] for color in order], :]
    else:
        height = data_extent
        grid = make_grid(height, marker_count, bg)
        for c, color in enumerate(original):
            grid[0, c] = color
            grid[1:, c] = rng.choice(data_colors, size=height - 1, replace=True)
        index_by_marker = {int(grid[0, c]): c for c in range(marker_count)}
        output = grid[:, [index_by_marker[color] for color in order]]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(output),
        "metadata": {
            "background_color": bg,
            "axis": axis,
            "marker_order": [int(c) for c in order],
            "marker_position": "left_column" if axis == "rows" else "top_row",
            "movement_rule": "move_full_marked_rows_or_columns",
        },
    }


def _alignment_target_start(box: tuple[int, int, int, int], axis: str, landmark: str, target: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if axis == "row":
        if landmark == "top_edge":
            return target, c0
        if landmark == "bottom_edge":
            return target - (r1 - r0), c0
        return target - (r1 - r0) // 2, c0
    if landmark == "left_edge":
        return r0, target
    if landmark == "right_edge":
        return r0, target - (c1 - c0)
    return r0, target - (c1 - c0) // 2


def _aligned_fits(box: tuple[int, int, int, int], axis: str, landmark: str, target: int, limit: int) -> bool:
    r0, c0, r1, c1 = box
    new_top, new_left = _alignment_target_start(box, axis, landmark, target)
    if axis == "row":
        return 0 <= new_top and new_top + (r1 - r0) < limit
    return 0 <= new_left and new_left + (c1 - c0) < limit


@register("199")
def generate_translate_objects_to_alignment_line(rng: np.random.Generator) -> dict:
    axis = str(rng.choice(["row", "column"]))
    landmark = str(rng.choice(["top_edge", "bottom_edge", "center_row"] if axis == "row" else ["left_edge", "right_edge", "center_column"]))
    center = landmark in ("center_row", "center_column")
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        object_color, line_color = unused_colors(rng, 2, bg)
        # H and W sampled independently across the ARC-typical span.
        grid_h = int(rng.integers(10, 21))
        grid_w = int(rng.integers(10, 21))
        grid = make_grid(grid_h, grid_w, bg)
        n_objects = int(rng.integers(2, 5))
        # Each object lives in its own band on the perpendicular axis (a disjoint
        # column block for a row-alignment, a disjoint row block otherwise), so
        # translation along the alignment axis can never make two objects collide
        # and the per-object perpendicular position is preserved (tie policy).
        if axis == "row":
            along, perp = grid_h, grid_w
        else:
            along, perp = grid_w, grid_h
        # Reserve a target line and place objects so each fits before AND after
        # alignment. Object extent along the alignment axis is bounded so the
        # aligned object always fits inside `along`.
        max_ext = min(4, along - 2)
        if max_ext < 1:
            continue
        target = int(rng.integers(0, along))
        # Carve `n_objects` disjoint perpendicular bands of width >= shape width+gap.
        band_w = max(2, perp // (n_objects + 1))
        starts = list(range(0, perp - band_w + 1, band_w + 1))
        rng.shuffle(starts)
        starts = sorted(int(s) for s in starts[:n_objects])
        if len(starts) < n_objects:
            continue
        objects = []
        boxes: list[tuple[int, int, int, int]] = []
        placed = True
        for band_start in starts:
            ok_obj = False
            for _attempt in range(60):
                # Extent along alignment axis (odd if center landmark, for an exact center).
                ext = int(rng.integers(1, max_ext + 1))
                if center and ext % 2 == 0:
                    ext = ext - 1 if ext > 1 else 1
                wid = int(rng.integers(1, max(2, band_w)))
                if axis == "row":
                    shape = {(rr, cc) for rr in range(ext) for cc in range(wid)}
                    sh, sw = ext, wid
                    top = int(rng.integers(0, along - sh + 1))
                    left = band_start
                    box = (top, left, top + sh - 1, left + sw - 1)
                else:
                    shape = {(rr, cc) for rr in range(wid) for cc in range(ext)}
                    sh, sw = wid, ext
                    left = int(rng.integers(0, along - sw + 1))
                    top = band_start
                    box = (top, left, top + sh - 1, left + sw - 1)
                # Input must avoid the target line and fit after alignment.
                line_cells = set(range(box[0], box[2] + 1)) if axis == "row" else set(range(box[1], box[3] + 1))
                if target in line_cells:
                    continue
                if not _aligned_fits(box, axis, landmark, target, along):
                    continue
                cells = _translated_shape(shape, box[0], box[1])
                if any(boxes_overlap(box, other, gap=1) for other in boxes):
                    continue
                _stamp(grid, cells, object_color)
                boxes.append(box)
                objects.append({"color": int(object_color), "cells": cells, "bbox": box, "shape": shape})
                ok_obj = True
                break
            if not ok_obj:
                placed = False
                break
        if not placed:
            continue
        # Paint the target line over background only (objects keep their cells).
        if axis == "row":
            grid[target, :] = np.where(grid[target, :] == bg, line_color, grid[target, :])
        else:
            grid[:, target] = np.where(grid[:, target] == bg, line_color, grid[:, target])
        # Require the objects to actually move (not already on the line) for at
        # least one object, so the transform is non-trivial.
        starts_aligned = [_alignment_target_start(o["bbox"], axis, landmark, target) for o in objects]
        if all(
            (o["bbox"][0], o["bbox"][1]) == s for o, s in zip(objects, starts_aligned)
        ):
            continue
        # Texture is added after the line + objects are stamped (so the
        # background guard keeps it off both); it uses a colour disjoint from the
        # object and line colours. Object detection filters by object colour, so
        # texture never participates in the alignment.
        texture = _add_background_texture(
            grid, rng, bg=bg, exclude_colors=(object_color, line_color),
        )
        output = grid.copy()
        for obj in objects:
            for r, c in obj["cells"]:
                output[r, c] = bg
        for obj in objects:
            new_top, new_left = _alignment_target_start(obj["bbox"], axis, landmark, target)
            _stamp(output, _translated_shape(obj["shape"], new_top, new_left), object_color)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(output),
            "metadata": {
                "background_texture_color": texture["color"],
                "background_color": bg,
                "object_color": object_color,
                "target_line_color": line_color,
                "axis": axis,
                "landmark": landmark,
                "target_index": target,
                "tie_policy": "objects_keep_their_perpendicular_position",
            },
        }
    raise RuntimeError("Task 199: failed to generate valid alignment example")


def _hole_count_for_component(grid: np.ndarray, comp: dict, bg: int) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or sub[r, c] != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                touches_edge |= cr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and sub[nr, nc] == bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            holes += int(not touches_edge)
    return holes


def _holey_shape(rng: np.random.Generator, hole_count: int) -> set[tuple[int, int]]:
    """Build a solid rectangle (random size) with exactly `hole_count` enclosed
    1x1 background holes, each fully surrounded by object cells and mutually
    separated by at least one object cell."""
    if hole_count == 0:
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        return {(r, c) for r in range(h) for c in range(w)}
    # Interior hole slots live on a sub-lattice so that surrounding object
    # cells always separate them (no two holes are 4-adjacent and none touches
    # the bounding-box edge). Each hole occupies an interior cell at rows/cols
    # of the form 1 + 2*k, guaranteeing a border ring and inter-hole walls.
    n_slots = hole_count
    # Number of interior slots per axis; choose a layout that fits hole_count.
    cols_of_slots = int(rng.integers(1, n_slots + 1))
    rows_of_slots = -(-n_slots // cols_of_slots)  # ceil
    # Inner region needs rows_of_slots/cols_of_slots interior cells with gaps.
    inner_h = rows_of_slots * 2 - 1
    inner_w = cols_of_slots * 2 - 1
    # Pad with extra object thickness so border is >= 1 on every side.
    h = inner_h + 2
    w = inner_w + 2
    cells = {(r, c) for r in range(h) for c in range(w)}
    hole_positions = [
        (1 + 2 * sr, 1 + 2 * sc)
        for sr in range(rows_of_slots)
        for sc in range(cols_of_slots)
    ]
    for pos in hole_positions[:hole_count]:
        cells.discard(pos)
    return cells


@register("200")
def generate_recolor_by_hole_count(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        object_color = unused_colors(rng, 1, bg)[0]
        # H and W sampled independently across a wide span (holey rectangles reach
        # a 7-cell bbox side, so the lower bound leaves room for several + gaps).
        grid_h = int(rng.integers(10, 23))
        grid_w = int(rng.integers(10, 23))
        grid = make_grid(grid_h, grid_w, bg)
        n_objects = int(rng.integers(3, 7))
        # Sample a hole count per object from {0,1,2,3}; require >= 2 distinct
        # values so the task cannot be solved by silhouette memorization.
        hole_counts = [int(rng.integers(0, 4)) for _ in range(n_objects)]
        if len(set(hole_counts)) < 2:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        placed = True
        for hc in hole_counts:
            shape = _holey_shape(rng, hc)
            sh, sw = _shape_size(shape)
            for _attempt in range(120):
                top = int(rng.integers(0, grid_h - sh + 1))
                left = int(rng.integers(0, grid_w - sw + 1))
                try:
                    _place_shape(grid, shape, top, left, object_color, boxes, gap=1)
                    break
                except ValueError:
                    continue
            else:
                placed = False
                break
        if not placed:
            continue
        comps = _components_of(grid, bg, object_color)
        if len(comps) != n_objects:
            continue
        present = sorted({_hole_count_for_component(grid, comp, bg) for comp in comps})
        if len(present) < 2:
            continue
        map_colors = unused_colors(rng, len(present), bg, object_color)
        mapping = _mapping_list(present, map_colors, "hole_count")
        lookup = _mapping_lookup(mapping, "hole_count")
        # Protect every cell inside each object's bounding box so texture stays
        # strictly outside the objects and cannot alter any enclosed-hole count.
        protected = {
            (r, c)
            for (r0, c0, r1, c1) in boxes
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
        }
        texture = _add_background_texture(
            grid, rng, bg=bg,
            exclude_colors=(object_color, *map_colors),
            protected_cells=protected,
        )
        output = grid.copy()
        for comp in comps:
            color = lookup[_hole_count_for_component(grid, comp, bg)]
            for r, c in comp["cells"]:
                output[r, c] = color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(output),
            "metadata": {
                "background_texture_color": texture["color"],
                "background_color": bg,
                "object_color": object_color,
                "hole_connectivity": 4,
                "hole_count_to_color": mapping,
            },
        }
    raise RuntimeError("Task 200: failed to generate valid hole-count example")


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _min_manhattan_gap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _graph_edges(objects: list[dict], relation: str, threshold: int) -> set[tuple[int, int]]:
    edges = set()
    for i, a in enumerate(objects):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for j, b in enumerate(objects[i + 1:], start=i + 1):
            br0, bc0, br1, bc1 = b["bbox"]
            if relation == "same_row_band":
                linked = _intervals_overlap(ar0, ar1, br0, br1)
            elif relation == "same_column_band":
                linked = _intervals_overlap(ac0, ac1, bc0, bc1)
            else:
                linked = _min_manhattan_gap(a["cells"], b["cells"]) <= threshold
            if linked:
                edges.add((i, j))
    return edges


_DEGREE_SHAPES = [
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
]


def _degree_shape(rng: np.random.Generator) -> frozenset:
    """A small object: half the time a canonical mini-shape, half a random
    polyomino, broadening the shape vocabulary for the degree task."""
    if rng.random() < 0.5:
        return frozenset(_DEGREE_SHAPES[int(rng.integers(0, len(_DEGREE_SHAPES)))])
    return random_shape(rng, (2, 4), connectivity=4)


@register("201")
def generate_recolor_by_graph_degree(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        object_color = unused_colors(rng, 1, bg)[0]
        relation = str(rng.choice(["same_row_band", "same_column_band", "manhattan_gap_le_threshold"]))
        threshold = int(rng.integers(2, 4))
        # H and W sampled independently across the ARC-typical span.
        grid_h = int(rng.integers(10, 21))
        grid_w = int(rng.integers(10, 21))
        grid = make_grid(grid_h, grid_w, bg)
        n_objects = int(rng.integers(4, 8))
        boxes: list[tuple[int, int, int, int]] = []
        placed = True
        for _i in range(n_objects):
            shape = _degree_shape(rng)
            cells = place_shape(grid, rng, shape, object_color, bg, boxes, gap=1, max_attempts=200)
            if cells is None:
                placed = False
                break
        if not placed:
            continue
        objects = _components_of(grid, bg, object_color)
        if len(objects) != n_objects:
            continue
        edges = _graph_edges(objects, relation, threshold)
        degrees = [0] * len(objects)
        for i, j in edges:
            degrees[i] += 1
            degrees[j] += 1
        # Require degree variety so the recoloring is not trivially constant.
        present = sorted(set(degrees))
        if len(present) < 2:
            continue
        degree_colors = unused_colors(rng, len(present), bg, object_color)
        mapping = _mapping_list(present, degree_colors, "degree")
        lookup = _mapping_lookup(mapping, "degree")
        object_cells = {cell for obj in objects for cell in obj["cells"]}
        texture = _add_background_texture(
            grid, rng, bg=bg,
            exclude_colors=(object_color, *degree_colors),
            protected_cells=object_cells,
        )
        output = grid.copy()
        for obj, degree in zip(objects, degrees):
            for r, c in obj["cells"]:
                output[r, c] = lookup[degree]
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(output),
            "metadata": {
                "background_texture_color": texture["color"],
                "background_color": bg,
                "object_color": object_color,
                "adjacency_relation": relation,
                "gap_threshold": threshold,
                "degree_to_color": mapping,
                "tie_policy": "no_ties_degree_is_computed_from_all_qualifying_edges",
            },
        }
    raise RuntimeError("Task 201: failed to generate valid graph-degree example")


@register("202")
def generate_fill_unique_aligned_pair_gaps(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, 10))
    color_a, color_b, fill = unused_colors(rng, 3, bg)
    axis = str(rng.choice(["horizontal", "vertical"]))
    # Sample the two grid sides independently across the ARC-typical span. Each
    # pair lives on its own line (row for horizontal, column for vertical); the
    # cross axis must hold the two endpoints with a >=1 background gap, and the
    # along axis holds pair lines spaced >=2 apart (endpoints never 4-adjacent).
    if axis == "horizontal":
        grid_w = int(rng.integers(7, 21))
        grid_h = int(rng.integers(7, 21))
        slots = list(range(1, grid_h - 1, 2))
        rng.shuffle(slots)
        n_pairs = int(rng.integers(2, max(3, len(slots) + 1)))
        lines = sorted(int(s) for s in slots[:n_pairs])
        grid = make_grid(grid_h, grid_w, bg)
        pairs = []
        for r in lines:
            c1 = int(rng.integers(0, max(1, grid_w // 2 - 1)))
            c2 = int(rng.integers(c1 + 2, grid_w))
            grid[r, c1] = color_a
            grid[r, c2] = color_b
            pairs.append(((r, c1), (r, c2)))
    else:
        grid_w = int(rng.integers(7, 21))
        grid_h = int(rng.integers(7, 21))
        slots = list(range(1, grid_w - 1, 2))
        rng.shuffle(slots)
        n_pairs = int(rng.integers(2, max(3, len(slots) + 1)))
        lines = sorted(int(s) for s in slots[:n_pairs])
        grid = make_grid(grid_h, grid_w, bg)
        pairs = []
        for c in lines:
            r1 = int(rng.integers(0, max(1, grid_h // 2 - 1)))
            r2 = int(rng.integers(r1 + 2, grid_h))
            grid[r1, c] = color_a
            grid[r2, c] = color_b
            pairs.append(((r1, c), (r2, c)))
    # Collect the gap cells per pair (the straight background run between the
    # endpoints) so texture can be kept off every fill track and the endpoints.
    fill_paths: list[tuple[int, int]] = []
    for a, b in pairs:
        dr = int(np.sign(b[0] - a[0]))
        dc = int(np.sign(b[1] - a[1]))
        r, c = a[0] + dr, a[1] + dc
        while (r, c) != b:
            fill_paths.append((r, c))
            r += dr
            c += dc
    protected = set(fill_paths)
    for a, b in pairs:
        protected.add(a)
        protected.add(b)
    # Pairing is keyed strictly by colour (each endpoint is found by an exact
    # color_a / color_b match) and the fill colour is disjoint, so decoration in
    # a fourth disjoint colour cannot be mistaken for an endpoint or a fill cell.
    # Scatter varied isolated decoration shapes (single cells through small
    # polyominoes) off every fill track and endpoint — this both varies the
    # background (solid+decorated mix) and broadens the shape vocabulary, while
    # the prompt records and ignores the decoration colour.
    texture = _scatter_decoration_shapes(
        grid, rng, bg=bg, exclude_colors=(color_a, color_b, fill),
        protected_cells=protected,
    )
    output = grid.copy()
    for r, c in fill_paths:
        output[r, c] = fill
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(output),
        "metadata": {
            "background_texture_color": texture["color"],
            "background_color": bg,
            "object_a_color": color_a,
            "object_b_color": color_b,
            "fill_color": fill,
            "pairing_axis": axis,
            "pairing_rule": "unique_opposite_color_object_sharing_the_same_row_or_column",
        },
    }


def _bbox_property(comp: dict, prop: str) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if prop == "bbox_height":
        return height
    if prop == "bbox_width":
        return width
    if prop == "bbox_area":
        return height * width
    return 2 * (height + width)


@register("203")
def generate_recolor_by_bbox_property(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        object_color = unused_colors(rng, 1, bg)[0]
        prop = str(rng.choice(["bbox_height", "bbox_width", "bbox_area", "bbox_perimeter"]))
        # H and W sampled independently across the ARC-typical small-to-large span.
        grid_h = int(rng.integers(9, 21))
        grid_w = int(rng.integers(9, 21))
        grid = make_grid(grid_h, grid_w, bg)
        n_objects = int(rng.integers(3, 7))
        boxes: list[tuple[int, int, int, int]] = []
        placed = True
        for _i in range(n_objects):
            # Solid rectangles of random size; bbox == the rectangle itself.
            rh = int(rng.integers(1, 5))
            rw = int(rng.integers(1, 5))
            shape = {(rr, cc) for rr in range(rh) for cc in range(rw)}
            for _attempt in range(120):
                top = int(rng.integers(0, grid_h - rh + 1))
                left = int(rng.integers(0, grid_w - rw + 1))
                try:
                    _place_shape(grid, shape, top, left, object_color, boxes, gap=1)
                    break
                except ValueError:
                    continue
            else:
                placed = False
                break
        if not placed:
            continue
        comps = _components_of(grid, bg, object_color)
        if len(comps) != n_objects:
            continue
        values = sorted({_bbox_property(comp, prop) for comp in comps})
        # Require >= 2 distinct property values so the recoloring is not a
        # trivial constant fill, and keep the color demand within the palette.
        if not (2 <= len(values) <= 8):
            continue
        value_colors = unused_colors(rng, len(values), bg, object_color)
        mapping = _mapping_list(values, value_colors, "property_value")
        lookup = _mapping_lookup(mapping, "property_value")
        object_cells = {cell for comp in comps for cell in comp["cells"]}
        texture = _add_background_texture(
            grid, rng, bg=bg,
            exclude_colors=(object_color, *value_colors),
            protected_cells=object_cells,
        )
        output = grid.copy()
        for comp in comps:
            color = lookup[_bbox_property(comp, prop)]
            for r, c in comp["cells"]:
                output[r, c] = color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(output),
            "metadata": {
                "background_texture_color": texture["color"],
                "background_color": bg,
                "object_color": object_color,
                "bbox_property": prop,
                "property_to_color": mapping,
            },
        }
    raise RuntimeError("Task 203: failed to generate valid bbox-property example")


def _equivalent(a: dict, b: dict, relation: str) -> bool:
    if relation == "same_shape":
        return normalize_shape(a["cells"]) == normalize_shape(b["cells"])
    if relation == "same_bbox_size":
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        return (ar1 - ar0 + 1, ac1 - ac0 + 1) == (br1 - br0 + 1, bc1 - bc0 + 1)
    return len(a["cells"]) == len(b["cells"])


def _apply_class_action(output: np.ndarray, comp: dict, action: str, color: int, bg: int) -> None:
    if action == "recolor_object":
        for r, c in comp["cells"]:
            output[r, c] = color
    elif action == "mark_first_cell":
        output[min(comp["cells"])] = color
    else:
        for r, c in comp["cells"]:
            output[r, c] = bg


def _equivalence_key(shape: set[tuple[int, int]] | frozenset[tuple[int, int]], relation: str):
    if relation == "same_shape":
        return normalize_shape(shape)
    r0, c0, r1, c1 = bounding_box(shape)
    if relation == "same_bbox_size":
        return (r1 - r0 + 1, c1 - c0 + 1)
    return len(shape)


def _place_shape_anywhere(
    grid: np.ndarray,
    rng: np.random.Generator,
    shape: set[tuple[int, int]] | frozenset[tuple[int, int]],
    color: int,
    boxes: list[tuple[int, int, int, int]],
    attempts: int = 120,
) -> bool:
    h, w = grid.shape
    sh, sw = _shape_size(shape)
    if sh > h or sw > w:
        return False
    for _ in range(attempts):
        top = int(rng.integers(0, h - sh + 1))
        left = int(rng.integers(0, w - sw + 1))
        try:
            _place_shape(grid, shape, top, left, color, boxes, gap=1)
            return True
        except ValueError:
            continue
    return False


@register("227")
def generate_equivalence_class_transform(rng: np.random.Generator) -> dict:
    relation = str(rng.choice(["same_shape", "same_bbox_size", "same_area"]))
    matched_action = str(rng.choice(["recolor_object", "mark_first_cell", "erase_object"]))
    unmatched_options = [a for a in ["recolor_object", "mark_first_cell", "erase_object"] if a != matched_action]
    unmatched_action = str(rng.choice(unmatched_options))
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        target_color, reference_color, matched_color, unmatched_color = unused_colors(rng, 4, bg)
        # H and W sampled independently across the ARC-typical span.
        grid_h = int(rng.integers(10, 21))
        grid_w = int(rng.integers(10, 21))
        n_targets = int(rng.integers(3, 6))
        n_refs = int(rng.integers(2, 5))
        # Build target shapes; then choose reference shapes so the matched set is
        # guaranteed non-empty AND a proper subset (some target stays unmatched).
        target_shapes = [random_shape(rng, (3, 5), connectivity=4) for _ in range(n_targets)]
        target_keys = [_equivalence_key(s, relation) for s in target_shapes]
        distinct_keys = list({k: None for k in target_keys})
        if len(distinct_keys) < 2:
            continue
        # Match exactly the first distinct key's targets via a reference; leave at
        # least one other key unreferenced so its targets are unmatched.
        idx = int(rng.integers(0, len(distinct_keys)))
        matched_key = distinct_keys[idx]
        matched_shape = next(s for s, k in zip(target_shapes, target_keys) if k == matched_key)
        reference_shapes = [matched_shape]
        # Remaining references must NOT introduce any other target key as a match.
        forbidden = set(distinct_keys) - {matched_key}
        guard = 0
        while len(reference_shapes) < n_refs and guard < 200:
            guard += 1
            cand = random_shape(rng, (3, 5), connectivity=4)
            if _equivalence_key(cand, relation) not in forbidden:
                reference_shapes.append(cand)
        grid = make_grid(grid_h, grid_w, bg)
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for shape in target_shapes:
            if not _place_shape_anywhere(grid, rng, shape, target_color, boxes):
                ok = False
                break
        if ok:
            for shape in reference_shapes:
                if not _place_shape_anywhere(grid, rng, shape, reference_color, boxes):
                    ok = False
                    break
        if not ok:
            continue
        targets = _components_of(grid, bg, target_color)
        refs = _components_of(grid, bg, reference_color)
        if len(targets) != n_targets or len(refs) != len(reference_shapes):
            continue
        classes = [any(_equivalent(t, ref, relation) for ref in refs) for t in targets]
        # Enforce a genuine mix: at least one matched and one unmatched target.
        if not (any(classes) and not all(classes)):
            continue
        semantic_cells = {
            cell for comp in targets + refs for cell in comp["cells"]
        }
        texture = _add_background_texture(
            grid, rng, bg=bg,
            exclude_colors=(target_color, reference_color, matched_color, unmatched_color),
            protected_cells=semantic_cells,
        )
        output = grid.copy()
        for target in targets:
            matched = any(_equivalent(target, ref, relation) for ref in refs)
            if matched:
                _apply_class_action(output, target, matched_action, matched_color, bg)
            else:
                _apply_class_action(output, target, unmatched_action, unmatched_color, bg)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(output),
            "metadata": {
                "background_texture_color": texture["color"],
                "background_color": bg,
                "target_color": target_color,
                "reference_color": reference_color,
                "equivalence_relation": relation,
                "matched_action": matched_action,
                "unmatched_action": unmatched_action,
                "matched_color": matched_color,
                "unmatched_color": unmatched_color,
            },
        }
    raise RuntimeError("Task 227: failed to generate valid equivalence example")


def _count_color_components(grid: np.ndarray, color: int, bg: int) -> int:
    return len(_components_of(grid, bg, color))


def _count_set(grid: np.ndarray, color: int, bg: int, count_type: str) -> int:
    if count_type == "cells":
        return int(np.count_nonzero(grid == color))
    return _count_color_components(grid, color, bg)


def _count_operation(a: int, b: int, op: str) -> int:
    if op == "sum":
        return a + b
    if op == "absolute_difference":
        return abs(a - b)
    if op == "a_minus_b_floor0":
        return max(0, a - b)
    if op == "b_minus_a_floor0":
        return max(0, b - a)
    return a * b


def _place_count_items(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    count_type: str,
    item_count: int,
    reserved: set[tuple[int, int]],
) -> None:
    """Place `item_count` color items, each its own 4-connected component
    separated from everything else by a one-cell halo.

    For ``count_type == "cells"`` items are single cells, so the cell count
    equals the item count. For ``count_type == "objects"`` items are varied
    small polyominoes (2-4 cells) so the object (component) count equals the
    item count while widening the shape vocabulary.
    """
    h, w = grid.shape
    positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(positions)
    placed = 0
    for r, c in positions:
        if placed >= item_count:
            return
        if count_type == "cells":
            shape = {(0, 0)}
        else:
            size = int(rng.integers(2, 5))
            shape = random_shape(rng, (size, size), connectivity=4)
        cells = {(r + sr, c + sc) for sr, sc in shape}
        halo = set()
        for cr, cc in cells:
            halo.update((rr, hc) for rr in range(cr - 1, cr + 2) for hc in range(cc - 1, cc + 2))
        if any((rr, cc) in reserved or not (0 <= rr < h and 0 <= cc < w) for rr, cc in halo):
            continue
        for cell in cells:
            grid[cell] = color
        reserved.update(halo)
        placed += 1
    if placed != item_count:
        raise RuntimeError("failed to place count items")


@register("228")
def generate_count_operation_text(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        color_a, color_b = unused_colors(rng, 2, bg)
        count_a_type = str(rng.choice(["cells", "objects"]))
        count_b_type = str(rng.choice(["cells", "objects"]))
        operation = str(rng.choice(["sum", "absolute_difference", "a_minus_b_floor0", "b_minus_a_floor0", "product"]))
        # H and W sampled independently across the ARC-typical span (text output
        # is tiny, so the grid can reach the high end of the envelope).
        grid_h = int(rng.integers(8, 23))
        grid_w = int(rng.integers(8, 23))
        grid = make_grid(grid_h, grid_w, bg)
        reserved: set[tuple[int, int]] = set()
        n_a = int(rng.integers(1, 6))
        n_b = int(rng.integers(1, 6))
        try:
            _place_count_items(rng, grid, color_a, count_a_type, n_a, reserved)
            _place_count_items(rng, grid, color_b, count_b_type, n_b, reserved)
        except RuntimeError:
            continue
        count_a = _count_set(grid, color_a, bg, count_a_type)
        count_b = _count_set(grid, color_b, bg, count_b_type)
        result = _count_operation(count_a, count_b, operation)
        # Texture uses a colour disjoint from both counted colours, so it never
        # affects either cell- or object-count.
        texture = _add_background_texture(
            grid, rng, bg=bg, exclude_colors=(color_a, color_b),
        )
        return {
            "input": grid_to_list(grid),
            "output": str(result),
            "metadata": {
                "background_texture_color": texture["color"],
                "background_color": bg,
                "set_a_color": color_a,
                "set_b_color": color_b,
                "set_a_count_type": count_a_type,
                "set_b_count_type": count_b_type,
                "operation": operation,
            },
        }
    raise RuntimeError("Task 228: failed to generate valid count example")


@register("229")
def generate_count_expression_bar(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, 10))
        color_a, color_b, anchor_color, bar_color = unused_colors(rng, 4, bg)
        count_a_type = str(rng.choice(["cells", "objects"]))
        count_b_type = str(rng.choice(["cells", "objects"]))
        operation = str(rng.choice(["sum", "absolute_difference", "a_minus_b_floor0", "b_minus_a_floor0"]))
        direction = str(rng.choice(["east", "south"]))
        # H and W sampled independently across the ARC-typical span. The drawn
        # bar (a grid output) keeps the upper bound moderate so bundles still fit.
        grid_h = int(rng.integers(8, 21))
        grid_w = int(rng.integers(8, 21))
        grid = make_grid(grid_h, grid_w, bg)
        anchor = (0, 0)
        grid[anchor] = anchor_color
        reserved = {(0, 0), (0, 1), (1, 0), (1, 1)}
        n_a = int(rng.integers(1, 6))
        n_b = int(rng.integers(1, 6))
        try:
            _place_count_items(rng, grid, color_a, count_a_type, n_a, reserved)
            _place_count_items(rng, grid, color_b, count_b_type, n_b, reserved)
        except RuntimeError:
            continue
        count_a = _count_set(grid, color_a, bg, count_a_type)
        count_b = _count_set(grid, color_b, bg, count_b_type)
        length = _count_operation(count_a, count_b, operation)
        max_len = grid.shape[1] if direction == "east" else grid.shape[0]
        if 1 <= length <= max_len:
            dr, dc = (0, 1) if direction == "east" else (1, 0)
            # Keep the bar track clear of texture so the input bar track is plain
            # background; texture uses a colour disjoint from every drawn colour.
            bar_cells = {(anchor[0] + dr * k, anchor[1] + dc * k) for k in range(max_len)}
            texture = _add_background_texture(
                grid, rng, bg=bg,
                exclude_colors=(color_a, color_b, anchor_color, bar_color),
                protected_cells=bar_cells,
            )
            output = grid.copy()
            for k in range(length):
                output[anchor[0] + dr * k, anchor[1] + dc * k] = bar_color
            return {
                "input": grid_to_list(grid),
                "output": grid_to_list(output),
                "metadata": {
                    "background_texture_color": texture["color"],
                    "background_color": bg,
                    "set_a_color": color_a,
                    "set_b_color": color_b,
                    "set_a_count_type": count_a_type,
                    "set_b_count_type": count_b_type,
                    "operation": operation,
                    "anchor_color": anchor_color,
                    "bar_color": bar_color,
                    "bar_direction": direction,
                    "bar_includes_anchor": True,
                },
            }
    raise RuntimeError("Task 229: failed to generate valid bar example")
