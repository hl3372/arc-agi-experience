"""Symmetry and comparison second-batch task generators, slice F."""
from __future__ import annotations

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from common.objects import bounding_box, random_polyomino
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    place_shape,
    unused_colors,
)
from registry import register


Cell = tuple[int, int]
Shape = frozenset[Cell]


def _normalize_shape(cells: set[Cell] | frozenset[Cell]) -> Shape:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _transpose_shape(shape: Shape) -> Shape:
    return _normalize_shape({(c, r) for r, c in shape})


def _shape_size(shape: Shape) -> tuple[int, int]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    return (max_r + 1, max_c + 1)


def _is_rectangle(shape: Shape) -> bool:
    h, w = _shape_size(shape)
    return len(shape) == h * w


def _scene_shape_banks() -> dict[str, dict[int, list[Shape]]]:
    height_banks = {
        1: [
            frozenset({(0, 0), (0, 1)}),
            frozenset({(0, 0), (0, 1), (0, 2)}),
        ],
        2: [
            frozenset({(0, 0), (1, 0)}),
            frozenset({(0, 0), (1, 0), (1, 1)}),
            frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        ],
        3: [
            frozenset({(0, 0), (1, 0), (2, 0)}),
            frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
            frozenset({(0, 1), (1, 0), (1, 1), (2, 0)}),
        ],
    }
    width_banks = {
        measure: [_transpose_shape(shape) for shape in shapes]
        for measure, shapes in height_banks.items()
    }
    cell_count_banks = {
        2: [
            frozenset({(0, 0), (0, 1)}),
            frozenset({(0, 0), (1, 0)}),
        ],
        4: [
            frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
            frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
            frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
        ],
        6: [
            frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}),
            frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}),
            frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
        ],
    }
    return {
        "height": height_banks,
        "width": width_banks,
        "cell_count": cell_count_banks,
    }


SCENE_SHAPE_BANKS = _scene_shape_banks()


def _choose_scene_shapes(
    rng: np.random.Generator,
    measure_dimension: str,
    measure_values: list[int],
) -> list[tuple[int, Shape]]:
    banks = SCENE_SHAPE_BANKS[measure_dimension]
    duplicate_measure = int(rng.choice(measure_values))
    requested_measures = measure_values + [duplicate_measure]
    rng.shuffle(requested_measures)
    chosen: list[tuple[int, Shape]] = []
    used_shapes: dict[int, set[Shape]] = {}
    for measure in requested_measures:
        options = list(banks[measure])
        prior = used_shapes.get(measure, set())
        if prior:
            distinct_options = [shape for shape in options if shape not in prior]
            if distinct_options:
                options = distinct_options
        shape = options[int(rng.integers(0, len(options)))]
        used_shapes.setdefault(measure, set()).add(shape)
        chosen.append((measure, shape))
    if any(not _is_rectangle(shape) for _, shape in chosen):
        return chosen
    for idx, (measure, shape) in enumerate(chosen):
        non_rect_options = [candidate for candidate in banks[measure] if not _is_rectangle(candidate) and candidate != shape]
        if non_rect_options:
            replacement = non_rect_options[int(rng.integers(0, len(non_rect_options)))]
            chosen[idx] = (measure, replacement)
            return chosen
    return chosen


def _mirror_cell(r: int, c: int, shape: tuple[int, int], axis: str) -> Cell:
    h, w = shape
    if axis == "horizontal":
        return (h - 1 - r, c)
    if axis == "vertical":
        return (r, w - 1 - c)
    if axis == "main_diagonal":
        return (c, r)
    if axis == "anti_diagonal":
        return (w - 1 - c, h - 1 - r)
    raise ValueError(f"unknown axis: {axis}")


def _rotate_cell(r: int, c: int, shape: tuple[int, int], angle: int) -> Cell:
    h, w = shape
    if angle == 180:
        return (h - 1 - r, w - 1 - c)
    if angle == 90:
        return (c, h - 1 - r)
    if angle == 270:
        return (w - 1 - c, r)
    raise ValueError(f"unknown angle: {angle}")


def _orbit(
    start: Cell,
    transform,
    shape: tuple[int, int],
    param,
) -> list[Cell]:
    cells = []
    cur = start
    while cur not in cells:
        cells.append(cur)
        cur = transform(cur[0], cur[1], shape, param)
    return cells


def _collect_orbits(
    shape: tuple[int, int],
    transform,
    param,
) -> list[list[Cell]]:
    seen: set[Cell] = set()
    orbits = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if (r, c) in seen:
                continue
            orbit = _orbit((r, c), transform, shape, param)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _fill_orbits(
    rng: np.random.Generator,
    shape: tuple[int, int],
    transform,
    param,
    colors: list[int],
) -> np.ndarray:
    grid = make_grid(shape[0], shape[1], fill=colors[0])
    for orbit in _collect_orbits(shape, transform, param):
        color = int(colors[int(rng.integers(0, len(colors)))])
        for rr, cc in orbit:
            grid[rr, cc] = color
    return grid


def _is_symmetric(grid: np.ndarray, label: str) -> bool:
    if label == "horizontal_reflection":
        return np.array_equal(grid, np.flipud(grid))
    if label == "vertical_reflection":
        return np.array_equal(grid, np.fliplr(grid))
    if label == "main_diagonal_reflection":
        return grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T)
    if label == "anti_diagonal_reflection":
        return grid.shape[0] == grid.shape[1] and np.array_equal(grid, np.fliplr(np.flipud(grid)).T)
    if label == "rotational_180":
        return np.array_equal(grid, np.rot90(grid, 2))
    raise ValueError(f"unknown label: {label}")


def _fill_horizontal_vertical_orbits(
    rng: np.random.Generator,
    shape: tuple[int, int],
    colors: list[int],
) -> np.ndarray:
    h, w = shape
    grid = make_grid(h, w, fill=colors[0])
    seen: set[Cell] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (h - 1 - r, c),
                (r, w - 1 - c),
                (h - 1 - r, w - 1 - c),
            }
            color = int(colors[int(rng.integers(0, len(colors)))])
            for rr, cc in orbit:
                grid[rr, cc] = color
            seen.update(orbit)
    return grid


def _fill_full_reflection_orbits(
    rng: np.random.Generator,
    size: int,
    colors: list[int],
) -> np.ndarray:
    grid = make_grid(size, size, fill=colors[0])
    seen: set[Cell] = set()
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (size - 1 - r, c),
                (r, size - 1 - c),
                (size - 1 - r, size - 1 - c),
                (c, r),
                (size - 1 - c, r),
                (c, size - 1 - r),
                (size - 1 - c, size - 1 - r),
            }
            color = int(colors[int(rng.integers(0, len(colors)))])
            for rr, cc in orbit:
                grid[rr, cc] = color
            seen.update(orbit)
    return grid


def _patterned_fill(
    rng: np.random.Generator,
    shape: tuple[int, int],
    colors: list[int],
) -> tuple[np.ndarray, str]:
    mode = str(rng.choice(["solid", "noise", "checker", "stripes"]))
    h, w = shape
    if mode == "solid":
        return make_grid(h, w, fill=int(colors[int(rng.integers(0, len(colors)))])), mode
    if mode == "noise":
        grid = np.array(rng.choice(colors, size=(h, w), replace=True), dtype=np.uint8)
        return grid, mode
    grid = make_grid(h, w, fill=int(colors[0]))
    if mode == "checker":
        for r in range(h):
            for c in range(w):
                grid[r, c] = int(colors[(r + c) % len(colors)])
        return grid, mode
    vertical = bool(rng.integers(0, 2))
    stripe_width = int(rng.integers(1, 4))
    for r in range(h):
        for c in range(w):
            idx = c if vertical else r
            grid[r, c] = int(colors[(idx // stripe_width) % len(colors)])
    return grid, "vertical_stripes" if vertical else "horizontal_stripes"


# Task 101 moved to generators/symmetry_violations_101.py in the September 2026
# rebuild; every helper above stays, because 103 and 281 still use them.
# Task 102 moved to generators/symmetry_report_102.py in the September 2026
# rebuild; every helper above stays, because 103 and 281 still use them.
@register("103")
def generate_visual_legend_attribute_to_color(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg, object_color, separator_color, distractor_color = unused_colors(rng, 4)
        mapped_colors = unused_colors(rng, 3, bg, object_color, separator_color, distractor_color)
        order = list(range(3))
        rng.shuffle(order)
        separator_axis = str(rng.choice(["horizontal", "vertical"]))
        if separator_axis == "horizontal":
            legend_side = str(rng.choice(["top", "bottom"]))
        else:
            legend_side = str(rng.choice(["left", "right"]))
        measure_dimension = str(rng.choice(["height", "width", "cell_count"]))
        grid = make_grid(14, 14, fill=bg)
        out = grid.copy()
        legend_entries = []
        if measure_dimension == "cell_count":
            specs = [
                {"height": 1, "width": 2, "measure_value": 2},
                {"height": 2, "width": 2, "measure_value": 4},
                {"height": 2, "width": 3, "measure_value": 6},
            ]
        else:
            specs = [
                {
                    "height": measure if measure_dimension == "height" else 2,
                    "width": measure if measure_dimension == "width" else 2,
                    "measure_value": measure,
                }
                for measure in [1, 2, 3]
            ]
        measure_values = [int(spec["measure_value"]) for spec in specs]
        scene_shapes = _choose_scene_shapes(rng, measure_dimension, measure_values)

        # Geometry is parametrized so the grid is not a fixed square. Two
        # independent axes:
        #   * legend-axis (perpendicular to the separator): holds the three
        #     swatches spaced `swatch_gap` apart, plus a margin on each end.
        #   * split-axis (parallel-stacking of legend / separator / scene):
        #     legend band is `legend_depth` deep, scene band is `scene_extent`.
        # For a horizontal separator the legend-axis is the WIDTH and the
        # split-axis is the HEIGHT; for a vertical separator the roles swap.
        # Because the separator orientation is random, H and W each take both
        # roles across seeds, so the size distribution is wide and decorrelated.
        max_swatch_w = max(int(spec["width"]) for spec in specs)
        max_swatch_h = max(int(spec["height"]) for spec in specs)
        swatch_gap = int(rng.integers(4, 7))
        margin_lo = int(rng.integers(1, 3))
        margin_hi = int(rng.integers(1, 4))
        # legend band along the swatch direction: object cells start at +1, the
        # swatch row/col sits at legend_depth-2, leaving a clear margin.
        legend_depth = int(rng.integers(7, 10))
        scene_extent = int(rng.integers(6, 13))

        legend_axis_len = margin_lo + 2 * swatch_gap + max(max_swatch_w, max_swatch_h) + margin_hi
        split_axis_len = legend_depth + 1 + scene_extent

        if separator_axis == "horizontal":
            grid_h = split_axis_len
            grid_w = legend_axis_len
        else:
            grid_h = legend_axis_len
            grid_w = split_axis_len
        if grid_h > 30 or grid_w > 30:
            continue
        grid = make_grid(grid_h, grid_w, fill=bg)
        out = grid.copy()

        separator_row = None
        separator_col = None
        legend_row0 = 0
        legend_col0 = 0
        scene_row0 = 0
        scene_col0 = 0
        scene_h = 0
        scene_w = 0

        if separator_axis == "horizontal":
            if legend_side == "top":
                separator_row = legend_depth
                legend_region = "rows_above_separator"
                scene_region = "rows_below_separator"
                scene_side = "bottom"
                legend_row0, legend_col0 = 0, 0
                scene_row0, scene_col0, scene_h, scene_w = legend_depth + 1, 0, scene_extent, grid_w
            else:
                separator_row = scene_extent
                legend_region = "rows_below_separator"
                scene_region = "rows_above_separator"
                scene_side = "top"
                scene_row0, scene_col0, scene_h, scene_w = 0, 0, scene_extent, grid_w
                legend_row0, legend_col0 = scene_extent + 1, 0
        else:
            if legend_side == "left":
                separator_col = legend_depth
                legend_region = "cols_left_of_separator"
                scene_region = "cols_right_of_separator"
                scene_side = "right"
                legend_row0, legend_col0 = 0, 0
                scene_row0, scene_col0, scene_h, scene_w = 0, legend_depth + 1, grid_h, scene_extent
            else:
                separator_col = scene_extent
                legend_region = "cols_right_of_separator"
                scene_region = "cols_left_of_separator"
                scene_side = "left"
                scene_row0, scene_col0, scene_h, scene_w = 0, 0, grid_h, scene_extent
                legend_row0, legend_col0 = 0, scene_extent + 1

        if separator_axis == "horizontal":
            grid[separator_row, :] = separator_color
            out[separator_row, :] = separator_color
            swatch_row = legend_row0 + legend_depth - 2
            for idx, spec in enumerate(specs):
                color = int(mapped_colors[order[idx]])
                r0 = legend_row0 + 1
                c0 = legend_col0 + margin_lo + idx * swatch_gap
                grid[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = object_color
                grid[swatch_row, c0 : c0 + spec["width"]] = color
                out[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = object_color
                out[swatch_row, c0 : c0 + spec["width"]] = color
                legend_entries.append({"measure_value": spec["measure_value"], "output_color": color})
        else:
            grid[:, separator_col] = separator_color
            out[:, separator_col] = separator_color
            swatch_col = legend_col0 + legend_depth - 2
            for idx, spec in enumerate(specs):
                color = int(mapped_colors[order[idx]])
                r0 = legend_row0 + margin_lo + idx * swatch_gap
                c0 = legend_col0 + 1
                grid[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = object_color
                grid[r0, swatch_col] = color
                out[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = object_color
                out[r0, swatch_col] = color
                legend_entries.append({"measure_value": spec["measure_value"], "output_color": color})

        scene_grid = make_grid(scene_h, scene_w, fill=bg)
        placed_boxes: list[tuple[int, int, int, int]] = []
        distractor_shape = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
        distractor_cells = place_shape(
            scene_grid,
            rng,
            distractor_shape,
            distractor_color,
            bg,
            placed_boxes,
            gap=1,
        )
        if distractor_cells is None:
            continue
        scene_objects = []
        for measure_value, shape in scene_shapes:
            cells = place_shape(
                scene_grid,
                rng,
                shape,
                object_color,
                bg,
                placed_boxes,
                gap=1,
            )
            if cells is None:
                break
            scene_objects.append({
                "measure_value": int(measure_value),
                "cells": cells,
                "normalized_shape": _normalize_shape(cells),
            })
        else:
            # Optional patterned background, applied to scene background cells
            # only. The rule selects scene objects by `object_color` connected
            # components and the distractor by `distractor_color`, so texture in
            # any *other* color is provably ignored by both the rule and the
            # verifier; we exclude every semantic color to keep that invariant.
            # The same textured scene is copied into both grid and out, and the
            # output recolors object cells only, so textured cells are identical
            # in input and output (left unchanged, as the prompt states).
            protected = set(distractor_cells)
            for obj in scene_objects:
                protected |= set(obj["cells"])
            bg_texture = add_patterned_background(
                scene_grid,
                rng,
                bg=bg,
                exclude_colors=[object_color, separator_color, distractor_color, *mapped_colors],
                protected_cells=protected,
                solid_probability=0.35,
            )
            grid[scene_row0 : scene_row0 + scene_h, scene_col0 : scene_col0 + scene_w] = scene_grid
            out[scene_row0 : scene_row0 + scene_h, scene_col0 : scene_col0 + scene_w] = scene_grid
            mapping = {
                int(item["measure_value"]): int(item["output_color"])
                for item in legend_entries
            }
            for obj in scene_objects:
                mapped = mapping[obj["measure_value"]]
                for local_r, local_c in obj["cells"]:
                    out[scene_row0 + local_r, scene_col0 + local_c] = mapped
            return {
                "input": grid_to_list(grid),
                "output": grid_to_list(out),
                "metadata": {
                    "background_color": bg,
                    "object_color": object_color,
                    "separator_color": separator_color,
                    "distractor_color": distractor_color,
                    "separator_axis": separator_axis,
                    "legend_side": legend_side,
                    "scene_side": scene_side,
                    "measure_dimension": measure_dimension,
                    "separator_row": separator_row,
                    "separator_col": separator_col,
                    "legend_rule": f"{measure_dimension}_to_scene_object_color",
                    "legend_region": legend_region,
                    "scene_region": scene_region,
                    "legend_entries": legend_entries,
                    "scene_object_count": len(scene_objects),
                    "background_pattern": bg_texture,
                },
            }
    raise RuntimeError("Task 103: failed to generate example")


@register("104")
def generate_apply_inferred_color_permutation(rng: np.random.Generator) -> dict:
    n_colors = int(rng.integers(4, 7))
    colors = unused_colors(rng, n_colors)
    targets = colors.copy()
    for _ in range(100):
        rng.shuffle(targets)
        if all(a != b for a, b in zip(colors, targets)):
            break
    else:
        targets = colors[1:] + colors[:1]
    mapping = {int(a): int(b) for a, b in zip(colors, targets)}

    def random_palette_grid() -> np.ndarray:
        # Sample H and W independently in the ARC-small range; every panel must
        # contain all source colors at least once so the mapping is fully
        # inferable (before/after) and every source color is queried (query).
        # min cell budget is n_colors, so the lower bound clamps to fit it.
        min_side = 4
        h = int(rng.integers(min_side, 16))
        w = int(rng.integers(min_side, 16))
        while h * w < n_colors:
            w += 1
        grid = np.array(rng.choice(colors, size=(h, w), replace=True), dtype=np.uint8)
        seed_positions = rng.choice(h * w, size=n_colors, replace=False)
        for color, flat in zip(colors, seed_positions):
            grid.flat[int(flat)] = color
        return grid

    before = random_palette_grid()
    after = np.vectorize(mapping.__getitem__, otypes=[np.uint8])(before)
    query = random_palette_grid()
    out = np.vectorize(mapping.__getitem__, otypes=[np.uint8])(query)
    return {
        "input": {
            "before": grid_to_list(before),
            "after": grid_to_list(after),
            "query": grid_to_list(query),
        },
        "output": grid_to_list(out),
        "metadata": {
            "color_mapping": [
                {"from_color": int(a), "to_color": int(mapping[int(a)])}
                for a in colors
            ],
            "mapping_type": "one_to_one_color_permutation",
        },
    }


def _header_colors(rng: np.random.Generator, n: int, blank: int) -> list[int]:
    """Pick n header colors. Distinct while the palette allows it, then fall back
    to repeats. The fill rule is positional (it reads the header *cell*, not its
    color identity), so repeated header colors keep the answer well-defined and
    let the table grow past the distinct-color capacity."""
    pool = [c for c in range(NUM_COLORS) if c != blank]
    rng.shuffle(pool)
    colors = pool[:n]
    while len(colors) < n:
        colors.append(int(pool[int(rng.integers(0, len(pool)))]))
    return [int(c) for c in colors]


@register("107")
def generate_fill_visual_table_by_headers(rng: np.random.Generator) -> dict:
    blank = int(rng.integers(0, NUM_COLORS))
    # Sample table height and width independently and widely. The combine rule is
    # positional, so header colors may repeat on large tables (see _header_colors).
    n_rows = int(rng.integers(3, 16))
    n_cols = int(rng.integers(3, 16))
    n_row_headers = n_rows - 1
    n_col_headers = n_cols - 1
    row_colors = _header_colors(rng, n_row_headers, blank)
    col_colors = _header_colors(rng, n_col_headers, blank)
    even_source = str(rng.choice(["row_header", "column_header"]))
    odd_source = "column_header" if even_source == "row_header" else "row_header"
    grid = make_grid(n_rows, n_cols, fill=blank)
    grid[1:, 0] = row_colors
    grid[0, 1:] = col_colors
    out = grid.copy()
    for r in range(1, n_rows):
        for c in range(1, n_cols):
            source = even_source if (r + c) % 2 == 0 else odd_source
            out[r, c] = grid[r, 0] if source == "row_header" else grid[0, c]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "blank_color": blank,
            "row_header_colors": [int(c) for c in row_colors],
            "column_header_colors": [int(c) for c in col_colors],
            "combine_rule": "row_column_parity",
            "even_parity_source": even_source,
            "odd_parity_source": odd_source,
        },
    }


@register("120")
def generate_compare_two_grids_cellwise(rng: np.random.Generator) -> dict:
    bg, match_color, diff_color = unused_colors(rng, 3)
    base_colors = unused_colors(rng, 4, match_color, diff_color)
    # Two same-shape panels compared cell-by-cell; sample height and width
    # independently across the ARC-small-to-mid range (two grids per example
    # keep the token budget in check at the top of the range).
    h = int(rng.integers(5, 16))
    w = int(rng.integers(5, 16))
    left = np.array(rng.choice(base_colors, size=(h, w), replace=True), dtype=np.uint8)
    right = left.copy()
    n_diff = int(rng.integers(max(2, h * w // 6), max(3, h * w // 3)))
    positions = rng.choice(h * w, size=n_diff, replace=False)
    for flat in positions:
        r, c = divmod(int(flat), w)
        choices = [color for color in base_colors if color != int(left[r, c])]
        right[r, c] = int(choices[int(rng.integers(0, len(choices)))])
    out = make_grid(h, w, fill=bg)
    out[left == right] = match_color
    out[left != right] = diff_color
    return {
        "input": {"left": grid_to_list(left), "right": grid_to_list(right)},
        "output": grid_to_list(out),
        "metadata": {
            "match_color": match_color,
            "difference_color": diff_color,
            "comparison_rule": "same_position_equal_values",
        },
    }


def _oriented_offsets(cells: set[Cell], turns: int) -> set[Cell]:
    pts = set(cells)
    for _ in range(turns % 4):
        pts = {(c, -r) for r, c in pts}
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return {(r - min_r, c - min_c) for r, c in pts}


def _is_connected(cells: set[Cell]) -> bool:
    if not cells:
        return False
    todo = [next(iter(cells))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return seen == cells


# (base_shape, decoration_subset) pairs. Each base shape has four distinct
# rotational footprints, so a placed object's orientation is uniquely
# recoverable from its cells; each decoration is a subset of the base shape and
# stays a subset under every rotation, so it always lands on object cells.
_DECORATED_SHAPE_BANK: list[tuple[frozenset[Cell], frozenset[Cell]]] = [
    (frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}), frozenset({(0, 0), (2, 2)})),
    (frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}), frozenset({(0, 0), (0, 2)})),
    (frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}), frozenset({(0, 1), (2, 0)})),
    (frozenset({(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)}), frozenset({(1, 0), (3, 1)})),
    (frozenset({(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}), frozenset({(0, 2), (2, 0)})),
]


def _oriented_shape_and_decoration(
    turns: int,
    base: frozenset[Cell] = frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    decoration: frozenset[Cell] = frozenset({(0, 0), (2, 2)}),
) -> tuple[set[Cell], set[Cell]]:
    raw_shape = set(base)
    raw_decoration = set(decoration)
    for _ in range(turns % 4):
        raw_shape = {(c, -r) for r, c in raw_shape}
        raw_decoration = {(c, -r) for r, c in raw_decoration}
    min_r = min(r for r, _ in raw_shape)
    min_c = min(c for _, c in raw_shape)
    shape = {(r - min_r, c - min_c) for r, c in raw_shape}
    deco = {(r - min_r, c - min_c) for r, c in raw_decoration}
    return shape, deco


@register("127")
def generate_copy_decoration_to_oriented_objects(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg, object_color, decoration_color = unused_colors(rng, 3)
        base_shape, decoration = _DECORATED_SHAPE_BANK[
            int(rng.integers(0, len(_DECORATED_SHAPE_BANK)))
        ]
        # Sample height and width independently across a wide range; objects are
        # placed with retries, so smaller grids simply hold fewer of them.
        h = int(rng.integers(9, 22))
        w = int(rng.integers(9, 22))
        grid = make_grid(h, w, fill=bg)
        # Scale the object count to the available area so small grids stay
        # placeable while large grids carry more decorated/plain objects.
        max_objects = min(6, max(3, (h * w) // 18))
        n_objects = int(rng.integers(3, max_objects + 1))
        # The decorated example plus the plain targets all share this base shape.
        orientations = [int(rng.integers(0, 4)) for _ in range(n_objects)]
        placed_boxes: list[tuple[int, int, int, int]] = []
        objects: list[dict] = []
        placement_cells: list[tuple[set[Cell], set[Cell], int]] = []
        ok = True
        for turns in orientations:
            shape, deco = _oriented_shape_and_decoration(turns, base_shape, decoration)
            cells = place_shape(grid, rng, frozenset(shape), object_color, bg, placed_boxes, gap=1)
            if cells is None:
                ok = False
                break
            min_r = min(r for r, _ in cells)
            min_c = min(c for _, c in cells)
            anchor = (min_r, min_c)
            deco_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in deco}
            placement_cells.append((cells, deco_cells, turns))
            objects.append({
                "anchor": [int(anchor[0]), int(anchor[1])],
                "rotation_degrees_clockwise": int(turns * 90),
            })
        if not ok:
            continue
        out = grid.copy()
        decorated_idx = int(rng.integers(0, len(placement_cells)))
        for idx, (cells, deco_cells, _turns) in enumerate(placement_cells):
            if idx == decorated_idx:
                for r, c in deco_cells:
                    grid[r, c] = decoration_color
                    out[r, c] = decoration_color
                objects[idx]["role"] = "decorated_example"
            else:
                for r, c in deco_cells:
                    out[r, c] = decoration_color
                objects[idx]["role"] = "plain_target"
        if not any(obj["role"] == "plain_target" for obj in objects):
            continue
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "object_color": object_color,
                "decoration_color": decoration_color,
                "base_shape_offsets": sorted([list(p) for p in base_shape]),
                "decoration_offsets": sorted([list(p) for p in decoration]),
                "orientation_rule": "rotate_decoration_with_matching_object_orientation",
                "objects": objects,
            },
        }
    raise RuntimeError("Task 127: failed to place oriented objects")


@register("130")
def generate_shape_edit_mask(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg, source_color, target_color, divider_color, add_color, remove_color = unused_colors(rng, 6)
        # Sample panel height and width independently across the ARC-small range
        # so the two panels (and the h x w mask output) are not locked to a fixed
        # square. The full input grid is h x (2w + 1).
        h = int(rng.integers(5, 14))
        w = int(rng.integers(5, 14))
        # Sample a varied connected source shape and place it with a one-cell
        # margin inside the panel so removal/addition edits stay in bounds.
        n_cells = int(rng.integers(5, max(6, (h - 2) * (w - 2) // 2 + 1)))
        base = random_polyomino(rng, n_cells, connectivity=4)
        sr0, sc0, sr1, sc1 = bounding_box(base)
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        if sh > h - 2 or sw > w - 2:
            continue
        off_r = int(rng.integers(1, h - sh)) if h - sh - 1 >= 1 else 1
        off_c = int(rng.integers(1, w - sw)) if w - sw - 1 >= 1 else 1
        source = {(r - sr0 + off_r, c - sc0 + off_c) for r, c in base}
        target = set(source)
        remove_count = int(rng.integers(1, 3))
        for _remove in range(remove_count):
            boundary = [
                cell for cell in target
                if any((cell[0] + dr, cell[1] + dc) not in target for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
            ]
            rng.shuffle(boundary)
            for cell in boundary:
                candidate = set(target)
                candidate.remove(cell)
                if _is_connected(candidate):
                    target = candidate
                    break
        add_count = int(rng.integers(2, 4))
        for _add in range(add_count):
            candidates = sorted({
                (r + dr, c + dc)
                for r, c in target
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in target
            })
            if not candidates:
                break
            cell = candidates[int(rng.integers(0, len(candidates)))]
            target.add(cell)
        if not _is_connected(source) or not _is_connected(target):
            continue
        add = target - source
        remove = source - target
        if add and remove:
            break
    else:
        raise RuntimeError("Task 130: failed to generate connected edit pair")
    grid = make_grid(h, 2 * w + 1, fill=bg)
    grid[:, w] = divider_color
    for r, c in source:
        grid[r, c] = source_color
    for r, c in target:
        grid[r, w + 1 + c] = target_color
    out = make_grid(h, w, fill=bg)
    for r, c in add:
        out[r, c] = add_color
    for r, c in remove:
        out[r, c] = remove_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "source_color": source_color,
            "target_color": target_color,
            "divider_color": divider_color,
            "add_color": add_color,
            "remove_color": remove_color,
            "panel_width": w,
            "mask_rule": "add_cells_where_target_has_shape_and_source_is_blank_remove_cells_where_source_has_shape_and_target_is_blank",
        },
    }


@register("281")
def generate_identify_framed_subgrid_symmetry(rng: np.random.Generator) -> dict:
    labels = [
        "horizontal_reflection",
        "vertical_reflection",
        "main_diagonal_reflection",
        "anti_diagonal_reflection",
        "rotational_180",
    ]
    mode = str(
        rng.choice(
            [
                "horizontal",
                "vertical",
                "main_diagonal",
                "anti_diagonal",
                "rotational_180",
                "hv_combo",
                "all_reflections",
                "random",
            ]
        )
    )
    for _ in range(MAX_RETRIES):
        grid_h = int(rng.integers(5, 31))
        grid_w = int(rng.integers(5, 31))
        if min(grid_h, grid_w) < 5:
            continue
        frame_color = unused_colors(rng, 1)[0]
        color_pool = unused_colors(rng, 5, frame_color)
        outside_count = int(rng.integers(1, 4))
        outside_colors = color_pool[:outside_count]
        inner_colors = color_pool[outside_count:]
        if len(inner_colors) < 2:
            inner_colors = color_pool[-2:]

        square_only = mode in {"main_diagonal", "anti_diagonal", "all_reflections"}
        if square_only:
            max_side = min(grid_h - 2, grid_w - 2)
            if max_side < 3:
                continue
            inner_h = int(rng.integers(3, max_side + 1))
            inner_w = inner_h
        else:
            max_inner_h = grid_h - 2
            max_inner_w = grid_w - 2
            if min(max_inner_h, max_inner_w) < 3:
                continue
            inner_h = int(rng.integers(3, max_inner_h + 1))
            inner_w = int(rng.integers(3, max_inner_w + 1))

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        top = int(rng.integers(0, grid_h - frame_h + 1))
        left = int(rng.integers(0, grid_w - frame_w + 1))
        bottom = top + frame_h - 1
        right = left + frame_w - 1

        grid, outside_mode = _patterned_fill(rng, (grid_h, grid_w), outside_colors)
        if mode == "horizontal":
            inner = _fill_orbits(rng, (inner_h, inner_w), _mirror_cell, "horizontal", inner_colors)
        elif mode == "vertical":
            inner = _fill_orbits(rng, (inner_h, inner_w), _mirror_cell, "vertical", inner_colors)
        elif mode == "main_diagonal":
            inner = _fill_orbits(rng, (inner_h, inner_w), _mirror_cell, "main_diagonal", inner_colors)
        elif mode == "anti_diagonal":
            inner = _fill_orbits(rng, (inner_h, inner_w), _mirror_cell, "anti_diagonal", inner_colors)
        elif mode == "rotational_180":
            inner = _fill_orbits(rng, (inner_h, inner_w), _rotate_cell, 180, inner_colors)
        elif mode == "hv_combo":
            inner = _fill_horizontal_vertical_orbits(rng, (inner_h, inner_w), inner_colors)
        elif mode == "all_reflections":
            inner = _fill_full_reflection_orbits(rng, inner_h, inner_colors)
        else:
            inner = np.array(rng.choice(inner_colors, size=(inner_h, inner_w), replace=True), dtype=np.uint8)

        grid[top, left : right + 1] = frame_color
        grid[bottom, left : right + 1] = frame_color
        grid[top : bottom + 1, left] = frame_color
        grid[top : bottom + 1, right] = frame_color
        grid[top + 1 : bottom, left + 1 : right] = inner
        output = "\n".join(
            f"{label}: {'yes' if _is_symmetric(inner, label) else 'no'}"
            for label in labels
        )
        return {
            "input": grid_to_list(grid),
            "output": output,
            "metadata": {
                "frame_color": frame_color,
                "frame_bbox": [top, left, bottom, right],
                "interior_height": inner_h,
                "interior_width": inner_w,
                "outside_fill_mode": outside_mode,
                "inner_generation_mode": mode,
                "allowed_symmetries": labels,
                "selection_rule": "report_each_allowed_symmetry_for_the_framed_interior_only",
                "output_format_rule": "one_label_per_line_in_allowed_symmetries_order_with_yes_no_values",
            },
        }
    raise RuntimeError("Task 281: failed to generate example")
