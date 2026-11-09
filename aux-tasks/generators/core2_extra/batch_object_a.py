"""Object-focused second-batch task generators, slice A.

Tasks 97 and 98 moved out on 2026-09-15 (``generators/odd_shape_97.py`` and
``generators/sorted_strip_98.py``); everything the eight tasks that stayed rely
on -- the shape pools, the size sampler, the placement helpers and the
measures -- is unchanged, and their examples are byte-identical.
"""
from __future__ import annotations

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid, pick_colors
from common.objects import bounding_box, connected_components, normalize_shape
from generators.core2_extra.utils import (
    MAX_RETRIES,
    boxes_overlap,
    place_shape,
)
from registry import register


Shape = frozenset[tuple[int, int]]
Cell = tuple[int, int]


SHAPES: list[Shape] = [
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)}),
]


EIGHT_CONNECTED_SHAPES: list[Shape] = [
    frozenset({(0, 0), (1, 1), (2, 2)}),
    frozenset({(0, 2), (1, 1), (2, 0)}),
    frozenset({(0, 0), (1, 1), (2, 0)}),
    frozenset({(0, 1), (1, 0), (1, 2), (2, 1)}),
    frozenset({(0, 0), (1, 1), (1, 2), (2, 3)}),
]


def _unused_colors(rng: np.random.Generator, n: int, *exclude: int) -> list[int]:
    return [int(c) for c in pick_colors(rng, n, exclude=[int(c) for c in exclude])]


def _sample_grid_size(
    rng: np.random.Generator,
    n_objects: int,
    *,
    extra: int = 6,
    max_side: int = 22,
    square_prob: float = 0.35,
) -> tuple[int, int]:
    """Sample (H, W) for an object scene.

    ARC grids are dominated by small sizes (median side ~10) spanning 1-30, so we
    shift the center down and broaden rather than fixing a mid-teens value. H and W
    are sampled independently to decorrelate them, while still emitting some square
    grids (ARC is ~62% square). The lower bound scales with ``n_objects`` so that the
    chosen size can actually hold that many gap-separated objects plus distractors;
    the upper bound stays inside the grid->grid token budget (a 22x22 spaced bundle
    is well under the 2900-token cap).
    """
    lo = extra + int(n_objects)
    hi = max_side
    if lo > hi:
        lo = hi
    if rng.random() < square_prob:
        side = int(rng.integers(lo, hi + 1))
        return side, side
    h = int(rng.integers(lo, hi + 1))
    w = int(rng.integers(lo, hi + 1))
    return h, w


def _components(
    grid: np.ndarray,
    bg: int,
    color: int | None = None,
    connectivity: int = 4,
) -> list[dict]:
    comps = connected_components(grid, connectivity=connectivity, background=bg)
    if color is not None:
        comps = [comp for comp in comps if comp["color"] == color]
    return comps


def _shape_key(cells: set[Cell]) -> tuple[Cell, ...]:
    return tuple(sorted(normalize_shape(cells)))


def _has_multiple_4_components(cells: set[Cell]) -> bool:
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
        if count > 1:
            return True
    return False


def _equivalence_key(comp: dict, rule: str) -> tuple[Cell, ...] | int:
    if rule == "same_normalized_shape":
        return _shape_key(comp["cells"])
    if rule == "same_cell_count":
        return len(comp["cells"])
    raise ValueError(f"unknown equivalence rule: {rule}")


def _bbox_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _measure(comp: dict, measure: str) -> int:
    if measure == "cell_count":
        return len(comp["cells"])
    if measure == "bounding_box_area":
        h, w = _bbox_size(comp["bbox"])
        return h * w
    if measure == "bbox_height":
        return _bbox_size(comp["bbox"])[0]
    if measure == "bbox_width":
        return _bbox_size(comp["bbox"])[1]
    raise ValueError(f"unknown measure: {measure}")


def _landmark(comp: dict, name: str) -> Cell:
    r0, c0, r1, c1 = comp["bbox"]
    if name == "top_left":
        return (r0, c0)
    if name == "top_right":
        return (r0, c1)
    if name == "bottom_left":
        return (r1, c0)
    if name == "bottom_right":
        return (r1, c1)
    if name == "medoid":
        return _medoid(comp["cells"])
    raise ValueError(f"unknown landmark: {name}")


def _medoid(cells: set[Cell]) -> Cell:
    ordered = sorted(cells)
    return min(
        ordered,
        key=lambda p: (
            sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in ordered),
            p[0],
            p[1],
        ),
    )


def _distance_to_marker(comp: dict, marker: Cell, metric: str) -> int:
    if metric == "min_manhattan":
        return min(abs(r - marker[0]) + abs(c - marker[1]) for r, c in comp["cells"])
    if metric == "top_left_manhattan":
        r, c = _landmark(comp, "top_left")
        return abs(r - marker[0]) + abs(c - marker[1])
    raise ValueError(f"unknown distance metric: {metric}")


def _place_objects(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    bg: int,
    shapes: list[Shape],
    colors: list[int],
    gap: int = 2,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = make_grid(h, w, fill=bg)
    boxes: list[tuple[int, int, int, int]] = []
    objects: list[dict] = []
    for shape, color in zip(shapes, colors):
        cells = place_shape(grid, rng, shape, int(color), bg, boxes, gap=gap, max_attempts=300)
        if cells is None:
            return None
        objects.append({"color": int(color), "cells": cells, "bbox": bounding_box(cells)})
    return grid, objects


def _place_marker(
    grid: np.ndarray,
    rng: np.random.Generator,
    color: int,
    bg: int,
    boxes: list[tuple[int, int, int, int]],
    gap: int = 1,
) -> Cell | None:
    h, w = grid.shape
    for _ in range(300):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if grid[r, c] != bg:
            continue
        box = (r, c, r, c)
        if any(boxes_overlap(box, other, gap=gap) for other in boxes):
            continue
        grid[r, c] = color
        boxes.append(box)
        return (r, c)
    return None


def _object_boxes(objects: list[dict]) -> list[tuple[int, int, int, int]]:
    return [tuple(int(v) for v in obj["bbox"]) for obj in objects]


def _add_distractor_objects(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    exclude_colors: list[int] | tuple[int, ...],
    boxes: list[tuple[int, int, int, int]],
    min_count: int = 1,
    max_count: int = 3,
) -> list[int]:
    n = int(rng.integers(min_count, max_count + 1))
    colors = _unused_colors(rng, n, bg, *exclude_colors)
    placed_colors: list[int] = []
    for color in colors:
        candidates = SHAPES.copy()
        rng.shuffle(candidates)
        for shape in candidates:
            cells = place_shape(grid, rng, shape, int(color), bg, boxes, gap=1, max_attempts=120)
            if cells is not None:
                placed_colors.append(int(color))
                break
    return placed_colors


@register("105")
def generate_recolor_by_nearest_marker(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        target_color = _unused_colors(rng, 1, bg)[0]
        n_objects = int(rng.integers(3, 5))
        n_markers = int(rng.integers(2, 4))
        marker_colors = _unused_colors(rng, n_markers, bg, target_color)
        shapes = [SHAPES[int(i)] for i in rng.choice(len(SHAPES), size=n_objects, replace=False)]
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(
            rng, h=h, w=w, bg=bg, shapes=shapes, colors=[target_color] * n_objects, gap=2
        )
        if scene is None:
            continue
        grid, objects = scene
        boxes = _object_boxes(objects)
        markers: list[dict] = []
        for color in marker_colors:
            pos = _place_marker(grid, rng, color, bg, boxes, gap=1)
            if pos is None:
                break
            markers.append({"color": int(color), "position": pos})
        if len(markers) != n_markers:
            continue
        distractor_colors = _add_distractor_objects(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, *marker_colors],
            boxes=boxes,
        )
        if not distractor_colors:
            continue
        metric = str(rng.choice(["min_manhattan", "top_left_manhattan"]))
        comps = _components(grid, bg, target_color)
        alternate_metric = "top_left_manhattan" if metric == "min_manhattan" else "min_manhattan"
        if not any(
            min(
                markers,
                key=lambda m: (
                    _distance_to_marker(comp, m["position"], metric),
                    m["position"][0],
                    m["position"][1],
                ),
            )["color"]
            != min(
                markers,
                key=lambda m: (
                    _distance_to_marker(comp, m["position"], alternate_metric),
                    m["position"][0],
                    m["position"][1],
                ),
            )["color"]
            for comp in comps
        ):
            continue
        out = grid.copy()
        for comp in comps:
            marker = min(
                markers,
                key=lambda m: (
                    _distance_to_marker(comp, m["position"], metric),
                    m["position"][0],
                    m["position"][1],
                ),
            )
            for r, c in comp["cells"]:
                out[r, c] = marker["color"]
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target_color,
                "marker_colors": marker_colors,
                "distractor_colors": distractor_colors,
                "connectivity": 4,
                "distance_metric": metric,
                "tie_rule": "choose the tied marker with row-major marker position",
            },
        }
    raise RuntimeError("Task 105: failed to generate valid example")


@register("106")
def generate_rank_objects_recolor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        target_color = _unused_colors(rng, 1, bg)[0]
        n_objects = int(rng.integers(3, 6))
        output_colors = _unused_colors(rng, n_objects, bg, target_color)
        size_measure = str(rng.choice(["cell_count", "bounding_box_area"]))
        rank_order = str(rng.choice(["smallest_to_largest", "largest_to_smallest"]))
        candidates = SHAPES.copy()
        rng.shuffle(candidates)
        shapes: list[Shape] = []
        used_values: set[int] = set()
        for shape in candidates:
            value = _measure({"cells": set(shape), "bbox": bounding_box(shape)}, size_measure)
            if value in used_values:
                continue
            shapes.append(shape)
            used_values.add(value)
            if len(shapes) == n_objects:
                break
        if len(shapes) != n_objects:
            continue
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(
            rng, h=h, w=w, bg=bg, shapes=shapes, colors=[target_color] * n_objects, gap=2
        )
        if scene is None:
            continue
        grid, objects = scene
        boxes = _object_boxes(objects)
        distractor_colors = _add_distractor_objects(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, *output_colors],
            boxes=boxes,
        )
        if not distractor_colors:
            continue
        comps = _components(grid, bg, target_color)
        if len(comps) != n_objects:
            continue
        by_cell_count = [
            comp["bbox"]
            for comp in sorted(comps, key=lambda comp: (_measure(comp, "cell_count"), comp["bbox"][0], comp["bbox"][1]))
        ]
        by_bbox_area = [
            comp["bbox"]
            for comp in sorted(
                comps,
                key=lambda comp: (_measure(comp, "bounding_box_area"), comp["bbox"][0], comp["bbox"][1]),
            )
        ]
        if by_cell_count == by_bbox_area:
            continue
        reverse = rank_order == "largest_to_smallest"
        ordered = sorted(
            comps,
            key=lambda comp: (_measure(comp, size_measure), comp["bbox"][0], comp["bbox"][1]),
            reverse=reverse,
        )
        out = grid.copy()
        mapping = []
        for rank, (comp, color) in enumerate(zip(ordered, output_colors), start=1):
            mapping.append({"rank": rank, "color": int(color)})
            for r, c in comp["cells"]:
                out[r, c] = color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target_color,
                "distractor_colors": distractor_colors,
                "connectivity": 4,
                "size_measure": size_measure,
                "rank_order": rank_order,
                "rank_to_color_mapping": mapping,
                "tie_rule": "sizes are unique",
            },
        }
    raise RuntimeError("Task 106: failed to generate valid example")


@register("108")
def generate_frequency_rank_color_remap(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        n_colors = int(rng.integers(3, 6))
        input_colors = _unused_colors(rng, n_colors, bg)
        # Independent H/W shifted toward the ARC-small range and broadened. The floor
        # grows with the color count so the grid can hold one distinct-count block per
        # color; over-tight draws are rejected by the count guard below and retried.
        floor = max(6, n_colors + 3)
        h = int(rng.integers(floor, 23))
        w = int(rng.integers(floor, 23))
        grid = make_grid(h, w, fill=bg)
        max_cells = h * w - 8
        base_counts = list(range(5, 5 + 3 * n_colors, 3))
        rng.shuffle(base_counts)
        counts = [min(c, max_cells // n_colors + 8) for c in base_counts[:n_colors]]
        if len(set(counts)) != len(counts) or sum(counts) >= h * w:
            continue
        positions = [(r, c) for r in range(h) for c in range(w)]
        chosen = rng.choice(len(positions), size=sum(counts), replace=False)
        offset = 0
        for color, count in zip(input_colors, counts):
            for idx in chosen[offset : offset + count]:
                r, c = positions[int(idx)]
                grid[r, c] = color
            offset += count
        frequencies = {color: int(np.sum(grid == color)) for color in input_colors}
        if len(set(frequencies.values())) != len(frequencies):
            continue
        ordered_colors = sorted(input_colors, key=lambda color: -frequencies[color])
        for _attempt in range(50):
            output_colors = _unused_colors(rng, n_colors, bg)
            if all(int(src) != int(dst) for src, dst in zip(ordered_colors, output_colors)):
                break
        else:
            continue
        mapping = [
            {"rank": rank, "color": int(color)}
            for rank, color in enumerate(output_colors, start=1)
        ]
        rank_color = {color: output_colors[i] for i, color in enumerate(ordered_colors)}
        out = grid.copy()
        for color, mapped in rank_color.items():
            out[grid == color] = mapped
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "ranking": "most_frequent_to_least_frequent",
                "tie_rule": "non-background color frequencies are unique; no tie exists",
                "source_color_frequencies": [
                    {"color": int(color), "count": int(frequencies[color])}
                    for color in sorted(input_colors)
                ],
                "rank_to_color_mapping": mapping,
            },
        }
    raise RuntimeError("Task 108: failed to generate valid example")


@register("143")
def generate_order_index_recolor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        target_color = _unused_colors(rng, 1, bg)[0]
        n_objects = int(rng.integers(3, 6))
        output_colors = _unused_colors(rng, n_objects, bg, target_color)
        shapes = [SHAPES[int(i)] for i in rng.choice(len(SHAPES), size=n_objects, replace=False)]
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(
            rng, h=h, w=w, bg=bg, shapes=shapes, colors=[target_color] * n_objects, gap=2
        )
        if scene is None:
            continue
        grid, objects = scene
        boxes = _object_boxes(objects)
        distractor_colors = _add_distractor_objects(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, *output_colors],
            boxes=boxes,
        )
        if not distractor_colors:
            continue
        ordering = str(
            rng.choice([
                "top_to_bottom_left_to_right",
                "left_to_right_top_to_bottom",
                "bottom_to_top_left_to_right",
            ])
        )
        comps = _components(grid, bg, target_color)
        if ordering == "top_to_bottom_left_to_right":
            key = lambda comp: (comp["bbox"][0], comp["bbox"][1])
        elif ordering == "left_to_right_top_to_bottom":
            key = lambda comp: (comp["bbox"][1], comp["bbox"][0])
        else:
            key = lambda comp: (-comp["bbox"][2], comp["bbox"][1])
        ordered = sorted(comps, key=key)
        out = grid.copy()
        mapping = []
        for index, (comp, color) in enumerate(zip(ordered, output_colors), start=1):
            mapping.append({"index": index, "color": int(color)})
            for r, c in comp["cells"]:
                out[r, c] = color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target_color,
                "distractor_colors": distractor_colors,
                "connectivity": 4,
                "object_ordering": ordering,
                "index_to_color_mapping": mapping,
            },
        }
    raise RuntimeError("Task 143: failed to generate valid example")


@register("144")
def generate_mark_object_medoids(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, marker_color = _unused_colors(rng, 2, bg)
        n_objects = int(rng.integers(3, 6))
        shapes = [SHAPES[int(i)] for i in rng.choice(len(SHAPES), size=n_objects, replace=False)]
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(
            rng, h=h, w=w, bg=bg, shapes=shapes, colors=[target_color] * n_objects, gap=2
        )
        if scene is None:
            continue
        grid, objects = scene
        boxes = _object_boxes(objects)
        distractor_colors = _add_distractor_objects(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, marker_color],
            boxes=boxes,
        )
        if not distractor_colors:
            continue
        comps = _components(grid, bg, target_color)
        out = grid.copy()
        for comp in comps:
            out[_medoid(comp["cells"])] = marker_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target_color,
                "marker_color": marker_color,
                "distractor_colors": distractor_colors,
                "connectivity": 4,
                "tie_rule": "if several cells minimize total distance, choose the row-major first one",
            },
        }
    raise RuntimeError("Task 144: failed to generate valid example")


@register("152")
def generate_keep_object_associated_with_marker(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, marker_color = _unused_colors(rng, 2, bg)
        n_objects = int(rng.integers(3, 6))
        shapes = [SHAPES[int(i)] for i in rng.choice(len(SHAPES), size=n_objects, replace=False)]
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(
            rng, h=h, w=w, bg=bg, shapes=shapes, colors=[target_color] * n_objects, gap=2
        )
        if scene is None:
            continue
        grid, objects = scene
        boxes = _object_boxes(objects)
        marker_pos = _place_marker(grid, rng, marker_color, bg, boxes, gap=1)
        if marker_pos is None:
            continue
        distractor_colors = _add_distractor_objects(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, marker_color],
            boxes=boxes,
        )
        if not distractor_colors:
            continue
        association_rule = str(rng.choice(["nearest_min_manhattan", "nearest_top_left_manhattan"]))
        metric = "min_manhattan" if association_rule == "nearest_min_manhattan" else "top_left_manhattan"
        comps = _components(grid, bg, target_color)
        keep = min(
            comps,
            key=lambda comp: (
                _distance_to_marker(comp, marker_pos, metric),
                comp["bbox"][0],
                comp["bbox"][1],
            ),
        )
        out = grid.copy()
        for comp in comps:
            if comp is keep:
                continue
            for r, c in comp["cells"]:
                out[r, c] = bg
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target_color,
                "marker_color": marker_color,
                "distractor_colors": distractor_colors,
                "connectivity": 4,
                "association_rule": association_rule,
                "tie_rule": "choose the tied object with row-major top-left bounding-box corner",
            },
        }
    raise RuntimeError("Task 152: failed to generate valid example")


@register("161")
def generate_replace_objects_with_landmark_cell(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        n_objects = int(rng.integers(3, 6))
        object_colors = _unused_colors(rng, n_objects, bg)
        marker_color = _unused_colors(rng, 1, bg, *object_colors)[0]
        shapes = [SHAPES[int(i)] for i in rng.choice(len(SHAPES), size=n_objects, replace=False)]
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(rng, h=h, w=w, bg=bg, shapes=shapes, colors=object_colors, gap=2)
        if scene is None:
            continue
        grid, _objects = scene
        comps = _components(grid, bg)
        landmark = str(rng.choice(["top_left", "bottom_right", "medoid"]))
        color_rule = str(rng.choice(["original_color", "marker_color"]))
        out = make_grid(*grid.shape, fill=bg)
        for comp in comps:
            color = comp["color"] if color_rule == "original_color" else marker_color
            out[_landmark(comp, landmark)] = color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "connectivity": 4,
                "target_rule": "all non-background objects",
                "landmark": landmark,
                "color_rule": color_rule,
                "marker_color": marker_color,
                "tie_rule": "medoid ties use row-major order",
            },
        }
    raise RuntimeError("Task 161: failed to generate valid example")


@register("172")
def generate_replace_objects_with_count_bars(rng: np.random.Generator) -> dict:
    dirs = {
        "right": (0, 1),
        "left": (0, -1),
        "down": (1, 0),
        "up": (-1, 0),
    }
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        n_objects = int(rng.integers(3, 6))
        object_colors = _unused_colors(rng, n_objects, bg)
        shapes = [SHAPES[int(i)] for i in rng.choice(len(SHAPES), size=n_objects, replace=False)]
        h, w = _sample_grid_size(rng, n_objects)
        scene = _place_objects(rng, h=h, w=w, bg=bg, shapes=shapes, colors=object_colors, gap=3)
        if scene is None:
            continue
        grid, _objects = scene
        direction = str(rng.choice(list(dirs)))
        landmark_choices = {
            "right": ["top_left", "bottom_left"],
            "left": ["top_right", "bottom_right"],
            "down": ["top_left", "top_right"],
            "up": ["bottom_left", "bottom_right"],
        }[direction]
        landmark = str(rng.choice(landmark_choices))
        comps = _components(grid, bg)
        out = make_grid(*grid.shape, fill=bg)
        used: set[Cell] = set()
        dr, dc = dirs[direction]
        ok = True
        bars: list[tuple[dict, list[Cell]]] = []
        for comp in comps:
            start = _landmark(comp, landmark)
            cells = [(start[0] + dr * i, start[1] + dc * i) for i in range(len(comp["cells"]))]
            if any(r < 0 or r >= out.shape[0] or c < 0 or c >= out.shape[1] for r, c in cells):
                ok = False
                break
            if any(cell in used for cell in cells):
                ok = False
                break
            used.update(cells)
            bars.append((comp, cells))
        if not ok:
            continue
        for comp, cells in bars:
            for r, c in cells:
                out[r, c] = comp["color"]
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "connectivity": 4,
                "target_rule": "all non-background objects",
                "landmark": landmark,
                "direction": direction,
                "length_rule": "object cell count",
                "color_rule": "original_color",
            },
        }
    raise RuntimeError("Task 172: failed to generate valid example")
