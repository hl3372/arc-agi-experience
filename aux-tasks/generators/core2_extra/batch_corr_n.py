"""Object-correlation and correspondence generators for core2 batch N."""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from generators.core2_extra.utils import (
    add_patterned_background,
    boxes_overlap,
    unused_colors,
)
from registry import register


Cell = tuple[int, int]
Shape = frozenset[Cell]


SHAPES: list[Shape] = [
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}),
]

# Elongated shapes for task 225 so an object's medoid (center) can point toward a
# different marker than its nearest edge cell, letting the association relation matter.
TASK_225_SHAPES: list[Shape] = [
    frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}),
    frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}),
    frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    frozenset({(0, 0), (1, 0), (2, 0), (3, 0)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
]

# Orientation-tagged shape banks for task 224 (bbox wider/taller/square). Each
# list holds normalized shapes whose bounding box is strictly wider, strictly
# taller, or square, so a scene can guarantee all three classes are present
# while still varying which specific shapes and how many appear.
TASK_224_WIDE: list[Shape] = [
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
]
TASK_224_TALL: list[Shape] = [
    frozenset({(0, 0), (1, 0), (2, 0)}),
    frozenset({(0, 0), (1, 0), (2, 0), (3, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 0)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
]
TASK_224_SQUARE: list[Shape] = [
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)}),
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)}),
    frozenset({(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)}),
]


def _normalize(cells: set[Cell] | Shape) -> Shape:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _bbox(cells: set[Cell] | Shape) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _abs(shape: Shape, anchor: Cell) -> set[Cell]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in shape}


def _stamp(grid: np.ndarray, shape: Shape, anchor: Cell, color: int) -> set[Cell]:
    cells = _abs(shape, anchor)
    for r, c in cells:
        grid[r, c] = color
    return cells


def _transform_shape(shape: Shape, name: str) -> Shape:
    if name == "identity":
        cells = set(shape)
    elif name == "rotate90":
        cells = {(c, -r) for r, c in shape}
    elif name == "rotate180":
        cells = {(-r, -c) for r, c in shape}
    elif name == "rotate270":
        cells = {(-c, r) for r, c in shape}
    elif name == "reflect_horizontal":
        cells = {(r, -c) for r, c in shape}
    elif name == "reflect_vertical":
        cells = {(-r, c) for r, c in shape}
    else:
        raise ValueError(f"unknown transform {name}")
    return _normalize(cells)


def _shape_key(shape: Shape) -> tuple[Cell, ...]:
    return tuple(sorted(_normalize(shape)))


def _canonical(shape: Shape, relation: str) -> tuple[Cell, ...]:
    if relation == "same_normalized_shape":
        return _shape_key(shape)
    if relation == "same_shape_under_rotation":
        names = ["identity", "rotate90", "rotate180", "rotate270"]
    elif relation == "same_shape_under_reflection":
        names = ["identity", "reflect_horizontal", "reflect_vertical", "rotate180"]
    else:
        raise ValueError(f"unknown relation {relation}")
    return min(tuple(sorted(_transform_shape(shape, name))) for name in names)


def _components(grid: np.ndarray, bg: int, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg or (colors is not None and color not in colors):
                seen[r, c] = True
                continue
            q: deque[Cell] = deque([(r, c)])
            seen[r, c] = True
            cells: set[Cell] = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3], comp["color"]))


def _records(comps: list[dict], prefix: str) -> list[dict]:
    records = []
    for i, comp in enumerate(comps, start=1):
        records.append({
            "label": f"{prefix}{i}",
            "color": int(comp["color"]),
            "bbox": [int(v) for v in comp["bbox"]],
            "cells": [[int(r), int(c)] for r, c in sorted(comp["cells"])],
            "normalized_shape": [[int(r), int(c)] for r, c in sorted(_normalize(comp["cells"]))],
        })
    return records


def _apply_transform(out: np.ndarray, cells: set[Cell], bg: int, transform: dict) -> None:
    if transform["type"] == "recolor":
        color = int(transform["color"])
    elif transform["type"] == "erase":
        color = int(bg)
    else:
        raise ValueError(f"unknown object transform {transform['type']}")
    for r, c in cells:
        out[r, c] = color


def _property_condition(comp: dict, condition: str, threshold: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if condition == "cell_count_at_least":
        return len(comp["cells"]) >= threshold
    if condition == "bbox_wider_than_tall":
        return width > height
    if condition == "cell_count_even":
        return len(comp["cells"]) % 2 == 0
    raise ValueError(f"unknown condition {condition}")


def _bbox_orientation(comp: dict) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width > height:
        return "wide"
    if height > width:
        return "tall"
    return "square"


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


def _distance_to_marker(comp: dict, marker: Cell, relation: str) -> int:
    if relation == "nearest_marker_by_min_manhattan":
        return min(abs(r - marker[0]) + abs(c - marker[1]) for r, c in comp["cells"])
    if relation == "nearest_marker_by_medoid":
        r, c = _medoid(comp["cells"])
        return abs(r - marker[0]) + abs(c - marker[1])
    raise ValueError(f"unknown marker relation {relation}")


def _region_for(comp: dict, split_row: int, split_col: int) -> str:
    r, c = _medoid(comp["cells"])
    if r < split_row and c < split_col:
        return "top_left"
    if r < split_row and c > split_col:
        return "top_right"
    if r > split_row and c < split_col:
        return "bottom_left"
    return "bottom_right"


def _scene(shapes: list[Shape], anchors: list[Cell], bg: int, colors: list[int], *, h: int = 18, w: int = 18) -> np.ndarray:
    grid = make_grid(h, w, fill=bg)
    for shape, anchor, color in zip(shapes, anchors, colors):
        _stamp(grid, shape, anchor, int(color))
    return grid


def _grid_arrangement(rng: np.random.Generator, n: int) -> tuple[int, int]:
    """Pick a rows x cols block arrangement (<=5 each) that can hold n blocks."""
    arrangements = []
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        if rows * cols >= n and rows <= 5 and cols <= 5:
            arrangements.append((rows, cols))
    rows, cols = arrangements[int(rng.integers(0, len(arrangements)))]
    return rows, cols


def _layout_anchors(
    rng: np.random.Generator,
    n: int,
    h: int,
    w: int,
    *,
    cell: int = 4,
) -> list[Cell] | None:
    """Lay out n anchors on a jittered block grid inside (h, w) so shapes up to
    (cell-1) on a side never overlap and always keep a >=1-cell gap (objects
    share a colour, so touching would merge their components). Every object gets
    a distinct top-left corner from the block columns/rows. Returns None if it
    will not fit."""
    rows, cols = _grid_arrangement(rng, n)
    if rows * cell + 1 > h or cols * cell + 1 > w:
        return None
    block_h = (h - 1) // rows
    block_w = (w - 1) // cols
    if block_h < cell or block_w < cell:
        return None
    slots = list(range(rows * cols))
    chosen = sorted(int(s) for s in rng.choice(np.array(slots), size=n, replace=False))
    anchors: list[Cell] = []
    for slot in chosen:
        br, bc = divmod(slot, cols)
        r0 = 1 + br * block_h
        c0 = 1 + bc * block_w
        jr = int(rng.integers(0, block_h - (cell - 1)))
        jc = int(rng.integers(0, block_w - (cell - 1)))
        anchors.append((r0 + jr, c0 + jc))
    return anchors


def _block_size(
    rng: np.random.Generator,
    n_max: int,
    *,
    cell: int = 4,
    h_min: int = 10,
    h_max: int = 21,
    w_min: int = 10,
    w_max: int = 21,
) -> tuple[int, int]:
    """Pick an independent (H, W) large enough to lay out up to n_max blocks in
    some arrangement, widened down toward the ARC small-grid mode and up to the
    two-grid token budget. Min/max span >8 so the size axis is genuinely wide."""
    # Need room for at least a ceil(sqrt) x ceil(sqrt) arrangement of n_max.
    import math
    side = math.ceil(math.sqrt(n_max))
    lo_h = max(h_min, side * cell + 1)
    lo_w = max(w_min, side * cell + 1)
    lo_h = min(lo_h, h_max)
    lo_w = min(lo_w, w_max)
    return _independent_hw(rng, lo_h, h_max, lo_w, w_max, square_prob=0.30)


def _pick_unique_shapes(
    rng: np.random.Generator,
    n: int,
    relation: str = "same_normalized_shape",
    exclude: set[tuple[Cell, ...]] | None = None,
    pool: list[Shape] | None = None,
) -> list[Shape]:
    excluded = set(exclude or set())
    shape_pool = pool if pool is not None else SHAPES
    indices = list(range(len(shape_pool)))
    rng.shuffle(indices)
    out: list[Shape] = []
    seen: set[tuple[Cell, ...]] = set()
    for idx in indices:
        shape = shape_pool[int(idx)]
        key = _canonical(shape, relation)
        if key in excluded or key in seen:
            continue
        seen.add(key)
        out.append(shape)
        if len(out) == n:
            return out
    raise RuntimeError("not enough unique shapes")


def _permutation(rng: np.random.Generator, n: int) -> list[int]:
    order = [int(i) for i in rng.permutation(n)]
    if order == list(range(n)):
        order = order[1:] + order[:1]
    return order


def _correspondence_from_grids(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    bg: int,
    object_color: int,
    relation: str,
) -> str:
    """Build the input->output correspondence string from the actual top-left
    component order of each grid (independent of placement order), matching by
    `relation`. This mirrors the verifier exactly, so the answer is the unique
    canonical string regardless of how the objects were jittered into place."""
    inputs = _components(input_grid, bg, {object_color})
    outputs = _components(output_grid, bg, {object_color})
    output_by_key = {_canonical(c["cells"], relation): idx for idx, c in enumerate(outputs, start=1)}
    parts = [
        f"input{idx}->output{output_by_key[_canonical(c['cells'], relation)]}"
        for idx, c in enumerate(inputs, start=1)
    ]
    return ";".join(parts)


def _status_summary_from_grids(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    bg: int,
    object_color: int,
) -> str:
    """Build the matched/deleted/copied/new classification string from the
    top-left component order of each grid (mirrors the verifier)."""
    inputs = _components(input_grid, bg, {object_color})
    outputs = _components(output_grid, bg, {object_color})
    input_keys = [_canonical(c["cells"], "same_normalized_shape") for c in inputs]
    output_keys = [_canonical(c["cells"], "same_normalized_shape") for c in outputs]
    matched: list[str] = []
    deleted: list[str] = []
    copied: list[str] = []
    for input_idx, key in enumerate(input_keys, start=1):
        labels = [f"output{j}" for j, ok in enumerate(output_keys, start=1) if ok == key]
        if not labels:
            deleted.append(f"input{input_idx}")
        elif len(labels) == 1:
            matched.append(f"input{input_idx}->{labels[0]}")
        else:
            copied.extend(f"input{input_idx}->{label}" for label in labels)
    new = [
        f"output{idx}"
        for idx, key in enumerate(output_keys, start=1)
        if key not in set(input_keys)
    ]
    return (
        f"matched={','.join(matched)};"
        f"deleted={','.join(deleted)};"
        f"copied={','.join(copied)};"
        f"new={','.join(new)}"
    )


def _pair_metadata(
    *,
    bg: int,
    object_color: int,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    relation: str,
    extra: dict | None = None,
) -> dict:
    meta = {
        "background_color": int(bg),
        "object_color": int(object_color),
        "label_order_rule": "top_left_bounding_box_order",
        "matching_rule": relation,
        "input_objects": _records(_components(input_grid, bg, {object_color}), "input"),
        "output_objects": _records(_components(output_grid, bg, {object_color}), "output"),
    }
    if extra:
        meta.update(extra)
    return meta


def _distinct_top_left_order(comps: list[dict]) -> bool:
    """True when the objects have a strict total order under (top, then left)."""
    corners = [(comp["bbox"][0], comp["bbox"][1]) for comp in comps]
    return len(set(corners)) == len(corners)


# ---------------------------------------------------------------------------
# Grid-size + element-count diversity helpers (second review)
#
# ARC is dominated by small grids spanning 1..30 (p25=6, median=10). The
# original generators pinned every scene to a single fixed H,W and a fixed
# object count on hard-coded anchor bands. These helpers sample H and W
# independently from a per-task envelope, place a *variable* number of shapes
# at random non-overlapping positions (1-cell gap so 4-connected components
# stay distinct), and guarantee a strict top-left labelling order. The answer
# rule is unchanged; only the layout/size/count distribution widens.
# ---------------------------------------------------------------------------


def _try_place_shapes(
    rng: np.random.Generator,
    grid: np.ndarray,
    shapes: list[Shape],
    color: int,
    bg: int,
    *,
    gap: int = 1,
    max_attempts: int = 60,
) -> list[set[Cell]] | None:
    """Place each shape (in order) at a random spot, keeping a `gap`-cell margin
    between bounding boxes so components stay 4-connected-separate. Returns the
    placed cell-sets in input order, or None on failure."""
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_cells: list[set[Cell]] = []
    for shape in shapes:
        r0, c0, r1, c1 = _bbox(shape)
        sh = r1 - r0 + 1
        sw = c1 - c0 + 1
        if sh > h or sw > w:
            return None
        ok = False
        for _ in range(max_attempts):
            tr = int(rng.integers(0, h - sh + 1))
            tc = int(rng.integers(0, w - sw + 1))
            box = (tr, tc, tr + sh - 1, tc + sw - 1)
            if any(boxes_overlap(box, other, gap=gap) for other in placed_boxes):
                continue
            cells = {(tr + r - r0, tc + c - c0) for r, c in shape}
            if any(grid[rr, cc] != bg for rr, cc in cells):
                continue
            for rr, cc in cells:
                grid[rr, cc] = color
            placed_boxes.append(box)
            placed_cells.append(cells)
            ok = True
            break
        if not ok:
            return None
    return placed_cells


def _independent_hw(
    rng: np.random.Generator,
    h_min: int,
    h_max: int,
    w_min: int,
    w_max: int,
    *,
    square_prob: float = 0.30,
) -> tuple[int, int]:
    """Sample H and W independently from [min,max]; with `square_prob` force a
    square grid (ARC is ~62% square, so keep some, but stop pinning H==W)."""
    h = int(rng.integers(h_min, h_max + 1))
    w = int(rng.integers(w_min, w_max + 1))
    if rng.random() < square_prob:
        lo = max(h_min, w_min)
        hi = min(h_max, w_max)
        if hi >= lo:
            s = int(rng.integers(lo, hi + 1))
            return s, s
    return h, w


@register("193")
def generate_count_indexed_target_action(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    condition_color, target_color, action_color, distractor_color = unused_colors(rng, 4, bg)
    # Vary the object count so the index range varies seed to seed.
    n_targets = int(rng.integers(3, 7))
    # Independent H,W, widened down toward the ARC small-grid mode and up to the
    # grid->grid token budget. A top marker row reserves row 0; the objects need
    # ~3-row tall shapes with a 1-cell gap, so keep a comfortable floor.
    h_min, h_max = 9, 22
    w_min, w_max = max(8, 2 * n_targets), 22
    condition_count = int(rng.integers(1, n_targets + 1))

    grid = None
    comps: list[dict] = []
    h = w = 0
    for _ in range(400):
        h, w = _independent_hw(rng, h_min, h_max, w_min, w_max)
        # Marker slots evenly distributed across the top row.
        step = max(2, w // (n_targets + 1))
        marker_cells = [(0, min(w - 1, 1 + i * step)) for i in range(n_targets)]
        if len({c for _, c in marker_cells}) != n_targets:
            continue
        shapes = _pick_unique_shapes(rng, n_targets)
        candidate = make_grid(h, w, fill=bg)
        for r, c in marker_cells[:condition_count]:
            candidate[r, c] = condition_color
        for r, c in marker_cells[condition_count:]:
            candidate[r, c] = distractor_color
        protected = {cell for cell in marker_cells}
        # Reserve the marker row from object placement.
        work = candidate.copy()
        work[0, :] = condition_color  # sentinel so shapes avoid the marker row
        placed = _try_place_shapes(rng, work, shapes, target_color, bg)
        if placed is None:
            continue
        for cells in placed:
            for rr, cc in cells:
                candidate[rr, cc] = target_color
        candidate_comps = _components(candidate, bg, {target_color})
        if len(candidate_comps) != n_targets:
            continue
        # Ordering must be unambiguous: objects strictly ordered by (top, then left).
        if not _distinct_top_left_order(candidate_comps):
            continue
        for cells in placed:
            protected |= cells
        bg_info = add_patterned_background(
            candidate, rng, bg=bg,
            exclude_colors=[condition_color, target_color, action_color, distractor_color],
            protected_cells=protected,
        )
        grid = candidate
        comps = candidate_comps
        break
    if grid is None:
        raise RuntimeError("could not place task 193 scene")

    action_type = str(rng.choice(["recolor", "erase"]))
    transform = {"type": "recolor", "color": action_color} if action_type == "recolor" else {"type": "erase"}
    selected = comps[condition_count - 1]
    out = grid.copy()
    _apply_transform(out, selected["cells"], bg, transform)

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_style": bg_info,
            "background_color": bg,
            "condition_color": condition_color,
            "target_color": target_color,
            "distractor_color": distractor_color,
            "action_color": action_color,
            "condition": "cell_color_equals_condition_color",
            "indexing": "one_based",
            "action_type": action_type,
            "action_transform": transform,
            "label_order_rule": "top_left_bounding_box_order",
            "target_objects": _records(comps, "target"),
        },
    }


@register("223")
def generate_property_condition_two_transforms(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, true_color, false_color = unused_colors(rng, 3, bg)
    transform_names = ["identity", "rotate90", "rotate180", "rotate270", "reflect_horizontal", "reflect_vertical"]
    n_objects = int(rng.integers(4, 7))
    # Independent H,W widened down to the structural minimum (objects span up to
    # ~3 cells + 1-cell gaps) and up to the grid->grid token budget.
    h_min, h_max = 10, 22
    w_min, w_max = 10, 22

    condition = str(rng.choice(["cell_count_at_least", "bbox_wider_than_tall", "cell_count_even"]))
    threshold = int(rng.integers(4, 7)) if condition == "cell_count_at_least" else 5

    grid = None
    comps: list[dict] = []
    for _ in range(400):
        h, w = _independent_hw(rng, h_min, h_max, w_min, w_max)
        base_shapes = _pick_unique_shapes(rng, n_objects)
        # Apply a random orientation to each shape so cell counts and bbox orientation
        # vary across seeds rather than being a fixed memorizable partition.
        shapes = [
            _transform_shape(shape, transform_names[int(rng.integers(0, len(transform_names)))])
            for shape in base_shapes
        ]
        candidate = make_grid(h, w, fill=bg)
        placed = _try_place_shapes(rng, candidate, shapes, object_color, bg)
        if placed is None:
            continue
        candidate_comps = _components(candidate, bg, {object_color})
        if len(candidate_comps) != n_objects:
            continue
        if not _distinct_top_left_order(candidate_comps):
            continue
        # The property partition must be non-trivial: some objects pass, some fail.
        verdicts = [_property_condition(comp, condition, threshold) for comp in candidate_comps]
        if not (any(verdicts) and not all(verdicts)):
            continue
        protected = set().union(*(comp["cells"] for comp in candidate_comps))
        bg_info = add_patterned_background(
            candidate, rng, bg=bg,
            exclude_colors=[object_color, true_color, false_color],
            protected_cells=protected,
        )
        grid = candidate
        comps = candidate_comps
        break
    if grid is None:
        raise RuntimeError("could not place task 223 scene")

    transform_case = str(rng.choice(["both_recolor", "true_recolor_false_erase", "true_erase_false_recolor"]))
    if transform_case == "both_recolor":
        true_transform = {"type": "recolor", "color": true_color}
        false_transform = {"type": "recolor", "color": false_color}
    elif transform_case == "true_recolor_false_erase":
        true_transform = {"type": "recolor", "color": true_color}
        false_transform = {"type": "erase"}
    else:
        true_transform = {"type": "erase"}
        false_transform = {"type": "recolor", "color": false_color}

    out = grid.copy()
    for comp in comps:
        transform = true_transform if _property_condition(comp, condition, threshold) else false_transform
        _apply_transform(out, comp["cells"], bg, transform)

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "background_style": bg_info,
            "object_color": object_color,
            "true_color": true_color,
            "false_color": false_color,
            "property_condition": condition,
            "threshold": threshold,
            "transform_A": true_transform,
            "transform_B": false_transform,
            "label_order_rule": "top_left_bounding_box_order",
            "objects": _records(comps, "object"),
        },
    }


@register("224")
def generate_property_value_transform_mapping(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, wide_color, tall_color, square_color = unused_colors(rng, 4, bg)
    mapping = {
        "wide": {"type": "recolor", "color": wide_color},
        "tall": {"type": "recolor", "color": tall_color},
        "square": {"type": "recolor", "color": square_color},
    }
    # Vary object count while guaranteeing all three orientation classes appear
    # (the rule needs wide, tall, and square objects to all be exercised).
    n_objects = int(rng.integers(3, 7))
    h_min, h_max = 10, 22
    w_min, w_max = 10, 22

    grid = None
    comps: list[dict] = []
    for _ in range(400):
        h, w = _independent_hw(rng, h_min, h_max, w_min, w_max)
        # One shape from each orientation class, then fill the remainder from the
        # union so the orientation multiset varies across seeds.
        chosen: list[Shape] = [
            TASK_224_WIDE[int(rng.integers(0, len(TASK_224_WIDE)))],
            TASK_224_TALL[int(rng.integers(0, len(TASK_224_TALL)))],
            TASK_224_SQUARE[int(rng.integers(0, len(TASK_224_SQUARE)))],
        ]
        pool = TASK_224_WIDE + TASK_224_TALL + TASK_224_SQUARE
        for _ in range(n_objects - 3):
            chosen.append(pool[int(rng.integers(0, len(pool)))])
        order = list(rng.permutation(len(chosen)))
        shapes = [chosen[int(i)] for i in order]
        candidate = make_grid(h, w, fill=bg)
        placed = _try_place_shapes(rng, candidate, shapes, object_color, bg)
        if placed is None:
            continue
        candidate_comps = _components(candidate, bg, {object_color})
        if len(candidate_comps) != n_objects:
            continue
        if not _distinct_top_left_order(candidate_comps):
            continue
        # All three orientation classes must be represented.
        present = {_bbox_orientation(comp) for comp in candidate_comps}
        if present != {"wide", "tall", "square"}:
            continue
        protected = set().union(*(comp["cells"] for comp in candidate_comps))
        bg_info = add_patterned_background(
            candidate, rng, bg=bg,
            exclude_colors=[object_color, wide_color, tall_color, square_color],
            protected_cells=protected,
        )
        grid = candidate
        comps = candidate_comps
        break
    if grid is None:
        raise RuntimeError("could not place task 224 scene")

    out = grid.copy()
    for comp in comps:
        _apply_transform(out, comp["cells"], bg, mapping[_bbox_orientation(comp)])

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "background_style": bg_info,
            "object_color": object_color,
            "classification_property": "bbox_orientation",
            "property_values": ["wide", "tall", "square"],
            "value_to_transform": [{"value": value, "transform": mapping[value]} for value in ["wide", "tall", "square"]],
            "label_order_rule": "top_left_bounding_box_order",
            "objects": _records(comps, "object"),
        },
    }


def _nearest_marker_unique(comp: dict, markers: list[dict], relation: str) -> dict | None:
    """Return the strictly nearest marker, or None when the minimum distance is tied."""
    scored = [
        (_distance_to_marker(comp, tuple(item["position"]), relation), item)
        for item in markers
    ]
    best = min(d for d, _ in scored)
    winners = [item for d, item in scored if d == best]
    if len(winners) != 1:
        return None
    return winners[0]


@register("225")
def generate_marker_associated_object_transform(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    # Vary object and marker counts; colours are object + n_markers distinct markers.
    n_objects = int(rng.integers(3, 6))
    n_markers = int(rng.integers(2, 5))
    object_color, *marker_colors = unused_colors(rng, 1 + n_markers, bg)
    n_markers = len(marker_colors)
    # Independent H,W. Markers are scattered single cells, so the grid must stay
    # large enough for objects + markers with a gap; widen down toward ~12.
    h_min, h_max = 12, 22
    w_min, w_max = 12, 22
    relation = str(rng.choice(["nearest_marker_by_min_manhattan", "nearest_marker_by_medoid"]))

    grid = None
    comps: list[dict] = []
    markers: list[dict] = []
    require_divergence = True
    for attempt in range(1200):
        # After many attempts, relax the "relations must disagree" constraint so the
        # generator always terminates; the geometry diversity fix still applies.
        if attempt == 900:
            require_divergence = False
        h, w = _independent_hw(rng, h_min, h_max, w_min, w_max)
        shapes = _pick_unique_shapes(rng, n_objects, pool=TASK_225_SHAPES)
        candidate = make_grid(h, w, fill=bg)
        placed = _try_place_shapes(rng, candidate, shapes, object_color, bg)
        if placed is None:
            continue
        candidate_comps = _components(candidate, bg, {object_color})
        if len(candidate_comps) != n_objects:
            continue
        if not _distinct_top_left_order(candidate_comps):
            continue
        # Markers at random background cells (with a 1-cell gap between markers so
        # they read as separate single cells) scattered across the grid so the two
        # association relations are free to diverge and it is not a quadrant lookup.
        bg_positions = [tuple(p) for p in np.argwhere(candidate == bg)]
        if len(bg_positions) < n_markers:
            continue
        marker_positions: list[Cell] = []
        idx_order = list(rng.permutation(len(bg_positions)))
        for idx in idx_order:
            r, c = bg_positions[int(idx)]
            if any(abs(r - mr) + abs(c - mc) <= 1 for mr, mc in marker_positions):
                continue
            marker_positions.append((int(r), int(c)))
            if len(marker_positions) == n_markers:
                break
        if len(marker_positions) != n_markers:
            continue
        candidate_markers = [
            {"label": f"marker{i}", "color": int(color), "position": [int(r), int(c)]}
            for i, (color, (r, c)) in enumerate(zip(marker_colors, marker_positions), start=1)
        ]
        # The answer must be uniquely determined: every object has a strictly nearest
        # marker under the active relation (no tied minimum distance).
        assigned = [_nearest_marker_unique(comp, candidate_markers, relation) for comp in candidate_comps]
        if any(item is None for item in assigned):
            continue
        # Require that at least two distinct markers are used so the task is not
        # trivially "every object maps to the same marker".
        if len({item["label"] for item in assigned}) < 2:
            continue
        if require_divergence:
            # Prefer layouts where the two association relations actually disagree for
            # some object, so the relation the prompt varies is not functionally inert.
            other_relation = (
                "nearest_marker_by_medoid"
                if relation == "nearest_marker_by_min_manhattan"
                else "nearest_marker_by_min_manhattan"
            )
            other_assigned = [
                _nearest_marker_unique(comp, candidate_markers, other_relation)
                for comp in candidate_comps
            ]
            if any(item is None for item in other_assigned):
                continue
            if [item["label"] for item in assigned] == [item["label"] for item in other_assigned]:
                continue
        # Patterned background: texture must avoid object colour and every marker
        # colour (markers are single coloured cells located by colour) and must
        # not overwrite object cells or marker positions.
        protected = set().union(*(comp["cells"] for comp in candidate_comps))
        protected |= {tuple(m["position"]) for m in candidate_markers}
        bg_info = add_patterned_background(
            candidate, rng, bg=bg,
            exclude_colors=[object_color, *marker_colors],
            protected_cells=protected,
        )
        grid = candidate
        comps = candidate_comps
        markers = candidate_markers
        break
    if grid is None:
        raise RuntimeError("could not place task 225 scene")

    for marker in markers:
        r, c = marker["position"]
        grid[r, c] = int(marker["color"])

    out = grid.copy()
    for comp in comps:
        marker = _nearest_marker_unique(comp, markers, relation)
        _apply_transform(out, comp["cells"], bg, {"type": "recolor", "color": marker["color"]})

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "background_style": bg_info,
            "object_color": object_color,
            "marker_colors": [int(c) for c in marker_colors],
            "association_relation": relation,
            "associated_transform": "recolor_object_to_marker_color",
            "label_order_rule": "top_left_bounding_box_order",
            "objects": _records(comps, "object"),
            "markers": markers,
        },
    }


def _place_in_window(
    rng: np.random.Generator,
    grid: np.ndarray,
    shape: Shape,
    color: int,
    bg: int,
    r_lo: int,
    r_hi: int,
    c_lo: int,
    c_hi: int,
    *,
    max_attempts: int = 80,
) -> set[Cell] | None:
    """Place a shape whose bounding box fits entirely inside the inclusive
    window [r_lo,r_hi] x [c_lo,c_hi] (used to keep an object inside one quadrant)."""
    r0, c0, r1, c1 = _bbox(shape)
    sh = r1 - r0 + 1
    sw = c1 - c0 + 1
    if r_hi - r_lo + 1 < sh or c_hi - c_lo + 1 < sw:
        return None
    for _ in range(max_attempts):
        tr = int(rng.integers(r_lo, r_hi - sh + 2))
        tc = int(rng.integers(c_lo, c_hi - sw + 2))
        cells = {(tr + r - r0, tc + c - c0) for r, c in shape}
        if all(grid[rr, cc] == bg for rr, cc in cells):
            for rr, cc in cells:
                grid[rr, cc] = color
            return cells
    return None


@register("226")
def generate_region_membership_transform_mapping(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    divider_color, object_color, tl_color, tr_color, bl_color, br_color = unused_colors(rng, 6, bg)
    mapping = {
        "top_left": {"type": "recolor", "color": tl_color},
        "top_right": {"type": "recolor", "color": tr_color},
        "bottom_left": {"type": "recolor", "color": bl_color},
        "bottom_right": {"type": "recolor", "color": br_color},
    }
    # Independent H,W with room for a 4-quadrant split; widen down toward ~13.
    h_min, h_max = 13, 22
    w_min, w_max = 13, 22
    # Vary how many objects sit in each quadrant (each quadrant gets >=1 so all
    # four regions are always exercised; some quadrants get a second object).
    n_extra = int(rng.integers(0, 4))

    grid = None
    comps: list[dict] = []
    split_row = split_col = 0
    for _ in range(400):
        h, w = _independent_hw(rng, h_min, h_max, w_min, w_max)
        # Split lines kept off the borders so every quadrant has interior room.
        split_row = int(rng.integers(5, h - 4))
        split_col = int(rng.integers(5, w - 4))
        candidate = make_grid(h, w, fill=bg)
        candidate[split_row, :] = divider_color
        candidate[:, split_col] = divider_color
        # Inclusive interior windows for each quadrant (1-cell gap from divider/border).
        windows = {
            "top_left": (1, split_row - 2, 1, split_col - 2),
            "top_right": (1, split_row - 2, split_col + 2, w - 2),
            "bottom_left": (split_row + 2, h - 2, 1, split_col - 2),
            "bottom_right": (split_row + 2, h - 2, split_col + 2, w - 2),
        }
        regions = ["top_left", "top_right", "bottom_left", "bottom_right"]
        # Quadrants receiving a second object.
        extra_regions = [regions[int(i)] for i in rng.permutation(4)[:n_extra]]
        placement_plan = regions + extra_regions
        pool_keys: set[tuple[Cell, ...]] = set()
        cells_list: list[set[Cell]] = []
        ok = True
        for region in placement_plan:
            r_lo, r_hi, c_lo, c_hi = windows[region]
            shape = _pick_unique_shapes(rng, 1)[0]
            cells = _place_in_window(rng, candidate, shape, object_color, bg, r_lo, r_hi, c_lo, c_hi)
            if cells is None:
                ok = False
                break
            cells_list.append(cells)
        if not ok:
            continue
        candidate_comps = _components(candidate, bg, {object_color})
        if len(candidate_comps) != len(placement_plan):
            continue
        if not _distinct_top_left_order(candidate_comps):
            continue
        seen = {_region_for(comp, split_row, split_col) for comp in candidate_comps}
        if seen != {"top_left", "top_right", "bottom_left", "bottom_right"}:
            continue
        protected = set().union(*(comp["cells"] for comp in candidate_comps))
        # Divider cells are protected implicitly (not background); texture only
        # fills background cells in colours disjoint from all semantic colours.
        bg_info = add_patterned_background(
            candidate, rng, bg=bg,
            exclude_colors=[divider_color, object_color, tl_color, tr_color, bl_color, br_color],
            protected_cells=protected,
        )
        grid = candidate
        comps = candidate_comps
        break
    if grid is None:
        raise RuntimeError("could not place task 226 scene")

    out = grid.copy()
    for comp in comps:
        _apply_transform(out, comp["cells"], bg, mapping[_region_for(comp, split_row, split_col)])

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "background_style": bg_info,
            "divider_color": divider_color,
            "object_color": object_color,
            "region_membership_rule": "object_medoid_quadrant",
            "split_row": split_row,
            "split_col": split_col,
            "region_to_transform": [{"region": region, "transform": mapping[region]} for region in ["top_left", "top_right", "bottom_left", "bottom_right"]],
            "label_order_rule": "top_left_bounding_box_order",
            "objects": _records(comps, "object"),
        },
    }


@register("232")
def generate_selected_input_corresponding_output_highlight(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, selected_color, highlight_color = unused_colors(rng, 3, bg)
    n_objects = int(rng.integers(3, 6))

    input_grid = output_grid = None
    shapes: list[Shape] = []
    output_shapes: list[Shape] = []
    output_order: list[int] = []
    anchors: list[Cell] = []
    selected_idx = 0
    for _ in range(200):
        h, w = _block_size(rng, n_objects)
        anchors = _layout_anchors(rng, n_objects, h, w)
        if anchors is None:
            continue
        shapes = _pick_unique_shapes(rng, n_objects)
        selected_idx = int(rng.integers(0, n_objects))
        input_colors = [object_color] * n_objects
        input_colors[selected_idx] = selected_color
        cand_in = _scene(shapes, anchors, bg, input_colors, h=h, w=w)
        output_order = _permutation(rng, n_objects)
        output_shapes = [shapes[i] for i in output_order]
        cand_out = _scene(output_shapes, anchors, bg, [object_color] * n_objects, h=h, w=w)
        in_comps = _components(cand_in, bg, {object_color, selected_color})
        out_comps = _components(cand_out, bg, {object_color})
        if len(in_comps) != n_objects or len(out_comps) != n_objects:
            continue
        if not (_distinct_top_left_order(in_comps) and _distinct_top_left_order(out_comps)):
            continue
        input_grid, output_grid = cand_in, cand_out
        break
    if input_grid is None:
        raise RuntimeError("could not place task 232 scene")

    answer_grid = output_grid.copy()
    selected_output_idx = output_order.index(selected_idx)
    selected_cells = _abs(output_shapes[selected_output_idx], anchors[selected_output_idx])
    for r, c in selected_cells:
        answer_grid[r, c] = highlight_color

    input_comps = _components(input_grid, bg, {object_color, selected_color})
    output_comps = _components(output_grid, bg, {object_color})
    return {
        "input": {"input_grid": grid_to_list(input_grid), "output_grid": grid_to_list(output_grid)},
        "output": grid_to_list(answer_grid),
        "metadata": {
            "background_color": bg,
            "object_color": object_color,
            "selected_color": selected_color,
            "highlight_color": highlight_color,
            "label_order_rule": "top_left_bounding_box_order",
            "matching_rule": "same_normalized_shape",
            "selected_input_label": f"input{selected_idx + 1}",
            "corresponding_output_label": f"output{selected_output_idx + 1}",
            "input_objects": _records(input_comps, "input"),
            "output_objects": _records(output_comps, "output"),
        },
    }


@register("233")
def generate_same_count_one_to_one_correspondence(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    object_color = unused_colors(rng, 1, bg)[0]
    n_objects = int(rng.integers(3, 7))

    input_grid = output_grid = None
    for _ in range(200):
        h, w = _block_size(rng, n_objects)
        anchors = _layout_anchors(rng, n_objects, h, w)
        if anchors is None:
            continue
        shapes = _pick_unique_shapes(rng, n_objects)
        cand_in = _scene(shapes, anchors, bg, [object_color] * n_objects, h=h, w=w)
        output_order = _permutation(rng, n_objects)
        cand_out = _scene([shapes[i] for i in output_order], anchors, bg, [object_color] * n_objects, h=h, w=w)
        in_comps = _components(cand_in, bg, {object_color})
        out_comps = _components(cand_out, bg, {object_color})
        if len(in_comps) != n_objects or len(out_comps) != n_objects:
            continue
        if not (_distinct_top_left_order(in_comps) and _distinct_top_left_order(out_comps)):
            continue
        input_grid, output_grid = cand_in, cand_out
        break
    if input_grid is None:
        raise RuntimeError("could not place task 233 scene")

    return {
        "input": {"input_grid": grid_to_list(input_grid), "output_grid": grid_to_list(output_grid)},
        "output": _correspondence_from_grids(input_grid, output_grid, bg, object_color, "same_normalized_shape"),
        "metadata": _pair_metadata(
            bg=bg,
            object_color=object_color,
            input_grid=input_grid,
            output_grid=output_grid,
            relation="same_normalized_shape",
            extra={"output_format": "semicolon_separated_input_to_output_pairs"},
        ),
    }


@register("234")
def generate_equivalent_object_correspondence(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    object_color = unused_colors(rng, 1, bg)[0]
    relation = str(rng.choice(["same_shape_under_rotation", "same_shape_under_reflection"]))
    n_objects = int(rng.integers(3, 7))
    transforms = ["rotate90", "rotate180", "rotate270"] if relation == "same_shape_under_rotation" else ["reflect_horizontal", "reflect_vertical"]

    input_grid = output_grid = None
    output_order: list[int] = []
    output_transforms: list[str] = []
    for _ in range(200):
        h, w = _block_size(rng, n_objects)
        anchors = _layout_anchors(rng, n_objects, h, w)
        if anchors is None:
            continue
        shapes = _pick_unique_shapes(rng, n_objects, relation=relation)
        cand_in = _scene(shapes, anchors, bg, [object_color] * n_objects, h=h, w=w)
        output_order = _permutation(rng, n_objects)
        output_transforms = [str(rng.choice(transforms)) for _ in output_order]
        output_shapes = [_transform_shape(shapes[i], output_transforms[j]) for j, i in enumerate(output_order)]
        cand_out = _scene(output_shapes, anchors, bg, [object_color] * n_objects, h=h, w=w)
        in_comps = _components(cand_in, bg, {object_color})
        out_comps = _components(cand_out, bg, {object_color})
        if len(in_comps) != n_objects or len(out_comps) != n_objects:
            continue
        if not (_distinct_top_left_order(in_comps) and _distinct_top_left_order(out_comps)):
            continue
        # The matching must stay unique under the relation: each output object's
        # equivalence-class key must match exactly one input object.
        in_keys = [_canonical(s, relation) for s in shapes]
        out_keys = [_canonical(s, relation) for s in output_shapes]
        if len(set(in_keys)) != n_objects or len(set(out_keys)) != n_objects:
            continue
        if set(in_keys) != set(out_keys):
            continue
        input_grid, output_grid = cand_in, cand_out
        break
    if input_grid is None:
        raise RuntimeError("could not place task 234 scene")

    return {
        "input": {"input_grid": grid_to_list(input_grid), "output_grid": grid_to_list(output_grid)},
        "output": _correspondence_from_grids(input_grid, output_grid, bg, object_color, relation),
        "metadata": _pair_metadata(
            bg=bg,
            object_color=object_color,
            input_grid=input_grid,
            output_grid=output_grid,
            relation=relation,
            extra={
                "output_transforms_in_output_order": output_transforms,
                "output_format": "semicolon_separated_input_to_output_pairs",
            },
        ),
    }


@register("235")
def generate_correspondence_statuses_with_count_changes(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    object_color = unused_colors(rng, 1, bg)[0]
    # k input objects; the output has k+1 objects: one input copied (2x), one
    # deleted (0x), the remaining (k-2) matched (1x each), plus 1 new shape.
    n_inputs = int(rng.integers(3, 6))
    n_outputs = n_inputs + 1

    input_grid = output_grid = None
    input_shapes: list[Shape] = []
    output_items: list[tuple[str, int | None, Shape]] = []
    copied_idx = deleted_idx = 0
    matched_idxs: list[int] = []
    input_keys: set[tuple[Cell, ...]] = set()
    for _ in range(200):
        h, w = _block_size(rng, n_outputs)
        in_anchors = _layout_anchors(rng, n_inputs, h, w)
        out_anchors = _layout_anchors(rng, n_outputs, h, w)
        if in_anchors is None or out_anchors is None:
            continue
        input_shapes = _pick_unique_shapes(rng, n_inputs)
        input_keys = {_canonical(shape, "same_normalized_shape") for shape in input_shapes}
        new_shape = _pick_unique_shapes(rng, 1, exclude=input_keys)[0]
        role_order = [int(i) for i in rng.permutation(n_inputs)]
        copied_idx = role_order[0]
        deleted_idx = role_order[1]
        matched_idxs = sorted(role_order[2:])
        items: list[tuple[str, int | None, Shape]] = [
            ("copy", copied_idx, input_shapes[copied_idx]),
            ("copy", copied_idx, input_shapes[copied_idx]),
            ("new", None, new_shape),
        ]
        for mi in matched_idxs:
            items.append(("matched", mi, input_shapes[mi]))
        rng.shuffle(items)
        cand_in = _scene(input_shapes, in_anchors, bg, [object_color] * n_inputs, h=h, w=w)
        cand_out = _scene([it[2] for it in items], out_anchors, bg, [object_color] * n_outputs, h=h, w=w)
        in_comps = _components(cand_in, bg, {object_color})
        out_comps = _components(cand_out, bg, {object_color})
        if len(in_comps) != n_inputs or len(out_comps) != n_outputs:
            continue
        if not (_distinct_top_left_order(in_comps) and _distinct_top_left_order(out_comps)):
            continue
        input_grid, output_grid = cand_in, cand_out
        output_items = items
        break
    if input_grid is None:
        raise RuntimeError("could not place task 235 scene")

    # Build the answer from the actual top-left component order of each grid
    # (mirrors the verifier and is independent of the jittered placement order).
    output_text = _status_summary_from_grids(input_grid, output_grid, bg, object_color)

    # Derive a correct role summary from the input components (labels are by the
    # input grid's top-left order, matching the answer's label space).
    input_comps = _components(input_grid, bg, {object_color})
    output_comps = _components(output_grid, bg, {object_color})
    out_key_counts = Counter(_canonical(c["cells"], "same_normalized_shape") for c in output_comps)
    role_copied = role_deleted = None
    role_matched: list[str] = []
    for idx, comp in enumerate(input_comps, start=1):
        cnt = out_key_counts.get(_canonical(comp["cells"], "same_normalized_shape"), 0)
        if cnt == 0:
            role_deleted = f"input{idx}"
        elif cnt >= 2:
            role_copied = f"input{idx}"
        else:
            role_matched.append(f"input{idx}")

    return {
        "input": {"input_grid": grid_to_list(input_grid), "output_grid": grid_to_list(output_grid)},
        "output": output_text,
        "metadata": _pair_metadata(
            bg=bg,
            object_color=object_color,
            input_grid=input_grid,
            output_grid=output_grid,
            relation="same_normalized_shape",
            extra={
                "classification_format": "matched=...;deleted=...;copied=...;new=...",
                "role_summary": {
                    "copied_input_label": role_copied,
                    "deleted_input_label": role_deleted,
                    "matched_input_labels": role_matched,
                },
            },
        ),
    }
