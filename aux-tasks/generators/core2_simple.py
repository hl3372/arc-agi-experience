"""Simple core task generators for selected second-batch tasks."""
from __future__ import annotations

from collections import Counter

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid, pick_color, pick_colors
from common.objects import bounding_box, connected_components, random_shape, stamp_rectangle
from registry import register

MAX_RETRIES = 50


def _unused_colors(rng: np.random.Generator, n: int, *exclude: int) -> list[int]:
    return pick_colors(rng, n, exclude=[int(c) for c in exclude])


def _sample_dim(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Sample one grid side in [lo, hi], weighted toward small (ARC-like).

    Real ARC grids are dominated by small sizes (p25=6, median 10) but span up to
    30. A flat uniform over a wide range over-represents large grids, so we mix a
    small-biased triangular draw with an occasional uniform draw to keep the full
    span covered (which is what clears the narrow-span warning) while still
    centring near the ARC median.
    """
    lo = int(lo)
    hi = int(hi)
    if hi <= lo:
        return lo
    if rng.random() < 0.78:
        # Triangular biased toward lo (mode at lo) -> plenty of small grids.
        v = int(rng.triangular(lo, lo, hi + 1 - 1e-9))
    else:
        # Uniform across the whole envelope -> guarantees the large end appears,
        # giving a wide H/W span and decorrelated H vs W.
        v = int(rng.integers(lo, hi + 1))
    return min(hi, max(lo, v))


def _sample_hw(
    rng: np.random.Generator,
    *,
    h_lo: int,
    h_hi: int,
    w_lo: int | None = None,
    w_hi: int | None = None,
    square_prob: float = 0.18,
) -> tuple[int, int]:
    """Sample (H, W) independently over an ARC-aligned envelope.

    H and W are drawn independently (decorrelated) so the H/W correlation stays
    low, but with probability ``square_prob`` we force H == W because real ARC is
    ~62% square and a model should still see square cases. The per-axis span is
    wide enough (>8) to clear the narrow-grid warning while small grids stay the
    most common, matching the measured ARC distribution.
    """
    w_lo = h_lo if w_lo is None else w_lo
    w_hi = h_hi if w_hi is None else w_hi
    h = _sample_dim(rng, h_lo, h_hi)
    if rng.random() < square_prob:
        lo = max(h_lo, w_lo)
        hi = min(h_hi, w_hi)
        if lo <= h <= hi:
            return h, h
    w = _sample_dim(rng, w_lo, w_hi)
    return h, w


def _bbox_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_shape(
    grid: np.ndarray,
    rng: np.random.Generator,
    shape: frozenset[tuple[int, int]],
    color: int,
    bg: int,
    placed_boxes: list[tuple[int, int, int, int]],
    gap: int = 1,
    margin: int = 0,
    max_attempts: int = 200,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = bounding_box(shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    if sh + 2 * margin > h or sw + 2 * margin > w:
        return None
    for _ in range(max_attempts):
        r0 = int(rng.integers(margin, h - sh - margin + 1))
        c0 = int(rng.integers(margin, w - sw - margin + 1))
        box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        if any(_boxes_overlap(box, other, gap=gap) for other in placed_boxes):
            continue
        cells = {(r0 + r - sr0, c0 + c - sc0) for r, c in shape}
        if all(grid[r, c] == bg for r, c in cells):
            for r, c in cells:
                grid[r, c] = color
            placed_boxes.append(box)
            return cells
    return None


def _count_4_components(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> int:
    cells = set(cells)
    seen = set()
    count = 0
    for cell in cells:
        if cell in seen:
            continue
        count += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return count


def _has_multiple_4_components(shape: frozenset[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    return _count_4_components(shape) > 1


def _random_object_shape(
    rng: np.random.Generator,
    *,
    min_cells: int,
    max_cells: int,
    connectivity: int,
    prefer_diagonal: bool = False,
) -> frozenset[tuple[int, int]]:
    for _ in range(50):
        size = int(rng.integers(min_cells, max_cells + 1))
        shape = random_shape(rng, (size, size), connectivity=connectivity, allow_named=(connectivity == 4))
        if not prefer_diagonal or _has_multiple_4_components(shape):
            return shape
    return frozenset((i, i) for i in range(min(3, max_cells)))


def _random_scene_objects(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    bg: int,
    n_objects: int,
    same_color: int | None = None,
    min_cells: int = 4,
    max_cells: int = 9,
    gap: int = 1,
    connectivity: int = 4,
    require_diagonal_object: bool = False,
    margin: int = 0,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = make_grid(h, w, fill=bg)
    colors = (
        [int(same_color)] * n_objects
        if same_color is not None
        else _unused_colors(rng, n_objects, bg)
    )
    objects: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    diagonal_done = False
    for index, color in enumerate(colors):
        for _ in range(50):
            prefer_diagonal = bool(
                connectivity == 8
                and require_diagonal_object
                and not diagonal_done
                and index == len(colors) - 1
            )
            shape = _random_object_shape(
                rng,
                min_cells=min_cells,
                max_cells=max_cells,
                connectivity=connectivity,
                prefer_diagonal=prefer_diagonal,
            )
            cells = _place_shape(grid, rng, shape, int(color), bg, boxes, gap=gap, margin=margin)
            if cells is not None:
                diagonal_done = diagonal_done or _has_multiple_4_components(shape)
                objects.append({"color": int(color), "cells": cells, "bbox": bounding_box(cells)})
                break
        else:
            return None
    return grid, objects


def _add_non_target_distractors(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    target_color: int,
    placed_boxes: list[tuple[int, int, int, int]],
    extra_exclude: list[int] | tuple[int, ...] = (),
) -> list[int]:
    distractor_colors = _unused_colors(rng, int(rng.integers(1, 4)), bg, target_color, *extra_exclude)
    h, w = grid.shape
    for color in distractor_colors:
        if rng.random() < 0.65:
            shape = _random_object_shape(rng, min_cells=2, max_cells=7, connectivity=int(rng.choice([4, 8])))
            _place_shape(grid, rng, shape, int(color), bg, placed_boxes, gap=1, max_attempts=100)
        else:
            for _ in range(int(rng.integers(3, 9))):
                r = int(rng.integers(0, h))
                c = int(rng.integers(0, w))
                if grid[r, c] == bg:
                    grid[r, c] = int(color)
    return [int(c) for c in distractor_colors]


def _inside_any_box(r: int, c: int, boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes)


def _available_colors(*exclude: int) -> list[int]:
    excluded = {int(c) for c in exclude}
    return [c for c in range(NUM_COLORS) if c not in excluded]


def _add_background_texture(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    exclude_colors: list[int] | tuple[int, ...] = (),
    protected_boxes: list[tuple[int, int, int, int]] | None = None,
    protected_cells: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    density_scale: float = 1.0,
    solid_probability: float = 0.30,
) -> dict:
    protected_boxes = protected_boxes or []
    protected_cells = set(protected_cells or set())
    if rng.random() < solid_probability:
        return {"style": "solid", "colors": [], "period": 0}
    palette = _available_colors(bg, *exclude_colors)
    if not palette:
        return {"style": "solid", "colors": [], "period": 0}
    n_colors = min(len(palette), int(rng.integers(1, 3)))
    colors = [int(c) for c in rng.choice(np.array(palette, dtype=np.uint8), size=n_colors, replace=False)]
    style = str(rng.choice(["checkerboard", "horizontal_stripes", "vertical_stripes", "diagonal_stripes", "dotted_grid"]))
    period = int(rng.integers(2, 5 if density_scale >= 0.8 else 6))
    offset = int(rng.integers(0, period))
    used: set[int] = set()
    h, w = grid.shape

    def can_fill(rr: int, cc: int) -> bool:
        return (
            grid[rr, cc] == bg
            and (rr, cc) not in protected_cells
            and not _inside_any_box(rr, cc, protected_boxes)
        )

    for r in range(h):
        for c in range(w):
            if not can_fill(r, c):
                continue
            pattern_slot: int | None = None
            if style == "checkerboard" and ((r // period) + (c // period) + offset) % 2 == 0:
                pattern_slot = ((r // period) + (c // period)) % len(colors)
            elif style == "horizontal_stripes" and (r + offset) % period == 0:
                pattern_slot = (r // period) % len(colors)
            elif style == "vertical_stripes" and (c + offset) % period == 0:
                pattern_slot = (c // period) % len(colors)
            elif style == "diagonal_stripes" and (r + c + offset) % period == 0:
                pattern_slot = ((r + c) // period) % len(colors)
            elif style == "dotted_grid" and (r + offset) % period == 0 and (c + offset) % period == 0:
                pattern_slot = ((r // period) + (c // period)) % len(colors)
            if pattern_slot is not None:
                color = colors[pattern_slot]
                grid[r, c] = color
                used.add(color)

    return {"style": style, "colors": sorted(used), "period": period}


def _add_random_distractor_cells(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    exclude_colors: list[int] | tuple[int, ...],
    density: float = 0.08,
) -> list[int]:
    colors = _unused_colors(rng, int(rng.integers(1, 4)), bg, *exclude_colors)
    h, w = grid.shape
    attempts = max(4, int(h * w * density))
    used: set[int] = set()
    for _ in range(attempts):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if grid[r, c] != bg:
            continue
        color = int(rng.choice(colors))
        grid[r, c] = color
        used.add(color)
    return sorted(used)


# Tasks 95 and 96 moved out of this file on 2026-09-15: the bounding-box outline
# task lives in ``generators/bbox_outline_95.py`` and the frame-interior task in
# ``generators/frame_extract_96.py``, both rebuilt on the shared shape and
# background libraries.  Every other task in this file is unchanged.


@register("118")
def generate_block_compression(rng: np.random.Generator) -> dict:
    block_size = int(rng.choice([2, 3, 4]))
    out_h = int(rng.integers(3, 8))
    out_w = int(rng.integers(3, 8))
    bg = int(rng.integers(0, NUM_COLORS))
    rule = str(rng.choice(["top_left_color", "majority_color", "any_target_presence"]))
    target_color = pick_color(rng, [bg])
    true_color, false_color = _unused_colors(rng, 2, target_color)
    grid = make_grid(out_h * block_size, out_w * block_size, fill=bg)
    out = make_grid(out_h, out_w, fill=bg)
    palette = _unused_colors(rng, min(5, NUM_COLORS - 1), bg)
    palette.append(bg)

    for br in range(out_h):
        for bc in range(out_w):
            block = rng.choice(np.array(palette, dtype=np.uint8), size=(block_size, block_size))
            if rule == "any_target_presence" and rng.random() < 0.5:
                block[int(rng.integers(0, block_size)), int(rng.integers(0, block_size))] = target_color
            grid[
                br * block_size : (br + 1) * block_size,
                bc * block_size : (bc + 1) * block_size,
            ] = block
            flat = [int(v) for v in block.flatten()]
            if rule == "top_left_color":
                out[br, bc] = int(block[0, 0])
            elif rule == "majority_color":
                counts = Counter(flat)
                max_count = max(counts.values())
                majority_colors = [color for color, count in counts.items() if count == max_count]
                out[br, bc] = majority_colors[0] if len(majority_colors) == 1 else bg
            else:
                out[br, bc] = true_color if target_color in flat else false_color

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "block_size": block_size,
            "summary_rule": rule,
            "background_color": bg,
            "target_color": int(target_color),
            "true_color": int(true_color),
            "false_color": int(false_color),
            "majority_tie_rule": "if no color is uniquely most frequent, output the background color",
        },
    }


@register("133")
def generate_same_size_binary_mask(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 16))
    w = int(rng.integers(6, 16))
    bg = int(rng.integers(0, NUM_COLORS))
    target_color, true_color, false_color = _unused_colors(rng, 3, bg)
    grid = make_grid(h, w, fill=bg)
    palette = [bg] + _unused_colors(rng, 3, bg, target_color)
    for r in range(h):
        for c in range(w):
            if rng.random() < 0.25:
                grid[r, c] = target_color
            elif rng.random() < 0.2:
                grid[r, c] = int(rng.choice(palette))
    predicate = str(rng.choice(["color_equals", "non_background"]))
    if predicate == "color_equals":
        mask = grid == target_color
        desc = f"cell color equals {target_color}"
    else:
        mask = grid != bg
        desc = f"cell color is not background color {bg}"
    out = np.where(mask, true_color, false_color).astype(np.uint8)
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "predicate": predicate,
            "predicate_description": desc,
            "target_color": int(target_color),
            "background_color": bg,
            "true_color": int(true_color),
            "false_color": int(false_color),
        },
    }


@register("134")
def generate_copy_marked_row_or_column(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=5, h_hi=24)
    bg = int(rng.integers(0, NUM_COLORS))
    source_marker, target_marker = _unused_colors(rng, 2, bg)
    fill_colors = _unused_colors(rng, 4, bg, source_marker, target_marker)
    grid = rng.choice(np.array([bg] + fill_colors, dtype=np.uint8), size=(h, w))
    axis = str(rng.choice(["row", "column"]))
    policy = str(rng.choice(["copy_all", "preserve_target_marker"]))
    out = grid.copy()
    if axis == "row":
        source_idx, target_idx = rng.choice(h, size=2, replace=False).tolist()
        source_pos = int(rng.integers(0, w))
        target_pos = int(rng.integers(0, w))
        grid[source_idx, :] = rng.choice(np.array([bg] + fill_colors, dtype=np.uint8), size=w)
        grid[target_idx, :] = rng.choice(np.array([bg] + fill_colors, dtype=np.uint8), size=w)
        grid[source_idx, source_pos] = source_marker
        grid[target_idx, target_pos] = target_marker
        out = grid.copy()
        out[target_idx, :] = grid[source_idx, :]
        if policy == "preserve_target_marker":
            out[target_idx, target_pos] = target_marker
    else:
        source_idx, target_idx = rng.choice(w, size=2, replace=False).tolist()
        source_pos = int(rng.integers(0, h))
        target_pos = int(rng.integers(0, h))
        grid[:, source_idx] = rng.choice(np.array([bg] + fill_colors, dtype=np.uint8), size=h)
        grid[:, target_idx] = rng.choice(np.array([bg] + fill_colors, dtype=np.uint8), size=h)
        grid[source_pos, source_idx] = source_marker
        grid[target_pos, target_idx] = target_marker
        out = grid.copy()
        out[:, target_idx] = grid[:, source_idx]
        if policy == "preserve_target_marker":
            out[target_pos, target_idx] = target_marker
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "axis": axis,
            "source_marker_color": int(source_marker),
            "target_marker_color": int(target_marker),
            "marker_policy": policy,
            "background_color": bg,
        },
    }


@register("135")
def generate_delete_rows_columns_by_condition(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, h_lo=5, h_hi=24)
        bg = int(rng.integers(0, NUM_COLORS))
        condition_color = pick_color(rng, [bg])
        palette = [bg] + _unused_colors(rng, 4, bg, condition_color)
        grid = rng.choice(np.array(palette, dtype=np.uint8), size=(h, w))
        axis = str(rng.choice(["rows", "columns", "both"]))
        condition = str(rng.choice(["all_background", "contains_color"]))
        if condition == "all_background":
            if axis in ("rows", "both"):
                for r in rng.choice(h, size=max(1, h // 4), replace=False):
                    grid[int(r), :] = bg
            if axis in ("columns", "both"):
                for c in rng.choice(w, size=max(1, w // 4), replace=False):
                    grid[:, int(c)] = bg
        else:
            if axis in ("rows", "both"):
                for r in rng.choice(h, size=max(1, h // 4), replace=False):
                    grid[int(r), int(rng.integers(0, w))] = condition_color
            if axis in ("columns", "both"):
                for c in rng.choice(w, size=max(1, w // 4), replace=False):
                    grid[int(rng.integers(0, h)), int(c)] = condition_color

        row_keep = np.ones(h, dtype=bool)
        col_keep = np.ones(w, dtype=bool)
        if axis in ("rows", "both"):
            if condition == "all_background":
                row_keep = np.any(grid != bg, axis=1)
            else:
                row_keep = ~np.any(grid == condition_color, axis=1)
        if axis in ("columns", "both"):
            if condition == "all_background":
                col_keep = np.any(grid != bg, axis=0)
            else:
                col_keep = ~np.any(grid == condition_color, axis=0)
        if np.any(row_keep) and np.any(col_keep):
            out = grid[row_keep][:, col_keep]
            return {
                "input": grid_to_list(grid),
                "output": grid_to_list(out),
                "metadata": {
                    "axis": axis,
                    "condition": condition,
                    "condition_color": int(condition_color),
                    "background_color": bg,
                    "order_rule": "preserve the order of remaining rows and columns",
                },
            }
    raise RuntimeError("Task 135: failed to generate valid example")


@register("136")
def generate_row_column_marker_intersections(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=6, h_hi=22)
    bg = int(rng.integers(0, NUM_COLORS))
    row_marker, column_marker, intersection_color = _unused_colors(rng, 3, bg)
    grid = make_grid(h, w, fill=bg)
    row_count = int(rng.integers(1, min(4, h) + 1))
    col_count = int(rng.integers(1, min(4, w) + 1))
    rows = rng.choice(h, size=row_count, replace=False).tolist()
    cols = rng.choice(w, size=col_count, replace=False).tolist()
    for r in rows:
        grid[int(r), int(rng.integers(0, w))] = row_marker
    for c in cols:
        candidates = [r for r in range(h) if grid[r, int(c)] != row_marker]
        grid[int(candidates[int(rng.integers(0, len(candidates)))]), int(c)] = column_marker
    distractor_colors = _add_random_distractor_cells(
        grid,
        rng,
        bg=bg,
        exclude_colors=[row_marker, column_marker, intersection_color],
        density=0.05,
    )
    rows = np.where(np.any(grid == row_marker, axis=1))[0].tolist()
    cols = np.where(np.any(grid == column_marker, axis=0))[0].tolist()
    out = grid.copy()
    for r in rows:
        for c in cols:
            out[int(r), int(c)] = intersection_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "row_marker_color": int(row_marker),
            "column_marker_color": int(column_marker),
            "intersection_color": int(intersection_color),
            "background_color": bg,
            "distractor_colors": distractor_colors,
            "row_rule": "any row containing the row marker",
            "column_rule": "any column containing the column marker",
            "overwrite_rule": "intersection color overwrites whatever is currently in an intersection cell",
        },
    }


@register("137")
def generate_extremal_position_object_recolor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(10, 20))
        w = int(rng.integers(10, 20))
        bg = int(rng.integers(0, NUM_COLORS))
        n_objects = int(rng.integers(3, 7))
        connectivity = int(rng.choice([4, 8]))
        scene = _random_scene_objects(
            rng,
            h=h,
            w=w,
            bg=bg,
            n_objects=n_objects,
            gap=2,
            connectivity=connectivity,
            require_diagonal_object=(connectivity == 8),
        )
        if scene is None:
            continue
        grid, objects = scene
        rule = str(rng.choice(["topmost", "bottommost", "leftmost", "rightmost"]))
        score = {
            "topmost": lambda obj: obj["bbox"][0],
            "bottommost": lambda obj: -obj["bbox"][2],
            "leftmost": lambda obj: obj["bbox"][1],
            "rightmost": lambda obj: -obj["bbox"][3],
        }[rule]
        ranked = sorted(objects, key=lambda obj: (score(obj), obj["bbox"][0], obj["bbox"][1]))
        if len(ranked) > 1 and score(ranked[0]) == score(ranked[1]):
            continue
        new_color = pick_color(rng, [bg, ranked[0]["color"]])
        texture = _add_background_texture(
            grid,
            rng,
            bg=bg,
            exclude_colors=[new_color, *[int(obj["color"]) for obj in objects]],
            protected_cells={cell for obj in objects for cell in obj["cells"]},
            density_scale=0.7,
        )
        # Ensure the chosen target object is the unique most-extremal NON-BACKGROUND
        # connected component in the requested direction, so a solver that does not
        # discount the patterned background still derives the same answer.
        detail_colors = set(texture["colors"])
        target_score = score(ranked[0])
        target_cells = {(int(r), int(c)) for r, c in ranked[0]["cells"]}
        texture_competes = False
        for comp in connected_components(grid, connectivity=connectivity, background=bg):
            if comp["color"] not in detail_colors:
                continue
            comp_obj = {"bbox": comp["bbox"]}
            if score(comp_obj) <= target_score:
                texture_competes = True
                break
        if texture_competes:
            continue
        out = grid.copy()
        for r, c in target_cells:
            out[r, c] = new_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "position_rule": rule,
                "target_color": int(new_color),
                "background_color": bg,
                "connectivity": connectivity,
                "background_style": texture["style"],
                "background_detail_colors": texture["colors"],
                "tie_break": "row-major by top-left bounding-box corner, though generated examples avoid ties",
            },
        }
    raise RuntimeError("Task 137: failed to generate valid example")


@register("138")
def generate_boundary_interior_recolor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, h_lo=9, h_hi=20)
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, boundary_color, interior_color = _unused_colors(rng, 3, bg)
        grid = make_grid(h, w, fill=bg)
        boxes: list[tuple[int, int, int, int]] = []
        for _rect in range(int(rng.integers(2, 5))):
            for _attempt in range(100):
                rh = int(rng.integers(3, 7))
                rw = int(rng.integers(3, 7))
                r0 = int(rng.integers(0, h - rh + 1))
                c0 = int(rng.integers(0, w - rw + 1))
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if not any(_boxes_overlap(box, other, gap=1) for other in boxes):
                    stamp_rectangle(grid, r0, c0, rh, rw, target_color, filled=True)
                    boxes.append(box)
                    break
        if not boxes:
            continue
        distractor_colors = _add_non_target_distractors(
            grid,
            rng,
            bg=bg,
            target_color=target_color,
            placed_boxes=boxes,
            extra_exclude=[boundary_color, interior_color],
        )
        texture = _add_background_texture(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, boundary_color, interior_color, *distractor_colors],
            protected_cells={
                cell
                for comp in connected_components(grid, connectivity=4, background=bg)
                if comp["color"] in {target_color, *distractor_colors}
                for cell in comp["cells"]
            },
            density_scale=0.7,
        )
        out = grid.copy()
        for comp in connected_components(grid, connectivity=4, background=bg):
            if comp["color"] != target_color:
                continue
            cells = comp["cells"]
            for r, c in cells:
                is_boundary = any((r + dr, c + dc) not in cells for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
                out[r, c] = boundary_color if is_boundary else interior_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "target_color": int(target_color),
                "boundary_color": int(boundary_color),
                "interior_color": int(interior_color),
                "background_color": bg,
                "distractor_colors": distractor_colors,
                "background_style": texture["style"],
                "background_detail_colors": texture["colors"],
                "connectivity": 4,
                "boundary_rule": "a target cell with at least one 4-neighbor outside its object",
            },
        }
    raise RuntimeError("Task 138: failed to generate valid example")


@register("145")
def generate_row_column_compaction(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=5, h_hi=24)
    bg = int(rng.integers(0, NUM_COLORS))
    colors = _unused_colors(rng, 5, bg)
    grid = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            if rng.random() < 0.32:
                grid[r, c] = int(rng.choice(colors))
    axis = str(rng.choice(["row", "column"]))
    direction = str(rng.choice(["left", "right"] if axis == "row" else ["top", "bottom"]))
    out = make_grid(h, w, fill=bg)
    if axis == "row":
        for r in range(h):
            vals = [int(v) for v in grid[r, :] if int(v) != bg]
            if direction == "left":
                out[r, : len(vals)] = vals
            else:
                out[r, w - len(vals) :] = vals
    else:
        for c in range(w):
            vals = [int(v) for v in grid[:, c] if int(v) != bg]
            if direction == "top":
                out[: len(vals), c] = vals
            else:
                out[h - len(vals) :, c] = vals
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "axis": axis,
            "direction": direction,
            "background_color": bg,
            "movable_condition": "non-background cells",
            "order_rule": "preserve order within each row or column",
        },
    }


@register("146")
def generate_mark_first_last_occurrences(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=5, h_hi=24)
    bg = int(rng.integers(0, NUM_COLORS))
    target_color, marker_color = _unused_colors(rng, 2, bg)
    grid = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            if rng.random() < 0.25:
                grid[r, c] = target_color
            elif rng.random() < 0.12:
                grid[r, c] = pick_color(rng, [bg, target_color, marker_color])
    axis = str(rng.choice(["row", "column"]))
    occurrence = str(rng.choice(["first", "last"]))
    scan_direction = str(rng.choice(["left_to_right", "right_to_left"] if axis == "row" else ["top_to_bottom", "bottom_to_top"]))
    out = grid.copy()
    if axis == "row":
        order = list(range(w)) if scan_direction == "left_to_right" else list(range(w - 1, -1, -1))
        for r in range(h):
            hits = [(r, c) for c in order if grid[r, c] == target_color]
            if hits:
                rr, cc = hits[0 if occurrence == "first" else -1]
                out[rr, cc] = marker_color
    else:
        order = list(range(h)) if scan_direction == "top_to_bottom" else list(range(h - 1, -1, -1))
        for c in range(w):
            hits = [(r, c) for r in order if grid[r, c] == target_color]
            if hits:
                rr, cc = hits[0 if occurrence == "first" else -1]
                out[rr, cc] = marker_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "axis": axis,
            "occurrence": occurrence,
            "scan_direction": scan_direction,
            "target_color": int(target_color),
            "marker_color": int(marker_color),
            "background_color": bg,
        },
    }


@register("147")
def generate_mark_maximal_runs(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=5, h_hi=24)
    bg = int(rng.integers(0, NUM_COLORS))
    target_color, marker_color = _unused_colors(rng, 2, bg)
    grid = make_grid(h, w, fill=bg)
    axis = str(rng.choice(["horizontal", "vertical"]))
    condition = str(rng.choice(["at_least", "exactly"]))
    length = int(rng.integers(2, 5))
    for r in range(h):
        for c in range(w):
            if rng.random() < 0.28:
                grid[r, c] = target_color
    if axis == "horizontal":
        run_row = int(rng.integers(0, h))
        max_start = w - length
        run_start = int(rng.integers(0, max_start + 1))
        grid[run_row, run_start:run_start + length] = target_color
        if condition == "exactly":
            if run_start > 0:
                grid[run_row, run_start - 1] = bg
            if run_start + length < w:
                grid[run_row, run_start + length] = bg
    else:
        run_col = int(rng.integers(0, w))
        max_start = h - length
        run_start = int(rng.integers(0, max_start + 1))
        grid[run_start:run_start + length, run_col] = target_color
        if condition == "exactly":
            if run_start > 0:
                grid[run_start - 1, run_col] = bg
            if run_start + length < h:
                grid[run_start + length, run_col] = bg
    distractor_colors = _add_random_distractor_cells(
        grid,
        rng,
        bg=bg,
        exclude_colors=[target_color, marker_color],
        density=0.10,
    )
    out = grid.copy()
    outer = h if axis == "horizontal" else w
    inner = w if axis == "horizontal" else h
    for i in range(outer):
        j = 0
        while j < inner:
            r, c = (i, j) if axis == "horizontal" else (j, i)
            if grid[r, c] != target_color:
                j += 1
                continue
            start = j
            while j < inner:
                rr, cc = (i, j) if axis == "horizontal" else (j, i)
                if grid[rr, cc] != target_color:
                    break
                j += 1
            run_len = j - start
            ok = run_len >= length if condition == "at_least" else run_len == length
            if ok:
                for k in range(start, j):
                    rr, cc = (i, k) if axis == "horizontal" else (k, i)
                    out[rr, cc] = marker_color
        # loop continues after maximal run
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "axis": axis,
            "condition": condition,
            "length": length,
            "target_color": int(target_color),
            "marker_color": int(marker_color),
            "background_color": bg,
            "distractor_colors": distractor_colors,
            "run_rule": "maximal contiguous runs along the stated axis",
        },
    }


@register("153")
def generate_swap_marked_rows_columns(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=5, h_hi=24)
    bg = int(rng.integers(0, NUM_COLORS))
    marker_a, marker_b = _unused_colors(rng, 2, bg)
    palette = [bg] + _unused_colors(rng, 4, bg, marker_a, marker_b)
    grid = rng.choice(np.array(palette, dtype=np.uint8), size=(h, w))
    axis = str(rng.choice(["row", "column"]))
    if axis == "row":
        a_idx, b_idx = rng.choice(h, size=2, replace=False).tolist()
        grid[int(a_idx), int(rng.integers(0, w))] = marker_a
        grid[int(b_idx), int(rng.integers(0, w))] = marker_b
        out = grid.copy()
        out[[int(a_idx), int(b_idx)], :] = out[[int(b_idx), int(a_idx)], :]
    else:
        a_idx, b_idx = rng.choice(w, size=2, replace=False).tolist()
        grid[int(rng.integers(0, h)), int(a_idx)] = marker_a
        grid[int(rng.integers(0, h)), int(b_idx)] = marker_b
        out = grid.copy()
        out[:, [int(a_idx), int(b_idx)]] = out[:, [int(b_idx), int(a_idx)]]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "axis": axis,
            "marker_color_a": int(marker_a),
            "marker_color_b": int(marker_b),
            "background_color": bg,
        },
    }


@register("154")
def generate_fill_object_bounding_boxes(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(10, 20))
        w = int(rng.integers(10, 20))
        bg = int(rng.integers(0, NUM_COLORS))
        n_objects = int(rng.integers(3, 7))
        connectivity = int(rng.choice([4, 8]))
        scene = _random_scene_objects(
            rng,
            h=h,
            w=w,
            bg=bg,
            n_objects=n_objects,
            gap=2,
            connectivity=connectivity,
            require_diagonal_object=(connectivity == 8),
        )
        if scene is None:
            continue
        grid, objects = scene
        texture = _add_background_texture(
            grid,
            rng,
            bg=bg,
            exclude_colors=[int(obj["color"]) for obj in objects],
            protected_cells={cell for obj in objects for cell in obj["cells"]},
            density_scale=0.7,
        )
        detail_colors = set(texture["colors"])
        comps = connected_components(grid, connectivity=connectivity, background=bg)
        # The patterned background must never resemble an object: every detail
        # (texture) component must be a thin 1-D run (filling its bbox is a no-op,
        # so it cannot be mistaken for a solid blob) and no detail cell may lie
        # inside an object's bounding box (otherwise the order of bbox filling
        # would matter). Under these constraints the output is identical whether
        # a solver treats the texture as background or fills every component's
        # bbox, so the answer is uniquely derivable.
        object_boxes = [comp["bbox"] for comp in comps if comp["color"] not in detail_colors]
        texture_object_like = False
        for comp in comps:
            if comp["color"] not in detail_colors:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if r1 > r0 and c1 > c0:
                texture_object_like = True
                break
            if any(
                br0 <= r <= br1 and bc0 <= c <= bc1
                for (r, c) in comp["cells"]
                for (br0, bc0, br1, bc1) in object_boxes
            ):
                texture_object_like = True
                break
        if texture_object_like:
            continue
        out = grid.copy()
        for comp in comps:
            if comp["color"] in detail_colors:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = comp["color"]
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "connectivity": connectivity,
                "background_style": texture["style"],
                "background_detail_colors": texture["colors"],
                "target_condition": "all non-background single-color objects",
                "fill_rule": "use each object's original color",
            },
        }
    raise RuntimeError("Task 154: failed to generate valid example")


@register("157")
def generate_extract_specified_diagonal(rng: np.random.Generator) -> dict:
    # Sample H and W independently. The diagonal is anchored at a named corner and
    # stepped one cell at a time until it leaves the grid, so on a rectangular grid
    # it has length min(H, W) -- a single, unambiguous corner-to-edge run. The
    # output is just a thin strip, so the input may use the full ARC size span.
    h, w = _sample_hw(rng, h_lo=4, h_hi=30, square_prob=0.25)
    grid = rng.integers(0, NUM_COLORS, size=(h, w), dtype=np.uint8)
    diag_len = min(h, w)
    diagonal = str(rng.choice(["main", "anti"]))
    if diagonal == "main":
        # Anchor top-left (0,0), step down-right until an edge is hit.
        coords = [(i, i) for i in range(diag_len)]
        directions = ["top_left_to_bottom_right", "bottom_right_to_top_left"]
    else:
        # Anchor top-right (0,W-1), step down-left until an edge is hit.
        coords = [(i, w - 1 - i) for i in range(diag_len)]
        directions = ["top_right_to_bottom_left", "bottom_left_to_top_right"]
    read_direction = str(rng.choice(directions))
    if read_direction in ("bottom_right_to_top_left", "bottom_left_to_top_right"):
        coords = list(reversed(coords))
    values = [int(grid[r, c]) for r, c in coords]
    output_orientation = str(rng.choice(["row", "column"]))
    out = np.array([values], dtype=np.uint8) if output_orientation == "row" else np.array(values, dtype=np.uint8).reshape(-1, 1)
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "diagonal": diagonal,
            "read_direction": read_direction,
            "output_orientation": output_orientation,
            "diagonal_length": diag_len,
        },
    }


@register("158")
def generate_mark_midpoint_between_markers(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, h_lo=6, h_hi=22)
        bg = int(rng.integers(0, NUM_COLORS))
        marker_color, midpoint_color = _unused_colors(rng, 2, bg)
        grid = make_grid(h, w, fill=bg)
        alignment = str(rng.choice(["horizontal", "vertical", "main_diagonal", "anti_diagonal"]))
        if alignment == "horizontal":
            r = int(rng.integers(0, h))
            c1 = int(rng.integers(0, w - 2))
            c2 = int(rng.integers(c1 + 2, w))
            if (c2 - c1) % 2 == 1:
                c2 -= 1
            if c2 <= c1:
                continue
            p1, p2 = (r, c1), (r, c2)
        elif alignment == "vertical":
            c = int(rng.integers(0, w))
            r1 = int(rng.integers(0, h - 2))
            r2 = int(rng.integers(r1 + 2, h))
            if (r2 - r1) % 2 == 1:
                r2 -= 1
            if r2 <= r1:
                continue
            p1, p2 = (r1, c), (r2, c)
        else:
            max_step = min((h - 1) // 2, (w - 1) // 2)
            step = int(rng.integers(1, max_step + 1))
            if alignment == "main_diagonal":
                r1 = int(rng.integers(0, h - 2 * step))
                c1 = int(rng.integers(0, w - 2 * step))
                p1, p2 = (r1, c1), (r1 + 2 * step, c1 + 2 * step)
            else:
                r1 = int(rng.integers(0, h - 2 * step))
                c1 = int(rng.integers(2 * step, w))
                p1, p2 = (r1, c1), (r1 + 2 * step, c1 - 2 * step)
        grid[p1] = marker_color
        grid[p2] = marker_color
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        texture = _add_background_texture(
            grid,
            rng,
            bg=bg,
            exclude_colors=[marker_color, midpoint_color],
            protected_cells={p1, p2, mid},
            density_scale=0.8,
        )
        out = grid.copy()
        out[mid] = midpoint_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "marker_color": int(marker_color),
                "midpoint_color": int(midpoint_color),
                "background_color": bg,
                "alignment": alignment,
                "background_style": texture["style"],
                "background_detail_colors": texture["colors"],
                "midpoint_rule": "average the two marker coordinates; examples always have an integer midpoint",
            },
        }
    raise RuntimeError("Task 158: failed to generate valid example")


@register("162")
def generate_delete_duplicate_rows_columns(rng: np.random.Generator) -> dict:
    h, w = _sample_hw(rng, h_lo=5, h_hi=22)
    grid = rng.integers(0, NUM_COLORS, size=(h, w), dtype=np.uint8)
    axis = str(rng.choice(["rows", "columns"]))
    keep = str(rng.choice(["first", "last"]))
    if axis == "rows":
        dup_from, dup_to = rng.choice(h, size=2, replace=False).tolist()
        grid[int(dup_to), :] = grid[int(dup_from), :]
        seen: set[tuple[int, ...]] = set()
        keep_indices: list[int] = []
        order = range(h) if keep == "first" else range(h - 1, -1, -1)
        for i in order:
            key = tuple(int(v) for v in grid[i, :])
            if key not in seen:
                seen.add(key)
                keep_indices.append(i)
        keep_indices.sort()
        out = grid[keep_indices, :]
    else:
        dup_from, dup_to = rng.choice(w, size=2, replace=False).tolist()
        grid[:, int(dup_to)] = grid[:, int(dup_from)]
        seen = set()
        keep_indices = []
        order = range(w) if keep == "first" else range(w - 1, -1, -1)
        for i in order:
            key = tuple(int(v) for v in grid[:, i])
            if key not in seen:
                seen.add(key)
                keep_indices.append(i)
        keep_indices.sort()
        out = grid[:, keep_indices]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "axis": axis,
            "keep_occurrence": keep,
            "order_rule": "preserve original order of retained rows or columns",
        },
    }


@register("164")
def generate_pad_grid_borders(rng: np.random.Generator) -> dict:
    # Output grows by 2*thickness per padded side, so cap the input below the
    # grid->grid budget so the padded output still fits the token envelope.
    h, w = _sample_hw(rng, h_lo=3, h_hi=22)
    grid = rng.integers(0, NUM_COLORS, size=(h, w), dtype=np.uint8)
    padding_color = int(rng.integers(0, NUM_COLORS))
    thickness = int(rng.integers(1, 4))
    sides = str(rng.choice(["all", "top", "bottom", "left", "right", "top_bottom", "left_right"]))
    pad = {
        "all": ((thickness, thickness), (thickness, thickness)),
        "top": ((thickness, 0), (0, 0)),
        "bottom": ((0, thickness), (0, 0)),
        "left": ((0, 0), (thickness, 0)),
        "right": ((0, 0), (0, thickness)),
        "top_bottom": ((thickness, thickness), (0, 0)),
        "left_right": ((0, 0), (thickness, thickness)),
    }[sides]
    out = np.pad(grid, pad, constant_values=padding_color)
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "sides": sides,
            "thickness": thickness,
            "padding_color": padding_color,
        },
    }
