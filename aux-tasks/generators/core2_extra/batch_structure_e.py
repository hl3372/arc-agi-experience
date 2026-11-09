"""Structure-oriented second-batch task generators."""
from __future__ import annotations

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from common.objects import connected_components, random_shape
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    available_colors,
    place_shape,
    random_scene_objects,
    unused_colors,
)
from registry import register


def _minimal_periods(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_period = h
    col_period = w
    for p in range(1, h + 1):
        if h % p == 0 and np.array_equal(grid, np.tile(grid[:p, :], (h // p, 1))):
            row_period = p
            break
    for p in range(1, w + 1):
        if w % p == 0 and np.array_equal(grid, np.tile(grid[:, :p], (1, w // p))):
            col_period = p
            break
    return row_period, col_period


def _primitive_block(
    rng: np.random.Generator,
    height: int,
    width: int,
    colors: list[int],
) -> np.ndarray:
    for _ in range(200):
        block = np.array(rng.choice(colors, size=(height, width), replace=True), dtype=np.uint8)
        if _minimal_periods(block) == (height, width):
            return block
    block = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            block[r, c] = colors[(r * width + c) % len(colors)]
    block[-1, -1] = colors[-1]
    return block


def _classes_for_period(mode: str, shape: tuple[int, int], meta: dict) -> list[list[tuple[int, int]]]:
    h, w = shape
    if mode == "row_wise":
        p = meta["period_length"]
        return [[(r, c) for c in range(res, w, p)] for r in range(h) for res in range(p)]
    if mode == "column_wise":
        p = meta["period_length"]
        return [[(r, c) for r in range(res, h, p)] for c in range(w) for res in range(p)]
    th, tw = meta["tile_height"], meta["tile_width"]
    return [
        [(r, c) for r in range(rr, h, th) for c in range(cc, w, tw)]
        for rr in range(th)
        for cc in range(tw)
    ]


def _mask_periodic_cells(
    rng: np.random.Generator,
    full: np.ndarray,
    unknown_color: int,
    mode: str,
    meta: dict,
) -> np.ndarray:
    grid = full.copy()
    masked = 0
    masked_classes = 0
    for cls in _classes_for_period(mode, full.shape, meta):
        if len(cls) < 3:
            continue
        max_mask = len(cls) - 2
        count = int(rng.integers(0, min(3, max_mask) + 1))
        if count == 0:
            continue
        chosen = rng.choice(len(cls), size=count, replace=False)
        for idx in chosen:
            r, c = cls[int(idx)]
            grid[r, c] = unknown_color
            masked += 1
        masked_classes += 1
    eligible = [cls for cls in _classes_for_period(mode, full.shape, meta) if len(cls) >= 3]
    while masked == 0 or (len(eligible) > 1 and masked_classes < 2):
        unmasked_classes = [cls for cls in eligible if all(grid[r, c] != unknown_color for r, c in cls)]
        choices = unmasked_classes or eligible
        cls = choices[int(rng.integers(0, len(choices)))]
        candidates = [(r, c) for r, c in cls if grid[r, c] != unknown_color]
        if len(candidates) <= 2:
            break
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        grid[r, c] = unknown_color
        masked += 1
        masked_classes += 1
    return grid


def _pattern_cells(pattern_type: str, h: int, w: int, meta: dict) -> set[tuple[int, int]]:
    spacing = meta["spacing"]
    if pattern_type == "vertical_stripes":
        return {(r, c) for r in range(h) for c in range(meta["column_offset"], w, spacing)}
    if pattern_type == "horizontal_stripes":
        return {(r, c) for r in range(meta["row_offset"], h, spacing) for c in range(w)}
    if pattern_type == "rectangular_lattice":
        rows = set(range(meta["row_offset"], h, spacing))
        cols = set(range(meta["column_offset"], w, spacing))
        return {(r, c) for r in range(h) for c in range(w) if r in rows or c in cols}
    if meta["diagonal_direction"] == "down_right":
        return {(r, c) for r in range(h) for c in range(w) if (r - c - meta["diagonal_offset"]) % spacing == 0}
    return {(r, c) for r in range(h) for c in range(w) if (r + c - meta["diagonal_offset"]) % spacing == 0}


def _interface_cells(grid: np.ndarray, meta: dict) -> set[tuple[int, int]]:
    a = meta["object_a_color"]
    b = meta["object_b_color"]
    a_cells = {tuple(p) for p in np.argwhere(grid == a)}
    b_cells = {tuple(p) for p in np.argwhere(grid == b)}
    a_touching = set()
    b_touching = set()
    for r, c in a_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in b_cells:
                a_touching.add((r, c))
                b_touching.add((r + dr, c + dc))
    convention = meta["contact_convention"]
    if convention == "object_a_side_cells":
        return a_touching
    if convention == "object_b_side_cells":
        return b_touching
    return a_touching | b_touching


def _run_representatives(start: int, end: int, rule: str) -> set[int]:
    if rule == "first":
        return {start}
    if rule == "last":
        return {end - 1}
    if rule == "endpoints":
        return {start, end - 1}
    return {(start + end - 1) // 2}


def _replace_runs(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    target = meta["target_color"]
    marker = meta["marker_color"]
    bg = meta["background_color"]
    axis = meta["axis"]
    outer = grid.shape[0] if axis == "horizontal" else grid.shape[1]
    inner = grid.shape[1] if axis == "horizontal" else grid.shape[0]
    for i in range(outer):
        j = 0
        while j < inner:
            r, c = (i, j) if axis == "horizontal" else (j, i)
            if grid[r, c] != target:
                j += 1
                continue
            start = j
            while j < inner:
                rr, cc = (i, j) if axis == "horizontal" else (j, i)
                if grid[rr, cc] != target:
                    break
                j += 1
            reps = _run_representatives(start, j, meta["representative_rule"])
            for k in range(start, j):
                rr, cc = (i, k) if axis == "horizontal" else (k, i)
                out[rr, cc] = marker if k in reps else bg
    return out


def _pair_distance(a: tuple[int, int], b: tuple[int, int], metric: str) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc if metric == "manhattan" else max(dr, dc)


def _distance_marks(grid: np.ndarray, meta: dict) -> set[tuple[int, int]]:
    a_cells = sorted(map(tuple, np.argwhere(grid == meta["object_a_color"])))
    b_cells = sorted(map(tuple, np.argwhere(grid == meta["object_b_color"])))
    pairs = [
        (a, b, _pair_distance(a, b, meta["distance_metric"]))
        for a in a_cells
        for b in b_cells
    ]
    min_dist = min(d for _, _, d in pairs)
    winners = sorted((a, b) for a, b, d in pairs if d == min_dist)
    policy = meta["tie_policy"]
    if policy == "lexicographically_first_pair":
        a, b = winners[0]
        return {a, b}
    if policy == "object_a_minimizers":
        return {a for a, _ in winners}
    if policy == "object_b_minimizers":
        return {b for _, b in winners}
    return {p for pair in winners for p in pair}


def _directional_boundary(cells: set[tuple[int, int]], direction: str) -> set[tuple[int, int]]:
    delta = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]
    dr, dc = delta
    return {(r, c) for r, c in cells if (r + dr, c + dc) not in cells}


def _line_counts(grid: np.ndarray, meta: dict) -> list[int]:
    axis = meta["axis"]
    count_rule = meta["count_rule"]
    lines = [grid[r, :] for r in range(grid.shape[0])] if axis == "row" else [grid[:, c] for c in range(grid.shape[1])]
    counts = []
    for line in lines:
        if count_rule == "target_color_count":
            counts.append(int(np.count_nonzero(line == meta["target_color"])))
        elif count_rule == "non_background_count":
            counts.append(int(np.count_nonzero(line != meta["background_color"])))
        else:
            counts.append(len(set(int(v) for v in line)))
    return counts


def _select_line(grid: np.ndarray, meta: dict) -> int:
    counts = _line_counts(grid, meta)
    best = max(counts) if meta["extremum"] == "maximum" else min(counts)
    winners = [i for i, count in enumerate(counts) if count == best]
    return winners[0] if meta["tie_policy"] == "first" else winners[-1]


@register("116")
def generate_smallest_repeating_tile(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        tile_h = int(rng.choice([2, 3, 4]))
        tile_w = int(rng.choice([2, 3, 4]))
        repeats_r = int(rng.integers(2, 5))
        repeats_c = int(rng.integers(2, 5))
        colors = unused_colors(rng, int(rng.integers(2, 5)))
        tile = _primitive_block(rng, tile_h, tile_w, colors)
        grid = np.tile(tile, (repeats_r, repeats_c))
        if _minimal_periods(grid) != (tile_h, tile_w):
            continue
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(tile),
            "metadata": {
                "tile_height": tile_h,
                "tile_width": tile_w,
                "repeat_rows": repeats_r,
                "repeat_columns": repeats_c,
                "selection_rule": "smallest_rectangular_translation_tile",
            },
        }
    raise RuntimeError("Task 116: failed to generate valid example")


@register("117")
def generate_fill_periodic_unknowns(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        mode = str(rng.choice(["row_wise", "column_wise", "full_grid"]))
        unknown = int(rng.integers(0, NUM_COLORS))
        colors = unused_colors(rng, int(rng.integers(2, 5)), unknown)
        meta = {"period_mode": mode, "unknown_color": unknown}
        if mode == "row_wise":
            period = int(rng.choice([2, 3, 4]))
            h = int(rng.choice([5, 7, 8, 9]))
            w = period * int(rng.integers(3, 6))
            rows = [_primitive_block(rng, 1, period, colors)[0] for _ in range(h)]
            full = np.vstack([np.tile(row, int(w // period)) for row in rows]).astype(np.uint8)
            if _minimal_periods(full) != (h, period):
                continue
            meta["period_length"] = period
        elif mode == "column_wise":
            period = int(rng.choice([2, 3, 4]))
            h = period * int(rng.integers(3, 6))
            w = int(rng.choice([5, 7, 8, 9]))
            cols = [_primitive_block(rng, period, 1, colors)[:, 0] for _ in range(w)]
            full = np.column_stack([np.tile(col, int(h // period)) for col in cols]).astype(np.uint8)
            if _minimal_periods(full) != (period, w):
                continue
            meta["period_length"] = period
        else:
            th = int(rng.choice([2, 3]))
            tw = int(rng.choice([2, 3, 4]))
            tile = _primitive_block(rng, th, tw, colors)
            full = np.tile(tile, (int(rng.integers(3, 5)), int(rng.integers(3, 5))))
            if _minimal_periods(full) != (th, tw):
                continue
            meta["tile_height"] = th
            meta["tile_width"] = tw
        grid = _mask_periodic_cells(rng, full, unknown, mode, meta)
        return {"input": grid_to_list(grid), "output": grid_to_list(full), "metadata": meta}
    raise RuntimeError("Task 117: failed to generate valid example")


@register("119")
def generate_repair_stripe_lattice(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 17))
    w = int(rng.integers(6, 17))
    bg, pattern = unused_colors(rng, 2)
    pattern_type = str(rng.choice([
        "vertical_stripes",
        "horizontal_stripes",
        "rectangular_lattice",
        "diagonal_stripes",
    ]))
    spacing = int(rng.choice([3, 4]))
    meta = {
        "background_color": bg,
        "pattern_color": pattern,
        "pattern_type": pattern_type,
        "spacing": spacing,
    }
    if pattern_type in ("vertical_stripes", "rectangular_lattice"):
        meta["column_offset"] = int(rng.integers(0, spacing))
    if pattern_type in ("horizontal_stripes", "rectangular_lattice"):
        meta["row_offset"] = int(rng.integers(0, spacing))
    if pattern_type == "diagonal_stripes":
        meta["diagonal_direction"] = str(rng.choice(["down_right", "down_left"]))
        meta["diagonal_offset"] = int(rng.integers(0, spacing))
    cells = sorted(_pattern_cells(pattern_type, h, w, meta))
    full = make_grid(h, w, fill=bg)
    for r, c in cells:
        full[r, c] = pattern
    grid = full.copy()
    n_missing = max(2, min(len(cells) // 4, int(rng.integers(3, 8))))
    for idx in rng.choice(len(cells), size=n_missing, replace=False):
        r, c = cells[int(idx)]
        grid[r, c] = bg
    meta["missing_color"] = bg
    return {"input": grid_to_list(grid), "output": grid_to_list(full), "metadata": meta}


@register("155")
def generate_mark_contact_interface(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 17))
    w = int(rng.integers(6, 17))
    bg, color_a, color_b, contact = unused_colors(rng, 4)
    grid = make_grid(h, w, fill=bg)
    placed_boxes: list[tuple[int, int, int, int]] = []
    orientation = str(rng.choice(["horizontal", "vertical"]))
    contact_style = "full_side"
    if orientation == "horizontal":
        rh = int(rng.integers(3, min(6, h) + 1))
        aw = int(rng.integers(2, 4))
        bw = int(rng.integers(2, 4))
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - aw - bw + 1))
        grid[r0:r0 + rh, c0:c0 + aw] = color_a
        grid[r0:r0 + rh, c0 + aw:c0 + aw + bw] = color_b
        placed_boxes.append((r0, c0, r0 + rh - 1, c0 + aw + bw - 1))
        if rh > 2 and rng.random() < 0.7:
            contact_style = "partial_staggered"
            open_rows = rng.choice(np.arange(r0, r0 + rh), size=int(rng.integers(1, rh)), replace=False)
            for row in open_rows:
                if rng.random() < 0.5:
                    grid[int(row), c0 + aw - 1] = bg
                else:
                    grid[int(row), c0 + aw] = bg
    else:
        ah = int(rng.integers(2, 4))
        bh = int(rng.integers(2, 4))
        rw = int(rng.integers(3, min(6, w) + 1))
        r0 = int(rng.integers(0, h - ah - bh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        grid[r0:r0 + ah, c0:c0 + rw] = color_a
        grid[r0 + ah:r0 + ah + bh, c0:c0 + rw] = color_b
        placed_boxes.append((r0, c0, r0 + ah + bh - 1, c0 + rw - 1))
        if rw > 2 and rng.random() < 0.7:
            contact_style = "partial_staggered"
            open_cols = rng.choice(np.arange(c0, c0 + rw), size=int(rng.integers(1, rw)), replace=False)
            for col in open_cols:
                if rng.random() < 0.5:
                    grid[r0 + ah - 1, int(col)] = bg
                else:
                    grid[r0 + ah, int(col)] = bg
    # Distractor objects in colors other than A/B/contact: vary the component count
    # without affecting the A-B contact interface the rule depends on.
    distractor_palette = available_colors(bg, color_a, color_b, contact)
    n_distract = int(rng.integers(0, 4))
    for _ in range(n_distract):
        if not distractor_palette:
            break
        dcolor = int(rng.choice(distractor_palette))
        shape = random_shape(rng, (2, 5), connectivity=4)
        place_shape(grid, rng, shape, dcolor, bg, placed_boxes, gap=1)
    meta = {
        "background_color": bg,
        "object_a_color": color_a,
        "object_b_color": color_b,
        "contact_color": contact,
        "contact_convention": str(rng.choice([
            "object_a_side_cells",
            "object_b_side_cells",
            "both_objects_side_cells",
        ])),
        "output_mode": str(rng.choice(["preserve_scene", "contact_mask"])),
        "adjacency": "4-neighbor",
        "contact_style": contact_style,
    }
    # Sparse patterned background in non-semantic colors; protect every object cell so the
    # A/B contact geometry is untouched. The mask output ignores it (it is rebuilt on solid bg).
    bg_meta = add_patterned_background(
        grid,
        rng,
        bg=bg,
        exclude_colors=[color_a, color_b, contact],
        protected_boxes=placed_boxes,
        solid_probability=0.5,
    )
    meta["background_style"] = bg_meta["style"]
    out = grid.copy() if meta["output_mode"] == "preserve_scene" else make_grid(h, w, fill=bg)
    for r, c in _interface_cells(grid, meta):
        out[r, c] = contact
    return {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}


@register("168")
def generate_replace_runs_with_representatives(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 17))
    w = int(rng.integers(6, 17))
    bg, target, marker = unused_colors(rng, 3)
    grid = make_grid(h, w, fill=bg)
    axis = str(rng.choice(["horizontal", "vertical"]))
    protected: set[tuple[int, int]] = set()
    if axis == "horizontal":
        available_rows = np.arange(0, h, 2)
        rows = rng.choice(available_rows, size=int(rng.integers(1, min(5, len(available_rows)) + 1)), replace=False)
        for r in rows:
            start = int(rng.integers(0, w - 3))
            length = int(rng.integers(2, min(6, w - start) + 1))
            grid[int(r), start:start + length] = target
            # Protect the whole line so background texture cannot split or extend the run.
            protected.update((int(r), c) for c in range(w))
    else:
        available_cols = np.arange(0, w, 2)
        cols = rng.choice(available_cols, size=int(rng.integers(1, min(5, len(available_cols)) + 1)), replace=False)
        for c in cols:
            start = int(rng.integers(0, h - 3))
            length = int(rng.integers(2, min(6, h - start) + 1))
            grid[start:start + length, int(c)] = target
            protected.update((r, int(c)) for r in range(h))
    meta = {
        "background_color": bg,
        "target_color": target,
        "marker_color": marker,
        "axis": axis,
        "run_count": int(len(rows) if axis == "horizontal" else len(cols)),
        "representative_rule": str(rng.choice(["first", "last", "middle", "endpoints"])),
    }
    # Patterned background lives only off the run lines (those lines are protected so texture
    # cannot split or extend a run). The output is computed from the textured input, so the
    # texture cells round-trip unchanged and only run cells differ.
    bg_meta = add_patterned_background(
        grid,
        rng,
        bg=bg,
        exclude_colors=[target, marker],
        protected_cells=protected,
        solid_probability=0.5,
    )
    meta["background_style"] = bg_meta["style"]
    out = _replace_runs(grid, meta)
    return {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}


@register("169")
def generate_mark_closest_object_cells(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(7, 17))
        w = int(rng.integers(7, 17))
        bg, color_a, color_b, marker = unused_colors(rng, 4)
        grid = make_grid(h, w, fill=bg)
        ah = int(rng.integers(2, 4))
        aw = int(rng.integers(2, 4))
        bh = int(rng.integers(2, 4))
        bw = int(rng.integers(2, 4))
        layout = str(rng.choice(["right", "below", "diagonal"]))
        # Gap is clamped to whatever the (independently sized) grid can fit.
        if layout == "right":
            max_gap = min(4, w - aw - bw - 1)
        elif layout == "below":
            max_gap = min(4, h - ah - bh - 1)
        else:
            max_gap = min(4, h - ah - bh - 1, w - aw - bw - 1)
        if max_gap < 2:
            continue
        gap = int(rng.integers(2, max_gap + 1))
        placed_boxes: list[tuple[int, int, int, int]] = []
        if layout == "right":
            if h - max(ah, bh) - 2 <= 2:
                continue
            r0 = int(rng.integers(2, h - max(ah, bh) - 2))
            c0 = int(rng.integers(0, w - aw - bw - gap + 1))
            br0 = min(h - bh - 1, max(1, r0 + int(rng.choice([-1, 0, 1]))))
            bc0 = c0 + aw + gap
        elif layout == "below":
            if w - max(aw, bw) - 2 <= 2:
                continue
            r0 = int(rng.integers(0, h - ah - bh - gap + 1))
            c0 = int(rng.integers(2, w - max(aw, bw) - 2))
            br0 = r0 + ah + gap
            bc0 = min(w - bw - 1, max(1, c0 + int(rng.choice([-1, 0, 1]))))
        else:
            r0 = int(rng.integers(0, h - ah - bh - gap + 1))
            c0 = int(rng.integers(0, w - aw - bw - gap + 1))
            br0 = r0 + ah + gap
            bc0 = c0 + aw + gap
        grid[r0:r0 + ah, c0:c0 + aw] = color_a
        grid[br0:br0 + bh, bc0:bc0 + bw] = color_b
        placed_boxes.append((r0, c0, r0 + ah - 1, c0 + aw - 1))
        placed_boxes.append((br0, bc0, br0 + bh - 1, bc0 + bw - 1))
        shape_variant = "rectangles"
        if rng.random() < 0.7:
            shape_variant = "corner_notched"
            grid[r0, c0] = bg
            grid[br0 + bh - 1, bc0 + bw - 1] = bg
        # Distractor objects (other colors): vary the component count; they never lie between
        # A and B at the minimum distance because they are placed with a gap from both boxes,
        # and the rule measures only A-to-B distances.
        distractor_palette = available_colors(bg, color_a, color_b, marker)
        for _ in range(int(rng.integers(0, 4))):
            if not distractor_palette:
                break
            dcolor = int(rng.choice(distractor_palette))
            shape = random_shape(rng, (2, 5), connectivity=4)
            place_shape(grid, rng, shape, dcolor, bg, placed_boxes, gap=1)
        meta = {
            "background_color": bg,
            "object_a_color": color_a,
            "object_b_color": color_b,
            "marker_color": marker,
            "distance_metric": str(rng.choice(["manhattan", "chebyshev"])),
            "shape_variant": shape_variant,
            "tie_policy": str(rng.choice([
                "all_minimizing_cells",
                "lexicographically_first_pair",
                "object_a_minimizers",
                "object_b_minimizers",
            ])),
        }
        bg_meta = add_patterned_background(
            grid,
            rng,
            bg=bg,
            exclude_colors=[color_a, color_b, marker],
            protected_boxes=placed_boxes,
            solid_probability=0.5,
        )
        meta["background_style"] = bg_meta["style"]
        out = grid.copy()
        for r, c in _distance_marks(grid, meta):
            out[r, c] = marker
        return {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}
    raise RuntimeError("Task 169: failed to generate valid example")


@register("174")
def generate_mark_missing_rectangle_corner(rng: np.random.Generator) -> dict:
    h = int(rng.integers(5, 17))
    w = int(rng.integers(5, 17))
    bg, corner = unused_colors(rng, 2)
    r0 = int(rng.integers(0, h - 4))
    r1 = int(rng.integers(r0 + 3, h))
    c0 = int(rng.integers(0, w - 4))
    c1 = int(rng.integers(c0 + 3, w))
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    corner_set = set(corners)
    missing = corners[int(rng.integers(0, 4))]
    grid = make_grid(h, w, fill=bg)
    for r, c in corners:
        if (r, c) != missing:
            grid[r, c] = corner
    # Distractor cells in non-corner colors: vary element count / shape vocabulary
    # without touching the corner-color cells the rule depends on.
    distractor_palette = available_colors(bg, corner)
    n_distract = int(rng.integers(0, 6))
    if distractor_palette and n_distract:
        for _ in range(n_distract * 3):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if (r, c) in corner_set or grid[r, c] != bg:
                continue
            grid[r, c] = int(rng.choice(distractor_palette))
            n_distract -= 1
            if n_distract == 0:
                break
    # Optional sparse patterned background (never the corner color); protect corner cells.
    bg_meta = add_patterned_background(
        grid,
        rng,
        bg=bg,
        exclude_colors=[corner],
        protected_cells=corner_set | {missing},
        solid_probability=0.5,
    )
    out = grid.copy()
    out[missing] = corner
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "corner_color": corner,
            "rectangle_type": "axis_aligned",
            "preserve_existing_corners": True,
            "background_style": bg_meta["style"],
        },
    }


@register("177")
def generate_mark_directional_boundaries(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(7, 17))
        w = int(rng.integers(7, 17))
        bg, target, edge = unused_colors(rng, 3)
        scene = random_scene_objects(
            rng,
            h=h,
            w=w,
            bg=bg,
            n_objects=int(rng.integers(2, 6)),
            same_color=target,
            min_cells=4,
            max_cells=8,
            gap=2,
        )
        if scene is None:
            continue
        grid, objects = scene
        # Sparse patterned background in non-target / non-edge colors. The rule only ever looks
        # at target-color components (it filters on comp["color"] == target), so these texture
        # cells are recorded and ignored; protect the target object cells from being overwritten.
        protected = {cell for obj in objects for cell in obj["cells"]}
        bg_meta = add_patterned_background(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target, edge],
            protected_cells=protected,
            solid_probability=0.5,
        )
        direction = str(rng.choice(["top", "bottom", "left", "right"]))
        out = grid.copy()
        for comp in connected_components(grid, connectivity=4, background=bg):
            if comp["color"] != target:
                continue
            for r, c in _directional_boundary(comp["cells"], direction):
                out[r, c] = edge
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target,
                "edge_color": edge,
                "direction": direction,
                "connectivity": 4,
                "background_style": bg_meta["style"],
            },
        }
    raise RuntimeError("Task 177: failed to generate valid example")


@register("181")
def generate_recolor_trigger_bands(rng: np.random.Generator) -> dict:
    h = int(rng.integers(6, 17))
    w = int(rng.integers(6, 17))
    bg, trigger, band, filler = unused_colors(rng, 4)
    grid = make_grid(h, w, fill=bg)
    axis = str(rng.choice(["rows", "columns"]))
    n_lines = int(rng.integers(2, 5))
    line_count = h if axis == "rows" else w
    chosen = rng.choice(line_count, size=min(n_lines, line_count), replace=False)
    for line in chosen:
        if axis == "rows":
            trigger_col = int(rng.integers(0, w))
            grid[int(line), trigger_col] = trigger
            if rng.random() < 0.7:
                choices = [c for c in range(w) if c != trigger_col]
                grid[int(line), choices[int(rng.integers(0, len(choices)))]] = filler
        else:
            trigger_row = int(rng.integers(0, h))
            grid[trigger_row, int(line)] = trigger
            if rng.random() < 0.7:
                choices = [r for r in range(h) if r != trigger_row]
                grid[choices[int(rng.integers(0, len(choices)))], int(line)] = filler
    for _ in range(int(rng.integers(3, 8))):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if grid[r, c] == bg:
            grid[r, c] = filler
    # Patterned background in non-trigger / non-band colors. Selection depends only on whether a
    # line contains a trigger cell, so texture never changes which lines are selected; selected
    # lines are fully overwritten with the band color and non-selected lines keep their detail.
    bg_meta = add_patterned_background(
        grid,
        rng,
        bg=bg,
        exclude_colors=[trigger, band],
        solid_probability=0.5,
    )
    out = grid.copy()
    if axis == "rows":
        out[np.any(grid == trigger, axis=1), :] = band
    else:
        out[:, np.any(grid == trigger, axis=0)] = band
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "trigger_color": trigger,
            "band_color": band,
            "axis": axis,
            "overwrite_rule": "recolor_entire_selected_band",
            "background_style": bg_meta["style"],
        },
    }


@register("182")
def generate_select_extreme_count_line(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(6, 17))
        w = int(rng.integers(6, 17))
        bg, target, highlight, extra1, extra2, extra3 = unused_colors(rng, 6)
        grid = make_grid(h, w, fill=bg)
        axis = str(rng.choice(["row", "column"]))
        extremum = str(rng.choice(["maximum", "minimum"]))
        count_rule = str(rng.choice(["target_color_count", "non_background_count", "distinct_color_count"]))
        line_count = h if axis == "row" else w
        line_len = w if axis == "row" else h
        selected = int(rng.integers(0, line_count))
        palette = [target, extra1, extra2, extra3]
        for i in range(line_count):
            if count_rule == "distinct_color_count":
                distinct = 4 if (i == selected and extremum == "maximum") else 1 if i == selected else 2
                if extremum == "minimum" and i != selected:
                    distinct = 3
                values = ([bg] if distinct == 1 else palette[:distinct])
                cells = [values[j % len(values)] for j in range(line_len)]
            else:
                if i == selected:
                    count = min(line_len - 1, 5) if extremum == "maximum" else 0
                else:
                    count = int(rng.integers(0, min(3, line_len))) if extremum == "maximum" else int(rng.integers(2, min(6, line_len) + 1))
                cells = [bg] * line_len
                if count:
                    positions = rng.choice(line_len, size=count, replace=False)
                    for pos in positions:
                        cells[int(pos)] = target if count_rule == "target_color_count" else palette[int(rng.integers(0, len(palette)))]
            if axis == "row":
                grid[i, :] = cells
            else:
                grid[:, i] = cells
        meta = {
            "background_color": bg,
            "target_color": target,
            "highlight_color": highlight,
            "axis": axis,
            "extremum": extremum,
            "count_rule": count_rule,
            "output_mode": str(rng.choice(["recolor_entire_line", "mark_line_endpoints"])),
            "tie_policy": "first",
        }
        counts = _line_counts(grid, meta)
        best = max(counts) if extremum == "maximum" else min(counts)
        if counts.count(best) != 1 or counts[selected] != best:
            continue
        chosen = _select_line(grid, meta)
        out = grid.copy()
        if axis == "row":
            if meta["output_mode"] == "recolor_entire_line":
                out[chosen, :] = highlight
            else:
                out[chosen, 0] = highlight
                out[chosen, -1] = highlight
        else:
            if meta["output_mode"] == "recolor_entire_line":
                out[:, chosen] = highlight
            else:
                out[0, chosen] = highlight
                out[-1, chosen] = highlight
        return {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}
    raise RuntimeError("Task 182: failed to generate valid example")
