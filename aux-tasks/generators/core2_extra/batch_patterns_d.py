"""Additional pattern-oriented second-batch generators."""
from __future__ import annotations

from collections import Counter

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid, pick_colors
from common.objects import connected_components
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    boxes_overlap,
    unused_colors,
)
from registry import register


def _choice_int(rng: np.random.Generator, values: list[int]) -> int:
    return int(values[int(rng.integers(0, len(values)))])


def _sample_hw(
    rng: np.random.Generator,
    h_lo: int,
    h_hi: int,
    w_lo: int,
    w_hi: int,
    square_prob: float = 0.35,
) -> tuple[int, int]:
    """Sample H and W independently across [lo, hi] (inclusive), but with
    ``square_prob`` snap to a square grid where feasible.

    ARC grids are ~62% square yet span the full 1..30 range on each side, so we
    decorrelate H from W (kills hw_coupled / grid_size_narrow) while preserving a
    realistic square fraction instead of forcing H == W.
    """
    h = int(rng.integers(h_lo, h_hi + 1))
    w = int(rng.integers(w_lo, w_hi + 1))
    if rng.random() < square_prob:
        lo = max(h_lo, w_lo)
        hi = min(h_hi, w_hi)
        if lo <= hi:
            side = int(rng.integers(lo, hi + 1))
            return side, side
    return h, w


def _pattern_matches(grid: np.ndarray, motif: np.ndarray, r0: int, c0: int) -> bool:
    mh, mw = motif.shape
    return bool(np.array_equal(grid[r0 : r0 + mh, c0 : c0 + mw], motif))


def _replace_motif(grid: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = grid.copy()
    mh, mw = source.shape
    hits = [
        (r, c)
        for r in range(grid.shape[0] - mh + 1)
        for c in range(grid.shape[1] - mw + 1)
        if _pattern_matches(grid, source, r, c)
    ]
    for r, c in hits:
        out[r : r + mh, c : c + mw] = target
    return out


def _make_random_core(rng: np.random.Generator, h: int, w: int, colors: list[int]) -> np.ndarray:
    return np.array(rng.choice(colors, size=(h, w), replace=True), dtype=np.uint8)


def _summary_line(values: list[int], meta: dict) -> int:
    rule = meta["summary_rule"]
    if rule == "majority_color":
        counts = Counter(values)
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        if len(winners) == 1:
            return int(winners[0])
        return int(meta["background_color"])
    if rule == "first_non_background":
        bg = int(meta["background_color"])
        for value in values:
            if int(value) != bg:
                return int(value)
        return bg
    if rule == "target_presence":
        return int(meta["true_color"] if int(meta["target_color"]) in values else meta["false_color"])
    raise ValueError(f"unknown summary rule: {rule}")


def _fill_between_same_object(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    bg = meta["background_color"]
    target = meta["target_color"]
    fill = meta["fill_color"]
    axis = meta["axis"]
    for comp in connected_components(grid, connectivity=meta.get("connectivity", 4), background=bg):
        if comp["color"] != target:
            continue
        cells = comp["cells"]
        if axis in ("rows", "both"):
            rows = sorted({r for r, _ in cells})
            for r in rows:
                cols = sorted(c for rr, c in cells if rr == r)
                if len(cols) >= 2:
                    for c in range(cols[0] + 1, cols[-1]):
                        if out[r, c] == bg:
                            out[r, c] = fill
        if axis in ("columns", "both"):
            cols = sorted({c for _, c in cells})
            for c in cols:
                rows = sorted(r for r, cc in cells if cc == c)
                if len(rows) >= 2:
                    for r in range(rows[0] + 1, rows[-1]):
                        if out[r, c] == bg:
                            out[r, c] = fill
    return out


@register("131")
def generate_replace_local_motif(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        a, b, x, y, noise = unused_colors(rng, 5, bg)
        motif_options = [
            ([[a, b], [b, a]], [[x, x], [y, x]]),
            ([[a, a, b], [b, a, b]], [[x, y, x], [y, y, x]]),
            ([[a, b], [a, a], [b, a]], [[x, y], [y, x], [x, x]]),
        ]
        source_rows, target_rows = motif_options[int(rng.integers(0, len(motif_options)))]
        source = np.array(source_rows, dtype=np.uint8)
        target = np.array(target_rows, dtype=np.uint8)
        mh, mw = source.shape
        # Independent H/W across a wide envelope: down toward the room needed for
        # two separated copies, up toward the ARC token-budget cap. Square-biased.
        h, w = _sample_hw(rng, 2 * mh + 1, 18, 2 * mw + 1, 18)
        grid = make_grid(h, w, fill=bg)

        n_hits = int(rng.integers(2, 6))
        boxes: list[tuple[int, int, int, int]] = []
        for _hit in range(n_hits):
            for _attempt in range(200):
                r0 = int(rng.integers(0, h - mh + 1))
                c0 = int(rng.integers(0, w - mw + 1))
                box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
                if any(
                    not (box[2] + 1 < other[0] or other[2] + 1 < box[0] or box[3] + 1 < other[1] or other[3] + 1 < box[1])
                    for other in boxes
                ):
                    continue
                grid[r0 : r0 + mh, c0 : c0 + mw] = source
                boxes.append(box)
                break
            else:
                break
        if len(boxes) < 2:
            continue

        for _noise in range(int(rng.integers(3, 8))):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if all(not (r0 <= r <= r1 and c0 <= c <= c1) for r0, c0, r1, c1 in boxes):
                grid[r, c] = noise

        # Optional sparse background detail. The match rule looks for exact source
        # motifs (built only from colors a/b); texture in any other color can never
        # form a source window, and we protect the placed-motif boxes outright.
        bg_info = add_patterned_background(
            grid,
            rng,
            bg=bg,
            exclude_colors=[a, b, x, y, noise],
            protected_boxes=boxes,
            solid_probability=0.4,
        )

        out = _replace_motif(grid, source, target)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "source_motif": grid_to_list(source),
                "target_motif": grid_to_list(target),
                "match_rule": "exact_rectangular_motif",
                "application": "simultaneous",
                "background_style": bg_info["style"],
            },
        }
    raise RuntimeError("Task 131: failed to generate valid example")


@register("132")
def generate_color_to_motif_substitution(rng: np.random.Generator) -> dict:
    table_size = int(rng.integers(3, 6))
    colors = pick_colors(rng, table_size)
    # Sample the input dimensions first across a wide envelope (capped so even the
    # smallest 2x motif keeps the expanded output within the grid->grid token cap),
    # then choose a motif size bounded by the remaining budget. This lets the INPUT
    # span the full small-grid range independently on H and W with the output side
    # never exceeding ~22.
    out_cap = 22
    h, w = _sample_hw(rng, 2, out_cap // 2, 2, out_cap // 2)
    motif_h = int(rng.integers(2, min(3, out_cap // h) + 1))
    motif_w = int(rng.integers(2, min(3, out_cap // w) + 1))
    table = []
    for color in colors:
        motif_palette = pick_colors(rng, int(rng.integers(2, 4)))
        motif = np.array(rng.choice(motif_palette, size=(motif_h, motif_w), replace=True), dtype=np.uint8)
        table.append({"color": int(color), "motif": grid_to_list(motif)})

    input_grid = np.array(rng.choice(colors, size=(h, w), replace=True), dtype=np.uint8)
    lookup = {entry["color"]: np.array(entry["motif"], dtype=np.uint8) for entry in table}
    out = make_grid(h * motif_h, w * motif_w, fill=0)
    for r in range(h):
        for c in range(w):
            out[r * motif_h : (r + 1) * motif_h, c * motif_w : (c + 1) * motif_w] = lookup[int(input_grid[r, c])]
    return {
        "input": grid_to_list(input_grid),
        "output": grid_to_list(out),
        "metadata": {
            "replacement_table": table,
            "motif_height": motif_h,
            "motif_width": motif_w,
            "layout_rule": "replace each cell by its motif block, preserving grid order",
        },
    }


@register("139")
def generate_fill_between_target_cells(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, fill_color, distractor = unused_colors(rng, 3, bg)
        axis = str(rng.choice(["rows", "columns", "both"]))
        # Independent H/W, shifted down toward room for two small frames and up to
        # the grid->grid token-budget cap (~16-17). Square-biased.
        h, w = _sample_hw(rng, 7, 16, 7, 16)
        grid = make_grid(h, w, fill=bg)
        boxes: list[tuple[int, int, int, int]] = []
        for _frame in range(int(rng.integers(2, 6))):
            for _attempt in range(100):
                rh = int(rng.integers(3, min(7, h - 1) + 1))
                rw = int(rng.integers(3, min(7, w - 1) + 1))
                r0 = int(rng.integers(0, h - rh + 1))
                c0 = int(rng.integers(0, w - rw + 1))
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if any(boxes_overlap(box, other, gap=1) for other in boxes):
                    continue
                r1, c1 = box[2], box[3]
                grid[r0:r1 + 1, c0] = target_color
                grid[r0:r1 + 1, c1] = target_color
                grid[r0, c0:c1 + 1] = target_color
                grid[r1, c0:c1 + 1] = target_color
                boxes.append(box)
                break
        if len(boxes) < 2:
            continue

        for _noise in range(int(rng.integers(3, 8))):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if grid[r, c] == bg:
                grid[r, c] = distractor

        # Optional sparse background detail OUTSIDE the frames. Frames are made of
        # the target color and the fill is computed per target object, so detail in
        # other colors outside the frame boxes is provably irrelevant.
        bg_info = add_patterned_background(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, fill_color, distractor],
            protected_boxes=boxes,
            solid_probability=0.4,
        )

        meta = {
            "background_color": bg,
            "target_color": int(target_color),
            "fill_color": int(fill_color),
            "axis": axis,
            "connectivity": 4,
            "background_style": bg_info["style"],
            "fill_rule": "fill only background cells strictly between cells from the same target object on each selected line",
        }
        out = _fill_between_same_object(grid, meta)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": meta,
        }
    raise RuntimeError("Task 139: failed to generate valid example")


@register("150")
def generate_rectangle_from_corner_markers(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    marker_color, draw_color, distractor = unused_colors(rng, 3, bg)
    # Independent H/W from the structural minimum (a 3-cell rectangle needs >=4)
    # up to the grid->grid token-budget cap. Square-biased.
    h, w = _sample_hw(rng, 5, 16, 5, 16)
    grid = make_grid(h, w, fill=bg)
    r0 = int(rng.integers(0, h - 3))
    r1 = int(rng.integers(r0 + 2, h))
    c0 = int(rng.integers(0, w - 3))
    c1 = int(rng.integers(c0 + 2, w))
    diagonal = str(rng.choice(["top_left_bottom_right", "top_right_bottom_left"]))
    if diagonal == "top_left_bottom_right":
        grid[r0, c0] = marker_color
        grid[r1, c1] = marker_color
    else:
        grid[r0, c1] = marker_color
        grid[r1, c0] = marker_color
    mode = str(rng.choice(["outline", "filled"]))
    distractor_count = 0
    for _ in range(int(rng.integers(2, 7))):
        for _attempt in range(40):
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if grid[r, c] == bg:
                grid[r, c] = distractor
                distractor_count += 1
                break
    # Optional sparse background detail OUTSIDE the rectangle. The rule recovers the
    # rectangle from the two unique marker-colored corner cells, so detail in any
    # other color outside the rectangle box is provably ignored.
    bg_info = add_patterned_background(
        grid,
        rng,
        bg=bg,
        exclude_colors=[marker_color, draw_color, distractor],
        protected_boxes=[(r0, c0, r1, c1)],
        solid_probability=0.4,
    )
    out = grid.copy()
    if mode == "filled":
        out[r0 : r1 + 1, c0 : c1 + 1] = draw_color
    else:
        out[r0 : r1 + 1, c0] = draw_color
        out[r0 : r1 + 1, c1] = draw_color
        out[r0, c0 : c1 + 1] = draw_color
        out[r1, c0 : c1 + 1] = draw_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "corner_marker_color": int(marker_color),
            "draw_color": int(draw_color),
            "distractor_color": int(distractor),
            "distractor_count": distractor_count,
            "rectangle_mode": mode,
            "corner_diagonal": diagonal,
            "background_style": bg_info["style"],
            "corner_role": "the two marker cells are opposite rectangle corners",
        },
    }


@register("160")
def generate_fill_marked_region_checkerboard(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    border_color, blank_color, anchor_color, other_color = unused_colors(rng, 4, bg)
    # Interior at least 2x2 so both checkerboard colors necessarily appear.
    interior_h = int(rng.integers(2, 8))
    interior_w = int(rng.integers(2, 8))
    # Sample H and W independently across a wide envelope (down to the structural
    # minimum interior+3, up toward the ARC token-budget cap) instead of a fixed
    # margin band. The +2 accounts for the one-cell border ring on each axis.
    region_h = interior_h + 2
    region_w = interior_w + 2
    h, w = _sample_hw(rng, region_h + 1, max(region_h + 1, 20), region_w + 1, max(region_w + 1, 20))
    grid = make_grid(h, w, fill=bg)
    r0 = int(rng.integers(0, h - region_h + 1))
    c0 = int(rng.integers(0, w - region_w + 1))
    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = blank_color
    ar = int(rng.integers(r0 + 1, r1))
    ac = int(rng.integers(c0 + 1, c1))
    grid[ar, ac] = anchor_color

    # Optional sparse background detail OUTSIDE the marked region. The rule locates
    # its region by the unique border color and the unique anchor color, so detail
    # in any other color is provably ignored; we protect the whole region box and
    # exclude the four rule colors so the texture can never mimic them.
    bg_info = add_patterned_background(
        grid,
        rng,
        bg=bg,
        exclude_colors=[border_color, blank_color, anchor_color, other_color],
        protected_boxes=[(r0, c0, r1, c1)],
        solid_probability=0.34,
    )

    out = grid.copy()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[r, c] = anchor_color if (r - ar + c - ac) % 2 == 0 else other_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "region_border_color": int(border_color),
            "blank_color": int(blank_color),
            "anchor_color": int(anchor_color),
            "other_checker_color": int(other_color),
            "background_style": bg_info["style"],
            "anchor_rule": "the anchor color cell is phase 0 of the checkerboard",
        },
    }


@register("163")
def generate_marker_induced_subgrid(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    row_marker, column_marker = unused_colors(rng, 2, bg)
    fill_colors = [bg] + unused_colors(rng, 4, bg, row_marker, column_marker)
    # Independent H/W across a wide small-grid envelope (the dense input grid is the
    # bundle-dominant side; the extracted subgrid stays tiny). Square-biased.
    h, w = _sample_hw(rng, 5, 18, 5, 18)
    grid = _make_random_core(rng, h, w, fill_colors)
    # Keep at least one unselected row and column so markers can sit off the
    # selected lines and the extracted subgrid stays marker-free.
    n_rows = int(rng.integers(2, min(6, h - 1) + 1))
    n_cols = int(rng.integers(2, min(6, w - 1) + 1))
    rows = sorted(int(v) for v in rng.choice(h, size=n_rows, replace=False))
    cols = sorted(int(v) for v in rng.choice(w, size=n_cols, replace=False))
    marker_col_choices = [c for c in range(w) if c not in cols] or list(range(w))
    marker_row_choices = [r for r in range(h) if r not in rows] or list(range(h))
    for r in rows:
        grid[r, _choice_int(rng, marker_col_choices)] = row_marker
    for c in cols:
        grid[_choice_int(rng, marker_row_choices), c] = column_marker
    out = grid[np.ix_(rows, cols)]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "row_marker_color": int(row_marker),
            "column_marker_color": int(column_marker),
            "row_rule": "select rows containing the row marker color",
            "column_rule": "select columns containing the column marker color",
        },
    }


@register("165")
def generate_compress_lines_summary(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    target_color, true_color, false_color = unused_colors(rng, 3, bg)
    palette = [bg, target_color] + unused_colors(rng, 4, bg, target_color, true_color, false_color)
    # Independent H/W across a wide small-grid envelope (the dense input is the
    # bundle-dominant side; the summary strip stays thin). Square-biased.
    h, w = _sample_hw(rng, 5, 18, 5, 18)
    grid = _make_random_core(rng, h, w, palette)
    axis = str(rng.choice(["rows", "columns"]))
    rule = str(rng.choice(["majority_color", "first_non_background", "target_presence"]))
    meta = {
        "axis": axis,
        "summary_rule": rule,
        "background_color": bg,
        "target_color": int(target_color),
        "true_color": int(true_color),
        "false_color": int(false_color),
    }
    if rule == "majority_color":
        meta["tie_rule"] = "use the background color when multiple colors share the maximum count"
    if axis == "rows":
        out = np.array([[_summary_line([int(v) for v in grid[r, :]], meta)] for r in range(h)], dtype=np.uint8)
    else:
        out = np.array([[_summary_line([int(v) for v in grid[:, c]], meta) for c in range(w)]], dtype=np.uint8)
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": meta,
    }


@register("166")
def generate_count_to_bars(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    target_color, bar_color = unused_colors(rng, 2, bg)
    noise_colors = unused_colors(rng, 3, bg, target_color, bar_color)
    # Independent H/W across a wide small-grid envelope, capped at the grid->grid
    # token-budget bound (input and bar output are the same size). Square-biased.
    h, w = _sample_hw(rng, 4, 16, 4, 16)
    grid = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            roll = float(rng.random())
            if roll < 0.25:
                grid[r, c] = target_color
            elif roll < 0.45:
                grid[r, c] = int(rng.choice(noise_colors))
    axis = str(rng.choice(["rows", "columns"]))
    edge = str(rng.choice(["left", "right"] if axis == "rows" else ["top", "bottom"]))
    out = make_grid(h, w, fill=bg)
    if axis == "rows":
        for r in range(h):
            count = int(np.count_nonzero(grid[r, :] == target_color))
            if edge == "left":
                out[r, :count] = bar_color
            else:
                out[r, w - count :] = bar_color
    else:
        for c in range(w):
            count = int(np.count_nonzero(grid[:, c] == target_color))
            if edge == "top":
                out[:count, c] = bar_color
            else:
                out[h - count :, c] = bar_color
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "target_color": int(target_color),
            "bar_color": int(bar_color),
            "count_axis": axis,
            "edge": edge,
            "output_rule": "draw bars on a blank background grid of the same size",
        },
    }


@register("171")
def generate_distinct_color_palette(rng: np.random.Generator) -> dict:
    n_colors = int(rng.integers(3, 8))
    colors = pick_colors(rng, n_colors)
    # Independent H/W across a wide small-grid envelope (the dense input is the
    # bundle-dominant side; the palette strip stays tiny). Square-biased. The grid
    # always has room (>=16 cells) to seed each distinct color at least once.
    h, w = _sample_hw(rng, 4, 18, 4, 18)
    grid = np.array(rng.choice(colors, size=(h, w), replace=True), dtype=np.uint8)
    for i, color in enumerate(colors):
        grid.flat[i] = int(color)
    ordering = str(rng.choice(["frequency_desc", "first_seen_row_major"]))
    orientation = str(rng.choice(["row", "column"]))
    flat_values = [int(v) for v in grid.flatten()]
    first_seen = []
    seen = set()
    for value in flat_values:
        if value not in seen:
            first_seen.append(value)
            seen.add(value)
    if ordering == "frequency_desc":
        counts = Counter(flat_values)
        ordered = sorted(first_seen, key=lambda color: -counts[color])
    else:
        ordered = first_seen
    out = np.array([ordered], dtype=np.uint8) if orientation == "row" else np.array([[c] for c in ordered], dtype=np.uint8)
    metadata = {
        "ordering": ordering,
        "output_orientation": orientation,
    }
    if ordering == "frequency_desc":
        metadata["tie_break"] = "first_seen_row_major"
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": metadata,
    }


@register("180")
def generate_remove_outer_padding(rng: np.random.Generator) -> dict:
    padding_color = int(rng.integers(0, NUM_COLORS))
    core_colors = unused_colors(rng, int(rng.integers(3, 6)), padding_color)
    core_h = int(rng.integers(4, 9))
    core_w = int(rng.integers(4, 9))
    core = _make_random_core(rng, core_h, core_w, core_colors)
    all_sides = ["top", "bottom", "left", "right"]
    side_mask = int(rng.integers(1, 1 << len(all_sides)))
    sides = [side for index, side in enumerate(all_sides) if side_mask & (1 << index)]
    thickness = int(rng.integers(1, 4))
    top = thickness if "top" in sides else 0
    bottom = thickness if "bottom" in sides else 0
    left = thickness if "left" in sides else 0
    right = thickness if "right" in sides else 0
    grid = np.pad(core, ((top, bottom), (left, right)), constant_values=padding_color)
    out = grid[top : grid.shape[0] - bottom if bottom else grid.shape[0], left : grid.shape[1] - right if right else grid.shape[1]]
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "padding_color": padding_color,
            "sides": sides,
            "thickness": thickness,
            "removal_rule": "remove exactly this many outer rows or columns from each stated side",
        },
    }
