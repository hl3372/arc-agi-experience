"""Fit, placement, and candidate-selection generators for core2 batch J."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    unused_colors,
)
from registry import register


Coord = tuple[int, int]
Shape = frozenset[Coord]

LABELS = ["A", "B", "C"]

BASE_SHAPES: list[Shape] = [
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2)}),
]


def _normalize(cells: set[Coord] | frozenset[Coord]) -> Shape:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _bbox(cells: set[Coord] | frozenset[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_size(shape: Shape) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _expanded_shape(shape: Shape) -> Shape:
    for r, c in sorted(shape):
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            cell = (r + dr, c + dc)
            if cell not in shape:
                return _normalize(set(shape) | {cell})
    raise ValueError("cannot expand empty shape")


def _transform(shape: Shape, name: str) -> Shape:
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


def _orientations(shape: Shape, allowed: list[str]) -> list[Shape]:
    seen: set[Shape] = set()
    out: list[Shape] = []
    for name in allowed:
        orient = _transform(shape, name)
        if orient not in seen:
            seen.add(orient)
            out.append(orient)
    return out


def _absolute(shape: Shape, anchor: Coord) -> set[Coord]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in shape}


def _in_bounds(cells: set[Coord], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _stamp(grid: np.ndarray, shape: Shape, anchor: Coord, color: int) -> set[Coord]:
    cells = _absolute(shape, anchor)
    for r, c in cells:
        grid[r, c] = color
    return cells


def _components(grid: np.ndarray, color: int) -> list[set[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return sorted(comps, key=lambda comp: _bbox(comp))


def _candidate_marker_list(label_colors: list[int]) -> list[dict]:
    return [{"label": label, "marker_color": int(color)} for label, color in zip(LABELS, label_colors)]


def _valid_anchor(
    grid: np.ndarray,
    shape: Shape,
    anchor: Coord,
    *,
    bg: int,
    anchor_color: int,
    allowed_transforms: list[str],
) -> bool:
    h, w = grid.shape
    for orient in _orientations(shape, allowed_transforms):
        cells = _absolute(orient, anchor)
        if not _in_bounds(cells, h, w):
            continue
        if all(int(grid[r, c]) in (bg, anchor_color) for r, c in cells):
            return True
    return False


def _count_fitting_orientations(
    grid: np.ndarray,
    shape: Shape,
    anchor: Coord,
    *,
    bg: int,
    anchor_color: int,
    allowed_transforms: list[str],
) -> int:
    h, w = grid.shape
    count = 0
    for orient in _orientations(shape, allowed_transforms):
        cells = _absolute(orient, anchor)
        if _in_bounds(cells, h, w) and all(int(grid[r, c]) in (bg, anchor_color) for r, c in cells):
            count += 1
    return count


def _first_valid_orientation(
    grid: np.ndarray,
    shape: Shape,
    anchor: Coord,
    *,
    bg: int,
    anchor_color: int,
    allowed_transforms: list[str],
) -> Shape | None:
    h, w = grid.shape
    for orient in _orientations(shape, allowed_transforms):
        cells = _absolute(orient, anchor)
        if _in_bounds(cells, h, w) and all(int(grid[r, c]) in (bg, anchor_color) for r, c in cells):
            return orient
    return None


def _apply_placement(
    grid: np.ndarray,
    shape: Shape,
    source_cells: set[Coord],
    anchor: Coord,
    *,
    bg: int,
    object_color: int,
    anchor_color: int,
    allowed_transforms: list[str],
    copy_move_mode: str,
) -> np.ndarray:
    out = grid.copy()
    if copy_move_mode == "move":
        for r, c in source_cells:
            out[r, c] = bg
    orient = _first_valid_orientation(
        grid,
        shape,
        anchor,
        bg=bg,
        anchor_color=anchor_color,
        allowed_transforms=allowed_transforms,
    )
    if orient is None:
        raise ValueError("no valid orientation for selected anchor")
    for r, c in _absolute(orient, anchor):
        out[r, c] = object_color
    return out


def _block_anchor(
    grid: np.ndarray,
    shape: Shape,
    anchor: Coord,
    *,
    bg: int,
    anchor_color: int,
    blocker_color: int,
    allowed_transforms: list[str],
) -> None:
    h, w = grid.shape
    blocked: set[Coord] = set()
    for orient in _orientations(shape, allowed_transforms):
        cells = sorted(_absolute(orient, anchor))
        if not _in_bounds(set(cells), h, w):
            continue
        for cell in cells:
            if cell != anchor and int(grid[cell]) in (bg, anchor_color):
                blocked.add(cell)
                break
    for r, c in blocked:
        grid[r, c] = blocker_color


def _count_items(grid: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(grid == color))


@register("230")
def generate_count_comparison_action(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W over the ARC-aligned envelope. The rule is purely a
        # cell-count comparison, so it is valid on any grid large enough to hold
        # the sampled cells.
        h = int(rng.integers(6, 23))
        w = int(rng.integers(6, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        color_a, color_b, action_color = unused_colors(rng, 3, bg)
        count_case = str(rng.choice(["a_greater", "b_greater", "tie"]))
        # Scale the cell counts up on larger grids so the scene is not trivially
        # sparse, while keeping the grid background-dominant.
        cap = max(2, (h * w) // 6)
        hi = min(8, cap)
        lo_g = min(5, hi)
        if count_case == "a_greater":
            count_a, count_b = int(rng.integers(lo_g, hi + 1)), int(rng.integers(2, max(3, lo_g)))
        elif count_case == "b_greater":
            count_b, count_a = int(rng.integers(lo_g, hi + 1)), int(rng.integers(2, max(3, lo_g)))
        else:
            count_a = count_b = int(rng.integers(3, max(4, min(6, cap // 2) + 1)))
        grid = make_grid(h, w, fill=bg)
        cells = [(r, c) for r in range(h) for c in range(w)]
        chosen = rng.choice(len(cells), size=count_a + count_b, replace=False).tolist()
        for idx in chosen[:count_a]:
            grid[cells[int(idx)]] = color_a
        for idx in chosen[count_a:]:
            grid[cells[int(idx)]] = color_b

        target_category = str(rng.choice(["greater_count", "lesser_count", "tied"]))
        fallback_category = str(rng.choice(["set_A", "set_B", "both_sets", "neither_set"]))
        action_type = str(rng.choice(["recolor", "erase"]))

        if count_a > count_b:
            category_colors = {"greater_count": [color_a], "lesser_count": [color_b], "tied": []}
        elif count_b > count_a:
            category_colors = {"greater_count": [color_b], "lesser_count": [color_a], "tied": []}
        else:
            category_colors = {"greater_count": [], "lesser_count": [], "tied": [color_a, color_b]}
        fallback_colors = {
            "set_A": [color_a],
            "set_B": [color_b],
            "both_sets": [color_a, color_b],
            "neither_set": [],
        }[fallback_category]
        selected = category_colors[target_category] or fallback_colors
        if not selected:
            continue
        # Optional patterned background (a solid/patterned MIX). Texture never uses
        # the two set colours or the action colour, so the per-colour counts and
        # the recolour/erase action are unaffected; texture cells stay untouched.
        bg_style = add_patterned_background(
            grid, rng, bg=bg, exclude_colors=[color_a, color_b, action_color]
        )
        out = grid.copy()
        replacement = action_color if action_type == "recolor" else bg
        for color in selected:
            out[grid == color] = replacement
        if np.array_equal(out, grid):
            continue

        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "set_a_color": int(color_a),
                "set_b_color": int(color_b),
                "action_color": int(action_color),
                "count_a": count_a,
                "count_b": count_b,
                "target_category": target_category,
                "fallback_category": fallback_category,
                "action_type": action_type,
                "fallback_rule": "use fallback_category when target_category is empty for this comparison",
                "background_style": bg_style,
            },
        }
    raise RuntimeError("Task 230: failed to generate non-degenerate example")


@register("231")
def generate_linewise_count_comparison(rng: np.random.Generator) -> dict:
    # Independent H/W over the ARC-aligned envelope; the per-line count rule is
    # valid at any size (per-line fills are clamped to the line length below).
    h = int(rng.integers(6, 23))
    w = int(rng.integers(6, 23))
    bg = int(rng.integers(0, NUM_COLORS))
    color_a, color_b, greater_color, equal_color, less_color = unused_colors(rng, 5, bg)
    axis = str(rng.choice(["row", "column"]))
    output_form = str(rng.choice(["summary_strip", "paint_full_lines", "mark_line_start"]))
    grid = make_grid(h, w, fill=bg)
    outer = h if axis == "row" else w
    inner = w if axis == "row" else h
    relations = ["greater", "equal", "less"]
    for i in range(outer):
        relation = relations[(i + int(rng.integers(0, 3))) % 3]
        if relation == "greater":
            na, nb = int(rng.integers(3, min(inner, 5) + 1)), int(rng.integers(0, 3))
        elif relation == "less":
            nb, na = int(rng.integers(3, min(inner, 5) + 1)), int(rng.integers(0, 3))
        else:
            na = nb = int(rng.integers(1, max(2, inner // 2)))
        na = min(na, inner)
        nb = min(nb, inner - na)
        positions = rng.choice(inner, size=na + nb, replace=False).tolist()
        for j in positions[:na]:
            r, c = (i, int(j)) if axis == "row" else (int(j), i)
            grid[r, c] = color_a
        for j in positions[na:]:
            r, c = (i, int(j)) if axis == "row" else (int(j), i)
            grid[r, c] = color_b

    line_colors = []
    for i in range(outer):
        line = grid[i, :] if axis == "row" else grid[:, i]
        ca = _count_items(line, color_a)
        cb = _count_items(line, color_b)
        line_colors.append(greater_color if ca > cb else less_color if ca < cb else equal_color)

    # Optional patterned background (solid/patterned MIX) on the input only.
    # Texture excludes both set colours and all three result colours, so the
    # per-line counts and the recomputed output are unaffected.
    bg_style = add_patterned_background(
        grid, rng, bg=bg,
        exclude_colors=[color_a, color_b, greater_color, equal_color, less_color],
    )

    if output_form == "summary_strip":
        out = np.array(line_colors, dtype=np.uint8).reshape((outer, 1) if axis == "row" else (1, outer))
    elif output_form == "paint_full_lines":
        out = make_grid(h, w, fill=bg)
        for i, color in enumerate(line_colors):
            if axis == "row":
                out[i, :] = color
            else:
                out[:, i] = color
    else:
        out = make_grid(h, w, fill=bg)
        for i, color in enumerate(line_colors):
            if axis == "row":
                out[i, 0] = color
            else:
                out[0, i] = color

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "set_a_color": int(color_a),
            "set_b_color": int(color_b),
            "greater_color": int(greater_color),
            "equal_color": int(equal_color),
            "less_color": int(less_color),
            "axis": axis,
            "output_form": output_form,
            "background_style": bg_style,
        },
    }


def _placement_metadata(
    *,
    bg: int,
    object_color: int,
    source_anchor: Coord,
    source_shape: Shape,
    allowed_transforms: list[str],
    copy_move_mode: str,
    extra: dict,
) -> dict:
    meta = {
        "background_color": bg,
        "object_color": int(object_color),
        "source_anchor": list(source_anchor),
        "source_shape": [list(p) for p in sorted(source_shape)],
        "allowed_transforms": allowed_transforms,
        "copy_move_mode": copy_move_mode,
        "anchor_landmark": "top_left_of_transformed_bounding_box",
        "overwrite_behavior": "object may overwrite the selected anchor or mask cells only; all other occupied cells block placement",
    }
    meta.update(extra)
    return meta


@register("249")
def generate_place_at_unique_fit_anchor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W over the ARC-aligned envelope. Floor leaves room for the
        # source band (rows 1..~4) and a good anchor below it.
        h = int(rng.integers(9, 23))
        w = int(rng.integers(9, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        object_color, anchor_color, blocker_color = unused_colors(rng, 3, bg)
        allowed_options = [
            ["identity"],
            ["identity", "rotate90", "rotate180", "rotate270"],
            ["identity", "reflect_horizontal"],
        ]
        allowed = list(allowed_options[int(rng.integers(0, len(allowed_options)))])
        shape = BASE_SHAPES[int(rng.integers(0, len(BASE_SHAPES)))]
        sh, sw = _shape_size(shape)
        grid = make_grid(h, w, fill=bg)
        source_anchor = (1, 1)
        source_cells = _stamp(grid, shape, source_anchor, object_color)
        good_anchor = (int(rng.integers(5, h - sh)), int(rng.integers(3, w - sw)))
        candidates = [good_anchor, (h - 1, w - 1), (good_anchor[0], 0), (0, good_anchor[1])]
        for anchor in candidates:
            grid[anchor] = anchor_color
        for anchor in candidates[1:]:
            _block_anchor(
                grid,
                shape,
                anchor,
                bg=bg,
                anchor_color=anchor_color,
                blocker_color=blocker_color,
                allowed_transforms=allowed,
            )
        # Inert blocker-coloured distractor cells vary the component count and the
        # scene density. They are added before the validity gate below, so any
        # distractor that would disturb anchor- or orientation-uniqueness simply
        # triggers a retry; correctness is enforced by the same gate as before.
        protect_cells = set(source_cells) | set(candidates)
        free = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if grid[r, c] == bg and (r, c) not in protect_cells
        ]
        n_distract = int(rng.integers(0, 7))
        if free and n_distract:
            idx = rng.choice(len(free), size=min(n_distract, len(free)), replace=False).tolist()
            for i in idx:
                grid[free[int(i)]] = blocker_color
        valid = [
            tuple(p) for p in candidates
            if _valid_anchor(grid, shape, tuple(p), bg=bg, anchor_color=anchor_color, allowed_transforms=allowed)
        ]
        if valid != [good_anchor]:
            continue
        # Anchor uniqueness is not enough: the placed orientation must also be
        # unique, otherwise a solver cannot know which fitting orientation to use
        # when several allowed transforms fit at the selected anchor.
        if _count_fitting_orientations(
            grid, shape, good_anchor, bg=bg, anchor_color=anchor_color, allowed_transforms=allowed
        ) != 1:
            continue
        mode = str(rng.choice(["copy", "move"]))
        out = _apply_placement(
            grid,
            shape,
            source_cells,
            good_anchor,
            bg=bg,
            object_color=object_color,
            anchor_color=anchor_color,
            allowed_transforms=allowed,
            copy_move_mode=mode,
        )
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": _placement_metadata(
                bg=bg,
                object_color=object_color,
                source_anchor=source_anchor,
                source_shape=shape,
                allowed_transforms=allowed,
                copy_move_mode=mode,
                extra={
                    "anchor_color": int(anchor_color),
                    "blocker_color": int(blocker_color),
                    "candidate_anchors": [list(p) for p in candidates],
                    "fit_constraint": "transformed object cells must fit inside the grid and collide only with the selected anchor marker",
                },
            ),
        }
    raise RuntimeError("Task 249: failed to generate valid example")


def _place_blobs(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    shapes: list[Shape],
    halo: int,
    max_attempts: int = 400,
) -> list[Coord] | None:
    """Find anchors for a list of shapes so their cells, dilated by `halo`, never
    overlap; keeps separately stamped blobs (and any frame ring drawn around
    them) from merging. Returns anchors in input order or None."""
    def dilate_n(cells: set[Coord], n: int) -> set[Coord]:
        out = set(cells)
        for _ in range(n):
            grown: set[Coord] = set()
            for r, c in out:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        grown.add((r + dr, c + dc))
            out = grown
        return out

    for _ in range(max_attempts):
        occupied: set[Coord] = set()
        anchors: list[Coord] = []
        ok = True
        for shape in shapes:
            sh, sw = _shape_size(shape)
            placed = False
            for _ in range(120):
                r0 = int(rng.integers(0, max(1, h - sh + 1)))
                c0 = int(rng.integers(0, max(1, w - sw + 1)))
                cells = _absolute(shape, (r0, c0))
                if not _in_bounds(cells, h, w):
                    continue
                if dilate_n(cells, halo) & occupied:
                    continue
                occupied |= cells
                anchors.append((r0, c0))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return anchors
    return None


@register("250")
def generate_place_into_unique_mask_region(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W over the ARC-aligned envelope. The source object plus
        # three separated mask blobs (with an optional frame ring) set the floor.
        h = int(rng.integers(10, 23))
        w = int(rng.integers(10, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        object_color, mask_color, frame_color = unused_colors(rng, 3, bg)
        allowed = list([
            ["identity"],
            ["identity", "rotate90", "rotate180", "rotate270"],
            ["identity", "reflect_horizontal"],
        ][int(rng.integers(0, 3))])
        shape = BASE_SHAPES[int(rng.integers(0, len(BASE_SHAPES)))]
        valid_shape = _orientations(shape, allowed)[-1]
        region_type = str(rng.choice(["mask_region", "hole", "frame_interior"]))
        decoy_shapes = [
            _expanded_shape(valid_shape),
            BASE_SHAPES[(BASE_SHAPES.index(shape) + 1) % len(BASE_SHAPES)],
        ]
        blob_shapes = [valid_shape, decoy_shapes[0], decoy_shapes[1]]
        # The framed variants draw a one-cell ring, so they need a wider halo to
        # stop adjacent frames from merging blobs of the mask colour.
        halo = 2 if region_type != "mask_region" else 1
        # Place the source object + the three mask blobs, all separated.
        layout = _place_blobs(rng, h=h, w=w, shapes=[shape, *blob_shapes], halo=halo)
        if layout is None:
            continue
        source_anchor = layout[0]
        mask_anchors = layout[1:]
        grid = make_grid(h, w, fill=bg)
        source_cells = _stamp(grid, shape, source_anchor, object_color)
        if region_type == "mask_region":
            for anchor, blob in zip(mask_anchors, blob_shapes):
                _stamp(grid, blob, anchor, mask_color)
        else:
            for anchor, blob in zip(mask_anchors, blob_shapes):
                for r, c in _absolute(blob, anchor):
                    grid[r, c] = mask_color
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                            grid[nr, nc] = frame_color
        comps = _components(grid, mask_color)
        matching = [comp for comp in comps if _normalize(comp) in _orientations(shape, allowed)]
        if len(matching) != 1:
            continue
        mode = str(rng.choice(["copy", "move"]))
        out = grid.copy()
        if mode == "move":
            for r, c in source_cells:
                out[r, c] = bg
        for r, c in matching[0]:
            out[r, c] = object_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": _placement_metadata(
                bg=bg,
                object_color=object_color,
                source_anchor=source_anchor,
                source_shape=shape,
                allowed_transforms=allowed,
                copy_move_mode=mode,
                extra={
                    "mask_color": int(mask_color),
                    "frame_color": int(frame_color),
                    "region_type": region_type,
                    "fit_constraint": "choose the unique mask-color component whose normalized cells equal an allowed transform of the source object",
                },
            ),
        }
    raise RuntimeError("Task 250: failed to generate valid example")


def _alignment_ok(anchor: Coord, relation: str, guide: Coord) -> bool:
    if relation == "same_row_as_guide":
        return anchor[0] == guide[0]
    if relation == "same_column_as_guide":
        return anchor[1] == guide[1]
    if relation == "above_guide_same_column":
        return anchor[1] == guide[1] and anchor[0] < guide[0]
    if relation == "left_of_guide_same_row":
        return anchor[0] == guide[0] and anchor[1] < guide[1]
    raise ValueError(f"unknown relation {relation}")


@register("251")
def generate_place_by_alignment(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W over the ARC-aligned envelope; the source band plus a
        # guide cell with room on both sides set the structural floor.
        h = int(rng.integers(9, 23))
        w = int(rng.integers(9, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        object_color, anchor_color, guide_color = unused_colors(rng, 3, bg)
        shape = BASE_SHAPES[int(rng.integers(0, len(BASE_SHAPES)))]
        allowed = ["identity"]
        # Reserve the top-left band for the source object so candidate anchors
        # cannot collide with it; randomize everything else per seed.
        source_anchor = (1, 1)
        reserved_rows = max(_shape_size(s)[0] for s in BASE_SHAPES) + 2
        relation = str(rng.choice([
            "same_row_as_guide", "same_column_as_guide",
            "above_guide_same_column", "left_of_guide_same_row",
        ]))
        guide = (int(rng.integers(reserved_rows + 1, h - 2)), int(rng.integers(3, w - 3)))

        # Build the good anchor on the guide's line, on the side the relation
        # requires for the strict variants.
        if relation in ("same_column_as_guide", "above_guide_same_column"):
            col = guide[1]
            if relation == "above_guide_same_column":
                lo = reserved_rows
                if guide[0] - 1 < lo:
                    continue
                row = int(rng.integers(lo, guide[0]))
            else:
                choices = [r for r in range(reserved_rows, h) if r != guide[0]]
                if not choices:
                    continue
                row = int(rng.choice(choices))
            good_anchor = (row, col)
        else:  # same_row / left_of, on the guide's row
            row = guide[0]
            if relation == "left_of_guide_same_row":
                if guide[1] - 1 < 1:
                    continue
                col = int(rng.integers(1, guide[1]))
            else:
                choices = [c for c in range(1, w - 1) if c != guide[1]]
                if not choices:
                    continue
                col = int(rng.choice(choices))
            good_anchor = (row, col)

        # Distractor anchors. For the strict relations include a deliberate
        # wrong-side distractor that satisfies the non-strict part (same line)
        # but violates the inequality, so '<' is load-bearing.
        candidates = [good_anchor]
        if relation == "above_guide_same_column" and guide[0] + 1 <= h - 1:
            candidates.append((int(rng.integers(guide[0] + 1, h)), guide[1]))  # below guide, same column
        elif relation == "left_of_guide_same_row" and guide[1] + 1 <= w - 1:
            candidates.append((guide[0], int(rng.integers(guide[1] + 1, w))))  # right of guide, same row

        # Fill remaining candidate slots with off-line distractors that satisfy
        # no part of the relation.
        attempts = 0
        while len(candidates) < 4 and attempts < 200:
            attempts += 1
            r = int(rng.integers(reserved_rows, h))
            c = int(rng.integers(1, w - 1))
            cand = (r, c)
            if cand == guide or cand in candidates:
                continue
            if _alignment_ok(cand, relation, guide):
                continue  # never let a distractor satisfy the relation
            candidates.append(cand)
        if len(candidates) < 4:
            continue

        grid = make_grid(h, w, fill=bg)
        source_cells = _stamp(grid, shape, source_anchor, object_color)
        grid[guide] = guide_color
        for anchor in candidates:
            grid[anchor] = anchor_color
        # Inert distractor cells (a spare colour, never the object/anchor/guide
        # colour) vary scene density and component count. They are added before
        # the validity gate, so any distractor that perturbs the unique aligned
        # fit triggers a retry; correctness is enforced by the same gate.
        spare = unused_colors(rng, 1, bg, object_color, anchor_color, guide_color)
        if spare:
            protect_cells = set(source_cells) | set(candidates) | {guide}
            free = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if grid[r, c] == bg and (r, c) not in protect_cells
            ]
            n_distract = int(rng.integers(0, 7))
            if free and n_distract:
                idx = rng.choice(len(free), size=min(n_distract, len(free)), replace=False).tolist()
                for i in idx:
                    grid[free[int(i)]] = spare[0]
        valid = [
            p for p in candidates
            if _alignment_ok(p, relation, guide)
            and _valid_anchor(grid, shape, p, bg=bg, anchor_color=anchor_color, allowed_transforms=allowed)
        ]
        if valid != [good_anchor]:
            continue
        # The placed object must not cover the guide or any other candidate
        # marker (it may only overwrite its own selected anchor).
        footprint = _absolute(shape, good_anchor)
        protect = {guide} | (set(candidates) - {good_anchor})
        if footprint & protect:
            continue
        mode = str(rng.choice(["copy", "move"]))
        out = _apply_placement(
            grid,
            shape,
            source_cells,
            good_anchor,
            bg=bg,
            object_color=object_color,
            anchor_color=anchor_color,
            allowed_transforms=allowed,
            copy_move_mode=mode,
        )
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": _placement_metadata(
                bg=bg,
                object_color=object_color,
                source_anchor=source_anchor,
                source_shape=shape,
                allowed_transforms=allowed,
                copy_move_mode=mode,
                extra={
                    "anchor_color": int(anchor_color),
                    "guide_color": int(guide_color),
                    "guide_position": list(guide),
                    "candidate_anchors": [list(p) for p in candidates],
                    "alignment_constraint": relation,
                },
            ),
        }
    raise RuntimeError("Task 251: failed to generate valid example")


def _contact_ok(grid: np.ndarray, shape: Shape, anchor: Coord, contact_color: int, side: str) -> bool:
    delta = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
    }[side]
    h, w = grid.shape
    for r, c in _absolute(shape, anchor):
        nr, nc = r + delta[0], c + delta[1]
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == contact_color:
            return True
    return False


@register("252")
def generate_place_by_contact(rng: np.random.Generator) -> dict:
    delta = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}
    for _ in range(MAX_RETRIES):
        # Independent H/W over the ARC-aligned envelope; the source band plus a
        # good anchor with an external contact cell set the structural floor.
        h = int(rng.integers(9, 23))
        w = int(rng.integers(9, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        object_color, anchor_color, contact_color = unused_colors(rng, 3, bg)
        shape = BASE_SHAPES[int(rng.integers(0, len(BASE_SHAPES)))]
        allowed = ["identity"]
        sh, sw = _shape_size(shape)
        # Reserve the top-left band for the source object.
        reserved_rows = max(_shape_size(s)[0] for s in BASE_SHAPES) + 2
        # Choose the contact side first, then place the good anchor and a single
        # contact cell adjacent to its footprint on that side; this keeps the
        # side distribution balanced instead of relying on rejection.
        side = str(rng.choice(["north", "south", "west", "east"]))
        dr, dc = delta[side]
        good_anchor = (
            int(rng.integers(reserved_rows, h - sh)),
            int(rng.integers(1, w - sw)),
        )
        good_foot_set = _absolute(shape, good_anchor)
        # Pick a contact base among the object cells whose side-neighbor points
        # OUTSIDE the object, so a single external contact cell is always
        # realizable for the chosen side (keeps all four sides balanced).
        external = sorted(
            (r, c) for (r, c) in good_foot_set if (r + dr, c + dc) not in good_foot_set
        )
        if not external:
            continue
        contact_base = external[int(rng.integers(0, len(external)))]
        contact_pos = (contact_base[0] + dr, contact_base[1] + dc)
        if not (0 <= contact_pos[0] < h and 0 <= contact_pos[1] < w):
            continue
        if contact_pos in good_foot_set:
            continue

        # Pick three distractor anchors that do not (yet) get a contact cell on
        # their chosen side, and place them so neither the good anchor's
        # footprint nor the contact cell collides with them.
        good_foot = good_foot_set
        blocked = good_foot | {contact_pos}
        candidates = [good_anchor]
        attempts = 0
        while len(candidates) < 4 and attempts < 300:
            attempts += 1
            cand = (int(rng.integers(reserved_rows, h)), int(rng.integers(1, w - 1)))
            if cand in candidates or cand in blocked:
                continue
            cand_foot = _absolute(shape, cand)
            if not _in_bounds(cand_foot, h, w):
                continue
            # keep distractor footprints disjoint from the good placement and
            # contact cell so they never accidentally satisfy the contact rule
            if cand_foot & blocked:
                continue
            candidates.append(cand)
            blocked |= cand_foot
        if len(candidates) < 4:
            continue

        grid = make_grid(h, w, fill=bg)
        source_cells = _stamp(grid, shape, source_anchor := (1, 1), object_color)
        for anchor in candidates:
            grid[anchor] = anchor_color
        grid[contact_pos] = contact_color
        # Inert distractor cells (a spare colour, never the contact colour) vary
        # scene density and component count. A spare-colour cell can never create
        # contact (that needs the contact colour), and any distractor that breaks
        # the unique valid+contact fit triggers a retry via the gate below.
        spare = unused_colors(rng, 1, bg, object_color, anchor_color, contact_color)
        if spare:
            protect_cells = set(source_cells) | set(candidates) | {contact_pos}
            free = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if grid[r, c] == bg and (r, c) not in protect_cells
            ]
            n_distract = int(rng.integers(0, 7))
            if free and n_distract:
                idx = rng.choice(len(free), size=min(n_distract, len(free)), replace=False).tolist()
                for i in idx:
                    grid[free[int(i)]] = spare[0]
        valid = [
            p for p in candidates
            if _valid_anchor(grid, shape, p, bg=bg, anchor_color=anchor_color, allowed_transforms=allowed)
            and _contact_ok(grid, shape, p, contact_color, side)
        ]
        if valid != [good_anchor]:
            continue
        mode = str(rng.choice(["copy", "move"]))
        out = _apply_placement(
            grid,
            shape,
            source_cells,
            good_anchor,
            bg=bg,
            object_color=object_color,
            anchor_color=anchor_color,
            allowed_transforms=allowed,
            copy_move_mode=mode,
        )
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": _placement_metadata(
                bg=bg,
                object_color=object_color,
                source_anchor=source_anchor,
                source_shape=shape,
                allowed_transforms=allowed,
                copy_move_mode=mode,
                extra={
                    "anchor_color": int(anchor_color),
                    "contact_color": int(contact_color),
                    "candidate_anchors": [list(p) for p in candidates],
                    "contact_constraint": f"placed object must have a {side} 4-neighbor of the contact color",
                    "contact_side": side,
                },
            ),
        }
    raise RuntimeError("Task 252: failed to generate valid example")


def _candidate_meta(label_colors: list[int], origins: list[Coord]) -> list[dict]:
    return [
        {"label": label, "marker_color": int(color), "origin": list(origin)}
        for label, color, origin in zip(LABELS, label_colors, origins)
    ]


def _shape_footprint_with_marker(shape: Shape, anchor: Coord) -> set[Coord]:
    """Cells the shape occupies at `anchor` plus its marker cell one row above
    the object's top-left cell."""
    cells = _absolute(shape, anchor)
    origin = min(cells)
    return cells | {(origin[0] - 1, origin[1])}


def _dilate(cells: set[Coord]) -> set[Coord]:
    """All cells within Chebyshev distance 1 of any cell (a 1-cell keep-out
    halo so separately placed pieces never touch or merge)."""
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _place_reference_and_candidates(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    ref_shape: Shape,
    cand_shapes: list[Shape],
    ref_color: int,
    piece_color: int,
    marker_colors: list[int],
    bg: int,
    max_attempts: int = 400,
) -> tuple[np.ndarray, Coord, list[Coord]] | None:
    """Randomly place a reference object and a set of marked candidate pieces so
    that no piece (including its marker and a one-cell halo) overlaps another and
    every marker is in-bounds. Returns (grid, ref_origin, candidate_origins) or
    None if placement could not be realised in this H x W."""
    grid = make_grid(h, w, fill=bg)
    occupied: set[Coord] = set()

    def try_place(shape: Shape, color: int, marker_color: int | None) -> Coord | None:
        sh, sw = _shape_size(shape)
        # marker sits one row above the object's top cell, so the top row must be >=1
        for _ in range(120):
            r0 = int(rng.integers(1, max(2, h - sh + 1)))
            c0 = int(rng.integers(0, max(1, w - sw + 1)))
            anchor = (r0, c0)
            cells = _absolute(shape, anchor)
            if not _in_bounds(cells, h, w):
                continue
            origin = min(cells)
            footprint = set(cells)
            if marker_color is not None:
                marker = (origin[0] - 1, origin[1])
                if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                    continue
                footprint = footprint | {marker}
            if _dilate(footprint) & occupied:
                continue
            for cell in cells:
                grid[cell] = color
            if marker_color is not None:
                grid[(origin[0] - 1, origin[1])] = marker_color
            occupied.update(footprint)
            return origin
        return None

    for _ in range(max_attempts):
        grid[:] = bg
        occupied.clear()
        ref_origin = try_place(ref_shape, ref_color, None)
        if ref_origin is None:
            continue
        cand_origins: list[Coord] = []
        ok = True
        for shape, mcolor in zip(cand_shapes, marker_colors):
            o = try_place(shape, piece_color, mcolor)
            if o is None:
                ok = False
                break
            cand_origins.append(o)
        if ok:
            return grid, ref_origin, cand_origins
    return None


def _scatter_distractors(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    color: int,
    avoid: set[Coord],
    max_count: int,
) -> None:
    """Sprinkle up to `max_count` inert single cells of `color` on background
    cells, keeping a one-cell halo away from semantic cells. Used only to vary
    the component count; never the candidate/piece color."""
    h, w = grid.shape
    halo = _dilate(set(avoid))
    free = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == bg and (r, c) not in halo
    ]
    n = int(rng.integers(0, max_count + 1))
    if not free or n == 0:
        return
    idx = rng.choice(len(free), size=min(n, len(free)), replace=False).tolist()
    for i in idx:
        grid[free[int(i)]] = color


def _draw_labeled_shapes(
    grid: np.ndarray,
    shapes: list[Shape],
    origins: list[Coord],
    marker_colors: list[int],
    piece_color: int,
) -> list[Coord]:
    occupied_origins: list[Coord] = []
    for shape, origin, marker_color in zip(shapes, origins, marker_colors):
        cells = _absolute(shape, origin)
        object_origin = min(cells)
        marker = (object_origin[0] - 1, object_origin[1])
        occupied_origins.append(object_origin)
        grid[marker] = marker_color
        _stamp(grid, shape, origin, piece_color)
    return occupied_origins


def _candidate_shapes_for_match(base: Shape, relation: str, winner_index: int) -> list[Shape]:
    if relation == "same_shape_under_rotation":
        winner = _transform(base, "rotate90")
        decoys = [
            _expanded_shape(base),
            BASE_SHAPES[(BASE_SHAPES.index(base) + 1) % len(BASE_SHAPES)],
        ]
    elif relation == "same_shape_under_reflection":
        winner = _transform(base, "reflect_horizontal")
        decoys = [
            _expanded_shape(base),
            BASE_SHAPES[(BASE_SHAPES.index(base) + 2) % len(BASE_SHAPES)],
        ]
    elif relation == "same_cell_count":
        winner = BASE_SHAPES[(BASE_SHAPES.index(base) + 1) % len(BASE_SHAPES)]
        while len(winner) != len(base):
            winner = _transform(base, "rotate180")
            break
        decoys = [frozenset({(0, 0), (0, 1)}), frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)})]
    else:
        winner = base
        decoys = [
            _expanded_shape(base),
            BASE_SHAPES[(BASE_SHAPES.index(base) + 1) % len(BASE_SHAPES)],
        ]
    shapes = decoys[:]
    shapes.insert(winner_index, winner)
    return shapes[:3]


@register("253")
def generate_choose_equivalent_candidate_object(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Sample H and W independently across the ARC-aligned envelope. The
        # structural floor is set by the reference plus three separated marked
        # candidate pieces (each up to ~5 cells tall for the cell-count decoy).
        h = int(rng.integers(10, 23))
        w = int(rng.integers(10, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        ref_color, piece_color, *label_colors = unused_colors(rng, 5, bg)
        relation = str(rng.choice(["same_shape", "same_shape_under_rotation", "same_shape_under_reflection", "same_cell_count"]))
        base = BASE_SHAPES[int(rng.integers(0, 3))]
        winner_index = int(rng.integers(0, 3))
        shapes = _candidate_shapes_for_match(base, relation, winner_index)
        placement = _place_reference_and_candidates(
            rng, h=h, w=w, ref_shape=base, cand_shapes=shapes,
            ref_color=ref_color, piece_color=piece_color,
            marker_colors=label_colors, bg=bg,
        )
        if placement is None:
            continue
        grid, ref_origin, occupied_origins = placement
        # Inert distractors (a spare color, never the reference/candidate color)
        # vary the component count without affecting the shape-match answer.
        spare = unused_colors(rng, 1, bg, ref_color, piece_color, *label_colors)
        if spare:
            avoid = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != bg}
            _scatter_distractors(grid, rng, bg=bg, color=spare[0], avoid=avoid, max_count=6)
        # Optional patterned background (solid/patterned MIX). Texture excludes the
        # reference, candidate and every marker colour, so it never forms a phantom
        # candidate component nor overwrites a marker (markers are non-background).
        bg_style = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[ref_color, piece_color, *label_colors, *spare],
        )
        return {
            "input": grid_to_list(grid),
            "output": LABELS[winner_index],
            "metadata": {
                "background_color": bg,
                "reference_color": int(ref_color),
                "candidate_color": int(piece_color),
                "reference_origin": list(ref_origin),
                "reference_shape": [list(p) for p in sorted(base)],
                "equivalence_relation": relation,
                "candidate_labels": _candidate_meta(label_colors, occupied_origins),
                "label_encoding": "each candidate label is the marker color directly above that candidate object",
                "background_style": bg_style,
            },
        }


def _patch_matches(cand: np.ndarray, ref: np.ndarray, relation: str) -> bool:
    if relation == "exact_contents":
        return np.array_equal(cand, ref)
    if relation == "rotate_180_contents":
        return np.array_equal(cand, np.rot90(ref, 2))
    return sorted(int(v) for v in cand.ravel()) == sorted(int(v) for v in ref.ravel())


def _patch_variants(
    rng: np.random.Generator, patch: np.ndarray, relation: str, winner_index: int
) -> list[np.ndarray] | None:
    """Build the winner and two decoys for the patch-matching relation. Returns
    None if a unique-winner configuration could not be built for this patch."""
    ph, pw = patch.shape
    flat = patch.reshape(-1)
    n = flat.size

    def perm_diff_layout() -> np.ndarray | None:
        # a permutation of the cells that keeps the colour multiset but differs
        # from the reference array (so 'same histogram, different layout')
        for _ in range(40):
            order = rng.permutation(n)
            arr = flat[order].reshape(ph, pw)
            if not np.array_equal(arr, patch):
                return arr
        return None

    def mutate_one() -> np.ndarray | None:
        # change a single cell to a different present colour -> breaks exact and
        # histogram match while staying inside the content palette
        present = sorted(set(int(v) for v in flat))
        if len(present) < 2:
            return None
        for _ in range(40):
            idx = int(rng.integers(0, n))
            old = int(flat[idx])
            choices = [c for c in present if c != old]
            arr = flat.copy()
            arr[idx] = choices[int(rng.integers(0, len(choices)))]
            return arr.reshape(ph, pw)
        return None

    if relation == "exact_contents":
        winner = patch.copy()
    elif relation == "rotate_180_contents":
        winner = np.rot90(patch, 2).copy()
    else:
        w = perm_diff_layout()
        if w is None:
            return None
        winner = w

    decoys: list[np.ndarray] = []
    d1 = mutate_one()
    rot = np.rot90(patch, 2).copy()
    candidates_for_decoy = [d1, rot, np.full((ph, pw), int(flat[0]), dtype=patch.dtype)]
    for d in candidates_for_decoy:
        if d is None:
            continue
        if not _patch_matches(d, patch, relation):
            decoys.append(d)
        if len(decoys) == 2:
            break
    if len(decoys) < 2:
        return None
    variants = decoys[:2]
    variants.insert(winner_index, winner)
    variants = variants[:3]
    # gate: exactly the winner matches the relation
    matches = [i for i, v in enumerate(variants) if _patch_matches(v, patch, relation)]
    if matches != [winner_index]:
        return None
    return variants


def _place_patches(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    ph: int,
    pw: int,
    n: int,
    bg: int,
    max_attempts: int = 400,
) -> list[Coord] | None:
    """Randomly choose origins for `n` patches of size ph x pw plus a marker row
    above each, so footprints (with a one-cell halo) never overlap. Returns the
    list of top-left origins, or None."""
    occupied: set[Coord] = set()
    origins: list[Coord] = []

    def cells_of(r0: int, c0: int) -> set[Coord]:
        body = {(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)}
        body.add((r0 - 1, c0))  # marker
        return body

    for _ in range(max_attempts):
        occupied.clear()
        origins.clear()
        ok = True
        for _i in range(n):
            placed = False
            for _ in range(120):
                r0 = int(rng.integers(1, max(2, h - ph + 1)))
                c0 = int(rng.integers(0, max(1, w - pw + 1)))
                foot = cells_of(r0, c0)
                if any(not (0 <= r < h and 0 <= c < w) for r, c in foot):
                    continue
                if _dilate(foot) & occupied:
                    continue
                occupied.update(foot)
                origins.append((r0, c0))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return list(origins)
    return None


@register("254")
def generate_choose_matching_candidate_patch(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Independent H/W; patch dimensions vary too, which widens both the
        # grid-size envelope and the component-shape vocabulary.
        h = int(rng.integers(8, 23))
        w = int(rng.integers(8, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        ph = int(rng.integers(2, 4))
        pw = int(rng.integers(2, 4))
        # distinct content colours + 3 marker colours, drawn together so they
        # never collide; a spare distractor colour is drawn afterwards.
        n_content = int(rng.integers(2, min(5, ph * pw) + 1))
        drawn = unused_colors(rng, n_content + 3, bg)
        if len(drawn) < n_content + 3:
            continue
        palette = drawn[:n_content]
        label_colors = drawn[n_content:n_content + 3]
        relation = str(rng.choice(["exact_contents", "rotate_180_contents", "same_color_histogram"]))
        # Build a patch that uses every palette colour at least once (so the
        # histogram/exact decoys are realisable), arranged randomly.
        cells = list(palette)
        while len(cells) < ph * pw:
            cells.append(int(palette[int(rng.integers(0, len(palette)))]))
        rng.shuffle(cells)
        patch = np.array(cells, dtype=np.uint8).reshape(ph, pw)
        winner_index = int(rng.integers(0, 3))
        variants = _patch_variants(rng, patch, relation, winner_index)
        if variants is None:
            continue
        origins_all = _place_patches(rng, h=h, w=w, ph=ph, pw=pw, n=4, bg=bg)
        if origins_all is None:
            continue
        ref_origin = origins_all[0]
        cand_origins = origins_all[1:]
        grid = make_grid(h, w, fill=bg)
        grid[ref_origin[0]:ref_origin[0] + ph, ref_origin[1]:ref_origin[1] + pw] = patch
        candidate_bboxes = []
        for label_color, origin, candidate in zip(label_colors, cand_origins, variants):
            grid[origin[0] - 1, origin[1]] = label_color
            grid[origin[0]:origin[0] + ph, origin[1]:origin[1] + pw] = candidate
            candidate_bboxes.append([origin[0], origin[1], origin[0] + ph - 1, origin[1] + pw - 1])
        # inert distractors (a spare colour distinct from content + label colours)
        spare = unused_colors(rng, 1, bg, *palette, *label_colors)
        if spare:
            avoid = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != bg}
            _scatter_distractors(grid, rng, bg=bg, color=spare[0], avoid=avoid, max_count=6)
        # Optional patterned background (solid/patterned MIX). Texture excludes the
        # content and marker colours; every patch cell and marker is non-background,
        # so texture stays strictly outside the compared patch regions.
        bg_style = add_patterned_background(
            grid, rng, bg=bg, exclude_colors=[*palette, *label_colors, *spare]
        )
        return {
            "input": grid_to_list(grid),
            "output": LABELS[winner_index],
            "metadata": {
                "background_color": bg,
                "content_colors": [int(c) for c in palette],
                "reference_bbox": [ref_origin[0], ref_origin[1], ref_origin[0] + ph - 1, ref_origin[1] + pw - 1],
                "candidate_bboxes": candidate_bboxes,
                "match_relation": relation,
                "candidate_labels": _candidate_meta(label_colors, cand_origins),
                "label_encoding": "each candidate label is the marker color directly above that candidate region",
                "background_style": bg_style,
            },
        }
    raise RuntimeError("Task 254: failed to generate valid example")


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _spatial_winner(positions: list[Coord], ref: Coord, relation: str) -> int | None:
    """Index of the unique winning candidate, or None if it is not unique."""
    if relation in ("nearest_to_reference", "farthest_from_reference"):
        dists = [_manhattan(p, ref) for p in positions]
        target = min(dists) if relation == "nearest_to_reference" else max(dists)
        winners = [i for i, d in enumerate(dists) if d == target]
    elif relation == "same_row_right_of_reference":
        winners = [i for i, (r, c) in enumerate(positions) if r == ref[0] and c > ref[1]]
    else:  # below_same_column
        winners = [i for i, (r, c) in enumerate(positions) if c == ref[1] and r > ref[0]]
    return winners[0] if len(winners) == 1 else None


@register("255")
def generate_choose_spatial_relation_candidate(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Sample H and W independently over the ARC-aligned envelope (small grids
        # dominate); the reference + 3 marked candidate cells fit in any grid >=6.
        h = int(rng.integers(6, 21))
        w = int(rng.integers(6, 21))
        bg = int(rng.integers(0, NUM_COLORS))
        ref_color, candidate_color, *label_colors = unused_colors(rng, 5, bg)
        relation = str(rng.choice([
            "nearest_to_reference", "farthest_from_reference",
            "same_row_right_of_reference", "below_same_column",
        ]))
        # Reference cell with room for markers above candidates.
        ref = (int(rng.integers(2, h - 1)), int(rng.integers(1, w - 1)))
        # Sample 3 distinct candidate cells (row >= 1 so each has a marker above),
        # none coinciding with the reference, its marker row, or each other's
        # markers.
        used: set[Coord] = {ref}
        positions: list[Coord] = []
        attempts = 0
        while len(positions) < 3 and attempts < 400:
            attempts += 1
            pos = (int(rng.integers(1, h)), int(rng.integers(0, w)))
            marker = (pos[0] - 1, pos[1])
            if pos in used or marker in used or pos == ref:
                continue
            # avoid placing a candidate on another candidate's marker or vice versa
            if any(pos == m or marker == p for p, m in
                   ((p, (p[0] - 1, p[1])) for p in positions)):
                continue
            positions.append(pos)
            used.add(pos)
            used.add(marker)
        if len(positions) < 3:
            continue
        # _spatial_winner returns an index only when the relation selects exactly
        # one candidate (a strict-unique nearest/farthest under Manhattan, or a
        # single candidate satisfying the row/column predicate).
        winner_index = _spatial_winner(positions, ref, relation)
        if winner_index is None:
            continue
        grid = make_grid(h, w, fill=bg)
        grid[ref] = ref_color
        protected: set[Coord] = {ref}
        for color, pos in zip(label_colors, positions):
            grid[pos] = candidate_color
            grid[pos[0] - 1, pos[1]] = color
            protected.add(pos)
            protected.add((pos[0] - 1, pos[1]))
        # Inert distractor cells (a spare color, never the candidate color) vary
        # the component count and break the "every off-background cell matters"
        # shortcut. They are never the candidate color, so the candidate scan and
        # the chosen winner are unaffected.
        distractor_pool = unused_colors(rng, 1, bg, ref_color, candidate_color, *label_colors)
        if distractor_pool:
            distractor_color = distractor_pool[0]
            n_distract = int(rng.integers(0, 7))
            free = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if (r, c) not in protected and grid[r, c] == bg
            ]
            if free and n_distract:
                idx = rng.choice(len(free), size=min(n_distract, len(free)), replace=False).tolist()
                for i in idx:
                    grid[free[int(i)]] = distractor_color
        # Optional patterned background (solid/patterned MIX). Texture excludes the
        # candidate, reference, marker and distractor colours, so it cannot form a
        # phantom candidate cell nor overwrite a marker (markers are non-background).
        bg_style = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[ref_color, candidate_color, *label_colors, *distractor_pool],
        )
        return {
            "input": grid_to_list(grid),
            "output": LABELS[winner_index],
            "metadata": {
                "background_color": bg,
                "reference_color": int(ref_color),
                "candidate_color": int(candidate_color),
                "reference_position": list(ref),
                "spatial_relation": relation,
                "distance_metric": "manhattan",
                "candidate_labels": [
                    {"label": label, "marker_color": int(color), "position": list(pos)}
                    for label, color, pos in zip(LABELS, label_colors, positions)
                ],
                "label_encoding": "each candidate label is the marker color immediately above that candidate cell",
                "background_style": bg_style,
            },
        }
    raise RuntimeError("Task 255: failed to generate valid example")


@register("256")
def generate_choose_piece_covering_target(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Sample H and W independently; structural floor is the target slot plus
        # three separated marked pieces (each up to ~4 cells tall).
        h = int(rng.integers(9, 23))
        w = int(rng.integers(9, 23))
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, piece_color, *label_colors = unused_colors(rng, 5, bg)
        allowed = list([
            ["identity"],
            ["identity", "rotate90", "rotate180", "rotate270"],
            ["identity", "reflect_horizontal"],
        ][int(rng.integers(0, 3))])
        target_shape = BASE_SHAPES[int(rng.integers(0, len(BASE_SHAPES)))]
        winner_index = int(rng.integers(0, 3))
        winner_shape = _orientations(target_shape, allowed)[-1]
        decoys = [
            _expanded_shape(target_shape),
            BASE_SHAPES[(BASE_SHAPES.index(target_shape) + 1) % len(BASE_SHAPES)],
        ]
        shapes = decoys[:]
        shapes.insert(winner_index, winner_shape)
        shapes = shapes[:3]
        matches = [i for i, shape in enumerate(shapes) if target_shape in _orientations(shape, allowed)]
        if matches != [winner_index]:
            continue
        placement = _place_reference_and_candidates(
            rng, h=h, w=w, ref_shape=target_shape, cand_shapes=shapes,
            ref_color=target_color, piece_color=piece_color,
            marker_colors=label_colors, bg=bg,
        )
        if placement is None:
            continue
        grid, target_origin, occupied_origins = placement
        spare = unused_colors(rng, 1, bg, target_color, piece_color, *label_colors)
        if spare:
            avoid = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != bg}
            _scatter_distractors(grid, rng, bg=bg, color=spare[0], avoid=avoid, max_count=6)
        # Optional patterned background (solid/patterned MIX). Texture excludes the
        # target, piece and marker colours, so it never forms a phantom piece
        # component nor overwrites a marker (markers are non-background).
        bg_style = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[target_color, piece_color, *label_colors, *spare],
        )
        return {
            "input": grid_to_list(grid),
            "output": LABELS[winner_index],
            "metadata": {
                "background_color": bg,
                "target_color": int(target_color),
                "piece_color": int(piece_color),
                "target_origin": list(target_origin),
                "target_shape": [list(p) for p in sorted(target_shape)],
                "allowed_transforms": allowed,
                "cover_rule": "choose the unique labeled piece whose cells exactly cover the target under an allowed transform",
                "candidate_labels": _candidate_meta(label_colors, occupied_origins),
                "label_encoding": "each candidate label is the marker color directly above that candidate piece",
                "background_style": bg_style,
            },
        }
    raise RuntimeError("Task 256: failed to generate valid example")
