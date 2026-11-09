"""Motion and multi-grid second-batch task generators, slice G."""
from __future__ import annotations

from collections import Counter

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from common.objects import bounding_box, connected_components
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    boxes_overlap,
    unused_colors,
)
from registry import register


Cell = tuple[int, int]
Pattern = list[list[int]]


SOLID_SHAPES: list[frozenset[Cell]] = [
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}),
]

OBJECT_SHAPES: list[frozenset[Cell]] = [
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
]

VECTORS: list[Cell] = [
    (-4, 0), (-3, -2), (-3, 2), (-2, -3), (-2, 3), (0, -4),
    (0, 4), (2, -3), (2, 3), (3, -2), (3, 2), (4, 0),
]


def _sample_hw(
    rng: np.random.Generator, lo: int, hi: int, square_prob: float = 0.4
) -> tuple[int, int]:
    """Sample H and W in [lo, hi].

    With probability `square_prob` force H == W (ARC grids are ~62% square),
    otherwise draw H and W independently so the envelope is decorrelated. The
    resulting H-W correlation stays well under the profiler's 0.9 coupling
    threshold while keeping a healthy square fraction.
    """
    h = int(rng.integers(lo, hi + 1))
    if rng.random() < square_prob:
        return h, h
    return h, int(rng.integers(lo, hi + 1))


def _shape_size(shape: frozenset[Cell]) -> tuple[int, int]:
    r0, c0, r1, c1 = bounding_box(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _normalize_cells(cells: set[Cell]) -> set[Cell]:
    r0, c0, _r1, _c1 = bounding_box(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _paint_cells(grid: np.ndarray, cells: set[Cell], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _shift(cells: set[Cell], dr: int, dc: int) -> set[Cell]:
    return {(r + dr, c + dc) for r, c in cells}


def _in_bounds(cells: set[Cell], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _landmark(cells: set[Cell], name: str) -> Cell:
    r0, c0, r1, c1 = bounding_box(cells)
    if name == "top_left":
        return (r0, c0)
    if name == "top_right":
        return (r0, c1)
    if name == "bottom_left":
        return (r1, c0)
    if name == "bottom_right":
        return (r1, c1)
    if name == "medoid":
        ordered = sorted(cells)
        return min(
            ordered,
            key=lambda p: (
                sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in ordered),
                p[0],
                p[1],
            ),
        )
    raise ValueError(f"unknown landmark: {name}")


def _occupied_landmark(cells: set[Cell], name: str) -> Cell:
    if name == "top_left":
        return min(cells, key=lambda p: (p[0], p[1]))
    if name == "top_right":
        return min(cells, key=lambda p: (p[0], -p[1]))
    if name == "bottom_left":
        return min(cells, key=lambda p: (-p[0], p[1]))
    if name == "bottom_right":
        return min(cells, key=lambda p: (-p[0], -p[1]))
    return _landmark(cells, name)


def _place_shape_at(top: int, left: int, shape: frozenset[Cell]) -> set[Cell]:
    r0, c0, _r1, _c1 = bounding_box(shape)
    return {(top + r - r0, left + c - c0) for r, c in shape}


def _try_place_shape_pair(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    shape: frozenset[Cell],
    dr: int,
    dc: int,
    blocked: set[Cell],
    avoid_overlap: bool = True,
) -> tuple[set[Cell], set[Cell]] | None:
    sh, sw = _shape_size(shape)
    for _ in range(300):
        r0 = int(rng.integers(0, h - sh + 1))
        c0 = int(rng.integers(0, w - sw + 1))
        src = _place_shape_at(r0, c0, shape)
        dst = _shift(src, dr, dc)
        if not _in_bounds(dst, h, w):
            continue
        if src & blocked or dst & blocked:
            continue
        if avoid_overlap and src & dst:
            continue
        return src, dst
    return None


def _make_colored_pattern(rng: np.random.Generator, colors: list[int], h: int, w: int) -> np.ndarray:
    pattern = np.empty((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            pattern[r, c] = colors[(r * w + c + int(rng.integers(0, len(colors)))) % len(colors)]
    return pattern


def _stamp_pattern(grid: np.ndarray, top: int, left: int, pattern: np.ndarray) -> None:
    ph, pw = pattern.shape
    grid[top : top + ph, left : left + pw] = pattern


def _apply_transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "rotate_cw":
        return np.rot90(grid, k=-1).copy()
    if name == "rotate_ccw":
        return np.rot90(grid, k=1).copy()
    if name == "flip_horizontal":
        return np.fliplr(grid).copy()
    if name == "flip_vertical":
        return np.flipud(grid).copy()
    if name == "transpose_main":
        return grid.T.copy()
    if name == "transpose_anti":
        return np.rot90(grid.T, 2).copy()
    raise ValueError(f"unknown transform: {name}")


def _inverse_transform_name(name: str) -> str:
    return {
        "rotate_cw": "rotate_ccw",
        "rotate_ccw": "rotate_cw",
        "flip_horizontal": "flip_horizontal",
        "flip_vertical": "flip_vertical",
        "transpose_main": "transpose_main",
        "transpose_anti": "transpose_anti",
    }[name]


def _ring_slot_tops(h: int, w: int, n_slots: int, margin: int) -> list[Cell] | None:
    """Slot top-left coords on a rectangular ring (clockwise from top-left).

    Each slot occupies a 3x3 footprint: a 2x2 payload at (r, c)..(r+1, c+1)
    plus a marker at (r+2, c+2). Returns the clockwise cycle of payload
    top-lefts, or None if the grid is too small for a clean, non-overlapping
    ring with the requested slot count.
    """
    foot = 3  # 2x2 payload + marker corner
    top = margin
    bottom = h - foot - margin
    left = margin
    right = w - foot - margin
    if bottom <= top or right <= left:
        return None
    # Edge midpoints (centre of the inner span); keep them clear of corners.
    mid_r = (top + bottom) // 2
    mid_c = (left + right) // 2
    corners = {
        "tl": (top, left),
        "tr": (top, right),
        "br": (bottom, right),
        "bl": (bottom, left),
    }
    if n_slots == 4:
        cycle = [corners["tl"], corners["tr"], corners["br"], corners["bl"]]
    elif n_slots == 6:
        cycle = [
            corners["tl"], (top, mid_c), corners["tr"],
            corners["br"], (bottom, mid_c), corners["bl"],
        ]
    else:  # 8
        cycle = [
            corners["tl"], (top, mid_c), corners["tr"], (mid_r, right),
            corners["br"], (bottom, mid_c), corners["bl"], (mid_r, left),
        ]
    # Reject if any two footprints overlap or touch (need a clear gap so the
    # ring reads as distinct slots, and payloads never collide on copy).
    boxes = [(r, c, r + foot - 1, c + foot - 1) for r, c in cycle]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j], gap=1):
                return None
    return cycle


@register("128")
def generate_move_objects_to_next_marked_slot(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        marker_color, *payload_colors = unused_colors(rng, 7, bg)
        n_slots = int(rng.choice([4, 6, 8]))
        direction = str(rng.choice(["clockwise", "counterclockwise"]))
        # Independent (square-biased) H/W; lower bound large enough that the
        # ring's corner footprints clear the edge midpoints, upper bound capped
        # so that even a 20x20 single example keeps the grid->grid bundle >=2
        # (a 20x20 in+out spaced pair is ~2.7k tokens, under the 2900 budget
        # for n=2).
        h, w = _sample_hw(rng, 11, 20)
        margin = int(rng.integers(0, 3))
        cycle = _ring_slot_tops(h, w, n_slots, margin)
        if cycle is None:
            continue
        if direction == "counterclockwise":
            cycle = [cycle[0]] + list(reversed(cycle[1:]))

        grid = make_grid(h, w, fill=bg)
        slot_payloads: list[list[list[int]]] = []
        for i, (r0, c0) in enumerate(cycle):
            grid[r0 + 2, c0 + 2] = marker_color
            colors = [payload_colors[(i + j) % len(payload_colors)] for j in range(2)]
            payload = _make_colored_pattern(rng, colors, 2, 2)
            slot_payloads.append(grid_to_list(payload))
            _stamp_pattern(grid, r0, c0, payload)

        out = make_grid(h, w, fill=bg)
        for r0, c0 in cycle:
            out[r0 + 2, c0 + 2] = marker_color
        for i, payload in enumerate(slot_payloads):
            nr, nc = cycle[(i + 1) % n_slots]
            _stamp_pattern(out, nr, nc, np.array(payload, dtype=np.uint8))

        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "slot_marker_color": marker_color,
                "cycle_direction": direction,
                "slot_marker_positions": [[r + 2, c + 2] for r, c in cycle],
                "slot_payload_top_lefts": [[r, c] for r, c in cycle],
                "payload_size": [2, 2],
                "transfer_rule": "move each slot payload to the next marked slot and clear old payload cells",
            },
        }
    raise RuntimeError("Task 128: failed to generate valid slot ring")


@register("129")
def generate_undo_transform_sequence(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Square only by construction: rotations / transposes are in the
        # transform set and a non-square grid would change shape under them,
        # so H and W stay coupled. Widen the square span down toward the ARC
        # median (~10) and up to a size that keeps the grid->grid bundle >=3.
        n = int(rng.integers(5, 17))
        bg = int(rng.integers(0, NUM_COLORS))
        colors = unused_colors(rng, 4, bg)
        original = make_grid(n, n, fill=bg)
        for _dot in range(int(rng.integers(n + 5, n + 12))):
            r = int(rng.integers(0, n))
            c = int(rng.integers(0, n))
            original[r, c] = colors[int(rng.integers(0, len(colors)))]
        original[0, 1] = colors[0]
        original[1, 0] = colors[1]
        original[n - 2, n - 1] = colors[2]
        original[n - 1, n - 3] = colors[3]
        # The rule transforms the WHOLE grid uniformly (no object detection),
        # so a structured background carries through the transform and is
        # recovered exactly by the undo — it never spawns spurious objects.
        # Exclude the content colors so the dots stay readable on the texture.
        bg_style = add_patterned_background(original, rng, bg=bg, exclude_colors=colors)

        choices = ["rotate_cw", "rotate_ccw", "flip_horizontal", "flip_vertical", "transpose_main", "transpose_anti"]
        length = int(rng.integers(2, 5))
        sequence = [str(rng.choice(choices)) for _seq in range(length)]
        transformed = original.copy()
        for name in sequence:
            transformed = _apply_transform(transformed, name)
        if np.array_equal(transformed, original):
            continue
        inverse = [_inverse_transform_name(name) for name in reversed(sequence)]
        return {
            "input": grid_to_list(transformed),
            "output": grid_to_list(original),
            "metadata": {
                "background_color": bg,
                "applied_sequence": sequence,
                "undo_sequence": inverse,
                "transform_domain": "whole square grid",
                "background_style": bg_style["style"],
            },
        }
    raise RuntimeError("Task 129: failed to generate non-identity transform sequence")


@register("148")
def generate_overlay_priority_collision(rng: np.random.Generator) -> dict:
    # Per-cell overlay of two same-size grids; widen H/W (square-biased,
    # decorrelated) down to the structural minimum and up to a cap that keeps
    # the two-grid bundle >=2.
    h, w = _sample_hw(rng, 5, 16)
    bg = int(rng.integers(0, NUM_COLORS))
    colors = unused_colors(rng, 5, bg)
    rule = str(rng.choice(["top_priority", "bottom_priority", "collision_color"]))
    collision_color = colors[-1]
    bottom = make_grid(h, w, fill=bg)
    top = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            roll = float(rng.random())
            if roll < 0.18:
                bottom[r, c] = colors[int(rng.integers(0, 2))]
                top[r, c] = colors[int(rng.integers(2, 4))]
            elif roll < 0.34:
                bottom[r, c] = colors[int(rng.integers(0, 2))]
            elif roll < 0.50:
                top[r, c] = colors[int(rng.integers(2, 4))]
    bottom[1, 1] = colors[0]
    top[1, 1] = colors[2]

    out = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            b = int(bottom[r, c])
            t = int(top[r, c])
            if b == bg and t == bg:
                out[r, c] = bg
            elif b == bg:
                out[r, c] = t
            elif t == bg:
                out[r, c] = b
            elif rule == "top_priority":
                out[r, c] = t
            elif rule == "bottom_priority":
                out[r, c] = b
            else:
                out[r, c] = collision_color
    return {
        "input": {"bottom": grid_to_list(bottom), "top": grid_to_list(top)},
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "overlay_rule": rule,
            "collision_color": collision_color,
            "foreground_condition": "cell color is not the background color",
        },
    }


@register("175")
def generate_apply_marker_offset_to_target(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Square-biased, decorrelated H/W widened down toward the structural
        # minimum (must hold two markers an offset apart plus an object and its
        # shifted copy) and up to a token-safe cap. Vary the number of target
        # objects 1..3: the rule shifts every target-colored cell by the same
        # vector, so any count stays uniquely solvable. Cap the side at 18 so a
        # largest-case square example keeps the grid->grid bundle >=2.
        h, w = _sample_hw(rng, 9, 18)
        bg = int(rng.integers(0, NUM_COLORS))
        marker_a_color, marker_b_color, target_color = unused_colors(rng, 3, bg)
        dr, dc = VECTORS[int(rng.integers(0, len(VECTORS)))]
        ar = int(rng.integers(max(0, -dr), min(h, h - dr)))
        ac = int(rng.integers(max(0, -dc), min(w, w - dc)))
        marker_a = (ar, ac)
        marker_b = (ar + dr, ac + dc)
        n_objects = int(rng.integers(1, 4))
        blocked: set[Cell] = {marker_a, marker_b}
        srcs: list[set[Cell]] = []
        dsts: list[set[Cell]] = []
        for _obj in range(n_objects):
            shape = OBJECT_SHAPES[int(rng.integers(0, len(OBJECT_SHAPES)))]
            pair = _try_place_shape_pair(
                rng, h=h, w=w, shape=shape, dr=dr, dc=dc, blocked=blocked,
            )
            if pair is None:
                break
            src, dst = pair
            srcs.append(src)
            dsts.append(dst)
            # Block both the source and its landing so objects never collide
            # and a moved object never lands on another's source cells.
            blocked |= src | dst
        if len(srcs) != n_objects:
            continue
        operation = str(rng.choice(["move", "copy"]))
        grid = make_grid(h, w, fill=bg)
        grid[marker_a] = marker_a_color
        grid[marker_b] = marker_b_color
        for src in srcs:
            _paint_cells(grid, src, target_color)
        out = grid.copy()
        if operation == "move":
            for src in srcs:
                _paint_cells(out, src, bg)
            out[marker_a] = marker_a_color
            out[marker_b] = marker_b_color
        for dst in dsts:
            _paint_cells(out, dst, target_color)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "marker_a_color": marker_a_color,
                "marker_b_color": marker_b_color,
                "target_color": target_color,
                "marker_a_position": list(marker_a),
                "marker_b_position": list(marker_b),
                "offset_vector": [dr, dc],
                "operation": operation,
                "object_count": n_objects,
                "collision_rule": "generated so shifted target cells are background and in bounds; marker cells remain unchanged",
            },
        }
    raise RuntimeError("Task 175: failed to generate valid example")


@register("176")
def generate_align_object_landmark_to_anchor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Square-biased, decorrelated H/W widened down toward the structural
        # minimum (one solid object plus its translated copy, offset up to 4
        # cells) and up to a token-safe cap. Element count is intentionally
        # fixed at one movable object + one anchor: the rule aligns *the* object
        # to *the* anchor, so adding more objects/anchors would make the
        # object->anchor pairing ambiguous and break unique solvability.
        h, w = _sample_hw(rng, 8, 20)
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, anchor_color = unused_colors(rng, 2, bg)
        # Use only bounding-box corner landmarks: for the rectangular SOLID_SHAPES
        # these are unique occupied cells. "medoid" is dropped because its
        # min-total-Manhattan-distance set is non-singleton for 3 of the 4 shapes
        # (e.g. all 4 cells of a 2x2), so the hidden tie-break would make the
        # target placement ambiguous to a solver.
        landmark = str(rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"]))
        shape = SOLID_SHAPES[int(rng.integers(0, len(SOLID_SHAPES)))]
        dr, dc = VECTORS[int(rng.integers(0, len(VECTORS)))]
        pair = _try_place_shape_pair(rng, h=h, w=w, shape=shape, dr=dr, dc=dc, blocked=set())
        if pair is None:
            continue
        src, dst = pair
        anchor = _landmark(dst, landmark)
        if anchor in src:
            continue
        grid = make_grid(h, w, fill=bg)
        _paint_cells(grid, src, target_color)
        grid[anchor] = anchor_color
        out = make_grid(h, w, fill=bg)
        _paint_cells(out, dst, target_color)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": target_color,
                "anchor_color": anchor_color,
                "landmark": landmark,
                "anchor_position": list(anchor),
                "translation_vector": [anchor[0] - _landmark(src, landmark)[0], anchor[1] - _landmark(src, landmark)[1]],
                "anchor_handling": "the moved object overwrites the anchor marker cell",
            },
        }
    raise RuntimeError("Task 176: failed to generate valid example")


@register("179")
def generate_copy_source_region_to_target(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Square-biased, decorrelated H/W widened down toward the structural
        # minimum (two rectangular regions with a 2-cell gap) and up to a
        # token-safe cap; the retry loop rerolls region sizes when a tight grid
        # can't fit two.
        h, w = _sample_hw(rng, 9, 20)
        bg = int(rng.integers(0, NUM_COLORS))
        colors = unused_colors(rng, 6, bg)
        mode = str(rng.choice(["frame", "region"]))
        rh = int(rng.integers(3, 6))
        rw = int(rng.integers(3, 6))
        boxes: list[tuple[int, int, int, int]] = []
        placed: list[tuple[int, int]] = []
        for _place in range(300):
            r0 = int(rng.integers(1, h - rh))
            c0 = int(rng.integers(1, w - rw))
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(boxes_overlap(box, other, gap=2) for other in boxes):
                continue
            boxes.append(box)
            placed.append((r0, c0))
            if len(placed) == 2:
                break
        if len(placed) != 2:
            continue
        (sr, sc), (tr, tc) = placed
        grid = make_grid(h, w, fill=bg)
        out = grid.copy()
        pattern = _make_colored_pattern(rng, colors[:3], rh - 2 if mode == "frame" else rh, rw - 2 if mode == "frame" else rw)
        if mode == "frame":
            source_frame_color, target_frame_color = colors[3], colors[4]
            grid[sr : sr + rh, sc : sc + rw] = bg
            grid[tr : tr + rh, tc : tc + rw] = bg
            grid[sr, sc : sc + rw] = source_frame_color
            grid[sr + rh - 1, sc : sc + rw] = source_frame_color
            grid[sr : sr + rh, sc] = source_frame_color
            grid[sr : sr + rh, sc + rw - 1] = source_frame_color
            grid[tr, tc : tc + rw] = target_frame_color
            grid[tr + rh - 1, tc : tc + rw] = target_frame_color
            grid[tr : tr + rh, tc] = target_frame_color
            grid[tr : tr + rh, tc + rw - 1] = target_frame_color
            _stamp_pattern(grid, sr + 1, sc + 1, pattern)
            out = grid.copy()
            _stamp_pattern(out, tr + 1, tc + 1, pattern)
        else:
            source_marker_color, target_marker_color = colors[3], colors[4]
            _stamp_pattern(grid, sr, sc, pattern)
            source_corners = [(sr, sc), (sr, sc + rw - 1), (sr + rh - 1, sc), (sr + rh - 1, sc + rw - 1)]
            target_corners = [(tr, tc), (tr, tc + rw - 1), (tr + rh - 1, tc), (tr + rh - 1, tc + rw - 1)]
            for cell in source_corners:
                grid[cell] = source_marker_color
            for cell in target_corners:
                grid[cell] = target_marker_color
            out = grid.copy()
            out[tr : tr + rh, tc : tc + rw] = grid[sr : sr + rh, sc : sc + rw]
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "copy_mode": mode,
                "source_top_left": [sr, sc],
                "target_top_left": [tr, tc],
                "region_size": [rh, rw],
                "source_frame_color": colors[3],
                "target_frame_color": colors[4],
                "source_corner_positions": [list(p) for p in source_corners] if mode == "region" else None,
                "target_corner_positions": [list(p) for p in target_corners] if mode == "region" else None,
                "copy_rule": "frame mode copies only interiors; region mode copies the full same-size rectangle",
            },
        }
    raise RuntimeError("Task 179: failed to generate valid example")


@register("188")
def generate_apply_mask_to_source(rng: np.random.Generator) -> dict:
    # Per-cell mask gate over two same-size grids; widen H/W (square-biased,
    # decorrelated) down to the structural minimum and up to a cap that keeps
    # the two-grid bundle >=2.
    h, w = _sample_hw(rng, 5, 16)
    source_bg = int(rng.integers(0, NUM_COLORS))
    mask_bg, mask_color, fill_color = unused_colors(rng, 3, source_bg)
    source_colors = unused_colors(rng, 4, source_bg, mask_bg, mask_color, fill_color)
    source = make_grid(h, w, fill=source_bg)
    mask = make_grid(h, w, fill=mask_bg)
    condition = str(rng.choice(["mask_equals_color", "mask_non_background", "mask_not_color"]))
    for r in range(h):
        for c in range(w):
            if float(rng.random()) < 0.45:
                source[r, c] = source_colors[int(rng.integers(0, len(source_colors)))]
            if float(rng.random()) < 0.42:
                mask[r, c] = mask_color if float(rng.random()) < 0.65 else source_colors[int(rng.integers(0, len(source_colors)))]
    mask[0, 0] = mask_color
    source[0, 0] = source_colors[0]
    out = make_grid(h, w, fill=fill_color)
    for r in range(h):
        for c in range(w):
            m = int(mask[r, c])
            keep = (
                (condition == "mask_equals_color" and m == mask_color)
                or (condition == "mask_non_background" and m != mask_bg)
                or (condition == "mask_not_color" and m != mask_color)
            )
            if keep:
                out[r, c] = source[r, c]
    return {
        "input": {"source": grid_to_list(source), "mask": grid_to_list(mask)},
        "output": grid_to_list(out),
        "metadata": {
            "source_background_color": source_bg,
            "mask_background_color": mask_bg,
            "mask_color": mask_color,
            "fill_color": fill_color,
            "mask_condition": condition,
        },
    }


@register("189")
def generate_cellwise_consensus(rng: np.random.Generator) -> dict:
    # Per-cell consensus over N same-size grids. Widen H/W (square-biased,
    # decorrelated) down toward the structural minimum and up, but this is the
    # most token-heavy task in the file (up to 5 input grids + 1 output), so cap
    # the grid COUNT by the grid area: a single rendered example must stay under
    # the ~2900 token budget. ~2 chars/cell, (n_grids+1) grids => keep
    # area*(n+1) bounded.
    h, w = _sample_hw(rng, 4, 14)
    # tokens ~= 2*area*(n_grids+1)/1.2 + prompt; cap so single example fits.
    max_grids = max(2, min(5, int(2400 * 1.2 / (2 * h * w)) - 1))
    n_grids = int(rng.integers(2, max_grids + 1))
    bg = int(rng.integers(0, NUM_COLORS))
    colors = unused_colors(rng, 4, bg)
    tie_color = colors[-1]
    rule = str(rng.choice(["unanimous_non_background", "majority_non_background", "plurality_with_tie_color"]))
    grids = [make_grid(h, w, fill=bg) for _ in range(n_grids)]
    for r in range(h):
        for c in range(w):
            base = colors[int(rng.integers(0, 3))]
            mode = int(rng.integers(0, 4))
            if mode == 0:
                vals = [base] * n_grids
            elif mode == 1:
                vals = [base if i < (n_grids // 2 + 1) else bg for i in range(n_grids)]
            elif mode == 2:
                vals = [colors[(i + r + c) % 3] for i in range(n_grids)]
            else:
                vals = [bg] * n_grids
            rng.shuffle(vals)
            for i, value in enumerate(vals):
                grids[i][r, c] = value
    out = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            vals = [int(g[r, c]) for g in grids]
            counts = Counter(vals)
            if rule == "unanimous_non_background":
                out[r, c] = vals[0] if len(counts) == 1 and vals[0] != bg else bg
            elif rule == "majority_non_background":
                non_bg_counts = [(color, count) for color, count in counts.items() if color != bg]
                max_count = max((count for _color, count in non_bg_counts), default=0)
                winners = [color for color, count in non_bg_counts if count == max_count]
                out[r, c] = winners[0] if len(winners) == 1 and max_count > n_grids / 2 else bg
            else:
                ranked = counts.most_common()
                max_count = ranked[0][1]
                winners = [color for color, count in ranked if count == max_count]
                out[r, c] = winners[0] if len(winners) == 1 else tie_color
    payload = {f"grid_{i + 1}": grid_to_list(grid) for i, grid in enumerate(grids)}
    return {
        "input": payload,
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "grid_count": n_grids,
            "consensus_rule": rule,
            "tie_color": tie_color,
            "tie_rule": "plurality ties become the tie color; majority ties become background",
        },
    }


@register("194")
def generate_translate_scene_to_anchor(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Square-biased, decorrelated H/W widened down toward the structural
        # minimum (a small irregular scene plus its translated copy, offset up
        # to 4 cells) and up to a token-safe cap (side <=18 keeps a
        # largest-case square single example's grid->grid bundle >=2).
        h, w = _sample_hw(rng, 9, 18)
        bg = int(rng.integers(0, NUM_COLORS))
        colors = unused_colors(rng, 5, bg)
        anchor_color = colors[-1]
        # Restrict to the two corner landmarks that are unambiguous on an irregular
        # scene: the generator forces base[0,0] and base[scene_h-1,scene_w-1] to be
        # foreground, so the bounding-box top-left and bottom-right corners are always
        # occupied and unique. top_right/bottom_left are dropped because, for an
        # irregular scene, row-priority vs column-priority readings of the corner
        # disagree on ~1/3 of cases; "medoid" is dropped because its minimizing set is
        # frequently non-singleton, leaving the alignment cell (and thus the whole
        # translation) ambiguous to a solver.
        landmark = str(rng.choice(["top_left", "bottom_right"]))
        scene_h = int(rng.integers(3, 6))
        scene_w = int(rng.integers(3, 6))
        base = make_grid(scene_h, scene_w, fill=bg)
        for r in range(scene_h):
            for c in range(scene_w):
                if float(rng.random()) < 0.72:
                    base[r, c] = colors[int(rng.integers(0, 4))]
        base[0, 0] = colors[0]
        base[scene_h - 1, scene_w - 1] = colors[1]
        scene_cells = {(r, c) for r in range(scene_h) for c in range(scene_w) if int(base[r, c]) != bg}
        dr, dc = VECTORS[int(rng.integers(0, len(VECTORS)))]
        pair = _try_place_shape_pair(rng, h=h, w=w, shape=frozenset(scene_cells), dr=dr, dc=dc, blocked=set())
        if pair is None:
            continue
        src_cells, dst_cells = pair
        src_lm = _occupied_landmark(src_cells, landmark)
        dst_lm = (src_lm[0] + dr, src_lm[1] + dc)
        if dst_lm in src_cells:
            continue
        src_r0, src_c0, _src_r1, _src_c1 = bounding_box(src_cells)
        grid = make_grid(h, w, fill=bg)
        for r, c in scene_cells:
            grid[src_r0 + r, src_c0 + c] = base[r, c]
        grid[dst_lm] = anchor_color
        out = make_grid(h, w, fill=bg)
        for r, c in scene_cells:
            out[src_r0 + r + dr, src_c0 + c + dc] = base[r, c]
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "anchor_color": anchor_color,
                "landmark": landmark,
                "anchor_position": list(dst_lm),
                "translation_vector": [dr, dc],
                "scene_rule": "translate every non-background cell except the anchor marker as one scene",
                "anchor_handling": "the translated scene overwrites the anchor marker if it lands there",
            },
        }
    raise RuntimeError("Task 194: failed to generate valid example")


@register("195")
def generate_copy_prototype_to_marker_anchors(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Square-biased, decorrelated H/W widened down toward the structural
        # minimum (a small prototype plus 3-5 marker anchors, each a separated
        # box) and up to a token-safe cap (side <=18 keeps a largest-case square
        # single example's grid->grid bundle >=2); the retry loop rerolls when a
        # tight grid can't fit all the marker boxes.
        h, w = _sample_hw(rng, 9, 18)
        bg = int(rng.integers(0, NUM_COLORS))
        colors = unused_colors(rng, 6, bg)
        marker_color = colors[-1]
        # Use only bounding-box corner landmarks: the prototype is a full rectangle,
        # so all four corners are unique occupied cells. "medoid" is dropped because
        # its min-total-Manhattan-distance set is non-singleton for non-square
        # prototypes (e.g. all 4 cells of a 2x2), which would shift every copy by the
        # hidden tie-break and make the placement ambiguous to a solver.
        landmark = str(rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"]))
        ph = int(rng.integers(2, 4))
        pw = int(rng.integers(2, 4))
        pattern = _make_colored_pattern(rng, colors[:3], ph, pw)
        proto_cells = {(r, c) for r in range(ph) for c in range(pw)}
        proto_landmark = _landmark(proto_cells, landmark)
        boxes: list[tuple[int, int, int, int]] = []
        placements: list[tuple[int, int]] = []
        needed = int(rng.integers(3, 6)) + 1
        for _attempt in range(500):
            top = int(rng.integers(0, h - ph + 1))
            left = int(rng.integers(0, w - pw + 1))
            box = (top, left, top + ph - 1, left + pw - 1)
            if any(boxes_overlap(box, other, gap=1) for other in boxes):
                continue
            boxes.append(box)
            placements.append((top, left))
            if len(placements) == needed:
                break
        if len(placements) != needed:
            continue
        prototype_top_left = placements[0]
        marker_tops = placements[1:]
        grid = make_grid(h, w, fill=bg)
        _stamp_pattern(grid, prototype_top_left[0], prototype_top_left[1], pattern)
        marker_positions: list[Cell] = []
        for top, left in marker_tops:
            anchor = (top + proto_landmark[0], left + proto_landmark[1])
            grid[anchor] = marker_color
            marker_positions.append(anchor)
        out = grid.copy()
        for top, left in marker_tops:
            _stamp_pattern(out, top, left, pattern)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "marker_color": marker_color,
                "landmark": landmark,
                "prototype_top_left": list(prototype_top_left),
                "prototype_size": [ph, pw],
                "prototype_landmark_offset": list(proto_landmark),
                "marker_positions": [list(p) for p in marker_positions],
                "copy_rule": "copy the prototype to every marker; the copied prototype overwrites marker cells",
            },
        }
    raise RuntimeError("Task 195: failed to generate valid example")
