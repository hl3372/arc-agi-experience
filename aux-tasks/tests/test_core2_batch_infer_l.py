"""Verifier tests for rule-inference batch L tasks."""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate
from sft_wrapper import format_single


TASK_IDS = ["211", "212", "213", "214", "215", "216", "217", "218", "219", "220"]
N_SEEDS = 30
PALETTE = tuple("abcdefghij")
DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

RAY_LABELS = [
    "ray_until_wall_E",
    "ray_until_wall_NE",
    "ray_until_edge_S",
    "ray_hit_wall_SW",
]
OVERLAY_LABELS = [
    "overlay_top_priority",
    "overlay_bottom_priority",
    "overlay_intersection_only",
    "overlay_union_recolor",
]
MARKER_LABELS = [
    "copy_to_markers_top_left",
    "copy_to_markers_center",
    "move_to_marker_top_left",
    "move_to_marker_center",
    "stamp_cross_on_markers_center",
]
SCALE_LABELS = [
    "scale_2x",
    "tile_2x2",
    "repeat_rows_3",
    "block_compress_2x",
]
EXTRACT_LABELS = [
    "extract_bbox_target_color",
    "extract_inside_frame",
    "extract_marker_box_interior",
    "extract_top_left_quadrant",
]
MASK_LABELS = [
    "mask_union",
    "mask_intersection",
    "mask_xor",
    "mask_a_minus_b",
    "mask_b_minus_a",
]
PROPERTY_LABELS = [
    "property_size_rank",
    "property_bbox_width",
    "property_touching_border",
    "property_hole_count",
]
MORPH_LABELS = [
    "dilate_1",
    "dilate_2",
    "erode_1",
    "open_1",
    "close_1",
]
REGION_LABELS = [
    "fill_enclosed_regions",
    "fill_border_connected_background",
    "fill_marker_adjacent_region",
    "fill_smallest_background_region",
]
PACK_LABELS = [
    "pack_left_to_right_by_area",
    "pack_left_to_right_by_width",
    "pack_top_to_bottom_by_area",
    "pack_top_to_bottom_by_width",
]

MOVE_SHAPE = frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)})
STAMP_CROSS = frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)})


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        assert isinstance(ex["input"], dict), seed
        assert isinstance(ex["output"], str), seed
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed in {0, 7, 13}:
            rendered_prompt, answer = format_single(task_id, seed, color_palette=PALETTE)
            assert "[No prompt" not in rendered_prompt
            assert answer == ex["output"]
        fn(seed, ex)
        assert generate(task_id, seed) == ex


def _arr(grid) -> np.ndarray:
    return np.array(grid, dtype=np.uint8)


def _coords(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    return set(map(tuple, np.argwhere(grid == color).tolist()))


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(set(cells))
    return frozenset((r - r0, c - c0) for r, c in cells)


def _stamp(grid: np.ndarray, shape: set[tuple[int, int]] | frozenset[tuple[int, int]], top: int, left: int, color: int) -> bool:
    h, w = grid.shape
    for r, c in shape:
        nr, nc = top + r, left + c
        if not (0 <= nr < h and 0 <= nc < w):
            return False
    for r, c in shape:
        grid[top + r, left + c] = color
    return True


def _components(grid: np.ndarray, bg: int | None = None, color: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            comp_color = int(grid[r, c])
            if (bg is not None and comp_color == bg) or (color is not None and comp_color != color):
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == comp_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": comp_color, "cells": cells, "bbox": _bbox(cells)})
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))


def _line_transform(before: np.ndarray, meta: dict, label: str) -> np.ndarray | None:
    source = meta["source_color"]
    wall = meta["wall_color"]
    ray = meta["ray_color"]
    sources = sorted(_coords(before, source))
    if not sources:
        return None
    out = before.copy()
    if label == "ray_until_wall_E":
        direction = (0, 1)
        stop = "wall_draw"
    elif label == "ray_until_wall_NE":
        direction = (-1, 1)
        stop = "wall_draw"
    elif label == "ray_until_edge_S":
        direction = (1, 0)
        stop = "edge_draw"
    else:
        direction = (1, -1)
        stop = "wall_hit"

    h, w = before.shape
    for sr, sc in sources:
        r, c = sr + direction[0], sc + direction[1]
        path = []
        hit_wall = None
        while 0 <= r < h and 0 <= c < w:
            if int(before[r, c]) == wall:
                hit_wall = (r, c)
                break
            if int(before[r, c]) != meta["background_color"]:
                return None
            path.append((r, c))
            r += direction[0]
            c += direction[1]
        if stop == "edge_draw":
            if hit_wall is not None:
                return None
            for cell in path:
                out[cell] = ray
        elif stop == "wall_draw":
            if hit_wall is None:
                return None
            for cell in path:
                out[cell] = ray
        else:
            if hit_wall is None:
                return None
            out[hit_wall] = ray
    return out


def test_task_211():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        matches = [
            label
            for label in RAY_LABELS
            if (candidate := _line_transform(before, ex["metadata"], label)) is not None
            and candidate.tolist() == after.tolist()
        ]
        assert matches == [ex["output"]], (seed, matches, ex["output"])
    _verify("211", check)


def test_task_212():
    def check(seed, ex):
        a = _arr(ex["input"]["source_a"])
        b = _arr(ex["input"]["source_b"])
        composite = _arr(ex["input"]["composite"])
        meta = ex["metadata"]
        bg = meta["background_color"]
        expected = {}
        top = np.full_like(a, bg)
        top[b != bg] = meta["source_b_color"]
        top[a != bg] = meta["source_a_color"]
        expected["overlay_top_priority"] = top
        bottom = np.full_like(a, bg)
        bottom[a != bg] = meta["source_a_color"]
        bottom[b != bg] = meta["source_b_color"]
        expected["overlay_bottom_priority"] = bottom
        intersection = np.full_like(a, bg)
        intersection[(a != bg) & (b != bg)] = meta["result_color"]
        expected["overlay_intersection_only"] = intersection
        union = np.full_like(a, bg)
        union[(a != bg) | (b != bg)] = meta["result_color"]
        expected["overlay_union_recolor"] = union
        matches = [label for label in OVERLAY_LABELS if expected[label].tolist() == composite.tolist()]
        assert matches == [ex["output"]], (seed, matches)
    _verify("212", check)


def _marker_candidate(before: np.ndarray, meta: dict, label: str) -> np.ndarray | None:
    bg = meta["background_color"]
    object_color = meta["object_color"]
    marker_color = meta["marker_color"]
    stamp_color = meta["stamp_color"]
    markers = sorted(_coords(before, marker_color))
    if label == "stamp_cross_on_markers_center":
        if not markers or _coords(before, object_color):
            return None
        out = before.copy()
        for mr, mc in markers:
            for dr, dc in STAMP_CROSS:
                nr, nc = mr + dr, mc + dc
                if not (0 <= nr < before.shape[0] and 0 <= nc < before.shape[1]):
                    return None
                out[nr, nc] = stamp_color
        return out

    object_cells = _coords(before, object_color)
    comps = _components(before, bg=bg, color=object_color)
    if len(comps) != 1 or not markers:
        return None
    if label.startswith("copy_") and len(markers) < 2:
        return None
    if label.startswith("move_") and len(markers) != 1:
        return None
    source_shape = _normalize(object_cells)
    h = max(r for r, _ in source_shape) + 1
    w = max(c for _, c in source_shape) + 1
    out = before.copy()
    if label.startswith("move_"):
        for cell in object_cells:
            out[cell] = bg
    for mr, mc in markers:
        if label.endswith("_center"):
            top, left = mr - h // 2, mc - w // 2
        else:
            top, left = mr, mc
        if not _stamp(out, source_shape, top, left, object_color):
            return None
    return out


def test_task_213():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        matches = [
            label
            for label in MARKER_LABELS
            if (candidate := _marker_candidate(before, ex["metadata"], label)) is not None
            and candidate.tolist() == after.tolist()
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("213", check)


def _scale_candidate(before: np.ndarray, label: str) -> np.ndarray | None:
    if label == "scale_2x":
        return np.repeat(np.repeat(before, 2, axis=0), 2, axis=1)
    if label == "tile_2x2":
        return np.tile(before, (2, 2))
    if label == "repeat_rows_3":
        return np.repeat(before, 3, axis=0)
    if before.shape[0] % 2 or before.shape[1] % 2:
        return None
    compressed = before[::2, ::2]
    if np.repeat(np.repeat(compressed, 2, axis=0), 2, axis=1).tolist() != before.tolist():
        return None
    return compressed


def test_task_214():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        matches = [
            label
            for label in SCALE_LABELS
            if (candidate := _scale_candidate(before, label)) is not None
            and candidate.tolist() == after.tolist()
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("214", check)


def _extract_candidate(source: np.ndarray, meta: dict, label: str) -> np.ndarray | None:
    if label == "extract_bbox_target_color":
        cells = _coords(source, meta["target_color"])
        if not cells:
            return None
        r0, c0, r1, c1 = _bbox(cells)
        return source[r0:r1 + 1, c0:c1 + 1]
    if label == "extract_inside_frame":
        cells = _coords(source, meta["frame_color"])
        if not cells:
            return None
        r0, c0, r1, c1 = _bbox(cells)
        return source[r0 + 1:r1, c0 + 1:c1]
    if label == "extract_marker_box_interior":
        cells = sorted(_coords(source, meta["marker_color"]))
        if len(cells) != 2:
            return None
        rows = sorted([cells[0][0], cells[1][0]])
        cols = sorted([cells[0][1], cells[1][1]])
        return source[rows[0] + 1:rows[1], cols[0] + 1:cols[1]]
    return source[: source.shape[0] // 2, : source.shape[1] // 2]


def test_task_215():
    def check(seed, ex):
        source = _arr(ex["input"]["source"])
        extracted = _arr(ex["input"]["extracted"])
        matches = [
            label
            for label in EXTRACT_LABELS
            if (candidate := _extract_candidate(source, ex["metadata"], label)) is not None
            and candidate.tolist() == extracted.tolist()
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("215", check)


def test_task_216():
    def check(seed, ex):
        a = _arr(ex["input"]["source_a"])
        b = _arr(ex["input"]["source_b"])
        combined = _arr(ex["input"]["combined"])
        meta = ex["metadata"]
        bg = meta["background_color"]
        mask_a = a != bg
        mask_b = b != bg
        masks = {
            "mask_union": mask_a | mask_b,
            "mask_intersection": mask_a & mask_b,
            "mask_xor": mask_a ^ mask_b,
            "mask_a_minus_b": mask_a & ~mask_b,
            "mask_b_minus_a": mask_b & ~mask_a,
        }
        matches = []
        for label, mask in masks.items():
            expected = np.full_like(a, bg)
            expected[mask] = meta["result_color"]
            if expected.tolist() == combined.tolist():
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
    _verify("216", check)


def _hole_count(grid: np.ndarray, comp: dict, bg: int) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(sub[r, c]) != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                touches_edge |= cr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(sub[nr, nc]) == bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _property_values(before: np.ndarray, comps: list[dict], bg: int, label: str) -> list[int]:
    if label == "property_size_rank":
        sizes = [len(comp["cells"]) for comp in comps]
        rank = {size: idx for idx, size in enumerate(sorted(set(sizes)), start=1)}
        return [rank[size] for size in sizes]
    if label == "property_bbox_width":
        return [comp["bbox"][3] - comp["bbox"][1] + 1 for comp in comps]
    if label == "property_touching_border":
        h, w = before.shape
        return [
            int(any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp["cells"]))
            for comp in comps
        ]
    return [_hole_count(before, comp, bg) for comp in comps]


def _property_matches(before: np.ndarray, after: np.ndarray, meta: dict, label: str) -> bool:
    bg = meta["background_color"]
    comps = _components(before, bg=bg, color=meta["object_color"])
    if len(comps) < 2:
        return False
    object_cells = set().union(*(comp["cells"] for comp in comps))
    for r in range(before.shape[0]):
        for c in range(before.shape[1]):
            if (r, c) not in object_cells and int(before[r, c]) != int(after[r, c]):
                return False
    values = _property_values(before, comps, bg, label)
    if len(set(values)) < 2:
        return False
    value_to_color: dict[int, int] = {}
    color_to_value: dict[int, int] = {}
    for comp, value in zip(comps, values):
        after_colors = {int(after[cell]) for cell in comp["cells"]}
        if len(after_colors) != 1:
            return False
        color = after_colors.pop()
        if value in value_to_color and value_to_color[value] != color:
            return False
        if color in color_to_value and color_to_value[color] != value:
            return False
        value_to_color[value] = color
        color_to_value[color] = value
    return len(value_to_color) == len(color_to_value)


def test_task_217():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        matches = [label for label in PROPERTY_LABELS if _property_matches(before, after, ex["metadata"], label)]
        assert matches == [ex["output"]], (seed, matches)
    _verify("217", check)


def _dilate(mask: set[tuple[int, int]], h: int, w: int, radius: int) -> set[tuple[int, int]]:
    out = set(mask)
    for _ in range(radius):
        expanded = set(out)
        for r, c in out:
            for dr, dc in DELTAS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
        out = expanded
    return out


def _erode(mask: set[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out = set(mask)
    for _ in range(radius):
        out = {
            (r, c)
            for r, c in out
            if all((r + dr, c + dc) in out for dr, dc in [(0, 0), *DELTAS_4])
        }
    return out


def _morph(mask: set[tuple[int, int]], h: int, w: int, label: str) -> set[tuple[int, int]]:
    if label == "dilate_1":
        return _dilate(mask, h, w, 1)
    if label == "dilate_2":
        return _dilate(mask, h, w, 2)
    if label == "erode_1":
        return _erode(mask, 1)
    if label == "open_1":
        return _dilate(_erode(mask, 1), h, w, 1)
    if label == "close_1":
        return _erode(_dilate(mask, h, w, 1), 1)
    raise AssertionError(label)


def test_task_218():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        meta = ex["metadata"]
        mask = _coords(before, meta["foreground_color"])
        matches = []
        for label in MORPH_LABELS:
            expected = np.full_like(before, meta["background_color"])
            for cell in _morph(mask, before.shape[0], before.shape[1], label):
                expected[cell] = meta["foreground_color"]
            if expected.tolist() == after.tolist():
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
    _verify("218", check)


def _background_components(grid: np.ndarray, bg: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _region_candidate(before: np.ndarray, meta: dict, label: str) -> np.ndarray:
    bg = meta["background_color"]
    comps = _background_components(before, bg)
    h, w = before.shape
    markers = _coords(before, meta["marker_color"])
    selected: set[tuple[int, int]] = set()
    if label == "fill_enclosed_regions":
        for comp in comps:
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                selected |= comp
    elif label == "fill_border_connected_background":
        for comp in comps:
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                selected |= comp
    elif label == "fill_marker_adjacent_region":
        for comp in comps:
            if any((r + dr, c + dc) in markers for r, c in comp for dr, dc in DELTAS_4):
                selected |= comp
    else:
        selected = set(min(comps, key=len))
    out = before.copy()
    for cell in selected:
        out[cell] = meta["fill_color"]
    return out


def test_task_219():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        matches = [
            label
            for label in REGION_LABELS
            if _region_candidate(before, ex["metadata"], label).tolist() == after.tolist()
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("219", check)


def _pack_candidate(before: np.ndarray, bg: int, label: str) -> np.ndarray:
    comps = _components(before, bg=bg)
    if label.endswith("_by_area"):
        key_fn = lambda comp: len(comp["cells"])
    else:
        key_fn = lambda comp: comp["bbox"][3] - comp["bbox"][1] + 1
    ordered = sorted(comps, key=lambda comp: (-key_fn(comp), comp["bbox"][0], comp["bbox"][1]))
    out = np.full_like(before, bg)
    cursor = 1
    horizontal = label.startswith("pack_left_to_right")
    for comp in ordered:
        shape = _normalize(comp["cells"])
        if horizontal:
            top, left = 1, cursor
            width = max(c for _, c in shape) + 1
            cursor += width + 1
        else:
            top, left = cursor, 1
            height = max(r for r, _ in shape) + 1
            cursor += height + 1
        assert _stamp(out, shape, top, left, comp["color"])
    return out


def test_task_220():
    def check(seed, ex):
        before = _arr(ex["input"]["before"])
        after = _arr(ex["input"]["after"])
        bg = ex["metadata"]["background_color"]
        comps = _components(before, bg=bg)
        areas = [len(comp["cells"]) for comp in comps]
        widths = [comp["bbox"][3] - comp["bbox"][1] + 1 for comp in comps]
        assert len(areas) == len(set(areas)), seed
        assert len(widths) == len(set(widths)), seed
        prompt = get_prompt("220", ex["metadata"]).lower()
        metadata_text = json.dumps(ex["metadata"]).lower()
        ordinal_terms = (
            "color index",
            "color order",
            "color ordinal",
            "lower color",
            "greater color",
            "earlier color",
            "later color",
        )
        assert not any(term in prompt for term in ordinal_terms), seed
        assert not any(term in metadata_text for term in ordinal_terms), seed
        matches = [
            label
            for label in PACK_LABELS
            if _pack_candidate(before, bg, label).tolist() == after.tolist()
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("220", check)


def test_candidate_label_metadata_is_isolated():
    for task_id in TASK_IDS:
        ex = generate(task_id, 0)
        labels = ex["metadata"]["candidate_labels"]
        labels.append("__bad_label__")
        fresh = generate(task_id, 0)
        fresh_labels = fresh["metadata"]["candidate_labels"]
        assert "__bad_label__" not in fresh_labels, task_id
        assert fresh["output"] in fresh_labels, task_id


if __name__ == "__main__":
    tests = [
        test_task_211,
        test_task_212,
        test_task_213,
        test_task_214,
        test_task_215,
        test_task_216,
        test_task_217,
        test_task_218,
        test_task_219,
        test_task_220,
        test_candidate_label_metadata_is_isolated,
    ]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
