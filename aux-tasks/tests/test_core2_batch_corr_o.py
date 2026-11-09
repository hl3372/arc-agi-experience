"""Verifier tests for correspondence/change batch O tasks."""
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


TASK_IDS = ["236", "237", "238", "239", "240", "241", "242", "243"]
N_SEEDS = 30
PALETTE = tuple("abcdefghij")
PROPERTY_ORDER = [
    "color_preserved",
    "shape_preserved",
    "size_preserved",
    "position_preserved",
    "connectedness_preserved",
    "slot_preserved",
]
GLOBAL_PROPERTY_ORDER = [
    "background_color_preserved",
    "grid_size_preserved",
    "label_layout_preserved",
    "object_count_preserved",
    "slot_order_preserved",
]


def _verify(task_id: str, fn) -> None:
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed < 4:
            user_text, answer_text = format_single(task_id, seed, color_palette=PALETTE)
            assert user_text and answer_text
        fn(seed, ex)
        assert generate(task_id, seed) == ex


def _slot_bounds(meta: dict, index: int) -> tuple[int, int]:
    left = index * (meta["slot_width"] + meta["slot_gap"])
    return left, left + meta["slot_width"]


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _is_connected(shape: frozenset[tuple[int, int]]) -> bool:
    if not shape:
        return False
    q = deque([next(iter(shape))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in shape and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen == set(shape)


def _scale_shape(shape: frozenset[tuple[int, int]], factor: int = 2) -> frozenset[tuple[int, int]]:
    return frozenset(
        (r * factor + dr, c * factor + dc)
        for r, c in shape
        for dr in range(factor)
        for dc in range(factor)
    )


def _parse_scene(grid: np.ndarray, meta: dict, bg: int | None = None) -> list[dict | None]:
    objects: list[dict | None] = []
    if bg is None:
        bg = meta["background_color"]
    for index in range(meta["num_slots"]):
        left, right = _slot_bounds(meta, index)
        cells = {
            (int(r), int(c))
            for r in range(meta["object_top_row"], grid.shape[0])
            for c in range(left, right)
            if int(grid[r, c]) != bg
        }
        if not cells:
            objects.append(None)
            continue
        colors = {int(grid[cell]) for cell in cells}
        assert len(colors) == 1
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        objects.append({
            "cells": cells,
            "color": colors.pop(),
            "anchor": (min(rows), min(cols)),
            "shape": _normalize(cells),
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
        })
    return objects


def _pair_objects(ex: dict) -> tuple[list[dict | None], list[dict | None], dict]:
    meta = ex["metadata"]
    before = np.array(ex["input"]["input_scene"], dtype=np.uint8)
    after = np.array(ex["input"]["output_scene"], dtype=np.uint8)
    return _parse_scene(before, meta), _parse_scene(after, meta), meta


def _change_type(before: dict | None, after: dict | None) -> str:
    if before is None and after is None:
        return "none"
    if before is None:
        return "new"
    if after is None:
        return "deleted"
    same_color = before["color"] == after["color"]
    same_shape = before["shape"] == after["shape"]
    same_anchor = before["anchor"] == after["anchor"]
    if same_color and same_shape and same_anchor:
        return "unchanged"
    if same_color and same_shape and not same_anchor:
        return "translated"
    if not same_color and same_shape and same_anchor:
        return "recolored"
    if same_color and same_anchor and after["shape"] == _scale_shape(before["shape"], 2):
        return "scaled_2x"
    return "changed"


def _class_line(before: list[dict | None], after: list[dict | None], meta: dict, detailed: bool) -> str:
    parts = []
    for index, label in enumerate(meta["input_labels"]):
        change = _change_type(before[index], after[index])
        value = change if detailed else ("unchanged" if change == "unchanged" else "changed")
        parts.append(f"{label}:{value}")
    return ";".join(parts)


def _descriptor(before: dict, after: dict, change: str) -> str:
    if change == "translated":
        dr = after["anchor"][0] - before["anchor"][0]
        dc = after["anchor"][1] - before["anchor"][1]
        return f"translated(dr={dr},dc={dc})"
    if change == "recolored":
        return "recolored(to_output_color)"
    if change == "scaled_2x":
        return "scaled_2x(factor=2)"
    return change


def _properties(before: dict, after: dict) -> tuple[list[str], list[str]]:
    preserved = []
    changed = []
    if before["color"] == after["color"]:
        preserved.append("color_preserved")
    else:
        changed.append("color_changed")
    if before["shape"] == after["shape"]:
        preserved.append("shape_preserved")
    else:
        changed.append("shape_changed")
    if len(before["shape"]) == len(after["shape"]):
        preserved.append("size_preserved")
    else:
        changed.append("size_changed")
    if before["anchor"] == after["anchor"]:
        preserved.append("position_preserved")
    else:
        changed.append("position_changed")
    if _is_connected(before["shape"]) == _is_connected(after["shape"]):
        preserved.append("connectedness_preserved")
    preserved.append("slot_preserved")
    return [label for label in PROPERTY_ORDER if label in preserved], changed


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _relation_preserved(before: list[dict | None], after: list[dict | None], candidate: str) -> bool:
    before_objs = [obj for obj in before if obj is not None]
    after_objs = [obj for obj in after if obj is not None]
    assert len(before_objs) == len(after_objs)
    if candidate == "left_to_right_order":
        before_order = sorted(range(len(before_objs)), key=lambda i: before_objs[i]["bbox"][1])
        after_order = sorted(range(len(after_objs)), key=lambda i: after_objs[i]["bbox"][1])
        return before_order == after_order
    if candidate == "top_row_order":
        return all(
            _sign(before_objs[i]["bbox"][0] - before_objs[j]["bbox"][0])
            == _sign(after_objs[i]["bbox"][0] - after_objs[j]["bbox"][0])
            for i in range(len(before_objs))
            for j in range(i + 1, len(before_objs))
        )
    if candidate == "cell_count_order":
        return all(
            _sign(len(before_objs[i]["shape"]) - len(before_objs[j]["shape"]))
            == _sign(len(after_objs[i]["shape"]) - len(after_objs[j]["shape"]))
            for i in range(len(before_objs))
            for j in range(i + 1, len(before_objs))
        )
    assert candidate == "color_equality_pattern"
    return all(
        (before_objs[i]["color"] == before_objs[j]["color"])
        == (after_objs[i]["color"] == after_objs[j]["color"])
        for i in range(len(before_objs))
        for j in range(i + 1, len(before_objs))
    )


def test_task_236() -> None:
    def check(seed, ex):
        before, after, meta = _pair_objects(ex)
        assert ex["output"] == _class_line(before, after, meta, detailed=False), seed
    _verify("236", check)


def test_task_237() -> None:
    def check(seed, ex):
        before, after, meta = _pair_objects(ex)
        assert ex["output"] == _class_line(before, after, meta, detailed=True), seed
    _verify("237", check)


def test_task_238() -> None:
    def check(seed, ex):
        before, after, meta = _pair_objects(ex)
        index = meta["input_labels"].index(meta["selected_input_label"])
        assert before[index] is not None and after[index] is not None
        change = _change_type(before[index], after[index])
        expected = (
            f"{meta['input_labels'][index]}->{meta['output_labels'][index]}:"
            f"{_descriptor(before[index], after[index], change)}"
        )
        assert ex["output"] == expected, seed
        assert "->output_label" not in get_prompt("238", meta)
    _verify("238", check)


def test_task_239() -> None:
    def check(seed, ex):
        meta = ex["metadata"]
        before_grid = np.array(ex["input"]["input_scene"], dtype=np.uint8)
        after_grid = np.array(ex["input"]["output_scene"], dtype=np.uint8)
        before = _parse_scene(before_grid, meta)
        after = _parse_scene(after_grid, meta)
        expected = np.full(before_grid.shape, meta["background_color"], dtype=np.uint8)
        layout = (before_grid == meta["label_marker_color"]) | (before_grid == meta["divider_color"])
        expected[layout] = before_grid[layout]
        category_colors = {entry["category"]: entry["color"] for entry in meta["category_to_color"]}
        for index in range(meta["num_slots"]):
            b_obj = before[index]
            a_obj = after[index]
            if b_obj is not None and a_obj is not None:
                change = _change_type(b_obj, a_obj)
                category = "unchanged" if change == "unchanged" else "changed"
                cells = a_obj["cells"]
            elif b_obj is not None:
                category = "deleted"
                cells = b_obj["cells"]
            elif a_obj is not None:
                category = "new"
                cells = a_obj["cells"]
            else:
                continue
            for cell in cells:
                expected[cell] = category_colors[category]
        assert np.array(ex["output"], dtype=np.uint8).tolist() == expected.tolist(), seed
    _verify("239", check)


def test_task_240() -> None:
    def check(seed, ex):
        before, after, meta = _pair_objects(ex)
        index = meta["input_labels"].index(meta["selected_input_label"])
        assert before[index] is not None and after[index] is not None
        preserved, changed = _properties(before[index], after[index])
        preserved_text = ",".join(preserved) if preserved else "none"
        changed_text = ",".join(changed) if changed else "none"
        expected = (
            f"{meta['input_labels'][index]}->{meta['output_labels'][index]}:"
            f"preserved={preserved_text};changed={changed_text}"
        )
        assert ex["output"] == expected, seed
        assert "->output_label" not in get_prompt("240", meta)
    _verify("240", check)


def test_task_241() -> None:
    def check(seed, ex):
        before, after, _meta = _pair_objects(ex)
        common = set(PROPERTY_ORDER)
        for b_obj, a_obj in zip(before, after):
            assert b_obj is not None and a_obj is not None
            preserved, _changed = _properties(b_obj, a_obj)
            common &= set(preserved)
        expected = ";".join(label for label in PROPERTY_ORDER if label in common) or "none"
        assert ex["output"] == expected, seed
    _verify("241", check)


def test_task_242() -> None:
    def check(seed, ex):
        meta = ex["metadata"]
        before_grid = np.array(ex["input"]["input_scene"], dtype=np.uint8)
        after_grid = np.array(ex["input"]["output_scene"], dtype=np.uint8)
        before_bg = meta["background_color"]
        after_bg = meta.get("after_background_color", before_bg)
        before = _parse_scene(before_grid, meta, before_bg)
        after = _parse_scene(after_grid, meta, after_bg)
        preserved = []
        if int(before_grid[1, 0]) == before_bg and int(after_grid[1, 0]) == after_bg and before_bg == after_bg:
            preserved.append("background_color_preserved")
        if before_grid.shape == after_grid.shape:
            preserved.append("grid_size_preserved")
        before_layout = (before_grid == meta["label_marker_color"]) | (before_grid == meta["divider_color"])
        after_layout = (after_grid == meta["label_marker_color"]) | (after_grid == meta["divider_color"])
        if before_layout.shape == after_layout.shape and np.array_equal(before_layout, after_layout):
            preserved.append("label_layout_preserved")
        if sum(obj is not None for obj in before) == sum(obj is not None for obj in after):
            preserved.append("object_count_preserved")
        if all((b_obj is not None) == (a_obj is not None) for b_obj, a_obj in zip(before, after)):
            preserved.append("slot_order_preserved")
        expected = ";".join(label for label in GLOBAL_PROPERTY_ORDER if label in preserved) or "none"
        assert ex["output"] == expected, seed
    _verify("242", check)


def test_task_243() -> None:
    def check(seed, ex):
        before, after, meta = _pair_objects(ex)
        preserved = _relation_preserved(before, after, meta["candidate_relation"])
        expected = "relation_preserved" if preserved else "relation_not_preserved"
        assert ex["output"] == expected, seed
    _verify("243", check)


if __name__ == "__main__":
    tests = [
        test_task_236,
        test_task_237,
        test_task_238,
        test_task_239,
        test_task_240,
        test_task_241,
        test_task_242,
        test_task_243,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
