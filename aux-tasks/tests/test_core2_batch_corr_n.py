"""Verifier tests for core2 batch N correspondence tasks."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate
from sft_wrapper import format_single
from symbol_remap import palette_for_example


TASK_IDS = ["193", "223", "224", "225", "226", "232", "233", "234", "235"]
N_SEEDS = 40
RAW_COLOR_RE = re.compile(r"\bcolor [0-9]\b(?!-)", re.I)


def _primary_dims(ex) -> tuple[int, int]:
    """Largest grid found anywhere in the example input (h, w)."""
    grids = []

    def walk(v):
        if isinstance(v, list) and v and isinstance(v[0], list) and (not v[0] or not isinstance(v[0][0], list)):
            grids.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(ex["input"])
    g = max(grids, key=lambda x: len(x) * (len(x[0]) if x else 0))
    return len(g), len(g[0])


def test_size_and_count_diversity():
    """The second-review widening must produce independently-varying H and W and
    a varied object count for every task (not a single fixed scene size)."""
    for task_id in TASK_IDS:
        heights, widths, comp_counts = set(), set(), set()
        for seed in range(60):
            ex = generate(task_id, seed)
            h, w = _primary_dims(ex)
            heights.add(h)
            widths.add(w)
            grid = np.array(ex["input"]["input_grid"] if isinstance(ex["input"], dict) else ex["input"], dtype=np.uint8)
            meta = ex["metadata"]
            obj = meta.get("object_color", meta.get("target_color"))
            if obj is not None:
                comp_counts.add(len(_components(grid, meta["background_color"], {int(obj)})))
        assert len(heights) >= 4, (task_id, "H not varied", sorted(heights))
        assert len(widths) >= 4, (task_id, "W not varied", sorted(widths))
        # H and W independent: not every grid is square.
        assert not all(generate(task_id, s) and _primary_dims(generate(task_id, s))[0] == _primary_dims(generate(task_id, s))[1] for s in range(20)), (task_id, "H==W always")
        assert len(comp_counts) >= 2, (task_id, "object count not varied", sorted(comp_counts))


def _components(grid: np.ndarray, bg: int, colors: set[int] | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg or (colors is not None and color not in colors):
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
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
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append({
                "color": color,
                "cells": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            })
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3], comp["color"]))
    return comps


def _normalize(cells) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _transform_shape(shape, name: str) -> frozenset[tuple[int, int]]:
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
        raise AssertionError(name)
    return _normalize(cells)


def _canonical(shape, relation: str) -> tuple[tuple[int, int], ...]:
    shape = _normalize(shape)
    if relation == "same_normalized_shape":
        return tuple(sorted(shape))
    if relation == "same_shape_under_rotation":
        names = ["identity", "rotate90", "rotate180", "rotate270"]
    elif relation == "same_shape_under_reflection":
        names = ["identity", "reflect_horizontal", "reflect_vertical", "rotate180"]
    else:
        raise AssertionError(relation)
    return min(tuple(sorted(_transform_shape(shape, name))) for name in names)


def _apply_transform(expected: np.ndarray, cells: set[tuple[int, int]], meta: dict, transform: dict) -> None:
    if transform["type"] == "erase":
        color = meta["background_color"]
    elif transform["type"] == "recolor":
        color = transform["color"]
    else:
        raise AssertionError(transform)
    for r, c in cells:
        expected[r, c] = color


def _condition(comp: dict, meta: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    name = meta["property_condition"]
    if name == "cell_count_at_least":
        return len(comp["cells"]) >= meta["threshold"]
    if name == "bbox_wider_than_tall":
        return width > height
    if name == "cell_count_even":
        return len(comp["cells"]) % 2 == 0
    raise AssertionError(name)


def _orientation(comp: dict) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width > height:
        return "wide"
    if height > width:
        return "tall"
    return "square"


def _medoid(cells: set[tuple[int, int]]) -> tuple[int, int]:
    ordered = sorted(cells)
    return min(
        ordered,
        key=lambda p: (
            sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in ordered),
            p[0],
            p[1],
        ),
    )


def _distance(comp: dict, marker: tuple[int, int], relation: str) -> int:
    if relation == "nearest_marker_by_min_manhattan":
        return min(abs(r - marker[0]) + abs(c - marker[1]) for r, c in comp["cells"])
    if relation == "nearest_marker_by_medoid":
        r, c = _medoid(comp["cells"])
        return abs(r - marker[0]) + abs(c - marker[1])
    raise AssertionError(relation)


def _region(comp: dict, meta: dict) -> str:
    r, c = _medoid(comp["cells"])
    if r < meta["split_row"] and c < meta["split_col"]:
        return "top_left"
    if r < meta["split_row"] and c > meta["split_col"]:
        return "top_right"
    if r > meta["split_row"] and c < meta["split_col"]:
        return "bottom_left"
    return "bottom_right"


def _correspondence(input_grid: np.ndarray, output_grid: np.ndarray, meta: dict) -> str:
    bg = meta["background_color"]
    object_color = meta["object_color"]
    relation = meta["matching_rule"]
    inputs = _components(input_grid, bg, {object_color})
    outputs = _components(output_grid, bg, {object_color})
    output_by_key = {_canonical(comp["cells"], relation): idx for idx, comp in enumerate(outputs, start=1)}
    parts = []
    for idx, comp in enumerate(inputs, start=1):
        parts.append(f"input{idx}->output{output_by_key[_canonical(comp['cells'], relation)]}")
    return ";".join(parts)


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        palette = palette_for_example(task_id, seed)
        rendered_prompt, answer = format_single(task_id, seed, color_palette=palette)
        assert "[No prompt" not in rendered_prompt
        assert "Input:" in rendered_prompt and isinstance(answer, str) and answer
        assert not RAW_COLOR_RE.search(rendered_prompt), (task_id, seed, rendered_prompt)
        fn(seed, ex)
        assert generate(task_id, seed) == ex


def test_task_193():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        count = int(np.count_nonzero(grid == meta["condition_color"]))
        comps = _components(grid, meta["background_color"], {meta["target_color"]})
        assert 1 <= count <= len(comps), seed
        expected = grid.copy()
        _apply_transform(expected, comps[count - 1]["cells"], meta, meta["action_transform"])
        assert out.tolist() == expected.tolist(), seed
    _verify("193", check)


def test_task_223():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        comps = _components(grid, meta["background_color"], {meta["object_color"]})
        expected = grid.copy()
        seen_values = []
        for comp in comps:
            ok = _condition(comp, meta)
            seen_values.append(ok)
            _apply_transform(expected, comp["cells"], meta, meta["transform_A"] if ok else meta["transform_B"])
        assert any(seen_values) and not all(seen_values), seed
        assert out.tolist() == expected.tolist(), seed
    _verify("223", check)


def test_task_224():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        mapping = {item["value"]: item["transform"] for item in meta["value_to_transform"]}
        expected = grid.copy()
        values = set()
        for comp in _components(grid, meta["background_color"], {meta["object_color"]}):
            value = _orientation(comp)
            values.add(value)
            _apply_transform(expected, comp["cells"], meta, mapping[value])
        assert values == {"wide", "tall", "square"}, seed
        assert out.tolist() == expected.tolist(), seed
    _verify("224", check)


def test_task_225():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        markers = []
        for color in meta["marker_colors"]:
            hits = [tuple(p) for p in np.argwhere(grid == color)]
            assert len(hits) == 1, (seed, color, hits)
            markers.append({"color": color, "position": hits[0]})
        expected = grid.copy()
        for comp in _components(grid, meta["background_color"], {meta["object_color"]}):
            marker = min(
                markers,
                key=lambda item: (
                    _distance(comp, item["position"], meta["association_relation"]),
                    item["position"][0],
                    item["position"][1],
                ),
            )
            _apply_transform(expected, comp["cells"], meta, {"type": "recolor", "color": marker["color"]})
        assert out.tolist() == expected.tolist(), seed
    _verify("225", check)


def test_task_226():
    def check(seed, ex):
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        mapping = {item["region"]: item["transform"] for item in meta["region_to_transform"]}
        expected = grid.copy()
        seen = set()
        for comp in _components(grid, meta["background_color"], {meta["object_color"]}):
            region = _region(comp, meta)
            seen.add(region)
            _apply_transform(expected, comp["cells"], meta, mapping[region])
        assert seen == {"top_left", "top_right", "bottom_left", "bottom_right"}, seed
        assert out.tolist() == expected.tolist(), seed
    _verify("226", check)


def test_task_232():
    def check(seed, ex):
        input_grid = np.array(ex["input"]["input_grid"], dtype=np.uint8)
        output_grid = np.array(ex["input"]["output_grid"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        meta = ex["metadata"]
        input_comps = _components(input_grid, meta["background_color"], {meta["object_color"], meta["selected_color"]})
        output_comps = _components(output_grid, meta["background_color"], {meta["object_color"]})
        selected = [comp for comp in input_comps if comp["color"] == meta["selected_color"]]
        assert len(selected) == 1, seed
        selected_key = _canonical(selected[0]["cells"], meta["matching_rule"])
        matches = [comp for comp in output_comps if _canonical(comp["cells"], meta["matching_rule"]) == selected_key]
        assert len(matches) == 1, seed
        expected = output_grid.copy()
        for r, c in matches[0]["cells"]:
            expected[r, c] = meta["highlight_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("232", check)


def test_task_233():
    def check(seed, ex):
        input_grid = np.array(ex["input"]["input_grid"], dtype=np.uint8)
        output_grid = np.array(ex["input"]["output_grid"], dtype=np.uint8)
        expected = _correspondence(input_grid, output_grid, ex["metadata"])
        assert ex["output"] == expected, seed
    _verify("233", check)


def test_task_234():
    def check(seed, ex):
        input_grid = np.array(ex["input"]["input_grid"], dtype=np.uint8)
        output_grid = np.array(ex["input"]["output_grid"], dtype=np.uint8)
        expected = _correspondence(input_grid, output_grid, ex["metadata"])
        assert ex["output"] == expected, seed
    _verify("234", check)


def test_task_235():
    def check(seed, ex):
        input_grid = np.array(ex["input"]["input_grid"], dtype=np.uint8)
        output_grid = np.array(ex["input"]["output_grid"], dtype=np.uint8)
        meta = ex["metadata"]
        inputs = _components(input_grid, meta["background_color"], {meta["object_color"]})
        outputs = _components(output_grid, meta["background_color"], {meta["object_color"]})
        input_keys = [_canonical(comp["cells"], meta["matching_rule"]) for comp in inputs]
        output_keys = [_canonical(comp["cells"], meta["matching_rule"]) for comp in outputs]
        matched = []
        deleted = []
        copied = []
        for input_idx, key in enumerate(input_keys, start=1):
            output_labels = [f"output{j}" for j, out_key in enumerate(output_keys, start=1) if out_key == key]
            if not output_labels:
                deleted.append(f"input{input_idx}")
            elif len(output_labels) == 1:
                matched.append(f"input{input_idx}->{output_labels[0]}")
            else:
                copied.extend(f"input{input_idx}->{label}" for label in output_labels)
        new = [
            f"output{idx}"
            for idx, key in enumerate(output_keys, start=1)
            if key not in set(input_keys)
        ]
        expected = f"matched={','.join(matched)};deleted={','.join(deleted)};copied={','.join(copied)};new={','.join(new)}"
        assert matched and deleted and copied and new, seed
        assert ex["output"] == expected, seed
    _verify("235", check)


if __name__ == "__main__":
    for test_fn in [
        test_task_193,
        test_task_223,
        test_task_224,
        test_task_225,
        test_task_226,
        test_task_232,
        test_task_233,
        test_task_234,
        test_task_235,
        test_size_and_count_diversity,
    ]:
        test_fn()
        print(f"  PASS  {test_fn.__name__}")
