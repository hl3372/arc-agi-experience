"""Verifier tests for selected simple core second-batch tasks.

Tasks 95 and 96 moved to ``generators/bbox_outline_95.py`` and
``generators/frame_extract_96.py`` on 2026-09-15; their checks live in
``tests/test_outline_frames_95_96.py``.
"""
from __future__ import annotations

import ast
import os
import sys
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate

TASK_IDS = [
    "118", "133", "134", "135", "136", "137",
    "138", "145", "146", "147", "153", "154", "157", "158", "162", "164",
]
N_SEEDS = 25


def _components(grid: np.ndarray, bg: int, connectivity: int = 4, color: int | None = None) -> list[dict]:
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg or (color is not None and grid[r, c] != color):
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            cell_color = int(grid[r, c])
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == cell_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append({
                "color": cell_color,
                "cells": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            })
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        fn(seed, np.array(ex["input"], dtype=np.uint8), np.array(ex["output"], dtype=np.uint8) if not isinstance(ex["output"], str) else ex["output"], ex["metadata"])
        assert generate(task_id, seed) == ex


def _assert_background_style_mix(task_id: str):
    solid_cases = 0
    patterned_cases = 0
    styles = set()
    for seed in range(100):
        meta = generate(task_id, seed)["metadata"]
        style = meta.get("background_style")
        styles.add(style)
        if style == "solid":
            solid_cases += 1
            assert meta.get("background_detail_colors", []) == []
        else:
            patterned_cases += 1
    assert solid_cases > 0, styles
    assert patterned_cases > 0, styles


def test_task_118():
    def check(seed, grid, out, meta):
        n = meta["block_size"]
        expected = np.zeros((grid.shape[0] // n, grid.shape[1] // n), dtype=np.uint8)
        for br in range(expected.shape[0]):
            for bc in range(expected.shape[1]):
                block = grid[br * n:(br + 1) * n, bc * n:(bc + 1) * n]
                flat = [int(v) for v in block.flatten()]
                if meta["summary_rule"] == "top_left_color":
                    expected[br, bc] = int(block[0, 0])
                elif meta["summary_rule"] == "majority_color":
                    counts = Counter(flat)
                    max_count = max(counts.values())
                    majority_colors = [color for color, count in counts.items() if count == max_count]
                    expected[br, bc] = majority_colors[0] if len(majority_colors) == 1 else meta["background_color"]
                else:
                    expected[br, bc] = meta["true_color"] if meta["target_color"] in flat else meta["false_color"]
        assert out.tolist() == expected.tolist(), seed
        assert meta["majority_tie_rule"] == "if no color is uniquely most frequent, output the background color"
        prompt = get_prompt("118", meta).lower()
        forbidden = [
            "low" + "est",
            "low" + "er color",
            "color " + "index",
            "color " + "order",
            "ear" + "lier color",
            "lat" + "er color",
        ]
        assert not any(term in prompt for term in forbidden), prompt
    _verify("118", check)


def test_task_133():
    def check(seed, grid, out, meta):
        if meta["predicate"] == "color_equals":
            mask = grid == meta["target_color"]
        else:
            mask = grid != meta["background_color"]
        expected = np.where(mask, meta["true_color"], meta["false_color"])
        assert out.tolist() == expected.tolist(), seed
    _verify("133", check)


def test_task_134():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        src = meta["source_marker_color"]
        dst = meta["target_marker_color"]
        if meta["axis"] == "row":
            src_row = int(np.where(np.any(grid == src, axis=1))[0][0])
            dst_row = int(np.where(np.any(grid == dst, axis=1))[0][0])
            dst_pos = tuple(np.argwhere(grid == dst)[0])
            expected[dst_row, :] = grid[src_row, :]
            if meta["marker_policy"] == "preserve_target_marker":
                expected[dst_pos] = dst
        else:
            src_col = int(np.where(np.any(grid == src, axis=0))[0][0])
            dst_col = int(np.where(np.any(grid == dst, axis=0))[0][0])
            dst_pos = tuple(np.argwhere(grid == dst)[0])
            expected[:, dst_col] = grid[:, src_col]
            if meta["marker_policy"] == "preserve_target_marker":
                expected[dst_pos] = dst
        assert out.tolist() == expected.tolist(), seed
    _verify("134", check)


def test_task_135():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        cond_color = meta["condition_color"]
        rows = np.ones(grid.shape[0], dtype=bool)
        cols = np.ones(grid.shape[1], dtype=bool)
        if meta["axis"] in ("rows", "both"):
            rows = np.any(grid != bg, axis=1) if meta["condition"] == "all_background" else ~np.any(grid == cond_color, axis=1)
        if meta["axis"] in ("columns", "both"):
            cols = np.any(grid != bg, axis=0) if meta["condition"] == "all_background" else ~np.any(grid == cond_color, axis=0)
        assert out.tolist() == grid[rows][:, cols].tolist(), seed
    _verify("135", check)
    prompts = [get_prompt("135", generate("135", seed)["metadata"]) for seed in range(100)]
    assert any("row and column" in prompt for prompt in prompts)
    assert not any("every both" in prompt for prompt in prompts)


def test_task_136():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        rows = np.where(np.any(grid == meta["row_marker_color"], axis=1))[0]
        cols = np.where(np.any(grid == meta["column_marker_color"], axis=0))[0]
        for r in rows:
            for c in cols:
                expected[r, c] = meta["intersection_color"]
        assert out.tolist() == expected.tolist(), seed
        for color in meta.get("distractor_colors", []):
            mask = (grid == color) & (expected != meta["intersection_color"])
            assert np.array_equal(out[mask], grid[mask]), seed
    _verify("136", check)


def test_task_137():
    def check(seed, grid, out, meta):
        detail_colors = set(meta.get("background_detail_colors", []))
        cleaned = grid.copy()
        for color in detail_colors:
            cleaned[cleaned == color] = meta["background_color"]
        comps = _components(cleaned, meta["background_color"], meta["connectivity"])
        key = {
            "topmost": lambda comp: (comp["bbox"][0], comp["bbox"][0], comp["bbox"][1]),
            "bottommost": lambda comp: (-comp["bbox"][2], comp["bbox"][0], comp["bbox"][1]),
            "leftmost": lambda comp: (comp["bbox"][1], comp["bbox"][0], comp["bbox"][1]),
            "rightmost": lambda comp: (-comp["bbox"][3], comp["bbox"][0], comp["bbox"][1]),
        }[meta["position_rule"]]
        chosen = sorted(comps, key=key)[0]
        expected = grid.copy()
        for r, c in chosen["cells"]:
            expected[r, c] = meta["target_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("137", check)
    variants = {generate("137", seed)["metadata"]["connectivity"] for seed in range(100)}
    assert variants == {4, 8}, variants
    assert any(generate("137", seed)["metadata"]["background_detail_colors"] for seed in range(100))
    _assert_background_style_mix("137")


def test_task_138():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        for comp in _components(grid, meta["background_color"], 4, meta["target_color"]):
            for r, c in comp["cells"]:
                boundary = any((r + dr, c + dc) not in comp["cells"] for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
                expected[r, c] = meta["boundary_color"] if boundary else meta["interior_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("138", check)
    assert any(generate("138", seed)["metadata"]["distractor_colors"] for seed in range(100))
    assert any(generate("138", seed)["metadata"]["background_detail_colors"] for seed in range(100))
    _assert_background_style_mix("138")


def test_task_145():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        expected = np.full_like(grid, bg)
        if meta["axis"] == "row":
            for r in range(grid.shape[0]):
                vals = [v for v in grid[r, :] if v != bg]
                if meta["direction"] == "left":
                    expected[r, :len(vals)] = vals
                else:
                    expected[r, grid.shape[1] - len(vals):] = vals
        else:
            for c in range(grid.shape[1]):
                vals = [v for v in grid[:, c] if v != bg]
                if meta["direction"] == "top":
                    expected[:len(vals), c] = vals
                else:
                    expected[grid.shape[0] - len(vals):, c] = vals
        assert out.tolist() == expected.tolist(), seed
    _verify("145", check)


def test_task_146():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        target = meta["target_color"]
        if meta["axis"] == "row":
            order = list(range(grid.shape[1])) if meta["scan_direction"] == "left_to_right" else list(range(grid.shape[1] - 1, -1, -1))
            for r in range(grid.shape[0]):
                hits = [(r, c) for c in order if grid[r, c] == target]
                if hits:
                    expected[hits[0 if meta["occurrence"] == "first" else -1]] = meta["marker_color"]
        else:
            order = list(range(grid.shape[0])) if meta["scan_direction"] == "top_to_bottom" else list(range(grid.shape[0] - 1, -1, -1))
            for c in range(grid.shape[1]):
                hits = [(r, c) for r in order if grid[r, c] == target]
                if hits:
                    expected[hits[0 if meta["occurrence"] == "first" else -1]] = meta["marker_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("146", check)


def test_task_147():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        outer = grid.shape[0] if meta["axis"] == "horizontal" else grid.shape[1]
        inner = grid.shape[1] if meta["axis"] == "horizontal" else grid.shape[0]
        for i in range(outer):
            j = 0
            while j < inner:
                r, c = (i, j) if meta["axis"] == "horizontal" else (j, i)
                if grid[r, c] != meta["target_color"]:
                    j += 1
                    continue
                start = j
                while j < inner:
                    rr, cc = (i, j) if meta["axis"] == "horizontal" else (j, i)
                    if grid[rr, cc] != meta["target_color"]:
                        break
                    j += 1
                run_len = j - start
                ok = run_len >= meta["length"] if meta["condition"] == "at_least" else run_len == meta["length"]
                if ok:
                    for k in range(start, j):
                        rr, cc = (i, k) if meta["axis"] == "horizontal" else (k, i)
                        expected[rr, cc] = meta["marker_color"]
        assert out.tolist() == expected.tolist(), seed
        assert np.any(out != grid), seed
    _verify("147", check)
    assert any(generate("147", seed)["metadata"]["distractor_colors"] for seed in range(100))


def test_task_153():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        a = meta["marker_color_a"]
        b = meta["marker_color_b"]
        if meta["axis"] == "row":
            ai = int(np.where(np.any(grid == a, axis=1))[0][0])
            bi = int(np.where(np.any(grid == b, axis=1))[0][0])
            expected[[ai, bi], :] = expected[[bi, ai], :]
        else:
            ai = int(np.where(np.any(grid == a, axis=0))[0][0])
            bi = int(np.where(np.any(grid == b, axis=0))[0][0])
            expected[:, [ai, bi]] = expected[:, [bi, ai]]
        assert out.tolist() == expected.tolist(), seed
    _verify("153", check)


def test_task_154():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        detail_colors = set(meta.get("background_detail_colors", []))
        for comp in _components(grid, meta["background_color"], meta["connectivity"]):
            if comp["color"] in detail_colors:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            expected[r0:r1 + 1, c0:c1 + 1] = comp["color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("154", check)
    variants = {generate("154", seed)["metadata"]["connectivity"] for seed in range(100)}
    assert variants == {4, 8}, variants
    assert any(generate("154", seed)["metadata"]["background_detail_colors"] for seed in range(100))
    _assert_background_style_mix("154")


def test_task_157():
    def check(seed, grid, out, meta):
        h, w = grid.shape
        # Corner-anchored diagonal of length min(H, W): main starts at the
        # top-left stepping down-right; anti starts at the top-right stepping
        # down-left. Walk until an edge is hit.
        n = min(h, w)
        assert meta["diagonal_length"] == n, seed
        coords = [(i, i) for i in range(n)] if meta["diagonal"] == "main" else [(i, w - 1 - i) for i in range(n)]
        if meta["read_direction"] in ("bottom_right_to_top_left", "bottom_left_to_top_right"):
            coords = list(reversed(coords))
        vals = [int(grid[r, c]) for r, c in coords]
        expected = np.array([vals], dtype=np.uint8) if meta["output_orientation"] == "row" else np.array(vals, dtype=np.uint8).reshape(-1, 1)
        assert out.tolist() == expected.tolist(), seed
    _verify("157", check)
    # The widened contract must produce non-square grids (decorrelated H/W).
    shapes = [np.array(generate("157", seed)["input"]).shape for seed in range(60)]
    assert any(h != w for h, w in shapes), shapes
    assert any(h == w for h, w in shapes), shapes


def test_task_158():
    def check(seed, grid, out, meta):
        points = [tuple(p) for p in np.argwhere(grid == meta["marker_color"])]
        assert len(points) == 2, seed
        mid = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
        expected = grid.copy()
        expected[mid] = meta["midpoint_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("158", check)
    assert any(generate("158", seed)["metadata"]["background_detail_colors"] for seed in range(100))
    _assert_background_style_mix("158")


def test_task_162():
    def check(seed, grid, out, meta):
        axis_len = grid.shape[0] if meta["axis"] == "rows" else grid.shape[1]
        order = range(axis_len) if meta["keep_occurrence"] == "first" else range(axis_len - 1, -1, -1)
        seen = set()
        kept = []
        for i in order:
            key = tuple(int(v) for v in (grid[i, :] if meta["axis"] == "rows" else grid[:, i]))
            if key not in seen:
                seen.add(key)
                kept.append(i)
        kept.sort()
        expected = grid[kept, :] if meta["axis"] == "rows" else grid[:, kept]
        assert out.tolist() == expected.tolist(), seed
    _verify("162", check)


def test_task_164():
    def check(seed, grid, out, meta):
        t = meta["thickness"]
        pad = {
            "all": ((t, t), (t, t)),
            "top": ((t, 0), (0, 0)),
            "bottom": ((0, t), (0, 0)),
            "left": ((0, 0), (t, 0)),
            "right": ((0, 0), (0, t)),
            "top_bottom": ((t, t), (0, 0)),
            "left_right": ((0, 0), (t, t)),
        }[meta["sides"]]
        assert out.tolist() == np.pad(grid, pad, constant_values=meta["padding_color"]).tolist(), seed
    _verify("164", check)


if __name__ == "__main__":
    tests = [
        test_task_118,
        test_task_133, test_task_134, test_task_135, test_task_136, test_task_137,
        test_task_138, test_task_145, test_task_146, test_task_147, test_task_153,
        test_task_154, test_task_157, test_task_158, test_task_162, test_task_164,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
