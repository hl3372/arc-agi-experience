"""Verifier tests for motion and multi-grid second-batch slice G."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate
from sft_wrapper import format_single


TASK_IDS = ["128", "129", "148", "175", "176", "179", "188", "189", "194", "195"]
N_SEEDS = 30
TEST_PALETTE = tuple("µßàáâãåæçè")


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
    raise AssertionError(f"unknown transform {name}")


def _components(grid: np.ndarray, bg: int, color: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            cell_color = int(grid[r, c])
            if cell_color == bg or (color is not None and cell_color != color):
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
                        and int(grid[nr, nc]) == cell_color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append({
                "color": cell_color,
                "cells": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            })
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _landmark(cells: set[tuple[int, int]], name: str) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
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
    raise AssertionError(f"unknown landmark {name}")


def _occupied_landmark(cells: set[tuple[int, int]], name: str) -> tuple[int, int]:
    if name == "top_left":
        return min(cells, key=lambda p: (p[0], p[1]))
    if name == "top_right":
        return min(cells, key=lambda p: (p[0], -p[1]))
    if name == "bottom_left":
        return min(cells, key=lambda p: (-p[0], p[1]))
    if name == "bottom_right":
        return min(cells, key=lambda p: (-p[0], -p[1]))
    return _landmark(cells, name)


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed < 5:
            rendered_prompt, answer = format_single(task_id, seed, color_palette=TEST_PALETTE)
            assert isinstance(rendered_prompt, str) and isinstance(answer, str)
            assert "Input:" in rendered_prompt and answer
        fn(seed, ex["input"], np.array(ex["output"], dtype=np.uint8), ex["metadata"])
        assert generate(task_id, seed) == ex


def test_all_prompts_exist():
    for task_id in TASK_IDS:
        ex = generate(task_id, 0)
        assert "[No prompt" not in get_prompt(task_id, ex["metadata"])


def test_task_128():
    def check(seed, inp, out, meta):
        grid = np.array(inp, dtype=np.uint8)
        bg = meta["background_color"]
        ph, pw = meta["payload_size"]
        tops = [tuple(p) for p in meta["slot_payload_top_lefts"]]
        expected = np.full_like(grid, bg)
        for r, c in meta["slot_marker_positions"]:
            expected[r, c] = meta["slot_marker_color"]
        for i, (sr, sc) in enumerate(tops):
            tr, tc = tops[(i + 1) % len(tops)]
            expected[tr : tr + ph, tc : tc + pw] = grid[sr : sr + ph, sc : sc + pw]
        assert out.tolist() == expected.tolist(), seed

    _verify("128", check)


def test_task_129():
    def check(seed, inp, out, meta):
        expected = np.array(inp, dtype=np.uint8)
        for name in meta["undo_sequence"]:
            expected = _apply_transform(expected, name)
        assert out.tolist() == expected.tolist(), seed
        assert inp != out.tolist(), seed

    _verify("129", check)


def test_task_148():
    def check(seed, inp, out, meta):
        bottom = np.array(inp["bottom"], dtype=np.uint8)
        top = np.array(inp["top"], dtype=np.uint8)
        bg = meta["background_color"]
        expected = np.full_like(bottom, bg)
        for r in range(bottom.shape[0]):
            for c in range(bottom.shape[1]):
                b = int(bottom[r, c])
                t = int(top[r, c])
                if b == bg and t == bg:
                    expected[r, c] = bg
                elif b == bg:
                    expected[r, c] = t
                elif t == bg:
                    expected[r, c] = b
                elif meta["overlay_rule"] == "top_priority":
                    expected[r, c] = t
                elif meta["overlay_rule"] == "bottom_priority":
                    expected[r, c] = b
                else:
                    expected[r, c] = meta["collision_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("148", check)


def test_task_175():
    def check(seed, inp, out, meta):
        grid = np.array(inp, dtype=np.uint8)
        dr, dc = meta["offset_vector"]
        target = {tuple(p) for p in np.argwhere(grid == meta["target_color"])}
        expected = grid.copy()
        if meta["operation"] == "move":
            for cell in target:
                expected[cell] = meta["background_color"]
            expected[tuple(meta["marker_a_position"])] = meta["marker_a_color"]
            expected[tuple(meta["marker_b_position"])] = meta["marker_b_color"]
        for r, c in target:
            expected[r + dr, c + dc] = meta["target_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("175", check)


def test_task_176():
    def check(seed, inp, out, meta):
        grid = np.array(inp, dtype=np.uint8)
        cells = {tuple(p) for p in np.argwhere(grid == meta["target_color"])}
        anchor = tuple(meta["anchor_position"])
        lm = _landmark(cells, meta["landmark"])
        dr, dc = anchor[0] - lm[0], anchor[1] - lm[1]
        expected = np.full_like(grid, meta["background_color"])
        for r, c in cells:
            expected[r + dr, c + dc] = meta["target_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("176", check)


def test_task_179():
    def check(seed, inp, out, meta):
        grid = np.array(inp, dtype=np.uint8)
        sr, sc = meta["source_top_left"]
        tr, tc = meta["target_top_left"]
        rh, rw = meta["region_size"]
        expected = grid.copy()
        if meta["copy_mode"] == "frame":
            expected[tr + 1 : tr + rh - 1, tc + 1 : tc + rw - 1] = grid[
                sr + 1 : sr + rh - 1, sc + 1 : sc + rw - 1
            ]
        else:
            expected[tr : tr + rh, tc : tc + rw] = grid[sr : sr + rh, sc : sc + rw]
        assert out.tolist() == expected.tolist(), seed

    _verify("179", check)


def test_task_188():
    def check(seed, inp, out, meta):
        source = np.array(inp["source"], dtype=np.uint8)
        mask = np.array(inp["mask"], dtype=np.uint8)
        expected = np.full_like(source, meta["fill_color"])
        for r in range(source.shape[0]):
            for c in range(source.shape[1]):
                m = int(mask[r, c])
                keep = (
                    (meta["mask_condition"] == "mask_equals_color" and m == meta["mask_color"])
                    or (meta["mask_condition"] == "mask_non_background" and m != meta["mask_background_color"])
                    or (meta["mask_condition"] == "mask_not_color" and m != meta["mask_color"])
                )
                if keep:
                    expected[r, c] = source[r, c]
        assert out.tolist() == expected.tolist(), seed

    _verify("188", check)


def test_task_189():
    def check(seed, inp, out, meta):
        grids = [np.array(inp[f"grid_{i + 1}"], dtype=np.uint8) for i in range(meta["grid_count"])]
        bg = meta["background_color"]
        expected = np.full_like(grids[0], bg)
        for r in range(expected.shape[0]):
            for c in range(expected.shape[1]):
                vals = [int(grid[r, c]) for grid in grids]
                counts = Counter(vals)
                if meta["consensus_rule"] == "unanimous_non_background":
                    expected[r, c] = vals[0] if len(counts) == 1 and vals[0] != bg else bg
                elif meta["consensus_rule"] == "majority_non_background":
                    non_bg = [(color, count) for color, count in counts.items() if color != bg]
                    max_count = max((count for _color, count in non_bg), default=0)
                    winners = [color for color, count in non_bg if count == max_count]
                    expected[r, c] = winners[0] if len(winners) == 1 and max_count > meta["grid_count"] / 2 else bg
                else:
                    max_count = max(counts.values())
                    winners = [color for color, count in counts.items() if count == max_count]
                    expected[r, c] = winners[0] if len(winners) == 1 else meta["tie_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("189", check)


def test_task_194():
    def check(seed, inp, out, meta):
        grid = np.array(inp, dtype=np.uint8)
        bg = meta["background_color"]
        scene = {
            tuple(p)
            for p in np.argwhere((grid != bg) & (grid != meta["anchor_color"]))
        }
        anchor = tuple(meta["anchor_position"])
        lm = _occupied_landmark(scene, meta["landmark"])
        dr, dc = anchor[0] - lm[0], anchor[1] - lm[1]
        expected = np.full_like(grid, bg)
        for r, c in scene:
            expected[r + dr, c + dc] = grid[r, c]
        assert out.tolist() == expected.tolist(), seed

    _verify("194", check)


def test_task_195():
    def check(seed, inp, out, meta):
        grid = np.array(inp, dtype=np.uint8)
        pr, pc = meta["prototype_top_left"]
        ph, pw = meta["prototype_size"]
        lr, lc = meta["prototype_landmark_offset"]
        pattern = grid[pr : pr + ph, pc : pc + pw].copy()
        expected = grid.copy()
        for mr, mc in meta["marker_positions"]:
            top, left = mr - lr, mc - lc
            expected[top : top + ph, left : left + pw] = pattern
        assert out.tolist() == expected.tolist(), seed

    _verify("195", check)


if __name__ == "__main__":
    tests = [
        test_all_prompts_exist,
        test_task_128,
        test_task_129,
        test_task_148,
        test_task_175,
        test_task_176,
        test_task_179,
        test_task_188,
        test_task_189,
        test_task_194,
        test_task_195,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
