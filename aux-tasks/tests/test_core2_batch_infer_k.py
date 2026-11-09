"""Verifier tests for finite inference/meta batch K tasks."""
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


TASK_IDS = ["121", "122", "123", "204", "205", "206", "207", "208", "209", "210"]
N_SEEDS = 40
PALETTE = tuple("abcdefghij")

TRANSFORM_LABELS = [
    "identity",
    "rotate_cw_90",
    "rotate_180",
    "rotate_ccw_90",
    "reflect_vertical",
    "reflect_horizontal",
]
GEOM_OPS = {
    "translate_up_1": (-1, 0),
    "translate_down_1": (1, 0),
    "translate_left_1": (0, -1),
    "translate_right_1": (0, 1),
}
RAW_COLOR_RE = re.compile(r"\bcolor [0-9]\b(?!-)", re.I)
LABEL_RE = re.compile(r"^[A-Za-z0-9_:]+$")


def _grid(ex: dict, key: str) -> np.ndarray:
    return np.array(ex["input"][key], dtype=np.uint8)


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid.tolist())


def _apply_transform(grid: np.ndarray, label: str) -> np.ndarray:
    if label == "identity":
        return grid.copy()
    if label == "rotate_cw_90":
        return np.rot90(grid, -1).copy()
    if label == "rotate_180":
        return np.rot90(grid, 2).copy()
    if label == "rotate_ccw_90":
        return np.rot90(grid, 1).copy()
    if label == "reflect_vertical":
        return np.fliplr(grid).copy()
    if label == "reflect_horizontal":
        return np.flipud(grid).copy()
    raise AssertionError(label)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, *, bg: int | None = None, color: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            cell_color = int(grid[r, c])
            if (bg is not None and cell_color == bg) or (color is not None and cell_color != color):
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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == cell_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": cell_color, "cells": cells, "bbox": _bbox(cells)})
    return sorted(comps, key=lambda comp: comp["bbox"])


def _verify(task_id: str, check):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        assert isinstance(ex["input"], dict), seed
        assert isinstance(ex["output"], str), seed
        assert LABEL_RE.fullmatch(ex["output"]), (seed, ex["output"])
        meta = ex["metadata"]
        assert ex["output"] in meta["allowed_labels"], (seed, ex["output"], meta["allowed_labels"])
        prompt = get_prompt(task_id, meta)
        assert "[No prompt" not in prompt
        if seed < 5:
            rendered_prompt, answer = format_single(task_id, seed, color_palette=PALETTE)
            assert "[No prompt" not in rendered_prompt
            assert not RAW_COLOR_RE.search(rendered_prompt), rendered_prompt
            assert answer == ex["output"], (seed, answer, ex["output"])
        check(seed, ex)
        assert generate(task_id, seed) == ex


def test_task_121():
    def check(seed, ex):
        inp = _grid(ex, "input")
        out = _grid(ex, "output")
        matches = [label for label in TRANSFORM_LABELS if _grid_key(_apply_transform(inp, label)) == _grid_key(out)]
        assert matches == [ex["output"]], (seed, matches)
    _verify("121", check)


def test_task_122():
    def check(seed, ex):
        inp = _grid(ex, "input")
        expected = _apply_transform(inp, ex["metadata"]["rule_label"])
        winners = [
            label for label in ex["metadata"]["candidate_labels"]
            if _grid_key(_grid(ex, label)) == _grid_key(expected)
        ]
        assert winners == [ex["output"]], (seed, winners)
    _verify("122", check)


def test_task_123():
    def check(seed, ex):
        inp = _grid(ex, "input")
        candidate = _grid(ex, "candidate")
        expected = _apply_transform(inp, ex["metadata"]["rule_label"])
        truth = _grid_key(candidate) == _grid_key(expected)
        assert ex["output"] == ("true" if truth else "false"), seed
    _verify("123", check)


def _changed_mapping(inp: np.ndarray, out: np.ndarray) -> dict[int, int]:
    mapping = {}
    for color in sorted(int(v) for v in np.unique(inp)):
        mask = inp == color
        values = {int(v) for v in np.unique(out[mask])}
        assert len(values) == 1
        target = values.pop()
        if target != color:
            mapping[color] = target
    return mapping


def _candidate_mapping(item: dict) -> dict[int, int]:
    return {int(pair["from_color"]): int(pair["to_color"]) for pair in item["pairs"]}


def test_task_204():
    def check(seed, ex):
        inp = _grid(ex, "input")
        out = _grid(ex, "output")
        expected = _changed_mapping(inp, out)
        assert expected, seed
        winners = [
            item["label"] for item in ex["metadata"]["candidate_mappings"]
            if _candidate_mapping(item) == expected
        ]
        assert winners == [ex["output"]], (seed, winners, expected)
    _verify("204", check)


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _apply_geom(grid: np.ndarray, bg: int, label: str, selected: dict | None = None) -> np.ndarray:
    scope, op = label.split(":", 1)
    dr, dc = GEOM_OPS[op]
    comps = _components(grid, bg=bg)
    moving = comps if scope == "all_foreground" else [selected]
    assert moving and all(item is not None for item in moving)
    out = grid.copy()
    for comp in moving:
        for r, c in comp["cells"]:
            out[r, c] = bg
    for comp in moving:
        for r, c in _shift(comp["cells"], dr, dc):
            out[r, c] = comp["color"]
    return out


def test_task_205():
    def check(seed, ex):
        grid = _grid(ex, "input")
        out = _grid(ex, "output")
        bg = ex["metadata"]["background_color"]
        comps = _components(grid, bg=bg)
        matches = []
        for label in ex["metadata"]["allowed_labels"]:
            scope = label.split(":", 1)[0]
            if scope == "all_foreground":
                if _grid_key(_apply_geom(grid, bg, label)) == _grid_key(out):
                    matches.append(label)
            else:
                if any(_grid_key(_apply_geom(grid, bg, label, comp)) == _grid_key(out) for comp in comps):
                    matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
    _verify("205", check)


def _select_component(comps: list[dict], rule: str) -> dict:
    if rule == "largest_area":
        key = lambda comp: len(comp["cells"])
        best = max(key(comp) for comp in comps)
    elif rule == "smallest_area":
        key = lambda comp: len(comp["cells"])
        best = min(key(comp) for comp in comps)
    elif rule == "topmost":
        key = lambda comp: comp["bbox"][0]
        best = min(key(comp) for comp in comps)
    elif rule == "bottommost":
        key = lambda comp: comp["bbox"][2]
        best = max(key(comp) for comp in comps)
    elif rule == "leftmost":
        key = lambda comp: comp["bbox"][1]
        best = min(key(comp) for comp in comps)
    else:
        key = lambda comp: comp["bbox"][3]
        best = max(key(comp) for comp in comps)
    winners = [comp for comp in comps if key(comp) == best]
    assert len(winners) == 1
    return winners[0]


def _apply_selection_action(grid: np.ndarray, comp: dict, action: str, meta: dict) -> np.ndarray:
    out = grid.copy()
    if action == "recolor":
        for r, c in comp["cells"]:
            out[r, c] = meta["action_color"]
    elif action == "erase":
        for r, c in comp["cells"]:
            out[r, c] = meta["background_color"]
    else:
        out[min(comp["cells"])] = meta["action_color"]
    return out


def test_task_206():
    def check(seed, ex):
        grid = _grid(ex, "input")
        out = _grid(ex, "output")
        meta = ex["metadata"]
        comps = _components(grid, color=meta["object_color"])
        matches = []
        for label in meta["allowed_labels"]:
            rule, action = label.split(":", 1)
            comp = _select_component(comps, rule)
            expected = _apply_selection_action(grid, comp, action, meta)
            if _grid_key(expected) == _grid_key(out):
                matches.append(label)
        assert matches == [ex["output"]], (seed, matches)
    _verify("206", check)


def _apply_line_label(grid: np.ndarray, label: str, meta: dict) -> np.ndarray:
    axis, operation, condition = label.split(":")
    marker = meta["marker_color"]
    outer = grid.shape[0] if axis == "rows" else grid.shape[1]
    marker_lines = set()
    for idx in range(outer):
        line = grid[idx, :] if axis == "rows" else grid[:, idx]
        if np.any(line == marker):
            marker_lines.add(idx)
    selected = marker_lines if condition == "contains_marker" else set(range(outer)) - marker_lines
    replacement = meta["background_color"] if operation == "erase" else meta["line_color"]
    out = grid.copy()
    for idx in selected:
        if axis == "rows":
            out[idx, :] = replacement
        else:
            out[:, idx] = replacement
    return out


def test_task_207():
    def check(seed, ex):
        grid = _grid(ex, "input")
        out = _grid(ex, "output")
        matches = [
            label for label in ex["metadata"]["allowed_labels"]
            if _grid_key(_apply_line_label(grid, label, ex["metadata"])) == _grid_key(out)
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("207", check)


def _apply_symmetry_label(grid: np.ndarray, label: str, bg: int) -> np.ndarray:
    size = grid.shape[0]
    mid = size // 2
    out = grid.copy()
    if label == "vertical_axis:add_from_left":
        for r in range(size):
            for c in range(mid):
                mc = size - 1 - c
                if grid[r, c] != bg and grid[r, mc] == bg:
                    out[r, mc] = grid[r, c]
    elif label == "vertical_axis:remove_right_extra":
        for r in range(size):
            for c in range(mid + 1, size):
                mc = size - 1 - c
                if grid[r, c] != bg and grid[r, mc] == bg:
                    out[r, c] = bg
    elif label == "vertical_axis:recolor_right_to_left":
        for r in range(size):
            for c in range(mid):
                mc = size - 1 - c
                if grid[r, c] != bg and grid[r, mc] != bg and grid[r, mc] != grid[r, c]:
                    out[r, mc] = grid[r, c]
    elif label == "horizontal_axis:add_from_top":
        for r in range(mid):
            mr = size - 1 - r
            for c in range(size):
                if grid[r, c] != bg and grid[mr, c] == bg:
                    out[mr, c] = grid[r, c]
    elif label == "horizontal_axis:remove_bottom_extra":
        for r in range(mid + 1, size):
            mr = size - 1 - r
            for c in range(size):
                if grid[r, c] != bg and grid[mr, c] == bg:
                    out[r, c] = bg
    else:
        for r in range(mid):
            mr = size - 1 - r
            for c in range(size):
                if grid[r, c] != bg and grid[mr, c] != bg and grid[mr, c] != grid[r, c]:
                    out[mr, c] = grid[r, c]
    return out


def test_task_208():
    def check(seed, ex):
        grid = _grid(ex, "input")
        out = _grid(ex, "output")
        bg = ex["metadata"]["background_color"]
        matches = [
            label for label in ex["metadata"]["allowed_labels"]
            if _grid_key(_apply_symmetry_label(grid, label, bg)) == _grid_key(out)
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("208", check)


def _reachable(grid: np.ndarray, meta: dict, connectivity: int) -> set[tuple[int, int]]:
    starts = [tuple(p) for p in np.argwhere(grid == meta["seed_color"])]
    assert len(starts) == 1
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = grid.shape
    q = deque([starts[0]])
    seen = {starts[0]}
    reached = set()
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                continue
            if grid[nr, nc] != meta["target_color"]:
                continue
            seen.add((nr, nc))
            reached.add((nr, nc))
            q.append((nr, nc))
    return reached


def _apply_connectivity(grid: np.ndarray, meta: dict, label: str) -> np.ndarray:
    connectivity = 4 if label == "connectivity_4" else 8
    out = grid.copy()
    for r, c in _reachable(grid, meta, connectivity):
        out[r, c] = meta["mark_color"]
    return out


def test_task_209():
    def check(seed, ex):
        grid = _grid(ex, "input")
        out = _grid(ex, "output")
        matches = [
            label for label in ex["metadata"]["allowed_labels"]
            if _grid_key(_apply_connectivity(grid, ex["metadata"], label)) == _grid_key(out)
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("209", check)


def _apply_distance(grid: np.ndarray, meta: dict, label: str) -> np.ndarray:
    metric, distance_text = label.split(":")
    distance = int(distance_text.rsplit("_", 1)[1])
    starts = [tuple(p) for p in np.argwhere(grid == meta["source_color"])]
    assert len(starts) == 1
    sr, sc = starts[0]
    out = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) == (sr, sc):
                continue
            if metric == "manhattan":
                d = abs(r - sr) + abs(c - sc)
            else:
                d = max(abs(r - sr), abs(c - sc))
            if d == distance:
                out[r, c] = meta["mark_color"]
    return out


def test_task_210():
    def check(seed, ex):
        grid = _grid(ex, "input")
        out = _grid(ex, "output")
        matches = [
            label for label in ex["metadata"]["allowed_labels"]
            if _grid_key(_apply_distance(grid, ex["metadata"], label)) == _grid_key(out)
        ]
        assert matches == [ex["output"]], (seed, matches)
    _verify("210", check)


if __name__ == "__main__":
    for fn in [
        test_task_121,
        test_task_122,
        test_task_123,
        test_task_204,
        test_task_205,
        test_task_206,
        test_task_207,
        test_task_208,
        test_task_209,
        test_task_210,
    ]:
        fn()
        print(f"  PASS  {fn.__name__}")
