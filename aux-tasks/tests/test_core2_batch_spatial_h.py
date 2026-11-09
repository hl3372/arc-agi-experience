"""Verifier tests for core2 batch spatial H tasks."""
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


TASK_IDS = ["178", "183", "184", "185", "186", "187", "190", "191", "192", "196"]
N_SEEDS = 30

DIRECTIONS = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
    "northwest": (-1, -1),
    "northeast": (-1, 1),
    "southwest": (1, -1),
    "southeast": (1, 1),
}

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate_cw(cells):
    return _normalize({(c, -r) for r, c in cells})


def _transform(shape, name: str):
    out = shape
    for _ in range({"rot0": 0, "rot90": 1, "rot180": 2, "rot270": 3}[name]):
        out = _rotate_cw(out)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    comps.sort(key=lambda cells: _bbox(cells))
    return comps


def _components_not_wall(grid: np.ndarray, wall_color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == wall_color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != wall_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _single_cell(grid: np.ndarray, color: int, seed: int):
    points = [tuple(p) for p in np.argwhere(grid == color)]
    assert len(points) == 1, seed
    return points[0]


def _distance(a, b, metric: str):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc if metric == "manhattan" else max(dr, dc)


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        fn(seed, grid, out, ex["metadata"])
        assert generate(task_id, seed) == ex


def test_prompts_and_formatting():
    palettes = [
        tuple("abcdefghij"),
        tuple("klmnopqrst"),
        tuple("uvwxyzABCD"),
    ]
    for task_id in TASK_IDS:
        for seed, palette in zip([0, 7, 19], palettes):
            prompt, answer = format_single(task_id, seed, color_palette=palette)
            assert "Input:" in prompt
            assert answer.strip()
            assert "[No prompt" not in prompt


def test_task_178():
    def check(seed, grid, out, meta):
        prototype_comps = _components_of_color(grid, meta["prototype_color"])
        assert len(prototype_comps) == 1, seed
        prototype = _normalize(prototype_comps[0])
        color_map = {item["transform"]: item["color"] for item in meta["orientation_color_map"]}
        expected = grid.copy()
        for comp in _components_of_color(grid, meta["target_object_color"]):
            shape = _normalize(comp)
            matches = [name for name in meta["allowed_transforms"] if shape == _transform(prototype, name)]
            assert len(matches) == 1, (seed, matches)
            for r, c in comp:
                expected[r, c] = color_map[matches[0]]
        assert out.tolist() == expected.tolist(), seed
    _verify("178", check)


def test_task_183():
    def check(seed, grid, out, meta):
        cells = {tuple(p) for p in np.argwhere(grid == meta["path_color"])}
        assert cells, seed
        expected = grid.copy()
        for r, c in cells:
            neighbors = [(r + dr, c + dc) for dr, dc in ORTHO_DIRS if (r + dr, c + dc) in cells]
            assert len(neighbors) in (1, 2), seed
            if len(neighbors) == 2:
                v0 = (neighbors[0][0] - r, neighbors[0][1] - c)
                v1 = (neighbors[1][0] - r, neighbors[1][1] - c)
                if (v0[0] + v1[0], v0[1] + v1[1]) != (0, 0):
                    expected[r, c] = meta["turn_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("183", check)


def test_task_184():
    def check(seed, grid, out, meta):
        comps = _components_of_color(grid, meta["target_object_color"])
        assert len(comps) == 1, seed
        r0, c0, r1, c1 = _bbox(comps[0])
        expected = grid.copy()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r, c] == meta["background_color"]:
                    expected[r, c] = meta["gap_color"]
        assert out.tolist() == expected.tolist(), seed
        for color in meta.get("distractor_colors", []):
            mask = grid == color
            assert np.array_equal(out[mask], grid[mask]), seed
    _verify("184", check)
    assert any(generate("184", seed)["metadata"]["distractor_colors"] for seed in range(100))


def test_task_185():
    def check(seed, grid, out, meta):
        del seed
        sources = []
        for color in meta["source_colors"]:
            points = [tuple(p) for p in np.argwhere(grid == color)]
            assert len(points) == 1
            sources.append((points[0], color))
        expected = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != meta["background_color"]:
                    continue
                distances = [_distance((r, c), pos, meta["metric"]) for pos, _ in sources]
                best = min(distances)
                if distances.count(best) == 1:
                    expected[r, c] = sources[distances.index(best)][1]
        assert out.tolist() == expected.tolist()
    _verify("185", check)


def test_task_186():
    def check(seed, grid, out, meta):
        source = _single_cell(grid, meta["source_color"], seed)
        expected = grid.copy()
        h, w = grid.shape
        for name in meta["directions"]:
            dr, dc = DIRECTIONS[name]
            r, c = source[0] + dr, source[1] + dc
            while 0 <= r < h and 0 <= c < w:
                if grid[r, c] == meta["hit_color"]:
                    expected[r, c] = meta["mark_color"]
                    break
                r += dr
                c += dc
        assert out.tolist() == expected.tolist(), seed
    _verify("186", check)


def test_task_187():
    def check(seed, grid, out, meta):
        cells = {tuple(p) for p in np.argwhere(grid == meta["line_color"])}
        assert cells, seed
        expected = grid.copy()
        for r, c in cells:
            degree = sum((r + dr, c + dc) in cells for dr, dc in ORTHO_DIRS)
            if degree == 1:
                expected[r, c] = meta["endpoint_marker_color"]
            elif degree >= 3:
                expected[r, c] = meta["branchpoint_marker_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("187", check)


def test_task_190():
    def check(seed, grid, out, meta):
        del seed
        bh, bw = meta["block_height"], meta["block_width"]
        mapping = {tuple(tuple(row) for row in item["motif"]): item["color"] for item in meta["motif_color_map"]}
        expected = grid.copy()
        assert grid.shape[0] % bh == 0
        assert grid.shape[1] % bw == 0
        for r0 in range(0, grid.shape[0], bh):
            for c0 in range(0, grid.shape[1], bw):
                block = tuple(tuple(int(v) for v in row) for row in grid[r0 : r0 + bh, c0 : c0 + bw])
                assert block in mapping
                expected[r0 : r0 + bh, c0 : c0 + bw] = mapping[block]
        assert out.tolist() == expected.tolist()
    _verify("190", check)


def test_task_191():
    def check(seed, grid, out, meta):
        comps = _components_of_color(grid, meta["object_color"])
        assert len(comps) >= 2, seed
        boxes = [_bbox(comp) for comp in comps]
        depth_map = {item["depth"]: item["color"] for item in meta["depth_color_map"]}
        expected = grid.copy()
        for comp, box in zip(comps, boxes):
            r0, c0, r1, c1 = box
            depth = 0
            for other in boxes:
                if other == box:
                    continue
                or0, oc0, or1, oc1 = other
                if or0 < r0 and oc0 < c0 and or1 > r1 and oc1 > c1:
                    depth += 1
            assert depth in depth_map
            for r, c in comp:
                expected[r, c] = depth_map[depth]
        assert out.tolist() == expected.tolist(), seed
    _verify("191", check)


def test_task_192():
    def check(seed, grid, out, meta):
        del seed
        marker_colors = set(meta["marker_colors"])
        expected = grid.copy()
        for comp in _components_not_wall(grid, meta["wall_color"]):
            markers = [(r, c) for r, c in comp if int(grid[r, c]) in marker_colors]
            if len(markers) == 1:
                fill_color = int(grid[markers[0]])
                for r, c in comp:
                    if grid[r, c] == meta["background_color"]:
                        expected[r, c] = fill_color
        assert out.tolist() == expected.tolist()
    _verify("192", check)


def test_task_196():
    def check(seed, grid, out, meta):
        del seed
        expected = grid.copy()
        idx = meta["divider_index"]
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != meta["background_color"]:
                    continue
                on_side = (
                    (meta["divider_orientation"] == "horizontal" and meta["side"] == "above" and r < idx)
                    or (meta["divider_orientation"] == "horizontal" and meta["side"] == "below" and r > idx)
                    or (meta["divider_orientation"] == "vertical" and meta["side"] == "left" and c < idx)
                    or (meta["divider_orientation"] == "vertical" and meta["side"] == "right" and c > idx)
                    or (meta["divider_orientation"] == "main_diagonal" and meta["side"] == "below_left" and c - r < idx)
                    or (meta["divider_orientation"] == "main_diagonal" and meta["side"] == "above_right" and c - r > idx)
                    or (meta["divider_orientation"] == "anti_diagonal" and meta["side"] == "above_left" and r + c < idx)
                    or (meta["divider_orientation"] == "anti_diagonal" and meta["side"] == "below_right" and r + c > idx)
                )
                if on_side:
                    expected[r, c] = meta["side_color"]
        assert out.tolist() == expected.tolist()
    _verify("196", check)


if __name__ == "__main__":
    tests = [
        test_prompts_and_formatting,
        test_task_178,
        test_task_183,
        test_task_184,
        test_task_185,
        test_task_186,
        test_task_187,
        test_task_190,
        test_task_191,
        test_task_192,
        test_task_196,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
