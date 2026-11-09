"""Verifier tests for core2 batch paths B tasks."""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate


TASK_IDS = ["109", "110", "112", "115", "140", "141", "142", "149", "159", "170"]
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


def _neighbors(connectivity: int):
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return deltas


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        output = np.array(ex["output"], dtype=np.uint8) if not isinstance(ex["output"], str) else ex["output"]
        fn(seed, np.array(ex["input"], dtype=np.uint8), output, ex["metadata"])
        assert generate(task_id, seed) == ex


def _single_cell(grid: np.ndarray, color: int, seed: int):
    points = [tuple(p) for p in np.argwhere(grid == color)]
    assert len(points) == 1, seed
    return points[0]


def _shortest_path(grid: np.ndarray, start, goal, wall_color: int, connectivity: int):
    q = deque([start])
    parent = {start: None}
    h, w = grid.shape
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in _neighbors(connectivity):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in parent and grid[nr, nc] != wall_color:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    assert goal in parent
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _shortest_path_count(grid: np.ndarray, start, goal, wall_color: int, connectivity: int):
    q = deque([start])
    dist = {start: 0}
    h, w = grid.shape
    while q:
        r, c = q.popleft()
        for dr, dc in _neighbors(connectivity):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != wall_color and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    assert goal in dist
    ways = {start: 1}
    for cell, _distance_value in sorted(dist.items(), key=lambda item: item[1]):
        r, c = cell
        for dr, dc in _neighbors(connectivity):
            nb = (r + dr, c + dc)
            if dist.get(nb) == dist[cell] + 1:
                ways[nb] = min(2, ways.get(nb, 0) + ways[cell])
    return ways.get(goal, 0)


def _trace_path(grid: np.ndarray, colors: set[int], connectivity: int = 4):
    cells = {tuple(p) for p in np.argwhere(np.isin(grid, list(colors)))}
    endpoints = []
    for r, c in cells:
        degree = sum((r + dr, c + dc) in cells for dr, dc in _neighbors(connectivity))
        if degree == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2
    start = min(endpoints)
    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [
            (cur[0] + dr, cur[1] + dc)
            for dr, dc in _neighbors(connectivity)
            if (cur[0] + dr, cur[1] + dc) in cells and (cur[0] + dr, cur[1] + dc) != prev
        ]
        assert len(nxt) <= 1
        if not nxt:
            return path
        prev, cur = cur, nxt[0]
        path.append(cur)


def _turns(path):
    total = 0
    prev = None
    for a, b in zip(path, path[1:]):
        step = (b[0] - a[0], b[1] - a[1])
        if prev is not None and step != prev:
            total += 1
        prev = step
    return total


def _components(grid: np.ndarray, color: int, connectivity: int):
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
                for dr, dc in _neighbors(connectivity):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _distance(a, b, metric: str):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc if metric == "manhattan" else max(dr, dc)


def _line_between(a, b):
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    cells = [a]
    cur = a
    while cur != b:
        cur = (cur[0] + dr, cur[1] + dc)
        cells.append(cur)
    return cells


def _assert_distractor_colors_present(grid: np.ndarray, meta: dict, seed: int):
    colors = meta.get("distractor_colors", [])
    assert colors, seed
    for color in colors:
        assert np.any(grid == color), (seed, color)


def test_task_109():
    def check(seed, grid, out, meta):
        start = _single_cell(grid, meta["start_color"], seed)
        goal = _single_cell(grid, meta["goal_color"], seed)
        path = _shortest_path(grid, start, goal, meta["wall_color"], meta["connectivity"])
        assert _shortest_path_count(grid, start, goal, meta["wall_color"], meta["connectivity"]) == 1, seed
        open_non_path = {tuple(p) for p in np.argwhere(grid != meta["wall_color"])} - set(path)
        assert open_non_path, seed
        expected = grid.copy()
        for p in path:
            expected[p] = meta["path_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("109", check)
    metas = [generate("109", seed)["metadata"] for seed in range(200)]
    assert {meta["path_shape"] for meta in metas} == {
        "horizontal",
        "vertical",
        "single_turn",
        "two_turn_horizontal",
        "two_turn_vertical",
    }
    prompts = [get_prompt("109", meta).lower() for meta in metas[:40]]
    assert all("zero-indexed row" in prompt and "zero-indexed column" in prompt for prompt in prompts)
    assert all("including the start and goal cells" in prompt for prompt in prompts)
    assert all("leave wall cells" in prompt and "non-path cell" in prompt for prompt in prompts)


def test_task_110():
    def check(seed, grid, out, meta):
        _assert_distractor_colors_present(grid, meta, seed)
        path = _trace_path(grid, {meta["path_color"]}, meta["connectivity"])
        length = len(path)
        turns = _turns(path)
        if meta["measure"] in {"turns", "both"}:
            assert turns > 0, seed
        if meta["measure"] == "length":
            expected = str(length)
        elif meta["measure"] == "turns":
            expected = str(turns)
        else:
            expected = f"length={length};turns={turns}"
        assert out == expected
    _verify("110", check)
    metas = [generate("110", seed)["metadata"] for seed in range(200)]
    assert {meta["measure"] for meta in metas} == {"length", "turns", "both"}
    assert any(meta["path_shape"] in {"horizontal", "vertical"} for meta in metas)
    assert any(meta["path_shape"] not in {"horizontal", "vertical"} for meta in metas)
    prompts = [get_prompt("110", meta).lower() for meta in metas[:40]]
    assert all("ignore every other color" in prompt for prompt in prompts)
    assert all("including both endpoints" in prompt for prompt in prompts)
    assert any("return only the integer length" in prompt for prompt in prompts)
    assert any("return only the integer turn count" in prompt for prompt in prompts)
    assert any("exact format length=n;turns=m" in prompt for prompt in prompts)


def test_task_112():
    def check(seed, grid, out, meta):
        source = _single_cell(grid, meta["source_color"], seed)
        _assert_distractor_colors_present(grid, meta, seed)
        expected = grid.copy()
        h, w = grid.shape
        for name in meta["directions"]:
            dr, dc = DIRECTIONS[name]
            r, c = source[0] + dr, source[1] + dc
            saw_ray_cell = False
            while 0 <= r < h and 0 <= c < w:
                if meta["stop_condition"] == "stop_color" and grid[r, c] == meta["stop_color"]:
                    assert saw_ray_cell, seed
                    assert out[r, c] == meta["stop_color"], seed
                    break
                expected[r, c] = meta["ray_color"]
                saw_ray_cell = True
                r += dr
                c += dc
            if meta["stop_condition"] == "stop_color":
                assert 0 <= r < h and 0 <= c < w, seed
            else:
                assert "stop_color" not in meta, seed
        assert out.tolist() == expected.tolist(), seed
        for color in meta["distractor_colors"]:
            mask = grid == color
            assert np.array_equal(out[mask], grid[mask]), seed
    _verify("112", check)
    metas = [generate("112", seed)["metadata"] for seed in range(200)]
    assert {meta["stop_condition"] for meta in metas} == {"grid_edge", "stop_color"}
    assert set().union(*(set(meta["directions"]) for meta in metas)) == set(DIRECTIONS)
    prompts = [get_prompt("112", meta).lower() for meta in metas[:40]]
    assert all("cast one ray in each of these directions" in prompt for prompt in prompts)
    assert all("ray starts at the adjacent cell" in prompt for prompt in prompts)
    assert all("outside the cast rays unchanged" in prompt for prompt in prompts)
    assert any("stop cells are not ray cells" in prompt for prompt in prompts)
    assert any("through the border cell" in prompt for prompt in prompts)


def test_task_115():
    def check(seed, grid, out, meta):
        _assert_distractor_colors_present(grid, meta, seed)
        cells = {tuple(p) for p in np.argwhere(grid == meta["line_color"])}
        endpoints = 0
        branchpoints = 0
        for r, c in cells:
            degree = sum((r + dr, c + dc) in cells for dr, dc in _neighbors(meta["connectivity"]))
            endpoints += int(degree == 1)
            branchpoints += int(degree >= 3)
        assert out == f"endpoints={endpoints};branchpoints={branchpoints}", seed
    _verify("115", check)
    branch_counts = {
        int(generate("115", seed)["output"].split("branchpoints=")[1])
        for seed in range(100)
    }
    assert len(branch_counts) >= 3, branch_counts
    metas = [generate("115", seed)["metadata"] for seed in range(200)]
    assert {meta["connectivity"] for meta in metas} == {4, 8}
    sensitive_cases = 0
    for seed in range(200):
        ex = generate("115", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        cells = {tuple(p) for p in np.argwhere(grid == meta["line_color"])}
        counts = {}
        for connectivity in (4, 8):
            endpoints = 0
            branchpoints = 0
            for r, c in cells:
                degree = sum((r + dr, c + dc) in cells for dr, dc in _neighbors(connectivity))
                endpoints += int(degree == 1)
                branchpoints += int(degree >= 3)
            counts[connectivity] = (endpoints, branchpoints)
        sensitive_cases += int(counts[4] != counts[8])
    assert sensitive_cases > 0
    prompts = [get_prompt("115", meta).lower() for meta in metas[:40]]
    assert all("ignore every other color" in prompt for prompt in prompts)
    assert all("neighboring line-color cells" in prompt for prompt in prompts)
    assert any("share an edge or a corner" in prompt for prompt in prompts)
    assert any("share an edge" in prompt for prompt in prompts)
    assert all("exact text endpoints=n;branchpoints=m" in prompt for prompt in prompts)


def test_task_140():
    def check(seed, grid, out, meta):
        points = [tuple(p) for p in np.argwhere(grid == meta["endpoint_color"])]
        assert len(points) == 2, seed
        segment = _line_between(points[0], points[1])
        expected = grid.copy()
        for p in segment:
            expected[p] = meta["line_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("140", check)


def test_task_141():
    def check(seed, grid, out, meta):
        del seed
        cells = [tuple(p) for p in np.argwhere(grid == meta["target_color"])]
        expected = grid.copy()
        h, w = grid.shape
        if meta["border"] in ("top", "bottom"):
            r = 0 if meta["border"] == "top" else h - 1
            for c in {c for _, c in cells}:
                expected[r, c] = meta["projection_color"]
        else:
            c = 0 if meta["border"] == "left" else w - 1
            for r in {r for r, _ in cells}:
                expected[r, c] = meta["projection_color"]
        assert out.tolist() == expected.tolist()
    _verify("141", check)


def test_task_142():
    def check(seed, grid, out, meta):
        anchor = _single_cell(grid, meta["anchor_color"], seed)
        if meta["condition"] == "target_cells":
            count = int(np.sum(grid == meta["target_color"]))
        else:
            count = len(_components(grid, meta["target_color"], meta["connectivity"]))
        dr, dc = DIRECTIONS[meta["direction"]]
        expected = grid.copy()
        for i in range(count):
            expected[anchor[0] + dr * i, anchor[1] + dc * i] = meta["bar_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("142", check)


def test_task_149():
    def check(seed, grid, out, meta):
        del seed
        sources = [tuple(p) for p in np.argwhere(grid == meta["source_color"])]
        expected = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != meta["background_color"]:
                    continue
                d = min(_distance((r, c), src, meta["metric"]) for src in sources)
                if d == meta["distance"]:
                    expected[r, c] = meta["ring_color"]
        assert out.tolist() == expected.tolist()
    _verify("149", check)


def test_task_159():
    def check(seed, grid, out, meta):
        start = _single_cell(grid, meta["start_endpoint_color"], seed)
        path = _trace_path(grid, {meta["path_color"], meta["start_endpoint_color"]}, meta["connectivity"])
        if path[0] != start:
            path = list(reversed(path))
        assert path[0] == start, seed
        expected = grid.copy()
        seq = meta["sequence_colors"]
        for i, p in enumerate(path):
            expected[p] = seq[i % len(seq)]
        assert out.tolist() == expected.tolist(), seed
    _verify("159", check)


def test_task_170():
    def check(seed, grid, out, meta):
        markers = [tuple(p) for p in np.argwhere(grid == meta["marker_color"])]
        assert len(markers) == 2, seed
        expected = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != meta["background_color"]:
                    continue
                if _distance((r, c), markers[0], meta["metric"]) == _distance((r, c), markers[1], meta["metric"]):
                    expected[r, c] = meta["bisector_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("170", check)


if __name__ == "__main__":
    tests = [
        test_task_109, test_task_110, test_task_112, test_task_115, test_task_140,
        test_task_141, test_task_142, test_task_149, test_task_159, test_task_170,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
