"""Verifier tests for object-focused second-batch slice A."""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate


# Tasks 97 and 98 were rebuilt on 2026-09-15 into generators/odd_shape_97.py and
# generators/sorted_strip_98.py; their tests live in tests/test_odd_shape_97.py
# and tests/test_sorted_strip_98.py.
TASK_IDS = ["105", "106", "108", "143", "144", "152", "161", "172"]
N_SEEDS = 30


def _components(
    grid: np.ndarray,
    bg: int,
    color: int | None = None,
    connectivity: int = 4,
) -> list[dict]:
    h, w = grid.shape
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
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
                for dr, dc in deltas:
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


def _bbox_size(box):
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _measure(comp: dict, name: str) -> int:
    if name == "cell_count":
        return len(comp["cells"])
    if name == "bounding_box_area":
        h, w = _bbox_size(comp["bbox"])
        return h * w
    if name == "bbox_height":
        return _bbox_size(comp["bbox"])[0]
    if name == "bbox_width":
        return _bbox_size(comp["bbox"])[1]
    raise AssertionError(f"unknown measure {name}")


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


def _landmark(comp: dict, name: str) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    if name == "top_left":
        return (r0, c0)
    if name == "top_right":
        return (r0, c1)
    if name == "bottom_left":
        return (r1, c0)
    if name == "bottom_right":
        return (r1, c1)
    if name == "medoid":
        return _medoid(comp["cells"])
    raise AssertionError(f"unknown landmark {name}")


def _distance(comp: dict, marker: tuple[int, int], metric: str) -> int:
    if metric in ("min_manhattan", "nearest_min_manhattan"):
        return min(abs(r - marker[0]) + abs(c - marker[1]) for r, c in comp["cells"])
    if metric in ("top_left_manhattan", "nearest_top_left_manhattan"):
        r, c = _landmark(comp, "top_left")
        return abs(r - marker[0]) + abs(c - marker[1])
    raise AssertionError(f"unknown metric {metric}")


def _nearest_marker_color(comp: dict, markers: list[dict], metric: str) -> int:
    marker = min(
        markers,
        key=lambda m: (
            _distance(comp, m["position"], metric),
            m["position"][0],
            m["position"][1],
        ),
    )
    return marker["color"]


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        fn(seed, grid, out, ex["metadata"])
        assert generate(task_id, seed) == ex


def _assert_colors_preserved(grid: np.ndarray, out: np.ndarray, colors: list[int], seed: int):
    for color in colors:
        mask = grid == color
        assert np.array_equal(out[mask], grid[mask]), seed
        assert np.array_equal(out[mask], np.full(int(np.sum(mask)), color, dtype=np.uint8)), seed


def test_task_105():
    def check(seed, grid, out, meta):
        markers = []
        for color in meta["marker_colors"]:
            hits = [tuple(p) for p in np.argwhere(grid == color)]
            assert len(hits) == 1, seed
            markers.append({"color": color, "position": hits[0]})
        expected = grid.copy()
        alternate_metric = "top_left_manhattan" if meta["distance_metric"] == "min_manhattan" else "min_manhattan"
        metric_sensitive = False
        for comp in _components(grid, meta["background_color"], meta["target_color"]):
            marker = min(
                markers,
                key=lambda m: (
                    _distance(comp, m["position"], meta["distance_metric"]),
                    m["position"][0],
                    m["position"][1],
                ),
            )
            metric_sensitive |= marker["color"] != _nearest_marker_color(comp, markers, alternate_metric)
            for r, c in comp["cells"]:
                expected[r, c] = marker["color"]
        assert metric_sensitive, seed
        assert out.tolist() == expected.tolist(), seed
        _assert_colors_preserved(grid, out, meta.get("distractor_colors", []), seed)
    _verify("105", check)
    metas = [generate("105", seed)["metadata"] for seed in range(120)]
    assert {meta["distance_metric"] for meta in metas} == {"min_manhattan", "top_left_manhattan"}
    assert {len(meta["marker_colors"]) for meta in metas} == {2, 3}
    prompts = [get_prompt("105", meta).lower() for meta in metas[:40]]
    assert not any(raw in prompt for prompt in prompts for raw in ["min_manhattan", "top_left_manhattan"])
    assert all("absolute row difference" in prompt and "absolute column difference" in prompt for prompt in prompts)
    assert all("smaller row" in prompt and "smaller column" in prompt for prompt in prompts)


def test_task_106():
    def check(seed, grid, out, meta):
        comps = _components(grid, meta["background_color"], meta["target_color"])
        assert len({_measure(comp, meta["size_measure"]) for comp in comps}) == len(comps), seed
        by_cell_count = [
            comp["bbox"]
            for comp in sorted(comps, key=lambda comp: (_measure(comp, "cell_count"), comp["bbox"][0], comp["bbox"][1]))
        ]
        by_bbox_area = [
            comp["bbox"]
            for comp in sorted(
                comps,
                key=lambda comp: (_measure(comp, "bounding_box_area"), comp["bbox"][0], comp["bbox"][1]),
            )
        ]
        assert by_cell_count != by_bbox_area, seed
        reverse = meta["rank_order"] == "largest_to_smallest"
        ordered = sorted(
            comps,
            key=lambda comp: (_measure(comp, meta["size_measure"]), comp["bbox"][0], comp["bbox"][1]),
            reverse=reverse,
        )
        color_by_rank = {item["rank"]: item["color"] for item in meta["rank_to_color_mapping"]}
        expected = grid.copy()
        for rank, comp in enumerate(ordered, start=1):
            for r, c in comp["cells"]:
                expected[r, c] = color_by_rank[rank]
        assert out.tolist() == expected.tolist(), seed
        _assert_colors_preserved(grid, out, meta.get("distractor_colors", []), seed)
    _verify("106", check)
    metas = [generate("106", seed)["metadata"] for seed in range(120)]
    assert {meta["size_measure"] for meta in metas} == {"cell_count", "bounding_box_area"}
    assert {meta["rank_order"] for meta in metas} == {"smallest_to_largest", "largest_to_smallest"}
    prompts = [get_prompt("106", meta).lower() for meta in metas[:40]]
    assert not any(raw in prompt for prompt in prompts for raw in ["cell_count", "bounding_box_area", "smallest_to_largest", "largest_to_smallest"])
    assert all("rank 1" in prompt and "measured sizes are unique" in prompt for prompt in prompts)


def test_task_108():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        colors = sorted(int(c) for c in np.unique(grid) if int(c) != bg)
        frequencies = {color: int(np.sum(grid == color)) for color in colors}
        assert len(set(frequencies.values())) == len(frequencies), seed
        ranked_colors = sorted(colors, key=lambda color: -frequencies[color])
        rank_to_color = {item["rank"]: item["color"] for item in meta["rank_to_color_mapping"]}
        expected = grid.copy()
        for rank, color in enumerate(ranked_colors, start=1):
            assert rank_to_color[rank] != color, seed
            expected[grid == color] = rank_to_color[rank]
        assert np.array_equal(out[grid == bg], grid[grid == bg]), seed
        assert out.tolist() == expected.tolist(), seed
    _verify("108", check)
    metas = [generate("108", seed)["metadata"] for seed in range(150)]
    assert {len(meta["rank_to_color_mapping"]) for meta in metas} == {3, 4, 5}
    for seed in range(150):
        example = generate("108", seed)
        grid = np.array(example["input"], dtype=np.uint8)
        bg = example["metadata"]["background_color"]
        freqs = [int(np.sum(grid == color)) for color in np.unique(grid) if int(color) != bg]
        assert len(set(freqs)) == len(freqs), seed
    prompts = [get_prompt("108", meta).lower() for meta in metas[:40]]
    assert all("count the cells" in prompt and "rank 1" in prompt for prompt in prompts)
    assert all("simultaneously" in prompt and "preserve background cells" in prompt for prompt in prompts)
    forbidden = ("lower", "greater", "earlier", "later", "color order", "color index", "ordered")
    assert all(term not in prompt for prompt in prompts for term in forbidden)


def test_task_143():
    def check(seed, grid, out, meta):
        comps = _components(grid, meta["background_color"], meta["target_color"])
        if meta["object_ordering"] == "top_to_bottom_left_to_right":
            key = lambda comp: (comp["bbox"][0], comp["bbox"][1])
        elif meta["object_ordering"] == "left_to_right_top_to_bottom":
            key = lambda comp: (comp["bbox"][1], comp["bbox"][0])
        else:
            key = lambda comp: (-comp["bbox"][2], comp["bbox"][1])
        ordered = sorted(comps, key=key)
        color_by_index = {item["index"]: item["color"] for item in meta["index_to_color_mapping"]}
        expected = grid.copy()
        for index, comp in enumerate(ordered, start=1):
            for r, c in comp["cells"]:
                expected[r, c] = color_by_index[index]
        assert out.tolist() == expected.tolist(), seed
        _assert_colors_preserved(grid, out, meta.get("distractor_colors", []), seed)
    _verify("143", check)


def test_task_144():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        for comp in _components(grid, meta["background_color"], meta["target_color"]):
            expected[_medoid(comp["cells"])] = meta["marker_color"]
        assert out.tolist() == expected.tolist(), seed
        _assert_colors_preserved(grid, out, meta.get("distractor_colors", []), seed)
    _verify("144", check)


def test_task_152():
    def check(seed, grid, out, meta):
        marker_hits = [tuple(p) for p in np.argwhere(grid == meta["marker_color"])]
        assert len(marker_hits) == 1, seed
        marker = marker_hits[0]
        comps = _components(grid, meta["background_color"], meta["target_color"])
        keep = min(
            comps,
            key=lambda comp: (
                _distance(comp, marker, meta["association_rule"]),
                comp["bbox"][0],
                comp["bbox"][1],
            ),
        )
        expected = grid.copy()
        for comp in comps:
            if comp == keep:
                continue
            for r, c in comp["cells"]:
                expected[r, c] = meta["background_color"]
        assert out.tolist() == expected.tolist(), seed
        _assert_colors_preserved(grid, out, meta.get("distractor_colors", []), seed)
    _verify("152", check)


def test_task_161():
    def check(seed, grid, out, meta):
        expected = np.full_like(grid, meta["background_color"])
        for comp in _components(grid, meta["background_color"]):
            color = comp["color"] if meta["color_rule"] == "original_color" else meta["marker_color"]
            expected[_landmark(comp, meta["landmark"])] = color
        assert out.tolist() == expected.tolist(), seed
    _verify("161", check)


def test_task_172():
    def check(seed, grid, out, meta):
        dirs = {
            "right": (0, 1),
            "left": (0, -1),
            "down": (1, 0),
            "up": (-1, 0),
        }
        dr, dc = dirs[meta["direction"]]
        expected = np.full_like(grid, meta["background_color"])
        for comp in _components(grid, meta["background_color"]):
            start = _landmark(comp, meta["landmark"])
            for i in range(len(comp["cells"])):
                expected[start[0] + dr * i, start[1] + dc * i] = comp["color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("172", check)


if __name__ == "__main__":
    tests = [
        test_task_97,
        test_task_98,
        test_task_105,
        test_task_106,
        test_task_108,
        test_task_143,
        test_task_144,
        test_task_152,
        test_task_161,
        test_task_172,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
