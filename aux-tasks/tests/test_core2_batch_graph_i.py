"""Verifier tests for graph/count batch I tasks."""
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


TASK_IDS = ["197", "198", "199", "200", "201", "202", "203", "227", "228", "229"]
N_SEEDS = 30
DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple("abcdefghij")


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        json.dumps(ex)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        if seed < 4:
            user_text, answer_text = format_single(task_id, seed, color_palette=PALETTE)
            assert user_text and answer_text
        output = np.array(ex["output"], dtype=np.uint8) if not isinstance(ex["output"], str) else ex["output"]
        fn(seed, np.array(ex["input"], dtype=np.uint8), output, ex["metadata"])
        assert generate(task_id, seed) == ex


def _components(grid: np.ndarray, bg: int, color: int | None = None) -> list[dict]:
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
            comp_color = int(grid[r, c])
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == comp_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append({
                "color": comp_color,
                "cells": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            })
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))
    return comps


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _lookup(mapping: list[dict], key: str) -> dict[int, int]:
    return {int(entry[key]): int(entry["color"]) for entry in mapping}


def _hole_count(grid: np.ndarray, comp: dict, bg: int) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or sub[r, c] != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                touches_edge |= cr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and sub[nr, nc] == bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _min_manhattan_gap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _edges(objects: list[dict], meta: dict) -> set[tuple[int, int]]:
    relation = meta["adjacency_relation"]
    edges = set()
    for i, a in enumerate(objects):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for j, b in enumerate(objects[i + 1:], start=i + 1):
            br0, bc0, br1, bc1 = b["bbox"]
            if relation == "same_row_band":
                linked = _intervals_overlap(ar0, ar1, br0, br1)
            elif relation == "same_column_band":
                linked = _intervals_overlap(ac0, ac1, bc0, bc1)
            else:
                linked = _min_manhattan_gap(a["cells"], b["cells"]) <= meta["gap_threshold"]
            if linked:
                edges.add((i, j))
    return edges


def _bbox_property(comp: dict, prop: str) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if prop == "bbox_height":
        return height
    if prop == "bbox_width":
        return width
    if prop == "bbox_area":
        return height * width
    return 2 * (height + width)


def _equivalent(a: dict, b: dict, relation: str) -> bool:
    if relation == "same_shape":
        return _normalize(a["cells"]) == _normalize(b["cells"])
    if relation == "same_bbox_size":
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        return (ar1 - ar0 + 1, ac1 - ac0 + 1) == (br1 - br0 + 1, bc1 - bc0 + 1)
    return len(a["cells"]) == len(b["cells"])


def _apply_action(expected: np.ndarray, comp: dict, action: str, color: int, bg: int) -> None:
    if action == "recolor_object":
        for cell in comp["cells"]:
            expected[cell] = color
    elif action == "mark_first_cell":
        expected[min(comp["cells"])] = color
    else:
        for cell in comp["cells"]:
            expected[cell] = bg


def _count_set(grid: np.ndarray, color: int, bg: int, count_type: str) -> int:
    if count_type == "cells":
        return int(np.count_nonzero(grid == color))
    return len(_components(grid, bg, color))


def _operation(a: int, b: int, op: str) -> int:
    if op == "sum":
        return a + b
    if op == "absolute_difference":
        return abs(a - b)
    if op == "a_minus_b_floor0":
        return max(0, a - b)
    if op == "b_minus_a_floor0":
        return max(0, b - a)
    return a * b


def test_task_197():
    def check(seed, grid, out, meta):
        row_marks = [tuple(p) for p in np.argwhere(grid[:, 0] == meta["row_marker_color"])]
        col_marks = [tuple(p) for p in np.argwhere(grid[0, :] == meta["column_marker_color"])]
        assert len(row_marks) == 1 and len(col_marks) == 1, seed
        mark_r = row_marks[0][0]
        mark_c = col_marks[0][0]
        divider = meta["divider_color"]
        panel = grid[1:, 1:]
        divider_rows = {r + 1 for r in range(panel.shape[0]) if np.all(panel[r, :] == divider)}
        divider_cols = {c + 1 for c in range(panel.shape[1]) if np.all(panel[:, c] == divider)}
        row_run = [r for r in range(1, grid.shape[0]) if r not in divider_rows and not any(d in range(min(r, mark_r) + 1, max(r, mark_r) + 1) for d in divider_rows)]
        col_run = [c for c in range(1, grid.shape[1]) if c not in divider_cols and not any(d in range(min(c, mark_c) + 1, max(c, mark_c) + 1) for d in divider_cols)]
        expected = grid[min(row_run):max(row_run) + 1, min(col_run):max(col_run) + 1]
        assert expected.shape == (meta["block_height"], meta["block_width"])
        assert out.tolist() == expected.tolist(), seed
    _verify("197", check)


def test_task_198():
    def check(seed, grid, out, meta):
        order = meta["marker_order"]
        if meta["axis"] == "rows":
            index = {int(grid[r, 0]): r for r in range(grid.shape[0])}
            expected = grid[[index[color] for color in order], :]
        else:
            index = {int(grid[0, c]): c for c in range(grid.shape[1])}
            expected = grid[:, [index[color] for color in order]]
        assert out.tolist() == expected.tolist(), seed
    _verify("198", check)


def test_task_199():
    def check(seed, grid, out, meta):
        comps = _components(grid, meta["background_color"], meta["object_color"])
        assert 2 <= len(comps) <= 4, seed
        expected = grid.copy()
        for comp in comps:
            for cell in comp["cells"]:
                expected[cell] = meta["background_color"]
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            if meta["axis"] == "row":
                if meta["landmark"] == "top_edge":
                    nr0 = meta["target_index"]
                elif meta["landmark"] == "bottom_edge":
                    nr0 = meta["target_index"] - (r1 - r0)
                else:
                    nr0 = meta["target_index"] - (r1 - r0) // 2
                nc0 = c0
            else:
                nr0 = r0
                if meta["landmark"] == "left_edge":
                    nc0 = meta["target_index"]
                elif meta["landmark"] == "right_edge":
                    nc0 = meta["target_index"] - (c1 - c0)
                else:
                    nc0 = meta["target_index"] - (c1 - c0) // 2
            for r, c in comp["cells"]:
                expected[nr0 + r - r0, nc0 + c - c0] = meta["object_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("199", check)


def test_task_200():
    def check(seed, grid, out, meta):
        lookup = _lookup(meta["hole_count_to_color"], "hole_count")
        expected = grid.copy()
        counts = []
        for comp in _components(grid, meta["background_color"], meta["object_color"]):
            count = _hole_count(grid, comp, meta["background_color"])
            counts.append(count)
            assert count in lookup, seed
            for cell in comp["cells"]:
                expected[cell] = lookup[count]
        # Hole counts vary per object; require at least two distinct counts so
        # the task cannot be solved by silhouette memorization, and the mapping
        # must cover exactly the present counts.
        assert len(set(counts)) >= 2, seed
        assert sorted(lookup) == sorted(set(counts)), seed
        assert out.tolist() == expected.tolist(), seed
    _verify("200", check)


def test_task_201():
    def check(seed, grid, out, meta):
        objects = _components(grid, meta["background_color"], meta["object_color"])
        assert 4 <= len(objects) <= 7, seed
        degrees = [0] * len(objects)
        for i, j in _edges(objects, meta):
            degrees[i] += 1
            degrees[j] += 1
        lookup = _lookup(meta["degree_to_color"], "degree")
        # Degrees vary; the mapping must cover exactly the present degrees.
        assert len(set(degrees)) >= 2, seed
        assert sorted(lookup) == sorted(set(degrees)), seed
        expected = grid.copy()
        for comp, degree in zip(objects, degrees):
            assert degree in lookup, seed
            for cell in comp["cells"]:
                expected[cell] = lookup[degree]
        assert out.tolist() == expected.tolist(), seed
    _verify("201", check)


def test_task_202():
    def check(seed, grid, out, meta):
        a_points = [tuple(p) for p in np.argwhere(grid == meta["object_a_color"])]
        b_points = [tuple(p) for p in np.argwhere(grid == meta["object_b_color"])]
        expected = grid.copy()
        for a in a_points:
            if meta["pairing_axis"] == "horizontal":
                candidates = [b for b in b_points if b[0] == a[0]]
            else:
                candidates = [b for b in b_points if b[1] == a[1]]
            assert len(candidates) == 1, seed
            b = candidates[0]
            dr = int(np.sign(b[0] - a[0]))
            dc = int(np.sign(b[1] - a[1]))
            r, c = a[0] + dr, a[1] + dc
            while (r, c) != b:
                assert expected[r, c] == meta["background_color"], seed
                expected[r, c] = meta["fill_color"]
                r += dr
                c += dc
        assert out.tolist() == expected.tolist(), seed
    _verify("202", check)


def test_task_203():
    def check(seed, grid, out, meta):
        lookup = _lookup(meta["property_to_color"], "property_value")
        expected = grid.copy()
        for comp in _components(grid, meta["background_color"], meta["object_color"]):
            value = _bbox_property(comp, meta["bbox_property"])
            assert value in lookup, seed
            for cell in comp["cells"]:
                expected[cell] = lookup[value]
        assert out.tolist() == expected.tolist(), seed
    _verify("203", check)


def test_task_227():
    def check(seed, grid, out, meta):
        targets = _components(grid, meta["background_color"], meta["target_color"])
        refs = _components(grid, meta["background_color"], meta["reference_color"])
        assert targets and refs, seed
        expected = grid.copy()
        classes = []
        for target in targets:
            matched = any(_equivalent(target, ref, meta["equivalence_relation"]) for ref in refs)
            classes.append(matched)
            if matched:
                _apply_action(expected, target, meta["matched_action"], meta["matched_color"], meta["background_color"])
            else:
                _apply_action(expected, target, meta["unmatched_action"], meta["unmatched_color"], meta["background_color"])
        assert any(classes) and not all(classes), seed
        assert meta["matched_action"] != meta["unmatched_action"], seed
        assert out.tolist() == expected.tolist(), seed
    _verify("227", check)


def test_task_228():
    def check(seed, grid, out, meta):
        count_a = _count_set(grid, meta["set_a_color"], meta["background_color"], meta["set_a_count_type"])
        count_b = _count_set(grid, meta["set_b_color"], meta["background_color"], meta["set_b_count_type"])
        expected = str(_operation(count_a, count_b, meta["operation"]))
        assert re.fullmatch(r"\d+", out), seed
        assert out == expected, seed
    _verify("228", check)


def test_task_229():
    def check(seed, grid, out, meta):
        count_a = _count_set(grid, meta["set_a_color"], meta["background_color"], meta["set_a_count_type"])
        count_b = _count_set(grid, meta["set_b_color"], meta["background_color"], meta["set_b_count_type"])
        length = _operation(count_a, count_b, meta["operation"])
        anchors = [tuple(p) for p in np.argwhere(grid == meta["anchor_color"])]
        assert len(anchors) == 1 and length > 0, seed
        expected = grid.copy()
        dr, dc = (0, 1) if meta["bar_direction"] == "east" else (1, 0)
        for k in range(length):
            expected[anchors[0][0] + dr * k, anchors[0][1] + dc * k] = meta["bar_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("229", check)


# Raw internal enum tokens that must never leak into a rendered prompt; the
# blind solver flagged 201's relation names and 227's first-cell action as
# undefined, so the prompts now spell out each meaning instead.
_LEAKED_ENUM_TOKENS = (
    "same_row_band", "same_column_band", "manhattan_gap_le_threshold",
    "same row band", "same column band", "manhattan gap le threshold",
    "mark_first_cell", "recolor_object", "erase_object",
)


def test_task_201_prompt_defines_relation():
    seen_relations = set()
    for seed in range(40):
        ex = generate("201", seed)
        meta = ex["metadata"]
        prompt = get_prompt("201", meta)
        for tok in _LEAKED_ENUM_TOKENS:
            assert tok not in prompt, (seed, tok)
        # Degree is defined explicitly and the mapping is present.
        assert "degree" in prompt, seed
        rel = meta["adjacency_relation"]
        seen_relations.add(rel)
        if rel == "same_row_band":
            assert "overlap in rows" in prompt, seed
        elif rel == "same_column_band":
            assert "overlap in columns" in prompt, seed
        else:
            # Manhattan-gap relation: the comparison and threshold are concrete.
            assert "Manhattan distance" in prompt, seed
            assert f"at most {meta['gap_threshold']}" in prompt, seed
    # All three relations should appear across the sweep (generator coverage).
    assert seen_relations == {"same_row_band", "same_column_band", "manhattan_gap_le_threshold"}


def test_task_227_prompt_marks_first_cell_keeps_rest():
    saw_mark = False
    for seed in range(40):
        ex = generate("227", seed)
        meta = ex["metadata"]
        prompt = get_prompt("227", meta)
        for tok in _LEAKED_ENUM_TOKENS:
            assert tok not in prompt, (seed, tok)
        if "mark_first_cell" in (meta["matched_action"], meta["unmatched_action"]):
            saw_mark = True
            # First cell is defined in row-major terms and the remaining cells
            # are explicitly said to keep their original target color.
            assert "first cell in row-major order" in prompt, seed
            assert "remaining cells" in prompt, seed
    assert saw_mark


if __name__ == "__main__":
    tests = [
        test_task_197,
        test_task_198,
        test_task_199,
        test_task_200,
        test_task_201,
        test_task_202,
        test_task_203,
        test_task_227,
        test_task_228,
        test_task_229,
        test_task_201_prompt_defines_relation,
        test_task_227_prompt_marks_first_cell_keeps_rest,
    ]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
