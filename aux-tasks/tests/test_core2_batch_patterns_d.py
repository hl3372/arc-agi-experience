"""Verifier tests for core2 pattern batch D tasks."""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.objects import connected_components
from prompts import get_prompt
from registry import generate


TASK_IDS = ["131", "132", "139", "150", "160", "163", "165", "166", "171", "180"]
N_SEEDS = 30


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        fn(seed, grid, out, ex["metadata"])
        assert generate(task_id, seed) == ex


def _line_summary(values: list[int], meta: dict) -> int:
    if meta["summary_rule"] == "majority_color":
        counts = Counter(values)
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        if len(winners) == 1:
            return int(winners[0])
        return int(meta["background_color"])
    if meta["summary_rule"] == "first_non_background":
        for value in values:
            if value != meta["background_color"]:
                return int(value)
        return int(meta["background_color"])
    if meta["summary_rule"] == "target_presence":
        return int(meta["true_color"] if meta["target_color"] in values else meta["false_color"])
    raise AssertionError(meta["summary_rule"])


def test_task_131():
    def check(seed, grid, out, meta):
        source = np.array(meta["source_motif"], dtype=np.uint8)
        target = np.array(meta["target_motif"], dtype=np.uint8)
        expected = grid.copy()
        mh, mw = source.shape
        hits = []
        for r in range(grid.shape[0] - mh + 1):
            for c in range(grid.shape[1] - mw + 1):
                if np.array_equal(grid[r:r + mh, c:c + mw], source):
                    hits.append((r, c))
        assert hits, seed
        for r, c in hits:
            expected[r:r + mh, c:c + mw] = target
        assert out.tolist() == expected.tolist(), seed
    _verify("131", check)


def test_task_132():
    def check(seed, grid, out, meta):
        table = {entry["color"]: np.array(entry["motif"], dtype=np.uint8) for entry in meta["replacement_table"]}
        mh = meta["motif_height"]
        mw = meta["motif_width"]
        expected = np.zeros((grid.shape[0] * mh, grid.shape[1] * mw), dtype=np.uint8)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                expected[r * mh:(r + 1) * mh, c * mw:(c + 1) * mw] = table[int(grid[r, c])]
        assert out.tolist() == expected.tolist(), seed
    _verify("132", check)


def test_task_139():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        bg = meta["background_color"]
        target = meta["target_color"]
        fill = meta["fill_color"]
        for comp in connected_components(grid, connectivity=meta["connectivity"], background=bg):
            if comp["color"] != target:
                continue
            cells = comp["cells"]
            if meta["axis"] in ("rows", "both"):
                for r in sorted({p[0] for p in cells}):
                    cols = sorted(c for rr, c in cells if rr == r)
                    if len(cols) >= 2:
                        for c in range(cols[0] + 1, cols[-1]):
                            if expected[r, c] == bg:
                                expected[r, c] = fill
            if meta["axis"] in ("columns", "both"):
                for c in sorted({p[1] for p in cells}):
                    rows = sorted(r for r, cc in cells if cc == c)
                    if len(rows) >= 2:
                        for r in range(rows[0] + 1, rows[-1]):
                            if expected[r, c] == bg:
                                expected[r, c] = fill
        assert out.tolist() == expected.tolist(), seed
    _verify("139", check)


def test_task_150():
    def check(seed, grid, out, meta):
        coords = np.argwhere(grid == meta["corner_marker_color"])
        assert coords.shape[0] == 2, seed
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        expected = grid.copy()
        if meta["rectangle_mode"] == "filled":
            expected[r0:r1 + 1, c0:c1 + 1] = meta["draw_color"]
        else:
            expected[r0:r1 + 1, c0] = meta["draw_color"]
            expected[r0:r1 + 1, c1] = meta["draw_color"]
            expected[r0, c0:c1 + 1] = meta["draw_color"]
            expected[r1, c0:c1 + 1] = meta["draw_color"]
        assert out.tolist() == expected.tolist(), seed
        distractors = np.argwhere(grid == meta["distractor_color"])
        assert all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in distractors), seed
    _verify("150", check)


def test_task_160():
    def check(seed, grid, out, meta):
        border_cells = np.argwhere(grid == meta["region_border_color"])
        r0, c0 = border_cells.min(axis=0)
        r1, c1 = border_cells.max(axis=0)
        anchors = np.argwhere(grid == meta["anchor_color"])
        assert anchors.shape[0] == 1, seed
        ar, ac = (int(anchors[0, 0]), int(anchors[0, 1]))
        expected = grid.copy()
        for r in range(int(r0) + 1, int(r1)):
            for c in range(int(c0) + 1, int(c1)):
                expected[r, c] = meta["anchor_color"] if (r - ar + c - ac) % 2 == 0 else meta["other_checker_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("160", check)


def test_task_163():
    def check(seed, grid, out, meta):
        rows = np.where(np.any(grid == meta["row_marker_color"], axis=1))[0]
        cols = np.where(np.any(grid == meta["column_marker_color"], axis=0))[0]
        expected = grid[np.ix_(rows, cols)]
        assert rows.size >= 2 and cols.size >= 2, seed
        assert out.tolist() == expected.tolist(), seed
    _verify("163", check)


def test_task_165():
    forbidden = ("lowest", "lower", "earlier", "later", "index", "ordinal", "ascending")

    def check(seed, grid, out, meta):
        prompt = get_prompt("165", meta).lower()
        metadata_text = str(meta).lower()
        assert all(term not in prompt for term in forbidden), seed
        assert all(term not in metadata_text for term in forbidden), seed
        if meta["axis"] == "rows":
            expected = np.array([[_line_summary([int(v) for v in grid[r, :]], meta)] for r in range(grid.shape[0])], dtype=np.uint8)
        else:
            expected = np.array([[_line_summary([int(v) for v in grid[:, c]], meta) for c in range(grid.shape[1])]], dtype=np.uint8)
        assert out.tolist() == expected.tolist(), seed
    _verify("165", check)

    tied_majority_seen = False
    for seed in range(N_SEEDS):
        ex = generate("165", seed)
        meta = ex["metadata"]
        if meta["summary_rule"] != "majority_color":
            continue
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        lines = [grid[r, :] for r in range(grid.shape[0])] if meta["axis"] == "rows" else [grid[:, c] for c in range(grid.shape[1])]
        for index, line in enumerate(lines):
            counts = Counter(int(v) for v in line)
            max_count = max(counts.values())
            if sum(1 for count in counts.values() if count == max_count) > 1:
                tied_majority_seen = True
                actual = int(out[index, 0]) if meta["axis"] == "rows" else int(out[0, index])
                assert actual == meta["background_color"], seed
    assert tied_majority_seen


def test_task_166():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        expected = np.full(grid.shape, bg, dtype=np.uint8)
        target = meta["target_color"]
        bar = meta["bar_color"]
        if meta["count_axis"] == "rows":
            for r in range(grid.shape[0]):
                count = int(np.count_nonzero(grid[r, :] == target))
                if meta["edge"] == "left":
                    expected[r, :count] = bar
                else:
                    expected[r, grid.shape[1] - count:] = bar
        else:
            for c in range(grid.shape[1]):
                count = int(np.count_nonzero(grid[:, c] == target))
                if meta["edge"] == "top":
                    expected[:count, c] = bar
                else:
                    expected[grid.shape[0] - count:, c] = bar
        assert out.tolist() == expected.tolist(), seed
    _verify("166", check)


def test_task_171():
    def check(seed, grid, out, meta):
        assert meta["ordering"] in {"frequency_desc", "first_seen_row_major"}, seed
        flat_values = [int(v) for v in grid.flatten()]
        first_seen = []
        seen = set()
        for value in flat_values:
            if value not in seen:
                first_seen.append(value)
                seen.add(value)
        if meta["ordering"] == "frequency_desc":
            assert meta["tie_break"] == "first_seen_row_major", seed
            counts = Counter(flat_values)
            ordered = sorted(first_seen, key=lambda color: -counts[color])
        else:
            ordered = first_seen
        expected = np.array([ordered], dtype=np.uint8) if meta["output_orientation"] == "row" else np.array([[c] for c in ordered], dtype=np.uint8)
        assert out.tolist() == expected.tolist(), seed
    _verify("171", check)


def test_task_180():
    def check(seed, grid, out, meta):
        t = meta["thickness"]
        top = t if "top" in meta["sides"] else 0
        bottom = t if "bottom" in meta["sides"] else 0
        left = t if "left" in meta["sides"] else 0
        right = t if "right" in meta["sides"] else 0
        expected = grid[top:grid.shape[0] - bottom if bottom else grid.shape[0], left:grid.shape[1] - right if right else grid.shape[1]]
        assert out.tolist() == expected.tolist(), seed
    _verify("180", check)
    side_sets = {tuple(generate("180", seed)["metadata"]["sides"]) for seed in range(200)}
    assert len(side_sets) == 15, side_sets


if __name__ == "__main__":
    for task_id in TASK_IDS:
        globals()[f"test_task_{task_id}"]()
