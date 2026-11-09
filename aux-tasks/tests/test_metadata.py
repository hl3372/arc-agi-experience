"""Tests for metadata family generators (tasks 1, 2, 25, 30, 75, 76)."""
from __future__ import annotations

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from registry import generate
from common.grid import NUM_COLORS

N_SEEDS = 100


def test_task_1_grid_size():
    for seed in range(N_SEEDS):
        ex = generate("1", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        h, w = grid.shape
        assert ex["output"] == f"{h}x{w}", f"seed={seed}: expected {h}x{w}, got {ex['output']}"


def test_task_2_coordinate_listing():
    for seed in range(N_SEEDS):
        ex = generate("2", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        target = meta["query_color_index"]

        # Parse coordinates from output
        coord_str = ex["output"]
        assert coord_str.startswith("[") and coord_str.endswith("]")
        inner = coord_str[1:-1]
        coords = []
        if inner:
            for pair in inner.split("),("):
                pair = pair.strip("()")
                r, c = pair.split(",")
                coords.append((int(r), int(c)))

        # Verify each coordinate has target color
        for r, c in coords:
            assert grid[r, c] == target, f"seed={seed}: ({r},{c}) is {grid[r,c]}, expected {target}"

        # Verify count matches
        actual = list(zip(*np.where(grid == target)))
        assert len(coords) == len(actual), f"seed={seed}: {len(coords)} coords but {len(actual)} cells"
        assert len(coords) <= 60, f"seed={seed}: too many coords: {len(coords)}"

        # Verify row-major order
        for i in range(len(coords) - 1):
            assert coords[i] <= coords[i + 1], f"seed={seed}: not row-major sorted"


# Task 25's background-inference tests live in test_background_color_25.py.


def test_task_2_unchanged_by_the_task_25_rebuild():
    """Task 2 shares the module; its 100-seed hash is pinned."""
    import hashlib, json

    h = hashlib.sha256()
    for seed in range(100):
        ex = generate("2", seed)
        h.update(json.dumps({"i": ex["input"], "o": ex["output"], "m": ex.get("metadata", {})},
                            sort_keys=True, default=str).encode())
    assert h.hexdigest() == TASK2_HASH_PIN


TASK2_HASH_PIN = "2b3921e9450b649cc32b3ebbbb13aab190212ed332cfc61db84903f4321209f3"


def test_task_30_count_cells():
    for seed in range(N_SEEDS):
        ex = generate("30", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        meta = ex["metadata"]
        target = meta["query_color"]
        actual_count = int(np.count_nonzero(grid == target))
        assert ex["output"] == str(actual_count), f"seed={seed}: expected {actual_count}, got {ex['output']}"


def test_task_75_center_color():
    for seed in range(N_SEEDS):
        ex = generate("75", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        h, w = grid.shape
        assert h % 2 == 1 and w % 2 == 1, f"seed={seed}: dims {h}x{w} not both odd"
        center_val = int(grid[h // 2, w // 2])
        # Generator emits the color index as a string (remapped to a palette
        # symbol at training time), not the color name.
        expected = str(center_val)
        assert ex["output"] == expected, f"seed={seed}: expected {expected}, got {ex['output']}"


def test_task_76_histogram():
    for seed in range(N_SEEDS):
        ex = generate("76", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        counts = np.bincount(grid.flatten(), minlength=NUM_COLORS)

        # Parse output
        pairs = []
        for entry in ex["output"].split(", "):
            name, cnt = entry.split(":")
            pairs.append((name.strip(), int(cnt)))

        # Verify counts match
        for name, cnt in pairs:
            from common.grid import COLOR_INDEX
            idx = COLOR_INDEX[name]
            assert counts[idx] == cnt, f"seed={seed}: {name} expected {counts[idx]}, got {cnt}"

        # Counts are unique, so no fallback rule is needed.
        seen_counts = [cnt for _, cnt in pairs]
        assert len(seen_counts) == len(set(seen_counts)), (
            f"seed={seed}: duplicate color counts in {pairs}"
        )

        # Verify sorted by count only.
        for i in range(len(pairs) - 1):
            _, cnt_a = pairs[i]
            _, cnt_b = pairs[i + 1]
            assert cnt_a < cnt_b, (
                f"seed={seed}: sort violated: {pairs[i]} before {pairs[i+1]}"
            )

        # At least 2 colors
        assert len(pairs) >= 2, f"seed={seed}: only {len(pairs)} colors"


def test_reproducibility():
    for task_id in ["1", "2", "25", "30", "75", "76"]:
        r1 = generate(task_id, seed=42)
        r2 = generate(task_id, seed=42)
        assert r1 == r2, f"Task {task_id} not reproducible"


if __name__ == "__main__":
    tests = [
        ("task_1_grid_size", test_task_1_grid_size),
        ("task_2_coordinate_listing", test_task_2_coordinate_listing),
        ("task_2_unchanged_by_the_task_25_rebuild", test_task_2_unchanged_by_the_task_25_rebuild),
        ("task_30_count_cells", test_task_30_count_cells),
        ("task_75_center_color", test_task_75_center_color),
        ("task_76_histogram", test_task_76_histogram),
        ("reproducibility", test_reproducibility),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
