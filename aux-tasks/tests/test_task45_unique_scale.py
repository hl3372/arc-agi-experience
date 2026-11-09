"""Regression checks for Task 45 inferred tiling-scale uniqueness."""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from registry import generate


def _exact_tiling_shapes(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return all top-left rectangular patterns that tile the grid exactly."""
    h, w = grid.shape
    shapes = []
    for tile_h in range(1, h + 1):
        if h % tile_h:
            continue
        for tile_w in range(1, w + 1):
            if w % tile_w:
                continue
            tile = grid[:tile_h, :tile_w]
            if np.array_equal(grid, np.tile(tile, (h // tile_h, w // tile_w))):
                shapes.append((tile_h, tile_w))
    return shapes


def _answer_to_repeats(answer: str) -> tuple[int, int]:
    vertical, horizontal = answer.lower().split("x")
    return int(vertical), int(horizontal)


def _assert_unique_task45_answer(seed: int) -> None:
    example = generate("45", seed)
    grid = np.array(example["input"], dtype=np.uint8)
    h, w = grid.shape
    m, n = _answer_to_repeats(example["output"])

    assert h % m == 0 and w % n == 0, seed
    expected_tile = (h // m, w // n)

    shapes = _exact_tiling_shapes(grid)
    min_area = min(tile_h * tile_w for tile_h, tile_w in shapes)
    smallest_shapes = [
        (tile_h, tile_w)
        for tile_h, tile_w in shapes
        if tile_h * tile_w == min_area
    ]

    assert smallest_shapes == [expected_tile], (
        seed,
        example["output"],
        grid.shape,
        example.get("metadata"),
        smallest_shapes,
        shapes,
    )


def test_task45_seed_sweep_has_unique_smallest_visible_tile():
    for seed in range(2000):
        _assert_unique_task45_answer(seed)


def test_task45_retains_repeat_count_variety():
    answers = Counter(generate("45", seed)["output"] for seed in range(1000))
    repeat_pairs = {_answer_to_repeats(answer) for answer in answers}
    assert len(answers) >= 50
    assert any(m == 1 and n > 1 for m, n in repeat_pairs)
    assert any(n == 1 and m > 1 for m, n in repeat_pairs)
