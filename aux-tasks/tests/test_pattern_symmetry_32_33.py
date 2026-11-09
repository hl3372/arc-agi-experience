"""Focused checks for Task 32/33 pattern-family symmetry variants."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import generators.pattern_symmetry as pattern_symmetry
from prompts import get_prompt
from registry import generate

FULL = "full_square_symmetry"


def _all_d4_transforms(grid: np.ndarray) -> list[np.ndarray]:
    return [
        grid,
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        np.flipud(grid),
        np.fliplr(grid),
        grid.T,
        np.flipud(np.fliplr(grid)).T,
    ]


def _occluder_box(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid == color)
    assert len(rows) > 0
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    assert np.all(grid[r0:r1 + 1, c0:c1 + 1] == color)
    return r0, c0, r1 - r0 + 1, c1 - c0 + 1


def test_full_square_symmetry_orbit_sizes():
    assert pattern_symmetry._symmetry_orbit(FULL, 2, 2, 5, 5) == ((2, 2),)
    assert set(pattern_symmetry._symmetry_orbit(FULL, 1, 1, 4, 4)) == {
        (1, 1), (1, 2), (2, 1), (2, 2)
    }
    assert set(pattern_symmetry._symmetry_orbit(FULL, 0, 0, 5, 5)) == {
        (0, 0), (0, 4), (4, 0), (4, 4)
    }
    assert len(pattern_symmetry._symmetry_orbit(FULL, 1, 1, 5, 5)) == 4
    assert len(pattern_symmetry._symmetry_orbit(FULL, 1, 2, 6, 6)) == 8


def test_full_square_symmetry_pattern_satisfies_all_square_transforms(monkeypatch):
    from generators import pattern_completion_32_33 as completion
    monkeypatch.setattr(completion, "FAMILIES", (FULL,))
    monkeypatch.setattr(completion, "WEIGHTS", (1.,))
    ex = generate("32", 123)
    grid = np.asarray(ex["output"])
    assert ex["metadata"]["pattern_family"] == FULL
    assert grid.shape[0] == grid.shape[1]
    assert len(np.unique(grid)) >= 2
    assert pattern_symmetry._grid_satisfies_family(grid, FULL)
    for transformed in _all_d4_transforms(grid):
        assert np.array_equal(grid, transformed)


def test_tasks_32_33_can_be_forced_to_full_square_symmetry(monkeypatch):
    from generators import pattern_completion_32_33 as completion
    monkeypatch.setattr(completion, "FAMILIES", (FULL,))
    monkeypatch.setattr(completion, "WEIGHTS", (1.,))

    for task_id in ("32", "33"):
        examples = [generate(task_id, seed) for seed in range(12)]
        assert all(ex["metadata"]["pattern_family"] == FULL for ex in examples)
        assert all(ex["metadata"]["symmetry_type"] == FULL for ex in examples)
        prompt = get_prompt(task_id, examples[0]["metadata"])
        assert "all 8 square symmetries" in prompt
        assert "90/180/270-degree rotations" in prompt

        for ex in examples:
            grid = np.array(ex["input"], dtype=np.uint8)
            assert grid.shape[0] == grid.shape[1]
            if task_id == "32":
                assert pattern_symmetry._valid_mask_for_meta(
                    grid,
                    int(ex["metadata"]["mask_color"]),
                    ex["metadata"],
                )
            else:
                occ_r, occ_c, occ_h, occ_w = _occluder_box(
                    grid,
                    int(ex["metadata"]["occluder_color"]),
                )
                assert pattern_symmetry._occluder_preserves_inference(
                    ex["metadata"],
                    occ_r,
                    occ_c,
                    occ_h,
                    occ_w,
                    grid.shape[0],
                    grid.shape[1],
                )
                assert len(np.unique(np.array(ex["output"], dtype=np.uint8))) >= 1


def test_full_square_symmetry_sampling_weight_is_low():
    weights = dict(zip(
        pattern_symmetry.PATTERN_FAMILY_CHOICES,
        pattern_symmetry.PATTERN_FAMILY_WEIGHTS,
    ))
    assert weights[FULL] <= 0.05
    assert abs(sum(pattern_symmetry.PATTERN_FAMILY_WEIGHTS) - 1.0) < 1e-9
