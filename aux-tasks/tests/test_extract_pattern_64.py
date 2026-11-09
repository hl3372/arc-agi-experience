"""Extraction correctness, witness loss and competing foreground regressions."""
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pytest

from generators import extract_pattern_64 as subject
from prompts import get_prompt
from registry import generate
from tests.extract_pattern_64_oracle import outputs, solve


@lru_cache(maxsize=1)
def examples():
    return [generate("64", seed) for seed in range(500)]


def test_literal_visible_reader():
    for seed, ex in enumerate(examples()):
        assert solve(ex["input"], get_prompt("64", ex["metadata"])) == ex["output"], seed


def test_arbitrary_letter_colors():
    for seed, ex in enumerate(examples()[:80]):
        palette = np.asarray(list("ABCDEFGHIJ"))[np.random.default_rng(seed+987).permutation(10)]
        prompt = get_prompt("64", ex["metadata"], color_palette=palette.tolist())
        assert solve(palette[np.asarray(ex["input"])].tolist(), prompt) == palette[np.asarray(ex["output"])].tolist()


def test_palette_and_familiar_shapes_scale_with_canvas():
    rows = [ex["metadata"] for ex in examples()]
    assert set(m["object_family"] for m in rows) == set(subject.SHAPES)
    assert set(m["background_family"] for m in rows) == {"periodic", *subject.SYMMETRIES}
    assert set(m["input_color_count"] for m in rows) == set(range(3, 11))
    assert set(len(m["object_colors"]) for m in rows) == {1, 2, 3}
    assert .45 < np.mean([len(m["object_colors"]) == 1 for m in rows]) < .65
    assert max(m["object_cells"] for m in rows) > 200
    assert np.mean([m["object_cells"] == 3 for m in rows]) < .05
    assert np.mean([m["object_occupancy"] for ex, m in zip(examples(), rows)
                    if np.asarray(ex["input"]).size >= 700]) > .10
    assert any(0 in m["object_colors"] for m in rows)
    assert len(set(m["canvas_color"] for m in rows)) == 10
    for m in rows:
        assert m["input_color_count"] == m["planned_color_count"]
        assert not set(m["object_colors"]) & set(m["background_colors"])


def test_sides_and_rejection_do_not_collapse_plans():
    shapes = np.array([np.asarray(ex["input"]).shape for ex in examples()])
    assert set(shapes[:, 0]) == set(range(5, 31)) == set(shapes[:, 1])
    assert abs(np.corrcoef(shapes.T)[0, 1]) < .15
    assert np.mean([ex["metadata"]["attempts"] for ex in examples()]) < 2
    for seed in range(10):
        assert generate("64", seed) == examples()[seed]


def test_hidden_classes_cannot_establish_a_pattern():
    grid = np.tile([[0, 1, 2], [3, 0, 4], [5, 6, 0]], (3, 3))
    removed = grid == 0
    # Six colors remain, but entire phases/orbits have no visible witnesses.
    grid[removed] = 9
    assert subject.pattern_model(grid, removed) is None


def test_reader_searches_more_than_three_foreground_colors():
    grid = np.tile([[0, 1], [1, 0]], (6, 6))
    grid[1:3, 1:5] = [6, 7, 8, 9]
    expected = np.full_like(grid, 5)
    expected[1:3, 1:5] = grid[1:3, 1:5]
    assert np.array_equal(subject.solve(grid, 5), expected)
    assert len(outputs(grid, 5)) == 1


def test_partial_color_extraction_is_detected():
    grid = np.tile(np.array([0, 1, 0])[:, None], (4, 25))
    grid[3, 2:11] = 4
    grid[3, 11:20] = 9
    assert len(outputs(grid, 3)) > 1
    with pytest.raises(ValueError, match="Ambiguous"):
        subject.solve(grid, 3)


@pytest.mark.parametrize("width,start,body", [(25, 11, [4, 9, 9]), (26, 10, [4, 4, 9, 9, 9])])
def test_tiny_competing_foreground_is_not_pruned(width, start, body):
    grid = np.tile(np.array([0, 1, 0])[:, None], (4, width))
    grid[3, start:start+len(body)] = body
    # Removing just color 4 (one/two cells) admits a mirror reading; removing
    # the complete 3+/5-cell body admits a periodic one. Source size cannot
    # exclude the small competing interpretation when the prompt does not.
    assert len(outputs(grid, 3)) > 1
    with pytest.raises(ValueError, match="Ambiguous"):
        subject.solve(grid, 3)


def test_wrong_output_and_tiny_canvases_raise():
    ex = deepcopy(examples()[0])
    ex["output"] = ex["input"]
    with pytest.raises(ValueError, match="disagrees"):
        subject.validate(ex)
    ex["input"] = [[0]*8 for _ in range(4)]
    with pytest.raises(ValueError, match="5..30"):
        subject.validate(ex)


@pytest.mark.parametrize("seed", [21, 73, 74, 295, 354, 393, 515, 590, 725, 767, 828, 895, 995,
                                  3193, 5355, 5561, 14875, 17971])
def test_preserved_plan_capacity_regressions(seed):
    ex = generate("64", seed)
    assert solve(ex["input"], get_prompt("64", ex["metadata"])) == ex["output"]
