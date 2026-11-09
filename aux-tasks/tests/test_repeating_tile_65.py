"""Input-only periodic-word oracle for task 65, including clipped edge copies."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from generators import repeating_tile_65 as G
from prompts import get_prompt
from registry import generate


def _word_period(values):
    # KMP longest-prefix/suffix lengths on whole row/column words.  This does not
    # use the production solver's shifted-array comparisons or planted periods.
    borders = [0] * len(values)
    for i in range(1, len(values)):
        j = borders[i - 1]
        while j and values[i] != values[j]:
            j = borders[j - 1]
        if values[i] == values[j]:
            j += 1
        borders[i] = j
    return len(values) - borders[-1]


def oracle(grid, prompt):
    assert "smallest rectangular tile" in prompt
    assert "unchanged copies" in prompt
    assert "top-left corner" in prompt
    assert "cut off at the bottom or right edge" in prompt
    rows = [tuple(row) for row in grid]
    cols = list(zip(*rows))
    h, w = _word_period(rows), _word_period(cols)
    return [list(row[:w]) for row in rows[:h]]


@pytest.fixture(scope="module")
def examples():
    return [generate("65", seed) for seed in range(1000)]


def test_visible_periods_and_symbol_remaps(examples):
    for seed, ex in enumerate(examples):
        prompt = get_prompt("65", ex["metadata"])
        assert oracle(ex["input"], prompt) == ex["output"]
        if seed < 150:
            palette = np.random.default_rng(seed + 658000).permutation(list("ABCDEFGHIJ")).tolist()
            inp = np.asarray(palette)[ex["input"]].tolist()
            out = np.asarray(palette)[ex["output"]].tolist()
            assert oracle(inp, get_prompt("65", ex["metadata"], color_palette=palette)) == out


def test_size_repetition_palette_and_family_coverage(examples):
    ms = [ex["metadata"] for ex in examples]
    # MIN_SIDE = 5 floor, 2026-09-15 (user: small sides are often ambiguous).
    for axis in ("grid_h", "grid_w"):
        assert set(m[axis] for m in ms) == set(range(5, 31))
    # The floor shrinks the pair ceiling from 891 to 26*26 = 676, so 1000 seeds
    # can no longer reach 550 distinct pairs; ~503 is the coupon-collector value.
    assert len({(m["grid_h"], m["grid_w"]) for m in ms}) > 460
    assert abs(np.corrcoef([[m["grid_h"] for m in ms], [m["grid_w"] for m in ms]])[0, 1]) < .12
    # 5 replaces 1 as the smallest side; it lands at the uniform-draw rate (~.075).
    assert .05 < np.mean([min(m["grid_h"], m["grid_w"]) == 5 for m in ms]) < .10
    assert .025 < np.mean([m["grid_h"] * m["grid_w"] >= 700 for m in ms]) < .08
    assert max(m["tile_area"] for m in ms) >= 150
    for axis in ("tile_height", "tile_width"):
        assert set(m[axis] for m in ms) == set(range(1, 16))
    assert len({(m["tile_height"], m["tile_width"]) for m in ms}) > 145
    assert set(m["n_colors"] for m in ms) == set(range(2, 11))
    assert .05 < np.mean([m["n_colors"] >= 8 for m in ms]) < .20
    family = Counter(m["family"] for m in ms)
    assert set(family) == set(G.FAMILIES)
    assert all(count > 50 for count in family.values())
    # Bands used to absorb the 1..4-wide canvases, where no other family fits;
    # with the floor at 5 every family is eligible far more often and bands
    # settle near their .10 weight instead of dominating the small sizes.
    assert .10 < family["bands"] / len(ms) < .42
    assert set(kind for m in ms for kind in m["motif_kinds"]) == set(G.MOTIFS)
    assert .70 < np.mean([m["has_partial_copies"] for m in ms]) < .93
    for ex, m in zip(examples, ms):
        assert m["n_colors"] == len(set(sum(ex["input"], [])))
        # No canvas is 1 cell wide any more, so both axes must really repeat.
        assert m["repeats_h"] >= 2
        assert m["repeats_w"] >= 2
        assert (m["tile_height"], m["tile_width"]) != (1, 1)
        assert m["attempts"] <= G.MAX_ATTEMPTS
    assert np.mean([m["attempts"] for m in ms]) < 3


def test_interiors_carry_period_without_losing_simple_patterns(examples):
    correct = []
    for ex in examples:
        a = ex["input"]
        guess = (_word_period([row[0] for row in a]), _word_period(a[0]))
        actual = (len(ex["output"]), len(ex["output"][0]))
        correct.append(guess == actual)
        assert bool(guess == actual) == ex["metadata"]["first_row_column_sufficient"]
    # Ordinary motifs supply the harder cases, without replacing simple stripes
    # and checks with an artificial equal-sized adversarial mode.
    assert .45 < np.mean(correct) < .80


def test_cropped_periods_and_internal_partial_periods_are_distinct():
    # The planted three-cell word has a two-cell prefix period on its own;
    # the complete repeated input proves that two cells are insufficient.
    ex = {"input": [[1, 2, 1, 1, 2, 1, 1, 2]], "output": [[1, 2, 1]]}
    G.validate_example(ex)
    assert G.smallest_tile(ex["input"]).tolist() == ex["output"]
    assert oracle(ex["input"], get_prompt("65", {})) == ex["output"]
    assert G.smallest_tile(ex["output"]).tolist() == [[1, 2]]
    tile = np.asarray([[1, 1, 2, 1], [1, 3, 3, 1], [1, 1, 1, 1]])
    a = np.tile(tile, (3, 3))[:7, :10]
    G.validate_example({"input": a.tolist(), "output": tile.tolist()})
    assert oracle(a.tolist(), get_prompt("65", {})) == tile.tolist()


def test_largest_feasible_tile_is_reachable():
    ex = generate("65", 116631)
    assert np.asarray(ex["input"]).shape == (30, 30)
    assert np.asarray(ex["output"]).shape == (15, 15)
    assert oracle(ex["input"], get_prompt("65", ex["metadata"])) == ex["output"]


@pytest.mark.parametrize("bad", [
    {"input": [[1]], "output": [[1]]},
    {"input": [[1, 2, 1]], "output": [[1, 2]]},
    {"input": [[1, 2], [1, 2]], "output": [[1, 2]]},
    {"input": [[1, 2, 1, 2, 1]], "output": [[2, 1]]},
    {"input": [[1, 2, 1, 2, 1]], "output": [[1, 2, 1, 2]]},
])
def test_validator_rejects_trivial_insufficient_and_wrong_phase_examples(bad):
    with pytest.raises(ValueError):
        G.validate_example(bad)


def test_validator_rejects_corrupted_input_output_and_statistics():
    ex = generate("65", 4)
    for where in ("input", "output"):
        bad = deepcopy(ex)
        bad[where][0][0] = (bad[where][0][0] + 1) % 10
        with pytest.raises(ValueError):
            G.validate_example(bad)
    for key in ("tile_height", "tile_width", "n_colors", "repeats_h", "partial_cols"):
        bad = deepcopy(ex)
        bad["metadata"][key] += 1
        with pytest.raises(ValueError, match="metadata"):
            G.validate_example(bad)


def test_failed_plan_raises_instead_of_shrinking_or_replacing(monkeypatch):
    plan = G._plan(np.random.default_rng(4))
    calls = []
    monkeypatch.setattr(G, "_plan", lambda rng: plan)

    def invalid_tile(rng, same_plan):
        assert same_plan is plan
        calls.append(same_plan)
        return np.full(plan["tile_shape"], plan["palette"][0])

    monkeypatch.setattr(G, "_draw", invalid_tile)
    with pytest.raises(ValueError, match="fixed tile plan"):
        G.generate_example(np.random.default_rng(4))
    assert len(calls) == G.MAX_ATTEMPTS
