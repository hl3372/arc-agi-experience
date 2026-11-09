"""Prompt-first oracles and drawing/rule coverage for tasks 66 and 70."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import re

import numpy as np
import pytest

from prompts import get_prompt
from registry import generate


def solve(grid, prompt, tid):
    """Use only visible cells and the rendered instruction, including letters."""
    h, w = len(grid), len(grid[0])
    if tid == "66":
        if "anti-diagonal" in prompt:
            return [[grid[h - 1 - c][w - 1 - r] for c in range(h)] for r in range(w)]
        assert "main diagonal" in prompt
        return [[grid[c][r] for c in range(h)] for r in range(w)]
    found = re.search(r"Swap every color (\S+) cell with color (\S+) and vice versa", prompt)
    assert found is not None
    a, b = found.groups()
    if not isinstance(grid[0][0], str):
        a, b = int(a), int(b)
    return [[b if cell == a else a if cell == b else cell for cell in row] for row in grid]


@lru_cache(maxsize=None)
def examples(tid):
    return [generate(tid, seed) for seed in range(800)]


@pytest.mark.parametrize("tid", ("66", "70"))
def test_independent_visible_grid_and_prompt_oracle(tid):
    for seed, ex in enumerate(examples(tid)):
        assert solve(ex["input"], get_prompt(tid, ex["metadata"]), tid) == ex["output"], seed
        assert solve(ex["output"], get_prompt(tid, ex["metadata"]), tid) == ex["input"], seed
        assert ex["input"] != ex["output"]


@pytest.mark.parametrize("tid", ("66", "70"))
def test_arbitrary_symbols(tid):
    for seed, ex in enumerate(examples(tid)[:300]):
        palette = np.asarray(list("ABCDEFGHIJ"))[np.random.default_rng(seed + 8601).permutation(10)]
        inp = palette[np.asarray(ex["input"])].tolist()
        out = palette[np.asarray(ex["output"])].tolist()
        prompt = get_prompt(tid, ex["metadata"], color_palette=palette.tolist())
        assert solve(inp, prompt, tid) == out, seed


@pytest.mark.parametrize("tid", ("66", "70"))
def test_sizes_palettes_and_determinism(tid):
    xs = examples(tid)
    sides = np.asarray([np.asarray(ex["input"]).shape for ex in xs])
    assert set(sides[:, 0]) == set(sides[:, 1]) == set(range(5, 31))
    assert abs(np.corrcoef(sides.T)[0, 1]) < .15
    assert .02 < np.mean(sides[:, 0] == sides[:, 1]) < .075
    assert .02 < np.mean(sides.prod(axis=1) >= 700) < .11
    counts = Counter(ex["metadata"]["n_colors"] for ex in xs)
    assert set(counts) == set(range(2, 11))
    assert .10 < counts[2] / len(xs) < .35
    assert .12 < sum(counts[n] for n in (8, 9, 10)) / len(xs) < .35
    for seed in range(20):
        assert generate(tid, seed) == xs[seed]


@pytest.mark.parametrize("tid", ("66", "70"))
def test_drawn_families_scale_and_keep_their_plans(tid):
    from generators.grid_content_66_70 import FAMILIES
    ms = [ex["metadata"] for ex in examples(tid)]
    families = Counter(m["content_family"] for m in ms)
    assert set(families) == set(FAMILIES)
    assert all(.08 < count / len(ms) < .31 for count in families.values())
    assert all(m["content_family"] == m["planned_family"] for m in ms)
    assert np.mean([m["attempts"] for m in ms]) < 2
    shapes = [m for m in ms if "shapes" in m]
    vocabulary = {s["kind"] for m in shapes for s in m["shapes"]}
    assert vocabulary == {"rectangle", "L", "T", "U", "H", "plus", "frame", "staircase", "chamfer", "diamond"}
    assert max(m["largest_shape_cells"] for m in shapes) > 150
    assert all(len(s["colors"]) <= 3 for m in shapes for s in m["shapes"])
    assert any(len(s["colors"]) == 3 for m in shapes for s in m["shapes"])
    assert all(m["multicolor_shape_count"] >= 1 for m in shapes
               if m["content_family"] == "multicolor_shapes" and m["n_colors"] >= 3)
    open_mean = np.mean([m["foreground_occupancy"] for m in shapes if m["planned_density"] == "open"])
    full_mean = np.mean([m["foreground_occupancy"] for m in shapes if m["planned_density"] == "full"])
    assert .10 < open_mean < .45
    assert full_mean > open_mean + .12
    big = [m for m in shapes if m["grid_h"] * m["grid_w"] >= 700]
    small = [m for m in shapes if m["grid_h"] * m["grid_w"] < 150]
    assert np.mean([m["foreground_occupancy"] for m in big]) > .22
    assert np.mean([m["foreground_occupancy"] for m in big]) > .65 * np.mean([m["foreground_occupancy"] for m in small])
    assert np.mean([m["largest_shape_cells"] for m in big]) > 45


def test_diagonal_balance_and_counterfactual():
    xs = examples("66")
    assert .42 < np.mean([ex["metadata"]["diagonal"] == "main" for ex in xs]) < .58
    assert np.mean([ex["metadata"]["diagonal_matters"] for ex in xs]) > .85
    for ex in xs:
        a = np.asarray(ex["input"])
        assert not np.array_equal(a, a.T)
        assert not np.array_equal(a, a[::-1, ::-1].T)


def test_diagonal_rule_cannot_change_the_scene(monkeypatch):
    from generators import grid_content_66_70 as g
    for seed in range(50):
        monkeypatch.setattr(g, "choose_diagonal", lambda rng: "main")
        main = generate("66", seed)
        monkeypatch.setattr(g, "choose_diagonal", lambda rng: "anti")
        anti = generate("66", seed)
        assert main["input"] == anti["input"]
        assert main["metadata"]["attempts"] == anti["metadata"]["attempts"]


def test_swap_roles_counts_and_color_identity():
    ms = [ex["metadata"] for ex in examples("70")]
    expected_bg = np.mean([2 / m["n_colors"] for m in ms])
    actual_bg = np.mean([m["swap_includes_background"] for m in ms])
    assert abs(actual_bg - expected_bg) < .065
    assert .35 < actual_bg < .70
    assert np.mean([m["count_ratio"] >= .6 for m in ms]) > .15
    assert np.mean([m["count_ratio"] <= .2 for m in ms]) > .15
    assert max(m["changed_cells"] for m in ms) >= 700
    assert np.mean([m["changed_fraction"] == 1 for m in ms]) > .10
    assert np.mean([m["changed_fraction"] < .05 for m in ms]) < .15
    assert np.mean([m["changed_cells"] <= 10 for m in ms]) < .15
    assert all(m["query_selection"] == "uniform_present_pair" for m in ms)
    hits = Counter(c for m in ms for c in (m["color_a"], m["color_b"]))
    assert set(hits) == set(range(10))
    assert all(.14 < hits[c] / len(ms) < .26 for c in hits)


def test_swap_pair_cannot_change_the_scene(monkeypatch):
    from generators import grid_content_66_70 as g
    for seed in range(50):
        monkeypatch.setattr(g, "choose_swap_pair", lambda rng, present: tuple(present[:2]))
        first = generate("70", seed)
        monkeypatch.setattr(g, "choose_swap_pair", lambda rng, present: tuple(present[-2:]))
        last = generate("70", seed)
        assert first["input"] == last["input"]


@pytest.mark.parametrize("tid", ("66", "70"))
def test_retry_keeps_size_palette_family_and_density(monkeypatch, tid):
    from generators import grid_content_66_70 as g
    original, seen = g._draw, []
    def fail_once(rng, plan):
        seen.append(deepcopy(plan))
        return None if len(seen) == 1 else original(rng, plan)
    monkeypatch.setattr(g, "_draw", fail_once)
    ex = generate(tid, 10)
    assert len(seen) >= 2 and all(plan == seen[0] for plan in seen)
    assert ex["metadata"]["attempts"] >= 2


def test_exhausted_fixed_plan_raises_instead_of_returning_canned_content(monkeypatch):
    from generators import grid_content_66_70 as g
    monkeypatch.setattr(g, "_draw", lambda rng, plan: None)
    with pytest.raises(RuntimeError, match="fixed 66/70 scene plan"):
        generate("66", 3)


@pytest.mark.parametrize("tid", ("66", "70"))
def test_full_shapes_cannot_erase_the_planned_base_color(tid):
    ex = generate(tid, 1500)
    assert set(np.unique(ex["input"])) == set(ex["metadata"]["palette"])
    assert ex["metadata"]["attempts"] > 1


@pytest.mark.parametrize("tid", ("66", "70"))
def test_validator_raises_on_wrong_output_size_palette_and_statistics(tid):
    from generators.grid_content_66_70 import validate_diagonal, validate_swap
    validate = validate_diagonal if tid == "66" else validate_swap
    for change in ("output", "size", "palette", "statistics", "family"):
        ex = deepcopy(examples(tid)[0])
        if change == "output":
            ex["output"][0][0] = (ex["output"][0][0] + 1) % 10
        elif change == "size":
            ex["input"] = ex["input"][:4]
        elif change == "palette":
            ex["metadata"]["palette"] = ex["metadata"]["palette"][:-1]
        elif change == "statistics":
            ex["metadata"]["color_counts"][0] += 1
        else:
            ex["metadata"]["planned_family"] = "different"
        with pytest.raises(ValueError):
            validate(ex)


@pytest.mark.parametrize("tid", ("66", "70"))
def test_prompt_has_only_the_operation_givens(tid):
    for ex in examples(tid):
        prompt = get_prompt(tid, ex["metadata"])
        assert "background" not in prompt
        assert "pattern" not in prompt and "family" not in prompt
        if tid == "66":
            assert "height-by-width grid becomes width-by-height" in prompt
