"""Center reading, ordinary scene content, and preservation of drawing plans."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from generators import center_cell_75 as module
from prompts import get_prompt
from registry import generate
from symbol_remap import render_color_label


def _read(grid, prompt):
    assert "center cell" in prompt
    rows = [r for r in range(len(grid)) if r == len(grid) - r - 1]
    cols = [c for c in range(len(grid[0])) if c == len(grid[0]) - c - 1]
    assert len(rows) == len(cols) == 1
    return str(grid[rows[0]][cols[0]])


def _isolated(grid):
    pad = np.pad(grid, 1, constant_values=-1)
    mask = np.ones(grid.shape, bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                mask &= grid != pad[1 + dr:1 + dr + len(grid), 1 + dc:1 + dc + len(grid[0])]
    return mask


@pytest.fixture(scope="module")
def examples():
    return [generate("75", seed) for seed in range(1000)]


def test_independent_center_reader(examples):
    for ex in examples:
        assert _read(ex["input"], get_prompt("75", ex["metadata"])) == ex["output"]


def test_odd_sizes_and_color_coverage(examples):
    sizes = np.array([np.shape(ex["input"]) for ex in examples])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(5, 30, 2))
    assert len(set(map(tuple, sizes))) >= 160
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .12
    assert .035 < np.mean(sizes[:, 0] == sizes[:, 1]) < .11
    assert .015 < np.mean(np.prod(sizes, axis=1) >= 700) < .07
    palettes = Counter(ex["metadata"]["n_colors"] for ex in examples)
    assert set(palettes) == set(range(2, 11))
    assert .12 < palettes[2] / len(examples) < .30
    assert .13 < sum(palettes[c] for c in (8, 9, 10)) / len(examples) < .35
    answers = Counter(ex["output"] for ex in examples)
    assert set(answers) == set(map(str, range(10)))
    assert all(.06 < n / len(examples) < .15 for n in answers.values())


def test_family_and_object_plans_are_preserved(examples):
    families = Counter()
    for ex in examples:
        m = ex["metadata"]
        families[m["content_family"]] += 1
        assert m["content_family"] == m["planned_family"]
        assert m["density"] == m["planned_density"]
        assert m["n_shapes"] == m["planned_n_shapes"] == len(m.get("shapes", []))
        assert set(np.unique(ex["input"])) == set(m["palette"])
        assert m["object_cells"] == sum(s["cells"] for s in m.get("shapes", []))
        if m["content_family"] in ("shapes", "multicolor_shapes"):
            assert m["n_shapes"] >= 1
        else:
            assert m["n_shapes"] == 0
    assert set(families) == set(module.FAMILIES)
    assert all(n >= 45 for n in families.values())
    assert .40 < (families["shapes"] + families["multicolor_shapes"]) / len(examples) < .66


def test_shapes_scale_and_rich_palettes_are_drawn(examples):
    shapes = [ex["metadata"] for ex in examples if ex["metadata"]["n_shapes"]]
    kinds = {s["kind"] for m in shapes for s in m["shapes"]}
    assert len(kinds) >= 9
    assert {m["density"] for m in shapes} == {"open", "full"}
    assert max(m["largest_shape_cells"] for m in shapes) > 200
    small = [m for m in shapes if m["grid_h"] * m["grid_w"] < 150]
    large = [m for m in shapes if m["grid_h"] * m["grid_w"] >= 700]
    occupancy = lambda ms: np.mean([m["object_cells"] / (m["grid_h"] * m["grid_w"]) for m in ms])
    assert len(large) >= 8
    assert occupancy(large) > .22
    assert occupancy(large) > .60 * occupancy(small)
    assert np.mean([m["largest_shape_cells"] for m in large]) > 3 * np.mean([m["largest_shape_cells"] for m in small])
    rich = [ex["metadata"] for ex in examples if ex["metadata"]["n_colors"] >= 8]
    assert len({m["content_family"] for m in rich}) >= 5
    assert all(m["background_style"] != "noisy" for m in rich)


def test_no_singleton_center_cue(examples):
    isolated_centers = guesses = applicable = 0
    roles = Counter()
    for ex in examples:
        grid = np.array(ex["input"])
        isolated = _isolated(grid)
        isolated_centers += bool(isolated[grid.shape[0] // 2, grid.shape[1] // 2])
        colors = set(grid[isolated])
        applicable += len(colors) == 1
        guesses += len(colors) == 1 and int(ex["output"]) in colors
        roles[ex["metadata"]["center_role"]] += 1
    assert isolated_centers / len(examples) < .025
    assert guesses / max(1, applicable) < .40
    assert guesses / len(examples) < .05
    assert all(roles[role] > 150 for role in ("field", "object", "pattern"))


def test_scene_is_not_changed_after_drawing(monkeypatch):
    original = module._draw_scene
    drawn = []

    def capture(rng, plan):
        result = original(rng, plan)
        drawn.append(result[0].copy())
        return result

    monkeypatch.setattr(module, "_draw_scene", capture)
    for seed in range(60):
        ex = generate("75", seed)
        assert np.array_equal(ex["input"], drawn[-1])


def test_retry_preserves_every_plan_choice(monkeypatch):
    original = module._draw
    plans = []

    def miss_first(rng, plan):
        plans.append(deepcopy(plan))
        return None if len(plans) == 1 else original(rng, plan)

    monkeypatch.setattr(module, "_draw", miss_first)
    ex = generate("75", 18)
    assert ex["metadata"]["attempts"] >= 2
    assert all(plan == plans[0] for plan in plans)


def test_no_canned_fallback(monkeypatch):
    monkeypatch.setattr(module, "_draw", lambda rng, plan: None)
    with pytest.raises(RuntimeError, match="fixed scene plan"):
        generate("75", 18)


@pytest.mark.parametrize("fault", ("answer", "center", "even_side", "color", "palette", "count"))
def test_validator_rejects_tampered_examples(fault):
    ex = deepcopy(generate("75", 18))
    if fault == "answer":
        ex["output"] = str((int(ex["output"]) + 1) % 10)
    elif fault == "center":
        a = ex["input"]
        a[len(a) // 2][len(a[0]) // 2] = (int(ex["output"]) + 1) % 10
    elif fault == "even_side":
        ex["input"] = ex["input"][:-1]
    elif fault == "color":
        ex["input"][0][0] = 10
    elif fault == "palette":
        ex["metadata"]["n_colors"] += 1
    else:
        ex["metadata"]["n_shapes"] += 1
    with pytest.raises(ValueError):
        module.validate_center_color(ex)


def test_arbitrary_symbols_and_repeatability(examples):
    for seed, ex in enumerate(examples[:100]):
        palette = np.random.default_rng(seed + 6189).permutation(list("QWERTYUIOP")).tolist()
        grid = [[palette[c] for c in row] for row in ex["input"]]
        prompt = get_prompt("75", ex["metadata"], color_palette=palette)
        assert "single color symbol" in prompt
        assert _read(grid, prompt) == render_color_label(ex["output"], palette)
        assert ex == generate("75", seed)
