"""Independent visible-input readers and plan/coverage contracts for 76/80."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from common.grid import COLOR_INDEX
from generators import grid_reading_76_80 as impl
from prompts import get_prompt
from registry import generate
from sft_wrapper import render_output


def histogram_reader(grid, prompt):
    assert "each present color has a different count" in prompt
    populations = Counter(value for row in grid for value in row)
    assert len(populations) == len(set(populations.values()))
    return sorted(populations.items(), key=lambda item: item[1])


def parse_histogram(answer, symbols=False):
    result = []
    for item in answer.split(","):
        label, value = item.strip().split(":")
        result.append((label if symbols else COLOR_INDEX[label], int(value)))
    return result


def half_reader(grid, prompt):
    side = re.search(r"Return the (top|bottom|left|right) half", prompt).group(1)
    h, w = len(grid), len(grid[0])
    if side in ("top", "bottom"):
        assert h % 2 == 0
        selected = range(h // 2) if side == "top" else range(h // 2, h)
        return [list(grid[r]) for r in selected]
    assert w % 2 == 0
    selected = range(w // 2) if side == "left" else range(w // 2, w)
    return [[row[c] for c in selected] for row in grid]


@pytest.mark.parametrize("task", ["76", "80"])
def test_visible_input_and_prompt_reader(task):
    for seed in range(400):
        example = generate(task, seed)
        prompt = get_prompt(task, example["metadata"])
        if task == "76":
            assert histogram_reader(example["input"], prompt) == parse_histogram(example["output"])
        else:
            assert half_reader(example["input"], prompt) == example["output"]


@pytest.mark.parametrize("task", ["76", "80"])
def test_arbitrary_symbols(task):
    for seed in range(40):
        palette = list("αβγδεζηθικ")
        np.random.default_rng(seed).shuffle(palette)
        example = generate(task, seed)
        inp = [[palette[v] for v in row] for row in example["input"]]
        prompt = get_prompt(task, example["metadata"], color_palette=palette)
        if task == "76":
            expected = parse_histogram(render_output(example["output"], task_id=task, color_palette=palette), True)
            assert histogram_reader(inp, prompt) == expected
        else:
            assert half_reader(inp, prompt) == [[palette[v] for v in row] for row in example["output"]]


@pytest.mark.parametrize("task", ["76", "80"])
def test_drawings_have_broad_meaningful_coverage(task):
    examples = [generate(task, seed) for seed in range(1200)]
    metas = [ex["metadata"] for ex in examples]
    assert {m["grid_h"] for m in metas} == set(range(5, 31))
    assert {m["grid_w"] for m in metas} == set(range(5, 31))
    assert {m["content_family"] for m in metas} == set(impl.FAMILIES)
    assert {m["n_colors"] for m in metas} == set(range(2, 11))
    assert np.mean([m["n_colors"] >= 8 for m in metas]) > .15
    assert np.mean([m["grid_h"] * m["grid_w"] >= 700 for m in metas]) > .025
    assert max(m["largest_shape_cells"] for m in metas) > 200
    big_shapes = [m for m in metas if m["n_shapes"] and m["grid_h"] * m["grid_w"] >= 700]
    assert np.mean([m["object_cells"] / (m["grid_h"] * m["grid_w"]) for m in big_shapes]) > .22
    kinds = {shape["kind"] for m in metas for shape in m.get("shapes", [])}
    assert len(kinds) >= 9
    assert all(m["content_family"] == m["planned_family"] and m["n_shapes"] == m["planned_n_shapes"] for m in metas)
    assert all(m["background_style"] != "noisy" for m in metas)
    if task == "76":
        assert max(max(m["color_counts"]) for m in metas) > 600
        assert np.mean([m["attempts"] for m in metas]) < 4
    else:
        assert np.mean([m["grid_h"] % 2 or m["grid_w"] % 2 for m in metas]) > .5
        assert .25 < np.mean([m["both_axes_even"] for m in metas]) < .4
        counts = Counter(m["which_half"] for m in metas)
        assert all(.20 < counts[side] / len(metas) < .30 for side in ("top", "bottom", "left", "right"))


def test_changing_only_requested_half_reuses_the_input(monkeypatch):
    for seed in range(30):
        initial = generate("80", seed)
        for side in initial["metadata"]["feasible_halves"]:
            with monkeypatch.context() as patch:
                patch.setattr(impl, "choose_half", lambda rng, grid, side=side: side)
                other = generate("80", seed)
            assert other["input"] == initial["input"]
            assert other["output"] == half_reader(other["input"], get_prompt("80", other["metadata"]))


def test_retry_keeps_the_complete_drawing_plan(monkeypatch):
    for histogram in (True, False):
        rng = np.random.default_rng(0)
        plan = impl._plan(rng, histogram=histogram)
        before = deepcopy(plan)
        original = impl._draw
        calls = []

        def first_failure(rng, actual_plan):
            calls.append(deepcopy(actual_plan))
            return None if len(calls) == 1 else original(rng, actual_plan)

        with monkeypatch.context() as patch:
            patch.setattr(impl, "_draw", first_failure)
            _, _, attempts = impl._draw_planned(rng, plan)
        assert attempts >= 2 and len(calls) >= 2
        assert all(value == before for value in calls) and plan == before


def test_validators_reject_visible_answer_and_statistics_tampering():
    histogram = generate("76", 0)
    bad = deepcopy(histogram)
    bad["output"] = bad["output"].rsplit(":", 1)[0] + ":999"
    with pytest.raises(ValueError):
        impl.validate_histogram(bad)
    half = generate("80", 0)
    bad = deepcopy(half)
    bad["output"][0][0] = (bad["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        impl.validate_half(bad)
    for example, validator in ((histogram, impl.validate_histogram), (half, impl.validate_half)):
        bad = deepcopy(example)
        bad["metadata"]["color_counts"][0] += 1
        with pytest.raises(ValueError):
            validator(bad)
        bad = deepcopy(example)
        bad["metadata"]["object_cells"] += 1
        with pytest.raises(ValueError):
            validator(bad)


def test_histogram_small_capacity_regressions():
    # Initial capacity accounting wrongly admitted zero feasible palette sizes
    # for these 5x5 patchwork plans; two unequal populations fit comfortably.
    for seed in (871, 5311, 5561):
        ex = generate("76", seed)
        assert ex["metadata"]["grid_h"] == ex["metadata"]["grid_w"] == 5
        impl.validate_histogram(ex)


def test_odd_cut_axis_is_rejected():
    ex = next(generate("80", seed) for seed in range(100)
              if generate("80", seed)["metadata"]["grid_h"] % 2)
    ex["metadata"]["which_half"] = "top"
    with pytest.raises(ValueError, match="split evenly"):
        impl.validate_half(ex)
