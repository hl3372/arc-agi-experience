"""Endpoint-run correctness, readability, and coverage regressions."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pytest

from prompts import get_prompt
from registry import generate
from generators import path_routes
from tests.path_runs_61_63_oracles import solve


TASKS = ("61", "62", "63")


@lru_cache(maxsize=None)
def examples(tid):
    return [generate(tid, seed) for seed in range(400)]


@pytest.mark.parametrize("tid", TASKS)
def test_independent_grid_and_prompt_solver(tid):
    for seed, ex in enumerate(examples(tid)):
        answer, stats = solve(ex["input"], get_prompt(tid, ex["metadata"]), tid)
        assert answer == ex["output"], (tid, seed)
        assert stats["path_length"] == ex["metadata"]["path_length"]
        assert stats["turns"] == ex["metadata"]["path_turns"]


@pytest.mark.parametrize("tid", TASKS)
def test_arbitrary_symbol_palette(tid):
    for seed, ex in enumerate(examples(tid)[:100]):
        palette = np.asarray(list("ABCDEFGHIJ"))[np.random.default_rng(seed+7100).permutation(10)]
        grid = palette[np.asarray(ex["input"])].tolist()
        expected = ex["output"] if tid == "62" else palette[np.asarray(ex["output"])].tolist()
        prompt = get_prompt(tid, ex["metadata"], color_palette=palette.tolist())
        assert solve(grid, prompt, tid)[0] == expected, (tid, seed)


@pytest.mark.parametrize("tid", TASKS)
def test_readable_canvas_palette_and_determinism(tid):
    xs = examples(tid)
    sizes = np.array([np.asarray(ex["input"]).shape for ex in xs])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(5, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .20
    assert np.mean(sizes[:, 0] == sizes[:, 1]) < .08
    assert .02 < np.mean(sizes.prod(axis=1) >= 700) < .11
    colors = Counter(ex["metadata"]["n_colors"] for ex in xs)
    assert set(colors) == set(range(2 if tid == "61" else 3, 11 if tid == "62" else 10))
    assert .08 < np.mean([ex["metadata"]["n_colors"] >= 8 for ex in xs]) < .40
    for seed, ex in enumerate(xs[:30]):
        assert ex == generate(tid, seed)
        assert ex["metadata"]["grid_h"] == len(ex["input"])
        assert ex["metadata"]["grid_w"] == len(ex["input"][0])
        if tid != "62":
            assert ex["metadata"]["highlight_color"] not in np.asarray(ex["input"])


@pytest.mark.parametrize("tid", TASKS)
def test_route_run_and_background_variety(tid):
    xs = examples(tid)
    ms = [ex["metadata"] for ex in xs]
    families = Counter(m["path_family"] for m in ms)
    assert set(families) == set(path_routes.FAMILIES)
    assert all(count/len(ms) > .04 for count in families.values())
    # The templated shapes stay the majority; irregular routes are the rest.
    irregular = sum(v for k, v in families.items() if k in path_routes.IRREGULAR_FAMILIES)
    assert .35 < irregular/len(ms) < .45
    assert all(count/len(ms) > .05 for family, count in families.items()
               if family in path_routes.TEMPLATE_FAMILIES)
    assert min(m["n"] for m in ms) == 1
    assert max(m["n"] for m in ms) >= (80 if tid == "61" else 120)
    assert max(m["path_length"] for m in ms) > 250
    assert .45 < np.mean([m["n"] <= 6 for m in ms]) < .80
    assert np.mean([m["n"] >= 12 for m in ms]) > .15
    assert all(m["n"] <= m["answer_cap"] for m in ms)
    assert np.mean([m["attempts"] for m in ms]) < 3
    styles = Counter(m["background_style"] for m in ms)
    assert .40 < styles["solid"]/len(ms) < .75
    assert .08 < styles["lattice"]/len(ms) < .30
    assert .12 < sum(v for k, v in styles.items() if k not in ("solid", "lattice"))/len(ms) < .35
    assert all(styles[k] > 0 for k in ("stripes", "diagonal", "checker", "split"))
    for m in ms:
        assert m["path_family"] != "serpentine" or m["path_turns"] >= 4
        assert m["path_family"] != "staircase" or m["path_turns"] >= 2
    big = [m["path_occupancy"] for ex, m in zip(xs, ms) if np.asarray(ex["input"]).size >= 700]
    assert np.mean(big) > .11


@pytest.mark.parametrize("tid", TASKS)
def test_traversal_matters_beyond_local_distance(tid):
    results = [solve(ex["input"], get_prompt(tid, ex["metadata"]), tid)[1]
               for ex in examples(tid)]
    assert np.mean([s["manhattan_shortcut_correct"] for s in results]) < .85


@pytest.mark.parametrize("tid", TASKS)
def test_n_does_not_change_the_display(monkeypatch, tid):
    from generators import path_runs_61_63 as g
    for seed in range(30):
        monkeypatch.setattr(g, "choose_n", lambda rng, length, task_id: 1)
        short = generate(tid, seed)
        monkeypatch.setattr(g, "choose_n", lambda rng, length, task_id:
                            length//2 if task_id == "61" else length-1)
        long = generate(tid, seed)
        assert short["input"] == long["input"]
        assert solve(long["input"], get_prompt(tid, long["metadata"]), tid)[0] == long["output"]


@pytest.mark.parametrize("tid", TASKS)
def test_prompt_rules(tid):
    for ex in examples(tid):
        m = ex["metadata"]
        prompt = get_prompt(tid, m)
        assert "connected horizontally or vertically" in prompt
        assert "background is" not in prompt and f"background color {m['bg_color']}" not in prompt
        if tid != "62":
            assert prompt.endswith("Leave every other cell unchanged.")
        if tid != "61":
            assert f"single start cell of color {m['start_color']}" in prompt
            assert "The start cell is not included." in prompt
            assert np.count_nonzero(np.asarray(ex["input"]) == m["start_color"]) == 1
        if m["n"] == 1:
            assert "first 1 cell " in prompt or "last 1 cell " in prompt
            assert "first 1 cells" not in prompt and "last 1 cells" not in prompt


@pytest.mark.parametrize("tid", ["62", "63"])
def test_named_marker_resolves_competing_endpoint_decorations(tid):
    grid = np.zeros((5, 7), int)
    grid[2, 1:6] = 1
    grid[2, 0], grid[2, 6] = 2, 3
    m = {"path_color": 1, "start_color": 2, "highlight_color": 4, "n": 2}
    answer, stats = solve(grid, get_prompt(tid, m), tid)
    assert set(stats["chosen"]) == {(2, 4), (2, 5)}
    other_answer, other_stats = solve(grid, get_prompt(tid, {**m, "start_color": 3}), tid)
    assert set(other_stats["chosen"]) == {(2, 1), (2, 2)}
    assert answer != other_answer


@pytest.mark.parametrize("seed", [4684, 6495])
def test_former_ambiguous_marker_seeds(seed):
    ex = generate("63", seed)
    assert solve(ex["input"], get_prompt("63", ex["metadata"]), "63")[0] == ex["output"]


@pytest.mark.parametrize("tid", TASKS)
def test_validator_rejects_corrupt_output_and_canvas(tid):
    from generators.path_runs_61_63 import validate_example
    ex = deepcopy(generate(tid, 0))
    ex["output"] = "wrong"
    with pytest.raises(ValueError, match="output disagrees"):
        validate_example(ex, tid)
    ex = deepcopy(generate(tid, 0))
    ex["input"] = ex["input"][:4]
    with pytest.raises(ValueError, match="between five and thirty"):
        validate_example(ex, tid)


@pytest.mark.parametrize("tid", ["62", "63"])
def test_validator_rejects_duplicate_or_interior_marker(tid):
    from generators.path_runs_61_63 import validate_example
    ex = deepcopy(generate(tid, 0))
    a, m = np.asarray(ex["input"]), ex["metadata"]
    point = tuple(np.argwhere(a == m["bg_color"])[0])
    a[point] = m["start_color"]
    ex["input"] = a.tolist()
    with pytest.raises(ValueError, match="exactly once"):
        validate_example(ex, tid)
    ex = deepcopy(generate(tid, 0))
    order = solve(ex["input"], get_prompt(tid, ex["metadata"]), tid)[1]["path"]
    a = np.asarray(ex["input"])
    a[order[0]] = m["path_color"]
    a[order[len(order)//2]] = m["start_color"]
    ex["input"] = a.tolist()
    with pytest.raises(ValueError, match="endpoint"):
        validate_example(ex, tid)


def test_validator_rejects_unconnected_cycle_of_path_cells():
    from generators.path_runs_61_63 import trace_visible
    grid = np.zeros((8, 8), int)
    grid[1, 1:5] = 1
    grid[4:6, 4:6] = 1
    with pytest.raises(ValueError, match="disconnected"):
        trace_visible(grid, 1)


def test_failed_decoration_keeps_scene_plan_and_raises(monkeypatch):
    from generators import path_runs_61_63 as g
    plans = []

    def cannot_decorate(rng, grid, protected, bg, colors, style, **kwargs):
        plans.append((grid.shape, bg, tuple(colors), style))
        raise ValueError("Deliberately unrealizable decoration")

    monkeypatch.setattr(g, "decorate", cannot_decorate)
    monkeypatch.setattr(g, "MAX_ATTEMPTS", 4)
    with pytest.raises(RuntimeError, match="fixed"):
        generate("63", 0)
    assert len(plans) == 4
    assert all(plan == plans[0] for plan in plans)
