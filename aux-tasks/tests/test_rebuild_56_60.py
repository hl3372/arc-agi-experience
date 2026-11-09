"""Prompt-only regression checks for contact, enclosure, search and path tasks."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pytest

from prompts import get_prompt
from registry import generate
from tests.rebuild_56_60_oracles import SOLVERS, components, generic59

TASKS = ("56", "57", "58", "59", "60")


@lru_cache(maxsize=None)
def sweep(tid):
    return [generate(tid, seed) for seed in range(400)]


@pytest.mark.parametrize("tid", TASKS)
def test_prompt_only_oracle(tid):
    for seed, ex in enumerate(sweep(tid)):
        actual, _ = SOLVERS[tid](ex["input"], get_prompt(tid, ex["metadata"]))
        assert actual == ex["output"], (tid, seed)


@pytest.mark.parametrize("tid", TASKS)
def test_permuted_letter_symbols(tid):
    for seed, ex in enumerate(sweep(tid)[:80]):
        palette = np.asarray(list("ABCDEFGHIJ"))[np.random.default_rng(seed+777).permutation(10)]
        grid = palette[np.asarray(ex["input"])].tolist()
        prompt = get_prompt(tid, ex["metadata"], color_palette=palette.tolist())
        expected = (palette[np.asarray(ex["output"])].tolist() if tid == "58"
                    else palette[int(ex["output"])] if tid == "57" else ex["output"])
        assert SOLVERS[tid](grid, prompt)[0] == expected, (tid, seed)


@pytest.mark.parametrize("tid", TASKS)
def test_full_canvas_palette_and_determinism(tid):
    examples = sweep(tid)
    shapes = np.array([np.asarray(ex["input"]).shape for ex in examples])
    # MIN_SIDE = 5 floor, 2026-09-15 (user: small sides are often ambiguous).
    assert shapes.min() == 5
    assert shapes.max() == 30
    assert abs(np.corrcoef(shapes.T)[0, 1]) < .20
    assert np.mean(shapes[:, 0] == shapes[:, 1]) < .08
    assert .01 < np.mean(np.prod(shapes, axis=1) >= 700) < .09
    colors = Counter(len(np.unique(ex["input"])) for ex in examples)
    assert set(colors) == set(range(3 if tid in ("56", "57") else 2, 10 if tid == "58" else 11))
    for seed, ex in enumerate(examples[:20]):
        assert ex == generate(tid, seed)
        assert np.asarray(ex["input"]).min() >= 0 and np.asarray(ex["input"]).max() <= 9


def test_contact_adjustment_remains_moderate():
    rows = [ex["metadata"] for ex in sweep("56")]
    assert .20 < np.mean([m["relation"] == "corner" for m in rows]) < .34
    assert .22 < np.mean([m["contact_mode"] == 8 for m in rows]) < .38
    assert .42 < np.mean([ex["output"] == "yes" for ex in sweep("56")]) < .58
    assert max(m["object_cells"] for m in rows) > 100
    large = [m for ex, m in zip(sweep("56"), rows) if np.asarray(ex["input"]).size >= 700]
    assert np.mean([(m["object_cells"]+m["target_cells"])/np.asarray(ex["input"]).size
                    for ex, m in zip(sweep("56"), rows) if np.asarray(ex["input"]).size >= 700]) > .08
    assert large and max(m["contact_length"] for m in rows) >= 8


def test_question_draw_cannot_change_the_input(monkeypatch):
    from generators import contact_56, shape_search_58_59
    for seed in range(60):
        monkeypatch.setattr(contact_56, "choose_connectivity", lambda rng: 4)
        a = generate("56", seed)
        monkeypatch.setattr(contact_56, "choose_connectivity", lambda rng: 8)
        b = generate("56", seed)
        assert a["input"] == b["input"]
        for tid in ("58", "59"):
            inputs = []
            for rule, mode in (("exact", "exact_pattern"), ("rotations", "exact_pattern"),
                               ("dihedral", "exact_pattern"), ("dihedral", "generic")):
                if tid == "58" and mode == "generic":
                    continue
                monkeypatch.setattr(shape_search_58_59, "choose_question", lambda rng, counting, r=rule, m=mode: (r, m))
                ex = generate(tid, seed)
                assert SOLVERS[tid](ex["input"], get_prompt(tid, ex["metadata"]))[0] == ex["output"]
                inputs.append(ex["input"])
            assert all(grid == inputs[0] for grid in inputs)


def test_enclosure_boundary_competition_and_background_policy():
    stats = [SOLVERS["57"](ex["input"], get_prompt("57", ex["metadata"]))[1] for ex in sweep("57")]
    assert .07 < np.mean([not s["bbox_answer_correct"] for s in stats]) < .28
    assert np.mean([s["inside_colors_counted"] >= 5 for s in stats]) > .13
    assert np.mean([s["inside_colors_counted"] in (2, 3) for s in stats]) > .35
    for ex in sweep("57"):
        m = ex["metadata"]
        prompt = get_prompt("57", m)
        assert "ignoring the background" in prompt
        assert "ignoring cells of color" not in prompt
        assert m["background_margin"] > 1.25
        assert m["winner_count"]-m["runner_up_count"] >= 2
        assert m["ignore_background"]


@pytest.mark.parametrize("seed", [2544, 3193, 3589, 3920, 3939, 4404, 5869, 6124, 7146, 7211, 8173, 8487, 9994])
def test_small_enclosure_feasibility_regressions(seed):
    ex = generate("57", seed)
    assert SOLVERS["57"](ex["input"], get_prompt("57", ex["metadata"]))[0] == ex["output"]


@pytest.mark.parametrize("tid", ["58", "59"])
def test_shape_search_variety_and_shortcut_resistance(tid):
    examples = sweep(tid)
    stats = [SOLVERS[tid](ex["input"], get_prompt(tid, ex["metadata"]))[1] for ex in examples]
    counts = Counter(s["n_matches"] for s in stats)
    assert .04 < counts[0]/len(stats) < .13
    assert counts[1] > 0 and max(counts) >= 16
    assert max(ex["metadata"]["template_cells"] for ex in examples) > 60
    for feature in ("size_only_correct", "box_only_correct"):
        values = [s[feature] for s in stats if feature in s]
        assert np.mean(values) < .55
    for ex in examples:
        prompt = get_prompt(tid, ex["metadata"])
        assert "4-connected" in prompt
        if tid == "59" and ex["metadata"]["query_mode"] == "generic":
            assert "whole groups" in prompt
        if "template" in prompt:
            assert ". = a position outside the shape" in prompt


def test_unequal_arm_l_recognition_in_every_orientation():
    from generators.shape_search_58_59 import family_matches
    original = np.zeros((3, 6), bool)
    original[:, 0] = True
    original[-1, :] = True
    for k in range(4):
        for mirrored in (False, True):
            a = np.rot90(original[:, ::-1] if mirrored else original, k)
            cells = set(map(tuple, np.argwhere(a)))
            assert generic59(cells, "L", "dihedral", "")
            assert family_matches(a, "L")


def test_paths_span_simple_and_long_routes():
    from generators import path_routes

    rows = [ex["metadata"] for ex in sweep("60")]
    families = Counter(m["path_family"] for m in rows)
    assert set(families) == set(path_routes.FAMILIES)
    assert all(v/len(rows) > .04 for v in families.values())
    # The templated shapes stay the majority; irregular routes are the rest.
    irregular = sum(v for k, v in families.items() if k in path_routes.IRREGULAR_FAMILIES)
    assert .35 < irregular/len(rows) < .45
    assert all(v/len(rows) > .05 for k, v in families.items()
               if k in path_routes.TEMPLATE_FAMILIES)
    assert min(m["n"] for m in rows) == 1 and max(m["n"] for m in rows) >= 80
    assert max(m["path_length"] for m in rows) >= 250
    assert all(2*m["n"] <= m["path_length"] for m in rows)
    for ex in sweep("60"):
        color = ex["metadata"]["path_color"]
        assert len(components(np.asarray(ex["input"]) == color)) == 1
        count = ex["metadata"]["n"]
        wording = "cell" if count == 1 else "cells"
        assert f"Return the last {count} {wording} from" in get_prompt("60", ex["metadata"])


@pytest.mark.parametrize("tid", TASKS)
def test_validators_reject_corrupt_outputs(tid):
    from generators import contact_56, enclosure_frequency_57, shape_search_58_59, path_ends_60
    ex = deepcopy(generate(tid, 0))
    if tid == "58":
        ex["output"][0][0] = (ex["output"][0][0]+1) % 10
    else:
        ex["output"] = "invalid"
    with pytest.raises(ValueError):
        if tid in ("58", "59"):
            shape_search_58_59.validate_example(ex, counting=tid == "59")
        else:
            {"56": contact_56, "57": enclosure_frequency_57, "60": path_ends_60}[tid].validate_example(ex)
