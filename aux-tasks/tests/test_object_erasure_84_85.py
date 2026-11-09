"""Independent prompt readers and capacity/geometry checks for erasure tasks."""
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import re

import numpy as np
import pytest

from prompts import get_prompt
from registry import generate


def objects(grid):
    """Union-find over the visible input, independent of generator flood-fill."""
    h, w = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    parent = list(range(h * w))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for r in range(h):
        for c in range(w):
            for rr, cc in ((r - 1, c), (r, c - 1)):
                if rr >= 0 and cc >= 0 and grid[rr][cc] == grid[r][c]:
                    parent[root(r * w + c)] = root(rr * w + cc)
    groups = defaultdict(set)
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                groups[root(r * w + c)].add((r, c))
    return bg, list(groups.values())


def normalized(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def equal_rotations(a, b):
    target = normalized(b)
    current = set(a)
    for _ in range(4):
        if normalized(current) == target:
            return True
        current = {(c, -r) for r, c in current}
    return False


def solve(grid, prompt, task):
    assert "Ignoring the background" in prompt
    assert "4-connected same-color component" in prompt
    assert "Leave every other cell unchanged." in prompt
    bg, groups = objects(grid)
    output = [list(row) for row in grid]
    if task == "84":
        assert "0, 90, 180 or 270" in prompt and "colors do not matter" in prompt
        assert "mirror images do not count" in prompt
        selected = [any(i != j and equal_rotations(a, b) for j, b in enumerate(groups)) for i, a in enumerate(groups)]
    else:
        # the criterion is prose, one clause per filter kind (2026-09-17 sweep)
        hit = re.search(r"Remove every object of color (\S+?)\.", prompt)
        if hit:
            kind, operation, want = "color", "=", hit.group(1)
        else:
            comparison, number = re.search(
                r"Remove every object with (fewer than|exactly|more than) (\d+) cells?\.", prompt).groups()
            kind = "size"
            operation = {"fewer than": "<", "exactly": "=", "more than": ">"}[comparison]
            want = int(number)
        selected = []
        for cells in groups:
            r, c = next(iter(cells))
            value = str(grid[r][c]) if kind == "color" else len(cells)
            selected.append(value == want if operation == "=" else value < want if operation == "<" else value > want)
    for cells, remove in zip(groups, selected):
        if remove:
            for r, c in cells:
                output[r][c] = bg
    return output


@lru_cache(maxsize=None)
def examples(task):
    return [generate(task, seed) for seed in range(600)]


@pytest.mark.parametrize("task", ("84", "85"))
def test_independent_prompt_reader(task):
    for seed, ex in enumerate(examples(task)):
        assert solve(ex["input"], get_prompt(task, ex["metadata"]), task) == ex["output"], seed


@pytest.mark.parametrize("task", ("84", "85"))
def test_shuffled_symbols_and_prompt_policy(task):
    for seed, ex in enumerate(examples(task)[:300]):
        palette = np.random.default_rng(seed + 8459).permutation(list("QWERTYUIOP"))
        inp = palette[np.asarray(ex["input"])].tolist()
        out = palette[np.asarray(ex["output"])].tolist()
        prompt = get_prompt(task, ex["metadata"], color_palette=palette.tolist())
        assert solve(inp, prompt, task) == out
        assert not re.search(r"\bcolor\s*[=: ]\s*[0-9]\b", prompt)
        assert "background is" not in prompt and "most frequent" not in prompt
        assert "cells that touch only at a corner are not connected" in prompt


@pytest.mark.parametrize("task", ("84", "85"))
def test_size_palette_and_drawn_geometry_coverage(task):
    ms = [x["metadata"] for x in examples(task)]
    sizes = np.asarray([[m["grid_h"], m["grid_w"]] for m in ms])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(5, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .15
    assert .01 < np.mean(sizes[:, 0] == sizes[:, 1]) < .085
    assert .02 < np.mean(sizes.prod(axis=1) >= 700) < .10
    palettes = Counter(m["n_colors"] for m in ms)
    assert set(palettes) == set(range(2, 11))
    assert .08 < palettes[2] / len(ms) < .30
    assert .15 < sum(palettes[n] for n in (8, 9, 10)) / len(ms) < .40
    assert max(m["largest_object_cells"] for m in ms) > 100
    assert min(m["n_objects"] for m in ms) <= 2
    assert max(m["n_objects"] for m in ms) >= 12
    assert np.mean([m["n_objects"] <= 3 for m in ms]) > .13
    kinds = {kind for m in ms for kind in m["object_kinds"]}
    assert {"L", "T", "U", "H", "frame", "plus", "staircase", "rectangle", "bar", "diamond", "chamfer"} <= kinds
    small = [m["foreground_occupancy"] for m in ms if m["grid_h"] * m["grid_w"] < 150]
    large = [m["foreground_occupancy"] for m in ms if m["grid_h"] * m["grid_w"] >= 700]
    assert np.mean(large) > .20
    assert np.mean(large) >= .85 * np.mean(small)
    assert all(m["background_margin"] >= 1.25 for m in ms)


@pytest.mark.parametrize("task", ("84", "85"))
def test_final_plans_and_metadata_are_visible_measures(task):
    for ex in examples(task):
        m = ex["metadata"]
        _, groups = objects(ex["input"])
        assert m["n_objects"] == m["planned_object_count"] == len(groups)
        assert m["n_colors"] == m["planned_n_colors"]
        assert m["attempts"] == 1
        assert m["placement_method"] == "free_moves_from_capacity_witness"
        assert Counter(map(len, groups)) == Counter(m["object_sizes"])
        for record in m["object_records"]:
            r, c, h, w = record["bounds"]
            if record["kind"] == "bar":
                assert min(h, w) == 1 and record["cells"] == max(h, w)
        assert m["n_removed"] + m["n_kept"] == m["n_objects"]
        assert m["removed_cells"] == sum(a != b for ra, rb in zip(ex["input"], ex["output"]) for a, b in zip(ra, rb))
    assert np.mean([x["metadata"]["freely_moved_objects"] > 0 for x in examples(task)]) > .90


def test_matching_modes_and_multiplicity():
    ms = [x["metadata"] for x in examples("84")]
    assert .025 < np.mean([m["outcome"] != "some" for m in ms]) < .13
    assert {m["outcome"] for m in ms} == {"none", "some", "all"}
    assert np.mean([max(m["class_sizes"]) >= 3 for m in ms]) > .20
    assert np.mean([m["mirror_negative_plan"] for m in ms]) > .04


def test_four_filters_and_numeric_boundaries():
    ms = [x["metadata"] for x in examples("85")]
    hist = Counter(m["filter_kind"] for m in ms)
    assert set(hist) == {"color", "size=", "size<", "size>"}
    assert all(.18 < n / len(ms) < .32 for n in hist.values())
    assert .025 < np.mean([m["outcome"] != "some" for m in ms]) < .15
    assert max(m["filter_value"] for m in ms if m["filter_kind"] != "color") > 50
    inequalities = [m for m in ms if m["filter_kind"] in ("size<", "size>") and m["outcome"] == "some"]
    assert np.mean([m["filter_value"] in m["object_sizes"] for m in inequalities]) > .25
    assert np.mean([m["filter_value"] not in m["object_sizes"] for m in inequalities]) > .20


def test_filter_kind_cannot_change_the_input(monkeypatch):
    from generators import object_erasure_84_85 as g
    for seed in range(30):
        inputs = []
        for kind in ("color", "size=", "size<", "size>"):
            monkeypatch.setattr(g, "choose_filter_kind", lambda rng, kind=kind: kind)
            ex = generate("85", seed)
            inputs.append(ex["input"])
        assert inputs.count(inputs[0]) == 4


@pytest.mark.parametrize("task", ("84", "85"))
def test_validator_rejects_changed_answer_or_plan(task):
    from generators.object_erasure_84_85 import validate
    ex = deepcopy(examples(task)[0])
    ex["output"][0][0] = (ex["output"][0][0] + 1) % 10
    with pytest.raises(ValueError, match="erasure"):
        validate(ex, task)
    ex = deepcopy(examples(task)[0])
    ex["metadata"]["planned_object_count"] += 1
    with pytest.raises(ValueError, match="object count"):
        validate(ex, task)
    ex = deepcopy(examples(task)[0])
    ex["metadata"]["object_records"][0]["bounds"][0] += 1
    with pytest.raises(ValueError, match="body records"):
        validate(ex, task)


def test_validator_rejects_prompt_filter_drift():
    from generators.object_erasure_84_85 import validate
    ex = deepcopy(next(x for x in examples("85") if x["metadata"]["filter_kind"] == "color"))
    ex["metadata"]["filter_color"] = (ex["metadata"]["filter_color"] + 1) % 10
    with pytest.raises(ValueError, match="rendered filter"):
        validate(ex, "85")


@pytest.mark.parametrize("seed", (1216, 1490, 2979, 3842, 4080, 5326, 5346, 6366, 7506, 7902,
                                 8193, 8888, 8991, 9756))
def test_small_capacity_regressions(seed):
    ex = generate("84", seed)
    assert solve(ex["input"], get_prompt("84", ex["metadata"]), "84") == ex["output"]


def test_rotation_reflection_all_copies_and_color_independence():
    grid = [[0] * 22 for _ in range(15)]
    base = {(0, 0), (1, 0), (2, 0), (2, 1)}
    masks = [base, {(c, -r) for r, c in base}, {(-r, -c) for r, c in base}, {(r, -c) for r, c in base}]
    copies = []
    for mask, color, (r0, c0) in zip(masks, (1, 2, 3, 1), ((2, 2), (2, 9), (8, 2), (8, 12))):
        cells = {(r + r0, c + c0) for r, c in normalized(mask)}
        copies.append(cells)
        for r, c in cells:
            grid[r][c] = color
    out = solve(grid, get_prompt("84"), "84")
    assert all(out[r][c] == 0 for cells in copies[:3] for r, c in cells)
    assert all(out[r][c] == 1 for r, c in copies[3])
