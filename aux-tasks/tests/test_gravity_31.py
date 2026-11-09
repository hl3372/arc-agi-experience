"""Prompt-level gravity, pass-through regression, and shared-scene coverage."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import gravity_31 as G
from prompts import get_prompt
from registry import generate


def _objects(grid, excluded=(), minimum=1):
    h, w = len(grid), len(grid[0])
    unseen = {(r, c) for r in range(h) for c in range(w) if grid[r][c] not in excluded}
    result, dots = [], set()
    while unseen:
        start = min(unseen)
        color = grid[start[0]][start[1]]
        todo, cells = [start], set()
        unseen.remove(start)
        while todo:
            r, c = todo.pop()
            cells.add((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen and grid[q[0]][q[1]] == color:
                    unseen.remove(q)
                    todo.append(q)
        if len(cells) >= minimum:
            result.append((cells, color))
        else:
            dots |= cells
    return result, dots


def _oracle(grid, prompt):
    h, w = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    objects, _ = _objects(grid, (bg,))
    vectors = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
    variant = "nearest_border" if "nearest grid border" in prompt else re.search(r"toward the (\w+) grid border", prompt)[1]
    points, directions = [], []
    for cells, _ in objects:
        rows, cols = zip(*cells)
        gaps = [min(rows), h - 1 - max(rows), min(cols), w - 1 - max(cols)]
        name = list(vectors)[gaps.index(min(gaps))] if variant == "nearest_border" else variant
        points.append(sorted(cells))
        directions.append(vectors[name])
    owners = [[-1] * w for _ in range(h)]
    for i, cells in enumerate(points):
        for r, c in cells:
            owners[r][c] = i
    while True:
        changes = 0
        for i, ((dr, dc), cells) in enumerate(zip(directions, points)):
            candidate = [(r + dr, c + dc) for r, c in cells]
            if not all(0 <= r < h and 0 <= c < w and owners[r][c] in (-1, i) for r, c in candidate):
                continue
            for r, c in cells:
                owners[r][c] = -1
            for r, c in candidate:
                owners[r][c] = i
            points[i] = candidate
            changes += 1
        if not changes:
            break
    return [[bg if i < 0 else objects[i][1] for i in row] for row in owners]


@pytest.fixture(scope="module")
def examples():
    return [generate("31", seed) for seed in range(600)]


def test_independent_prompt_oracle_and_all_variants(examples):
    for seed, ex in enumerate(examples):
        assert _oracle(ex["input"], get_prompt("31", ex["metadata"])) == ex["output"], seed
        if seed < 100:
            for v in G.VARIANTS:
                m = {"rule": "nearest_border" if v == "nearest_border" else "directional_gravity", "direction": v}
                assert _oracle(ex["input"], G.prompt31(m)) == G.solve_gravity(ex["input"], v).tolist()


def test_gravity_no_overtaking_and_tie_order():
    a = np.zeros((20, 15), dtype=int)
    a[10:12, 8] = 1
    a[9, 11:14] = 2
    a[10, 11] = 2
    out = G.solve_gravity(a, "nearest_border")
    assert out[10, 11] == 1 and out[10, 12] == 2 and out[10, 14] == 0
    assert Counter(a.ravel()) == Counter(out.ravel())
    b = np.zeros((11, 11), dtype=int)
    b[4, 4:6] = b[5, 4] = 7
    expected = np.zeros_like(b)
    expected[0, 4:6] = expected[1, 4] = 7
    assert np.array_equal(G.solve_gravity(b, "nearest_border"), expected)


def test_variant_choice_does_not_change_input(monkeypatch, examples):
    monkeypatch.setattr(G, "VARIANTS", G.VARIANTS[1:] + G.VARIANTS[:1])
    for seed, original in enumerate(examples[:120]):
        changed = generate("31", seed)
        assert changed["input"] == original["input"]
        assert _oracle(changed["input"], get_prompt("31", changed["metadata"])) == changed["output"]


def test_motion_variety_and_size_scaling(examples):
    sizes = np.asarray([(len(e["input"]), len(e["input"][0])) for e in examples])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(1, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .15
    assert set(e["metadata"]["n_colors"] for e in examples) == set(range(2, 11))
    assert max(n for e in examples for n in e["metadata"]["object_areas"]) >= 80
    assert len({f for e in examples for f in e["metadata"]["shape_families"]}) >= 8
    assert .015 <= np.mean([e["input"] == e["output"] for e in examples]) <= .13
    nearest = [e for e in examples if e["metadata"]["rule"] == "nearest_border"]
    assert np.mean([e["metadata"]["nearest_ties"] > 0 for e in nearest]) > .2
    assert np.mean([e["metadata"]["blocked_objects"] > 0 for e in examples]) > .25
    large = [e["metadata"]["foreground_cells"] / int(h * w) for e, (h, w) in zip(examples, sizes) if h * w >= 700]
    assert np.mean(large) > .13


def test_remapped_symbols_and_validator_rejects_wrong_answer(examples):
    for seed, ex in enumerate(examples[:150]):
        p = np.random.default_rng(seed + 50000).permutation(list("ABCDEFGHIJ"))
        inp, out = p[np.asarray(ex["input"])], p[np.asarray(ex["output"])]
        assert _oracle(inp.tolist(), get_prompt("31", ex["metadata"])) == out.tolist()
        variant = ex["metadata"]["direction"] if ex["metadata"]["rule"] != "nearest_border" else "nearest_border"
        assert np.array_equal(G.solve_gravity(inp, variant), out)
    bad = deepcopy(examples[0])
    bad["output"][0][0] = (bad["output"][0][0] + 1) % 10
    with pytest.raises(ValueError, match="reconstruction"):
        G.validate_example(bad)
