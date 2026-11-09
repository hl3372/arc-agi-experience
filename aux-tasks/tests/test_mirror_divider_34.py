"""Infer the divider and shapes; preserve symmetric AND asymmetric detail."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from generators import mirror_divider_34 as G
from registry import generate
from tests.test_gravity_31 import _objects


def _oracle(grid, *, naive=False):
    h, w = len(grid), len(grid[0])
    ground = Counter(x for row in grid for x in row).most_common(1)[0][0]
    candidates = []
    for axis, lines in enumerate((grid, list(zip(*grid)))):
        for n, line in enumerate(lines):
            if len(set(line)) == 1 and line[0] != ground:
                candidates.append((axis, n, line[0]))
    assert len(candidates) == 1
    axis, n, divider = candidates[0]
    objects, dots = _objects(grid, (ground, divider), minimum=2)
    cells = set().union(*(s for s, _ in objects))
    if naive:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if grid[r][c] not in (ground, divider)}
    out = [row[:] for row in grid]
    for r, c in cells:
        rr, cc = (2 * n - r, c) if axis == 0 else (r, 2 * n - c)
        if naive and not (0 <= rr < h and 0 <= cc < w):
            continue
        assert 0 <= rr < h and 0 <= cc < w
        if not naive:
            assert grid[rr][cc] == ground
        out[rr][cc] = grid[r][c]
    return out, dots


@pytest.fixture(scope="module")
def examples():
    return [generate("34", seed) for seed in range(600)]


def test_independent_visible_oracle(examples):
    for seed, ex in enumerate(examples):
        expected, dots = _oracle(ex["input"])
        assert expected == ex["output"], seed
        assert all(ex["input"][r][c] == ex["output"][r][c] for r, c in dots)


def test_both_background_symmetries_and_ignoring_detail_matters(examples):
    groups = Counter()
    for ex in examples:
        mode = ex["metadata"]["background_symmetry"]
        groups[mode] += 1
        naive, _ = _oracle(ex["input"], naive=True)
        assert (naive != ex["output"]) == (mode == "asymmetric")
    assert groups["symmetric"] > len(examples) * .12
    assert groups["asymmetric"] > len(examples) * .30
    assert groups["solid"] > len(examples) * .20


def test_shape_size_palette_and_layout_coverage(examples):
    sizes = np.asarray([(len(e["input"]), len(e["input"][0])) for e in examples])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(3, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .15
    assert set(e["metadata"]["n_colors"] for e in examples) == set(range(3, 11))
    assert len({f for e in examples for f in e["metadata"]["shape_families"]}) >= 8
    assert max(n for e in examples for n in e["metadata"]["object_areas"]) > 60
    assert len({e["metadata"]["background"] for e in examples}) == 7
    offcenter = sum(2 * e["metadata"]["divider_position"] != (h if e["metadata"]["divider_orientation"] == "horizontal" else w) - 1
                    for e, (h, w) in zip(examples, sizes))
    assert offcenter > .5 * len(examples)


def test_symbol_remap_and_validator_do_not_use_planned_objects(examples):
    for seed, ex in enumerate(examples[:150]):
        p = np.random.default_rng(seed + 65000).permutation(list("ABCDEFGHIJ"))
        a, out = p[np.asarray(ex["input"])], p[np.asarray(ex["output"])]
        assert np.array_equal(G.solve_mirror(a), out)
        assert _oracle(a.tolist())[0] == out.tolist()
    bad = deepcopy(examples[0])
    bad["metadata"].update(background_color=-100, divider_color=-100, divider_position=1000, object_areas=[])
    G.validate_example(bad)
    bad["output"][0][0] = (bad["output"][0][0] + 1) % 10
    with pytest.raises(ValueError, match="reconstruction"):
        G.validate_example(bad)


def test_shallow_dividers_can_retain_asymmetric_detail():
    for seed in range(100):
        ex = generate("34", seed)
        if ex["metadata"]["background_symmetry"] != "asymmetric":
            continue
        assert _oracle(ex["input"])[0] == ex["output"]
        assert _oracle(ex["input"], naive=True)[0] != ex["output"]


def test_original_shapes_occur_on_both_sides_without_collisions(examples):
    eligible = [e for e in examples if e["metadata"]["object_count"] >= 2]
    mixed = 0
    variants = set()
    for ex in eligible:
        a = ex["input"]
        bg = Counter(x for row in a for x in row).most_common(1)[0][0]
        m = ex["metadata"]
        objects, _ = _objects(a, (bg, m["divider_color"]), minimum=2)
        axis = 0 if m["divider_orientation"] == "horizontal" else 1
        counts = [sum(next(iter(cells))[axis] < m["divider_position"] for cells, _ in objects)]
        counts.append(len(objects) - counts[0])
        assert counts == m["source_side_counts"]
        assert bool(min(counts)) == m["mixed_source_sides"]
        if min(counts):
            mixed += 1
            variants.add(m["divider_orientation"])
        assert _oracle(a)[0] == ex["output"]
    assert .40 < mixed / len(eligible) < .60
    assert variants == {"horizontal", "vertical"}
    assert "either or both sides" in G.PROMPT34
    assert "All shapes lie on one side" not in G.PROMPT34
