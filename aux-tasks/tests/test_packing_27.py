"""Task 27 visible semantics, variant independence and meaningful shape coverage."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import packing_27 as G


def _oracle_scene(grid):
    """Set-based flood fill: no generator metadata or component helpers."""
    h, w = len(grid), len(grid[0])
    frequencies = Counter(x for row in grid for x in row)
    ground, n_ground = frequencies.most_common(1)[0]
    assert n_ground > h * w / 2
    remaining = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != ground}
    objects, dots = [], set()
    while remaining:
        start = min(remaining)
        component, frontier = set(), {start}
        while frontier:
            component |= frontier
            remaining -= frontier
            frontier = {(r + dr, c + dc) for r, c in frontier
                        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))} & remaining
        if len(component) == 1:
            dots |= component
        else:
            rows, cols = zip(*component)
            bh, bw = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
            perimeter = sum((r + dr, c + dc) not in component for r, c in component
                            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)))
            objects.append({"cells": component, "area": len(component), "h": bh, "w": bw,
                            "r": min(rows), "c": min(cols), "bbox": bh * bw, "perimeter": perimeter})
    return ground, objects, dots


def _oracle_pack(grid, edge, start, metric="area"):
    ground, objects, dots = _oracle_scene(grid)
    h, w = len(grid), len(grid[0])
    assert len({o["area"] for o in objects}) == len(objects)
    horizontal = edge in ("top", "bottom")
    assert sum(o["w" if horizontal else "h"] for o in objects) + len(objects) - 1 <= (w if horizontal else h)
    answer = [row[:] for row in grid]
    for obj in objects:
        for r, c in obj["cells"]:
            answer[r][c] = ground
    ordered = sorted(objects, key=lambda o: -o[metric])
    for i, obj in enumerate(ordered):
        if horizontal:
            extent = sum(o["w"] for o in ordered[:i]) + i
            r0 = 0 if edge == "top" else h - obj["h"]
            c0 = extent if start == "left" else w - extent - obj["w"]
        else:
            extent = sum(o["h"] for o in ordered[:i]) + i
            r0 = extent if start == "top" else h - extent - obj["h"]
            c0 = 0 if edge == "left" else w - obj["w"]
        for r, c in obj["cells"]:
            rr, cc = r - obj["r"] + r0, c - obj["c"] + c0
            assert (rr, cc) not in dots
            answer[rr][cc] = grid[r][c]
    return answer


@pytest.fixture(scope="module")
def examples():
    return [G.generate_pack_shapes(np.random.default_rng(seed)) for seed in range(600)]


def test_visible_input_oracle_and_all_eight_counterfactuals(examples):
    for seed, ex in enumerate(examples):
        prompt = G.prompt27(ex["metadata"])
        edge, start = re.search(r"against the (\w+) edge, starting at the (\w+) side", prompt).groups()
        assert _oracle_pack(ex["input"], edge, start) == ex["output"], seed
        answers = set()
        for alt_edge, alt_start in G.VARIANTS:
            expected = _oracle_pack(ex["input"], alt_edge, alt_start)
            assert G.solve_pack_shapes(ex["input"], alt_edge, alt_start).tolist() == expected, (seed, alt_edge, alt_start)
            answers.add(tuple(map(tuple, expected)))
        assert len(answers) == 8, seed


def test_ordinary_input_is_identical_when_variant_mapping_changes(monkeypatch, examples):
    original = G.VARIANTS
    ordinary = [seed for seed, ex in enumerate(examples[:45]) if not ex["metadata"]["identity_planned"]]
    for shift in range(1, 8):
        monkeypatch.setattr(G, "VARIANTS", original[shift:] + original[:shift])
        for seed in ordinary:
            alternate = G.generate_pack_shapes(np.random.default_rng(seed))
            assert alternate["input"] == examples[seed]["input"], (seed, shift)
            assert alternate["metadata"]["pack_edge"] != examples[seed]["metadata"]["pack_edge"] or alternate["metadata"]["start_position"] != examples[seed]["metadata"]["start_position"]
            assert _oracle_pack(alternate["input"], alternate["metadata"]["pack_edge"], alternate["metadata"]["start_position"]) == alternate["output"]


def test_spatial_connectivity_multicolor_accents_diagonal_separation_and_texture():
    grid = np.zeros((10, 10), dtype=int)
    grid[1, 1:3] = [1, 2]
    grid[2, 3], grid[3, 3], grid[3, 4] = 3, 4, 3
    grid[4, 8] = 6
    # The differently coloured singleton accent at (3,3) belongs to the L.
    # The L only touches the two-cell object diagonally and stays separate.
    expected = np.zeros_like(grid)
    expected[8, 0], expected[9, 0], expected[9, 1] = 3, 4, 3
    expected[9, 3:5] = [1, 2]
    expected[4, 8] = 6
    actual = G.solve_pack_shapes(grid, "bottom", "left")
    assert np.array_equal(actual, expected)
    assert _oracle_pack(grid.tolist(), "bottom", "left") == expected.tolist()


def test_palette_permutation_and_construction_metadata_are_irrelevant(examples):
    for seed, ex in enumerate(examples[:150]):
        permutation = np.random.default_rng(seed + 80000).permutation(10)
        grid = permutation[np.asarray(ex["input"])]
        expected = permutation[np.asarray(ex["output"])]
        actual = G.solve_prompt27(grid, G.prompt27(ex["metadata"]))
        assert np.array_equal(actual, expected), seed
    changed = deepcopy(examples[11])
    changed["metadata"].update(background_color=-900, object_areas=[999], object_bbox_areas=[999], shape_families=["invented"])
    G.validate_example(changed)


def test_validator_rejects_changed_answer_and_wrong_given(examples):
    changed = deepcopy(examples[17])
    changed["output"][0][0] = (changed["output"][0][0] + 1) % 10
    with pytest.raises(ValueError, match="reconstruction"):
        G.validate_example(changed)
    changed = deepcopy(examples[17])
    old = (changed["metadata"]["pack_edge"], changed["metadata"]["start_position"])
    for edge, start in G.VARIANTS:
        if (edge, start) != old and _oracle_pack(changed["input"], edge, start) != changed["output"]:
            changed["metadata"].update(pack_edge=edge, start_position=start)
            break
    with pytest.raises(ValueError, match="reconstruction"):
        G.validate_example(changed)


def test_solver_rejects_ties_overflow_and_fixed_detail_collision():
    tied = np.zeros((10, 10), dtype=int)
    tied[1, 1:3], tied[4, 5:7] = 1, 2
    with pytest.raises(ValueError, match="distinct"):
        G.solve_pack_shapes(tied, "bottom", "left")
    overflow = np.zeros((10, 5), dtype=int)
    overflow[1, 0:3], overflow[4, 0:4] = 1, 2
    with pytest.raises(ValueError, match="do not fit"):
        G.solve_pack_shapes(overflow, "bottom", "left")
    collision = np.zeros((10, 10), dtype=int)
    collision[1, 1:3], collision[4, 5:8], collision[9, 0] = 1, 2, 3
    with pytest.raises(ValueError, match="fixed background detail"):
        G.solve_pack_shapes(collision, "bottom", "left")


def test_retry_exhaustion_raises_instead_of_replacing_sampled_parameters(monkeypatch):
    monkeypatch.setattr(G, "MAX_GEOMETRY_ATTEMPTS", 3)
    calls = []

    def too_small(rng, h, w):
        calls.append((h, w))
        return np.ones((1, 1), dtype=bool), "rectangle"

    monkeypatch.setattr(G, "_random_kind", too_small)
    with pytest.raises(RuntimeError, match="without changing size/count/palette: \\(20, 20, 2, 10"):
        G._geometry(np.random.default_rng(42), 20, 20, 2, 10)
    assert len(calls) == 6


def test_distribution_and_nonrectangle_area_reversals(examples):
    n = len(examples)
    shapes = np.asarray([(len(e["input"]), len(e["input"][0])) for e in examples])
    assert set(shapes[:, 0]) == set(shapes[:, 1]) == set(range(3, 31))
    assert abs(np.corrcoef(shapes.T)[0, 1]) < .15
    assert ((shapes[:, 0] == shapes[:, 1]).mean()) < .07
    assert (shapes.prod(axis=1) >= 700).mean() > .035
    palette = Counter(len(np.unique(e["input"])) for e in examples)
    assert set(palette) == set(range(2, 11))
    assert max(palette.values()) / n < .18
    assert all(set(e["metadata"]["object_areas"]) == {o["area"] for o in _oracle_scene(e["input"])[1]} for e in examples)
    families = {f for e in examples for f in e["metadata"]["shape_families"]}
    assert {"rectangle", "L", "U", "H", "T", "frame", "staircase", "chamfer"} <= families
    assert max(max(e["metadata"]["object_areas"]) for e in examples) >= 100
    assert set(e["metadata"]["object_count"] for e in examples) == set(range(2, 7))
    assert .20 < sum(e["metadata"]["bbox_reversal"] for e in examples) / n < .50
    assert .18 < sum(e["metadata"]["perimeter_reversal"] for e in examples) / n < .50
    assert .04 < sum(e["input"] == e["output"] for e in examples) / n < .13
    assert any(e["metadata"]["texture_cells"] for e in examples)
    assert max(e["metadata"]["placement_attempts"] for e in examples) < 100
    assert max(e["metadata"]["geometry_attempts"] for e in examples) < 400


def test_cell_count_clause_changes_the_answer_against_bbox_and_perimeter(examples):
    tested = Counter()
    for ex in examples:
        ground, objects, dots = _oracle_scene(ex["input"])
        if dots:
            continue
        for metric in ("bbox", "perimeter"):
            reverse = any((a["area"] - b["area"]) * (a[metric] - b[metric]) < 0
                          for i, a in enumerate(objects) for b in objects[:i])
            if reverse:
                edge, start = ex["metadata"]["pack_edge"], ex["metadata"]["start_position"]
                assert _oracle_pack(ex["input"], edge, start, metric) != ex["output"]
                tested[metric] += 1
    assert min(tested.values()) > .15 * len(examples)


def test_roomy_scenes_have_meaningful_extra_objects_and_simple_fills(examples):
    roomy = [e for e in examples if min(np.shape(e["input"])) >= 16]
    assert sum(e["metadata"]["object_count"] == 2 for e in roomy) / len(roomy) < .25
    extra_scenes = [e for e in roomy if e["metadata"]["object_count"] > 2]
    shares = []
    meaningful = 0
    solid = total = 0
    for ex in extra_scenes:
        a = np.asarray(ex["input"])
        _, objects, _ = _oracle_scene(ex["input"])
        areas = sorted((o["area"] for o in objects), reverse=True)
        shares.append(sum(areas[:2]) / sum(areas))
        meaningful += areas[2] > 5
        for obj in objects:
            colors = {a[r, c] for r, c in obj["cells"]}
            assert len(colors) <= 3
            solid += len(colors) == 1
            total += 1
    assert np.median(shares) < .85
    assert meaningful > .65 * len(extra_scenes)
    assert solid / total > .45


def test_prompt_defines_geometry_without_naming_ground():
    for edge, start in G.VARIANTS:
        prompt = G.prompt27({"pack_edge": edge, "start_position": start})
        assert "4-connected" in prompt and "even when it contains several colors" in prompt
        assert "descending order of its number of cells" in prompt
        assert "bounding box" in prompt and "translation only" in prompt
        assert "at least two" in prompt and "leave them unchanged" in prompt
        assert not any(name in prompt for name in ("black", "blue", "red", "green", "yellow", "gray", "pink", "orange", "purple", "brown"))
