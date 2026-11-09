"""Verifier tests for core2_extra region and boundary tasks."""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate


TASK_IDS = ["111", "113", "114", "124", "125", "126", "151", "156", "167", "173"]
N_SEEDS = 30
DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DELTAS_8 = DELTAS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _deltas(connectivity: int):
    return DELTAS_8 if int(connectivity) == 8 else DELTAS_4


def _in_bounds(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    return 0 <= r < h and 0 <= c < w


def _neighbors(grid: np.ndarray, cell: tuple[int, int], connectivity: int = 4):
    r, c = cell
    for dr, dc in _deltas(connectivity):
        nr, nc = r + dr, c + dc
        if _in_bounds(grid, nr, nc):
            yield nr, nc


def _flood(grid: np.ndarray, starts: set[tuple[int, int]], passable, connectivity: int = 4, max_distance=None):
    seen = set()
    q = deque()
    for cell in starts:
        if passable(cell):
            seen.add(cell)
            q.append((cell, 0))
    while q:
        cell, dist = q.popleft()
        if max_distance is not None and dist >= max_distance:
            continue
        for nb in _neighbors(grid, cell, connectivity):
            if nb not in seen and passable(nb):
                seen.add(nb)
                q.append((nb, dist + 1))
    return seen


def _color_components(grid: np.ndarray, color: int, connectivity: int = 4):
    seen = set()
    comps = []
    for r, c in np.argwhere(grid == color):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        comp = _flood(grid, {cell}, lambda p: int(grid[p]) == color, connectivity)
        seen |= comp
        comps.append(comp)
    return comps


def _border_reachable_background(
    grid: np.ndarray,
    bg: int,
    barriers: set[int] | None = None,
    connectivity: int = 4,
):
    barriers = barriers or set()
    h, w = grid.shape
    starts = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    return _flood(
        grid,
        starts,
        lambda p: int(grid[p]) == bg and int(grid[p]) not in barriers,
        connectivity,
    )


def _articulation_cells(grid: np.ndarray, color: int, connectivity: int):
    comps = _color_components(grid, color, connectivity)
    assert comps
    cells = max(comps, key=len)
    result = set()
    for removed in cells:
        remaining = cells - {removed}
        if len(remaining) < 2:
            continue
        reached = _flood(
            grid,
            {min(remaining)},
            lambda p, remaining=remaining: p in remaining,
            connectivity,
        )
        if reached != remaining:
            result.add(removed)
    return result


def _corner_marked(grid: np.ndarray, meta: dict):
    target = meta["target_color"]
    object_cells = {tuple(int(v) for v in p) for p in np.argwhere(grid == target)}
    pairs = [
        ((-1, 0), (0, -1), (-1, -1)),
        ((-1, 0), (0, 1), (-1, 1)),
        ((1, 0), (0, -1), (1, -1)),
        ((1, 0), (0, 1), (1, 1)),
    ]
    expected = grid.copy()
    for r, c in object_cells:
        outer = False
        inner = False
        for (dr1, dc1), (dr2, dc2), (ddr, ddc) in pairs:
            side1 = (r + dr1, c + dc1) in object_cells
            side2 = (r + dr2, c + dc2) in object_cells
            diag = (r + ddr, c + ddc) in object_cells
            outer = outer or (not side1 and not side2)
            inner = inner or (side1 and side2 and not diag)
        if inner:
            expected[r, c] = meta["inner_corner_color"]
        elif outer:
            expected[r, c] = meta["outer_corner_color"]
    return expected


def _boundary_layer(grid: np.ndarray, color: int, layer_index: int, connectivity: int):
    remaining = {tuple(int(v) for v in p) for p in np.argwhere(grid == color)}
    layer = set()
    for _ in range(layer_index):
        layer = {
            (r, c)
            for r, c in remaining
            if any((r + dr, c + dc) not in remaining for dr, dc in _deltas(connectivity))
        }
        remaining -= layer
    return layer


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        fn(seed, grid, out, ex["metadata"])
        assert generate(task_id, seed) == ex


def test_all_prompts_exist():
    for task_id in TASK_IDS:
        ex = generate(task_id, 0)
        assert "[No prompt" not in get_prompt(task_id, ex["metadata"])


def test_task_111():
    def check(seed, grid, out, meta):
        seeds = {tuple(int(v) for v in p) for p in np.argwhere(grid == meta["seed_color"])}
        reachable = _flood(
            grid,
            seeds,
            lambda p: int(grid[p]) in {meta["background_color"], meta["seed_color"]},
            meta["connectivity"],
        )
        expected = grid.copy()
        for cell in reachable:
            expected[cell] = meta["fill_color"]
        assert out.tolist() == expected.tolist(), seed

        for color in meta["distractor_colors"]:
            mask = grid == color
            assert np.any(mask), seed
            assert np.array_equal(out[mask], grid[mask]), seed

        reachable_4 = _flood(
            grid,
            seeds,
            lambda p: int(grid[p]) in {meta["background_color"], meta["seed_color"]},
            4,
        )
        reachable_8 = _flood(
            grid,
            seeds,
            lambda p: int(grid[p]) in {meta["background_color"], meta["seed_color"]},
            8,
        )
        diagonal_only = reachable_8 - reachable_4
        assert diagonal_only, seed
        if meta["connectivity"] == 4:
            assert all(out[cell] != meta["fill_color"] for cell in diagonal_only), seed
        else:
            assert all(out[cell] == meta["fill_color"] for cell in diagonal_only), seed

    _verify("111", check)


def test_task_113():
    def check(seed, grid, out, meta):
        protected = set(meta["protected_colors"])
        assert len(protected) >= 2, seed
        starts = {tuple(int(v) for v in p) for p in np.argwhere(grid == meta["target_color"])}
        reached = _flood(
            grid,
            starts,
            lambda p: int(grid[p]) not in protected,
            meta["connectivity"],
            max_distance=meta["distance"],
        )
        expected = grid.copy()
        for cell in reached:
            if int(grid[cell]) not in protected:
                expected[cell] = meta["target_color"]
        assert out.tolist() == expected.tolist(), seed

        for color in protected:
            mask = grid == color
            assert np.any(mask), seed
            assert np.array_equal(out[mask], grid[mask]), seed

        distractor = meta["distractor_colors"][0]
        distractor_cells = {tuple(int(v) for v in p) for p in np.argwhere(grid == distractor)}
        assert distractor_cells, seed
        assert any(cell in reached and out[cell] == meta["target_color"] for cell in distractor_cells), seed
        assert any(cell not in reached and out[cell] == distractor for cell in distractor_cells), seed

        reached_4 = _flood(
            grid,
            starts,
            lambda p: int(grid[p]) not in protected,
            4,
            max_distance=meta["distance"],
        )
        reached_8 = _flood(
            grid,
            starts,
            lambda p: int(grid[p]) not in protected,
            8,
            max_distance=meta["distance"],
        )
        diagonal_only = reached_8 - reached_4
        assert diagonal_only, seed
        if meta["connectivity"] == 4:
            assert any(out[cell] != meta["target_color"] for cell in diagonal_only), seed
        else:
            assert all(out[cell] == meta["target_color"] for cell in diagonal_only), seed

    _verify("113", check)


def test_task_114():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        outside = _border_reachable_background(grid, bg, connectivity=meta["connectivity"])
        expected = grid.copy()
        target_holes = []
        ignored_matching_decoys = []
        for comp in _color_components(grid, bg, meta["connectivity"]):
            if comp & outside:
                continue
            touching = {
                int(grid[nb])
                for cell in comp
                for nb in _neighbors(grid, cell, 4)
                if int(grid[nb]) != bg
            }
            area = len(comp)
            if meta["area_condition"] == "at_most":
                ok = area <= meta["area_threshold"]
            else:
                ok = area >= meta["area_threshold"]
            if touching == {meta["target_color"]}:
                target_holes.append((comp, ok))
            elif ok and touching == set(meta["distractor_colors"]):
                ignored_matching_decoys.append(comp)
            if touching == {meta["target_color"]} and ok:
                for cell in comp:
                    expected[cell] = meta["fill_color"]
        assert out.tolist() == expected.tolist(), seed

        assert any(ok for _, ok in target_holes), seed
        assert any(not ok for _, ok in target_holes), seed
        assert ignored_matching_decoys, seed
        for comp in ignored_matching_decoys:
            assert all(out[cell] == bg for cell in comp), seed

        def target_hole_areas(connectivity):
            outside_for_conn = _border_reachable_background(grid, bg, connectivity=connectivity)
            areas = []
            for comp in _color_components(grid, bg, connectivity):
                if comp & outside_for_conn:
                    continue
                touching = {
                    int(grid[nb])
                    for cell in comp
                    for nb in _neighbors(grid, cell, 4)
                    if int(grid[nb]) != bg
                }
                if touching == {meta["target_color"]}:
                    areas.append(len(comp))
            return sorted(areas)

        assert len(target_hole_areas(4)) > len(target_hole_areas(8)), seed

    _verify("114", check)


def test_tasks_111_113_114_distributional_coverage():
    metas = {
        task_id: [generate(task_id, seed)["metadata"] for seed in range(100)]
        for task_id in ("111", "113", "114")
    }
    for task_id in ("111", "113", "114"):
        assert {meta["connectivity"] for meta in metas[task_id]} == {4, 8}, task_id

    assert {meta["distance"] for meta in metas["113"]} == {1, 2, 3}
    assert all(len(meta["protected_colors"]) >= 2 for meta in metas["113"])
    assert {meta["area_condition"] for meta in metas["114"]} == {"at_most", "at_least"}


def test_tasks_111_113_114_prompt_clarity():
    for task_id in ("111", "113", "114"):
        for seed in range(20):
            ex = generate(task_id, seed)
            meta = ex["metadata"]
            prompt = get_prompt(task_id, meta)
            lower = prompt.lower()
            assert "same-size output grid" in lower, (task_id, seed, prompt)
            assert f"{meta['connectivity']}-connectivity" in lower, (task_id, seed, prompt)
            assert "edge-neighbor" in lower, (task_id, seed, prompt)
            if meta["connectivity"] == 8:
                assert "diagonal" in lower and "corner-neighbor" in lower, (task_id, seed, prompt)
            assert "unchanged" in lower, (task_id, seed, prompt)

    prompt_111 = get_prompt("111", generate("111", 0)["metadata"]).lower()
    assert "unique seed" in prompt_111
    assert "barriers" in prompt_111
    assert "including the seed" in prompt_111

    prompt_113 = get_prompt("113", generate("113", 0)["metadata"]).lower()
    assert "distance 0" in prompt_113
    assert "protected" in prompt_113
    assert "never enter, cross, or overwrite" in prompt_113
    assert "non-protected distractor" in prompt_113

    prompt_114 = get_prompt("114", generate("114", 0)["metadata"]).lower()
    assert "not reachable from the grid border" in prompt_114
    assert "edge-neighbor touching" in prompt_114
    assert "ignore holes enclosed by distractor" in prompt_114
    assert "number of cells" in prompt_114


def test_task_124():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        fg = meta["foreground_color"]
        expected = np.full(grid.shape, bg, dtype=np.uint8)
        if meta["negative_space_mode"] == "inside_frame":
            outside = _border_reachable_background(grid, bg)
            for comp in _color_components(grid, bg, 4):
                if not (comp & outside):
                    for cell in comp:
                        expected[cell] = meta["output_color"]
        else:
            rows = np.where(np.any(grid == fg, axis=1))[0]
            for r in rows:
                fg_cols = np.where(grid[r] == fg)[0]
                left_end = int(np.max(fg_cols[fg_cols < np.median(fg_cols)]))
                right_start = int(np.min(fg_cols[fg_cols > np.median(fg_cols)]))
                for c in range(left_end + 1, right_start):
                    if grid[r, c] == bg:
                        expected[r, c] = meta["output_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("124", check)


def test_task_125():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        connectors = _articulation_cells(grid, meta["target_color"], meta["connectivity"])
        assert connectors, seed
        for cell in connectors:
            expected[cell] = meta["connector_color"]
        assert out.tolist() == expected.tolist(), seed
        # The object must be a single 4-connected component (the prompt says "the object").
        comps = _color_components(grid, meta["target_color"], meta["connectivity"])
        assert len(comps) == 1, (seed, len(comps))

    _verify("125", check)

    # Macro-structure (number of blobs) and connector count must both vary.
    metas = [generate("125", seed)["metadata"] for seed in range(100)]
    blob_counts = {meta["num_blobs"] for meta in metas}
    assert len(blob_counts) > 1, blob_counts
    connector_counts = {
        int((np.array(generate("125", seed)["input"]) != np.array(generate("125", seed)["output"])).sum())
        for seed in range(100)
    }
    assert len(connector_counts) > 3, connector_counts


def test_task_126():
    def check(seed, grid, out, meta):
        expected = _corner_marked(grid, meta)
        assert np.any(expected == meta["outer_corner_color"]), seed
        assert np.any(expected == meta["inner_corner_color"]), seed
        assert out.tolist() == expected.tolist(), seed

    _verify("126", check)
    changed_counts = {
        int((np.array(generate("126", seed)["input"]) != np.array(generate("126", seed)["output"])).sum())
        for seed in range(100)
    }
    notch_counts = {len(generate("126", seed)["metadata"].get("notch_sides", [])) for seed in range(100)}
    assert len(changed_counts) > 1, changed_counts
    assert len(notch_counts) > 1, notch_counts


def test_task_151():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        outside = _border_reachable_background(grid, bg, set(meta["barrier_colors"]))
        expected = grid.copy()
        for r, c in np.argwhere(grid == bg):
            cell = (int(r), int(c))
            expected[cell] = meta["outside_color"] if cell in outside else meta["inside_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("151", check)


def test_task_156():
    def check(seed, grid, out, meta):
        bg = meta["background_color"]
        outside = _border_reachable_background(grid, bg)
        expected = grid.copy()
        for comp in _color_components(grid, bg, 4):
            if comp & outside:
                continue
            touching = {
                int(grid[nb])
                for cell in comp
                for nb in _neighbors(grid, cell, 4)
                if int(grid[nb]) != bg
            }
            if len(touching) == 1:
                fill = next(iter(touching))
                for cell in comp:
                    expected[cell] = fill
        assert out.tolist() == expected.tolist(), seed

    _verify("156", check)

    # The discriminative negative case (a hole touching multiple object colors,
    # left unchanged) must be exercised in a meaningful fraction of seeds, and
    # every seed must still contain at least one filled (positive) hole.
    def has_multicolor_hole(seed):
        ex = generate("156", seed)
        grid = np.array(ex["input"], dtype=np.uint8)
        bg = ex["metadata"]["background_color"]
        outside = _border_reachable_background(grid, bg)
        positive = False
        multi = False
        for comp in _color_components(grid, bg, 4):
            if comp & outside:
                continue
            touching = {
                int(grid[nb])
                for cell in comp
                for nb in _neighbors(grid, cell, 4)
                if int(grid[nb]) != bg
            }
            if len(touching) == 1:
                positive = True
            elif len(touching) >= 2:
                multi = True
        return positive, multi

    flags = [has_multicolor_hole(seed) for seed in range(100)]
    assert all(positive for positive, _ in flags), "every seed needs a fillable hole"
    n_multi = sum(1 for _, multi in flags if multi)
    assert n_multi >= 20, n_multi


def test_task_167():
    def check(seed, grid, out, meta):
        layer = _boundary_layer(
            grid,
            meta["target_color"],
            meta["layer_index"],
            meta["connectivity"],
        )
        assert layer, seed
        expected = grid.copy()
        for cell in layer:
            expected[cell] = meta["layer_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("167", check)


def test_task_173():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        for r, c in np.argwhere(grid == meta["eligible_color"]):
            cell = (int(r), int(c))
            count = sum(
                1
                for nb in _neighbors(grid, cell, meta["connectivity"])
                if int(grid[nb]) == meta["neighbor_color"]
            )
            ok = count == meta["neighbor_count"] if meta["condition"] == "exactly" else count >= meta["neighbor_count"]
            if ok:
                expected[cell] = meta["marker_color"]
        assert out.tolist() == expected.tolist(), seed

    _verify("173", check)


if __name__ == "__main__":
    test_all_prompts_exist()
    test_task_111()
    test_task_113()
    test_task_114()
    test_task_124()
    test_task_125()
    test_task_126()
    test_task_151()
    test_task_156()
    test_task_167()
    test_task_173()
    print("ok")
