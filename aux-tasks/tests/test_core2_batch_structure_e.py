"""Verifier tests for structure batch E second-batch tasks."""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt
from registry import generate

TASK_IDS = ["116", "117", "119", "155", "168", "169", "174", "177", "181", "182"]
N_SEEDS = 30


def _verify(task_id: str, fn):
    for seed in range(N_SEEDS):
        ex = generate(task_id, seed)
        prompt = get_prompt(task_id, ex["metadata"])
        assert "[No prompt" not in prompt
        grid = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        fn(seed, grid, out, ex["metadata"])
        assert generate(task_id, seed) == ex


def _minimal_periods(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_period = h
    col_period = w
    for p in range(1, h + 1):
        if h % p == 0 and np.array_equal(grid, np.tile(grid[:p, :], (h // p, 1))):
            row_period = p
            break
    for p in range(1, w + 1):
        if w % p == 0 and np.array_equal(grid, np.tile(grid[:, :p], (1, w // p))):
            col_period = p
            break
    return row_period, col_period


def _fill_periodic(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    unknown = meta["unknown_color"]
    h, w = grid.shape
    if meta["period_mode"] == "row_wise":
        p = meta["period_length"]
        for r in range(h):
            for residue in range(p):
                coords = [(r, c) for c in range(residue, w, p)]
                known = [int(grid[rr, cc]) for rr, cc in coords if grid[rr, cc] != unknown]
                assert known and len(set(known)) == 1
                for rr, cc in coords:
                    out[rr, cc] = known[0]
    elif meta["period_mode"] == "column_wise":
        p = meta["period_length"]
        for c in range(w):
            for residue in range(p):
                coords = [(r, c) for r in range(residue, h, p)]
                known = [int(grid[rr, cc]) for rr, cc in coords if grid[rr, cc] != unknown]
                assert known and len(set(known)) == 1
                for rr, cc in coords:
                    out[rr, cc] = known[0]
    else:
        th, tw = meta["tile_height"], meta["tile_width"]
        for rr in range(th):
            for cc in range(tw):
                coords = [(r, c) for r in range(rr, h, th) for c in range(cc, w, tw)]
                known = [int(grid[r, c]) for r, c in coords if grid[r, c] != unknown]
                assert known and len(set(known)) == 1
                for r, c in coords:
                    out[r, c] = known[0]
    return out


def _period_class_coords(
    shape: tuple[int, int],
    meta: dict,
    r: int,
    c: int,
) -> list[tuple[int, int]]:
    h, w = shape
    if meta["period_mode"] == "row_wise":
        p = meta["period_length"]
        return [(r, cc) for cc in range(c % p, w, p)]
    if meta["period_mode"] == "column_wise":
        p = meta["period_length"]
        return [(rr, c) for rr in range(r % p, h, p)]
    th, tw = meta["tile_height"], meta["tile_width"]
    return [(rr, cc) for rr in range(r % th, h, th) for cc in range(c % tw, w, tw)]


def _pattern_cells(pattern_type: str, h: int, w: int, meta: dict) -> set[tuple[int, int]]:
    spacing = meta["spacing"]
    if pattern_type == "vertical_stripes":
        return {(r, c) for r in range(h) for c in range(meta["column_offset"], w, spacing)}
    if pattern_type == "horizontal_stripes":
        return {(r, c) for r in range(meta["row_offset"], h, spacing) for c in range(w)}
    if pattern_type == "rectangular_lattice":
        rows = set(range(meta["row_offset"], h, spacing))
        cols = set(range(meta["column_offset"], w, spacing))
        return {(r, c) for r in range(h) for c in range(w) if r in rows or c in cols}
    if meta["diagonal_direction"] == "down_right":
        return {(r, c) for r in range(h) for c in range(w) if (r - c - meta["diagonal_offset"]) % spacing == 0}
    return {(r, c) for r in range(h) for c in range(w) if (r + c - meta["diagonal_offset"]) % spacing == 0}


def _components(grid: np.ndarray, bg: int, color: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg or (color is not None and grid[r, c] != color):
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            comp_color = int(grid[r, c])
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == comp_color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append({
                "color": comp_color,
                "cells": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            })
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _interface_cells(grid: np.ndarray, meta: dict) -> set[tuple[int, int]]:
    a_cells = {tuple(p) for p in np.argwhere(grid == meta["object_a_color"])}
    b_cells = {tuple(p) for p in np.argwhere(grid == meta["object_b_color"])}
    a_touching = set()
    b_touching = set()
    for r, c in a_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in b_cells:
                a_touching.add((r, c))
                b_touching.add((r + dr, c + dc))
    if meta["contact_convention"] == "object_a_side_cells":
        return a_touching
    if meta["contact_convention"] == "object_b_side_cells":
        return b_touching
    return a_touching | b_touching


def _representatives(start: int, end: int, rule: str) -> set[int]:
    if rule == "first":
        return {start}
    if rule == "last":
        return {end - 1}
    if rule == "endpoints":
        return {start, end - 1}
    return {(start + end - 1) // 2}


def _replace_runs(grid: np.ndarray, meta: dict) -> np.ndarray:
    expected = grid.copy()
    axis = meta["axis"]
    outer = grid.shape[0] if axis == "horizontal" else grid.shape[1]
    inner = grid.shape[1] if axis == "horizontal" else grid.shape[0]
    for i in range(outer):
        j = 0
        while j < inner:
            r, c = (i, j) if axis == "horizontal" else (j, i)
            if grid[r, c] != meta["target_color"]:
                j += 1
                continue
            start = j
            while j < inner:
                rr, cc = (i, j) if axis == "horizontal" else (j, i)
                if grid[rr, cc] != meta["target_color"]:
                    break
                j += 1
            reps = _representatives(start, j, meta["representative_rule"])
            for k in range(start, j):
                rr, cc = (i, k) if axis == "horizontal" else (k, i)
                expected[rr, cc] = meta["marker_color"] if k in reps else meta["background_color"]
    return expected


def _distance(a: tuple[int, int], b: tuple[int, int], metric: str) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc if metric == "manhattan" else max(dr, dc)


def _distance_marks(grid: np.ndarray, meta: dict) -> set[tuple[int, int]]:
    a_cells = sorted(map(tuple, np.argwhere(grid == meta["object_a_color"])))
    b_cells = sorted(map(tuple, np.argwhere(grid == meta["object_b_color"])))
    pairs = [(a, b, _distance(a, b, meta["distance_metric"])) for a in a_cells for b in b_cells]
    best = min(d for _, _, d in pairs)
    winners = sorted((a, b) for a, b, d in pairs if d == best)
    if meta["tie_policy"] == "lexicographically_first_pair":
        return set(winners[0])
    if meta["tie_policy"] == "object_a_minimizers":
        return {a for a, _ in winners}
    if meta["tie_policy"] == "object_b_minimizers":
        return {b for _, b in winners}
    return {p for pair in winners for p in pair}


def _directional_boundary(cells: set[tuple[int, int]], direction: str) -> set[tuple[int, int]]:
    dr, dc = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]
    return {(r, c) for r, c in cells if (r + dr, c + dc) not in cells}


def _line_counts(grid: np.ndarray, meta: dict) -> list[int]:
    lines = [grid[r, :] for r in range(grid.shape[0])] if meta["axis"] == "row" else [grid[:, c] for c in range(grid.shape[1])]
    counts = []
    for line in lines:
        if meta["count_rule"] == "target_color_count":
            counts.append(int(np.count_nonzero(line == meta["target_color"])))
        elif meta["count_rule"] == "non_background_count":
            counts.append(int(np.count_nonzero(line != meta["background_color"])))
        else:
            counts.append(len(set(int(v) for v in line)))
    return counts


def _selected_line(grid: np.ndarray, meta: dict) -> int:
    counts = _line_counts(grid, meta)
    best = max(counts) if meta["extremum"] == "maximum" else min(counts)
    winners = [i for i, count in enumerate(counts) if count == best]
    assert len(winners) == 1 or meta["tie_policy"] in ("first", "last")
    return winners[0] if meta["tie_policy"] == "first" else winners[-1]


def test_task_116():
    def check(seed, grid, out, meta):
        th, tw = _minimal_periods(grid)
        assert (th, tw) == (meta["tile_height"], meta["tile_width"]), seed
        assert th >= 2 and tw >= 2, seed
        assert out.shape == (th, tw), seed
        expected = grid[:th, :tw]
        assert out.tolist() == expected.tolist(), seed
    _verify("116", check)


def test_task_116_distribution_and_prompt_clarity():
    sizes = []
    for seed in range(N_SEEDS):
        ex = generate("116", seed)
        meta = ex["metadata"]
        sizes.append((meta["tile_height"], meta["tile_width"]))
        prompt = get_prompt("116", meta).lower()
        assert "smallest rectangular tile" in prompt
        assert "top-left" in prompt
        assert "zero-indexed" in prompt
        assert "rows 0 through" in prompt
        assert "columns 0 through" in prompt
        assert f"height {meta['tile_height']}" in prompt
        assert f"width {meta['tile_width']}" in prompt
    assert all(h >= 2 and w >= 2 for h, w in sizes)
    assert len({h for h, _ in sizes}) >= 2
    assert len({w for _, w in sizes}) >= 2
    assert any(h != w for h, w in sizes)


def test_task_117():
    def check(seed, grid, out, meta):
        unknown = meta["unknown_color"]
        assert np.any(grid == unknown), seed
        assert not np.any(out == unknown), seed
        assert np.array_equal(out[grid != unknown], grid[grid != unknown]), seed
        expected = _fill_periodic(grid, meta)
        assert out.tolist() == expected.tolist(), seed
        for r, c in map(tuple, np.argwhere(grid == unknown)):
            coords = _period_class_coords(grid.shape, meta, r, c)
            known = [int(grid[rr, cc]) for rr, cc in coords if grid[rr, cc] != unknown]
            assert len(known) >= 2, seed
            assert len(set(known)) == 1, seed
        if meta["period_mode"] == "row_wise":
            assert _minimal_periods(out) == (out.shape[0], meta["period_length"]), seed
        elif meta["period_mode"] == "column_wise":
            assert _minimal_periods(out) == (meta["period_length"], out.shape[1]), seed
        else:
            assert _minimal_periods(out) == (meta["tile_height"], meta["tile_width"]), seed
    _verify("117", check)


def test_task_117_distribution_and_prompt_clarity():
    modes = set()
    for seed in range(N_SEEDS):
        ex = generate("117", seed)
        meta = ex["metadata"]
        modes.add(meta["period_mode"])
        prompt = get_prompt("117", meta).lower()
        assert str(meta["unknown_color"]) in prompt
        assert "zero-indexed" in prompt
        assert "known cells" in prompt
        assert "preserve" in prompt
        assert "output the full repaired grid" in prompt
        if meta["period_mode"] == "row_wise":
            assert "each row independently" in prompt
            assert "column remainder" in prompt
            assert f"modulo {meta['period_length']}" in prompt
        elif meta["period_mode"] == "column_wise":
            assert "each column independently" in prompt
            assert "row remainder" in prompt
            assert f"modulo {meta['period_length']}" in prompt
        else:
            assert "whole grid" in prompt
            assert f"height {meta['tile_height']}" in prompt
            assert f"width {meta['tile_width']}" in prompt
            assert f"row modulo {meta['tile_height']}" in prompt
            assert f"column modulo {meta['tile_width']}" in prompt
    assert modes == {"row_wise", "column_wise", "full_grid"}


def test_task_119():
    def check(seed, grid, out, meta):
        expected = np.full_like(grid, meta["background_color"])
        for r, c in _pattern_cells(meta["pattern_type"], *grid.shape, meta):
            expected[r, c] = meta["pattern_color"]
        assert np.any((grid != expected) & (expected == meta["pattern_color"])), seed
        assert out.tolist() == expected.tolist(), seed
    _verify("119", check)


def test_task_155():
    def check(seed, grid, out, meta):
        expected = grid.copy() if meta["output_mode"] == "preserve_scene" else np.full_like(grid, meta["background_color"])
        cells = _interface_cells(grid, meta)
        assert cells, seed
        for r, c in cells:
            expected[r, c] = meta["contact_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("155", check)
    styles = {generate("155", seed)["metadata"].get("contact_style") for seed in range(100)}
    assert "partial_staggered" in styles, styles


def test_task_168():
    def check(seed, grid, out, meta):
        expected = _replace_runs(grid, meta)
        assert out.tolist() == expected.tolist(), seed
    _verify("168", check)
    run_counts = {generate("168", seed)["metadata"]["run_count"] for seed in range(100)}
    assert len(run_counts) > 1, run_counts


def test_task_169():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        marks = _distance_marks(grid, meta)
        assert marks, seed
        for r, c in marks:
            expected[r, c] = meta["marker_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("169", check)
    variants = {generate("169", seed)["metadata"].get("shape_variant") for seed in range(100)}
    assert "corner_notched" in variants, variants


def test_task_174():
    def check(seed, grid, out, meta):
        points = [tuple(p) for p in np.argwhere(grid == meta["corner_color"])]
        assert len(points) == 3, seed
        rows = sorted({r for r, _ in points})
        cols = sorted({c for _, c in points})
        assert len(rows) == 2 and len(cols) == 2, seed
        all_corners = {(r, c) for r in rows for c in cols}
        missing = list(all_corners - set(points))
        assert len(missing) == 1, seed
        expected = grid.copy()
        expected[missing[0]] = meta["corner_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("174", check)


def test_task_177():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        comps = _components(grid, meta["background_color"], meta["target_color"])
        assert len(comps) >= 2, seed
        for comp in comps:
            for r, c in _directional_boundary(comp["cells"], meta["direction"]):
                expected[r, c] = meta["edge_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("177", check)


def test_task_181():
    def check(seed, grid, out, meta):
        expected = grid.copy()
        if meta["axis"] == "rows":
            selected = np.any(grid == meta["trigger_color"], axis=1)
            assert np.any(selected), seed
            expected[selected, :] = meta["band_color"]
        else:
            selected = np.any(grid == meta["trigger_color"], axis=0)
            assert np.any(selected), seed
            expected[:, selected] = meta["band_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("181", check)


def test_task_182():
    def check(seed, grid, out, meta):
        counts = _line_counts(grid, meta)
        best = max(counts) if meta["extremum"] == "maximum" else min(counts)
        assert counts.count(best) == 1, seed
        chosen = _selected_line(grid, meta)
        expected = grid.copy()
        if meta["axis"] == "row":
            if meta["output_mode"] == "recolor_entire_line":
                expected[chosen, :] = meta["highlight_color"]
            else:
                expected[chosen, 0] = meta["highlight_color"]
                expected[chosen, -1] = meta["highlight_color"]
        else:
            if meta["output_mode"] == "recolor_entire_line":
                expected[:, chosen] = meta["highlight_color"]
            else:
                expected[0, chosen] = meta["highlight_color"]
                expected[-1, chosen] = meta["highlight_color"]
        assert out.tolist() == expected.tolist(), seed
    _verify("182", check)


def _hw_span(task_id, n=120):
    hs, ws = [], []
    for seed in range(n):
        g = np.array(generate(task_id, seed)["input"], dtype=np.uint8)
        hs.append(g.shape[0])
        ws.append(g.shape[1])
    return min(hs), max(hs), min(ws), max(ws)


def test_grid_size_envelope_widened():
    # Tasks whose size envelope was widened in the second review: at least one axis must span > 8
    # (decorrelated H/W), shifted down toward small ARC grids while staying within the token budget.
    for task_id in ["119", "155", "168", "169", "174", "177", "181", "182"]:
        h_lo, h_hi, w_lo, w_hi = _hw_span(task_id)
        assert (h_hi - h_lo) > 8 or (w_hi - w_lo) > 8, (task_id, h_lo, h_hi, w_lo, w_hi)
        assert h_lo <= 8 and w_lo <= 8, (task_id, "floor too high", h_lo, w_lo)
        assert h_hi <= 16 and w_hi <= 16, (task_id, "exceeds grid->grid budget cap", h_hi, w_hi)


def test_background_style_variety():
    # Tasks that gained a solid+patterned background mix expose a background_style; both a solid and
    # at least one patterned style must appear across seeds.
    for task_id in ["155", "168", "169", "174", "177", "181"]:
        styles = {generate(task_id, seed)["metadata"].get("background_style") for seed in range(120)}
        assert "solid" in styles, (task_id, styles)
        assert len(styles - {"solid"}) >= 1, (task_id, styles)


def test_non_semantic_colors_appear():
    # The contact / distance / boundary tasks now mix in distractor or textured cells whose color is
    # none of the rule colors; confirm such non-semantic colors show up without breaking the rule.
    role_keys = {
        "155": ("background_color", "object_a_color", "object_b_color", "contact_color"),
        "169": ("background_color", "object_a_color", "object_b_color", "marker_color"),
        "177": ("background_color", "target_color", "edge_color"),
        "174": ("background_color", "corner_color"),
    }
    for task_id, keys in role_keys.items():
        saw_extra = False
        for seed in range(120):
            ex = generate(task_id, seed)
            meta = ex["metadata"]
            roles = {int(meta[k]) for k in keys}
            present = {int(v) for row in ex["input"] for v in row}
            if present - roles:
                saw_extra = True
                break
        assert saw_extra, (task_id, "no non-semantic colors ever appeared")


if __name__ == "__main__":
    tests = [
        test_task_116,
        test_task_117,
        test_task_119,
        test_task_155,
        test_task_168,
        test_task_169,
        test_task_174,
        test_task_177,
        test_task_181,
        test_task_182,
        test_grid_size_envelope_widened,
        test_background_style_variety,
        test_non_semantic_colors_appear,
    ]
    for fn in tests:
        fn()
        print(f"  PASS  {fn.__name__}")
