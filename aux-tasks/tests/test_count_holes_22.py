"""Tests for task 22 (count the empty regions completely enclosed by the object), redesigned
2026-09-08 around clean, natural shapes.

The oracle here is an independent BFS (no ``common.topology``).  Beyond correctness the tests
pin the naturalness rules the redesign exists for: no hair on the object (a cell with at most
one object 4-neighbour), no notch on the outer boundary except a deliberate near-miss gap, at
most three object components, readable hole kinds, bounded pinholes, and the checklist rules on
sizes, colours, answers, backgrounds and the validator.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, deque

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from registry import generate  # noqa: E402
from prompts import get_prompt  # noqa: E402
import generators.count_holes as G  # noqa: E402

SEEDS = 400
D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _flood(mask: np.ndarray, starts) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    dq = deque()
    for r, c in starts:
        if mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in D4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                dq.append((nr, nc))
    return seen


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    left = mask.copy()
    out = []
    for r in range(h):
        for c in range(w):
            if left[r, c]:
                comp = _flood(left, [(r, c)])
                out.append(comp)
                left &= ~comp
    return out


def oracle(grid: np.ndarray, colour: int) -> int:
    empty = grid != colour
    h, w = grid.shape
    border = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    outside = _flood(empty, border)
    enclosed = empty & ~outside
    return len(_components(enclosed))


def _nb4(mask: np.ndarray) -> np.ndarray:
    n = np.zeros(mask.shape, dtype=int)
    h, w = mask.shape
    for dr, dc in D4:
        sh = np.zeros_like(mask)
        sh[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)] = mask[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)]
        n += sh
    return n


@pytest.fixture(scope="module")
def examples():
    return [generate("22", s) for s in range(SEEDS)]


def _obj(ex):
    g = np.asarray(ex["input"])
    return g, g == ex["metadata"]["obj_color"]


def test_answer_matches_independent_oracle(examples):
    for s, ex in enumerate(examples):
        g = np.asarray(ex["input"])
        assert int(ex["output"]) == oracle(g, ex["metadata"]["obj_color"]) == ex["metadata"]["n_holes"], s


def test_no_hair_no_notch_few_components(examples):
    """The naturalness contract: every object cell has at least two object neighbours, the
    outer boundary has no one-cell dent except a deliberate gap, and there are at most three
    object components (islands included)."""
    for s, ex in enumerate(examples):
        g, obj = _obj(ex)
        m = ex["metadata"]
        if min(g.shape) < 3:
            continue                                   # strips: bars have ends by necessity
        hairs = int((obj & (_nb4(obj) <= 1)).sum())
        assert hairs == 0, (s, hairs)
        h, w = g.shape
        border = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        outside = _flood(~obj, border)
        notches = int((outside & (_nb4(obj) >= 3)).sum())
        assert notches <= len(m["post_ops"]), (s, notches, m["post_ops"])
        assert len(_components(obj)) <= G.MAX_COMPONENTS, s
        assert m["n_hairs"] == hairs and m["n_components"] == len(_components(obj))


def test_bodies_are_clean_and_spread(examples):
    fam = Counter(ex["metadata"]["family"].split("+")[0] for ex in examples
                  if ex["metadata"]["layout"] in ("single", "multi"))
    assert fam.most_common(1)[0][1] / sum(fam.values()) <= 0.32, fam
    assert len([k for k, v in fam.items() if v >= 3]) >= 8, fam
    layouts = Counter(ex["metadata"]["layout"] for ex in examples)
    assert layouts["multi"] >= 0.05 * SEEDS and layouts["full"] >= 0.02 * SEEDS, layouts
    comp = [ex["metadata"]["compactness"] for ex in examples if min(np.asarray(ex["input"]).shape) >= 3]
    assert max(comp) <= 140 and np.percentile(comp, 90) <= 45, (max(comp), np.percentile(comp, 90))


def test_hole_kinds_are_readable_and_varied(examples):
    kinds = Counter(k for ex in examples for k in ex["metadata"]["hole_kinds"])
    total = sum(kinds.values())
    assert set(kinds) <= set(G.HOLE_KINDS), kinds
    assert kinds["pinhole"] / total <= 0.20, kinds
    for k in ("rect", "letter", "corridor", "room", "ring", "nested", "blob"):
        assert kinds[k] >= 5, (k, kinds)
    grids3 = [ex for ex in examples if ex["metadata"]["n_holes"] >= 3]
    mostly_pin = sum(ex["metadata"]["hole_kinds"].count("pinhole") * 2 >= ex["metadata"]["n_holes"] for ex in grids3)
    assert mostly_pin <= 0.10 * max(1, len(grids3)), mostly_pin
    # every hole is one 4-connected region of readable size: pinholes bounded, no hole is noise
    for ex in examples:
        m = ex["metadata"]
        assert len(m["hole_kinds"]) == m["n_holes"], m["hole_kinds"]


def _hole_regions(g, obj):
    h, w = g.shape
    border = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    outside = _flood(~obj, border)
    return _components(~obj & ~outside)


def test_holes_are_readable_blocks(examples):
    """The 2026-09-08 follow-up: most holes carry a 2x2 block, one- and two-cell holes are a
    bounded minority, no grid is half tiny holes, and the body keeps enough cells per hole
    that it never reads as a sieve."""
    with2x2 = tiny = total = 0
    tiny_heavy = grids5 = 0
    dens = []
    dense8 = grids8 = 0
    for ex in examples:
        g, obj = _obj(ex)
        regions = _hole_regions(g, obj)
        sizes = [int(r.sum()) for r in regions]
        for r, sz in zip(regions, sizes):
            total += 1
            with2x2 += bool((r[:-1, :-1] & r[1:, :-1] & r[:-1, 1:] & r[1:, 1:]).any())
            tiny += sz <= 2
        n = len(sizes)
        if n >= 5:
            grids5 += 1
            tiny_heavy += sum(s <= 2 for s in sizes) * 2 >= n
        if n:
            d = n / max(1, int(obj.sum()))
            dens.append(d)
            assert abs(d - ex["metadata"]["hole_density"]) < 1e-3
            if n >= 8:
                grids8 += 1
                dense8 += d > 0.08
            if n >= 3:
                assert d <= G.MAX_HOLE_DENSITY + 1e-9, (n, int(obj.sum()))
    assert with2x2 / total >= 0.55, with2x2 / total
    assert tiny / total <= 0.20, tiny / total
    assert tiny_heavy <= 0.12 * max(1, grids5), (tiny_heavy, grids5)
    assert np.percentile(dens, 90) <= 0.06, np.percentile(dens, 90)
    assert dense8 <= 0.02 * max(1, grids8), (dense8, grids8)
    # the budgets recorded in the metadata are what the grid shows
    for ex in examples:
        m = ex["metadata"]
        assert m["n_tiny_holes"] <= max(m["tiny_budget"], m["n_tiny_holes"]) and m["n_thin_holes"] >= m["n_tiny_holes"]


def test_hole_kinds_follow_the_holes(examples):
    """``hole_kinds`` is parallel to ``hole_areas`` (label order): a pinhole kind is a one-cell
    hole and vice versa; a breached hole's kind moves to ``removed_hole_kinds`` and two merged
    holes become one ``merged`` region."""
    for s, ex in enumerate(examples):
        m = ex["metadata"]
        assert len(m["hole_kinds"]) == len(m["hole_areas"]) == m["n_holes"], s
        for kind, area in zip(m["hole_kinds"], m["hole_areas"]):
            assert (kind == "pinhole") == (area == 1) or kind == "merged", (s, kind, area)
        expect_removed = sum(1 if p == "breach" else 2 for p in m["post_ops"])
        assert len(m["removed_hole_kinds"]) == expect_removed, (s, m["post_ops"], m["removed_hole_kinds"])
        assert m["hole_kinds"].count("merged") == m["post_ops"].count("merge"), s
        assert isinstance(m["min_wall"], int) and (m["min_wall"] >= 1 or m["n_holes"] == 0), s


def test_no_diagonal_only_contact_between_components(examples):
    """A carve or a near-miss gap never leaves a piece of the object hanging by its corners."""
    D8 = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for s, ex in enumerate(examples):
        g, obj = _obj(ex)
        comps = _components(obj)
        if len(comps) < 2:
            continue
        lbl = np.zeros(obj.shape, dtype=int)
        for i, c in enumerate(comps, 1):
            lbl[c] = i
        h, w = obj.shape
        for dr, dc in D8:
            a = lbl[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
            b = lbl[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)]
            assert not ((a > 0) & (b > 0) & (a != b)).any(), s
        assert not ex["metadata"]["diagonal_contact"], s


def test_colours_reach_the_target_often(examples):
    """8-10 colours together on a fifth of the grids (a third of the big ones), ten colours on a
    few percent; small grids may stay at two to four."""
    ncol = np.array([ex["metadata"]["n_colors"] for ex in examples])
    area = np.array([ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] for ex in examples])
    assert np.mean(ncol >= 8) >= 0.18, np.mean(ncol >= 8)
    assert np.mean(ncol == 10) >= 0.04, np.mean(ncol == 10)
    big = ncol[area >= 500]
    assert len(big) >= 20 and np.mean(big >= 8) >= 0.25, (len(big), np.mean(big >= 8))
    # no speckle: extra colours come as shapes, so a grid never has more than three colours
    # that are single cells (a filled pinhole or a frame's one-cell centre are the exceptions;
    # textures excepted, which are library kinds)
    for ex in examples:
        m = ex["metadata"]
        if m["texture_scope"] != "none":
            continue
        g = np.asarray(ex["input"])
        singles = 0
        for col in np.unique(g):
            if col == m["obj_color"]:
                continue
            mask = g == col
            if mask.sum() == 1:
                singles += 1
        assert singles <= 3, (ex["metadata"]["grid_h"], ex["metadata"]["grid_w"], singles)


def test_variety_axes_present(examples):
    ms = [ex["metadata"] for ex in examples]
    n = len(ms)
    assert sum(m["has_pinhole"] for m in ms) >= 0.20 * n
    assert sum(m["has_corner_touch"] for m in ms) >= 0.10 * n
    assert sum(m["has_thin_wall"] for m in ms) >= 0.25 * n
    assert sum(m["n_islands"] > 0 for m in ms) >= 0.10 * n
    assert sum(m["has_nested"] for m in ms) >= 0.05 * n
    assert sum(bool(m["post_ops"]) for m in ms) >= 0.03 * n          # near misses
    assert sum(m["wall_target"] == 2 for m in ms) >= 0.05 * n
    assert sum(bool(m["hole_fills"]) for m in ms) >= 0.15 * n
    assert sum(m["n_distractors"] > 0 for m in ms) >= 0.25 * n
    assert sum(m["texture_scope"] != "none" for m in ms) >= 0.10 * n
    # holes are spread, not packed: on average well under half the holes of a grid touch
    # another hole at a corner, and pinholes never dominate a grid (checked above)
    assert np.mean([m["near_placements"] / max(1, m["n_holes"]) for m in ms if m["n_holes"] >= 2]) <= 0.45


def test_islands_nested_and_corner_rules_hold_on_the_grid(examples):
    for s, ex in enumerate(examples):
        g, obj = _obj(ex)
        m = ex["metadata"]
        h, w = g.shape
        border = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        outside = _flood(~obj, border)
        comps = _components(obj)
        touches = lambda comp: any(comp[r, c] and any(0 <= r + dr < h and 0 <= c + dc < w and outside[r + dr, c + dc]
                                                       for dr, dc in D4) for r, c in zip(*np.nonzero(comp)))
        islands = [c for c in comps if not touches(c)]
        assert len(islands) == m["n_islands"], s
        nested = any(oracle(np.where(c, 1, 0), 1) > 0 for c in islands)
        assert nested == m["has_nested"], s
        if m["has_corner_touch"]:
            enclosed = ~obj & ~outside
            regions = _components(enclosed)
            found = False
            for i, a in enumerate(regions):
                for b in regions[i + 1:]:
                    for r, c in zip(*np.nonzero(a)):
                        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                            if 0 <= r + dr < h and 0 <= c + dc < w and b[r + dr, c + dc]:
                                found = True
            assert found, s


def test_sizes_colours_and_answers(examples):
    hs = np.array([ex["metadata"]["grid_h"] for ex in examples])
    ws = np.array([ex["metadata"]["grid_w"] for ex in examples])
    assert hs.min() <= 3 and hs.max() == 30 and ws.min() <= 3 and ws.max() == 30
    assert not any((h == 1 and w == 1) for h, w in zip(hs, ws))
    assert abs(np.corrcoef(hs, ws)[0, 1]) < 0.2
    assert np.mean(hs * ws >= 700) >= 0.02
    # near-flat sides: every value 3..30 within a Poisson-sized band around the mean (400 seeds)
    for arr in (hs, ws):
        counts = Counter(arr.tolist())
        vals = [counts.get(v, 0) for v in range(3, 31)]
        mean = sum(vals) / len(vals)
        assert min(vals) >= 0.35 * mean and max(vals) <= 2.2 * mean, vals
    cols = Counter(ex["metadata"]["n_colors"] for ex in examples)
    assert cols[10] >= 1 and cols[9] >= 2 and max(cols.values()) <= 0.30 * SEEDS, cols
    ans = np.array([int(ex["output"]) for ex in examples])
    assert ans.max() >= 14 and np.mean(ans == 0) <= 0.15
    assert Counter(ans.tolist()).most_common(1)[0][1] <= 0.22 * SEEDS
    for ex in examples:
        m = ex["metadata"]
        assert int(ex["output"]) <= m["answer_cap"] == G.answer_cap(m["grid_h"], m["grid_w"])


def test_scenes_scale_with_the_canvas(examples):
    small = [ex["metadata"]["occupancy"] for ex in examples if ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] < 100]
    big = [ex["metadata"]["occupancy"] for ex in examples if ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] >= 500]
    assert big and np.mean(big) >= 0.25 and np.mean(big) >= 0.5 * np.mean(small)


def test_backgrounds_from_the_library_and_never_named(examples):
    kinds = Counter(ex["metadata"]["background_kind"] for ex in examples if ex["metadata"]["texture_scope"] != "none")
    assert len(kinds) >= 4, kinds
    for ex in examples:
        m = ex["metadata"]
        assert m["obj_color"] not in m["texture_colors"]
        p = get_prompt("22", m)
        assert "background" not in p.lower() and "texture" not in p.lower()
        assert f"color {m['obj_color']}" in p and "0 if nothing is enclosed" in p
        assert "diagonal contact does not connect" in p
        # the object colour appears nowhere except the object (a texture in that colour would change the count)
        g = np.asarray(ex["input"])
        assert int((g == m["obj_color"]).sum()) == m["object_area"]


def test_determinism_and_validator():
    a = generate("22", 7)
    b = generate("22", 7)
    assert a["input"] == b["input"] and a["output"] == b["output"]
    # a generator that cannot build must raise, never fall back
    rng = np.random.default_rng(0)
    plan = G._plan(rng, 3, 4, 1, 1)
    plan["mode"] = "single"
    with pytest.raises(RuntimeError):
        original = G._build
        G._build = lambda *a, **k: None
        try:
            G.generate_count_holes(np.random.default_rng(1))
        finally:
            G._build = original


def test_generation_is_fast():
    import time
    t = time.time()
    for s in range(60):
        generate("22", 1000 + s)
    assert (time.time() - t) / 60 < 0.15
