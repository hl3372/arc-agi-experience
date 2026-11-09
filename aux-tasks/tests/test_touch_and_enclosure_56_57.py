"""Regression checks for task 56 (does the object touch the target?) and task 57
(most frequent colour strictly inside a closed wall).

The oracles are plain-Python BFS over the visible grid, written independently
of ``common.topology`` / ``common.shapes`` so a bug in the shared libraries
cannot hide itself.  The sibling test replays hashes captured from the original
``generators/misc.py`` (before the 56/57 rewrite) for every other task in that
file.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402
from symbol_remap import palette_for_example  # noqa: E402

SWEEP = 2000
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sibling_hashes_misc_56_57.json")
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N8 = N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ---------------------------------------------------------------------------
# independent helpers
# ---------------------------------------------------------------------------

def _components(mask: np.ndarray, deltas) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                n += 1
                seen[r, c] = True
                dq = deque([(r, c)])
                while dq:
                    a, b = dq.popleft()
                    for dr, dc in deltas:
                        na, nb = a + dr, b + dc
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            dq.append((na, nb))
    return n


def _reach_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    dq: deque = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and passable[r, c]:
                seen[r, c] = True
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                dq.append((nr, nc))
    return seen


def _adjacent(a: np.ndarray, b: np.ndarray, deltas) -> bool:
    h, w = a.shape
    for r, c in zip(*np.nonzero(a)):
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and b[nr, nc]:
                return True
    return False


def _min_chebyshev(a: np.ndarray, b: np.ndarray) -> int:
    pa = np.argwhere(a)
    pb = np.argwhere(b)
    d = np.abs(pa[:, None, :] - pb[None, :, :]).max(axis=2)
    return int(d.min())


def _examples(task_id: str, n: int = SWEEP):
    for seed in range(n):
        ex = generate(task_id, seed)
        yield seed, ex, np.array(ex["input"], dtype=int), ex["metadata"]


def _share(counter: Counter, key, total: int) -> float:
    return counter[key] / total


# ---------------------------------------------------------------------------
# task 56
# ---------------------------------------------------------------------------

def _sweep56():
    rows = []
    for seed, ex, grid, meta in _examples("56"):
        rows.append((seed, ex, grid, meta))
    return rows


def test_task56_answer_matches_independent_oracle_and_prompt_states_the_rule():
    for seed, ex, grid, meta in _examples("56"):
        h, w = grid.shape
        assert 1 <= h <= 30 and 1 <= w <= 30, seed
        x, y, mode = meta["object_color"], meta["target_color"], meta["contact_mode"]
        assert x != y and mode in (4, 8), seed
        gx, gy = grid == x, grid == y
        # exactly one object of each colour (4-connected), nothing else in those colours
        assert _components(gx, N4) == 1 and _components(gy, N4) == 1, seed
        assert (meta["target_kind"] == "cell") == (int(gy.sum()) == 1), seed
        touching = _adjacent(gx, gy, N4 if mode == 4 else N8)
        assert ex["output"] == ("yes" if touching else "no"), seed
        # the placement bookkeeping agrees with the visible geometry
        cheb = _min_chebyshev(gx, gy)
        assert cheb == meta["min_chebyshev_distance"] >= 1, seed
        rel = meta["relation"]
        if rel == "edge":
            assert _adjacent(gx, gy, N4), seed
        elif rel == "corner":
            assert cheb == 1 and not _adjacent(gx, gy, N4), seed
        elif rel == "gap":
            assert cheb == 2, seed
        else:
            assert rel == "far" and cheb >= 3, seed
        prompt = get_prompt("56", meta)
        assert f"color {x} object" in prompt and f"color {y} target" in prompt, seed
        assert ("8-neighbor" in prompt) == (mode == 8), seed
        assert ("sharing an edge or a corner" in prompt) == (mode == 8), seed
        assert ("target object" in prompt) == (meta["target_kind"] == "object"), seed
        assert "Answer 'yes' or 'no'" in prompt, seed


def test_task56_distributions_cover_sizes_modes_shapes_and_backgrounds():
    rows = _sweep56()
    n = len(rows)
    assert .45 < np.mean([ex["output"] == "yes" for _, ex, _, _ in rows]) < .55
    modes = Counter(m["contact_mode"] for _, _, _, m in rows)
    assert .22 < modes[8]/n < .38
    relations = Counter(m["relation"] for _, _, _, m in rows)
    assert .22 < relations["corner"]/n < .33
    assert set(relations) == {"edge", "corner", "gap", "far"}
    shapes = np.array([g.shape for _, _, g, _ in rows])
    # MIN_SIDE = 5 floor, 2026-09-15 (user: small sides are often ambiguous).
    assert shapes.min() == 5 and shapes.max() == 30
    assert abs(np.corrcoef(shapes.T)[0, 1]) < .12
    assert .02 < np.mean(np.prod(shapes, axis=1) >= 700) < .08
    assert np.mean(shapes[:, 0] == shapes[:, 1]) < .06
    families = Counter(m["object_shape_family"] for _, _, _, m in rows)
    assert {"bar", "rectangle", "frame", "L", "T", "U", "H", "plus", "diamond", "chamfer", "staircase"} <= set(families)
    assert max(m["object_cells"] for _, _, _, m in rows) >= 150
    assert max(m["contact_length"] for _, _, _, m in rows) >= 10
    styles = Counter(m["background_style"] for _, _, _, m in rows)
    assert set(styles) == {"solid", "lattice", "stripes", "checker", "split", "diagonal"}
    assert .45 < styles["solid"]/n < .75
    assert set(len(np.unique(g)) for _, _, g, _ in rows) == set(range(3, 11))
    for seed, _, grid, m in rows:
        assert int((grid == m["object_color"]).sum()) == m["object_cells"], seed
        assert int((grid == m["target_color"]).sum()) == m["target_cells"], seed


def test_task56_symbol_mode_prompt_uses_palette_symbols_only():
    for seed in range(40):
        ex = generate("56", seed)
        prompt = get_prompt("56", ex["metadata"], color_palette=palette_for_example("56", seed))
        assert not re.search(r"color \d", prompt), prompt
        palette = palette_for_example("56", seed)
        assert palette[ex["metadata"]["object_color"]] in prompt
        assert palette[ex["metadata"]["target_color"]] in prompt


# ---------------------------------------------------------------------------
# task 57
# ---------------------------------------------------------------------------

def _oracle57(grid: np.ndarray, meta: dict) -> tuple[int, int, int, Counter]:
    """(winner, winner count, runner-up count, histogram) by a border flood blocked by the wall."""
    wall = meta["enclosure_color"]
    passable = grid != wall
    enclosed = passable & ~_reach_border(passable)
    assert _components(enclosed, N4) == 1
    hist = Counter(int(v) for v in grid[enclosed])
    if meta["ignore_background"]:
        hist.pop(meta["background_color"], None)
    else:
        assert meta["background_color"] not in hist
    ranked = sorted(hist.items(), key=lambda kv: -kv[1])
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0
    return ranked[0][0], top, second, hist


def test_task57_answer_matches_flood_fill_oracle_and_prompt_names_the_wall():
    for seed, ex, grid, meta in _examples("57"):
        h, w = grid.shape
        assert 3 <= h <= 30 and 3 <= w <= 30, seed
        wall, bg = meta["enclosure_color"], meta["background_color"]
        assert wall != bg, seed
        wmask = grid == wall
        assert _components(wmask, N8) == 1, seed
        winner, top, second, hist = _oracle57(grid, meta)
        assert ex["output"] == str(winner), seed
        assert winner not in (wall, bg), seed
        assert top - second >= 2, (seed, top, second)
        assert top == meta["winner_count"] and second == meta["runner_up_count"], seed
        assert len(hist) == meta["n_interior_colors"], seed
        assert sum(hist.values()) + round(meta["interior_bg_fraction"] * meta["interior_area"]) == meta["interior_area"], seed
        assert meta["interior_counts"][0] == [winner, top], seed
        prompt = get_prompt("57", meta)
        assert f"color {wall}" in prompt and "strictly inside" in prompt, seed
        assert "ignoring the background" in prompt and "ignoring cells of color" not in prompt, seed
        assert "up/down/left/right" in prompt, seed


def test_task57_distributions_cover_sizes_families_colors_and_outsides():
    rows = list(_examples("57"))
    n = len(rows)
    shapes = np.array([g.shape for _, _, g, _ in rows])
    # MIN_SIDE = 5 floor, 2026-09-15 (user: small sides are often ambiguous).
    assert shapes.min() == 5 and shapes.max() == 30
    assert abs(np.corrcoef(shapes.T)[0, 1]) < .15
    ncol = Counter(m["n_interior_colors"] for _, _, _, m in rows)
    assert set(ncol) == set(range(1, 9))
    assert (ncol[2]+ncol[3])/n > .40
    assert sum(v for k, v in ncol.items() if k >= 5)/n > .15
    families = Counter(m["enclosure_family"] for _, _, _, m in rows)
    assert set(families) == {"rectangle", "chamfer", "ellipse", "diamond", "L", "T", "plus", "step"}
    assert .30 < families["rectangle"]/n < .60
    assert np.mean([m["wall_thickness"] > 1 for _, _, _, m in rows]) > .05
    assert np.mean([m["interior_bg_fraction"] > 0 for _, _, _, m in rows]) > .25
    assert set(m["outside_style"] for _, _, _, m in rows) == {"solid", "lattice", "stripes", "checker", "diagonal"}
    assert set(m["interior_layout"] for _, _, _, m in rows) == {"scattered", "bands", "diagonal_bands", "blocks", "panels", "clusters"}
    assert max(m["interior_area"] for _, _, _, m in rows) >= 200
    assert .08 < np.mean([m["bbox_answer_differs"] for _, _, _, m in rows]) < .25
    assert set(len(np.unique(g)) for _, _, g, _ in rows) == set(range(3, 11))
    assert all(m["background_margin"] > 1.25 for _, _, _, m in rows)


def test_task57_symbol_mode_prompt_uses_palette_symbols_only():
    for seed in range(40):
        ex = generate("57", seed)
        palette = palette_for_example("57", seed)
        prompt = get_prompt("57", ex["metadata"], color_palette=palette)
        assert not re.search(r"color \d", prompt), prompt
        assert palette[ex["metadata"]["enclosure_color"]] in prompt
        assert "single color symbol" in prompt


# ---------------------------------------------------------------------------
# shared contracts
# ---------------------------------------------------------------------------

def test_determinism_and_cell_range():
    for tid in ("56", "57"):
        for seed in range(40):
            a = generate(tid, seed)
            b = generate(tid, seed)
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (tid, seed)
            grid = np.array(a["input"])
            assert grid.min() >= 0 and grid.max() <= 9 and isinstance(a["output"], str)


def test_generation_speed():
    for tid in ("56", "57"):
        times = []
        for seed in range(500):
            t0 = time.perf_counter()
            generate(tid, seed)
            times.append((time.perf_counter() - t0) * 1000.0)
        assert np.mean(times) <= 5.0, (tid, np.mean(times))
        assert np.percentile(times, 99) <= 30.0, (tid, np.percentile(times, 99))


def test_other_misc_tasks_are_byte_identical_to_original():
    with open(FIXTURE) as fh:
        expected = json.load(fh)
    assert set(expected) == {"26", "27", "28", "31", "35", "59", "80", "82", "83", "90", "91", "92"}
    for tid, hashes in expected.items():
        # Tasks 26–28 were rebuilt in the September pass and have independent
        # semantic/coverage suites.  The remaining siblings retain their pins.
        # Pins for 31/35 were refreshed after their approved September rebuild,
        # the 91/92 pins after their approved 2026-09-15 rebuilds and the 83 pins
        # after its approved 2026-09-16 background pass; all other fixture
        # entries still describe the original snapshot.
        if tid in {"26", "27", "28"}:
            continue
        for seed, digest in hashes.items():
            got = hashlib.sha1(json.dumps(generate(tid, int(seed)), sort_keys=True).encode()).hexdigest()
            assert got == digest, (tid, seed)
