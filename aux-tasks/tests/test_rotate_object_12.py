"""Tests for task 12 (generators/local_transforms.py): rotate one object in place.

Independent oracle: find the unique component of the target colour (own BFS at the
stated connectivity), check its bounding box is square, rotate its cells about the box
centre by the stated angle, set the vacated cells to the ground colour and compare with
the output.  Also: size envelope (5..30, no tilt), random background, near-flat colours
to 10, area-scaled scenes, irregular sparse textures only, no fallback signature,
determinism, prompt contents, performance.
"""
from __future__ import annotations

import os
import re
import statistics
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backgrounds import texture_cells_isolated  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
LETTERS = list("ABCDEFGHIJ")
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("12", s) for s in range(SWEEP))
    return _CACHE


def components(grid: np.ndarray, color: int, conn: int) -> list[set]:
    H, W = grid.shape
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if conn == 8:
        deltas += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r0, c0 in map(tuple, np.argwhere(grid == color)):
        if seen[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        seen[r0, c0] = True
        comp = set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc] and grid[rr, cc] == color:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)
    return comps


def oracle(ex: dict) -> np.ndarray:
    inp = np.array(ex["input"])
    m = ex["metadata"]
    conn = int(m["connectivity"])
    comps = components(inp, m["target_color"], conn)
    assert len(comps) == 1
    cells = comps[0]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    side = max(rs) - r0 + 1
    assert max(cs) - c0 + 1 == side, "target bounding box must be square"
    k = m["angle"] // 90
    # the ground colour: the most frequent colour once the target and texture dots are ignored
    vals, cnt = np.unique(inp, return_counts=True)
    bg = int(vals[np.argmax(cnt)])
    out = inp.copy()
    for r, c in cells:
        out[r, c] = bg
    for r, c in cells:
        rr, cc = r - r0, c - c0
        for _ in range(k):
            rr, cc = cc, side - 1 - rr
        out[r0 + rr, c0 + cc] = m["target_color"]
    return out


def test_oracle_reproduces_every_output():
    for s, ex in enumerate(examples()):
        assert np.array_equal(oracle(ex), np.array(ex["output"])), f"seed {s}"
        assert not np.array_equal(np.array(ex["input"]), np.array(ex["output"])), f"seed {s}"
        assert ex["metadata"]["background"] == int(np.bincount(np.array(ex["input"]).ravel(), minlength=10).argmax())


def test_sizes_have_no_tilt():
    exs = examples()
    hs = [len(ex["input"]) for ex in exs]
    ws = [len(ex["input"][0]) for ex in exs]
    assert min(hs) == 5 and max(hs) == 30 and min(ws) == 5 and max(ws) == 30
    sides = Counter(hs + ws)
    assert sides[5] + sides[6] + sides[7] >= 0.55 * (sides[28] + sides[29] + sides[30]), sides
    assert np.mean([h == w for h, w in zip(hs, ws)]) <= 0.08


def test_background_and_colours():
    exs = examples()
    assert len({ex["metadata"]["background"] for ex in exs}) == 10
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in exs)
    total = sum(seen.values())
    assert max(seen) == 10
    assert sum(v for k, v in seen.items() if k >= 8) / total >= 0.12, seen
    assert max(seen.values()) / total <= 0.35, seen


def test_scenes_scale_and_no_fallback():
    exs = examples()
    occ = {"small": [], "large": []}
    for ex in exs:
        g = np.array(ex["input"])
        bg = ex["metadata"]["background"]
        share = float((g != bg).mean())
        (occ["small"] if g.size < 150 else occ["large"] if g.size >= 600 else []).append(share)
    assert np.median(occ["large"]) >= 0.5 * np.median(occ["small"]), (np.median(occ["small"]), np.median(occ["large"]))
    assert np.median(occ["large"]) >= 0.10
    sides = Counter(ex["metadata"]["target_side"] for ex in exs)
    assert max(sides) >= 10 and sides[2] / len(exs) <= 0.30, sides
    assert max(ex["metadata"]["n_objects"] for ex in exs) >= 10
    # the old canned fallback was a 12x12 grid with background 0 and target colour 1
    fallback = sum(1 for ex in exs if len(ex["input"]) == 12 and len(ex["input"][0]) == 12
                   and ex["metadata"]["background"] == 0 and ex["metadata"]["target_color"] == 1
                   and ex["metadata"]["n_objects"] == 4)
    assert fallback <= 2


def test_textures_are_irregular_sparse_only():
    exs = examples()
    groups = Counter(ex["metadata"]["texture_group"] for ex in exs)
    assert 0.10 <= groups["sparse"] / len(exs) <= 0.30, groups
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        if m["texture_group"] is None:
            continue
        assert m["texture_kind"] in ("random_dots", "multi_speckle"), (s, m["texture_kind"])
        assert m["target_color"] not in m["texture_colors"]
        assert texture_cells_isolated(np.array(ex["input"]), m["texture_colors"]), f"seed {s}"


def test_determinism_prompt_performance():
    exs = examples()
    for s in range(200):
        assert generate("12", s) == exs[s]
    for ex in exs[:400]:
        m = ex["metadata"]
        p = get_prompt("12", m, color_palette=LETTERS)
        assert f"color {LETTERS[m['target_color']]} object ({m['connectivity']}-connected)" in p
        assert f"by {m['angle']} degrees clockwise" in p
        assert "Leave every other cell unchanged." in p and "become background" in p
        assert not re.search(r"color \d", p) and "background is" not in p and "background color" not in p
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate("12", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 3.0, statistics.mean(times)


def isolated(grid: np.ndarray) -> np.ndarray:
    """Cells with no 8-neighbour of their own colour (own implementation)."""
    H, W = grid.shape
    padded = np.full((H + 2, W + 2), -1, dtype=int)
    padded[1:-1, 1:-1] = grid
    same = np.zeros((H, W), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) != (0, 0):
                same |= padded[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc] == grid
    return ~same


def test_background_identifiable_on_every_example():
    """After isolated single cells are set aside the background is the strict plurality
    by >= 1.25x and owns the largest 4-connected region; the recorded margin matches."""
    for s, ex in enumerate(examples()):
        grid = np.array(ex["input"])
        bg = ex["metadata"]["background"]
        keep = ~isolated(grid)
        counts = np.bincount(grid[keep], minlength=10).astype(float)
        assert int(counts.argmax()) == bg, f"seed {s}"
        runner = float(np.sort(counts)[-2])
        margin = counts[bg] / max(1.0, runner)
        assert margin >= 1.25, f"seed {s}: margin {margin}"
        assert abs(margin - ex["metadata"]["background_margin"]) < 1e-3, f"seed {s}"
        largest_bg = max(len(c) for c in components(grid, bg, 4))
        for colour in set(np.unique(grid).tolist()) - {bg}:
            assert max(len(c) for c in components(grid, colour, 4)) < largest_bg, f"seed {s}: colour {colour}"


def test_target_variety():
    """Vocabulary shapes dominate (box walks <= 50%), the L-tromino is a small minority,
    and several families appear."""
    exs = examples()
    fams = Counter(ex["metadata"]["shape_family"] for ex in exs)
    assert fams["walk"] / len(exs) <= 0.50, fams
    assert len([f for f in fams if f != "walk" and fams[f] >= 0.02 * len(exs)]) >= 5, fams
    tromino = sum(1 for ex in exs if ex["metadata"]["target_side"] == 2 and ex["metadata"]["target_cells"] == 3)
    assert 0.02 <= tromino / len(exs) <= 0.15, tromino / len(exs)
    small = [ex for ex in exs if min(len(ex["input"]), len(ex["input"][0])) <= 7]
    assert any(ex["metadata"]["target_side"] >= 3 and ex["metadata"]["shape_family"] != "walk" for ex in small)
