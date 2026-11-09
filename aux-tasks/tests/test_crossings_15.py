"""Tests for task 15 (generators/lines_diamonds.py): count the cells on two or more lines.

Independent oracle written from the prompt: a line is a colour whose cells lie on one
straight H / V / 45-degree segment of at least five cells, every gap along it being a
non-background cell (a crossing showing the other line's colour); the answer is the
number of cells on two or more such segments.  Also: size envelope, answers and line
counts growing with the canvas, sparse textures only (isolated, clear of every line),
occupancy on large grids, byte-identity of every sibling in the four reworked files,
determinism, prompt, performance.  Follow-up (982e4e8 verification): the geometry guard
(no phantom diagonal crossings, parallel lines >= 2 apart, no line end touching another
line, >= 2 cells past every crossing), the answer histogram covering 7 / 11 / 13 / 14 /
17 / 18 / 19, and sides 7..10 as dense as the rest.
"""
from __future__ import annotations

import hashlib
import json
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
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "r4_siblings_11_12_14_15.json")
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("15", s) for s in range(SWEEP))
    return _CACHE


def oracle(grid: np.ndarray, bg: int) -> int:
    count = Counter(cell for seg in oracle_segments(grid, bg) for cell in seg)
    return sum(1 for v in count.values() if v >= 2)


def oracle_segments(grid: np.ndarray, bg: int) -> list[set]:
    segments = []
    for colour in np.unique(grid):
        if int(colour) == bg:
            continue
        cells = [(int(r), int(c)) for r, c in np.argwhere(grid == colour)]
        if len(cells) < 3:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r0 == r1:
            line = {(r0, c) for c in range(c0, c1 + 1)}
        elif c0 == c1:
            line = {(r, c0) for r in range(r0, r1 + 1)}
        elif r1 - r0 == c1 - c0 and all(r - r0 == c - c0 for r, c in cells):
            line = {(r0 + i, c0 + i) for i in range(r1 - r0 + 1)}
        elif r1 - r0 == c1 - c0 and all(r - r0 == c1 - c for r, c in cells):
            line = {(r0 + i, c1 - i) for i in range(r1 - r0 + 1)}
        else:
            continue
        if len(line) < 5 or not set(cells) <= line:
            continue
        if any(int(grid[r, c]) == bg for r, c in line - set(cells)):
            continue
        segments.append(line)
    return segments


def segment_kind(seg: set) -> tuple[str, int]:
    """('h', row) / ('v', col) / ('d', r - c) / ('u', r + c) of a straight segment."""
    cells = sorted(seg)
    rs = {r for r, _ in cells}
    cs = {c for _, c in cells}
    if len(rs) == 1:
        return "h", cells[0][0]
    if len(cs) == 1:
        return "v", cells[0][1]
    if all(r - c == cells[0][0] - cells[0][1] for r, c in cells):
        return "d", cells[0][0] - cells[0][1]
    return "u", cells[0][0] + cells[0][1]


def test_oracle_matches_every_answer():
    for s, ex in enumerate(examples()):
        grid = np.array(ex["input"])
        assert oracle(grid, ex["metadata"]["bg_color"]) == int(ex["output"]), f"seed {s}"
        assert int(ex["output"]) == ex["metadata"]["intersection_count"] >= 2


def test_sizes_and_scaling():
    exs = examples()
    hs = [len(ex["input"]) for ex in exs]
    ws = [len(ex["input"][0]) for ex in exs]
    assert min(hs) == 7 and max(hs) == 30 and min(ws) == 7 and max(ws) == 30
    assert np.mean([h == w for h, w in zip(hs, ws)]) <= 0.08
    lines = Counter(ex["metadata"]["line_count"] for ex in exs)
    assert max(lines) == 9 and lines[8] + lines[9] >= 0.04 * len(exs), lines
    answers = Counter(int(ex["output"]) for ex in exs)
    assert max(answers) >= 16 and sum(v for k, v in answers.items() if k >= 13) >= 0.05 * len(exs), answers
    assert max(answers.values()) / len(exs) <= 0.25
    big = [ex for ex in exs if len(ex["input"]) * len(ex["input"][0]) >= 600]
    small = [ex for ex in exs if len(ex["input"]) * len(ex["input"][0]) < 150]
    assert np.median([ex["metadata"]["line_count"] for ex in big]) >= np.median([ex["metadata"]["line_count"] for ex in small]) + 1

    def occupancy(ex):
        g = np.array(ex["input"])
        return float((g != ex["metadata"]["bg_color"]).mean())
    # lines are one cell wide: nine full-span lines on 30x30 are 30% of the cells, so the bar is 0.18
    assert np.median([occupancy(ex) for ex in big]) >= 0.18, np.median([occupancy(ex) for ex in big])


def test_textures_sparse_and_clear_of_lines():
    exs = examples()
    groups = Counter(ex["metadata"]["texture_group"] for ex in exs)
    assert 0.18 <= groups["sparse"] / len(exs) <= 0.32 and groups.get("solid", 0) == 0, groups
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        if m["texture_group"] is None:
            continue
        grid = np.array(ex["input"])
        assert texture_cells_isolated(grid, m["texture_colors"]), f"seed {s}"
        assert not set(m["texture_colors"]) & set(m["line_colors"])
        tex = np.isin(grid, m["texture_colors"])
        lines = np.isin(grid, m["line_colors"])
        H, W = grid.shape
        for r, c in np.argwhere(tex):
            assert not lines[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any(), f"seed {s}: texture touches a line"
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in exs)
    assert max(seen) == 10 and sum(v for k, v in seen.items() if k >= 8) / len(exs) >= 0.15, seen


def test_siblings_byte_identical():
    """Retain sibling pins, refreshing only separately approved task rebuilds."""
    expected = json.load(open(FIXTURE))
    for tid, hashes in expected.items():
        for s, want in enumerate(hashes):
            got = hashlib.sha1(json.dumps(generate(tid, s), sort_keys=True).encode()).hexdigest()
            assert got == want, f"task {tid} seed {s} changed"


def test_determinism_prompt_performance():
    exs = examples()
    for s in range(200):
        assert generate("15", s) == exs[s]
    p = get_prompt("15", exs[0]["metadata"], color_palette=list("ABCDEFGHIJ"))
    assert "Ignore isolated non-line distractor objects" in p and "Count the cells that lie on two or more lines" in p
    assert not re.search(r"color \d", p) and "background" not in p
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate("15", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 4.5  # 2.6 ms alone; headroom for parallel load, statistics.mean(times)


def test_geometry_guard():
    """No phantom crossing (opposite-slope diagonals meeting between cells inside both
    segments), parallel lines with overlapping extents >= 2 apart, no line end 8-adjacent
    to a line it does not share, every line >= 2 cells past every crossing."""
    for s, ex in enumerate(examples()):
        grid = np.array(ex["input"])
        segs = oracle_segments(grid, ex["metadata"]["bg_color"])
        assert len(segs) == ex["metadata"]["line_count"], f"seed {s}"
        assert ex["metadata"]["min_parallel_gap"] == 2
        info = [segment_kind(seg) for seg in segs]
        rows = [(min(r for r, _ in seg), max(r for r, _ in seg)) for seg in segs]
        cols = [(min(c for _, c in seg), max(c for _, c in seg)) for seg in segs]
        for i in range(len(segs)):
            ki, bi = info[i]
            for j in range(i + 1, len(segs)):
                kj, bj = info[j]
                if ki == kj:
                    along = (cols[i][0] <= cols[j][1] and cols[j][0] <= cols[i][1]) if ki == "h" else \
                        (rows[i][0] <= rows[j][1] and rows[j][0] <= rows[i][1])
                    assert not (abs(bi - bj) < 2 and along), f"seed {s}: adjacent parallel lines"
                elif {ki, kj} == {"d", "u"}:
                    bd, bu = (bi, bj) if ki == "d" else (bj, bi)
                    if (bd + bu) % 2:
                        r2, c2 = bd + bu, bu - bd
                        inside = all(2 * rows[x][0] < r2 < 2 * rows[x][1] and 2 * cols[x][0] < c2 < 2 * cols[x][1]
                                     for x in (i, j))
                        assert not inside, f"seed {s}: phantom crossing between cells"
        for i, seg in enumerate(segs):
            ordered = sorted(seg) if info[i][0] != "u" else sorted(seg, key=lambda rc: (rc[0], -rc[1]))
            for p in (ordered[0], ordered[-1]):
                for j, other in enumerate(segs):
                    if j == i or p in other:
                        continue
                    near = any((p[0] + dr, p[1] + dc) in other for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if (dr, dc) != (0, 0))
                    assert not near, f"seed {s}: line end touches another line"
            crossings = [k for k, cell in enumerate(ordered) if any(cell in o for j, o in enumerate(segs) if j != i)]
            if crossings:
                assert min(crossings) >= 2 and max(crossings) <= len(ordered) - 3, f"seed {s}: line ends at a crossing"


def test_answer_coverage_and_small_sides():
    exs = examples()
    answers = Counter(int(ex["output"]) for ex in exs)
    assert all(answers[k] > 0 for k in range(2, 21)), answers
    # the counts no plain lattice produces: >= 2.5% each over 10k seeds, 2% here for noise
    for k in (7, 11, 13, 14, 17, 18, 19):
        assert answers[k] / len(exs) >= 0.02, (k, answers)
    sides = Counter(len(ex["input"]) for ex in exs) + Counter(len(ex["input"][0]) for ex in exs)
    mean_big = np.mean([sides[k] for k in range(11, 31)])
    for k in (7, 8, 9, 10):
        assert sides[k] >= 0.7 * mean_big, (k, sides)
