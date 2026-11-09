"""Tests for task 15a (generators/lines_diamonds.py): count line crossings with multiplicity.

Independent oracle written from the prompt: a line is a colour whose cells lie on one
straight H / V / 45-degree segment of at least five cells, every gap along it being a
non-background cell; the answer is the number of pairs of lines sharing a cell, i.e. the
sum of C(k, 2) over the cells crossed by k lines.  Also: the geometry guard (no phantom
diagonal crossing, parallel lines >= 2 apart, no line end touching another line, >= 2
cells past every crossing), the multiplicity rule exercised (cells crossed by three or
four lines on a bounded share of examples), size envelope, lines and answers growing with
the canvas, sparse textures only (isolated, clear of every line), occupancy on large
grids, answer coverage 2..20, determinism, prompt under a letter palette, performance.
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
        _CACHE.extend(generate("15a", s) for s in range(SWEEP))
    return _CACHE


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


def oracle(grid: np.ndarray, bg: int) -> int:
    count = Counter(cell for seg in oracle_segments(grid, bg) for cell in seg)
    return sum(v * (v - 1) // 2 for v in count.values())


def segment_kind(seg: set) -> tuple[str, int]:
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
        m = ex["metadata"]
        assert oracle(grid, m["bg_color"]) == int(ex["output"]), f"seed {s}"
        assert int(ex["output"]) == m["pairwise_intersection_count"] >= 2
        count = Counter(cell for seg in oracle_segments(grid, m["bg_color"]) for cell in seg)
        assert sum(1 for v in count.values() if v >= 2) == m["intersection_cell_count"], f"seed {s}"
        assert max(count.values()) == m["max_lines_through_a_cell"], f"seed {s}"
        assert sum(1 for v in count.values() if v >= 3) == m["cells_with_three_or_more_lines"], f"seed {s}"


def test_multiplicity_rule_is_exercised():
    """Cells crossed by three or four lines appear on a bounded share of examples, and the
    answer there is the pairwise count, not the cell count."""
    exs = examples()
    multi = [ex for ex in exs if ex["metadata"]["cells_with_three_or_more_lines"] > 0]
    share = len(multi) / len(exs)
    assert 0.12 <= share <= 0.45, share
    assert any(ex["metadata"]["max_lines_through_a_cell"] == 4 for ex in multi)
    for ex in multi:
        m = ex["metadata"]
        assert m["pairwise_intersection_count"] > m["intersection_cell_count"]


def test_sizes_and_scaling():
    exs = examples()
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert hs.min() == 7 and ws.min() == 7 and hs.max() == 30 and ws.max() == 30
    assert np.mean(hs == ws) <= 0.08
    assert len({(h, w) for h, w in zip(hs, ws)}) >= 400
    sides = Counter(hs.tolist()) + Counter(ws.tolist())
    mean_big = np.mean([sides[k] for k in range(11, 31)])
    for k in (7, 8, 9, 10):
        assert sides[k] >= 0.7 * mean_big, (k, sides)
    areas = hs * ws
    small = [ex["metadata"]["line_count"] for ex, a in zip(exs, areas) if a < 200]
    big = [ex["metadata"]["line_count"] for ex, a in zip(exs, areas) if a >= 500]
    assert np.median(big) >= np.median(small) + 1, (np.median(small), np.median(big))
    occ_big = [float((np.array(ex["input"]) != ex["metadata"]["bg_color"]).mean()) for ex, a in zip(exs, areas) if a >= 700]
    assert len(occ_big) >= 40 and np.median(occ_big) >= 0.15, np.median(occ_big)


def test_colours_and_textures():
    exs = examples()
    ncol = Counter(len(np.unique(np.array(ex["input"]))) for ex in exs)
    assert max(ncol) >= 9 and max(ncol.values()) / len(exs) <= 0.35, ncol
    textured = 0
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        grid = np.array(ex["input"])
        for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
            assert key in m, key
        if m["texture_group"] is None:
            assert m["texture_colors"] == [] and not m["texture_named"]
            continue
        textured += 1
        assert m["texture_group"] == "sparse" and not m["texture_named"]
        assert texture_cells_isolated(grid, m["texture_colors"]), f"seed {s}"
        lines = set().union(*oracle_segments(grid, m["bg_color"]))
        for t in m["texture_colors"]:
            assert t != m["bg_color"] and t not in m["line_colors"] and t not in m["distractor_object_colors"]
            for r, c in np.argwhere(grid == t):
                assert not any((r + dr, c + dc) in lines for dr in (-1, 0, 1) for dc in (-1, 0, 1)), f"seed {s}"
    eligible = sum(1 for ex in exs if min(len(ex["input"]), len(ex["input"][0])) >= 8)
    assert 0.18 <= textured / eligible <= 0.40, textured / eligible


def test_geometry_guard():
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
                assert len(segs[i] & segs[j]) <= 1, f"seed {s}: two lines share more than one cell"
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


def test_answer_coverage():
    answers = Counter(int(ex["output"]) for ex in examples())
    assert all(answers[k] > 0 for k in range(2, 21)), answers
    assert max(answers.values()) / len(examples()) <= 0.30, answers
    for k in (7, 11, 13, 14, 17, 18, 19):
        assert answers[k] / len(examples()) >= 0.015, (k, answers)


def test_determinism_prompt_performance():
    exs = examples()
    for s in range(150):
        assert generate("15a", s) == exs[s]
    for ex in exs[:300]:
        p = get_prompt("15a", ex["metadata"], color_palette=LETTERS)
        assert "count every pair of them separately" in p and "Ignore isolated non-line distractor objects" in p
        assert not re.search(r"colou?r \d", p) and "background" not in p.lower()
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate("15a", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 4.5, statistics.mean(times)
