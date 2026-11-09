"""Tests for tasks 23 (draw the bouncing diagonal) and 24 (count the reflections).

Rewritten 2026-09-07 after the review: task 23's marker used to overwrite scene cells (two
singleton colours, undecidable) and its prompt never said how the marker is identified;
task 24 drew high answers from thin-grid size pools, so the grid size leaked the answer.

Both tasks are checked with independent oracles written from the prompt alone: task 23 is
re-simulated from the unique marker cell, the named direction and the reflection count;
task 24's trace is followed from the marker and its direction changes are counted.
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators import paths as P  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SEEDS = range(400)
DIRS = {"NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)}


# ----------------------------------------------------------------------------- oracles
def oracle_23(inp: np.ndarray, start_color: int, trace_color: int, direction: str, n: int) -> np.ndarray:
    """Independent simulation: reflect exactly n times, then straight to the wall cell."""
    h, w = inp.shape
    marks = np.argwhere(inp == start_color)
    assert len(marks) == 1
    r, c = int(marks[0][0]), int(marks[0][1])
    dr, dc = DIRS[direction]
    out = inp.copy()
    cells = [(r, c)]
    bounces = 0
    assert 0 <= r + dr < h and 0 <= c + dc < w
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            if bounces == n:
                break                      # the stop: the wall cell is the current cell
            if not 0 <= nr < h:
                dr = -dr
            if not 0 <= nc < w:
                dc = -dc
            bounces += 1
            nr, nc = r + dr, c + dc
        r, c = nr, nc
        cells.append((r, c))
    for rr, cc in cells:
        out[rr, cc] = trace_color
    out[cells[0]] = start_color
    return out


def oracle_24(grid: np.ndarray, start_color: int, trace_color: int) -> int:
    """Follow the trace from the marker; count direction changes; every trace cell used."""
    marks = np.argwhere(grid == start_color)
    assert len(marks) == 1
    cur = (int(marks[0][0]), int(marks[0][1]))
    cells = {(int(r), int(c)) for r, c in np.argwhere(grid == trace_color)}
    prev, d, n, seen = None, None, 0, {cur}
    while True:
        nxt = [(cur[0] + a, cur[1] + b) for a in (-1, 1) for b in (-1, 1)
               if (cur[0] + a, cur[1] + b) in cells and (cur[0] + a, cur[1] + b) != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "trace branches"
        step = (nxt[0][0] - cur[0], nxt[0][1] - cur[1])
        if d is not None and step != d:
            n += 1
        d, prev, cur = step, cur, nxt[0]
        assert cur not in seen, "trace revisits a cell"
        seen.add(cur)
    assert seen - {(int(marks[0][0]), int(marks[0][1]))} == cells, "trace cells left over"
    return n


# ----------------------------------------------------------------------------- task 23
def test_23_oracle_agrees_and_marker_is_clean():
    corners_hit = 0
    for seed in SEEDS:
        ex = generate("23", seed)
        m = ex["metadata"]
        inp = np.asarray(ex["input"])
        out = np.asarray(ex["output"])
        assert np.array_equal(oracle_23(inp, m["start_color"], m["trace_color"], m["direction"], m["n_bounces"]), out), seed
        # the marker is the only cell of its colour, sits on background, and every neighbour is background
        marks = np.argwhere(inp == m["start_color"])
        assert len(marks) == 1, seed
        r, c = marks[0]
        h, w = inp.shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < h and 0 <= cc < w:
                    assert inp[rr, cc] == m["bg_color"], seed
        # the trace colour is new in the output, the marker keeps its colour, nothing else changes
        assert not (inp == m["trace_color"]).any(), seed
        assert out[r, c] == m["start_color"], seed
        changed = np.argwhere(inp != out)
        assert all(out[a, b] == m["trace_color"] for a, b in changed), seed
        assert len(changed) >= 3, seed
        corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        corners_hit += any(out[a, b] == m["trace_color"] for a, b in corners)
        # the last cell of the trace is on a wall; the trace cells are counted from the marker
        assert m["trace_len"] >= 4
    assert corners_hit == 0


def test_23_answer_spread_sizes_colours_textures():
    n_hist = collections.Counter()
    H, W, cols, tex, scene = [], [], [], 0, 0
    for seed in SEEDS:
        m = generate("23", seed)["metadata"]
        n_hist[m["n_bounces"]] += 1
        H.append(m["grid_h"]); W.append(m["grid_w"]); cols.append(m["n_colors"])
        tex += m["texture_kind"] is not None
        scene += m["n_scene_objects"] > 0
    n = len(SEEDS)
    assert set(n_hist) == set(range(7)), n_hist                  # 0..6 all present
    assert n_hist[0] <= 0.12 * n and n_hist[0] >= 0.03 * n, n_hist
    assert min(n_hist[k] for k in range(1, 7)) >= 0.08 * n, n_hist
    assert min(H) == 4 and max(H) == 30 and min(W) == 4 and max(W) == 30
    assert abs(np.corrcoef(H, W)[0, 1]) < 0.15
    assert np.mean(np.array(H) == np.array(W)) < 0.10
    c = collections.Counter(cols)
    assert max(c) == 9 and min(c) == 2, c                          # 10 is structural: the trace colour is reserved
    assert all(c[k] >= 0.05 * n for k in range(2, 10)), c
    assert 0.20 <= tex / n <= 0.45, tex / n
    assert 0.6 <= scene / n <= 0.9, scene / n


def test_23_prompt_wording():
    seen_zero = False
    for seed in SEEDS:
        m = generate("23", seed)["metadata"]
        p = get_prompt("23", m)
        assert p.startswith(f"Starting from the single cell of color {m['start_color']}, draw a diagonal line in color {m['trace_color']} in the {m['direction']} (")
        assert p.endswith("Leave every other cell unchanged.")
        assert "reflections" not in p                                # no "1 reflections"
        if m["n_bounces"] == 0:
            seen_zero = True
            assert "The line does not reflect: it runs straight to the wall and stops on the wall cell it reaches, which is drawn." in p
            assert "may overlap" not in p and "The line is drawn on top of any existing cells except the marked start cell" in p
        else:
            word = P and {1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times"}[m["n_bounces"]]
            assert f"reflects off the grid walls exactly {word}: each reflection happens on the wall cell the line reaches, which is drawn, and the line turns there and continues;" in p
            assert "it continues straight and stops on the wall cell it reaches, which is drawn." in p
            assert "The trajectory may overlap itself" in p
        assert "background" not in p.lower().replace("the marked start cell", "")  # background never named
        letters = get_prompt("23", m, color_palette=list("ABCDEFGHIJ"))
        assert "color " + "ABCDEFGHIJ"[m["start_color"]] in letters
    assert seen_zero


def test_23_validator_raises_on_a_second_marker():
    ex = generate("23", 0)
    m = ex["metadata"]
    inp = np.asarray(ex["input"])
    out = np.asarray(ex["output"])
    P.validate_23(inp, out, m["start_color"], m["trace_color"], m["direction"], m["n_bounces"])
    bad = inp.copy()
    cells = np.argwhere(bad == m["bg_color"])
    bad[cells[-1][0], cells[-1][1]] = m["start_color"]
    with pytest.raises(ValueError):
        P.validate_23(bad, out, m["start_color"], m["trace_color"], m["direction"], m["n_bounces"])
    with pytest.raises(ValueError):
        P.validate_23(inp, inp, m["start_color"], m["trace_color"], m["direction"], m["n_bounces"])


# ----------------------------------------------------------------------------- task 24
def test_24_oracle_agrees_and_trace_is_clean():
    for seed in SEEDS:
        ex = generate("24", seed)
        m = ex["metadata"]
        grid = np.asarray(ex["input"])
        assert oracle_24(grid, m["start_color"], m["trace_color"]) == int(ex["output"]) == m["n_bounces"], seed
        h, w = grid.shape
        corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        for r, c in np.argwhere((grid == m["trace_color"]) | (grid == m["start_color"])):
            assert (int(r), int(c)) not in corners, seed
        # distractor objects never touch the trace, even diagonally
        semantic = (grid == m["trace_color"]) | (grid == m["start_color"])
        if m["n_distractors"]:
            others = ~semantic & (grid != m["bg_color"])
            if m["texture_kind"] is None:
                for r, c in np.argwhere(others):
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w:
                                assert not semantic[rr, cc], seed


def test_24_answers_flat_and_size_does_not_leak():
    n_hist = collections.Counter()
    size_by_answer = collections.defaultdict(collections.Counter)
    H, W, cols, tex = [], [], [], 0
    seeds = range(1500)
    for seed in seeds:
        m = generate("24", seed)["metadata"]
        b = m["n_bounces"]
        n_hist[b] += 1
        size_by_answer[b][(m["grid_h"], m["grid_w"])] += 1
        H.append(m["grid_h"]); W.append(m["grid_w"]); cols.append(m["n_colors"])
        tex += m["texture_kind"] is not None
    n = len(seeds)
    assert set(n_hist) == set(range(P.T24_MAX_BOUNCES + 1)) and P.T24_MAX_BOUNCES >= 7, n_hist
    assert 0.03 * n <= n_hist[0] <= 0.10 * n, n_hist
    # the size is drawn first, so the counts only thin sizes admit taper off (6 ~ 10 %, 7 ~ 3 %)
    assert min(n_hist[k] for k in range(1, 6)) >= 0.10 * n, n_hist
    assert n_hist[6] >= 0.06 * n and n_hist[7] >= 0.015 * n, n_hist
    # H and W independent (they were negatively correlated when the answer chose the size)
    assert abs(float(np.corrcoef(H, W)[0, 1])) < 0.08, np.corrcoef(H, W)[0, 1]
    assert 0.02 <= np.mean(np.array(H) == np.array(W)) <= 0.06
    # no size signature: for every answer the five most common sizes hold under 10 % of it
    # (or at most three draws per size when the sample is smaller than the feasible set),
    # and the sizes reached cover most of the feasible ones
    feasible = collections.Counter()
    for mx in P.T24_FEASIBLE.values():
        for b in range(mx + 1):
            feasible[b] += 1
    for b, sizes in size_by_answer.items():
        total = sum(sizes.values())
        top5 = sum(v for _, v in sizes.most_common(5))
        assert top5 <= max(0.10 * total, 15 if total < feasible[b] else 0), (b, sizes.most_common(5))
        assert len(sizes) >= 0.6 * min(feasible[b], total), (b, len(sizes), feasible[b], total)
    hc = collections.Counter(H)
    wc = collections.Counter(W)
    assert min(hc) == 5 and max(hc) == 30 and min(wc) == 5 and max(wc) == 30
    med_h = np.median(list(hc.values()))
    med_w = np.median(list(wc.values()))
    assert max(hc.values()) <= 2.0 * med_h and max(wc.values()) <= 2.0 * med_w, (hc, wc)
    assert np.mean(np.array(H) == np.array(W)) < 0.10
    c = collections.Counter(cols)
    assert max(c) == 10 and min(c) == 3, c
    assert all(c[k] >= 0.03 * n for k in range(3, 11)), c
    assert 0.30 <= tex / n <= 0.60, tex / n


def test_24_feasibility_table_spot_checks():
    assert set(P.T24_FEASIBLE) == {(h, w) for h in range(5, 31) for w in range(5, 31)}
    for hw in ((5, 5), (5, 12), (8, 20), (12, 12), (30, 5), (17, 23)):
        assert P.T24_FEASIBLE[hw] == P._max_clean_bounces(*hw), hw
    feasible = collections.Counter()
    for mx in P.T24_FEASIBLE.values():
        for b in range(mx + 1):
            feasible[b] += 1
    assert feasible[P.T24_MAX_BOUNCES] >= 50 and feasible[P.T24_MAX_BOUNCES + 1] < 50


def test_24_prompt_and_validator():
    m = generate("24", 0)["metadata"]
    p = get_prompt("24", m)
    assert f"A marked cell of color {m['start_color']}" in p and f"color {m['trace_color']} cells" in p
    assert "background detail to ignore" in p
    ex = generate("24", 1)
    grid = np.asarray(ex["input"])
    m = ex["metadata"]
    assert P.count_reflections_from_grid(grid, m["start_color"], m["trace_color"]) == int(ex["output"])
    bad = grid.copy()
    cells = np.argwhere(bad == m["bg_color"])
    bad[cells[0][0], cells[0][1]] = m["start_color"]
    with pytest.raises(ValueError):
        P.count_reflections_from_grid(bad, m["start_color"], m["trace_color"])
    bad = grid.copy()
    tr = np.argwhere(bad == m["trace_color"])
    bad[tr[len(tr) // 2][0], tr[len(tr) // 2][1]] = m["bg_color"]     # a gap in the trace leaves cells over
    with pytest.raises(ValueError):
        P.count_reflections_from_grid(bad, m["start_color"], m["trace_color"])


def test_generation_time():
    import time
    t0 = time.time()
    for seed in range(200):
        generate("23", seed)
        generate("24", seed)
    assert (time.time() - t0) / 400 < 0.02


def _size_bucket(h: int, w: int) -> tuple[int, int]:
    return (min(h, w) // 5, max(h, w) // 5)


def test_24_size_alone_predicts_the_answer_no_better_than_chance():
    """A fixed lookup classifier from the size bucket (min side // 5, max side // 5) trained on
    one half of the seeds and tested on the other half stays within 3.5 points of the
    majority-class rate.  Only the feasibility itself remains (7 needs a thin grid), which
    is inherent to clean traces."""
    rows = []
    for seed in range(3000):
        m = generate("24", seed)["metadata"]
        rows.append((m["grid_h"], m["grid_w"], m["n_bounces"]))
    train, test = rows[:1500], rows[1500:]
    table = collections.defaultdict(collections.Counter)
    for h, w, b in train:
        table[_size_bucket(h, w)][b] += 1
    majority = collections.Counter(b for _, _, b in train).most_common(1)[0][0]
    hits = 0
    for h, w, b in test:
        counts = table.get(_size_bucket(h, w))
        pred = counts.most_common(1)[0][0] if counts else majority
        hits += pred == b
    acc = hits / len(test)
    chance = collections.Counter(b for _, _, b in test).most_common(1)[0][1] / len(test)
    assert acc <= chance + 0.035, (acc, chance)


def test_24_infeasible_answers_match_an_exhaustive_sweep():
    """The table gives the maximum clean count per size; every lower count is reachable
    except the four listed combinations (a corner-free straight run of five cells does not
    fit).  Checked exhaustively on every size with a side of at most 7."""
    dirs = list(P._DIRS.values())
    found = set()
    for (h, w), mx in P.T24_FEASIBLE.items():
        if min(h, w) > 7:
            continue
        ok = set()
        for r in range(h):
            for c in range(w):
                for dr, dc in dirs:
                    if not P._first_step_ok(h, w, r, c, (dr, dc)):
                        continue
                    cells, turns = P._diag_path(h, w, r, c, dr, dc, stop_on_revisit=True)
                    clean = P._clean_prefix_len(cells, h, w)
                    for b in range(min(mx, P.T24_MAX_BOUNCES) + 1):
                        ext, wall = P._prefix_ends(turns, clean, b)
                        if ext or wall is not None:
                            ok.add(b)
        for b in range(min(mx, P.T24_MAX_BOUNCES) + 1):
            if b not in ok:
                found.add((h, w, b))
    assert found == set(P.T24_INFEASIBLE), found
    for (h, w, b) in P.T24_INFEASIBLE:
        assert b not in P.t24_feasible_answers(h, w)
    assert P.t24_feasible_answers(5, 5) == [2] and P.t24_feasible_answers(30, 30) == list(range(0, P.T24_FEASIBLE[(30, 30)] + 1))[: P.T24_MAX_BOUNCES + 1]


def test_24_prompt_defines_a_bounce():
    for seed in range(20):
        ex = generate("24", seed)
        p = get_prompt("24", ex["metadata"])
        assert "A bounce is a change of direction at a wall cell; the cell where the trace stops is not a bounce, even when it lies on a wall." in p
        assert p.endswith("How many times does the trace bounce off the grid walls before it stops? Answer with the number.")
