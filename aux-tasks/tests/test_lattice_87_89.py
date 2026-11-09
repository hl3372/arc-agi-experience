"""Independent oracles, coverage, leak and prompt checks for the lattice tasks 87-89.

Every oracle here reads the INPUT GRID and the PROMPT TEXT only: it recovers the
lattice from the constant rows and columns, rebuilds the blocks (clipped ones
included) and applies the rule the prompt states.  Task 81 shares
``generators/lattice.py`` and is pinned by a 100-seed hash.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from generators import lattice_blocks_87_89 as module
from prompts import get_prompt
from registry import generate

SEEDS = 400


# ---------------------------------------------------------------------------
# oracles written from the prompt text alone
# ---------------------------------------------------------------------------

def _blocks_from_prompt(grid):
    """The block decomposition the prompt describes, read off the visible grid.

    Anchor-free, exactly as the prompt reads: "some complete rows and complete
    columns ... are filled with one single color".  Cell (0, 0) means nothing -- the
    lattice has a phase per axis, so the grid can start with a clipped block row or
    block column.  The monochrome rows and columns must all carry ONE colour; any
    other complete monochrome line would be a forged lattice line.
    """
    g = np.asarray(grid)
    h, w = g.shape
    line_rows = [r for r in range(h) if len(set(int(v) for v in g[r])) == 1]
    line_cols = [c for c in range(w) if len(set(int(v) for v in g[:, c])) == 1]
    assert line_rows and line_cols, "no complete row line / column line"
    colors = {int(g[r][0]) for r in line_rows} | {int(g[0][c]) for c in line_cols}
    assert len(colors) == 1, f"forged lattice line: complete lines in colours {colors}"
    line = colors.pop()

    def runs(n, drop):
        out, start = [], None
        for i in range(n):
            if i in drop:
                if start is not None:
                    out.append((start, i))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            out.append((start, n))
        return out

    return line, runs(h, set(line_rows)), runs(w, set(line_cols))


def _oracle_87(grid, prompt):
    assert "0-indexed" in prompt and "cuts short still counts as a block" in prompt
    assert prompt.endswith("Leave every other cell unchanged.")
    match = re.search(r"Paint every cell of block \((\d+), (\d+)\) with color (\S+?),", prompt)
    assert match, prompt
    br, bc = int(match[1]), int(match[2])
    color = match[3] if isinstance(grid[0][0], str) else int(match[3])
    _, rows, cols = _blocks_from_prompt(grid)
    out = [list(row) for row in grid]
    r0, r1 = rows[br]
    c0, c1 = cols[bc]
    for r in range(r0, r1):
        for c in range(c0, c1):
            out[r][c] = color
    return out


def _oracle_88(grid, prompt):
    assert "Ignore the background" in prompt and "cuts short still counts as a block" in prompt
    assert "completely filled with one single color other than the background" in prompt
    assert "fill every block strictly between them with that same color" in prompt
    assert prompt.endswith("Leave every other cell unchanged.")
    line, rows, cols = _blocks_from_prompt(grid)
    counts = Counter()
    for r0, r1 in rows:
        for c0, c1 in cols:
            for r in range(r0, r1):
                for c in range(c0, c1):
                    counts[int(grid[r][c])] += 1
    ground = counts.most_common(1)[0][0]
    marked = {}
    for i, (r0, r1) in enumerate(rows):
        for j, (c0, c1) in enumerate(cols):
            values = {int(grid[r][c]) for r in range(r0, r1) for c in range(c0, c1)}
            if len(values) == 1:
                color = values.pop()
                if color not in (line, ground):
                    marked.setdefault(color, []).append((i, j))
    out = [list(row) for row in grid]

    def paint(i, j, color):
        r0, r1 = rows[i]
        c0, c1 = cols[j]
        for r in range(r0, r1):
            for c in range(c0, c1):
                out[r][c] = color

    for color, cells in marked.items():
        for axis in (0, 1):
            lanes = {}
            for cell in cells:
                lanes.setdefault(cell[axis], []).append(cell[1 - axis])
            for lane, others in lanes.items():
                if len(others) < 2:
                    continue
                for k in range(min(others) + 1, max(others)):
                    paint(*( (lane, k) if axis == 0 else (k, lane) ), color)
    return out


def _oracle_89(grid, prompt):
    assert prompt.endswith("Answer with the number.")
    assert "cuts short still counts as a block" in prompt
    assert "every one of its cells is that color" in prompt
    match = re.search(r"completely filled with color (\S+)\?", prompt)
    assert match, prompt
    color = match[1] if isinstance(grid[0][0], str) else int(match[1])
    _, rows, cols = _blocks_from_prompt(grid)
    total = 0
    for r0, r1 in rows:
        for c0, c1 in cols:
            if all(grid[r][c] == color for r in range(r0, r1) for c in range(c0, c1)):
                total += 1
    return str(total)


@pytest.fixture(scope="module")
def ex87():
    return [generate("87", s) for s in range(SEEDS)]


@pytest.fixture(scope="module")
def ex88():
    return [generate("88", s) for s in range(SEEDS)]


@pytest.fixture(scope="module")
def ex89():
    return [generate("89", s) for s in range(SEEDS)]


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------

def test_oracle_87_agrees(ex87):
    for ex in ex87:
        prompt = get_prompt("87", ex["metadata"])
        assert _oracle_87(ex["input"], prompt) == ex["output"]


def test_oracle_88_agrees(ex88):
    for ex in ex88:
        prompt = get_prompt("88", ex["metadata"])
        assert _oracle_88(ex["input"], prompt) == ex["output"]


def test_oracle_89_agrees(ex89):
    for ex in ex89:
        prompt = get_prompt("89", ex["metadata"])
        assert _oracle_89(ex["input"], prompt) == ex["output"]


def _runs_outside(n, drop):
    out, start = [], None
    for i in range(n):
        if i in drop:
            if start is not None:
                out.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        out.append((start, n))
    return out


def test_recovered_lattice_matches_metadata(ex87, ex88, ex89):
    for exs in (ex87, ex88, ex89):
        for ex in exs:
            meta = ex["metadata"]
            line, rows, cols = _blocks_from_prompt(ex["input"])
            assert line == meta["line_color"]
            assert len(rows) == meta["block_rows"] and len(cols) == meta["block_cols"]
            assert meta["line_thickness"] < min(meta["row_spacing"], meta["col_spacing"])
            # the metadata line starts, thickened, are exactly the visible lattice lines
            for n, starts, bands, phase, regular, spacing in (
                (meta["grid_h"], meta["row_line_starts"], rows, meta["row_phase"],
                 meta["row_regular"], meta["row_spacing"]),
                (meta["grid_w"], meta["col_line_starts"], cols, meta["col_phase"],
                 meta["col_regular"], meta["col_spacing"]),
            ):
                assert starts, "an axis with no lattice line at all"
                drawn = set()
                for start in starts:
                    drawn.update(range(start, min(start + meta["line_thickness"], n)))
                assert _runs_outside(n, drawn) == bands
                assert phase == min(starts)
                gaps = [b - a for a, b in zip(starts, starts[1:])]
                if regular:
                    assert set(gaps) <= {spacing}
                    assert [tuple(b) for b in module.axis_bands(
                        n, spacing, meta["line_thickness"], phase)[1]] == bands
                else:
                    assert len(set(gaps)) >= 2      # genuinely non-periodic
                    assert spacing == min(gaps)


def test_phase_coverage_and_clipping_on_every_side(ex87, ex88, ex89):
    for exs in (ex87, ex88, ex89):
        metas = [e["metadata"] for e in exs]
        n = len(metas)
        for key in ("row_phase", "col_phase"):
            zero = sum(1 for m in metas if m[key] == 0) / n
            assert 0.22 <= zero <= 0.60, (key, zero)
            assert max(m[key] for m in metas) >= 3
        sides = Counter(s for m in metas for s in m["clipped_sides"])
        for side in ("top", "bottom", "left", "right"):
            assert sides[side] >= 0.15 * n, (side, sides[side])
        both = sum(1 for m in metas
                   if "top" in m["clipped_sides"] and "left" in m["clipped_sides"])
        assert both >= 0.15 * n, both
        irregular = sum(1 for m in metas if not (m["row_regular"] and m["col_regular"]))
        assert 0.05 * n <= irregular <= 0.20 * n, irregular / n
        for ex in exs:
            meta = ex["metadata"]
            _, rows, cols = _blocks_from_prompt(ex["input"])
            # a positive phase means the grid starts inside a clipped block
            assert (rows[0][0] == 0) == (meta["row_phase"] > 0)
            assert (cols[0][0] == 0) == (meta["col_phase"] > 0)
            assert ("top" in meta["clipped_sides"]) == (meta["row_phase"] > 0)
            assert ("left" in meta["clipped_sides"]) == (meta["col_phase"] > 0)


def test_no_forged_lines_over_3000_grids():
    """A monochrome complete row or column of a non-line colour never occurs.

    It cannot: at least one complete column line crosses every row and at least one
    complete row line crosses every column, so no other complete line can be
    monochrome.  ``_blocks_from_prompt`` asserts it on every grid it reads.
    """
    for task in ("87", "88", "89"):
        for seed in range(1000):
            ex = generate(task, seed)
            g = np.asarray(ex["input"])
            line, rows, cols = _blocks_from_prompt(ex["input"])
            assert line == ex["metadata"]["line_color"]
            assert len(rows) == ex["metadata"]["block_rows"]
            assert len(cols) == ex["metadata"]["block_cols"]
            mono_rows = [r for r in range(g.shape[0]) if len(set(g[r].tolist())) == 1]
            mono_cols = [c for c in range(g.shape[1]) if len(set(g[:, c].tolist())) == 1]
            assert mono_rows and mono_cols
            assert all(int(g[r][0]) == line for r in mono_rows)
            assert all(int(g[0][c]) == line for c in mono_cols)


def test_block_content_never_uses_the_line_color(ex87, ex88, ex89):
    for exs in (ex87, ex88, ex89):
        for ex in exs:
            g = np.asarray(ex["input"])
            _, rows, cols = _blocks_from_prompt(ex["input"])
            for r0, r1 in rows:
                for c0, c1 in cols:
                    assert not (g[r0:r1, c0:c1] == ex["metadata"]["line_color"]).any()


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["87", "88", "89"])
def test_size_coverage(task, ex87, ex88, ex89):
    exs = {"87": ex87, "88": ex88, "89": ex89}[task]
    metas = [e["metadata"] for e in exs]
    hs = [m["grid_h"] for m in metas]
    ws = [m["grid_w"] for m in metas]
    assert min(hs) == 5 and max(hs) == 30 and min(ws) == 5 and max(ws) == 30
    for values in (hs, ws):
        counts = Counter(values)
        assert len(counts) == 26
        for side in range(5, 31):
            assert 0.010 <= counts[side] / len(values) <= 0.085, (side, counts[side])
    assert abs(np.corrcoef(hs, ws)[0, 1]) < 0.15
    squares = sum(1 for h, w in zip(hs, ws) if h == w)
    assert 0.005 <= squares / len(hs) <= 0.10
    assert sum(1 for h, w in zip(hs, ws) if h * w >= 700) >= 0.015 * len(hs)
    assert max(h * w for h, w in zip(hs, ws)) >= 780


@pytest.mark.parametrize("task", ["87", "88", "89"])
def test_color_coverage(task, ex87, ex88, ex89):
    exs = {"87": ex87, "88": ex88, "89": ex89}[task]
    counts = Counter(e["metadata"]["n_colors"] for e in exs)
    assert max(counts) == 10
    assert min(counts) <= 3
    rich = sum(counts[k] for k in (8, 9, 10)) / len(exs)
    assert rich >= 0.15, rich
    for k in range(3, 10):
        assert counts[k] >= 0.02 * len(exs), (k, counts[k])


@pytest.mark.parametrize("task", ["87", "88", "89"])
def test_lattice_variety(task, ex87, ex88, ex89):
    exs = {"87": ex87, "88": ex88, "89": ex89}[task]
    metas = [e["metadata"] for e in exs]
    spacings = Counter(m["row_spacing"] for m in metas) + Counter(m["col_spacing"] for m in metas)
    assert set(spacings) == set(range(2, 9))
    thickness = Counter(m["line_thickness"] for m in metas)
    assert set(thickness) >= {1, 2}
    assert thickness[2] >= 0.05 * len(exs)
    assert thickness[1] >= 0.5 * len(exs)
    non_square = sum(1 for m in metas if m["row_spacing"] != m["col_spacing"])
    assert non_square >= 0.6 * len(exs)
    clipped = sum(1 for m in metas if m["clipped_blocks"] > 0)
    assert 0.5 * len(exs) <= clipped <= 0.995 * len(exs)
    assert max(m["block_rows"] for m in metas) >= 8
    assert max(m["block_cols"] for m in metas) >= 8


@pytest.mark.parametrize("task", ["87", "88", "89"])
def test_texture_policy(task, ex87, ex88, ex89):
    exs = {"87": ex87, "88": ex88, "89": ex89}[task]
    textured = [e for e in exs if e["metadata"]["texture_group"]]
    assert 0.08 * len(exs) <= len(textured) <= 0.45 * len(exs)
    for ex in textured:
        meta = ex["metadata"]
        assert meta["texture_group"] == "sparse"
        assert meta["texture_named"] is False
        assert meta["texture_colors"]
        g = np.asarray(ex["input"])
        reserved = {meta["line_color"], meta["bg_color"]}
        reserved |= set(meta.get("group_colors", []))
        reserved |= {meta[k] for k in ("paint_color", "target_color") if k in meta}
        for color in meta["texture_colors"]:
            assert color not in reserved
            assert (g == color).any()
        # textures never sit on a lattice line
        line_mask = np.ones(g.shape, dtype=bool)
        _, rows, cols = _blocks_from_prompt(ex["input"])
        for r0, r1 in rows:
            for c0, c1 in cols:
                line_mask[r0:r1, c0:c1] = False
        for color in meta["texture_colors"]:
            assert not (g[line_mask] == color).any()


# ---------------------------------------------------------------------------
# task 87: no route to the target except the index
# ---------------------------------------------------------------------------

def test_87_target_is_not_findable_without_the_index(ex87):
    unique_blank = 0
    first_blank = 0
    for ex in ex87:
        meta = ex["metadata"]
        g = np.asarray(ex["input"])
        _, rows, cols = _blocks_from_prompt(ex["input"])
        plain = []
        for i, (r0, r1) in enumerate(rows):
            for j, (c0, c1) in enumerate(cols):
                view = g[r0:r1, c0:c1]
                if bool(np.all(view == view.flat[0])):
                    plain.append(((i, j), int(view.flat[0])))
        if not plain:
            continue                       # no completely filled block: the reader has nothing
        blank = Counter(c for _, c in plain).most_common(1)[0][0]
        blanks = [b for b, c in plain if c == blank]
        target = (meta["target_row"], meta["target_col"])
        if len(blanks) == 1 and blanks[0] == target:
            unique_blank += 1
        if blanks and blanks[0] == target:
            first_blank += 1
    assert unique_blank <= 0.05 * len(ex87), unique_blank / len(ex87)
    assert first_blank <= 0.15 * len(ex87), first_blank / len(ex87)


def test_87_target_variety(ex87):
    metas = [e["metadata"] for e in ex87]
    assert max(m["target_row"] for m in metas) >= 5
    assert max(m["target_col"] for m in metas) >= 5
    clipped = sum(1 for m in metas if m["target_is_clipped"])
    assert clipped >= 0.05 * len(metas)
    decorated = sum(1 for m in metas if m["n_decorated_blocks"] > 0)
    assert decorated >= 0.5 * len(metas)
    for ex in ex87:
        assert ex["input"] != ex["output"]


# ---------------------------------------------------------------------------
# task 88: scenes, no-ops and the background margin
# ---------------------------------------------------------------------------

def _oracle_88_color_blind(grid, prompt):
    """The reading that drops the prompt's "For each color separately".

    It fills between CONSECUTIVE marked blocks of a block row / block column whatever
    their colours, using the colour of the first of the pair -- the most charitable
    colour-blind reading, since on a lane whose markers all share a colour it agrees
    with the rule.  The task is only about colour groups if this reader is wrong
    often, so its accuracy is bounded here.
    """
    assert "For each color separately" in prompt
    line, rows, cols = _blocks_from_prompt(grid)
    counts = Counter()
    for r0, r1 in rows:
        for c0, c1 in cols:
            for r in range(r0, r1):
                for c in range(c0, c1):
                    counts[int(grid[r][c])] += 1
    ground = counts.most_common(1)[0][0]
    marked = []
    for i, (r0, r1) in enumerate(rows):
        for j, (c0, c1) in enumerate(cols):
            values = {int(grid[r][c]) for r in range(r0, r1) for c in range(c0, c1)}
            if len(values) == 1:
                color = values.pop()
                if color not in (line, ground):
                    marked.append(((i, j), color))
    out = [list(row) for row in grid]

    def paint(i, j, color):
        r0, r1 = rows[i]
        c0, c1 = cols[j]
        for r in range(r0, r1):
            for c in range(c0, c1):
                out[r][c] = color

    for axis in (0, 1):
        lanes = {}
        for cell, color in marked:
            lanes.setdefault(cell[axis], []).append((cell[1 - axis], color))
        for lane, items in lanes.items():
            if len(items) < 2:
                continue
            items.sort()
            for (pos, color), (nxt, _) in zip(items, items[1:]):
                for k in range(pos + 1, nxt):
                    paint(*((lane, k) if axis == 0 else (k, lane)), color)
    return out


def test_88_color_clause_is_load_bearing(ex88):
    right = 0
    for ex in ex88:
        prompt = get_prompt("88", ex["metadata"])
        blind = _oracle_88_color_blind(ex["input"], prompt)
        same = blind == ex["output"]
        right += int(same)
        assert same != ex["metadata"]["color_clause_matters"]
    assert right <= 0.55 * len(ex88), right / len(ex88)
    matters = sum(1 for e in ex88 if e["metadata"]["color_clause_matters"])
    assert matters >= 0.45 * len(ex88), matters / len(ex88)


def test_88_scenes_scale_with_the_canvas(ex88):
    bands = {}
    for ex in ex88:
        meta = ex["metadata"]
        touched = (meta["n_marked_blocks"] + meta["n_filled_blocks"]
                   + meta["n_decorated_blocks"]) / meta["n_blocks"]
        area = meta["grid_h"] * meta["grid_w"]
        key = "small" if area < 250 else ("mid" if area < 700 else "big")
        bands.setdefault(key, []).append(touched)
        bands.setdefault("blocks_" + key, []).append(meta["n_blocks"])
        bands.setdefault("groups_" + key, []).append(meta["n_groups"])
    assert set(bands) >= {"small", "mid", "big"}
    small, big = float(np.mean(bands["small"])), float(np.mean(bands["big"]))
    assert big >= small - 0.12, (small, big)
    # groups and markers follow the block count, they do not sit at a fixed cap
    assert np.mean(bands["groups_big"]) >= 1.8 * np.mean(bands["groups_small"])
    assert max(m["n_groups"] for m in (e["metadata"] for e in ex88)) >= 6


def test_88_background_survives_the_isolated_cell_reading(ex88):
    """A reader that first sets aside isolated single cells still sees the ground."""
    for ex in ex88:
        meta = ex["metadata"]
        g = np.asarray(ex["input"])
        _, rows, cols = _blocks_from_prompt(ex["input"])
        keep = ~module.isolated_cell_mask(g)
        counts = Counter()
        for r0, r1 in rows:
            for c0, c1 in cols:
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        if keep[r, c]:
                            counts[int(g[r, c])] += 1
        order = counts.most_common()
        assert order[0][0] == meta["bg_color"]
        rival = order[1][1] if len(order) > 1 else 0
        assert order[0][1] >= 1.25 * rival
        assert meta["background_margin_isolated"] >= 1.25


def test_88_scene_mix(ex88):
    metas = [e["metadata"] for e in ex88]
    groups = Counter(m["n_groups"] for m in metas)
    assert set(groups) >= {1, 2, 3}
    assert groups[1] <= 0.25 * len(metas), groups[1] / len(metas)
    orientations = Counter(o for m in metas for o in m["orientations"])
    assert orientations["row"] >= 0.2 * len(metas) and orientations["column"] >= 0.2 * len(metas)
    mixed = sum(1 for m in metas
                if "row" in m["orientations"] and "column" in m["orientations"])
    assert mixed >= 0.08 * len(metas), mixed / len(metas)
    markers = Counter(k for m in metas for k in m["markers_per_group"])
    assert markers[3] >= 0.02 * sum(markers.values())
    assert markers[4] >= 0.005 * sum(markers.values())
    noop = sum(1 for m in metas if m["n_noop_groups"] > 0)
    assert 0.20 * len(metas) <= noop <= 0.85 * len(metas), noop / len(metas)
    identity = sum(1 for m in metas if m["identity_output"])
    assert 0.03 * len(metas) <= identity <= 0.16 * len(metas), identity / len(metas)
    kinds = Counter(k for m in metas for k, o in zip(m["group_kinds"], m["orientations"])
                    if o == "none")
    assert set(kinds) == {"adjacent", "unaligned", "single", "decoy"}
    assert kinds["decoy"] >= 0.15 * len(metas), kinds["decoy"] / len(metas)


def test_88_background_is_identifiable(ex88):
    for ex in ex88:
        meta = ex["metadata"]
        assert meta["background_margin"] >= 1.25
        assert meta["background_cell_margin"] >= 1.25
        line, rows, cols = _blocks_from_prompt(ex["input"])
        counts = Counter()
        g = np.asarray(ex["input"])
        for r0, r1 in rows:
            for c0, c1 in cols:
                counts.update(int(v) for v in g[r0:r1, c0:c1].ravel())
        assert counts.most_common(1)[0][0] == meta["bg_color"]


def test_88_identity_matches_the_grids(ex88):
    for ex in ex88:
        assert (ex["input"] == ex["output"]) == ex["metadata"]["identity_output"]


def test_88_group_colors_are_distinct_and_never_the_ground(ex88):
    for ex in ex88:
        meta = ex["metadata"]
        colors = meta["group_colors"]
        assert len(set(colors)) == len(colors)
        assert meta["bg_color"] not in colors and meta["line_color"] not in colors


# ---------------------------------------------------------------------------
# task 89: "completely" is load-bearing, answers spread, no size leak
# ---------------------------------------------------------------------------

def test_89_answers(ex89):
    answers = [int(e["output"]) for e in ex89]
    counts = Counter(answers)
    assert counts[0] >= 0.05 * len(answers) and counts[0] <= 0.16 * len(answers)
    assert max(answers) >= 12
    for value in range(1, 9):
        assert counts[value] >= 0.02 * len(answers), (value, counts[value])
    for ex in ex89:
        meta = ex["metadata"]
        assert int(ex["output"]) == meta["n_full_blocks"] <= meta["answer_cap"]
        assert meta["answer_cap"] <= meta["n_blocks"] - 1


def test_89_partial_blocks_make_completely_load_bearing(ex89):
    """"Count every block that CONTAINS the colour" must be wrong most of the time."""
    metas = [e["metadata"] for e in ex89]
    partial = sum(1 for m in metas if m["n_partial_blocks"] > 0)
    assert partial >= 0.50 * len(metas), partial / len(metas)
    reader = 0
    for ex in ex89:
        prompt = get_prompt("89", ex["metadata"])
        _, rows, cols = _blocks_from_prompt(ex["input"])
        color = int(re.search(r"completely filled with color (\S+)\?", prompt)[1])
        contains = sum(1 for r0, r1 in rows for c0, c1 in cols
                       if any(ex["input"][r][c] == color
                              for r in range(r0, r1) for c in range(c0, c1)))
        assert contains == ex["metadata"]["n_blocks_containing_target"]
        reader += int(str(contains) == ex["output"])
    assert reader <= 0.45 * len(metas), reader / len(metas)
    # the partial cap follows the block count instead of a fixed three
    assert max(m["n_partial_blocks"] for m in metas) >= 5
    for m in metas:
        assert m["n_blocks_containing_target"] - m["n_full_blocks"] == m["n_partial_blocks"]
        assert m["n_partial_blocks"] <= max(1, m["n_blocks"] // 3)


def test_89_answer_is_not_predictable_from_the_size(ex89):
    answers = [int(e["output"]) for e in ex89]
    areas = [e["metadata"]["grid_h"] * e["metadata"]["grid_w"] for e in ex89]
    assert abs(np.corrcoef(answers, areas)[0, 1]) < 0.45
    majority = Counter(answers).most_common(1)[0][1] / len(answers)
    assert majority < 0.25


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["87", "88", "89"])
def test_prompt_policy(task, ex87, ex88, ex89):
    exs = {"87": ex87, "88": ex88, "89": ex89}[task]
    texts = set()
    for ex in exs:
        prompt = get_prompt(task, ex["metadata"])
        lowered = prompt.lower()
        assert "background is" not in lowered
        assert "most common" not in lowered and "most frequent" not in lowered
        assert "texture" not in lowered and "pattern is" not in lowered
        assert "cuts short still counts as a block" in prompt
        assert "not part of any block" in prompt
        for key in ("bg_color", "line_color"):
            assert f"color {ex['metadata'][key]}" not in prompt
        texts.add(re.sub(r"\d+", "#", prompt))
    assert len(texts) == 1, texts                      # one wording per task
    sample = get_prompt(task, exs[0]["metadata"])
    if task == "87":
        assert "0-indexed" in sample and "block (0, 0)" in sample
        assert sample.endswith("Leave every other cell unchanged.")
        assert "background detail" not in sample
    if task == "88":
        assert sample.endswith("Leave every other cell unchanged.")
        assert "Ignore the background, whether it is a solid color or a pattern." in sample
        assert "fill every block strictly between them with that same color" in sample
    if task == "89":
        assert sample.endswith("Answer with the number.")


def test_87_prompt_never_hides_the_background_colour_behind_the_paint_colour(ex87):
    # the paint colour is a task given and IS named; nothing else is
    for ex in ex87[:50]:
        meta = ex["metadata"]
        prompt = get_prompt("87", meta)
        assert f"color {meta['paint_color']}" in prompt
        assert prompt.count("with color") == 1


# ---------------------------------------------------------------------------
# validators raise when the example is tampered with
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault", ["output", "index", "color", "shape", "plain",
                                   "phase", "starts", "forged"])
def test_87_validator_raises(fault):
    ex = deepcopy(generate("87", 11))
    if fault == "phase":
        ex["metadata"]["row_phase"] += 1
    elif fault == "starts":
        ex["metadata"]["col_line_starts"] = [s + 1 for s in ex["metadata"]["col_line_starts"]]
    elif fault == "forged":
        # a whole row painted in a colour that is not the line colour would offer a
        # second reading of where the lattice lines are
        row = ex["metadata"]["grid_h"] - 1
        paint = ex["metadata"]["paint_color"]
        ex["input"][row] = [paint] * ex["metadata"]["grid_w"]
        ex["output"][row] = [paint] * ex["metadata"]["grid_w"]
    if fault == "output":
        ex["output"][1][1] = (ex["output"][1][1] + 1) % 10
    elif fault == "index":
        ex["metadata"]["target_row"] = (ex["metadata"]["target_row"] + 1) % ex["metadata"]["block_rows"]
    elif fault == "color":
        ex["metadata"]["paint_color"] = (ex["metadata"]["paint_color"] + 1) % 10
    elif fault == "shape":
        ex["metadata"]["target_block_h"] += 1
    else:
        ex["metadata"]["n_plain_blocks"] += 1
    with pytest.raises(ValueError):
        module.validate_paint_block(ex)


@pytest.mark.parametrize("fault", ["output", "identity", "marked", "margin", "bg",
                                   "isolated", "clause"])
def test_88_validator_raises(fault):
    ex = deepcopy(generate("88", 3))
    assert not ex["metadata"]["identity_output"]
    if fault == "output":
        ex["output"] = deepcopy(ex["input"])
    elif fault == "identity":
        ex["metadata"]["identity_output"] = True
    elif fault == "marked":
        ex["metadata"]["n_marked_blocks"] += 1
    elif fault == "margin":
        ex["metadata"]["background_margin"] = 1.0
    elif fault == "isolated":
        ex["metadata"]["background_margin_isolated"] = 1.0
    elif fault == "clause":
        ex["metadata"]["color_clause_matters"] = not ex["metadata"]["color_clause_matters"]
    else:
        ex["metadata"]["bg_color"] = ex["metadata"]["group_colors"][0]
    with pytest.raises(ValueError):
        module.validate_fill_between(ex)


@pytest.mark.parametrize("fault", ["answer", "count", "target", "contains", "phase"])
def test_89_validator_raises(fault):
    ex = deepcopy(generate("89", 5))
    if fault == "phase":
        ex["metadata"]["col_phase"] = ex["metadata"]["col_phase"] + 1
    if fault == "answer":
        ex["output"] = str(int(ex["output"]) + 1)
    elif fault == "count":
        ex["metadata"]["n_full_blocks"] = int(ex["output"]) + 1
    elif fault == "target":
        ex["metadata"]["target_color"] = ex["metadata"]["line_color"]
    elif fault == "contains":
        ex["metadata"]["n_blocks_containing_target"] += 1
    with pytest.raises(ValueError):
        module.validate_count_filled(ex)


def test_88_rule_is_stable_on_its_own_output(ex88):
    for ex in ex88[:120]:
        grid = np.asarray(ex["output"])
        line, rows, cols = module.recover_lattice(np.asarray(ex["input"]))
        ground = module.infer_ground(np.asarray(ex["input"]), rows, cols)
        again = module.apply_fill_rule(grid, rows, cols, line, ground)
        assert np.array_equal(again, grid)


# ---------------------------------------------------------------------------
# task 81 shares generators/lattice.py; 87-89 must not move when it changes.
# The pin was refreshed for the approved 2026-09-16 scene-variety rebuild of 81
# (value 440e07083257716482b7b3c3765c9cd141e9c3fbe793d42abd1c4a1e94dd674d before
# that rebuild) and again for its 2026-09-16 two-cell chamber floor, uniform
# frame depth and canonical connectivity wording (previous value
# 6b0ae09792f50df2ed37b68b21e3212e96942e570bc6119f9cd86963711b8d96).
# ---------------------------------------------------------------------------

def test_81_hash_pin():
    digest = hashlib.sha256()
    for seed in range(100):
        ex = generate("81", seed)
        digest.update(json.dumps({"i": ex["input"], "o": ex["output"],
                                  "m": ex.get("metadata", {})}, sort_keys=True,
                                 default=str).encode())
    assert digest.hexdigest() == "b9c077df52ac6b0f82805899f9bd4260f90084a2a2757e72a2bc7eecb2aa809f"
