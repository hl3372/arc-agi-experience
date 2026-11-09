"""Tests for the shape-counting tasks 4 and 5 (``generators/object_counting.py``).

The oracle here is INDEPENDENT of the generator: its own union-find labelling
of 8-connected same-colour components, its own bounding-box-mask classifier
and its own pattern matcher (parsed from the ``#``/``.`` text the prompt
shows).  It re-solves every example from the visible grid plus the metadata
the prompt carries (background colour, texture colour, query, size filter,
query colour).

Also covered: classifier positives/negatives per category, answer / size /
category / background distributions, hard zero cases, prompt contents,
determinism, performance, and byte-identity of the active sibling tasks
68a/68b against pins refreshed during their approved September 13 rebuild.
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
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators.object_counting import (  # noqa: E402
    COUNT_FILL,
    COUNT_FILL_SMALL,
    DISTRACTOR_RESERVE,
    MAX_COUNT,
    PATTERN_TEXTS,
    classify_cells,
    pattern_cells,
    query_matches,
)
from prompts import ISOLATED_CELL_RULE, get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
TASKS = ("4", "5")
SIBLINGS = ("68a", "68b")  # 69a/69b were merged into these tasks on 2026-09-13.
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "object_counting_sibling_hashes.json")
LETTERS = list("ABCDEFGHIJ")

_CACHE: dict[str, list[dict]] = {}


def examples(tid: str) -> list[dict]:
    if tid not in _CACHE:
        _CACHE[tid] = [generate(tid, s) for s in range(SWEEP)]
    return _CACHE[tid]


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------


def _components8(grid: np.ndarray, bg_colors: set[int]) -> list[tuple[int, frozenset]]:
    """Union-find over non-background cells; same colour + 8-adjacency joins."""
    H, W = grid.shape
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cells = [(r, c) for r in range(H) for c in range(W) if int(grid[r, c]) not in bg_colors]
    for cell in cells:
        parent[cell] = cell
    for r, c in cells:
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            nb = (r + dr, c + dc)
            if nb in parent and grid[nb] == grid[r, c]:
                ra, rb = find((r, c)), find(nb)
                if ra != rb:
                    parent[ra] = rb
    groups: dict[tuple[int, int], list] = {}
    for cell in cells:
        groups.setdefault(find(cell), []).append(cell)
    return [(int(grid[cs[0]]), frozenset(cs)) for cs in groups.values()]


def _mask(cells) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    m = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
    for r, c in cells:
        m[r - min(rs), c - min(cs)] = True
    return m


def _mask_from_text(text: str) -> np.ndarray:
    rows = text.split("//")
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _same_up_to_symmetry(a: np.ndarray, b: np.ndarray) -> bool:
    for k in range(4):
        rot = np.rot90(a, k)
        for cand in (rot, np.fliplr(rot)):
            if cand.shape == b.shape and np.array_equal(cand, b):
                return True
    return False


def _oracle_kind(cells) -> tuple[str, int, int]:
    m = _mask(cells)
    h, w = m.shape
    if m.sum() < 2:
        return "single", h, w
    if m.all():
        return "rectangle", h, w
    if h >= 2 and w >= 2:
        for ri in (0, h - 1):
            for ci in (0, w - 1):
                t = np.zeros_like(m)
                t[ri, :] = True
                t[:, ci] = True
                if np.array_equal(t, m):
                    return "L", h, w
    if h == w and h % 2 == 1 and h >= 3:
        t = np.zeros_like(m)
        t[h // 2, :] = True
        t[:, h // 2] = True
        if np.array_equal(t, m):
            return "plus", h, w
        eye = np.eye(h, dtype=bool)
        if np.array_equal(eye | np.fliplr(eye), m):
            return "X", h, w
    if h >= 3 and w >= 3:
        t = np.ones_like(m)
        t[1:-1, 1:-1] = False
        if np.array_equal(t, m):
            return "frame", h, w
    return "other", h, w


def _isolated_cells(grid: np.ndarray) -> np.ndarray:
    """Own implementation of the isolated-cell rule: no 8-neighbour of the same colour."""
    H, W = grid.shape
    iso = np.ones(grid.shape, dtype=bool)
    for r in range(H):
        for c in range(W):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < H and 0 <= c + dc < W and grid[r + dr, c + dc] == grid[r, c]:
                        iso[r, c] = False
    return iso


def _structural_view(grid: np.ndarray, m: dict) -> np.ndarray:
    """What the prompt lets a solver do: the background colour and the NAMED texture
    colours are background; then every isolated single cell of any colour is background."""
    bg = int(m["background"])
    named = {int(c) for c in m["texture_colors"]} if m["texture_named"] else set()
    view = np.where(np.isin(grid, list({bg} | named)), bg, grid)
    return np.where(_isolated_cells(view) & (view != bg), bg, view)




def _inferred_background(grid: np.ndarray, named: set, margin: float = 1.25):
    """Own implementation of the policy reading: set aside the named texture colours and
    every isolated single cell, take the most frequent remaining colour; it must beat the
    runner-up by ``margin`` and own the largest 4-connected region.  Returns (colour, ok)."""
    consider = ~_isolated_cells(grid)
    if named:
        consider &= ~np.isin(grid, list(named))
    counts = np.bincount(grid[consider].ravel(), minlength=10)
    order = np.argsort(-counts, kind="stable")
    color, runner = int(order[0]), int(order[1])
    # named texture cells are background in the prompt's reading, so for the region test
    # they join the candidate colour and bridge regions split by bands / grid lines
    if named:
        grid = np.where(np.isin(grid, list(named)), color, grid)
        consider = consider | np.isin(grid, [color])
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    best_size, best_color = 0, -1
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not consider[r, c]:
                continue
            col = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            n = 0
            while stack:
                a, b = stack.pop()
                n += 1
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < H and 0 <= nb < W and not seen[na, nb] and consider[na, nb] and grid[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            if n > best_size:
                best_size, best_color = n, col
    ok = counts[color] > counts[runner] and counts[color] >= margin * counts[runner] and best_color == color
    return color, bool(ok)


def oracle_count(ex: dict, structural: bool = True) -> int:
    m = ex["metadata"]
    grid = np.array(ex["input"])
    if structural:
        grid = _structural_view(grid, m)
        bg = {int(m["background"])}
    else:
        bg = {int(m["background"])} | {int(c) for c in m["texture_colors"]}
    query = m["query_shape"]
    size_filter = m["size_filter"]
    query_color = m.get("query_color")
    count = 0
    for color, cells in _components8(grid, bg):
        if query_color is not None and color != query_color:
            continue
        kind, h, w = _oracle_kind(cells)
        if query == "rectangle":
            ok = kind == "rectangle"
        elif query == "non_square_rectangle":
            ok = kind == "rectangle" and h != w
        elif query == "square":
            ok = kind == "rectangle" and h == w
        elif query == "line":
            ok = kind == "rectangle" and min(h, w) == 1
        elif query.startswith("pattern:"):
            ok = _same_up_to_symmetry(_mask_from_text(query.split(":", 1)[1]), _mask(cells))
        else:
            ok = kind == query
        if ok and size_filter:
            ok = sorted((h, w)) == sorted(int(v) for v in size_filter)
        count += int(ok)
    return count


# ---------------------------------------------------------------------------
# Classifier unit tests
# ---------------------------------------------------------------------------

_CLASSIFIER_CASES = [
    ("#", "single"),
    ("##", "rectangle"),                      # 2x1 bar: rectangle, non-square, line
    ("##//##", "rectangle"),                  # square
    ("###//###", "rectangle"),                # 2-wide bar: rectangle but NOT a line
    ("#.//##", "L"),                          # smallest L (2x2)
    ("#..//#..//###", "L"),
    ("..#//..#//###", "L"),                   # mirror image is still an L
    ("###//#..//#..", "L"),
    ("##..//##..//####//####", "other"),      # thick-armed L
    ("###//##.", "pattern:##//##//#."),       # 2x2 block + tail is the P pattern, not an L
    (".#.//###//.#.", "plus"),
    ("..#..//..#..//#####//..#..//..#..", "plus"),
    (".##.//.##.//####//####//.##.//.##.", "other"),  # thick plus
    (".#.//###//.#.//.#.", "other"),          # unequal arms
    ("#.#//.#.//#.#", "X"),
    ("#...#//.#.#.//..#..//.#.#.//#...#", "X"),
    ("##..#//###.#//.###.//#.###//#..##", "other"),   # thick X
    (".##//.#.//#.#", "other"),               # X with one arm shortened
    ("###//#.#//###", "frame"),
    ("####//#..#//####", "frame"),
    ("###//#.#//##.", "other"),               # broken frame
    ("#####//#...#//#.#.#//#...#//#####", "other"),  # frame with a loose centre is 2 objects; ring+centre here is one shape -> other
    ("###//.#.", "pattern:###//.#."),
    ("##.//.##//.#.", "pattern:.##//##.//.#."),      # mirrored F pentomino
    ("#.//.#", "other"),                      # diagonal pair
    ("#..//.#.//..#", "other"),               # diagonal chain
    ("#..//##.//.##", "pattern:#..//##.//.##"),
]


@pytest.mark.parametrize("text,expected", _CLASSIFIER_CASES)
def test_classifier_labels(text, expected):
    assert classify_cells(pattern_cells(text))["label"] == expected


def test_query_predicates_and_size_filter():
    bar = classify_cells(pattern_cells("###"))
    assert query_matches(bar, "rectangle") and query_matches(bar, "non_square_rectangle")
    assert query_matches(bar, "line") and not query_matches(bar, "square")
    square = classify_cells(pattern_cells("##//##"))
    assert query_matches(square, "rectangle") and query_matches(square, "square")
    assert not query_matches(square, "non_square_rectangle") and not query_matches(square, "line")
    thick = classify_cells(pattern_cells("###//###"))
    assert query_matches(thick, "rectangle") and query_matches(thick, "non_square_rectangle")
    assert not query_matches(thick, "line")
    single = classify_cells(pattern_cells("#"))
    assert not any(query_matches(single, q) for q in ("rectangle", "line", "square", "L", "plus", "X", "frame"))
    rect = classify_cells(pattern_cells("##//##//##"))
    assert query_matches(rect, "rectangle", (2, 3)) and query_matches(rect, "rectangle", (3, 2))
    assert not query_matches(rect, "rectangle", (2, 2))
    for text in PATTERN_TEXTS:
        assert classify_cells(pattern_cells(text))["label"] == "pattern:" + text


# ---------------------------------------------------------------------------
# Sweep: independent recount, format, determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tid", TASKS)
def test_answer_equals_independent_recount(tid):
    for s, ex in enumerate(examples(tid)):
        assert ex["output"] == str(oracle_count(ex)), f"task {tid} seed {s}"
        # the colour-based reading (every texture colour is background) agrees
        assert ex["output"] == str(oracle_count(ex, structural=False)), f"task {tid} seed {s}"


@pytest.mark.parametrize("tid", TASKS)
def test_format_and_determinism(tid):
    for s, ex in enumerate(examples(tid)):
        grid = np.array(ex["input"])
        assert grid.ndim == 2 and 6 <= grid.shape[0] <= 30 and 6 <= grid.shape[1] <= 30
        assert grid.min() >= 0 and grid.max() <= 9
        assert isinstance(ex["output"], str) and ex["output"].isdigit()
        m = ex["metadata"]
        assert int(ex["output"]) == m["answer"] == m["n_matches"]
        assert m["connectivity"] == 8
        for key in ("query_shape", "query_shape_desc", "query_category", "size_filter", "background",
                    "background_style", "texture_colors", "n_objects", "class_counts", "answer",
                    "k_max", "k_first_sampled"):
            assert key in m, key
        if tid == "5":
            assert "query_color" in m
        if s < 200:
            assert generate(tid, s) == ex


@pytest.mark.parametrize("tid", TASKS)
def test_background_identifiable_without_being_named(tid):
    """The prompt never names the background, so on every example the most frequent colour
    (named texture colours and isolated cells set aside) must be the generator's background,
    by a clear margin and as the largest 4-connected region; and no object is as large as it."""
    for s, ex in enumerate(examples(tid)):
        m = ex["metadata"]
        grid = np.array(ex["input"])
        named = {int(c) for c in m["texture_colors"]} if m["texture_named"] else set()
        color, ok = _inferred_background(grid, named)
        assert ok and color == m["background"], f"task {tid} seed {s}: background not identifiable"
        assert m["background_margin"] >= 1.25, (tid, s, m["background_margin"])
        view = _structural_view(grid, m)
        bg_cells = int((view == m["background"]).sum())
        assert all(len(cells) < bg_cells for _, cells in _components8(view, {m["background"]})), (tid, s)


@pytest.mark.parametrize("tid", TASKS)
def test_objects_have_at_least_two_cells(tid):
    for s, ex in enumerate(examples(tid)):
        m = ex["metadata"]
        bg = {m["background"]} | set(m["texture_colors"])
        comps = _components8(np.array(ex["input"]), bg)
        assert all(len(cells) >= 2 for _, cells in comps), f"task {tid} seed {s}"
        assert len(comps) == m["n_objects"] >= 1


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tid", TASKS)
def test_answer_distribution(tid):
    exs = examples(tid)
    hist = Counter(int(ex["output"]) for ex in exs)
    share = {k: v / len(exs) for k, v in hist.items()}
    assert max(share.values()) <= 0.35, share
    assert 0.10 <= share.get(0, 0.0) <= 0.20, share
    assert len(hist) >= 5
    sized = [int(ex["output"]) for ex in exs if ex["metadata"]["size_filter"]]
    assert len(sized) >= 0.15 * len(exs)
    sized_hist = Counter(sized)
    assert sized_hist[1] / len(sized) <= 0.40, sized_hist
    assert sized_hist[0] > 0 and sized_hist[2] > 0 and sized_hist[3] > 0


@pytest.mark.parametrize("tid", TASKS)
def test_size_envelope(tid):
    exs = examples(tid)
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert np.mean(np.minimum(hs, ws) <= 12) >= 0.35
    assert np.mean(np.minimum(hs, ws) <= 10) >= 0.25
    assert np.mean(np.maximum(hs, ws) >= 21) >= 0.15
    assert np.mean(hs == ws) <= 0.15  # squares only by coincidence of independent draws
    assert len(set(hs.tolist())) >= 20 and len(set(ws.tolist())) >= 20


@pytest.mark.parametrize("tid", TASKS)
def test_scene_scales_with_area(tid):
    """Large canvases hold large scenes: occupancy stays near the small-grid level,
    object sides go past the old cap of 9, and the answer range reaches 7-8."""
    exs = examples(tid)
    occ = {"small": [], "large": []}
    big_side = 0
    for ex in exs:
        m = ex["metadata"]
        grid = np.array(ex["input"])
        bg = {m["background"]} | set(m["texture_colors"])
        area = grid.size
        cells = int(sum((grid == c).sum() for c in range(10) if c not in bg))
        (occ["small"] if area < 150 else occ["large"] if area >= 600 else []).append(cells / area)
        if area >= 600:
            for _, cs in _components8(grid, bg):
                rs = [r for r, _ in cs]
                cols = [c for _, c in cs]
                big_side = max(big_side, max(rs) - min(rs) + 1, max(cols) - min(cols) + 1)
    assert len(occ["large"]) >= 100
    assert np.median(occ["large"]) >= 0.6 * np.median(occ["small"]), (np.median(occ["small"]), np.median(occ["large"]))
    assert np.median(occ["large"]) >= 0.12
    assert big_side >= 11, big_side
    answers = Counter(int(ex["output"]) for ex in exs)
    assert answers[7] + answers[8] >= 5, answers
    # the answer range reaches 12 on canvases of 400+ cells with a real tail
    assert sum(v for k, v in answers.items() if k >= 9) >= 10, answers
    assert max(answers) >= 11, answers


# ---------------------------------------------------------------------------
# Footprint-based answer cap
# ---------------------------------------------------------------------------


def _smallest_match_bbox(query: str, size_filter) -> tuple[int, int]:
    """Independent statement of the smallest object that can match a query."""
    if size_filter:
        return int(size_filter[0]), int(size_filter[1])
    if query in ("rectangle", "non_square_rectangle", "line"):
        return 1, 2
    if query in ("square", "L"):
        return 2, 2
    if query in ("plus", "X", "frame"):
        return 3, 3
    m = _mask_from_text(query.split(":", 1)[1])
    return m.shape[0], m.shape[1]


def _expected_cap(area: int, query: str, size_filter) -> int:
    h, w = _smallest_match_bbox(query, size_filter)
    fill = COUNT_FILL_SMALL if area < 150 else COUNT_FILL
    return max(1, min(MAX_COUNT, int((fill * area - DISTRACTOR_RESERVE) / ((h + 1) * (w + 1)))))


@pytest.mark.parametrize("tid", TASKS)
def test_footprint_cap_holds(tid):
    """Every answer is within the cap derived from the smallest matching object's footprint,
    recomputed here from the query, the size filter and the grid size."""
    for s, ex in enumerate(examples(tid)):
        m = ex["metadata"]
        area = len(ex["input"]) * len(ex["input"][0])
        cap = _expected_cap(area, m["query_shape"], m["size_filter"])
        assert m["k_max"] == cap, (s, m["query_shape"], m["size_filter"], area, m["k_max"], cap)
        assert 0 <= int(ex["output"]) <= cap, (s, ex["output"], cap)


@pytest.mark.parametrize("tid", TASKS)
def test_small_canvas_answer_reach(tid):
    """Under 100 cells, small-footprint queries reach 3+ (the old cap was 2 for everything);
    plus / X / frame queries stay at their footprint bound (2 at most there)."""
    small = [ex for ex in examples(tid) if len(ex["input"]) * len(ex["input"][0]) < 100]
    assert len(small) >= 150, len(small)
    reach = Counter()
    for ex in small:
        cat = ex["metadata"]["query_category"]
        reach[cat] = max(reach[cat], int(ex["output"]))
    for cat in ("line", "rectangle", "non_square_rectangle"):
        assert reach[cat] >= 3, (cat, reach)
    assert reach["L"] >= 3 or reach["square"] >= 3, reach
    for cat in ("plus", "X", "frame"):
        assert reach[cat] <= 2, (cat, reach)
    over2 = sum(1 for ex in small if int(ex["output"]) >= 3) / len(small)
    assert over2 >= 0.06, over2


@pytest.mark.parametrize("tid", TASKS)
def test_small_canvas_occupancy_guard(tid):
    """The user does not want busy small grids: under 150 cells the object share of the
    canvas stays modest even though the answer range widened."""
    occ = []
    for ex in examples(tid):
        grid = np.array(ex["input"])
        if grid.size >= 150:
            continue
        m = ex["metadata"]
        bg = {m["background"]} | set(m["texture_colors"])
        cells = int(sum((grid == c).sum() for c in range(10) if c not in bg))
        occ.append(cells / grid.size)
    assert len(occ) >= 300
    assert np.median(occ) <= 0.30, np.median(occ)
    assert np.percentile(occ, 90) <= 0.45, np.percentile(occ, 90)


@pytest.mark.parametrize("tid", TASKS)
def test_answer_tail_not_thinned_by_placement(tid):
    """Scenes that fail to place are retried for the same k, so the realized answer
    distribution matches the first-sampled one (no silent shrinking of the tail)."""
    exs = examples(tid)
    first = Counter(ex["metadata"]["k_first_sampled"] for ex in exs)
    got = Counter(int(ex["output"]) for ex in exs)
    for k in range(MAX_COUNT + 1):
        assert abs(got[k] - first[k]) / len(exs) <= 0.01, (k, got[k], first[k])
    assert got[12] >= 1 and got[11] + got[12] >= 5, got


@pytest.mark.parametrize("tid", TASKS)
def test_plus_x_sizes_and_thick_frames(tid):
    """Plus / X matches are not dominated by side 3, and thick frames appear as near misses."""
    sides = Counter()
    thick = 0
    for ex in examples(tid):
        m = ex["metadata"]
        if "thick_frame" in m["distractor_kinds"]:
            thick += 1
        if m["query_category"] in ("plus", "X"):
            bg = {m["background"]} | set(m["texture_colors"])
            for _, cells in _components8(np.array(ex["input"]), bg):
                kind, h, w = _oracle_kind(cells)
                if kind == m["query_category"]:
                    sides[h] += 1
    total = sum(sides.values())
    # 0.62, not 0.58: the footprint cap lets plus / X queries ask for more matches, and
    # crowded counts shrink the match shapes, so side 3 sits at ~58% on task 5 (was ~50%
    # right after the flattening, ~60% before it).
    assert sides[3] / total <= 0.62, sides
    assert sides[9] + sides[11] + sides[13] >= 0.03 * total, sides
    assert thick >= 0.01 * len(examples(tid)), thick


@pytest.mark.parametrize("tid", TASKS)
def test_palette_reaches_ten_colours(tid):
    """The full 10-colour palette is reachable in one grid (background + 9 object colours)."""
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in examples(tid))
    assert max(seen) >= (10 if tid == "4" else 9), seen
    assert seen[2] + seen[3] + seen[4] >= 0.4 * sum(seen.values()), seen


@pytest.mark.parametrize("tid", TASKS)
def test_category_and_pattern_coverage(tid):
    exs = examples(tid)
    cats = Counter(ex["metadata"]["query_category"] for ex in exs)
    for cat in ("rectangle", "non_square_rectangle", "square", "line", "L", "plus", "X", "frame", "pattern"):
        assert cats[cat] / len(exs) >= 0.03, (cat, cats)
    patterns = {ex["metadata"]["query_shape"] for ex in exs if ex["metadata"]["query_category"] == "pattern"}
    assert len(patterns) >= 17  # 19 pattern texts, incl. the six 8-9 cell ones
    colors = Counter(ex["metadata"]["n_colors"] for ex in exs)
    assert len(colors) >= 3


@pytest.mark.parametrize("tid", TASKS)
def test_backgrounds(tid):
    """Textures come from common/backgrounds.py: sparse ones are isolated cells, unnamed,
    <= 20% of the free cells; solid / structured ones are named; texture colours never
    collide with the background or an object colour; nothing on a short side under 8."""
    exs = examples(tid)
    groups = Counter()
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        grid = np.array(ex["input"])
        bg = m["background"]
        groups[m["texture_group"]] += 1
        for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
            assert key in m, key
        if m["texture_group"] is None:
            assert m["background_style"] == "uniform" and m["texture_colors"] == [] and not m["texture_named"]
            continue
        tex_colors = m["texture_colors"]
        assert tex_colors and bg not in tex_colors and not (set(tex_colors) & set(m["object_colors"]))
        assert min(grid.shape) >= 8, f"seed {s}: texture on a small grid"
        assert m["background_style"] == m["texture_kind"]
        tex = np.isin(grid, tex_colors)
        n_free = int((grid == bg).sum()) + int(tex.sum())
        if m["texture_group"] == "sparse":
            assert not m["texture_named"]
            assert tex.sum() <= 0.20 * n_free + 1e-9, f"seed {s}"
            assert _isolated_cells(np.where(tex, grid, -1))[tex].all(), f"seed {s}: sparse texture not isolated"
        else:
            assert m["texture_named"]
            assert m["texture_group"] in ("solid", "structured")
            limit = 0.55 if m["texture_group"] == "solid" else 0.60
            assert tex.sum() <= limit * n_free + 1e-9, f"seed {s}"
            if m["texture_group"] == "structured":
                assert min(grid.shape) >= 12
    n = len(exs)
    assert 0.20 <= 1 - groups[None] / n <= 0.40, groups
    assert groups["sparse"] >= 0.10 * n and groups["solid"] >= 0.06 * n and groups["structured"] >= 5, groups


@pytest.mark.parametrize("tid", TASKS)
def test_zero_answers_have_near_misses(tid):
    for s, ex in enumerate(examples(tid)):
        if ex["output"] != "0":
            continue
        m = ex["metadata"]
        grid = np.array(ex["input"])
        comps = _components8(grid, {m["background"]} | set(m["texture_colors"]))
        assert comps, f"seed {s}: zero answer on an empty grid"
        query = m["query_shape"]
        if m["size_filter"]:
            relaxed = dict(m, size_filter=None)
            if tid == "5":
                relaxed.pop("query_color")
            assert oracle_count({"input": ex["input"], "metadata": relaxed}) >= 1, f"seed {s}"
        if tid == "5":
            qc = m["query_color"]
            assert any(color == qc for color, _ in comps), f"seed {s}: no object of the query color"
            other = dict(m)
            other.pop("query_color")
            assert oracle_count({"input": ex["input"], "metadata": other}) >= 1, f"seed {s}: no same-shape other-color object"
        del query


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tid", TASKS)
def test_prompt_contents(tid):
    for s, ex in enumerate(examples(tid)[:600]):
        m = ex["metadata"]
        prompt = get_prompt(tid, m, color_palette=LETTERS)
        # policy: the background is never named or defined; the prompt only says to ignore it
        assert "background is color" not in prompt and "background color" not in prompt
        assert prompt.count("Ignore the background, whether it is a solid color or a pattern.") == 1
        assert "including diagonally" in prompt
        assert "at least two cells" in prompt and ISOLATED_CELL_RULE in prompt
        # Policy: nothing about the background is named -- no background colour, no texture
        # colour, whatever the texture kind; the solver infers it from the grid.
        assert "background texture, not objects" not in prompt
        for c in m["texture_colors"]:
            assert f"color {LETTERS[c]}" not in prompt.replace(f"objects of color {LETTERS[c]}", "")
        if m["texture_group"] == "structured":
            assert m["texture_kind"] == "mosaic"
        if m["size_filter"]:
            a, b = m["size_filter"]
            assert f"Count only those whose bounding box is {a}x{b}" in prompt
        else:
            assert "Count only those" not in prompt
        if m["query_category"] == "pattern":
            text = m["query_shape"].split(":", 1)[1]
            assert text in prompt and "#" in text and "rotated or mirrored" in prompt
            assert "# = object cell" in prompt
            assert "0" not in text and "1" not in text
        if m["query_category"] == "frame":
            assert "other objects may lie inside the ring" in prompt
        if tid == "5":
            assert f"objects of color {LETTERS[m['query_color']]}" in prompt
        # every colour is rendered through the palette: no "color <digit>" survives
        assert not re.search(r"color \d", prompt), prompt


# ---------------------------------------------------------------------------
# Siblings, performance
# ---------------------------------------------------------------------------


def test_sibling_tasks_byte_identical():
    expected = json.load(open(FIXTURE))
    for tid in SIBLINGS:
        for s in range(300):
            digest = hashlib.sha1(json.dumps(generate(tid, s), sort_keys=True).encode()).hexdigest()
            assert digest == expected[tid][s], f"task {tid} seed {s} changed"


@pytest.mark.parametrize("tid", TASKS)
def test_performance(tid):
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate(tid, s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 5.0, statistics.mean(times)
    assert np.percentile(times, 99) <= 30.0
