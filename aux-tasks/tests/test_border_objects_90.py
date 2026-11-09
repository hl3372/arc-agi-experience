"""Task 90: how many objects touch the outer border.

The oracle below is written from the prompt text alone (background inferred,
components under the stated connectivity, single cells dropped) with a numpy
labelling that shares no code with the generator's reader.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import os
import json
import re
import time

import numpy as np
import pytest

from generators import border_objects_90 as module
from prompts import get_prompt
from registry import generate

ORACLE_SEEDS = 400
SWEEP_SEEDS = 1200
PIECE_SEEDS = 800          # grids whose pieces are recounted from the picture


def _ex(seed):
    return generate("90", seed)


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------

def _connectivity_from_prompt(prompt):
    assert "Ignore the background, whether it is a solid color or a pattern." in prompt
    assert "at least two cells" in prompt
    assert "isolated single cells, whatever their color, are background texture" in prompt
    assert "always separate, even where they touch" in prompt
    assert prompt.endswith("Answer with the number.")
    match = re.search(r"[Tt]reat each (\d)-connected same-color group", prompt)
    return int(match.group(1))


def _label(mask, connectivity):
    """Row-major union-find labelling (a second implementation, not the module's)."""
    h, w = mask.shape
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    steps = [(-1, 0), (0, -1)] + ([(-1, -1), (-1, 1)] if connectivity == 8 else [])
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            parent.setdefault(r * w + c, r * w + c)
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    parent.setdefault(rr * w + cc, rr * w + cc)
                    union(r * w + c, rr * w + cc)
    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append((key // w, key % w))
    return list(groups.values())


def oracle(grid, prompt):
    connectivity = _connectivity_from_prompt(prompt)
    arr = np.asarray(grid)
    h, w = arr.shape
    # isolated single cells (no 8-neighbour of the same colour) are background texture
    lonely = np.ones((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.full((h, w), -1)
            src = arr[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
            shifted[max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)] = src
            lonely &= shifted != arr
    counts = Counter(arr[~lonely].ravel().tolist())
    background = counts.most_common(1)[0][0]
    answer = 0
    for color in np.unique(arr):
        if color == background:
            continue
        for cells in _label(arr == color, connectivity):
            if len(cells) < 2:
                continue
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                answer += 1
    return answer, int(background)


def test_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex(seed)
        answer, background = oracle(ex["input"], get_prompt("90", ex["metadata"]))
        assert str(answer) == ex["output"], (seed, answer, ex["output"])
        assert background == ex["metadata"]["bg_color"], seed


def test_determinism_and_cell_range():
    for seed in range(60):
        a, b = _ex(seed), _ex(seed)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), seed
        grid = np.asarray(a["input"])
        assert grid.min() >= 0 and grid.max() <= 9
        assert isinstance(a["output"], str) and a["output"].isdigit()


# ---------------------------------------------------------------------------
# sweep helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return [_ex(seed) for seed in range(SWEEP_SEEDS)]


def _meta(sweep):
    return [ex["metadata"] for ex in sweep]


def _pieces(grid, background, connectivity=4):
    """Same-colour components of >= 2 cells in the VISIBLE grid.

    This is what a reader sees.  The metadata's ``object_sizes`` holds one
    entry per FIGURE, so a corner-joined trio of three-cell pieces appears
    there as a single nine-cell object -- measuring the scene through that
    list is how a field of dominoes passed the old size tests.  Nothing below
    reads it.
    """
    arr = np.asarray(grid)
    h, w = arr.shape
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    todo = {(r, c) for r in range(h) for c in range(w) if arr[r, c] != background}
    out = []
    while todo:
        start = todo.pop()
        color = arr[start]
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in todo and arr[q] == color:
                    todo.discard(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) >= 2:
            out.append(comp)
    return out


@pytest.fixture(scope="module")
def piece_stats(sweep):
    """Per grid: (piece sizes, piece colours, area), all read off the picture.

    The background colour comes from the metadata, which ``test_independent_oracle``
    and ``test_background_is_identifiable_without_being_named`` have already
    checked against the grid itself.
    """
    stats = []
    for ex in sweep[:PIECE_SEEDS]:
        arr = np.asarray(ex["input"])
        comps = _pieces(arr, ex["metadata"]["bg_color"])
        sizes = [len(c) for c in comps]
        colors = {int(arr[c[0]]) for c in comps}
        stats.append((sizes, colors, int(arr.size)))
    return stats


def _heldout(feature, target):
    """Accuracy of the best per-value guess fitted on one half, scored on the other."""
    feature, target = list(feature), list(target)
    half = len(feature) // 2
    table = {}
    for f, t in zip(feature[:half], target[:half]):
        table.setdefault(f, Counter())[t] += 1
    fallback = Counter(target[:half]).most_common(1)[0][0]
    hits = sum(1 for f, t in zip(feature[half:], target[half:])
               if (table[f].most_common(1)[0][0] if f in table else fallback) == t)
    return hits / (len(feature) - half)


def _baseline(target):
    target = list(target)
    half = len(target) // 2
    guess = Counter(target[:half]).most_common(1)[0][0]
    return sum(1 for t in target[half:] if t == guess) / (len(target) - half)


# ---------------------------------------------------------------------------
# connectivity: balanced, per example, and invisible in the picture
# ---------------------------------------------------------------------------

def test_connectivity_is_balanced_and_stated(sweep):
    meta = _meta(sweep)
    share = np.mean([m["connectivity"] == 4 for m in meta])
    assert 0.44 <= share <= 0.56, share
    for ex in sweep[:120]:
        prompt = get_prompt("90", ex["metadata"])
        assert _connectivity_from_prompt(prompt) == ex["metadata"]["connectivity"]
        if ex["metadata"]["connectivity"] == 4:
            assert "cells that touch only at a corner are not connected" in prompt
        else:
            assert "horizontally, vertically, or diagonally" in prompt


def test_connectivity_not_predictable_from_the_input(sweep):
    """The scene is drawn before the rule: input-only classifiers stay at chance."""
    meta = _meta(sweep)
    rule = [m["connectivity"] for m in meta]
    base = _baseline(rule)
    # the two readings of the visible picture, its size and its palette
    features = [
        [f"{module.count_objects(ex['input'], 4)}_{module.count_objects(ex['input'], 8)}" for ex in sweep],
        [f"{m['grid_h']}x{m['grid_w']}_{m['n_colors']}" for m in meta],
        [f"{m['n_figures']}_{m['scene_mode']}" for m in meta],
    ]
    for feature in features:
        assert _heldout(feature, rule) <= base + 0.08, _heldout(feature, rule)


def test_rule_is_load_bearing(sweep):
    meta = _meta(sweep)
    differs = np.mean([m["differs_under_other_connectivity"] for m in meta])
    assert differs >= 0.40, differs
    # the other reading is genuinely computed, not asserted
    for ex in sweep[:80]:
        other = 8 if ex["metadata"]["connectivity"] == 4 else 4
        assert module.solve(ex["input"], other) == ex["metadata"]["answer_other_connectivity"]


# ---------------------------------------------------------------------------
# the answer does not leak
# ---------------------------------------------------------------------------

def test_object_count_does_not_give_the_answer(sweep):
    meta = _meta(sweep)
    answers = [m["n_border_objects"] for m in meta]
    base = _baseline(answers)
    from_objects = _heldout([m["n_objects"] for m in meta], answers)
    # a4 <= n_objects is structural; the bound keeps the residual small
    assert from_objects <= base + 0.07, (from_objects, base)
    with_rule = _heldout([f"{m['n_objects']}_{m['connectivity']}" for m in meta], answers)
    assert with_rule <= base + 0.08, (with_rule, base)


def test_size_does_not_give_the_answer(sweep):
    meta = _meta(sweep)
    answers = [m["n_border_objects"] for m in meta]
    base = _baseline(answers)
    from_size = _heldout([f"{m['grid_h']}x{m['grid_w']}" for m in meta], answers)
    assert from_size <= base + 0.04, (from_size, base)
    assert _heldout([(m["grid_h"] * m["grid_w"]) // 100 for m in meta], answers) <= base + 0.04


def test_no_size_conditioning_on_the_answer(sweep):
    """The legacy version used min_dim = 12 whenever the answer was >= 4."""
    meta = _meta(sweep)
    small = [m for m in meta if min(m["grid_h"], m["grid_w"]) < 12]
    assert np.mean([m["n_border_objects"] >= 4 for m in small]) >= 0.10


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_sides_are_flat_and_independent(sweep):
    meta = _meta(sweep)
    hs = np.array([m["grid_h"] for m in meta])
    ws = np.array([m["grid_w"] for m in meta])
    assert hs.min() == 5 and hs.max() == 30 and ws.min() == 5 and ws.max() == 30
    assert set(hs.tolist()) == set(range(5, 31))
    assert set(ws.tolist()) == set(range(5, 31))
    assert abs(np.corrcoef(hs, ws)[0, 1]) <= 0.10
    assert np.mean(hs == ws) <= 0.09                       # squares only by chance
    per_value = np.array([np.mean(hs == v) for v in range(5, 31)])
    assert per_value.max() <= 0.075 and per_value.min() >= 0.015
    area = hs * ws
    assert np.mean(area >= 700) >= 0.02                    # flat sides give ~4%
    assert area.max() >= 780


def test_colours(sweep):
    meta = _meta(sweep)
    counts = np.array([m["n_colors"] for m in meta])
    assert counts.min() == 2 and counts.max() == 10
    assert set(counts.tolist()) == set(range(2, 11))
    assert np.mean(counts >= 8) >= 0.18          # measured 0.22
    assert np.mean(counts == 2) <= 0.22
    assert np.mean([m["bg_color"] == 0 for m in meta]) >= 0.05   # 0 is an ordinary colour
    for m in meta:
        assert m["bg_color"] not in m["object_colors"]


def test_textures_are_sparse_and_inferable(sweep):
    meta = _meta(sweep)
    eligible = [m for m in meta if min(m["grid_h"], m["grid_w"]) >= 8]
    share = np.mean([bool(m["texture_colors"]) for m in eligible])
    assert 0.20 <= share <= 0.45, share
    kinds = set()
    for m in meta:
        if not m["texture_colors"]:
            continue
        assert m["texture_group"] == "sparse"        # a solid band would read as an object
        assert m["texture_named"] is False
        assert not set(m["texture_colors"]) & set(m["object_colors"] + [m["bg_color"]])
        kinds.add(m["texture_kind"])
    assert len(kinds) >= 5
    # every texture cell is isolated in the emitted grid
    for ex in sweep[:250]:
        colors = ex["metadata"]["texture_colors"]
        if not colors:
            continue
        arr = np.asarray(ex["input"])
        from common.backgrounds import isolated_cell_mask
        assert isolated_cell_mask(arr)[np.isin(arr, colors)].all()


def test_objects_scale_with_the_canvas(sweep, piece_stats):
    """Occupancy and piece sizes, both recomputed from the emitted grids."""
    area = np.array([a for _, _, a in piece_stats])
    occ = np.array([sum(sizes) / a for sizes, _, a in piece_stats])
    small = occ[area < 150].mean()
    big = occ[area >= 700].mean()
    assert big >= 0.15, big                       # no sea of background on 30x30
    assert big >= small - 0.08, (small, big)      # occupancy is flat by band
    assert small <= 0.33, small                   # ... and small grids are not packed
    sizes = np.array([s for sizes, _, _ in piece_stats for s in sizes])
    # a canvas that grew bought bigger bodies, not more of them
    big_sizes = np.array([s for sizes, _, a in piece_stats if a >= 400 for s in sizes])
    assert sizes.mean() >= 6.5, sizes.mean()                  # measured 8.3
    assert np.median(sizes) >= 5, np.median(sizes)            # measured 6
    assert big_sizes.mean() >= sizes.mean(), (big_sizes.mean(), sizes.mean())
    assert sizes.max() >= 60, sizes.max()                     # measured 210
    meta = _meta(sweep)
    families = Counter(k.replace("_clipped", "") for m in meta for k in m["object_kinds"])
    assert set(families) >= {"rect", "bar", "frame", "L", "staircase", "rect_blob", "square"}
    assert families.most_common(1)[0][1] / sum(families.values()) <= 0.40


def test_border_variety(sweep):
    meta = _meta(sweep)
    assert np.mean([m["has_span"] for m in meta]) >= 0.05          # edge to edge
    assert np.mean([m["has_clipped"] for m in meta]) >= 0.10       # cut by the border
    assert np.mean([m["has_contact"] for m in meta]) >= 0.10       # different colours touching
    assert np.mean([m["max_border_sides"] >= 2 for m in meta]) >= 0.20
    assert np.mean([m["n_interior_figures"] == 0 and m["n_border_objects"] > 0
                    for m in meta]) >= 0.04                        # all objects on the border
    assert np.mean([m["scene_mode"] == "plain" for m in meta]) >= 0.20


def test_answers(sweep):
    meta = _meta(sweep)
    answers = np.array([m["n_border_objects"] for m in meta])
    caps = np.array([m["answer_cap"] for m in meta])
    assert 0.05 <= np.mean(answers == 0) <= 0.18
    assert answers.max() >= 10
    hist = Counter(answers.tolist())
    assert max(hist.values()) / len(answers) <= 0.25
    assert (answers <= caps).all()
    ratio = answers / np.maximum(1, caps)
    assert 0.30 <= ratio.mean() <= 0.60            # near-flat inside each cap
    assert np.mean(answers == caps) >= 0.05


def _runs(cells):
    """(max runs in any row, max runs in any column) of a cell set."""
    by_row, by_col = {}, {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)
    def count(groups):
        worst = 0
        for keys in groups.values():
            keys = sorted(keys)
            worst = max(worst, 1 + sum(1 for a, b in zip(keys, keys[1:]) if b - a > 1))
        return worst
    return count(by_row), count(by_col)


def test_no_confetti(piece_stats):
    """The scene is bodies, not crumbs -- measured piece by piece on the grid.

    The September follow-up finding: ``per_piece`` was ``density * area /
    n_pieces`` with no floor while the piece count was free to reach
    ``area // 12``, so a full count forced every body down to two to four
    cells.  24.6% of pieces had <= 3 cells and a quarter of grids had a median
    piece of four cells or less.  The budget now follows the ink
    (``density * area / MIN_BODY``), so counts give and sizes do not.
    """
    sizes = np.array([s for sizes, _, _ in piece_stats for s in sizes])
    assert np.mean(sizes <= 3) <= 0.10, np.mean(sizes <= 3)          # measured 0.000
    tiny_share = np.array([np.mean([s <= 3 for s in sizes]) if sizes else 0.0
                           for sizes, _, _ in piece_stats])
    assert np.mean(tiny_share >= 0.60) <= 0.03, np.mean(tiny_share >= 0.60)
    medians = np.array([np.median(sizes) if sizes else 99.0
                        for sizes, _, _ in piece_stats])
    assert np.mean(medians >= 4) >= 0.95, np.mean(medians >= 4)      # measured 1.000
    # and no busy field of one colour: a scene with many objects has a palette
    mono_busy = [len(colors) == 1 and len(sizes) >= 8
                 for sizes, colors, _ in piece_stats]
    assert np.mean(mono_busy) <= 0.03, np.mean(mono_busy)            # measured 0.000


def test_objects_are_drawn_not_noise(sweep):
    """Every piece is one clean 4-connected body from the shape vocabulary.

    Rectangles, bars, frames, letters, staircases and unions of a few
    rectangles never break a row or a column into more than a few runs; a
    random walk or a carved blob would.
    """
    solid = thin = 0
    for ex in sweep[:250]:
        arr = np.asarray(ex["input"])
        bg = ex["metadata"]["bg_color"]
        texture = set(ex["metadata"]["texture_colors"])
        for color in np.unique(arr):
            if color == bg or color in texture:
                continue
            for cells in _label(arr == color, 4):
                assert len(cells) >= 2, (ex["metadata"], color)
                # every body is drawn at full size, never a leftover domino
                assert len(cells) >= 4, (ex["metadata"]["grid_h"], color, len(cells))
                row_runs, col_runs = _runs(cells)
                assert row_runs <= 3 and col_runs <= 3, (color, row_runs, col_runs)
                block = set(cells)
                if any((r + 1, c) in block and (r, c + 1) in block and (r + 1, c + 1) in block
                       for r, c in cells):
                    solid += 1
                else:
                    thin += 1
    assert solid / (solid + thin) >= 0.30, (solid, thin)


def test_color_separation_is_load_bearing(sweep):
    """"cells of different colors are always separate, even where they touch".

    Contacts used to land on 22% of grids and change the answer on 2.9%,
    because only one of the two touching objects reached the ring.  A planned
    contact is now aimed at the cells of ring-touching pieces, so both parties
    are objects the question counts and a colour-blind reader loses one.
    """
    meta = _meta(sweep)
    differs = np.mean([m["differs_under_color_blind"] for m in meta])
    assert differs >= 0.10, differs                       # measured 0.18
    assert np.mean([m["has_ring_contact"] for m in meta]) >= 0.08
    # the recorded alternative is recomputed, not asserted
    for ex in sweep[:80]:
        merged = module.solve_color_blind(ex["input"], ex["metadata"]["connectivity"])
        assert merged == ex["metadata"]["answer_color_blind"]
        assert (merged != int(ex["output"])) == ex["metadata"]["differs_under_color_blind"]


def test_background_is_identifiable_without_being_named(sweep):
    from common.backgrounds import infer_background
    for ex in sweep[:250]:
        info = infer_background(np.asarray(ex["input"]), margin=1.25)
        assert info["identifiable"], ex["metadata"]
        assert int(info["color"]) == ex["metadata"]["bg_color"]
        assert ex["metadata"]["background_margin"] >= 1.25


def test_prompt_policy(sweep):
    seen = set()
    for ex in sweep[:200]:
        prompt = get_prompt("90", ex["metadata"])
        seen.add(prompt)
        assert not re.search(r"\b(background|texture) (is|are|color)\b", prompt)
        assert "most frequent" not in prompt
        assert "outer border of the grid" in prompt
        assert prompt.endswith("Answer with the number.")
        # colours are never rendered, so no raw colour digit may appear
        assert not re.search(r"color \d", prompt)
    assert len(seen) == 2, seen        # exactly one wording per connectivity


def test_metadata_contract(sweep):
    required = {"grid_h", "grid_w", "n_colors", "connectivity", "n_objects",
                "n_border_objects", "differs_under_other_connectivity", "bg_color",
                "background_margin", "texture_group", "texture_kind", "texture_colors",
                "texture_named", "texture_coverage", "answer_cap", "scene_mode", "attempts",
                "answer_color_blind", "differs_under_color_blind", "has_ring_contact",
                "piece_sizes", "min_piece_cells", "piece_budget"}
    for ex in sweep[:200]:
        meta = ex["metadata"]
        assert required <= set(meta), required - set(meta)
        assert meta["connectivity"] in (4, 8)
        assert meta["n_border_objects"] <= meta["n_objects"]
        assert meta["n_border_objects"] == int(ex["output"])
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        # the piece list is one entry per 4-connected piece of the emitted grid
        assert meta["piece_sizes"] == sorted(
            len(c) for c in _pieces(ex["input"], meta["bg_color"]))
        assert meta["min_piece_cells"] >= 4
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep]) <= 2.0


# ---------------------------------------------------------------------------
# the validator is real
# ---------------------------------------------------------------------------

def test_validator_raises_on_a_tampered_example():
    ex = _ex(7)
    module.validate_example(deepcopy(ex))          # the honest example passes

    wrong = deepcopy(ex)
    wrong["output"] = str(int(wrong["output"]) + 1)
    with pytest.raises(ValueError):
        module.validate_example(wrong)

    flipped = deepcopy(ex)
    flipped["metadata"]["connectivity"] = 8 if flipped["metadata"]["connectivity"] == 4 else 4
    with pytest.raises(ValueError):
        module.validate_example(flipped)

    # paint an extra object onto the border: the recomputed count must move
    for seed in range(40):
        ex = _ex(seed)
        painted = deepcopy(ex)
        grid = [list(row) for row in painted["input"]]
        h, w = len(grid), len(grid[0])
        free = [c for c in range(10)
                if c not in set(v for row in grid for v in row)]
        if not free or w < 4:
            continue
        grid[0][1] = grid[0][2] = free[0]
        painted["input"] = grid
        with pytest.raises(ValueError):
            module.validate_example(painted)
        break
    else:
        pytest.fail("no seed offered a free colour for the tamper check")

    swapped = deepcopy(_ex(11))
    swapped["metadata"]["bg_color"] = (swapped["metadata"]["bg_color"] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(swapped)

    merged = deepcopy(_ex(13))
    merged["metadata"]["answer_color_blind"] += 1
    with pytest.raises(ValueError):
        module.validate_example(merged)

    listed = deepcopy(_ex(17))
    listed["metadata"]["piece_sizes"] = listed["metadata"]["piece_sizes"][:-1]
    with pytest.raises(ValueError):
        module.validate_example(listed)


def test_generation_speed():
    for seed in range(20):
        _ex(seed)
    times = []
    for seed in range(1000, 1300):
        start = time.perf_counter()
        _ex(seed)
        times.append((time.perf_counter() - start) * 1000)
    assert np.mean(times) <= 25.0, np.mean(times)
    assert np.percentile(times, 99) <= 120.0, np.percentile(times, 99)


# ---------------------------------------------------------------------------
# the siblings task 90 left behind in generators/misc.py are untouched
# ---------------------------------------------------------------------------

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "sibling_hashes_border_objects_90.json")


def test_misc_siblings_are_byte_identical():
    """Moving task 90 out of generators/misc.py changes nothing else in it.

    The 91 and 92 pins were refreshed for their approved 2026-09-15
    rebuilds, and 92b was dropped when it was merged into 92; the 83 pin was
    refreshed for its approved 2026-09-16 background pass.
    """
    with open(FIXTURE) as handle:
        expected = json.load(handle)
    assert set(expected["sha1"]) == {"31", "35", "56", "57", "58", "59", "80",
                                     "82", "83", "91", "92"}
    for tid, digest in sorted(expected["sha1"].items()):
        got = hashlib.sha1()
        for seed in range(expected["n_seeds"]):
            got.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert got.hexdigest() == digest, tid
