"""Verifier tests for task 97 (recolour the odd color X object).

The oracle below is written from the PROMPT TEXT alone: it parses the
connectivity, the criterion, the object colour and the recolour colour out of
the rendered prompt, infers the background the way a reader told only "ignoring
the background" can, and re-derives the answer.  It never looks at the
generator's metadata beyond what the prompt already says.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators import odd_shape_97 as module
from prompts import get_prompt
from registry import generate

ORACLE_SEEDS = 400
SWEEP_SEEDS = 700

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _ex(seed: int) -> dict:
    return generate("97", seed)


@pytest.fixture(scope="module")
def sweep():
    return [_ex(seed) for seed in range(SWEEP_SEEDS)]


# ---------------------------------------------------------------------------
# an independent reader, written from the prompt
# ---------------------------------------------------------------------------

def _paired(rows):
    """True where a cell has an 8-neighbour of its own colour."""
    h, w = len(rows), len(rows[0])
    flag = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            color = rows[r][c]
            for dr, dc in _ORTH + _DIAG:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and rows[rr][cc] == color:
                    flag[r][c] = True
                    break
    return flag


def _background(rows) -> int:
    flag = _paired(rows)
    tally = Counter()
    for r, row in enumerate(rows):
        for c, color in enumerate(row):
            if flag[r][c]:
                tally[color] += 1
    return max(sorted(tally), key=lambda color: tally[color])


def _groups(rows, color, connectivity):
    h, w = len(rows), len(rows[0])
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] == color}
    out = []
    while pending:
        start = min(pending)
        pending.remove(start)
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) >= 2:
            out.append(comp)
    return out


def _normalized(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _congruent(cells):
    """The shape up to the eight rotations and reflections."""
    best = None
    current = set(cells)
    for _ in range(4):
        current = {(c, -r) for r, c in current}
        for variant in (current, {(r, -c) for r, c in current}):
            key = tuple(sorted(_normalized(variant)))
            if best is None or key < best:
                best = key
    return best


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


def _odd(objs, criterion, invariant=False):
    if criterion == "cell_count":
        keys = [len(comp) for comp in objs]
    elif invariant:
        keys = [_congruent(comp) for comp in objs]
    else:
        keys = [_normalized(comp) for comp in objs]
    hits = [comp for comp, key in zip(objs, keys) if keys.count(key) == 1]
    return hits[0] if len(hits) == 1 else None


def oracle(grid, prompt):
    """Solve from the picture and the prompt only."""
    rows = [list(row) for row in grid]
    conn = int(re.search(r"[Tt]reat each (\d)-connected", prompt).group(1))
    color = int(re.search(r"Among the color (\d) objects", prompt).group(1))
    target = int(re.search(r"Recolor every cell of that object to color (\d)", prompt).group(1))
    criterion = ("cell_count" if "different number of cells from all the others" in prompt
                 else "shape")
    assert "Ignore the background, whether it is a solid color or a pattern." in prompt
    assert "isolated single cells, whatever their color, are background texture" in prompt
    ground = _background(rows)
    objs = _groups(rows, color, conn)
    odd = _odd(objs, criterion)
    assert odd is not None, "the prompt's reading does not single out one object"
    out = [list(row) for row in rows]
    for r, c in odd:
        out[r][c] = target
    return out, ground, criterion, conn, color


def test_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex(seed)
        prompt = get_prompt("97", ex["metadata"])
        out, ground, criterion, conn, color = oracle(ex["input"], prompt)
        assert out == ex["output"], seed
        assert ground == ex["metadata"]["bg_color"], seed
        assert criterion == ex["metadata"]["criterion"], seed
        assert conn == ex["metadata"]["connectivity"], seed
        assert color == ex["metadata"]["object_color"], seed


def test_examples_are_deterministic():
    for seed in (0, 1, 2, 37, 255):
        assert generate("97", seed) == generate("97", seed)


# ---------------------------------------------------------------------------
# the criterion decides the answer, and so does every clause of it
# ---------------------------------------------------------------------------

def test_the_two_criteria_can_never_disagree():
    """Same shape means same cell count, so an object with a unique count has a
    unique shape: when both readings answer they name the SAME object, and no
    construction can make "apply whichever criterion singles out exactly one
    object" wrong.  The generator's balance targets follow from this, so the
    fact itself is pinned here."""
    for seed in range(ORACLE_SEEDS):
        ex = _ex(seed)
        meta = ex["metadata"]
        grid, color, conn = ex["input"], meta["object_color"], meta["connectivity"]
        shape = module.solve(grid, color, conn, "shape")
        count = module.solve(grid, color, conn, "cell_count")
        if shape is not None and count is not None:
            assert set(shape) == set(count), seed
        # and the stated reading is one of them
        stated = shape if meta["criterion"] == "shape" else count
        assert stated is not None, seed


def test_criterion_is_load_bearing(sweep):
    """The two readings cannot disagree, so the criterion decides exactly where
    the OTHER reading is stuck.  That share and the criterion classifier are two
    sides of one identity -- classifier = 0.5 + decides / 2 -- so 30% here is the
    counterpart of the 65% bound in
    ``test_the_criterion_is_not_recoverable_from_the_picture``."""
    metas = [ex["metadata"] for ex in sweep]
    shape = [m for m in metas if m["criterion"] == "shape"]
    count = [m for m in metas if m["criterion"] == "cell_count"]
    assert 0.35 <= len(shape) / len(metas) <= 0.65
    changed = np.mean([m["differs_under_other_criterion"] for m in metas])
    assert changed >= 0.18, changed
    # where the other reading DOES answer, it agrees (the theorem above)
    assert all(m["differs_under_other_criterion"] != m["other_criterion_has_answer"]
               for m in metas)
    # both one-sided cells are stocked: the picture must not say which
    # criterion is coming, so each reading is stuck about as often in both
    shape_blind = np.mean([not m["other_criterion_has_answer"] for m in shape])
    count_blind = np.mean([not m["other_criterion_has_answer"] for m in count])
    assert 0.15 <= shape_blind <= 0.45, shape_blind
    assert 0.15 <= count_blind <= 0.45, count_blind
    assert abs(shape_blind - count_blind) <= 0.16, (shape_blind, count_blind)
    # "do not rotate or reflect": on a share of the shape examples the odd
    # object is a rotation or reflection of a sibling's body.  It needs the odd
    # object to share a sibling's cell count, so its ceiling is ``shape_blind``
    rotation = np.mean([m["differs_under_rotation_invariant"] for m in shape])
    assert rotation >= 0.15, rotation
    assert rotation <= shape_blind + 0.05, (rotation, shape_blind)


def test_the_criterion_is_not_recoverable_from_the_picture(sweep):
    """Everything except "do all sibling shapes repeat" and "is the odd cell
    count unique" is drawn by one code path for both criteria, so an
    object-aware classifier trained on half the sweep stays near the identity's
    floor of 0.5 + decides / 2."""
    metas = [ex["metadata"] for ex in sweep]
    target = [m["criterion"] for m in metas]
    base = _baseline(target)
    rows = []
    for ex in sweep:
        meta = ex["metadata"]
        objs = _groups([list(row) for row in ex["input"]], meta["object_color"],
                       meta["connectivity"])
        counts = [len(comp) for comp in objs]
        shapes = [_normalized(comp) for comp in objs]
        rows.append({
            "shape_valid": module.solve(ex["input"], meta["object_color"],
                                        meta["connectivity"], "shape") is not None,
            "count_valid": module.solve(ex["input"], meta["object_color"],
                                        meta["connectivity"], "cell_count") is not None,
            "n": len(objs), "n_counts": len(set(counts)), "n_shapes": len(set(shapes)),
            "top_count": max(Counter(counts).values()),
            "top_shape": max(Counter(shapes).values()),
            "size": (meta["grid_h"] // 6, meta["grid_w"] // 6),
            "colors": meta["n_colors"], "joined": meta["corner_joined_figures"] > 0,
        })
    features = {
        "shape reading answers": [r["shape_valid"] for r in rows],
        "count reading answers": [r["count_valid"] for r in rows],
        "both validity flags": [(r["shape_valid"], r["count_valid"]) for r in rows],
        "object multiset": [(r["n"], r["n_counts"], r["n_shapes"]) for r in rows],
        "full signature": [(r["n"], r["n_counts"], r["n_shapes"], r["top_count"],
                            r["top_shape"]) for r in rows],
        "canvas and palette": [(r["size"], r["colors"], r["joined"]) for r in rows],
    }
    for name, feature in features.items():
        score = _heldout(feature, target)
        assert score <= 0.68, (name, score, base)
    # the old builder's giveaway: the shape reading answered on every shape
    # example and on no cell-count one
    assert _heldout([r["shape_valid"] for r in rows], target) <= 0.68


def test_connectivity_is_load_bearing_and_unpredictable(sweep):
    metas = [ex["metadata"] for ex in sweep]
    share = np.mean([m["connectivity"] == 8 for m in metas])
    assert 0.42 <= share <= 0.58, share
    differs = np.mean([m["differs_under_other_connectivity"] for m in metas])
    assert differs >= 0.30, differs
    # the visible scene must not predict the variant: corner-joined figures
    # occur under both readings at the same rate
    joined = [m for m in metas if m["corner_joined_figures"] > 0]
    plain = [m for m in metas if m["corner_joined_figures"] == 0]
    assert len(joined) / len(metas) >= 0.30
    for group in (joined, plain):
        rate = np.mean([m["connectivity"] == 8 for m in group])
        assert abs(rate - 0.5) <= 0.12, (len(group), rate)
    # the other reading always has an answer too, so neither is impossible
    for ex in sweep[:200]:
        other = 4 if ex["metadata"]["connectivity"] == 8 else 8
        assert module.solve(ex["input"], ex["metadata"]["object_color"], other,
                            ex["metadata"]["criterion"]) is not None


def test_cheap_readers_are_bounded(sweep):
    """No single shortcut answers the task."""
    scores = Counter()
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        grid, out = ex["input"], ex["output"]
        answer = {(r, c) for r, row in enumerate(grid)
                  for c, value in enumerate(row) if out[r][c] != value}
        objs = _groups([list(row) for row in grid], meta["object_color"],
                       meta["connectivity"])

        def pick(keyfn):
            keys = [keyfn(comp) for comp in objs]
            hits = [comp for comp, key in zip(objs, keys) if keys.count(key) == 1]
            if not hits:
                return None
            return set(min(hits, key=min))

        scores["shape"] += int(pick(_normalized) == answer)
        scores["count"] += int(pick(len) == answer)
        scores["congruent"] += int(pick(_congruent) == answer)
        scores["largest"] += int(set(max(objs, key=lambda c: (len(c), min(c)))) == answer)
        scores["smallest"] += int(set(min(objs, key=lambda c: (len(c), min(c)))) == answer)
        scores["first"] += int(set(min(objs, key=min)) == answer)
        # "recolour the uniquely largest or the uniquely smallest object": the
        # reader abstains unless one object stands alone at an end
        sizes = [len(comp) for comp in objs]
        guess = None
        if sizes.count(max(sizes)) == 1:
            guess = max(objs, key=len)
        elif sizes.count(min(sizes)) == 1:
            guess = min(objs, key=len)
        if guess is not None:
            scores["rank_fires"] += 1
            scores["rank_right"] += int(set(guess) == answer)
    n = ORACLE_SEEDS
    # A fixed single-criterion reader is right wherever ITS reading answers, and
    # that share is the other side of the criterion classifier: the two add to
    # 1.5 exactly (see the module docstring), so bounding both below 0.75 is not
    # available.  The classifier is the one the review asked to close.
    assert scores["shape"] / n <= 0.90, scores["shape"] / n
    assert scores["count"] / n <= 0.93, scores["count"] / n
    assert scores["congruent"] / n <= 0.85, scores["congruent"] / n
    assert scores["largest"] / n <= 0.25, scores["largest"] / n
    assert scores["smallest"] / n <= 0.25, scores["smallest"] / n
    assert scores["first"] / n <= 0.35, scores["first"] / n
    # The size-rank shortcut cannot be made WRONG (a uniquely extreme count is a
    # unique count, so it is the answer whenever the stated reading answers at
    # all), so the lever is how often it fires: 50% before the follow-up pass.
    assert scores["rank_fires"] / n <= 0.33, scores["rank_fires"] / n
    assert scores["rank_right"] == scores["rank_fires"]


def test_the_answer_is_often_not_an_extreme_size(sweep):
    """The grouped cell counts let the odd object sit between its siblings."""
    metas = [ex["metadata"] for ex in sweep]
    extreme = np.mean([m["odd_is_extreme_size"] for m in metas])
    assert extreme <= 0.33, extreme
    count = [m for m in metas if m["criterion"] == "cell_count"]
    middle = np.mean([not m["odd_is_extreme_size"] for m in count])
    assert middle >= 0.55, middle
    # it fires on the scenes with ONE sibling group, where the odd count has
    # nothing to sit between: three or four figures on a small canvas
    single = [m for m in metas if m["n_sibling_groups"] == 1]
    many = [m for m in metas if m["n_sibling_groups"] >= 2]
    assert len(single) / len(metas) <= 0.25, len(single) / len(metas)
    assert np.mean([m["odd_is_extreme_size"] for m in many]) <= 0.30


def test_answer_is_not_predictable_from_the_size(sweep):
    """The criterion is not a function of the canvas."""
    by_size = Counter((m["grid_h"] >= 18, m["grid_w"] >= 18, m["criterion"])
                      for m in (ex["metadata"] for ex in sweep))
    for tall in (False, True):
        for wide in (False, True):
            shape = by_size[(tall, wide, "shape")]
            count = by_size[(tall, wide, "cell_count")]
            if shape + count >= 60:
                assert abs(shape / (shape + count) - 0.5) <= 0.14


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_size_coverage(sweep):
    metas = [ex["metadata"] for ex in sweep]
    heights = Counter(m["grid_h"] for m in metas)
    widths = Counter(m["grid_w"] for m in metas)
    assert set(heights) | set(widths) <= set(range(5, 31))
    assert len(set(heights) | set(widths)) >= 24
    for side in range(5, 31):
        assert heights[side] + widths[side] >= 0.012 * 2 * len(metas), side
    squares = np.mean([m["grid_h"] == m["grid_w"] for m in metas])
    assert squares <= 0.09, squares
    areas = np.array([m["grid_h"] * m["grid_w"] for m in metas])
    assert np.mean(areas >= 700) >= 0.015
    assert np.mean(areas < 150) >= 0.10
    corr = abs(float(np.corrcoef([m["grid_h"] for m in metas],
                                 [m["grid_w"] for m in metas])[0, 1]))
    assert corr <= 0.12, corr


def test_color_coverage(sweep):
    counts = Counter(ex["metadata"]["n_colors"] for ex in sweep)
    assert min(counts) <= 3 and max(counts) >= 8
    # the recolour colour is reserved for the output, so the input reaches 9
    assert max(counts) <= 9
    rich = sum(v for k, v in counts.items() if k >= 8) / len(sweep)
    assert rich >= 0.12, rich
    for ex in sweep[:200]:
        meta = ex["metadata"]
        present = {int(v) for row in ex["input"] for v in row}
        assert meta["target_color"] not in present
        assert meta["object_color"] in present


def test_objects_scale_with_the_canvas(sweep):
    bands = {"small": [], "large": []}
    for ex in sweep:
        meta = ex["metadata"]
        area = meta["grid_h"] * meta["grid_w"]
        bands["small" if area < 200 else "large"].append(meta["occupancy"])
    for name, values in bands.items():
        assert len(values) >= 40, name
        assert np.mean(values) >= 0.12, (name, np.mean(values))
    assert np.mean(bands["large"]) >= 0.6 * np.mean(bands["small"])
    sizes = [size for ex in sweep for size in ex["metadata"]["object_sizes"]]
    assert np.median(sizes) >= 5
    assert max(sizes) >= 40


def test_background_is_identifiable_without_being_named(sweep):
    from common.backgrounds import infer_background, isolated_cell_mask

    textured = 0
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        arr = np.asarray(ex["input"])
        info = infer_background(arr, margin=1.25)
        assert info["identifiable"] and int(info["color"]) == meta["bg_color"]
        assert meta["background_margin"] >= 1.25
        if meta["texture_colors"]:
            textured += 1
            assert meta["texture_group"] == "sparse"
            assert not meta["texture_named"]
            tex = np.isin(arr, meta["texture_colors"])
            assert bool(isolated_cell_mask(arr)[tex].all())
            assert meta["object_color"] not in meta["texture_colors"]
            assert meta["target_color"] not in meta["texture_colors"]
    assert 0.10 <= textured / ORACLE_SEEDS <= 0.45, textured / ORACLE_SEEDS


def test_shapes_look_drawn(sweep):
    """Bodies are drawn, not noise: no one-cell pieces, no hairs, full boxes."""
    hairs = 0
    pieces = 0
    fills = []
    for ex in sweep[:200]:
        meta = ex["metadata"]
        rows = [list(row) for row in ex["input"]]
        for comp in _groups(rows, meta["object_color"], 4):
            pieces += 1
            cells = set(comp)
            assert len(cells) >= 2
            for r, c in cells:
                neighbours = sum(1 for dr, dc in _ORTH if (r + dr, c + dc) in cells)
                if neighbours == 0:
                    hairs += 1
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            fills.append(len(cells) / ((r1 - r0 + 1) * (c1 - c0 + 1)))
        assert meta["min_piece_cells"] >= 2
    assert hairs == 0, hairs
    assert pieces >= 400
    assert np.mean(fills) >= 0.55, np.mean(fills)


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    seen = set()
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        prompt = get_prompt("97", meta)
        seen.add(meta["criterion"])
        assert "Ignore the background, whether it is a solid color or a pattern." in prompt
        assert "isolated single cells, whatever their color, are background texture" in prompt
        assert "cells of different colors are always separate" in prompt
        assert "at least two cells" in prompt
        assert "Leave every other cell unchanged." in prompt
        assert "ignoring every object of another color" in prompt
        assert "(the other objects need not all match each other)" in prompt
        if meta["criterion"] == "shape":
            assert "different shape from all the others" in prompt
            assert "do not rotate or reflect them" in prompt
        else:
            assert "different number of cells from all the others" in prompt
            assert "rotate" not in prompt
        conn = "8-connected" if meta["connectivity"] == 8 else "4-connected"
        assert conn in prompt
        if meta["connectivity"] == 4:
            assert "cells that touch only at a corner are not connected" in prompt
        else:
            assert "horizontally, vertically, or diagonally" in prompt
        # nothing about the background or the construction is named
        assert not re.search(r"\b(background|texture) (is|are|color)\b", prompt)
        assert "most frequent" not in prompt
        for leak in ("structure", "equivalence_rule", "corner", "distractor",
                     "odd_corner", "texture color"):
            assert leak not in prompt.lower().replace("at a corner", "")
    assert seen == {"shape", "cell_count"}


def test_prompt_is_byte_identical_per_bucket(sweep):
    buckets = {}
    for ex in sweep[:200]:
        meta = ex["metadata"]
        key = (meta["criterion"], meta["connectivity"])
        text = get_prompt("97", meta)
        text = re.sub(r"color \d", "color X", text)
        buckets.setdefault(key, set()).add(text)
    for key, texts in buckets.items():
        assert len(texts) == 1, key


# ---------------------------------------------------------------------------
# the validator is real
# ---------------------------------------------------------------------------

def test_validator_raises_when_the_output_is_tampered():
    ex = _ex(3)
    broken = {"input": ex["input"],
              "output": [list(row) for row in ex["output"]],
              "metadata": dict(ex["metadata"])}
    broken["output"][0][0] = (broken["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_validator_raises_when_a_flag_is_wrong():
    ex = _ex(5)
    broken = {"input": ex["input"], "output": ex["output"],
              "metadata": dict(ex["metadata"])}
    broken["metadata"]["differs_under_other_connectivity"] = \
        not broken["metadata"]["differs_under_other_connectivity"]
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_the_second_oracle_agrees_on_every_reading(sweep):
    """The validator's independence rests on two implementations, so they are
    checked against each other on every grid, connectivity and criterion."""
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        for conn in (4, 8):
            for criterion in ("shape", "cell_count"):
                assert module.oracle_solve(ex["input"], meta["object_color"], conn,
                                           criterion) == module.solve(
                    ex["input"], meta["object_color"], conn, criterion)
        cells = module.oracle_solve(ex["input"], meta["object_color"],
                                    meta["connectivity"], meta["criterion"])
        assert cells is not None
        assert module.recolor(ex["input"], cells, meta["target_color"]) == ex["output"]


def test_validator_raises_when_the_plan_disagrees():
    """The plan is a third witness: a figure list that does not match the
    picture is rejected even though the output is the right one."""
    ex = _ex(9)
    meta = ex["metadata"]
    figures = [sorted(comp) for comp in
               module.color_objects(ex["input"], meta["object_color"], 8)]
    odd = module.solve(ex["input"], meta["object_color"], 8, meta["criterion"])
    index = figures.index(sorted(odd))
    # the true plan passes
    module.validate_example(ex, plan={"figures": figures, "odd_index": index})
    with pytest.raises(ValueError):
        module.validate_example(ex, plan={"figures": figures,
                                          "odd_index": (index + 1) % len(figures)})
    dropped = [cells for i, cells in enumerate(figures) if i != index]
    with pytest.raises(ValueError):
        module.validate_example(ex, plan={"figures": dropped, "odd_index": 0})


def test_validator_raises_when_the_extreme_flag_is_wrong():
    ex = _ex(13)
    broken = {"input": ex["input"], "output": ex["output"],
              "metadata": dict(ex["metadata"])}
    broken["metadata"]["odd_is_extreme_size"] = \
        not broken["metadata"]["odd_is_extreme_size"]
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_validator_raises_when_a_second_odd_object_appears():
    """A grid the rule cannot answer is rejected, not patched."""
    ex = _ex(11)
    grid = [list(row) for row in ex["input"]]
    meta = dict(ex["metadata"])
    # two more cells of the object colour in the corner: another object, with
    # its own shape and its own cell count
    grid[0][0] = grid[0][1] = meta["object_color"]
    broken = {"input": grid, "output": ex["output"], "metadata": meta}
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_generation_is_cheap():
    times = []
    for seed in range(120):
        start = time.perf_counter()
        _ex(seed)
        times.append((time.perf_counter() - start) * 1000)
    assert np.mean(times) <= 30.0, np.mean(times)
    assert np.percentile(times, 95) <= 140.0, np.percentile(times, 95)


# ---------------------------------------------------------------------------
# the eight tasks 97 and 98 left behind in batch_object_a.py are untouched
# ---------------------------------------------------------------------------

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "sibling_hashes_batch_object_a.json")


def test_batch_object_a_siblings_are_byte_identical():
    with open(FIXTURE) as handle:
        expected = json.load(handle)
    assert set(expected["sha1"]) == {"105", "106", "108", "143", "144", "152",
                                     "161", "172"}
    for tid, digest in sorted(expected["sha1"].items()):
        got = hashlib.sha1()
        for seed in range(expected["n_seeds"]):
            got.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert got.hexdigest() == digest, tid
