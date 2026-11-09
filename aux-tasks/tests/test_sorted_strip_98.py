"""Verifier tests for task 98 (sort the objects and lay them out in a strip).

The oracle below is written from the PROMPT TEXT alone: it parses the
connectivity, the sort key, the strip axis, the start cell and the gap out of
the rendered prompt, infers the background the way a reader told only "ignoring
the background" can, and builds the whole output grid.  The sibling tasks that
stayed in ``generators/core2_extra/batch_object_a.py`` are pinned in
``tests/test_odd_shape_97.py``.
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators import sorted_strip_98 as module
from prompts import get_prompt
from registry import generate

ORACLE_SEEDS = 400
SWEEP_SEEDS = 800

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _ex(seed: int) -> dict:
    return generate("98", seed)


@pytest.fixture(scope="module")
def sweep():
    return [_ex(seed) for seed in range(SWEEP_SEEDS)]


# ---------------------------------------------------------------------------
# an independent reader, written from the prompt
# ---------------------------------------------------------------------------

def _background(rows) -> int:
    h, w = len(rows), len(rows[0])
    tally = Counter()
    for r in range(h):
        for c in range(w):
            color = rows[r][c]
            for dr, dc in _ORTH + _DIAG:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and rows[rr][cc] == color:
                    tally[color] += 1
                    break
    return max(sorted(tally), key=lambda color: tally[color])


def _objects(rows, ground, connectivity):
    h, w = len(rows), len(rows[0])
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] != ground}
    out = []
    while pending:
        start = min(pending)
        pending.remove(start)
        color = rows[start[0]][start[1]]
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending and rows[q[0]][q[1]] == color:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) >= 2:
            out.append({"cells": comp, "color": color,
                        "box": (min(r for r, _ in comp), min(c for _, c in comp),
                                max(r for r, _ in comp), max(c for _, c in comp))})
    return out


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


def _value(obj, key):
    r0, c0, r1, c1 = obj["box"]
    if key == "cells":
        return len(obj["cells"])
    if key == "rows":
        return r1 - r0 + 1
    return c1 - c0 + 1


def oracle(grid, prompt):
    """Solve from the picture and the prompt only."""
    rows = [list(row) for row in grid]
    conn = int(re.search(r"[Tt]reat each (\d)-connected", prompt).group(1))
    if "the number of cells each object contains" in prompt:
        key = "cells"
    elif "bounding-box height" in prompt:
        key = "rows"
    else:
        key = "cols"
    start = re.search(r"top-left corner at row (\d+), column (\d+)", prompt)
    start_r, start_c = int(start.group(1)), int(start.group(2))
    step = re.search(r"bounding box (to the right of|below) the previous object's "
                     r"bounding box, leaving exactly (\d+) empty (columns?|rows?) "
                     r"between the two bounding boxes", prompt)
    gap = int(step.group(2))
    axis = "horizontal" if step.group(1) == "to the right of" else "vertical"
    assert step.group(3) in (("column", "columns") if axis == "horizontal"
                             else ("row", "rows"))
    assert step.group(3).endswith("s") == (gap != 1)
    assert "The output has the same dimensions as the input." in prompt
    assert "is plain background" in prompt
    assert "0-indexed from the top-left corner" in prompt

    ground = _background(rows)
    objs = _objects(rows, ground, conn)
    ordered = sorted(objs, key=lambda o: (_value(o, key), o["box"][0], o["box"][1]))
    h, w = len(rows), len(rows[0])
    out = [[ground] * w for _ in range(h)]
    cur_r, cur_c = start_r, start_c
    for obj in ordered:
        r0, c0, r1, c1 = obj["box"]
        for r, c in obj["cells"]:
            out[cur_r + r - r0][cur_c + c - c0] = obj["color"]
        if axis == "horizontal":
            cur_c += (c1 - c0 + 1) + gap
        else:
            cur_r += (r1 - r0 + 1) + gap
    return out, ground, conn, key, axis, gap


def test_independent_oracle():
    keys = {"cells": "cell_count", "rows": "bbox_height", "cols": "bbox_width"}
    for seed in range(ORACLE_SEEDS):
        ex = _ex(seed)
        meta = ex["metadata"]
        out, ground, conn, key, axis, gap = oracle(ex["input"], get_prompt("98", meta))
        assert out == ex["output"], seed
        assert ground == meta["bg_color"], seed
        assert conn == meta["connectivity"], seed
        assert keys[key] == meta["sort_key"], seed
        assert axis == meta["strip_axis"] and gap == meta["gap"], seed


def test_examples_are_deterministic():
    for seed in (0, 1, 2, 37, 255):
        assert generate("98", seed) == generate("98", seed)


def test_output_has_the_input_dimensions_and_only_placed_objects(sweep):
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        grid, out = ex["input"], ex["output"]
        assert len(out) == len(grid) and len(out[0]) == len(grid[0])
        # every non-background cell of the output belongs to a placed object,
        # and the input's texture is gone
        painted = Counter(v for row in out for v in row if v != meta["bg_color"])
        objs = module.objects(grid, meta["connectivity"], meta["bg_color"])
        wanted = Counter()
        for obj in objs:
            wanted[obj["color"]] += len(obj["cells"])
        assert painted == wanted, meta["attempts"]
        if meta["texture_colors"]:
            assert not (set(meta["texture_colors"]) & set(painted))


# ---------------------------------------------------------------------------
# the clauses decide the answer
# ---------------------------------------------------------------------------

def test_sort_key_is_load_bearing(sweep):
    metas = [ex["metadata"] for ex in sweep]
    keys = Counter(m["sort_key"] for m in metas)
    assert set(keys) == {"cell_count", "bbox_height", "bbox_width"}
    assert min(keys.values()) >= 0.22 * len(metas)
    changed = np.mean([m["differs_under_other_key"] for m in metas])
    assert changed >= 0.62, changed


def test_strip_axis_is_load_bearing_and_balanced(sweep):
    metas = [ex["metadata"] for ex in sweep]
    axes = Counter(m["strip_axis"] for m in metas)
    assert min(axes.values()) >= 0.42 * len(metas)
    # the objects are laid out along the stated axis, so the other one always
    # paints a different picture
    assert all(m["differs_under_other_axis"] for m in metas)


def test_axis_and_key_are_not_recoverable_from_the_picture(sweep):
    """The whole scene is drawn before the axis and the key exist, and the
    object set is sized so that BOTH strips fit, so a classifier trained on half
    the sweep stays at chance.  The first rebuild sized the objects along the
    chosen axis and scored 92% here."""
    metas = [ex["metadata"] for ex in sweep]
    rows = []
    for ex in sweep:
        meta = ex["metadata"]
        objs = module.objects(ex["input"], 8, meta["bg_color"])
        heights = [obj["box"][2] - obj["box"][0] + 1 for obj in objs]
        widths = [obj["box"][3] - obj["box"][1] + 1 for obj in objs]
        counts = [len(obj["cells"]) for obj in objs]
        gap = meta["gap"]
        rows.append({
            "n": len(objs),
            "tall": sum(1 for h, w in zip(heights, widths) if h > w),
            "wide": sum(1 for h, w in zip(heights, widths) if w > h),
            "aspect": int(round(2 * float(np.mean([h / max(1, w)
                                                   for h, w in zip(heights, widths)])))),
            "fit_h": int(sum(heights) + gap * (len(objs) - 1)
                         + meta["strip_start_row"] <= meta["grid_h"]),
            "fit_w": int(sum(widths) + gap * (len(objs) - 1)
                         + meta["strip_start_col"] <= meta["grid_w"]),
            "fill_h": int(4 * sum(heights) / meta["grid_h"]),
            "fill_w": int(4 * sum(widths) / meta["grid_w"]),
            "tie_h": int(len(set(heights)) != len(heights)),
            "tie_w": int(len(set(widths)) != len(widths)),
            "tie_c": int(len(set(counts)) != len(counts)),
            "size": (meta["grid_h"] // 6, meta["grid_w"] // 6),
        })
    axis = [m["strip_axis"] for m in metas]
    base = _baseline(axis)
    axis_features = {
        "aspect ratios": [r["aspect"] for r in rows],
        "tall / wide counts": [(r["tall"], r["wide"]) for r in rows],
        "which strip fits": [(r["fit_h"], r["fit_w"]) for r in rows],
        "extent sums vs canvas": [(r["fill_h"], r["fill_w"]) for r in rows],
        "everything": [(r["fit_h"], r["fit_w"], r["tall"], r["wide"], r["size"])
                       for r in rows],
    }
    for name, feature in axis_features.items():
        score = _heldout(feature, axis)
        assert score <= base + 0.06, (name, score, base)
    # every scene fits BOTH ways, which is what makes the coin fair
    assert all(r["fit_h"] and r["fit_w"] for r in rows)

    key = [m["sort_key"] for m in metas]
    base = _baseline(key)
    key_features = {
        "tie pattern": [(r["tie_h"], r["tie_w"], r["tie_c"]) for r in rows],
        "tie pattern and count": [(r["tie_h"], r["tie_w"], r["tie_c"], r["n"])
                                  for r in rows],
        "object count": [r["n"] for r in rows],
        "canvas": [r["size"] for r in rows],
    }
    for name, feature in key_features.items():
        score = _heldout(feature, key)
        assert score <= base + 0.06, (name, score, base)


def test_ties_occur_and_the_tie_break_decides(sweep):
    metas = [ex["metadata"] for ex in sweep]
    share = np.mean([m["has_ties"] for m in metas])
    assert 0.12 <= share <= 0.40, share
    live = 0
    tied = 0
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        if not meta["has_ties"]:
            continue
        tied += 1
        objs = module.objects(ex["input"], meta["connectivity"], meta["bg_color"])
        ordered = module.sorted_objects(objs, meta["sort_key"])
        values = [module.key_value(obj, meta["sort_key"]) for obj in ordered]
        # swapping a tied pair must change the output, or the clause is dead
        for i in range(len(ordered) - 1):
            if values[i] != values[i + 1]:
                continue
            swapped = list(ordered)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            painted = module.layout(ex["input"], swapped, meta["bg_color"],
                                    meta["strip_axis"], meta["gap"],
                                    meta["strip_start_row"], meta["strip_start_col"])
            if painted != ex["output"]:
                live += 1
                break
    assert tied >= 20, tied
    # the rest are two same-colour bodies that happen to be congruent, which a
    # two-colour palette can always produce
    assert live / tied >= 0.90, (live, tied)


def test_reading_order_and_its_reverse_are_both_left_to_chance(sweep):
    """With two objects the sorted order is the reading order or exactly its
    reverse, so forbidding one hands the solver the other: the first rebuild
    rejected identity layouts and made the REVERSE reader right on 34% of
    examples (82% of the two-object scenes).  Neither is forced now, and three
    or more objects wherever the canvas allows is what keeps both rare."""
    metas = [ex["metadata"] for ex in sweep]
    forward = np.mean([m["order_equals_reading_order"] for m in metas])
    backward = np.mean([m["order_equals_reverse_reading_order"] for m in metas])
    assert 0.06 <= forward <= 0.30, forward
    assert 0.06 <= backward <= 0.30, backward
    assert abs(forward - backward) <= 0.10, (forward, backward)
    hits = Counter()
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        objs = module.objects(ex["input"], meta["connectivity"], meta["bg_color"])
        order = sorted(objs, key=lambda o: (o["box"][0], o["box"][1]))
        for name, candidate in (("forward", order), ("reverse", list(reversed(order)))):
            if not module.fits(candidate, meta["grid_h"], meta["grid_w"],
                               meta["strip_axis"], meta["gap"], meta["strip_start_row"],
                               meta["strip_start_col"]):
                continue
            painted = module.layout(ex["input"], candidate, meta["bg_color"],
                                    meta["strip_axis"], meta["gap"],
                                    meta["strip_start_row"], meta["strip_start_col"])
            hits[name] += int(painted == ex["output"])
    assert hits["forward"] / ORACLE_SEEDS <= 0.32, hits["forward"] / ORACLE_SEEDS
    assert hits["reverse"] / ORACLE_SEEDS <= 0.32, hits["reverse"] / ORACLE_SEEDS


def test_connectivity_is_load_bearing_and_unpredictable(sweep):
    metas = [ex["metadata"] for ex in sweep]
    share = np.mean([m["connectivity"] == 8 for m in metas])
    assert 0.42 <= share <= 0.58, share
    differs = np.mean([m["differs_under_other_connectivity"] for m in metas])
    assert differs >= 0.25, differs
    joined = [m for m in metas if m["corner_joined_figures"] > 0]
    plain = [m for m in metas if m["corner_joined_figures"] == 0]
    assert len(joined) / len(metas) >= 0.25
    for group in (joined, plain):
        rate = np.mean([m["connectivity"] == 8 for m in group])
        assert abs(rate - 0.5) <= 0.12, (len(group), rate)


def test_the_isolated_cell_clause_is_load_bearing(sweep):
    """A solver that laid out the texture dots too would be wrong."""
    textured = [ex for ex in sweep[:ORACLE_SEEDS] if ex["metadata"]["texture_colors"]]
    assert len(textured) >= 40, len(textured)
    for ex in textured:
        meta = ex["metadata"]
        assert meta["singleton_cells"] > 0
        rows = [list(row) for row in ex["input"]]
        singles = [{"cells": [(r, c)], "color": value, "box": (r, c, r, c)}
                   for r, row in enumerate(rows) for c, value in enumerate(row)
                   if value != meta["bg_color"] and all(
                       not (0 <= r + dr < meta["grid_h"] and 0 <= c + dc < meta["grid_w"]
                            and rows[r + dr][c + dc] == value)
                       for dr, dc in _ORTH + _DIAG)]
        objs = module.objects(ex["input"], meta["connectivity"], meta["bg_color"])
        ordered = module.sorted_objects(objs + singles, meta["sort_key"])
        if not module.fits(ordered, meta["grid_h"], meta["grid_w"], meta["strip_axis"],
                           meta["gap"], meta["strip_start_row"], meta["strip_start_col"]):
            continue
        painted = module.layout(ex["input"], ordered, meta["bg_color"],
                                meta["strip_axis"], meta["gap"],
                                meta["strip_start_row"], meta["strip_start_col"])
        assert painted != ex["output"]


def test_background_reading_is_load_bearing(sweep):
    """Filling the rest with colour 0 is only right when 0 IS the background."""
    hits = 0
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        objs = module.sorted_objects(
            module.objects(ex["input"], meta["connectivity"], meta["bg_color"]),
            meta["sort_key"])
        painted = module.layout(ex["input"], objs, 0, meta["strip_axis"], meta["gap"],
                                meta["strip_start_row"], meta["strip_start_col"])
        hits += int(painted == ex["output"])
    assert hits / ORACLE_SEEDS <= 0.20, hits / ORACLE_SEEDS


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
        assert heights[side] + widths[side] >= 0.010 * 2 * len(metas), side
    squares = np.mean([m["grid_h"] == m["grid_w"] for m in metas])
    assert squares <= 0.09, squares
    areas = np.array([m["grid_h"] * m["grid_w"] for m in metas])
    assert np.mean(areas >= 700) >= 0.02
    assert np.mean(areas < 150) >= 0.10
    corr = abs(float(np.corrcoef([m["grid_h"] for m in metas],
                                 [m["grid_w"] for m in metas])[0, 1]))
    assert corr <= 0.12, corr


def test_object_and_color_coverage(sweep):
    metas = [ex["metadata"] for ex in sweep]
    counts = Counter(m["n_objects"] for m in metas)
    assert min(counts) == 2 and max(counts) >= 5
    assert counts[2] / len(metas) <= 0.55
    assert sum(v for k, v in counts.items() if k >= 4) / len(metas) >= 0.20
    colors = Counter(m["n_colors"] for m in metas)
    assert min(colors) <= 3 and max(colors) >= 8
    assert sum(v for k, v in colors.items() if k >= 5) / len(metas) >= 0.20
    # Objects are single-coloured and the strip carries at most six of them
    # (n distinct extents on BOTH axes cost n(n+1)/2 cells each way, which is
    # what keeps the sort key unrecoverable), so the palette ceiling is one
    # ground colour, one per object and at most three texture colours.  Eight to
    # ten therefore needs four or more objects AND a texture.
    rich = sum(v for k, v in colors.items() if k >= 8) / len(metas)
    assert rich >= 0.04, rich
    big = [m for m in metas if m["grid_h"] * m["grid_w"] >= 500]
    assert len(big) >= 60
    assert np.mean([m["n_colors"] >= 8 for m in big]) >= 0.15
    assert all(m["n_colors"] <= 1 + m["n_objects"] + 3 for m in metas)
    axes = Counter(m["strip_axis"] for m in metas)
    assert min(axes.values()) >= 0.40 * len(metas)
    gaps = Counter(m["gap"] for m in metas)
    assert set(gaps) == {1, 2, 3}
    starts = Counter((m["strip_start_row"], m["strip_start_col"]) for m in metas)
    assert len(starts) >= 5


def test_objects_scale_with_the_canvas(sweep):
    bands = {"small": [], "large": []}
    for ex in sweep:
        meta = ex["metadata"]
        area = meta["grid_h"] * meta["grid_w"]
        bands["small" if area < 200 else "large"].append(meta["occupancy"])
    for name, values in bands.items():
        assert len(values) >= 40, name
        assert np.mean(values) >= 0.08, (name, np.mean(values))
    # Both strips have to fit, so the objects' heights AND widths each sum
    # inside the canvas: a roomy canvas carries more objects rather than
    # proportionally bigger ones, and occupancy falls off a little with the
    # object count.  It does not collapse.
    assert np.mean(bands["large"]) >= 0.55 * np.mean(bands["small"])
    sizes = [size for ex in sweep for size in ex["metadata"]["object_sizes"]]
    assert np.median(sizes) >= 6
    assert max(sizes) >= 40


def test_every_object_fits_the_strip(sweep):
    for ex in sweep:
        meta = ex["metadata"]
        assert meta["strip_length"] + (meta["strip_start_col"]
                                       if meta["strip_axis"] == "horizontal"
                                       else meta["strip_start_row"]) <= meta["strip_capacity"] \
            + (meta["strip_start_col"] if meta["strip_axis"] == "horizontal"
               else meta["strip_start_row"])
        assert module.fits(
            module.sorted_objects(
                module.objects(ex["input"], meta["connectivity"], meta["bg_color"]),
                meta["sort_key"]),
            meta["grid_h"], meta["grid_w"], meta["strip_axis"], meta["gap"],
            meta["strip_start_row"], meta["strip_start_col"])


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
            assert not (set(meta["texture_colors"]) & set(meta["object_colors"]))
    assert 0.12 <= textured / ORACLE_SEEDS <= 0.50, textured / ORACLE_SEEDS


def test_shapes_look_drawn(sweep):
    hairs = 0
    fills = []
    pieces = 0
    for ex in sweep[:200]:
        meta = ex["metadata"]
        for obj in module.objects(ex["input"], 4, meta["bg_color"]):
            pieces += 1
            cells = set(obj["cells"])
            assert len(cells) >= 2
            for r, c in cells:
                if sum(1 for dr, dc in _ORTH if (r + dr, c + dc) in cells) == 0:
                    hairs += 1
            r0, c0, r1, c1 = obj["box"]
            fills.append(len(cells) / ((r1 - r0 + 1) * (c1 - c0 + 1)))
    assert hairs == 0, hairs
    assert pieces >= 300
    assert np.mean(fills) >= 0.55, np.mean(fills)


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    seen = set()
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        prompt = get_prompt("98", meta)
        seen.add((meta["sort_key"], meta["strip_axis"]))
        assert "Ignore the background, whether it is a solid color or a pattern." in prompt
        assert "isolated single cells, whatever their color, are background texture" in prompt
        assert "cells of different colors are always separate" in prompt
        assert "at least two cells" in prompt
        assert "in ascending order of" in prompt
        assert "when two objects have the same value" in prompt
        assert "The output has the same dimensions as the input." in prompt
        assert "is plain background" in prompt
        assert "without any of the input's background texture" in prompt
        assert "Rows and columns are 0-indexed from the top-left corner." in prompt
        assert "each object keeps its own shape and color" in prompt
        # the gap is stated as the number of empty columns (rows) BETWEEN the
        # two bounding boxes; "K cells to the right of" also read as "shift the
        # left edge by K", which is not the implemented rule
        unit = "column" if meta["strip_axis"] == "horizontal" else "row"
        plural = unit if meta["gap"] == 1 else unit + "s"
        assert f"leaving exactly {meta['gap']} empty {plural} between the two " \
               "bounding boxes" in prompt
        assert f"{meta['gap']} cells to the right" not in prompt
        assert f"{meta['gap']} cells below" not in prompt
        if meta["strip_axis"] == "horizontal":
            assert "to the right of the previous object's bounding box" in prompt
            assert "with its top edge on the same row" in prompt
        else:
            assert "below the previous object's bounding box" in prompt
            assert "with its left edge in the same column" in prompt
        assert not re.search(r"\b(background|texture) (is|are|color)\b", prompt)
        assert "most frequent" not in prompt
        assert "Leave all other cells empty" not in prompt
        for leak in ("tie_rule", "has_ties", "corner-joined", "texture color"):
            assert leak not in prompt.lower()
    assert len(seen) == 6


def test_prompt_is_byte_identical_per_bucket(sweep):
    buckets = {}
    for ex in sweep[:300]:
        meta = ex["metadata"]
        key = (meta["sort_key"], meta["strip_axis"], meta["connectivity"], meta["gap"],
               meta["strip_start_row"], meta["strip_start_col"])
        buckets.setdefault(key, set()).add(get_prompt("98", meta))
    for key, texts in buckets.items():
        assert len(texts) == 1, key


# ---------------------------------------------------------------------------
# the validator is real
# ---------------------------------------------------------------------------

def test_validator_raises_when_the_output_is_tampered():
    ex = _ex(4)
    broken = {"input": ex["input"],
              "output": [list(row) for row in ex["output"]],
              "metadata": dict(ex["metadata"])}
    broken["output"][0][0] = (broken["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_validator_raises_when_the_gap_is_wrong():
    ex = _ex(6)
    broken = {"input": ex["input"], "output": ex["output"],
              "metadata": dict(ex["metadata"])}
    broken["metadata"]["gap"] = broken["metadata"]["gap"] % 3 + 1
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_the_second_oracle_rebuilds_every_output(sweep):
    """The validator's independence rests on two implementations of the whole
    rule, so they are checked against each other on every grid, connectivity,
    key and axis."""
    for ex in sweep[:ORACLE_SEEDS]:
        meta = ex["metadata"]
        twin = module.oracle_output(ex["input"], meta["connectivity"], meta["sort_key"],
                                    meta["strip_axis"], meta["gap"],
                                    meta["strip_start_row"], meta["strip_start_col"])
        assert twin == ex["output"], meta["attempts"]
        assert module.oracle_ground(ex["input"]) == meta["bg_color"]
    for ex in sweep[:120]:
        meta = ex["metadata"]
        for conn in (4, 8):
            for key in ("cell_count", "bbox_height", "bbox_width"):
                for axis in ("horizontal", "vertical"):
                    assert module.oracle_output(
                        ex["input"], conn, key, axis, meta["gap"],
                        meta["strip_start_row"], meta["strip_start_col"]) == module.solve(
                        ex["input"], conn, key, axis, meta["gap"],
                        meta["strip_start_row"], meta["strip_start_col"])


def test_validator_raises_when_the_axis_is_wrong():
    ex = _ex(8)
    broken = {"input": ex["input"], "output": ex["output"],
              "metadata": dict(ex["metadata"])}
    broken["metadata"]["strip_axis"] = (
        "vertical" if broken["metadata"]["strip_axis"] == "horizontal" else "horizontal")
    with pytest.raises(ValueError):
        module.validate_example(broken)


def test_validator_raises_when_the_plan_disagrees():
    """The plan is a third witness: boxes or cell counts that do not match the
    picture are rejected even though the output is the right one."""
    ex = _ex(10)
    meta = ex["metadata"]
    objs = module.objects(ex["input"], 8, meta["bg_color"])
    boxes = [(obj["box"][2] - obj["box"][0] + 1, obj["box"][3] - obj["box"][1] + 1)
             for obj in objs]
    cells = [len(obj["cells"]) for obj in objs]
    module.validate_example(ex, plan={"boxes": boxes, "cells": cells})
    with pytest.raises(ValueError):
        module.validate_example(ex, plan={"boxes": [(b[0] + 1, b[1]) for b in boxes],
                                          "cells": cells})
    with pytest.raises(ValueError):
        module.validate_example(ex, plan={"boxes": boxes,
                                          "cells": [c + 1 for c in cells]})


def test_validator_raises_when_an_object_would_not_fit():
    """A strip that runs off the canvas is rejected, not clipped."""
    for seed in range(60):
        ex = _ex(seed)
        meta = dict(ex["metadata"])
        capacity = (meta["grid_w"] if meta["strip_axis"] == "horizontal"
                    else meta["grid_h"])
        meta["strip_start_row"] = meta["grid_h"] - 1
        meta["strip_start_col"] = meta["grid_w"] - 1
        if meta["strip_length"] + 1 <= capacity:
            continue
        with pytest.raises(ValueError):
            module.validate_example({"input": ex["input"], "output": ex["output"],
                                     "metadata": meta})
        return
    raise AssertionError("no example to test the fit guard with")


def test_generation_is_cheap():
    times = []
    for seed in range(150):
        start = time.perf_counter()
        _ex(seed)
        times.append((time.perf_counter() - start) * 1000)
    assert np.mean(times) <= 25.0, np.mean(times)
    assert np.percentile(times, 95) <= 120.0, np.percentile(times, 95)
