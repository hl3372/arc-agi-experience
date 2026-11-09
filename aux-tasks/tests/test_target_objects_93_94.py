"""Tasks 93 (tight-crop the unique color X object) and 94 (list every color X box).

The oracles below are written from the prompt text alone -- the connectivity is
parsed out of the sentence, the objects are union-find components of the target
colour, and nothing here imports the generator's reader -- so a bug shared with
the module cannot hide itself.
"""
from collections import Counter
from copy import deepcopy
import ast
import json
import re
import time

import numpy as np
import pytest

from generators import target_objects_93_94 as module
from prompts import get_prompt
from registry import generate
from symbol_remap import palette_for_example

ORACLE_SEEDS = 400
SWEEP_SEEDS = 700


def _ex(task, seed):
    return generate(task, seed)


@pytest.fixture(scope="module")
def sweep93():
    return [_ex("93", seed) for seed in range(SWEEP_SEEDS)]


@pytest.fixture(scope="module")
def sweep94():
    return [_ex("94", seed) for seed in range(SWEEP_SEEDS)]


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------

def _connectivity_from_prompt(prompt):
    match = re.search(r"using (\d)-connectivity", prompt)
    assert match, prompt
    conn = int(match.group(1))
    if conn == 4:
        assert "connected horizontally or vertically" in prompt
        assert "cells that touch only at a corner are not connected" in prompt
    else:
        assert "connected horizontally, vertically, or diagonally" in prompt
    return conn


def _target_from_prompt(prompt):
    match = re.search(r"color (\d)", prompt)
    assert match, prompt
    return int(match.group(1))


def _label(mask, connectivity):
    """Row-major union-find labelling (a second implementation)."""
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


def _oracle_boxes(grid, prompt):
    conn = _connectivity_from_prompt(prompt)
    color = _target_from_prompt(prompt)
    arr = np.asarray(grid)
    boxes = []
    for cells in _label(arr == color, conn):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        boxes.append((min(rs), min(cs), max(rs), max(cs)))
    return sorted(boxes, key=lambda b: (b[0], b[1]))


def test_93_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex("93", seed)
        prompt = get_prompt("93", ex["metadata"])
        boxes = _oracle_boxes(ex["input"], prompt)
        assert len(boxes) == 1, (seed, boxes)          # the prompt's guarantee holds
        r0, c0, r1, c1 = boxes[0]
        crop = [list(row[c0:c1 + 1]) for row in ex["input"][r0:r1 + 1]]
        assert crop == ex["output"], seed
        assert [r0, c0, r1, c1] == list(ex["metadata"]["target_box"]), seed


def test_94_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex("94", seed)
        prompt = get_prompt("94", ex["metadata"])
        boxes = _oracle_boxes(ex["input"], prompt)
        assert boxes, seed
        assert [tuple(b) for b in ast.literal_eval(ex["output"])] == boxes, seed
        assert len(boxes) == ex["metadata"]["n_targets"], seed
        # the stated order is a total order on every example
        assert len({(b[0], b[1]) for b in boxes}) == len(boxes), seed


def test_determinism_and_cell_range():
    for task in ("93", "94"):
        for seed in range(50):
            a, b = _ex(task, seed), _ex(task, seed)
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (task, seed)
            grid = np.asarray(a["input"])
            assert grid.min() >= 0 and grid.max() <= 9
    out93 = _ex("93", 3)["output"]
    assert isinstance(out93, list) and isinstance(out93[0], list)
    assert isinstance(_ex("94", 3)["output"], str)


# ---------------------------------------------------------------------------
# helpers for the coverage / leak bounds
# ---------------------------------------------------------------------------

def _meta(sweep):
    return [ex["metadata"] for ex in sweep]


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
# sizes, colours, textures, objects (both tasks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["93", "94"])
def test_sides_are_flat_and_independent(task, sweep93, sweep94):
    meta = _meta(sweep93 if task == "93" else sweep94)
    hs = [m["grid_h"] for m in meta]
    ws = [m["grid_w"] for m in meta]
    assert min(hs) <= 6 and max(hs) >= 29, (min(hs), max(hs))
    assert min(ws) <= 6 and max(ws) >= 29, (min(ws), max(ws))
    assert abs(float(np.corrcoef(hs, ws)[0, 1])) <= 0.15
    squares = np.mean([h == w for h, w in zip(hs, ws)])
    assert squares <= 0.10, squares              # squares by chance only
    big = np.mean([h * w >= 700 for h, w in zip(hs, ws)])
    assert 0.01 <= big <= 0.09, big              # the flat-sides share
    # near-flat per value: no side value owns more than twice its fair share
    counts = Counter(hs + ws)
    fair = len(hs + ws) / 26
    assert max(counts.values()) <= 2.2 * fair, counts.most_common(3)
    assert min(counts.get(v, 0) for v in range(5, 31)) >= 0.35 * fair, counts


@pytest.mark.parametrize("task", ["93", "94"])
def test_colours(task, sweep93, sweep94):
    meta = _meta(sweep93 if task == "93" else sweep94)
    counts = [m["n_colors"] for m in meta]
    assert min(counts) == 2 and max(counts) == 10, (min(counts), max(counts))
    rich = np.mean([c >= 8 for c in counts])
    assert rich >= 0.15, rich                     # a real 8-10 tail
    assert np.mean([c <= 3 for c in counts]) >= 0.08
    for m in meta:
        assert m["target_color"] != m["background_color"]
        assert m["target_color"] not in (m["texture_colors"] or [])


@pytest.mark.parametrize("task", ["93", "94"])
def test_textures_come_from_the_library(task, sweep93, sweep94):
    sweep = sweep93 if task == "93" else sweep94
    groups = Counter(ex["metadata"]["texture_group"] for ex in sweep)
    assert groups["sparse"] >= 0.05 * len(sweep), groups
    assert groups["solid"] >= 0.03 * len(sweep), groups
    assert groups[None] >= 0.4 * len(sweep), groups
    for ex in sweep:
        meta = ex["metadata"]
        colors = meta["texture_colors"]
        if not colors:
            continue
        assert meta["texture_named"] is False
        arr = np.asarray(ex["input"])
        assert meta["target_color"] not in colors
        if meta["texture_group"] == "sparse":
            from common.backgrounds import isolated_cell_mask
            assert bool(isolated_cell_mask(arr)[np.isin(arr, colors)].all())
        # a texture never creates a target-colour cell
        assert int((arr == meta["target_color"]).sum()) == (
            meta["target_cells"] if task == "94" else meta["target_cells"])


@pytest.mark.parametrize("task", ["93", "94"])
def test_objects_scale_with_the_canvas(task, sweep93, sweep94):
    meta = _meta(sweep93 if task == "93" else sweep94)
    bands = {}
    for m in meta:
        area = m["grid_h"] * m["grid_w"]
        key = "<150" if area < 150 else ("150-399" if area < 400 else ("400-699" if area < 700 else "700+"))
        bands.setdefault(key, []).append(m["occupancy"])
    for key in ("<150", "150-399", "400-699"):
        assert np.mean(bands[key]) >= 0.12, (key, np.mean(bands[key]))
    if bands.get("700+"):
        # a big canvas is not two small objects in a sea of background
        assert np.mean(bands["700+"]) >= 0.6 * np.mean(bands["<150"]), (
            np.mean(bands["700+"]), np.mean(bands["<150"]))
    kinds = Counter(k for m in meta for k in m["object_kinds"] + m["distractor_kinds"])
    assert len(kinds) >= 8, kinds                 # the shared vocabulary, not one family
    assert max(kinds.values()) <= 0.5 * sum(kinds.values()), kinds.most_common(3)


# ---------------------------------------------------------------------------
# task 93: the crop, its contested box, the connectivity
# ---------------------------------------------------------------------------

def test_93_target_is_unique_and_sized(sweep93):
    meta = _meta(sweep93)
    for ex, m in zip(sweep93, meta):
        assert m["n_targets"] == 1
        assert m["target_cells"] >= 3
        arr = np.asarray(ex["input"])
        assert int((arr == m["target_color"]).sum()) == m["target_cells"]
    sizes = [m["target_cells"] for m in meta]
    assert min(sizes) <= 5 and max(sizes) >= 60, (min(sizes), max(sizes))
    areas = [m["output_h"] * m["output_w"] for m in meta]
    assert min(areas) <= 6 and max(areas) >= 150, (min(areas), max(areas))


def test_93_the_box_is_contested(sweep93):
    """"copying every input cell in that rectangle" has to matter."""
    meta = _meta(sweep93)
    with_object = np.mean([m["distractors_in_box"] > 0 for m in meta])
    assert 0.20 <= with_object <= 0.60, with_object
    with_any = np.mean([m["foreign_cells_in_box"] > 0 for m in meta])
    assert with_any >= with_object
    # and on those grids the crop really carries the other colour
    for ex, m in zip(sweep93, meta):
        if m["distractors_in_box"] == 0:
            continue
        flat = [v for row in ex["output"] for v in row]
        assert len(set(flat)) >= 2, m


def test_93_connectivity_is_stated_and_load_bearing(sweep93):
    meta = _meta(sweep93)
    share4 = np.mean([m["connectivity"] == 4 for m in meta])
    assert 0.35 <= share4 <= 0.65, share4
    differs = np.mean([m["differs_under_other_connectivity"] for m in meta])
    assert differs >= 0.30, differs
    for ex, m in zip(sweep93[:150], meta[:150]):
        assert _connectivity_from_prompt(get_prompt("93", m)) == m["connectivity"]
        # under the other reading the "unique object" premise fails exactly when
        # the flag says so
        other = 4 if m["connectivity"] == 8 else 8
        boxes = _oracle_boxes(ex["input"], get_prompt("93", dict(m, connectivity=other)))
        assert (len(boxes) != 1) == m["differs_under_other_connectivity"], m


def test_93_largest_object_shortcut_is_bounded(sweep93):
    """Cropping the biggest body instead of the named colour must not solve it."""
    hits = 0
    for ex in sweep93:
        arr = np.asarray(ex["input"])
        modal = int(np.bincount(arr.ravel(), minlength=10).argmax())
        best = None
        for color in np.unique(arr):
            if color == modal:
                continue
            for cells in _label(arr == color, 8):
                if best is None or len(cells) > len(best):
                    best = cells
        rs = [r for r, _ in best]
        cs = [c for _, c in best]
        hits += int([min(rs), min(cs), max(rs), max(cs)] == list(ex["metadata"]["target_box"]))
    assert hits / len(sweep93) <= 0.68, hits / len(sweep93)


# ---------------------------------------------------------------------------
# task 94: answers, the connectivity, the colour clause
# ---------------------------------------------------------------------------

def test_94_answers_are_spread(sweep94):
    meta = _meta(sweep94)
    counts = [m["n_targets"] for m in meta]
    hist = Counter(counts)
    assert min(counts) == 1, min(counts)
    assert max(counts) >= 9, max(counts)
    assert hist[1] / len(counts) <= 0.25, hist[1] / len(counts)
    assert max(hist.values()) / len(counts) <= 0.30, hist.most_common(3)
    assert len([v for v in hist if hist[v] >= 0.01 * len(counts)]) >= 8, hist
    for m in meta:
        assert m["n_targets"] <= m["answer_cap"] or m["n_targets"] <= module.MAX_BOXES
    singles = np.mean([m["single_cell_targets"] > 0 for m in meta])
    assert 0.20 <= singles <= 0.70, singles       # a lone cell is an object here
    sizes = [s for m in meta for s in m["target_object_sizes"]]
    assert min(sizes) == 1 and max(sizes) >= 40, (min(sizes), max(sizes))


def test_94_connectivity_is_load_bearing_and_unpredictable(sweep94):
    meta = _meta(sweep94)
    share4 = np.mean([m["connectivity"] == 4 for m in meta])
    assert 0.42 <= share4 <= 0.58, share4
    differs = np.mean([m["differs_under_other_connectivity"] for m in meta])
    assert differs >= 0.60, differs
    rule = [m["connectivity"] for m in meta]
    base = _baseline(rule)
    features = [
        [f"{len(module.solve_boxes(ex['input'], ex['metadata']['target_color'], 4))}_"
         f"{len(module.solve_boxes(ex['input'], ex['metadata']['target_color'], 8))}" for ex in sweep94],
        [f"{m['grid_h']}x{m['grid_w']}_{m['n_colors']}" for m in meta],
        [f"{m['target_cells']}_{m['n_distractors']}" for m in meta],
    ]
    for feature in features:
        assert _heldout(feature, rule) <= base + 0.08, _heldout(feature, rule)
    # the answer itself is not readable off the grid size
    answers = [m["n_targets"] for m in meta]
    size = [f"{m['grid_h'] // 6}_{m['grid_w'] // 6}" for m in meta]
    assert _heldout(size, answers) <= _baseline(answers) + 0.14, _heldout(size, answers)


def test_94_color_clause_is_load_bearing(sweep94):
    """Objects of other colours touch the targets: merging them changes the answer."""
    meta = _meta(sweep94)
    assert np.mean([m["distractors_touching_target"] > 0 for m in meta]) >= 0.25
    changed = 0
    for ex in sweep94[:250]:
        m = ex["metadata"]
        arr = np.asarray(ex["input"])
        mask = (arr != m["background_color"])
        blind = []
        for cells in _label(mask, m["connectivity"]):
            if not any(arr[r, c] == m["target_color"] for r, c in cells):
                continue
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            blind.append((min(rs), min(cs), max(rs), max(cs)))
        blind = sorted(blind, key=lambda b: (b[0], b[1]))
        changed += int(module.format_boxes(blind) != ex["output"])
    assert changed / 250 >= 0.30, changed / 250


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep93, sweep94):
    wordings = {"93": set(), "94": set()}
    for task, sweep in (("93", sweep93), ("94", sweep94)):
        for ex in sweep[:250]:
            meta = ex["metadata"]
            prompt = get_prompt(task, meta)
            wordings[task].add(re.sub(r"color \d", "color X", prompt))
            # the background is never named, defined or described
            assert "most frequent" not in prompt
            assert not re.search(r"background (is|color|colour)\b", prompt)
            assert not re.search(r"\btexture\b|\bpatterned background\b|\bdetail\b", prompt)
            assert "isolated single cells" not in prompt
            assert _connectivity_from_prompt(prompt) == meta["connectivity"]
            if task == "93":
                assert "background" not in prompt      # nothing to ignore: colour rule
                assert "copying every input cell in that rectangle exactly as it appears" in prompt
                assert "including any non-object cells inside the box" in prompt
                assert "exactly one such object is present" in prompt
            else:
                assert "Ignore the background, whether it is a solid color or a pattern." in prompt
                assert "on its own is an object" in prompt
                assert "cells of any other color are not objects" in prompt
                assert "inclusive 0-indexed coordinates" in prompt
                assert "[(top_row,left_column,bottom_row,right_column),...]" in prompt
    # exactly one wording per connectivity bucket on each task
    assert len(wordings["93"]) == 2, wordings["93"]
    assert len(wordings["94"]) == 2, wordings["94"]


def test_prompts_survive_the_symbol_palette():
    letters = list("ABCDEFGHIJ")
    for task in ("93", "94"):
        for seed in range(40):
            meta = generate(task, seed)["metadata"]
            prompt = get_prompt(task, meta, color_palette=letters)
            assert not re.search(r"\bcolor \d\b", prompt), (task, seed, prompt)
            assert re.search(r"\bcolor [A-J]\b", prompt), (task, seed, prompt)
            # the audit's own palette must not leave a raw digit either
            remapped = get_prompt(task, meta, color_palette=palette_for_example(task, seed))
            assert not re.search(r"\bcolor \d\b", remapped), (task, seed, remapped)


# ---------------------------------------------------------------------------
# metadata contract and the validators
# ---------------------------------------------------------------------------

def test_metadata_contract(sweep93, sweep94):
    shared = {"grid_h", "grid_w", "n_colors", "connectivity", "target_color", "n_targets",
              "target_cells", "n_distractors", "distractors_in_box", "texture_group",
              "texture_kind", "texture_colors", "texture_named", "texture_coverage",
              "attempts", "occupancy", "object_kinds", "distractor_kinds", "background_color"}
    for ex in sweep93[:200]:
        meta = ex["metadata"]
        assert shared <= set(meta), shared - set(meta)
        assert {"target_box", "output_h", "output_w", "foreign_cells_in_box",
                "n_target_components_other", "differs_under_other_connectivity",
                "target_pieces"} <= set(meta)
        assert meta["connectivity"] in (4, 8)
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        assert meta["output_h"] == len(ex["output"]) and meta["output_w"] == len(ex["output"][0])
    for ex in sweep94[:200]:
        meta = ex["metadata"]
        assert shared <= set(meta), shared - set(meta)
        assert {"answer_cap", "answer_anchor", "answer_other_connectivity",
                "differs_under_other_connectivity", "single_cell_targets",
                "target_object_sizes", "distractors_touching_target"} <= set(meta)
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        assert meta["n_targets"] == len(ast.literal_eval(ex["output"]))
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep93]) <= 2.0
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep94]) <= 2.0


def test_93_validator_raises_on_a_tampered_example():
    ex = _ex("93", 5)
    module.validate_93(deepcopy(ex))                      # the honest example passes

    wrong = deepcopy(ex)
    wrong["output"][0][0] = (wrong["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_93(wrong)

    # flipping the connectivity on a grid whose readings disagree breaks the
    # "unique object" premise the validator recomputes
    flip_seed = next(s for s in range(60)
                     if _ex("93", s)["metadata"]["differs_under_other_connectivity"])
    flipped = _ex("93", flip_seed)
    flipped["metadata"]["connectivity"] = 4 if flipped["metadata"]["connectivity"] == 8 else 8
    with pytest.raises(ValueError):
        module.validate_93(flipped)

    # a second object of the target colour destroys the "unique object" premise
    for seed in range(40):
        ex = _ex("93", seed)
        painted = deepcopy(ex)
        grid = [list(row) for row in painted["input"]]
        h, w = len(grid), len(grid[0])
        r0, c0, r1, c1 = ex["metadata"]["target_box"]
        spot = next(((r, c) for r in range(h) for c in range(w)
                     if not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)), None)
        if spot is None:
            continue
        grid[spot[0]][spot[1]] = ex["metadata"]["target_color"]
        painted["input"] = grid
        with pytest.raises(ValueError):
            module.validate_93(painted)
        break
    else:
        pytest.fail("no grid with room for a second target object")


def test_94_validator_raises_on_a_tampered_example():
    ex = _ex("94", 5)
    module.validate_94(deepcopy(ex))

    wrong = deepcopy(ex)
    boxes = [list(b) for b in ast.literal_eval(wrong["output"])]
    boxes[0][3] += 1
    wrong["output"] = module.format_boxes([tuple(b) for b in boxes])
    with pytest.raises(ValueError):
        module.validate_94(wrong)

    misordered = deepcopy(ex)
    boxes = [tuple(b) for b in ast.literal_eval(misordered["output"])]
    if len(boxes) >= 2:
        misordered["output"] = module.format_boxes(list(reversed(boxes)))
        with pytest.raises(ValueError):
            module.validate_94(misordered)

    flipped = deepcopy(ex)
    flipped["metadata"]["connectivity"] = 8 if flipped["metadata"]["connectivity"] == 4 else 4
    if ex["metadata"]["differs_under_other_connectivity"]:
        with pytest.raises(ValueError):
            module.validate_94(flipped)

    tampered = deepcopy(ex)
    tampered["metadata"]["n_targets"] += 1
    with pytest.raises(ValueError):
        module.validate_94(tampered)


def test_generation_speed():
    for task in ("93", "94"):
        times = []
        for seed in range(300):
            t0 = time.perf_counter()
            generate(task, seed)
            times.append((time.perf_counter() - t0) * 1000.0)
        assert np.mean(times) <= 8.0, (task, np.mean(times))
        assert np.percentile(times, 99) <= 60.0, (task, np.percentile(times, 99))
