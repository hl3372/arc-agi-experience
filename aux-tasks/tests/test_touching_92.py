"""Task 92: how many objects touch another object (4/8-connectivity per example).

Task 92b was merged into task 92 on 2026-09-15.  The oracle below is written
from the prompt text alone (infer the ground, group same-colour components of
at least two cells under the stated connectivity, count the ones with a
neighbour in another component) with a union-find labelling that shares no code
with the generator's reader.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
import re
import time

import numpy as np
import pytest

from generators import touching_92 as module
from prompts import PROMPT_TEMPLATES, get_prompt
from registry import RETIRED, generate, list_tasks

ORACLE_SEEDS = 400
SWEEP_SEEDS = 1200

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _ex(seed):
    return generate("92", seed)


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------

def _connectivity_from_prompt(prompt):
    assert "Ignore the background, whether it is a solid color or a pattern." in prompt
    assert "at least two cells" in prompt
    assert "isolated single cells, whatever their color, are background texture" in prompt
    assert "always separate, even where they touch" in prompt
    assert "Count objects, not touching pairs." in prompt
    assert prompt.endswith("Answer with the number.")
    match = re.search(r"[Tt]reat each (\d)-connected same-color group", prompt)
    assert match, prompt
    conn = int(match.group(1))
    if conn == 4:
        assert "side-adjacent to a cell of a different object" in prompt
        assert "side-or-corner" not in prompt
    else:
        assert "side-or-corner-adjacent to a cell of a different object" in prompt
    return conn


def _ground(arr):
    """The most frequent colour once every isolated single cell is set aside."""
    h, w = arr.shape
    lonely = np.ones((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.full((h, w), -1)
            src = arr[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
            shifted[max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)] = src
            lonely &= shifted != arr
    return Counter(arr[~lonely].ravel().tolist()).most_common(1)[0][0]


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


def _components(grid, connectivity):
    arr = np.asarray(grid)
    ground = _ground(arr)
    comps = []
    for color in np.unique(arr):
        if color == ground:
            continue
        for cells in _label(arr == color, connectivity):
            if len(cells) >= 2:
                comps.append(cells)
    return comps, int(ground)


def oracle(grid, prompt):
    connectivity = _connectivity_from_prompt(prompt)
    comps, ground = _components(grid, connectivity)
    owner = {}
    for i, comp in enumerate(comps):
        for cell in comp:
            owner[cell] = i
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    touching = set()
    for (r, c), i in owner.items():
        for dr, dc in steps:
            j = owner.get((r + dr, c + dc))
            if j is not None and j != i:
                touching.add(i)
                touching.add(j)
    return len(touching), ground, connectivity


def test_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex(seed)
        answer, ground, conn = oracle(ex["input"], get_prompt("92", ex["metadata"]))
        assert str(answer) == ex["output"], (seed, answer, ex["output"])
        assert ground == ex["metadata"]["bg_color"], seed
        assert conn == ex["metadata"]["connectivity"], seed


def test_determinism_and_cell_range():
    for seed in range(60):
        a, b = _ex(seed), _ex(seed)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), seed
        grid = np.asarray(a["input"])
        assert grid.min() >= 0 and grid.max() <= 9
        assert isinstance(a["output"], str) and a["output"].isdigit()


def test_shared_body_floor_matches_task_90():
    from generators import border_objects_90
    assert module.MIN_BODY == border_objects_90.MIN_BODY


# ---------------------------------------------------------------------------
# sweep helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return [_ex(seed) for seed in range(SWEEP_SEEDS)]


def _meta(sweep):
    return [ex["metadata"] for ex in sweep]


def _heldout(feature, target):
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
# 92b is gone
# ---------------------------------------------------------------------------

def test_92b_is_retired_into_92():
    assert RETIRED["92b"] == "92"
    assert "92b" not in list_tasks() and "92b" not in PROMPT_TEMPLATES
    assert "92" in list_tasks() and "92" in PROMPT_TEMPLATES
    with pytest.raises(KeyError, match="merged into task 92"):
        generate("92b", 0)


# ---------------------------------------------------------------------------
# connectivity: balanced, per example, and invisible in the picture
# ---------------------------------------------------------------------------

def test_connectivity_is_balanced_and_stated(sweep):
    meta = _meta(sweep)
    share = np.mean([m["connectivity"] == 4 for m in meta])
    assert 0.44 <= share <= 0.56, share
    for ex in sweep[:120]:
        prompt = get_prompt("92", ex["metadata"])
        assert _connectivity_from_prompt(prompt) == ex["metadata"]["connectivity"]


def test_connectivity_is_not_predictable_from_the_input(sweep):
    """The old 8-branch demanded a diagonal-only touch; an input-only reader
    named the variant on 76% of examples.  The scene is now drawn first."""
    meta = _meta(sweep)
    rule = [m["connectivity"] for m in meta]
    base = _baseline(rule)
    diagonal = [m["has_diagonal_only_touch"] for m in meta]
    # the share of grids with a corner-only touch is the same under both readings
    four = np.mean([d for d, m in zip(diagonal, meta) if m["connectivity"] == 4])
    eight = np.mean([d for d, m in zip(diagonal, meta) if m["connectivity"] == 8])
    assert abs(four - eight) <= 0.08, (four, eight)
    features = [
        diagonal,
        [f"{module.count_objects(ex['input'], 4)}_{module.count_objects(ex['input'], 8)}"
         for ex in sweep],
        [f"{m['grid_h']}x{m['grid_w']}_{m['n_colors']}" for m in meta],
        [f"{m['n_pieces']}_{m['n_decoys']}" for m in meta],
    ]
    for feature in features:
        assert _heldout(feature, rule) <= base + 0.08, _heldout(feature, rule)


def test_connectivity_is_load_bearing(sweep):
    meta = _meta(sweep)
    differs = np.mean([m["differs_under_other_connectivity"] for m in meta])
    assert differs >= 0.40, differs
    # the other reading is genuinely recomputed
    for ex in sweep[:80]:
        other = 8 if ex["metadata"]["connectivity"] == 4 else 4
        assert module.solve(ex["input"], other) == ex["metadata"]["answer_other_connectivity"]
    # both directions of the difference occur
    ups = [m for m in meta if m["answer_other_connectivity"] > m["n_touching"]]
    downs = [m for m in meta if m["answer_other_connectivity"] < m["n_touching"]]
    assert len(ups) / len(meta) >= 0.10 and len(downs) / len(meta) >= 0.10
    # the scene kinds that carry it are really there
    assert np.mean(["corner" in m["cluster_kinds"] for m in meta]) >= 0.20
    assert np.mean(["merge" in m["cluster_kinds"] for m in meta]) >= 0.12


def test_colour_separation_is_load_bearing(sweep):
    """A reader that merged touching colours into one object gets it wrong."""
    meta = _meta(sweep)
    assert np.mean([m["differs_under_color_blind"] for m in meta]) >= 0.50
    for ex in sweep[:80]:
        merged = module.solve(ex["input"], ex["metadata"]["connectivity"], color_blind=True)
        assert merged == ex["metadata"]["answer_color_blind"]


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------

def test_answers(sweep):
    meta = _meta(sweep)
    answers = np.array([m["n_touching"] for m in meta])
    caps = np.array([m["answer_cap"] for m in meta])
    assert (answers != 1).all()            # touching is symmetric: 1 is impossible
    assert 0.04 <= np.mean(answers == 0) <= 0.16, np.mean(answers == 0)
    assert answers.max() >= 10
    assert caps.max() == 12
    assert (answers <= caps).all()
    hist = Counter(answers.tolist())
    assert max(hist.values()) / len(answers) <= 0.32
    assert set(range(2, 11)) <= set(hist)
    ratio = answers / np.maximum(1, caps)
    assert 0.35 <= ratio.mean() <= 0.70
    assert np.mean(answers == caps) >= 0.05
    # both readings are spread, not just the stated one
    for key in ("answer_4", "answer_8"):
        values = np.array([m[key] for m in meta])
        assert (values != 1).all()
        assert values.max() >= 10


def test_answer_cap_is_footprint_derived(sweep):
    meta = _meta(sweep)
    area = np.array([m["grid_h"] * m["grid_w"] for m in meta])
    caps = np.array([m["answer_cap"] for m in meta])
    assert caps.min() == 2
    assert caps[area < 100].max() <= 6
    assert caps[area >= 600].min() >= 8
    for m in meta[:200]:
        assert m["answer_cap"] == module._answer_cap(m["grid_h"], m["grid_w"],
                                                     m["planned_density"])


def test_answer_does_not_follow_the_object_count(sweep):
    """The answer is capped by the object count, so some link is structural;
    the decoys and the corner chains keep the residual small."""
    meta = _meta(sweep)
    answers = [m["n_touching"] for m in meta]
    base = _baseline(answers)
    assert _heldout([m["n_objects"] for m in meta], answers) <= base + 0.12
    assert _heldout([f"{m['grid_h']}x{m['grid_w']}" for m in meta], answers) <= base + 0.05
    assert np.mean([m["n_decoys"] >= 1 for m in meta]) >= 0.50
    assert np.mean([m["n_touching"] < m["n_objects"] for m in meta]) >= 0.50


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
    assert np.mean(hs == ws) <= 0.09
    per_value = np.array([np.mean(hs == v) for v in range(5, 31)])
    assert per_value.max() <= 0.075 and per_value.min() >= 0.015
    area = hs * ws
    assert np.mean(area >= 700) >= 0.02
    assert area.max() >= 780


def test_colours(sweep):
    meta = _meta(sweep)
    counts = np.array([m["n_colors"] for m in meta])
    # Two-colour grids are structurally confined to the zero-answer bucket:
    # a touching pair needs two object colours, because two same-colour
    # objects that meet along a side are one object.
    assert counts.min() <= 3 and counts.max() == 10
    assert set(range(3, 11)) <= set(counts.tolist())
    assert np.mean(counts >= 8) >= 0.18          # measured 0.26
    assert np.mean([m["bg_color"] == 0 for m in meta]) >= 0.05
    for m in meta:
        assert m["bg_color"] not in m["object_colors"]
        if m["n_colors"] == 2:
            assert m["n_touching"] == 0


def test_pieces_are_bodies_not_confetti(sweep):
    meta = _meta(sweep)
    sizes = np.array([s for m in meta for s in m["piece_sizes"]])
    assert sizes.min() >= module.MIN_BODY
    assert np.mean(sizes <= 4) == 0.0
    assert sizes.mean() >= 7.0, sizes.mean()
    assert sizes.max() >= 60, sizes.max()
    assert min(m["min_piece_cells"] for m in meta) >= module.MIN_BODY
    for ex in sweep[:200]:
        assert ex["metadata"]["piece_sizes"] == sorted(
            len(c) for c in _components(ex["input"], 4)[0])


def test_occupancy_is_flat_by_area_band(sweep):
    meta = _meta(sweep)
    area = np.array([m["grid_h"] * m["grid_w"] for m in meta])
    occ = np.array([m["occupancy"] for m in meta])
    small = occ[area < 150].mean()
    big = occ[area >= 700].mean()
    assert big >= 0.13, big
    assert big >= small - 0.08, (small, big)
    assert small <= 0.35, small


def test_objects_are_drawn_not_noise(sweep):
    """Every piece is a clean body: a few runs per row and per column."""
    solid = thin = 0
    for ex in sweep[:200]:
        for cells in _components(ex["input"], 4)[0]:
            by_row, by_col = {}, {}
            for r, c in cells:
                by_row.setdefault(r, []).append(c)
                by_col.setdefault(c, []).append(r)

            def runs(groups):
                worst = 0
                for keys in groups.values():
                    keys = sorted(keys)
                    worst = max(worst, 1 + sum(1 for a, b in zip(keys, keys[1:]) if b - a > 1))
                return worst
            assert runs(by_row) <= 3 and runs(by_col) <= 3
            block = set(cells)
            if any((r + 1, c) in block and (r, c + 1) in block and (r + 1, c + 1) in block
                   for r, c in cells):
                solid += 1
            else:
                thin += 1
    assert solid / (solid + thin) >= 0.30, (solid, thin)


def test_textures_are_sparse_and_inferable(sweep):
    from common.backgrounds import isolated_cell_mask
    meta = _meta(sweep)
    eligible = [m for m in meta if min(m["grid_h"], m["grid_w"]) >= 8]
    share = np.mean([bool(m["texture_colors"]) for m in eligible])
    assert 0.20 <= share <= 0.45, share
    kinds = set()
    for m in meta:
        if not m["texture_colors"]:
            continue
        assert m["texture_group"] == "sparse"
        assert m["texture_named"] is False
        assert not set(m["texture_colors"]) & set(m["object_colors"] + [m["bg_color"]])
        kinds.add(m["texture_kind"])
    assert len(kinds) >= 5
    checked = 0
    for ex in sweep[:300]:
        colors = ex["metadata"]["texture_colors"]
        if not colors:
            continue
        checked += 1
        arr = np.asarray(ex["input"])
        assert isolated_cell_mask(arr)[np.isin(arr, colors)].all()
    assert checked >= 40


def test_background_is_identifiable_without_being_named(sweep):
    from common.backgrounds import infer_background
    for ex in sweep[:250]:
        info = infer_background(np.asarray(ex["input"]), margin=1.25)
        assert info["identifiable"], ex["metadata"]
        assert int(info["color"]) == ex["metadata"]["bg_color"]
        assert ex["metadata"]["background_margin"] >= 1.25


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    seen = set()
    for ex in sweep[:250]:
        prompt = get_prompt("92", ex["metadata"])
        seen.add(prompt)
        assert not re.search(r"\b(background|texture) (is|are|color)\b", prompt)
        assert "most frequent" not in prompt
        assert prompt.endswith("Answer with the number.")
        assert not re.search(r"color \d", prompt)
    assert len(seen) == 2, seen        # exactly one wording per connectivity


def test_metadata_contract(sweep):
    required = {"grid_h", "grid_w", "n_colors", "connectivity", "n_touching", "answer_cap",
                "answer_other_connectivity", "differs_under_other_connectivity",
                "answer_color_blind", "differs_under_color_blind", "answer_anchor",
                "answer_4", "answer_8", "n_objects", "n_objects_other_connectivity",
                "cluster_kinds", "cluster_sizes", "n_decoys", "n_pieces",
                "has_diagonal_only_touch", "piece_sizes", "min_piece_cells",
                "object_colors", "object_sizes", "bg_color", "background_margin",
                "occupancy", "texture_group", "texture_kind", "texture_colors",
                "texture_named", "texture_coverage", "attempts"}
    for ex in sweep[:250]:
        meta = ex["metadata"]
        assert required <= set(meta), required - set(meta)
        assert meta["connectivity"] in (4, 8)
        assert meta["n_touching"] == int(ex["output"])
        assert meta["n_touching"] <= meta["n_objects"]
        assert meta["n_objects"] >= 2          # the question needs a comparison
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        expected = meta["answer_4"] if meta["connectivity"] == 4 else meta["answer_8"]
        assert meta["n_touching"] == expected
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep]) <= 2.0


# ---------------------------------------------------------------------------
# the validator is real
# ---------------------------------------------------------------------------

def test_validator_raises_on_a_tampered_example():
    ex = _ex(3)
    module.validate_example(deepcopy(ex))          # the honest example passes

    wrong = deepcopy(ex)
    wrong["output"] = str(int(wrong["output"]) + 1)
    with pytest.raises(ValueError):
        module.validate_example(wrong)

    flipped = deepcopy(_ex(5))
    flipped["metadata"]["connectivity"] = 8 if flipped["metadata"]["connectivity"] == 4 else 4
    with pytest.raises(ValueError):
        module.validate_example(flipped)

    swapped = deepcopy(_ex(7))
    swapped["metadata"]["bg_color"] = (swapped["metadata"]["bg_color"] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(swapped)

    merged = deepcopy(_ex(9))
    merged["metadata"]["answer_color_blind"] += 1
    with pytest.raises(ValueError):
        module.validate_example(merged)

    listed = deepcopy(_ex(11))
    listed["metadata"]["piece_sizes"] = listed["metadata"]["piece_sizes"][:-1]
    with pytest.raises(ValueError):
        module.validate_example(listed)

    diagonal = deepcopy(_ex(13))
    diagonal["metadata"]["has_diagonal_only_touch"] = (
        not diagonal["metadata"]["has_diagonal_only_touch"])
    with pytest.raises(ValueError):
        module.validate_example(diagonal)

    # paint a second object against an existing one: the count must move
    for seed in range(60):
        ex = _ex(seed)
        painted = deepcopy(ex)
        grid = [list(row) for row in painted["input"]]
        h, w = len(grid), len(grid[0])
        bg = painted["metadata"]["bg_color"]
        free = [c for c in range(10) if c not in {v for row in grid for v in row}]
        spot = next(((r, c) for r in range(1, h - 1) for c in range(1, w - 2)
                     if grid[r][c] != bg and grid[r][c + 1] == bg and grid[r][c + 2] == bg), None)
        if not free or spot is None:
            continue
        grid[spot[0]][spot[1] + 1] = grid[spot[0]][spot[1] + 2] = free[0]
        painted["input"] = grid
        with pytest.raises(ValueError):
            module.validate_example(painted)
        break
    else:
        pytest.fail("no seed offered a free colour and a spot for the tamper check")


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
# the siblings tasks 91/92 left behind in generators/misc.py are untouched
# ---------------------------------------------------------------------------

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "sibling_hashes_pack_edge_91_92.json")


def test_misc_siblings_are_byte_identical():
    with open(FIXTURE) as handle:
        expected = json.load(handle)
    assert set(expected["sha1"]) == {"31", "35", "56", "57", "58", "59", "80",
                                     "82", "83", "90"}
    for tid, digest in sorted(expected["sha1"].items()):
        got = hashlib.sha1()
        for seed in range(expected["n_seeds"]):
            got.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert got.hexdigest() == digest, tid
