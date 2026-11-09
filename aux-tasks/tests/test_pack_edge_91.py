"""Task 91: pack every object against one edge.

The oracle below is written from the prompt text alone -- infer the ground,
group the remaining cells into same-colour 4-connected components of at least
two cells, slide them toward the named edge closest-first -- with a mask-based
packing that shares no code with the generator's readers.
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

from generators import pack_edge_91 as module
from prompts import get_prompt
from registry import generate

ORACLE_SEEDS = 400
SWEEP_SEEDS = 1200

_STEP = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


def _ex(seed):
    return generate("91", seed)


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------

def _direction_from_prompt(prompt):
    assert "Ignore the background, whether it is a solid color or a pattern." in prompt
    assert "at least two cells" in prompt
    assert "isolated single cells, whatever their color, are background texture" in prompt
    assert "always separate, even where they touch" in prompt
    assert "The cells an object leaves behind become background." in prompt
    assert prompt.endswith("Leave every other cell unchanged.")
    match = re.search(r"Pack every object against the (\w+) edge\.", prompt)
    assert match, prompt
    edge = match.group(1)
    assert f"Objects closest to the {edge} edge settle first" in prompt
    return edge


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
    counts = Counter(arr[~lonely].ravel().tolist())
    return counts.most_common(1)[0][0]


def _label(mask):
    """Row-major union-find labelling of a boolean mask (4-connectivity)."""
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

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            parent.setdefault(r * w + c, r * w + c)
            for dr, dc in ((-1, 0), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    parent.setdefault(rr * w + cc, rr * w + cc)
                    union(r * w + c, rr * w + cc)
    groups = {}
    for key in parent:
        groups.setdefault(find(key), []).append((key // w, key % w))
    return list(groups.values())


def _read(grid):
    """(objects, colours, texture cells, ground) exactly as the prompt defines them."""
    arr = np.asarray(grid)
    ground = _ground(arr)
    objects, colors = [], []
    covered = set()
    for color in np.unique(arr):
        if color == ground:
            continue
        for cells in _label(arr == color):
            if len(cells) < 2:
                continue
            objects.append(frozenset(cells))
            colors.append(int(color))
            covered |= set(cells)
    texture = [(int(r), int(c)) for r, c in np.argwhere(arr != ground)
               if (int(r), int(c)) not in covered]
    return objects, colors, texture, int(ground)


def _free_shift(mask, blocked, direction):
    """The largest shift of a boolean mask that stays on the grid and clear."""
    dr, dc = _STEP[direction]
    h, w = mask.shape
    shift = 0
    while True:
        nxt = shift + 1
        rr, cc = dr * nxt, dc * nxt
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows[0] + rr < 0 or rows[-1] + rr >= h or cols[0] + cc < 0 or cols[-1] + cc >= w:
            return shift
        moved = np.zeros_like(mask)
        src_r0, src_r1 = max(0, -rr), h - max(0, rr)
        src_c0, src_c1 = max(0, -cc), w - max(0, cc)
        moved[src_r0 + rr:src_r1 + rr, src_c0 + cc:src_c1 + cc] = mask[src_r0:src_r1, src_c0:src_c1]
        if (moved & blocked).any():
            return shift
        shift = nxt


def oracle(grid, prompt):
    """The packed grid, computed from masks rather than cell sets."""
    direction = _direction_from_prompt(prompt)
    arr = np.asarray(grid)
    h, w = arr.shape
    objects, colors, texture, ground = _read(arr)
    dr, dc = _STEP[direction]

    def distance(cells):
        if direction == "top":
            return min(r for r, _ in cells)
        if direction == "bottom":
            return h - 1 - max(r for r, _ in cells)
        if direction == "left":
            return min(c for _, c in cells)
        return w - 1 - max(c for _, c in cells)

    order = sorted(range(len(objects)),
                   key=lambda i: (distance(objects[i]), sorted(objects[i])))
    blocked = np.zeros((h, w), dtype=bool)
    out = np.full((h, w), ground)
    for i in order:
        mask = np.zeros((h, w), dtype=bool)
        for r, c in objects[i]:
            mask[r, c] = True
        shift = _free_shift(mask, blocked, direction)
        placed = {(r + dr * shift, c + dc * shift) for r, c in objects[i]}
        for r, c in placed:
            blocked[r, c] = True
            out[r, c] = colors[i]
    for r, c in texture:
        out[r, c] = arr[r, c]
    return out.tolist(), ground, direction


def test_independent_oracle():
    for seed in range(ORACLE_SEEDS):
        ex = _ex(seed)
        packed, ground, direction = oracle(ex["input"], get_prompt("91", ex["metadata"]))
        assert packed == ex["output"], seed
        assert ground == ex["metadata"]["bg_color"], seed
        assert direction == ex["metadata"]["direction"], seed


def test_shared_body_floor_matches_task_90():
    """The confetti floor is task 90's; a drift there would be silent."""
    from generators import border_objects_90
    assert module.MIN_BODY == border_objects_90.MIN_BODY


def test_determinism_and_cell_range():
    for seed in range(60):
        a, b = _ex(seed), _ex(seed)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), seed
        grid = np.asarray(a["input"])
        assert grid.min() >= 0 and grid.max() <= 9
        out = np.asarray(a["output"])
        assert out.shape == grid.shape


# ---------------------------------------------------------------------------
# sweep helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep():
    return [_ex(seed) for seed in range(SWEEP_SEEDS)]


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
    assert np.mean(hs == ws) <= 0.09                     # squares only by chance
    per_value = np.array([np.mean(hs == v) for v in range(5, 31)])
    assert per_value.max() <= 0.075 and per_value.min() >= 0.015
    area = hs * ws
    assert np.mean(area >= 700) >= 0.02                  # flat sides give ~4.7%
    assert area.max() >= 780


def test_colours(sweep):
    meta = _meta(sweep)
    counts = np.array([m["n_colors"] for m in meta])
    assert counts.min() == 2 and counts.max() == 10
    assert set(counts.tolist()) >= set(range(2, 11))
    assert np.mean(counts >= 8) >= 0.13                  # measured 0.18
    assert np.mean(counts == 2) <= 0.20
    assert np.mean([m["bg_color"] == 0 for m in meta]) >= 0.05   # 0 is ordinary
    # the palette no longer equals n_objects + 1
    assert np.mean([m["colors_repeat"] for m in meta]) >= 0.20
    for m in meta:
        assert m["bg_color"] not in m["object_colors"]


def test_colour_count_is_not_the_object_count(sweep):
    """The legacy version drew exactly ``n_objects`` object colours."""
    meta = _meta(sweep)
    exact = np.mean([m["n_object_colors"] == m["n_objects"] for m in meta])
    assert exact <= 0.80, exact
    answers = [m["n_objects"] for m in meta]
    assert _heldout([m["n_colors"] for m in meta], answers) <= _baseline(answers) + 0.30


def test_directions_are_balanced_and_stated(sweep):
    meta = _meta(sweep)
    share = Counter(m["direction"] for m in meta)
    assert set(share) == {"top", "bottom", "left", "right"}
    for value in share.values():
        assert 0.20 <= value / len(meta) <= 0.30
    for ex in sweep[:120]:
        prompt = get_prompt("91", ex["metadata"])
        assert _direction_from_prompt(prompt) == ex["metadata"]["direction"]


def test_objects_scale_with_the_canvas(sweep):
    meta = _meta(sweep)
    area = np.array([m["grid_h"] * m["grid_w"] for m in meta])
    occ = np.array([m["occupancy"] for m in meta])
    small = occ[area < 150].mean()
    big = occ[area >= 700].mean()
    assert big >= 0.12, big                    # no sea of background on 30x30
    assert big >= small - 0.08, (small, big)
    assert small <= 0.33, small
    sizes = np.array([s for m in meta for s in m["object_sizes"]])
    assert sizes.min() >= 5                    # the confetti floor
    assert sizes.mean() >= 7.0, sizes.mean()
    assert sizes.max() >= 60, sizes.max()
    big_sizes = np.array([s for m in meta if m["grid_h"] * m["grid_w"] >= 400
                          for s in m["object_sizes"]])
    assert big_sizes.mean() >= sizes.mean(), (big_sizes.mean(), sizes.mean())
    families = Counter(k.replace("_clipped", "") for m in meta for k in m["object_kinds"])
    assert set(families) >= {"rect", "bar", "frame", "L", "staircase", "square"}
    assert families.most_common(1)[0][1] / sum(families.values()) <= 0.40


def test_objects_are_drawn_not_noise(sweep):
    """Every object is one clean body: a few runs per row and per column."""
    solid = thin = 0
    for ex in sweep[:250]:
        objects, _, _, _ = _read(ex["input"])
        for cells in objects:
            assert len(cells) >= 5, (ex["metadata"]["grid_h"], len(cells))
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


def test_textures_are_sparse_inferable_and_static(sweep):
    from common.backgrounds import isolated_cell_mask
    meta = _meta(sweep)
    eligible = [m for m in meta if min(m["grid_h"], m["grid_w"]) >= 8]
    share = np.mean([bool(m["texture_colors"]) for m in eligible])
    assert 0.18 <= share <= 0.45, share
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
        out = np.asarray(ex["output"])
        tex = np.isin(arr, colors)
        assert isolated_cell_mask(arr)[tex].all()
        # texture is background: it never moves and nothing is ever slid over it
        assert (out[tex] == arr[tex]).all()
        objects, _, _, _ = _read(arr)
        packed = module.consistent_packing(objects, *arr.shape, ex["metadata"]["direction"])
        swept = module._swept(objects, packed, ex["metadata"]["direction"])
        assert not any(tex[r, c] for r, c in swept)
    assert checked >= 40


# ---------------------------------------------------------------------------
# the settle order is the skill
# ---------------------------------------------------------------------------

def test_objects_block_each_other(sweep):
    meta = _meta(sweep)
    assert np.mean([m["n_blocked"] >= 1 for m in meta]) >= 0.50
    assert np.mean([m["n_blocked"] >= 2 for m in meta]) >= 0.25
    assert max(m["n_blocked"] for m in meta) >= 5


def test_settle_order_is_load_bearing(sweep):
    """Reading "closest first" the other way round changes the output."""
    meta = _meta(sweep)
    assert np.mean([m["differs_under_farthest_first"] for m in meta]) >= 0.40
    assert np.mean([m["differs_under_reading_order"] for m in meta]) >= 0.20
    # the counterfactual is genuinely recomputed, not asserted
    for ex in sweep[:60]:
        objects, _, _, _ = _read(ex["input"])
        h, w = len(ex["input"]), len(ex["input"][0])
        direction = ex["metadata"]["direction"]
        packed = module.consistent_packing(objects, h, w, direction)
        alt = module.pack_closest_first(
            objects, h, w, direction,
            order=module._farthest_order(objects, h, w, direction))
        assert (alt != packed) == ex["metadata"]["differs_under_farthest_first"]


def test_readings_of_the_rule_agree_on_every_example(sweep):
    """"Closest settles first" and "slide until nothing moves" never disagree."""
    for ex in sweep[:250]:
        objects, _, _, _ = _read(ex["input"])
        h, w = len(ex["input"]), len(ex["input"][0])
        direction = ex["metadata"]["direction"]
        base = module.pack_closest_first(objects, h, w, direction)
        assert module.pack_relaxation(objects, h, w, direction) == base
        assert module.pack_relaxation(objects, h, w, direction, reverse=True) == base
        assert module.pack_closest_first_runs(objects, h, w, direction) == base


def test_colour_separation_is_load_bearing(sweep):
    """A reader who merged touching different-colour objects gets it wrong."""
    meta = _meta(sweep)
    assert np.mean([m["has_contact"] for m in meta]) >= 0.15
    assert np.mean([m["differs_under_color_blind"] for m in meta]) >= 0.08


def test_identity_and_motion(sweep):
    meta = _meta(sweep)
    identity = np.mean([m["identity_output"] for m in meta])
    assert 0.05 <= identity <= 0.20, identity
    for ex in sweep[:200]:
        same = ex["input"] == ex["output"]
        assert same == ex["metadata"]["identity_output"]
    moved = np.array([m["n_moved"] for m in meta])
    assert moved.max() >= 8
    assert np.mean([m["max_shift"] >= 10 for m in meta]) >= 0.30


def test_size_does_not_give_the_object_count(sweep):
    meta = _meta(sweep)
    counts = [m["n_objects"] for m in meta]
    base = _baseline(counts)
    assert _heldout([f"{m['grid_h']}x{m['grid_w']}" for m in meta], counts) <= base + 0.20


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    seen = set()
    for ex in sweep[:250]:
        prompt = get_prompt("91", ex["metadata"])
        seen.add(prompt)
        assert not re.search(r"\b(background|texture) (is|are|color)\b", prompt)
        assert "most frequent" not in prompt
        assert "The cells an object leaves behind become background." in prompt
        assert prompt.endswith("Leave every other cell unchanged.")
        assert "(gravity stacking)" in prompt
        assert not re.search(r"color \d", prompt)
    # only the edge word changes: one wording per direction
    assert len(seen) == 4, sorted(seen)
    stripped = {re.sub(r"\b(top|bottom|left|right)\b", "EDGE", p) for p in seen}
    assert len(stripped) == 1


def test_metadata_contract(sweep):
    required = {"grid_h", "grid_w", "n_colors", "direction", "n_objects", "n_blocked",
                "identity_output", "object_sizes", "object_colors", "object_kinds",
                "texture_group", "texture_kind", "texture_colors", "texture_named",
                "texture_coverage", "bg_color", "background_margin", "occupancy",
                "differs_under_farthest_first", "differs_under_reading_order",
                "differs_under_color_blind", "attempts"}
    for ex in sweep[:250]:
        meta = ex["metadata"]
        assert required <= set(meta), required - set(meta)
        assert meta["direction"] in ("top", "bottom", "left", "right")
        assert meta["grid_h"] == len(ex["input"]) and meta["grid_w"] == len(ex["input"][0])
        assert meta["n_objects"] >= 2
        assert meta["n_blocked"] <= meta["n_objects"]
        assert meta["object_sizes"] == sorted(len(c) for c in _read(ex["input"])[0])
        assert meta["background_margin"] >= 1.25
    assert np.mean([ex["metadata"]["attempts"] for ex in sweep]) <= 2.0


def test_background_is_identifiable_without_being_named(sweep):
    from common.backgrounds import infer_background
    for ex in sweep[:250]:
        info = infer_background(np.asarray(ex["input"]), margin=1.25)
        assert info["identifiable"], ex["metadata"]
        assert int(info["color"]) == ex["metadata"]["bg_color"]


# ---------------------------------------------------------------------------
# the validator is real
# ---------------------------------------------------------------------------

def test_validator_raises_on_a_tampered_example():
    ex = _ex(5)
    module.validate_example(deepcopy(ex))              # the honest example passes

    moved = deepcopy(ex)
    grid = [list(row) for row in moved["output"]]
    bg = moved["metadata"]["bg_color"]
    spot = next((r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg)
    grid[spot[0]][spot[1]] = bg
    moved["output"] = grid
    with pytest.raises(ValueError):
        module.validate_example(moved)

    turned = deepcopy(_ex(6))
    order = ["top", "bottom", "left", "right"]
    turned["metadata"]["direction"] = order[(order.index(turned["metadata"]["direction"]) + 1) % 4]
    with pytest.raises(ValueError):
        module.validate_example(turned)

    swapped = deepcopy(_ex(9))
    swapped["metadata"]["bg_color"] = (swapped["metadata"]["bg_color"] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(swapped)

    counted = deepcopy(_ex(11))
    counted["metadata"]["n_objects"] += 1
    with pytest.raises(ValueError):
        module.validate_example(counted)

    blocked = deepcopy(_ex(13))
    blocked["metadata"]["n_blocked"] += 1
    with pytest.raises(ValueError):
        module.validate_example(blocked)

    flagged = deepcopy(_ex(15))
    flagged["metadata"]["identity_output"] = not flagged["metadata"]["identity_output"]
    with pytest.raises(ValueError):
        module.validate_example(flagged)

    order_flag = deepcopy(_ex(17))
    order_flag["metadata"]["differs_under_farthest_first"] = (
        not order_flag["metadata"]["differs_under_farthest_first"])
    with pytest.raises(ValueError):
        module.validate_example(order_flag)


def test_consistent_packing_rejects_an_order_dependent_scene():
    """A layout where the two readings disagree is never emitted."""
    # the L and the block in its notch: "closest first" puts the block under the
    # L, "slide until nothing moves" lets the block pass it first
    ell = frozenset({(3, 0), (4, 0), (5, 0), (5, 1), (5, 2)})
    block = frozenset({(4, 1), (4, 2)})
    assert module.pack_closest_first([ell, block], 8, 4, "top") != \
        module.pack_relaxation([ell, block], 8, 4, "top")
    assert module.consistent_packing([ell, block], 8, 4, "top") is None


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
    """Moving tasks 91 and 92 out of generators/misc.py changes nothing else.

    The 83 pin was refreshed for its approved 2026-09-16 background pass.
    """
    with open(FIXTURE) as handle:
        expected = json.load(handle)
    assert set(expected["sha1"]) == {"31", "35", "56", "57", "58", "59", "80",
                                     "82", "83", "90"}
    for tid, digest in sorted(expected["sha1"].items()):
        got = hashlib.sha1()
        for seed in range(expected["n_seeds"]):
            got.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert got.hexdigest() == digest, tid
