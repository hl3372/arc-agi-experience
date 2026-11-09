"""Task 99 -- complete a symmetry (tasks 99 and 100 merged on 2026-09-15).

The oracle below is written from the prompt text alone: the transform comes out
of the coordinate sentence, the unknown colour out of the first sentence, and
the fill is a plain orbit walk.  Nothing here imports the generator's ``solve``,
so a bug shared with the module cannot hide itself.
"""
from collections import Counter
import hashlib
import json
import os
import re
import time

import pytest

from generators import symmetry_fill_99 as module
from prompts import get_prompt
from registry import generate
from symbol_remap import palette_for_example

ORACLE_SEEDS = 400
SWEEP_SEEDS = 700

TRANSFORMS = ("horizontal_mirror", "vertical_mirror", "main_diagonal",
              "anti_diagonal", "rotation_90", "rotation_180")
SQUARE_ONLY = {"main_diagonal", "anti_diagonal", "rotation_90"}
LIBRARY_FAMILIES = {"polyomino", "bar", "rect", "square", "frame", "L", "T", "U",
                    "plus", "H", "staircase", "blob", "big_blob", "scaled"}


@pytest.fixture(scope="module")
def sweep():
    return [generate("99", seed) for seed in range(SWEEP_SEEDS)]


# ---------------------------------------------------------------------------
# an oracle that reads the prompt, not the metadata
# ---------------------------------------------------------------------------

_RULES = {
    "row r, column c maps to row height - 1 - r, column c":
        lambda r, c, h, w: (h - 1 - r, c),
    "row r, column c maps to row r, column width - 1 - c":
        lambda r, c, h, w: (r, w - 1 - c),
    "row r, column c maps to row c, column r":
        lambda r, c, h, w: (c, r),
    "row r, column c maps to row width - 1 - c, column height - 1 - r":
        lambda r, c, h, w: (w - 1 - c, h - 1 - r),
    "row r, column c maps to row c, column height - 1 - r":
        lambda r, c, h, w: (c, h - 1 - r),
    "row r, column c maps to row height - 1 - r, column width - 1 - c":
        lambda r, c, h, w: (h - 1 - r, w - 1 - c),
}


def _rule_from_prompt(prompt):
    found = [rule for rule in _RULES if f"coordinate rule: {rule}." in prompt]
    assert len(found) == 1, prompt
    return _RULES[found[0]]


def _unknown_from_prompt(prompt):
    match = re.search(r"Only cells of unknown color (\d) may change\.", prompt)
    assert match, prompt
    return int(match.group(1))


def _oracle(rows, rule, unknown):
    """Walk the orbit of every cell and copy the one known colour onto it."""
    h, w = len(rows), len(rows[0])
    out = [list(row) for row in rows]
    for r in range(h):
        for c in range(w):
            seen = []
            cur = (r, c)
            while cur not in seen:
                seen.append(cur)
                cur = rule(cur[0], cur[1], h, w)
            known = {rows[a][b] for a, b in seen} - {unknown}
            assert len(known) <= 1, (r, c, known)
            if known:
                out[r][c] = known.pop()
    return out


def test_oracle_agrees_with_the_answer():
    for seed in range(ORACLE_SEEDS):
        ex = generate("99", seed)
        prompt = get_prompt("99", ex["metadata"])
        rule = _rule_from_prompt(prompt)
        unknown = _unknown_from_prompt(prompt)
        assert unknown == ex["metadata"]["unknown_color"], seed
        assert _oracle(ex["input"], rule, unknown) == ex["output"], seed


def test_examples_are_deterministic_and_serializable():
    for seed in range(40):
        ex = generate("99", seed)
        json.dumps(ex)
        assert generate("99", seed) == ex, seed


def test_prompt_renders_under_the_symbol_palette():
    from sft_wrapper import format_single
    for seed in range(12):
        prompt, answer = format_single("99", seed, color_palette=palette_for_example("99", seed))
        assert prompt and answer


# ---------------------------------------------------------------------------
# the transform is drawn per example; the canvas follows it
# ---------------------------------------------------------------------------

def test_transform_and_square_shares(sweep):
    kinds = Counter(ex["metadata"]["transform"] for ex in sweep)
    assert set(kinds) == set(TRANSFORMS), kinds
    for name, count in kinds.items():
        assert 0.10 <= count / len(sweep) <= 0.25, (name, count)
    for ex in sweep:
        meta = ex["metadata"]
        square = meta["grid_h"] == meta["grid_w"]
        assert meta["square"] is square
        if meta["transform"] in SQUARE_ONLY:
            assert square, meta
    # the three square-only rules force about half the grids; the axis mirrors
    # and the half turn draw their sides independently and land on a square only
    # by coincidence
    rect = [ex["metadata"] for ex in sweep if ex["metadata"]["transform"] not in SQUARE_ONLY]
    coincidental = sum(1 for m in rect if m["grid_h"] == m["grid_w"]) / len(rect)
    assert coincidental <= 0.12, coincidental
    overall = sum(1 for ex in sweep if ex["metadata"]["square"]) / len(sweep)
    assert 0.40 <= overall <= 0.60, overall


def test_size_coverage(sweep):
    sides = Counter()
    for ex in sweep:
        sides[ex["metadata"]["grid_h"]] += 1
        sides[ex["metadata"]["grid_w"]] += 1
    assert min(sides) == 5 and max(sides) == 30, sorted(sides)
    assert set(sides) == set(range(5, 31)), sorted(sides)
    total = sum(sides.values())
    for side, count in sides.items():
        assert 0.015 <= count / total <= 0.07, (side, count / total)
    areas = [ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] for ex in sweep]
    assert all(area <= 900 for area in areas)
    assert sum(1 for area in areas if area >= 700) / len(areas) >= 0.04


def test_color_coverage(sweep):
    counts = Counter(ex["metadata"]["n_colors"] for ex in sweep)
    assert min(counts) == 2, sorted(counts)
    # the unknown colour is reserved and never occurs in the picture, so the
    # answer reaches nine colours and the input reaches ten
    assert max(counts) == 9, sorted(counts)
    for value in range(2, 10):
        assert counts[value] / len(sweep) >= 0.02, (value, counts[value])
    assert sum(counts[v] for v in (7, 8, 9)) / len(sweep) >= 0.15
    for ex in sweep:
        meta = ex["metadata"]
        picture = {v for row in ex["output"] for v in row} - {meta["unknown_color"]}
        assert meta["unknown_color"] not in picture
        assert meta["n_colors"] == len(picture)
        assert meta["ground_color"] != meta["unknown_color"]
        assert meta["unknown_color"] not in meta["body_colors"]
        assert meta["ground_color"] not in meta["body_colors"]
        assert meta["unknown_color"] not in meta["texture_colors"]
        assert not set(meta["texture_colors"]) & set(meta["body_colors"])


def test_texture_and_occupancy(sweep):
    groups = Counter(ex["metadata"]["texture_group"] for ex in sweep)
    eligible = [ex for ex in sweep
                if min(ex["metadata"]["grid_h"], ex["metadata"]["grid_w"]) >= 8]
    textured = sum(1 for ex in eligible if ex["metadata"]["texture_group"] != "none")
    assert 0.12 <= textured / len(eligible) <= 0.40, textured / len(eligible)
    assert groups["sparse"] > 0 and groups["solid"] > 0
    for ex in sweep:
        meta = ex["metadata"]
        if meta["texture_group"] == "none":
            assert meta["texture_colors"] == [] and meta["texture_coverage"] == 0.0
        else:
            assert min(meta["grid_h"], meta["grid_w"]) >= 8
            assert meta["texture_coverage"] <= 0.55
            assert meta["texture_named"] is False
    occ = [ex["metadata"]["occupancy"] for ex in sweep]
    assert 0.18 <= sum(occ) / len(occ) <= 0.45, sum(occ) / len(occ)
    big = [ex["metadata"]["occupancy"] for ex in sweep
           if ex["metadata"]["grid_h"] * ex["metadata"]["grid_w"] >= 700]
    assert big and sum(big) / len(big) >= 0.18, sum(big) / len(big)


# ---------------------------------------------------------------------------
# masking: determined, bounded, and honest about what stays unknown
# ---------------------------------------------------------------------------

def _orbits(h, w, rule):
    seen = set()
    out = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            cells = []
            cur = (r, c)
            while cur not in cells:
                cells.append(cur)
                cur = rule(cur[0], cur[1], h, w)
            seen.update(cells)
            out.append(tuple(cells))
    return out


def test_every_masked_cell_is_decided(sweep):
    for ex in sweep:
        meta = ex["metadata"]
        rows, out = ex["input"], ex["output"]
        unknown = meta["unknown_color"]
        rule = _rule_from_prompt(get_prompt("99", meta))
        h, w = meta["grid_h"], meta["grid_w"]
        decoys = axis = 0
        for orbit in _orbits(h, w, rule):
            colors = {rows[r][c] for r, c in orbit} - {unknown}
            assert len(colors) <= 1
            if not colors:
                if len(orbit) == 1:
                    axis += 1
                else:
                    decoys += 1
                for r, c in orbit:
                    assert out[r][c] == unknown
            else:
                color = next(iter(colors))
                for r, c in orbit:
                    assert out[r][c] == color
        assert decoys == meta["n_decoy_orbits"]
        assert axis == meta["n_axis_unknowns"]
        # nothing known ever moves
        for r in range(h):
            for c in range(w):
                if rows[r][c] != unknown:
                    assert out[r][c] == rows[r][c]


def test_masking_bounds(sweep):
    n = len(sweep)
    decoy = sum(1 for ex in sweep if ex["metadata"]["n_decoy_orbits"] > 0) / n
    axis = sum(1 for ex in sweep if ex["metadata"]["n_axis_unknowns"] > 0) / n
    identity = sum(1 for ex in sweep if ex["metadata"]["identity_output"]) / n
    assert 0.20 <= decoy <= 0.50, decoy
    assert 0.15 <= axis <= 0.45, axis
    assert 0.02 <= identity <= 0.12, identity
    for ex in sweep:
        meta = ex["metadata"]
        area = meta["grid_h"] * meta["grid_w"]
        assert meta["n_unknown"] >= 1
        assert meta["n_unknown"] / area <= 0.2501, meta["unknown_share"]
        assert meta["identity_output"] == (ex["input"] == ex["output"])
        if not meta["identity_output"]:
            assert meta["n_filled"] >= 1
        else:
            assert meta["n_filled"] == 0
        # a cell on the axis can only be one of the fixed cells of the transform
        assert meta["n_axis_unknowns"] <= meta["n_fixed_cells"]
    filled = [ex["metadata"]["n_filled"] for ex in sweep]
    assert max(filled) >= 40


# ---------------------------------------------------------------------------
# naturalness: drawn bodies, not per-cell noise
# ---------------------------------------------------------------------------

def _components(rows, colors):
    h, w = len(rows), len(rows[0])
    seen = [[False] * w for _ in range(h)]
    sizes = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or rows[r][c] not in colors:
                continue
            color = rows[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            size = 0
            while stack:
                a, b = stack.pop()
                size += 1
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < h and 0 <= nb < w and not seen[na][nb] \
                            and rows[na][nb] == color:
                        seen[na][nb] = True
                        stack.append((na, nb))
            sizes.append(size)
    return sizes


def test_bodies_are_drawn_not_noise(sweep):
    tiny = 0
    total = 0
    for ex in sweep:
        meta = ex["metadata"]
        assert set(meta["body_families"]) <= LIBRARY_FAMILIES, meta["body_families"]
        assert meta["n_bodies"] >= 1
        # each body contributes at most one block per copy the transform makes
        assert meta["n_scene_components"] <= meta["max_orbit"] * meta["n_bodies"], meta
        assert meta["max_orbit"] == (4 if meta["transform"] == "rotation_90" else 2)
        if meta["n_decoy_orbits"] == 0 and meta["n_axis_unknowns"] == 0:
            # with nothing left unknown the answer IS the drawn picture, so the
            # component sizes can be recomputed from the example alone
            sizes = _components(ex["output"], set(meta["body_colors"]))
            assert len(sizes) == meta["n_scene_components"], meta
            assert sum(sizes) / len(sizes) >= 3.0, (meta["transform"], sizes)
            tiny += sum(1 for s in sizes if s == 1)
            total += len(sizes)
    assert total > 100
    assert tiny / total <= 0.06, tiny / total


def test_a_body_meets_its_own_image_on_a_real_share(sweep):
    share = sum(1 for ex in sweep if ex["metadata"]["n_self_symmetric_bodies"] > 0) / len(sweep)
    assert share >= 0.40, share


def test_the_picture_is_exactly_symmetric(sweep):
    for ex in sweep:
        meta = ex["metadata"]
        rule = _rule_from_prompt(get_prompt("99", meta))
        h, w = meta["grid_h"], meta["grid_w"]
        for r in range(h):
            for c in range(w):
                rr, cc = rule(r, c, h, w)
                assert ex["output"][rr][cc] == ex["output"][r][c], (r, c)


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    wordings = {}
    for ex in sweep[:400]:
        meta = ex["metadata"]
        prompt = get_prompt("99", meta)
        assert "Rows and columns are 0-indexed from the top-left corner." in prompt
        assert "Leave every other cell unchanged." in prompt
        assert "Only cells of unknown color" in prompt
        assert "coordinate rule" in prompt
        assert "Return the completed grid." in prompt
        assert "zero-indexed" not in prompt.lower()
        # the ground and the texture are never named or defined
        assert "background" not in prompt.lower()
        assert "most frequent" not in prompt.lower()
        wordings.setdefault(meta["transform"], set()).add(
            prompt.replace(f"color {meta['unknown_color']}", "color U"))
    assert set(wordings) == set(TRANSFORMS)
    for name, texts in wordings.items():
        assert len(texts) == 1, (name, texts)
    assert "rotationally symmetric by 90 degrees" in next(iter(wordings["rotation_90"]))
    assert "mapped group of four" in next(iter(wordings["rotation_90"]))
    assert "anti-diagonal" in next(iter(wordings["anti_diagonal"]))


LETTERS = list("ABCDEFGHIJ")


def test_only_the_unknown_color_is_named():
    for seed in range(60):
        ex = generate("99", seed)
        prompt = get_prompt("99", ex["metadata"], color_palette=LETTERS)
        named = set(re.findall(r"\b[A-J]\b", prompt))
        assert named == {LETTERS[ex["metadata"]["unknown_color"]]}, (seed, named)
        assert not re.search(r"\bcolor [0-9]\b", prompt), prompt
        symbol = get_prompt("99", ex["metadata"], color_palette=palette_for_example("99", seed))
        assert not re.search(r"\bcolor [0-9]\b", symbol), symbol


# ---------------------------------------------------------------------------
# the validator earns its place
# ---------------------------------------------------------------------------

def test_validator_raises_on_a_tampered_answer():
    ex = next(generate("99", seed) for seed in range(20)
              if generate("99", seed)["metadata"]["n_filled"] > 0)
    meta = ex["metadata"]
    unknown = meta["unknown_color"]

    bad = json.loads(json.dumps(ex))
    for r, row in enumerate(bad["input"]):
        for c, value in enumerate(row):
            if value == unknown:
                bad["output"][r][c] = (bad["output"][r][c] + 1) % 10
                break
        else:
            continue
        break
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["metadata"]["n_filled"] += 1
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["output"][0][0] = (bad["output"][0][0] + 1) % 10
    with pytest.raises(ValueError):
        module.validate_example(bad)

    module.validate_example(json.loads(json.dumps(ex)))


def test_transform_rejects_a_canvas_it_cannot_map():
    for name in sorted(SQUARE_ONLY):
        with pytest.raises(ValueError):
            module.map_cell(0, 0, (5, 7), name)
    with pytest.raises(ValueError):
        module.map_cell(0, 0, (5, 5), "spiral")


def test_task_100_is_retired():
    with pytest.raises(KeyError) as excinfo:
        generate("100", 0)
    assert "merged into task 99" in str(excinfo.value)
    from registry import RETIRED, list_tasks
    assert RETIRED["100"] == "99"
    assert "100" not in list_tasks()
    assert "99" in list_tasks()


def test_generation_is_cheap():
    start = time.perf_counter()
    for seed in range(200):
        generate("99", seed)
    per_seed = (time.perf_counter() - start) / 200
    assert per_seed < 0.09, per_seed


# ---------------------------------------------------------------------------
# the siblings left behind in generators/core2_extra/batch_symmetry_f.py
# ---------------------------------------------------------------------------

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "sibling_hashes_symmetry_fill_99.json")


def test_symmetry_batch_siblings_are_byte_identical():
    with open(FIXTURE) as handle:
        expected = json.load(handle)
    assert set(expected["sha1"]) == {"103", "104", "107", "120",
                                     "127", "130", "281"}
    for tid, digest in sorted(expected["sha1"].items()):
        got = hashlib.sha1()
        for seed in range(expected["n_seeds"]):
            got.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert got.hexdigest() == digest, tid
