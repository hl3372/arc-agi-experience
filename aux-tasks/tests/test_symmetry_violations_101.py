"""Task 101 -- erase the cells that break one colour's symmetry.

The oracle below is written from the prompt text alone: the transform comes out
of the coordinate sentence, the two colours out of the wording, and the edit is
one pass over the grid.  Nothing here imports the generator's ``solve``, so a
bug shared with the module cannot hide itself.
"""
from collections import Counter
import json
import re
import time

import pytest

from generators import symmetry_violations_101 as module
from prompts import get_prompt
from registry import generate
from symbol_remap import palette_for_example

ORACLE_SEEDS = 400
SWEEP_SEEDS = 700

TRANSFORMS = ("horizontal_mirror", "vertical_mirror", "main_diagonal",
              "anti_diagonal", "rotation_180")
SQUARE_ONLY = {"main_diagonal", "anti_diagonal"}
LIBRARY_FAMILIES = {"polyomino", "bar", "rect", "square", "frame", "L", "T", "U",
                    "plus", "H", "staircase", "blob", "big_blob", "scaled"}


@pytest.fixture(scope="module")
def sweep():
    return [generate("101", seed) for seed in range(SWEEP_SEEDS)]


# ---------------------------------------------------------------------------
# an oracle that reads the prompt, not the metadata
# ---------------------------------------------------------------------------

_RULES = {
    "row r, column c mirrors to row height - 1 - r, column c":
        lambda r, c, h, w: (h - 1 - r, c),
    "row r, column c mirrors to row r, column width - 1 - c":
        lambda r, c, h, w: (r, w - 1 - c),
    "row r, column c mirrors to row c, column r":
        lambda r, c, h, w: (c, r),
    "row r, column c mirrors to row width - 1 - c, column height - 1 - r":
        lambda r, c, h, w: (w - 1 - c, h - 1 - r),
    "row r, column c mirrors to row height - 1 - r, column width - 1 - c":
        lambda r, c, h, w: (h - 1 - r, w - 1 - c),
}


def _rule_from_prompt(prompt):
    found = [rule for rule in _RULES if f": {rule}." in prompt]
    assert len(found) == 1, prompt
    return _RULES[found[0]]


def _colors_from_prompt(prompt):
    target = re.search(r"Look only at cells of color (\d)\.", prompt)
    replacement = re.search(r"recolor that cell to color (\d)\.", prompt)
    assert target and replacement, prompt
    return int(target.group(1)), int(replacement.group(1))


def _oracle(rows, rule, target, replacement):
    """Every cell of the target colour whose mirrored partner is another colour."""
    h, w = len(rows), len(rows[0])
    out = [list(row) for row in rows]
    for r in range(h):
        for c in range(w):
            if rows[r][c] != target:
                continue
            rr, cc = rule(r, c, h, w)
            if rows[rr][cc] != target:
                out[r][c] = replacement
    return out


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


def test_oracle_agrees_with_the_answer():
    for seed in range(ORACLE_SEEDS):
        ex = generate("101", seed)
        prompt = get_prompt("101", ex["metadata"])
        rule = _rule_from_prompt(prompt)
        target, replacement = _colors_from_prompt(prompt)
        assert target == ex["metadata"]["target_color"], seed
        assert replacement == ex["metadata"]["replacement_color"], seed
        assert _oracle(ex["input"], rule, target, replacement) == ex["output"], seed


def test_examples_are_deterministic_and_serializable():
    for seed in range(40):
        ex = generate("101", seed)
        json.dumps(ex)
        assert generate("101", seed) == ex, seed


def test_prompt_renders_under_the_symbol_palette():
    from sft_wrapper import format_single
    for seed in range(12):
        prompt, answer = format_single("101", seed, color_palette=palette_for_example("101", seed))
        assert prompt and answer


# ---------------------------------------------------------------------------
# the transform is drawn per example; the canvas follows it
# ---------------------------------------------------------------------------

def test_transform_and_square_shares(sweep):
    kinds = Counter(ex["metadata"]["transform"] for ex in sweep)
    assert set(kinds) == set(TRANSFORMS), kinds
    for name, count in kinds.items():
        assert 0.12 <= count / len(sweep) <= 0.30, (name, count)
    coincidental = 0
    free = 0
    for ex in sweep:
        meta = ex["metadata"]
        square = meta["grid_h"] == meta["grid_w"]
        if meta["transform"] in SQUARE_ONLY:
            assert square, meta
            assert meta["requires_square"]
        else:
            assert not meta["requires_square"]
            free += 1
            coincidental += square
    # the two diagonal mirrors need H == W; nothing else does, so the other
    # three transforms keep the ~4% coincidence rate of independent sides
    assert coincidental / free <= 0.09, coincidental / free


def test_size_coverage(sweep):
    sides = Counter()
    big = 0
    for ex in sweep:
        meta = ex["metadata"]
        h, w = meta["grid_h"], meta["grid_w"]
        assert 5 <= h <= 30 and 5 <= w <= 30, meta
        assert h * w <= 900
        sides[h] += 1
        sides[w] += 1
        big += h * w >= 700
    assert min(sides) == 5 and max(sides) == 30
    assert len(sides) == 26, sorted(sides)
    counts = [sides[s] for s in range(5, 31)]
    assert max(counts) <= 3.0 * min(counts), counts     # near-flat, 700-seed noise
    assert 0.04 <= big / len(sweep) <= 0.16, big / len(sweep)


def test_color_coverage(sweep):
    hist = Counter(ex["metadata"]["n_colors"] for ex in sweep)
    assert set(hist) <= set(range(2, 11)), hist
    assert min(hist) == 2 and max(hist) == 10, hist
    rich = sum(v for k, v in hist.items() if k >= 8) / len(sweep)
    assert rich >= 0.12, rich
    assert hist[2] / len(sweep) >= 0.03, hist
    # colour 0 is an ordinary colour, in every role
    assert any(ex["metadata"]["ground_color"] == 0 for ex in sweep)
    assert any(ex["metadata"]["target_color"] == 0 for ex in sweep)
    assert any(ex["metadata"]["replacement_color"] == 0 for ex in sweep)


def test_texture_and_occupancy(sweep):
    textured = [ex for ex in sweep if ex["metadata"]["texture_kind"] != "none"]
    share = len(textured) / len(sweep)
    assert 0.12 <= share <= 0.35, share
    groups = Counter(ex["metadata"]["texture_group"] for ex in textured)
    assert set(groups) == {"sparse", "solid"}, groups
    for ex in textured:
        meta = ex["metadata"]
        assert min(meta["grid_h"], meta["grid_w"]) >= 8, meta
        assert meta["texture_named"] is False
        assert meta["target_color"] not in meta["texture_colors"]
        assert meta["ground_color"] not in meta["texture_colors"]
    bands = {"small": [], "large": []}
    for ex in sweep:
        meta = ex["metadata"]
        area = meta["grid_h"] * meta["grid_w"]
        bands["large" if area >= 500 else "small"].append(meta["occupancy"])
    for name, values in bands.items():
        assert len(values) >= 40, name
        assert sum(values) / len(values) >= 0.08, (name, sum(values) / len(values))


# ---------------------------------------------------------------------------
# what the answer has to decide
# ---------------------------------------------------------------------------

def test_violations_are_bodies_not_dots(sweep):
    for ex in sweep:
        meta = ex["metadata"]
        rows_in, rows_out = ex["input"], ex["output"]
        changed = [(r, c) for r in range(meta["grid_h"]) for c in range(meta["grid_w"])
                   if rows_in[r][c] != rows_out[r][c]]
        assert len(changed) == meta["answer_changed_cells"] == meta["n_violations"]
        for r, c in changed:
            assert rows_in[r][c] == meta["target_color"]
            assert rows_out[r][c] == meta["replacement_color"]
        if changed:
            assert meta["min_violation_body_cells"] >= 2, meta
    counts = Counter(ex["metadata"]["n_violation_bodies"] for ex in sweep)
    assert set(counts) >= {0, 1, 2, 3}, counts
    sizes = [ex["metadata"]["n_violations"] for ex in sweep]
    assert max(sizes) >= 10, max(sizes)
    assert len({s for s in sizes}) >= 8, sorted(set(sizes))


def test_identity_share(sweep):
    identity = [ex for ex in sweep if ex["metadata"]["identity_output"]]
    share = len(identity) / len(sweep)
    assert 0.04 <= share <= 0.15, share
    for ex in identity:
        assert ex["input"] == ex["output"]
        assert ex["metadata"]["n_violations"] == 0


def test_the_coloured_partner_clause_is_load_bearing(sweep):
    """"If the mirrored partner is also that color" must say more than "if the
    partner is empty": a violation facing a drawn cell of another colour is what
    separates the two readings, and it must be common without being universal."""
    with_coloured = sum(1 for ex in sweep
                        if ex["metadata"]["n_coloured_partner_violations"] > 0)
    share = with_coloured / len(sweep)
    assert 0.55 <= share <= 0.95, share
    non_identity = [ex for ex in sweep if not ex["metadata"]["identity_output"]]
    without = sum(1 for ex in non_identity
                  if ex["metadata"]["n_coloured_partner_violations"] == 0)
    assert without / len(non_identity) >= 0.10, without / len(non_identity)
    # the alternative reading ("erase a target cell whose partner is empty")
    # really does give another answer on those grids
    changed = 0
    for ex in sweep:
        meta = ex["metadata"]
        if meta["n_coloured_partner_violations"]:
            changed += 1
    assert changed / len(sweep) >= 0.55


def test_only_the_target_colour_is_the_rule(sweep):
    """Other colours break the symmetry too, so "look only at cells of color X"
    is not a free sentence: recolouring every asymmetric cell is wrong."""
    share = sum(1 for ex in sweep
                if ex["metadata"]["other_color_asymmetric_cells"] > 0) / len(sweep)
    assert share >= 0.55, share


def test_the_answer_restores_the_symmetry(sweep):
    for ex in sweep:
        meta = ex["metadata"]
        rule = _rule_from_prompt(get_prompt("101", meta))
        h, w = meta["grid_h"], meta["grid_w"]
        target = meta["target_color"]
        for r in range(h):
            for c in range(w):
                if ex["output"][r][c] != target:
                    continue
                rr, cc = rule(r, c, h, w)
                assert ex["output"][rr][cc] == target, (r, c)
        for r in range(h):
            for c in range(w):
                if ex["input"][r][c] != target:
                    assert ex["output"][r][c] == ex["input"][r][c], (r, c)


def test_cells_that_map_to_themselves_occur_and_are_kept(sweep):
    share = sum(1 for ex in sweep if ex["metadata"]["n_axis_target_cells"] > 0) / len(sweep)
    assert share >= 0.15, share
    for ex in sweep:
        meta = ex["metadata"]
        if not meta["n_axis_target_cells"]:
            continue
        rule = _rule_from_prompt(get_prompt("101", meta))
        h, w = meta["grid_h"], meta["grid_w"]
        for r in range(h):
            for c in range(w):
                if rule(r, c, h, w) == (r, c) and ex["input"][r][c] == meta["target_color"]:
                    assert ex["output"][r][c] == meta["target_color"], (r, c)


def test_bodies_are_drawn_not_noise(sweep):
    tiny = 0
    total = 0
    for ex in sweep:
        meta = ex["metadata"]
        assert set(meta["body_families"]) <= LIBRARY_FAMILIES, meta["body_families"]
        assert meta["n_bodies"] >= 1
        assert meta["n_single_cell_components"] == 0, meta
        sizes = _components(ex["input"], {meta["target_color"]})
        assert len(sizes) == meta["n_target_components"], meta
        # each body shows up as at most its own copy plus the transform's image
        assert meta["n_target_components"] <= 2 * meta["n_bodies"] + 1, meta
        assert sum(sizes) / len(sizes) >= 2.5, (meta["transform"], sizes)
        tiny += sum(1 for s in sizes if s <= 2)
        total += len(sizes)
    assert total > 500
    assert tiny / total <= 0.12, tiny / total


def test_a_body_meets_its_own_image_on_a_real_share(sweep):
    share = sum(1 for ex in sweep
                if "axis" in ex["metadata"]["body_kinds"]) / len(sweep)
    assert share >= 0.15, share
    limbs = sum(1 for ex in sweep if "paired_with_limb" in ex["metadata"]["body_kinds"])
    lone = sum(1 for ex in sweep if "lone" in ex["metadata"]["body_kinds"])
    assert limbs / len(sweep) >= 0.15, limbs / len(sweep)
    assert lone / len(sweep) >= 0.15, lone / len(sweep)


def test_the_transform_is_not_readable_from_the_scene(sweep):
    """Beyond the square canvas the two diagonal mirrors need, the scene is drawn
    the same way for every transform."""
    by_transform = {}
    for ex in sweep:
        meta = ex["metadata"]
        by_transform.setdefault(meta["transform"], []).append(
            (meta["n_colors"], meta["n_target_cells"] / (meta["grid_h"] * meta["grid_w"])))
    means = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
             for k, v in by_transform.items()}
    colors = [m[0] for m in means.values()]
    occupancy = [m[1] for m in means.values()]
    assert max(colors) - min(colors) <= 1.2, means
    assert max(occupancy) - min(occupancy) <= 0.08, means


# ---------------------------------------------------------------------------
# prompt policy
# ---------------------------------------------------------------------------

def test_prompt_policy(sweep):
    wordings = {}
    for ex in sweep[:400]:
        meta = ex["metadata"]
        prompt = get_prompt("101", meta)
        assert "Rows and columns are 0-indexed from the top-left corner." in prompt
        assert "check its mirrored partner in the input" in prompt
        assert "Leave every other cell unchanged." in prompt
        assert "Return the full grid." in prompt
        assert "Look only at cells of color" in prompt
        assert "is its own mirrored partner" in prompt
        assert "zero-indexed" not in prompt.lower()
        # the ground and the texture are never named or defined
        assert "background" not in prompt.lower()
        assert "most frequent" not in prompt.lower()
        generic = prompt.replace(f"color {meta['target_color']}", "color X") \
                        .replace(f"color {meta['replacement_color']}", "color R")
        wordings.setdefault(meta["transform"], set()).add(generic)
    assert set(wordings) == set(TRANSFORMS)
    for name, texts in wordings.items():
        assert len(texts) == 1, (name, texts)
    assert "anti-diagonal" in next(iter(wordings["anti_diagonal"]))
    assert "half turn" in next(iter(wordings["rotation_180"]))


LETTERS = list("ABCDEFGHIJ")


def test_only_the_two_given_colors_are_named():
    for seed in range(60):
        ex = generate("101", seed)
        meta = ex["metadata"]
        prompt = get_prompt("101", meta, color_palette=LETTERS)
        # the article "A" opens a sentence, so only "color <symbol>" counts
        named = set(re.findall(r"color ([A-J])\b", prompt))
        assert named == {LETTERS[meta["target_color"]], LETTERS[meta["replacement_color"]]}, \
            (seed, named)
        assert not re.search(r"\bcolor [0-9]\b", prompt), prompt
        symbol = get_prompt("101", meta, color_palette=palette_for_example("101", seed))
        assert not re.search(r"\bcolor [0-9]\b", symbol), symbol


def test_the_replacement_colour_is_never_the_ground(sweep):
    for ex in sweep:
        meta = ex["metadata"]
        assert meta["replacement_color"] != meta["ground_color"]
        assert meta["replacement_color"] != meta["target_color"]
    reserved = sum(1 for ex in sweep if ex["metadata"]["replacement_reserved"])
    assert 0.2 <= reserved / len(sweep) <= 0.9, reserved / len(sweep)


# ---------------------------------------------------------------------------
# the validator earns its place
# ---------------------------------------------------------------------------

def test_validator_raises_on_a_tampered_answer():
    ex = next(generate("101", seed) for seed in range(40)
              if generate("101", seed)["metadata"]["n_violations"] > 0)
    meta = ex["metadata"]

    bad = json.loads(json.dumps(ex))
    for r, row in enumerate(bad["input"]):
        for c, value in enumerate(row):
            if value == meta["target_color"]:
                bad["output"][r][c] = (value + 1) % 10
                break
        else:
            continue
        break
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["metadata"]["n_violations"] += 1
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["output"] = [list(row) for row in bad["input"]]
    with pytest.raises(ValueError):
        module.validate_example(bad)

    bad = json.loads(json.dumps(ex))
    bad["metadata"]["replacement_color"] = bad["metadata"]["ground_color"]
    with pytest.raises(ValueError):
        module.validate_example(bad)

    module.validate_example(json.loads(json.dumps(ex)))


def test_the_two_implementations_are_independent():
    """The validator's second reading is a whole-array flip, not the same loop."""
    for seed in range(60):
        ex = generate("101", seed)
        meta = ex["metadata"]
        args = (ex["input"], meta["transform"], meta["target_color"], meta["replacement_color"])
        assert module.solve(*args) == ex["output"], seed
        assert module.literal_solve(*args) == ex["output"], seed


def test_transform_rejects_a_canvas_it_cannot_map():
    for name in sorted(SQUARE_ONLY):
        with pytest.raises(ValueError):
            module.literal_solve([[0, 1, 2], [3, 4, 5]], name, 0, 1)


def test_generation_is_cheap():
    start = time.perf_counter()
    for seed in range(200):
        generate("101", seed)
    per_seed = (time.perf_counter() - start) / 200
    assert per_seed < 0.09, per_seed
