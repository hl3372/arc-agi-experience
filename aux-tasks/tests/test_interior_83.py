"""Literal boundary readings, scaled drawings, and independent rule selection."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from common.backgrounds import SPARSE_KINDS, texture_cells_isolated
from generators import interior_83 as module
from prompts import get_prompt
from registry import generate


def read_interior(grid, prompt):
    """Own component-and-exposed-boundary reader, using the displayed prompt.

    Returns ``(output, bodies, interiors, dots)``.  A dot is a one-cell group:
    the reader gives it no special treatment, and its single cell is always
    exposed, so the fill never reaches it.  ``bodies`` and ``interiors`` list the
    groups of two cells or more, which are the objects the plan drew.
    """
    token = re.search(r"interior cells with color (\S+)\.", prompt).group(1)
    fill = token if isinstance(grid[0][0], str) else int(token)
    assert "original input" in prompt and "same-color group connected through shared edges" in prompt
    assert "the area beyond the grid also counts as outside" in prompt
    corners = "through a shared edge or corner" in prompt
    h, w = len(grid), len(grid[0])
    ring = [grid[r][c] for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    ranked = Counter(ring).most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    background = ranked[0][0]
    remaining = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != background}
    result = [list(row) for row in grid]
    objects, interiors, dots = [], [], []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        color = grid[point[0]][point[1]]
        body, stack = {point}, [point]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining and grid[q[0]][q[1]] == color:
                    remaining.remove(q)
                    body.add(q)
                    stack.append(q)
        exposed = set()
        for r, c in body:
            neighbors = {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            if corners:
                neighbors |= {(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)}
            if neighbors - body:
                exposed.add((r, c))
        inside = body - exposed
        for r, c in inside:
            result[r][c] = fill
        if len(body) == 1:
            dots.append(next(iter(body)))
            assert not inside
            continue
        objects.append(body)
        interiors.append(inside)
    return result, objects, interiors, dots


@pytest.fixture(scope="module")
def examples():
    return [generate("83", seed) for seed in range(1200)]


def test_independent_actual_prompt_reader(examples):
    for ex in examples[:400]:
        assert read_interior(ex["input"], get_prompt("83", ex["metadata"]))[0] == ex["output"]


def test_shuffled_symbols(examples):
    for seed, ex in enumerate(examples[:300]):
        palette = np.random.default_rng(seed + 381).permutation(list("QWERTYUIOP")).tolist()
        grid = [[palette[c] for c in row] for row in ex["input"]]
        expected = [[palette[c] for c in row] for row in ex["output"]]
        prompt = get_prompt("83", ex["metadata"], color_palette=palette)
        assert read_interior(grid, prompt)[0] == expected


def test_sizes_counts_and_scaled_bodies(examples):
    dimensions = np.array([np.shape(ex["input"]) for ex in examples])
    assert set(dimensions[:, 0]) == set(dimensions[:, 1]) == set(range(5, 31))
    assert len(set(map(tuple, dimensions))) > 530
    assert abs(np.corrcoef(dimensions.T)[0, 1]) < .10
    assert .02 < np.mean(dimensions[:, 0] == dimensions[:, 1]) < .06
    assert .025 < np.mean(dimensions.prod(axis=1) >= 700) < .075
    counts = Counter(ex["metadata"]["n_objects"] for ex in examples)
    assert set(range(1, 11)) <= set(counts)
    assert .20 < counts[1] / len(examples) < .45
    assert max(max(ex["metadata"]["object_sizes"]) for ex in examples) > 250
    large = [ex["metadata"] for ex in examples if len(ex["input"]) * len(ex["input"][0]) >= 700]
    assert np.mean([m["occupancy"] for m in large]) > .23
    assert np.mean([max(m["object_sizes"]) for m in large]) > 80


def test_families_palette_and_thin_objects(examples):
    families = Counter(kind for ex in examples for kind in ex["metadata"]["object_kinds"])
    assert set(module.FAMILIES) | {"bar"} == set(families)
    assert families["rectangle"] / sum(families.values()) < .75
    assert all(families[family] >= 5 for family in module.FAMILIES)
    palette = Counter(ex["metadata"]["n_colors"] for ex in examples)
    # the input can never show all ten colors: the fill color is absent from it by
    # construction, so nine is the ceiling and the output is the histogram plus one
    assert set(range(2, 10)) <= set(palette)
    assert .18 < palette[2] / len(examples) < .40
    assert sum(palette[n] for n in (5, 6, 7, 8, 9)) > .18 * len(examples)
    assert sum(palette[n] for n in (7, 8, 9)) > .03 * len(examples)
    assert np.mean([ex["metadata"]["n_colors"] for ex in examples]) > 3.3
    assert .04 < np.mean([ex["metadata"]["unchanged_objects"] > 0 for ex in examples]) < .22
    for ex in examples:
        m = ex["metadata"]
        assert m["background_style"] in ("plain",) + SPARSE_KINDS
        assert m["fill_color"] not in np.unique(ex["input"])
        assert m["fill_color"] not in m["texture_colors"]
        assert len(np.unique(ex["output"])) == m["n_colors"] + 1


def test_rule_is_meaningful_and_balanced(examples):
    rules = Counter(ex["metadata"]["connectivity"] for ex in examples)
    assert .45 < rules[4] / len(examples) < .55
    assert .40 < np.mean([ex["metadata"]["connectivity_matters"] for ex in examples]) < .90
    for ex in examples[:400]:
        answers = []
        for rule in (4, 8):
            meta = dict(ex["metadata"], connectivity=rule)
            answer, _, inside, _ = read_interior(ex["input"], get_prompt("83", meta))
            answers.append(answer)
            assert sorted(map(len, inside)) == sorted(meta[f"interior_areas_{rule}"])
            assert answer != ex["input"]
        assert ex["metadata"]["connectivity_matters"] == (answers[0] != answers[1])


def test_completed_scene_is_independent_of_rule():
    class RuleRng:
        def __init__(self, seed, rule):
            self.base, self.rule = np.random.default_rng(seed), rule

        def __getattr__(self, name):
            return getattr(self.base, name)

        def choice(self, values, *args, **kwargs):
            if isinstance(values, tuple) and values == (4, 8):
                return self.rule
            return self.base.choice(values, *args, **kwargs)

    differing = 0
    for seed in range(150):
        four = module.generate_fill_leaving_shell(RuleRng(seed, 4))
        eight = module.generate_fill_leaving_shell(RuleRng(seed, 8))
        assert four["input"] == eight["input"]
        assert four["metadata"]["fill_color"] == eight["metadata"]["fill_color"]
        differing += four["output"] != eight["output"]
    assert differing > 60


def test_visible_metadata_and_preserved_plan(examples):
    for ex in examples[:400]:
        m = ex["metadata"]
        _, bodies, insides, dots = read_interior(ex["input"], get_prompt("83", m))
        assert m["n_objects"] == m["planned_object_count"] == len(bodies)
        assert sorted(m["object_sizes"]) == sorted(map(len, bodies))
        assert m["foreground_cells"] == sum(map(len, bodies))
        assert m["unchanged_objects"] == sum(not cells for cells in insides)
        assert m["attempts"] == 1 and m["relocation_rounds"] == 4
        assert set(m["planned_palette"]) | set(m["texture_colors"]) == set(np.unique(ex["input"]))
        assert len(dots) == m["texture_cells"]
        assert m["shell_thickness"] == 1


def test_corner_exposure_and_original_input_semantics():
    grid = np.zeros((9, 9), dtype=int)
    grid[2:7, 2:7] = 1
    grid[2, 2] = 0
    four = module.literal_solution(grid, 7, 4)
    eight = module.literal_solution(grid, 7, 8)
    assert four[3][3] == 7 and eight[3][3] == 1
    # Adjacent interior cells all fill; an earlier recolor never blocks the next.
    assert four[4][3:6] == [7, 7, 7]
    assert eight[4][3:6] == [7, 7, 7]
    edge = np.zeros((7, 8), dtype=int)
    edge[:3, 2:5] = 1
    for rule in (4, 8):
        answer = module.literal_solution(edge, 7, rule)
        assert answer[0][3] == 1 and answer[1][3] == 7


def test_independent_validator_rejects_tampering(examples, monkeypatch):
    ex = deepcopy(examples[0])
    ex["output"][0][0] = ex["metadata"]["fill_color"]
    with pytest.raises(ValueError, match="literal exterior-contact"):
        module.validate_example(ex)
    ex = deepcopy(examples[0])
    ex["metadata"]["planned_object_count"] += 1
    with pytest.raises(ValueError, match="object count"):
        module.validate_example(ex)
    original = module._inner
    def broken(mask, rule):
        value = original(mask, rule)
        if mask.shape[0] >= 20:
            return np.zeros_like(value)
        return value
    monkeypatch.setattr(module, "_inner", broken)
    with pytest.raises(ValueError, match="literal exterior-contact|interior-area"):
        module.generate_fill_leaving_shell(np.random.default_rng(0))


def test_prompts_keep_the_rule_simple_and_background_unnamed(examples):
    wordings = set()
    for ex in examples:
        prompt = get_prompt("83", ex["metadata"])
        assert "shell" not in prompt and "thickness" not in prompt
        assert "background color" not in prompt and "most frequent" not in prompt
        assert "Leave every other cell unchanged." in prompt
        assert ("shared edge or corner" in prompt) == (ex["metadata"]["connectivity"] == 8)
        # the house sentence, and nothing about the texture: no kind, no dots, no
        # isolated-cell rule, no hint of which grids carry one
        assert prompt.startswith("Ignore the background, whether it is a solid color or a pattern.")
        for word in ("texture", "dot", "isolated", "single cell", "speckle", "at least two cells"):
            assert word not in prompt
        wordings.add(prompt.replace(str(ex["metadata"]["fill_color"]), "F"))
    # one wording per rule variant, identical on plain and textured grids
    assert len(wordings) == 2


def test_sparse_textures_vary_the_ground_and_change_nothing(examples):
    """Every sparse kind appears, and a dot is an inert one-cell object."""
    eligible = [ex for ex in examples if min(np.shape(ex["input"])) >= 8]
    textured = [ex for ex in examples if ex["metadata"]["texture_kind"]]
    assert .22 < len(textured) / len(eligible) < .36
    kinds = Counter(ex["metadata"]["texture_kind"] for ex in textured)
    assert set(kinds) == set(SPARSE_KINDS) and min(kinds.values()) >= 10
    assert all(min(np.shape(ex["input"])) >= 8 for ex in textured)
    assert Counter(ex["metadata"]["background_style"] for ex in examples)["plain"] < .85 * len(examples)
    for ex in textured:
        m = ex["metadata"]
        grid, out = np.array(ex["input"]), np.array(ex["output"])
        colors = m["texture_colors"]
        cells = np.isin(grid, colors)
        assert m["texture_group"] == "sparse" and m["background_style"] == m["texture_kind"]
        assert m["texture_named"] is False
        assert 1 <= len(colors) <= 3 and len(set(colors)) == len(colors)
        assert m["background_color"] not in colors and m["fill_color"] not in colors
        assert not set(colors) & set(m["object_colors"])
        assert 4 <= m["texture_cells"] == int(cells.sum())
        assert 0 < m["texture_coverage"] <= .20
        # isolation: no two texture cells touch, and none touches an object
        assert texture_cells_isolated(grid, colors)
        objects = (grid != m["background_color"]) & ~cells
        assert not (module.halo(objects) & cells).any()
        # the ground still plainly owns the border ring the objects never reach
        share, rival = module.ring_shares(grid, m["background_color"])
        assert share >= module.RING_GROUND and rival <= module.RING_RIVAL
        # inert: the dots survive the edit, and erasing them changes no other cell
        assert (out[cells] == grid[cells]).all()
        plain = np.where(cells, m["background_color"], grid)
        expected = np.where(cells, m["background_color"], out)
        assert read_interior(plain.tolist(), get_prompt("83", m))[0] == expected.tolist()


def test_connectivity_is_not_hinted_by_the_scene(examples):
    """The rule is drawn after the picture, textures included; readers stay at chance."""
    metas = [ex["metadata"] for ex in examples]
    y = np.array([m["connectivity"] == 8 for m in metas])
    tex = np.array([bool(m["texture_kind"]) for m in metas])
    assert abs(y[tex].mean() - y[~tex].mean()) < .10
    for kind in SPARSE_KINDS:
        sel = np.array([m["texture_kind"] == kind for m in metas])
        assert .20 < y[sel].mean() < .80
    feats = np.array([[m["grid_h"], m["grid_w"], m["n_objects"], m["n_colors"], m["occupancy"],
                       m["foreground_cells"], float(bool(m["texture_kind"])),
                       m["texture_coverage"], m["texture_cells"]] for m in metas])
    half = len(feats) // 2
    best = .5
    for f in range(feats.shape[1]):
        for t in np.unique(np.quantile(feats[:half, f], np.linspace(.05, .95, 19))):
            pred = feats[half:, f] > t
            best = max(best, float((pred == y[half:]).mean()), float((~pred == y[half:]).mean()))
    assert best < .60


def test_validator_rejects_a_texture_against_an_object(examples):
    ex = deepcopy(next(e for e in examples if e["metadata"]["texture_kind"]))
    m = ex["metadata"]
    grid = np.array(ex["input"])
    cells = np.isin(grid, m["texture_colors"])
    objects = (grid != m["background_color"]) & ~cells
    touching = module.halo(objects) & ~objects & ~cells & ~module._ring(*grid.shape)
    source = np.argwhere(cells)[0]
    target = np.argwhere(touching)[0]
    grid[tuple(source)], grid[tuple(target)] = m["background_color"], m["texture_colors"][0]
    ex["input"] = grid.tolist()
    ex["output"] = module.literal_solution(grid, m["fill_color"], m["connectivity"])
    with pytest.raises(ValueError, match="isolated"):
        module.validate_example(ex)

    counted = deepcopy(next(e for e in examples if e["metadata"]["texture_kind"]))
    counted["metadata"]["texture_cells"] += 1
    with pytest.raises(ValueError, match="isolated single cells"):
        module.validate_example(counted)

    plain = deepcopy(next(e for e in examples if not e["metadata"]["texture_kind"]))
    plain["metadata"]["background_style"] = "random_dots"
    with pytest.raises(ValueError, match="background style"):
        module.validate_example(plain)
