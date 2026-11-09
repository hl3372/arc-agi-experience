"""Independent prompt-only oracles frozen during the pre-implementation review."""
import re
from collections import Counter
import copy
import numpy as np
import pytest

from registry import generate, list_tasks
from prompts import get_prompt, PROMPT_TEMPLATES
from generators import infer_translation_41 as motion
from generators import binary_logic_42 as binary
from generators import tiling_44_45 as tiling


def token(value):
    return int(value) if value.isdigit() else value


def clipped_translation_candidates(grids):
    """Independent point-dictionary reader, including stationary single cells."""
    a, b = np.asarray(grids['before']), np.asarray(grids['after'])
    if a.shape != b.shape:
        raise ValueError('Different canvas sizes')
    choices = {}
    h, w = a.shape
    for bg in np.unique(a):
        unseen = set(map(tuple, np.argwhere(a != bg)))
        movers, stationary = {}, {}
        while unseen:
            point = unseen.pop()
            todo, part = [point], {point}
            for r, c in todo:
                for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if q in unseen:
                        unseen.remove(q)
                        part.add(q)
                        todo.append(q)
            layer = movers if len(part) >= 2 else stationary
            layer.update({p: a[p].item() for p in part})
        if not movers:
            continue
        target = {tuple(p): b[tuple(p)].item() for p in np.argwhere(b != bg)}
        if target == stationary:
            raise ValueError('Infinitely many fully clipped offsets fit')
        changed = [(p, color) for p, color in target.items() if stationary.get(p) != color]
        if not changed:
            continue
        counts = Counter(movers.values())
        (rr, cc), color = min(changed, key=lambda item: counts[item[1]])
        for (r, c), source_color in movers.items():
            if source_color != color:
                continue
            dr, dc = rr-r, cc-c
            out = stationary.copy()
            out.update({(y+dr, x+dc): value for (y, x), value in movers.items()
                        if 0 <= y+dr < h and 0 <= x+dc < w})
            if out == target:
                choices.setdefault((int(dr), int(dc)), []).append(bg.item())
    return choices


def parse_binary(grid, prompt):
    a = np.asarray(grid)
    axis = 1 if 'side by side' in prompt else 0 if 'stacked vertically' in prompt else None
    if axis is None or a.shape[axis] % 2 != 1:
        raise ValueError('Cannot split equal binary subgrids')
    mid = a.shape[axis] // 2
    first, last = (a[:, :mid], a[:, mid + 1:]) if axis else (a[:mid], a[mid + 1:])
    true_colors = [token(v) for v in re.findall(r'color (\S+) means true', prompt)]
    false_color = token(re.search(r'color (\S+) means false', prompt).group(1))
    if len(true_colors) == 1:
        true_colors *= 2
        out_true = true_colors[0]
    else:
        out_true = token(re.search(r'using color (\S+) for true cells', prompt).group(1))
    for cells, fg in zip((first, last), true_colors):
        if not set(np.unique(cells)).issubset({fg, false_color}):
            raise ValueError('Unspecified color in binary subgrid')
    op = re.search(r'Compute their (\w+)', prompt).group(1)
    return first == true_colors[0], last == true_colors[1], op, out_true, false_color, axis


TRUTH = {'AND': (0, 0, 0, 1), 'OR': (0, 1, 1, 1), 'XOR': (0, 1, 1, 0),
         'NAND': (1, 1, 1, 0), 'NOR': (1, 0, 0, 0), 'XNOR': (1, 0, 0, 1)}


def binary_result(a, b, op, true_color, false_color):
    truth = np.asarray(TRUTH[op], dtype=bool)
    result = truth[a.astype(int) * 2 + b.astype(int)]
    return np.where(result, true_color, false_color).tolist()


def tile_factors(prompt):
    values = []
    for axis in ('vertically', 'horizontally'):
        match = re.search(r'(once|\d+ times) ' + axis, prompt)
        values.append(1 if match.group(1) == 'once' else int(match.group(1).split()[0]))
    return tuple(values)


def minimum_tile(grid):
    a = np.asarray(grid)
    h, w = a.shape
    # An exact rectangular tile must repeat independently on both axes.
    hs = [p for p in range(1, h + 1) if h % p == 0 and np.array_equal(a, a[np.arange(h) % p, :])]
    ws = [p for p in range(1, w + 1) if w % p == 0 and np.array_equal(a, a[:, np.arange(w) % p])]
    return min(hs), min(ws), hs, ws


@pytest.fixture(scope="module")
def sweep():
    return {task: [generate(task, seed) for seed in range(2000)] for task in ("41", "42", "44", "45")}


def independent_answer(task, inp, prompt):
    if task == "41":
        candidates = clipped_translation_candidates(inp)
        assert len(candidates) == 1, candidates
        return "(%s,%s)" % next(iter(candidates))
    if task == "42":
        a, b, op, true, false, _ = parse_binary(inp, prompt)
        return binary_result(a, b, op, true, false)
    if task == "44":
        m, n = tile_factors(prompt)
        h, w = len(inp), len(inp[0])
        return [[inp[r % h][c % w] for c in range(w * n)] for r in range(h * m)]
    h, w, _, _ = minimum_tile(inp)
    return f"{len(inp) // h}x{len(inp[0]) // w}"


@pytest.mark.parametrize("task", ["41", "42", "44", "45"])
def test_independent_prompt_oracle(sweep, task):
    for ex in sweep[task][:400]:
        assert independent_answer(task, ex["input"], get_prompt(task, ex["metadata"])) == ex["output"]


@pytest.mark.parametrize("task", ["41", "42", "44", "45"])
def test_permuted_letter_palette(sweep, task):
    for seed, ex in enumerate(sweep[task][:60]):
        palette = np.random.default_rng(seed + 10207).permutation(list("ABCDEFGHIJ")).tolist()
        def remap(grid):
            if isinstance(grid, dict):
                return {name: remap(a) for name, a in grid.items()}
            return np.asarray(palette)[grid].tolist()
        inp = remap(ex["input"])
        want = remap(ex["output"]) if isinstance(ex["output"], list) else ex["output"]
        assert independent_answer(task, inp, get_prompt(task, ex["metadata"], color_palette=palette)) == want


def test_translation_coverage_and_prompt(sweep):
    examples = sweep["41"]
    grids = [np.asarray(ex["input"]["before"]) for ex in examples]
    meta = [ex["metadata"] for ex in examples]
    assert .20 < np.mean([m['texture_cells'] > 0 for m in meta]) < .40
    assert len({m['background_kind'] for m in meta}) == 7
    assert all(m['texture_cells'] <= .06 * a.size for m, a in zip(meta, grids))
    for axis in (0, 1):
        assert {a.shape[axis] for a in grids} == set(range(1, 31))
    assert {len(np.unique(a)) for a in grids} == set(range(2, 11))
    assert sum(len(np.unique(a)) >= 8 for a in grids) > 350
    assert sum(min(a.shape) == 1 for a in grids) > 60
    assert 50 < sum(m["dr"] == m["dc"] == 0 for m in meta) < 190
    assert sum(m["clipped_cells"] > 0 for m in meta) > 750
    assert len({(m["dr"], m["dc"]) for m in meta}) > 350
    assert max(abs(m["dr"]) for m in meta) > 20
    assert max(abs(m["dc"]) for m in meta) > 20
    assert len({k for m in meta for k in m["shape_families"]}) >= 9
    large = [(a, m) for a, m in zip(grids, meta) if a.size >= 700]
    assert len(large) > 50
    assert np.mean([m["foreground_cells"] / a.size for a, m in large]) > .13
    for m in meta:
        assert m["background_margin"] > 1.25
        assert m["surviving_cells"] >= 1
        p = get_prompt("41", m).lower()
        assert "background" not in p and "clipped" in p and "zero" in p
        assert "isolated single cells in before are decorations, whatever their color" in p
        assert "stay fixed unless covered by a moved shape" in p
        assert p.index("decorations") < p.index("one or more shapes")


def test_translation_rejects_wrong_and_ambiguous_answers(sweep):
    ex = sweep["41"][0]
    with pytest.raises(ValueError):
        motion.validate(ex["input"]["before"], ex["input"]["after"], (300, 300))
    # Either source domino can become the sole visible domino after clipping.
    a = [[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]]
    b = [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
    assert len(motion.matching_offsets(a, b)) == 2
    with pytest.raises(ValueError):
        motion.validate(a, b, (0, 4))
    with pytest.raises(ValueError):
        motion.validate(a, [[0] * 10], (0, 10))
    # A second color interpretation can disagree despite a clear plurality.
    with pytest.raises(ValueError):
        motion.validate([[0, 0, 0, 0, 0, 1, 1]], [[1, 1, 0, 0, 0, 0, 0]], (0, -5))
    for seed in (6653, 8543):
        ex = generate("41", seed)
        assert independent_answer("41", ex["input"], get_prompt("41", ex["metadata"])) == ex["output"]


def test_boolean_encoding_rule_and_scene_coverage(sweep):
    examples = sweep["42"]
    meta = [ex["metadata"] for ex in examples]
    assert set(m["encoding"] for m in meta) == {"shared", "different"}
    assert all(n > 220 for n in Counter(m["operation"] for m in meta).values())
    assert all(n > 850 for n in Counter(m["encoding"] for m in meta).values())
    assert set(f for m in meta for f in m["mask_families"]) == set(binary.FAMILIES)
    assert len({m["relationship"] for m in meta}) == 6
    assert sum(m["constant_output"] for m in meta) < 260
    assert sum(all(m["truth_case_counts"]) for m in meta) > 1300
    assert sum(np.asarray(e["input"]).size >= 700 for e in examples) > 35
    assert any(len(e["output"]) == 1 for e in examples)
    assert any(len(e["output"][0]) == 1 for e in examples)
    for axis in (0, 1):
        assert max(np.asarray(e["output"]).shape[axis] for e in examples) == 30


def test_boolean_rule_cannot_change_the_input(monkeypatch):
    for seed in range(100):
        inputs = []
        for op in binary.OPERATIONS:
            monkeypatch.setattr(binary, "choose_operation", lambda rng, op=op: op)
            ex = generate("42", seed)
            inputs.append(ex["input"])
            assert independent_answer("42", ex["input"], get_prompt("42", ex["metadata"])) == ex["output"]
        assert all(a == inputs[0] for a in inputs)


def test_boolean_validator_reads_displayed_halves(sweep):
    ex = copy.deepcopy(sweep["42"][0])
    grid = np.asarray(ex["input"])
    m = ex["metadata"]
    axis = 1 if m["layout"] == "horizontal" else 0
    if axis:
        grid[:, grid.shape[1] // 2] = m["bg_color"]
    else:
        grid[grid.shape[0] // 2] = m["bg_color"]
    with pytest.raises(ValueError):
        binary.validate(grid, ex["output"], m)
    ex["output"][0][0] = m["bg_color"] if ex["output"][0][0] == m["output_true_color"] else m["output_true_color"]
    with pytest.raises(ValueError):
        binary.validate(ex["input"], ex["output"], m)


def test_tiling_palette_geometry_and_repetition(sweep):
    for task in ("44", "45"):
        examples = sweep[task]
        assert {len(np.unique(e["input"])) for e in examples} >= set(range(2, 11))
        assert sum(len(np.unique(e["input"])) >= 8 for e in examples) > 330
        assert len({e["metadata"]["content_family"] for e in examples}) == 7
        assert all(max(np.asarray(e["input"]).shape) <= 30 for e in examples)
        assert all(e["metadata"]["attempts"] < 100 for e in examples)
    repeat = sweep["44"]
    assert 50 < sum(e["input"] == e["output"] for e in repeat) < 190
    assert sum(e["metadata"]["input_periodic"] for e in repeat) > 400
    for axis in (0, 1):
        assert max(np.asarray(e["input"]).shape[axis] for e in repeat) == 30
    assert max(e["metadata"]["tile_rows"] for e in repeat) == 30
    assert max(e["metadata"]["tile_cols"] for e in repeat) >= 29
    scale = [e["metadata"] for e in sweep["45"]]
    assert len({(m["M"], m["N"]) for m in scale}) > 100
    assert 70 < sum(min(m["tile_h"], m["tile_w"]) == 1 for m in scale) < 250
    assert np.median([m["tile_h"] * m["tile_w"] for m in scale]) >= 25
    assert sum(m["tile_h"] * m["tile_w"] >= 100 for m in scale) > 120
    assert max(m["M"] for m in scale) >= 25 and max(m["N"] for m in scale) >= 25


def test_whole_input_and_smallest_tile_are_distinct_rules():
    a = [[1, 2, 1, 2], [3, 4, 3, 4]]
    out = [row * 3 for row in a] * 2
    tiling.validate_repeat(a, out, (2, 3))
    tiling.validate_scale(a, (1, 2))
    with pytest.raises(ValueError):
        tiling.validate_repeat(a, [row[:2] * 3 for row in a] * 2, (2, 3))
    with pytest.raises(ValueError):
        tiling.validate_scale(a, (1, 1))
    assert tiling.smallest_tile([[5, 7, 7]] * 7) == (1, 3)


def test_retired_43_is_not_sampled_or_silently_aliased():
    assert "43" not in list_tasks() and "43" not in PROMPT_TEMPLATES
    with pytest.raises(KeyError, match="merged into task 42"):
        generate("43", 0)
