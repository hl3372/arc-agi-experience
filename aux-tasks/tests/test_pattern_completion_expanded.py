"""Independent graph-based equality proof for tasks 32 and 33."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import pattern_completion_32_33 as G
from prompts import get_prompt
from registry import generate


def _oracle(grid, prompt):
    a = np.asarray(grid).astype(str)
    h, w = a.shape
    mask_color = re.search(r"color (\S+)", prompt)[1]
    hidden = a == mask_color
    parent = list(range(h * w))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        parent[root(i)] = root(j)

    period = re.search(r"tile size (\d+) rows? by (\d+) columns?", prompt)
    for r in range(h):
        for c in range(w):
            targets = []
            if period:
                th, tw = int(period[1]), int(period[2])
                if r + th < h:
                    targets.append((r + th, c))
                if c + tw < w:
                    targets.append((r, c + tw))
            elif "all 8 square symmetries" in prompt:
                targets = [(c, r), (h - 1 - r, c)]
            elif "both horizontal and vertical" in prompt:
                targets = [(h - 1 - r, c), (r, w - 1 - c)]
            elif "anti-diagonal" in prompt:
                targets = [(w - 1 - c, h - 1 - r)]
            elif "main-diagonal" in prompt:
                targets = [(c, r)]
            elif "180-degree rotational" in prompt:
                targets = [(h - 1 - r, w - 1 - c)]
            elif "horizontal mirror" in prompt:
                targets = [(h - 1 - r, c)]
            elif "vertical mirror" in prompt:
                targets = [(r, w - 1 - c)]
            else:
                raise ValueError(prompt)
            for rr, cc in targets:
                join(r * w + c, rr * w + cc)
    evidence = {}
    for i, value in enumerate(a.ravel()):
        if value == mask_color:
            continue
        k = root(i)
        if k in evidence and evidence[k] != value:
            raise ValueError("Visible pattern is inconsistent")
        evidence[k] = value
    if not hidden.any() or any(root(i) not in evidence for i in range(h * w)):
        raise ValueError("Completion is not unique")
    answer = np.asarray([evidence[root(i)] for i in range(h * w)]).reshape(h, w)
    if "Output exactly the hidden rectangle patch" in prompt:
        yy, xx = np.where(hidden)
        r, c, b, d = int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1
        assert min(b - r, d - c) >= 2 and hidden[r:b, c:d].all()
        answer = answer[r:b, c:d]
    return answer


@pytest.fixture(scope="module")
def examples():
    return {task: [generate(task, seed) for seed in range(600)] for task in ("32", "33")}


def test_independent_completion_proof_and_symbol_permutations(examples):
    for task, rows in examples.items():
        for seed, ex in enumerate(rows):
            assert np.array_equal(_oracle(ex["input"], get_prompt(task, ex["metadata"])), np.asarray(ex["output"]).astype(str))
            if seed < 150:
                p = np.random.default_rng(seed + 48000).permutation(list("ABCDEFGHIJ")).tolist()
                prompt = get_prompt(task, ex["metadata"], color_palette=p)
                grid = np.asarray(p)[np.asarray(ex["input"])]
                assert np.array_equal(_oracle(grid, prompt), np.asarray(p)[np.asarray(ex["output"])])


def test_family_palette_content_and_mask_coverage(examples):
    for task, rows in examples.items():
        ms = [e["metadata"] for e in rows]
        family = Counter(m["pattern_family"] for m in ms)
        assert set(family) == set(G.FAMILIES)
        assert .25 < family["periodic"] / len(ms) < .4
        assert .015 < family["full_square_symmetry"] / len(ms) < .065
        assert set(m["n_colors"] for m in ms) == set(range(3, 11))
        assert sum(m["n_colors"] >= 8 for m in ms) / len(ms) > .25
        assert set(m["pattern_content"] for m in ms) == set(G.CONTENT)
        assert sum(m["pattern_content"] == "mosaic" for m in ms) < .14 * len(ms)
        sizes = np.asarray([(len(e["input"]), len(e["input"][0])) for e in rows])
        assert (sizes.min(), sizes.max()) == ((1 if task == "32" else 2), 30)
        assert np.mean([m["masked_fraction"] for m in ms]) > (.20 if task == "32" else .15)
        assert max(m["masked_cells"] for m in ms) >= 300
        assert sum(m["masked_fraction"] >= .4 for m in ms) > .08 * len(ms)
        if task == "33":
            assert max(len(e["output"]) for e in rows) > 10
            assert max(len(e["output"][0]) for e in rows) > 10
            assert any(len(e["output"]) == len(e["input"]) or len(e["output"][0]) == len(e["input"][0]) for e in rows)


def test_entirely_hidden_classes_and_inconsistent_evidence_raise():
    grid = [[1, 0, 1, 2], [1, 0, 1, 2]]
    with pytest.raises(ValueError, match="entire class"):
        G.reconstruct(grid, "horizontal_mirror", 0)
    grid = [[0, 1, 2, 1], [1, 2, 2, 1]]
    with pytest.raises(ValueError, match="contradict"):
        G.reconstruct(grid, "horizontal_mirror", 0)


def test_validator_uses_visible_bounds_and_rejects_tampering(examples):
    for task in ("32", "33"):
        ex = deepcopy(examples[task][5])
        ex["metadata"].update(occluder_height=100, occluder_width=100, masked_cells=-1)
        G.validate_example(ex, task)
        ex["output"][0][0] = (ex["output"][0][0] + 1) % 10
        with pytest.raises(ValueError, match="reconstruction"):
            G.validate_example(ex, task)


def test_invalid_validation_is_not_replaced_by_a_canned_example(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("deliberate independent-check failure")
    monkeypatch.setattr(G, "reconstruct", broken)
    for task in ("32", "33"):
        with pytest.raises(ValueError, match="independent-check failure"):
            generate(task, 10)
