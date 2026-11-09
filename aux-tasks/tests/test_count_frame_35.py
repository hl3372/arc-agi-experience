"""Unambiguous unnamed frames and broad, literal interior counts."""
from collections import Counter
from copy import deepcopy
import re

import numpy as np
import pytest

from generators import count_frame_35 as G
from prompts import get_prompt
from registry import generate
from tests.test_crop_frame_26 import _oracle_frames


def _oracle(grid, prompt):
    # The integral-image frame oracle is independent of the production
    # run/corner detector and checks every colour, including the query colour.
    a = np.asarray(grid)
    symbols = list(dict.fromkeys(a.ravel().tolist()))
    encoded = np.asarray([[symbols.index(v) for v in row] for row in grid])
    frames = _oracle_frames(encoded)
    if len(frames) != 1:
        raise ValueError("Ambiguous frame")
    r, c, h, w, _ = frames[0]
    q = re.search(r"color (\S+)", prompt)[1]
    return str(sum(str(grid[y][x]) == q for y in range(r + 1, r + h - 1) for x in range(c + 1, c + w - 1)))


@pytest.fixture(scope="module")
def examples():
    return [generate("35", seed) for seed in range(400)]


def test_independent_frame_and_count_oracle(examples):
    for seed, ex in enumerate(examples):
        assert _oracle(ex["input"], get_prompt("35", ex["metadata"])) == ex["output"], seed


def test_prior_nested_frame_ambiguity_is_rejected():
    grid = [[int(c) for c in row] for row in ("422222444444", "428882644340", "428982444444", "428882444444", "422222444144")]
    assert len(_oracle_frames(grid)) == 2
    with pytest.raises(ValueError, match="one visible frame"):
        G.solve_count_in_frame(grid, 8)
    with pytest.raises(ValueError, match="Ambiguous"):
        _oracle(grid, "How many color 8 cells are inside the marked rectangle?")


def test_answer_capacity_and_tiny_interiors():
    a = np.full((30, 30), 4, dtype=int)
    a[1:-1, 1:-1] = 7
    assert G.solve_count_in_frame(a, 7) == "784"
    assert generate("35", 79227)["output"] == "784"
    assert G.solve_count_in_frame(a, 3) == "0"
    assert G.solve_count_in_frame([[4, 4, 4], [4, 7, 4], [4, 4, 4]], 7) == "1"
    for seed in (2286, 13122):
        ex = generate("35", seed)
        assert _oracle(ex["input"], get_prompt("35", ex["metadata"])) == ex["output"]


def test_size_palette_counts_and_coherent_content():
    rows = [generate("35", seed) for seed in range(2000)]
    sizes = np.asarray([(len(e["input"]), len(e["input"][0])) for e in rows])
    assert set(sizes[:, 0]) == set(sizes[:, 1]) == set(range(3, 31))
    assert abs(np.corrcoef(sizes.T)[0, 1]) < .12
    palette = Counter(e["metadata"]["n_colors"] for e in rows)
    assert set(palette) == set(range(2, 11)) and max(palette.values()) < .18 * len(rows)
    answers = [int(e["output"]) for e in rows]
    assert .04 <= answers.count(0) / len(rows) <= .09
    assert max(answers) >= 600 and len(set(answers)) > 180
    assert sum(x > 80 for x in answers) / len(rows) > .09
    assert sum(e["metadata"]["outside_query_count"] > 0 for e in rows) / len(rows) > .55
    assert len({e["metadata"]["content_family"] for e in rows}) == 12
    assert max(e["metadata"]["frame_bbox"][2] for e in rows) == 30
    assert max(e["metadata"]["frame_bbox"][3] for e in rows) == 30
    observed = expected = n = 0
    for ex in rows:
        if ex["output"] == "0":
            continue
        r, c, h, w = ex["metadata"]["frame_bbox"]
        counts = Counter(ex["input"][y][x] for y in range(r + 1, r + h - 1) for x in range(c + 1, c + w - 1))
        maximum = max(counts.values())
        observed += int(ex["output"]) == maximum
        expected += sum(v == maximum for v in counts.values()) / len(counts)
        n += 1
    assert abs(observed - expected) / n < .035


def test_symbol_remap_prompt_and_tampered_output(examples):
    for seed, ex in enumerate(examples[:80]):
        p = np.random.default_rng(seed + 92000).permutation(list("ABCDEFGHIJ")).tolist()
        inp = np.asarray(p)[np.asarray(ex["input"])].tolist()
        assert _oracle(inp, get_prompt("35", ex["metadata"], color_palette=p)) == ex["output"]
    prompt = get_prompt("35", examples[0]["metadata"])
    assert prompt.count("color ") == 1  # only the query is named
    bad = deepcopy(examples[0])
    bad["output"] = str(int(bad["output"]) + 1)
    with pytest.raises(ValueError, match="reconstruction"):
        G.validate_example(bad)
