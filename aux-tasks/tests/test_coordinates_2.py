"""Tests for task 2 (``generators/metadata.py``): coordinates of one colour.

Independent oracle: the answer must equal the row-major list of every cell of
the query colour in the displayed grid (parsed back from the answer string),
capped at 60, never truncated.  Also covered: size envelope (1..30 each side,
30x30 reachable), answer-length spread across the three bands, composition
and background mixes, the target colour never used by the background, the
10-colour palette reach, determinism, prompt contents and performance.
"""
from __future__ import annotations

import os
import re
import statistics
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators.metadata import TASK2_MAX_CELLS  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
LETTERS = list("ABCDEFGHIJ")
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("2", s) for s in range(SWEEP))
    return _CACHE


def parse(answer: str) -> list[tuple[int, int]]:
    assert answer.startswith("[") and answer.endswith("]")
    return [(int(r), int(c)) for r, c in re.findall(r"\((\d+),(\d+)\)", answer)]


def test_answer_is_exact_row_major_recount():
    for s, ex in enumerate(examples()):
        grid = np.array(ex["input"])
        target = ex["metadata"]["query_color_index"]
        coords = parse(ex["output"])
        expected = sorted((int(r), int(c)) for r, c in np.argwhere(grid == target))
        assert coords == expected, f"seed {s}"
        assert 1 <= len(coords) <= TASK2_MAX_CELLS == 60
        assert len(coords) == ex["metadata"]["n_cells"]
        assert ex["output"] == "[" + ",".join(f"({r},{c})" for r, c in coords) + "]"


def test_sizes_and_determinism():
    exs = examples()
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert hs.min() == 1 and ws.min() == 1 and hs.max() == 30 and ws.max() == 30
    assert (hs * ws).max() == 900
    assert len(set(hs.tolist())) == 30 and len(set(ws.tolist())) == 30
    assert np.mean(hs == ws) <= 0.08
    for s in range(200):
        assert generate("2", s) == exs[s]


def test_answer_length_bands():
    lengths = Counter(ex["metadata"]["n_cells"] for ex in examples())
    total = sum(lengths.values())
    short = sum(v for k, v in lengths.items() if k <= 8) / total
    mid = sum(v for k, v in lengths.items() if 9 <= k <= 30) / total
    long_ = sum(v for k, v in lengths.items() if k >= 31) / total
    assert short >= 0.35 and mid >= 0.20 and long_ >= 0.08, (short, mid, long_)
    assert max(lengths.values()) / total <= 0.15, lengths.most_common(3)
    assert lengths[60] > 0


def test_compositions_and_backgrounds():
    exs = examples()
    comp = Counter(ex["metadata"]["composition"] for ex in exs)
    for k in ("dots", "shapes", "big", "mixed"):
        assert comp[k] / len(exs) >= 0.10, comp
    bg = Counter(ex["metadata"]["background_style"] for ex in exs)
    assert bg["uniform"] >= 0.35 * len(exs) and bg["noisy"] >= 0.12 * len(exs) and bg["striped"] >= 0.10 * len(exs), bg
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        assert m["query_color_index"] != m["background"]


def test_target_pieces_do_not_touch_each_other():
    """Pieces of the target colour are placed one cell apart, so 'dots' really are isolated."""
    for s, ex in enumerate(examples()):
        m = ex["metadata"]
        if m["composition"] != "dots":
            continue
        grid = np.array(ex["input"])
        t = grid == m["query_color_index"]
        H, W = t.shape
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = t[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
            b = t[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)]
            assert not np.any(a & b), f"seed {s}"


def test_palette_reaches_ten_colours():
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in examples())
    assert max(seen) == 10, seen
    assert seen[2] + seen[3] >= 0.15 * sum(seen.values()), seen


def test_prompt_contents():
    for ex in examples()[:500]:
        m = ex["metadata"]
        prompt = get_prompt("2", m, color_palette=LETTERS)
        assert f"color {LETTERS[m['query_color_index']]}" in prompt
        assert "0-indexed" in prompt and "row-major" in prompt
        assert not re.search(r"color \d", prompt), prompt


def test_performance():
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate("2", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 5.0, statistics.mean(times)
    assert np.percentile(times, 99) <= 40.0
