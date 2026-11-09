"""Tests for task 9 (``generators/local_transforms.py``): infer the rotation between two images.

Independent oracle: split the visible combined grid at the divider (the unique
full row / column of a colour present nowhere else), check that exactly one
rotation in {90, 180, 270} maps the first image to the second, and that the
stated answer is that rotation in the stated direction.  Also covered: size
envelope (free dimension reaches 30, doubled dimension <= 14, even sides above
14 present, near-flat density, squares only by coincidence), content-family
coverage, colour distribution (10 reachable, 8-10 a fifth to two fifths),
divider colour absent from both halves, texture metadata, balance of answers /
directions / layouts, determinism, prompt contents under a letter palette, and
performance.
"""
from __future__ import annotations

import os
import re
import statistics
import sys
import time
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backgrounds import KIND_GROUP, texture_cells_isolated  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
LETTERS = list("ABCDEFGHIJ")
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("9", s) for s in range(SWEEP))
    return _CACHE


def _rot(a: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(a, -k)  # k quarter turns clockwise


def split_at_divider(grid: np.ndarray, layout: str):
    H, W = grid.shape
    n = H if layout == "vertical" else W
    found = []
    for i in range(n):
        line = grid[i, :] if layout == "vertical" else grid[:, i]
        if np.all(line == line[0]) and int((grid == line[0]).sum()) == len(line):
            found.append(i)
    assert len(found) == 1, found
    i = found[0]
    if layout == "vertical":
        return grid[:i, :], grid[i + 1:, :], int(grid[i, 0])
    return grid[:, :i], grid[:, i + 1:], int(grid[0, i])


def test_answer_is_the_unique_rotation_in_the_stated_direction():
    for s, ex in enumerate(examples()):
        m = ex["metadata"]
        g = np.array(ex["input"])
        a, b, div = split_at_divider(g, m["layout"])
        assert a.shape == (m["half_h"], m["half_w"]), s
        angles = [k * 90 for k in (1, 2, 3) if _rot(a, k).shape == b.shape and np.array_equal(_rot(a, k), b)]
        assert len(angles) == 1, (s, angles)
        cw = angles[0]
        expected = cw if m["rotation_direction"] == "clockwise" else (360 - cw) % 360 or 180
        assert int(ex["output"]) == expected == m["angle"], s
        assert m["cw_angle"] == cw
        assert not np.array_equal(a, b), s
        assert div == m["divider_color"] and div not in set(np.unique(a).tolist()), s


def test_size_envelope():
    exs = examples()
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert hs.max() == 30 and ws.max() == 30
    assert hs.min() == 3 and ws.min() == 3
    for ex in exs:
        m = ex["metadata"]
        doubled = m["half_w"] if m["layout"] == "horizontal" else m["half_h"]
        free = m["half_h"] if m["layout"] == "horizontal" else m["half_w"]
        assert 3 <= doubled <= 14 and 3 <= free <= 30
        if m["cw_angle"] in (90, 270):
            assert m["half_h"] == m["half_w"], "90/270 halves must be square (layout)"
        else:
            assert free >= 3
    even_big = np.mean(((hs > 14) & (hs % 2 == 0)) | ((ws > 14) & (ws % 2 == 0)))
    assert even_big >= 0.06, even_big
    assert np.mean(hs == ws) <= 0.08
    assert len({(int(h), int(w)) for h, w in zip(hs, ws)}) >= 300
    # 180-degree halves are rectangular often; 90/270 halves are never rectangular
    rect = [ex for ex in exs if ex["metadata"]["cw_angle"] == 180 and ex["metadata"]["half_h"] != ex["metadata"]["half_w"]]
    assert len(rect) >= 0.8 * sum(1 for ex in exs if ex["metadata"]["cw_angle"] == 180)
    # near-flat over the doubled side 3..14
    doubled_sides = Counter(ex["metadata"]["half_w"] if ex["metadata"]["layout"] == "horizontal" else ex["metadata"]["half_h"] for ex in exs)
    counts = np.array([doubled_sides[v] for v in range(3, 15)], dtype=float)
    assert counts.min() >= 0.5 * counts.max(), doubled_sides


def test_content_families_and_textures():
    exs = examples()
    fam = Counter(ex["metadata"]["content_family"] for ex in exs)
    for f in ("scene", "single", "dots", "mosaic", "structured"):
        assert fam[f] / len(exs) >= 0.05, fam
    pats = Counter(ex["metadata"].get("pattern") for ex in exs if ex["metadata"]["content_family"] == "structured")
    assert len(pats) == 5, pats
    textured = 0
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        if m["texture_group"] is None:
            assert m["texture_kind"] is None and m["texture_colors"] == []
            continue
        textured += 1
        assert m["content_family"] == "scene"
        assert KIND_GROUP[m["texture_kind"]] == m["texture_group"]
        assert min(m["half_h"], m["half_w"]) >= 8
        g = np.array(ex["input"])
        a, _b, div = split_at_divider(g, m["layout"])
        assert div not in m["texture_colors"]
        if m["texture_group"] == "sparse":
            assert texture_cells_isolated(a, m["texture_colors"]), s
    assert textured / len(exs) >= 0.05, textured


def test_colours_reach_ten_without_a_dominant_bin():
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in examples())
    total = sum(seen.values())
    assert max(seen) == 10, seen
    assert 0.15 <= sum(v for k, v in seen.items() if k >= 8) / total <= 0.40, seen
    assert max(seen.values()) / total <= 0.35, seen
    for ex in examples():
        assert ex["metadata"]["n_colors"] <= 9


def test_balance_and_determinism():
    exs = examples()
    ans = Counter(ex["metadata"]["angle"] for ex in exs)
    assert set(ans) == {90, 180, 270} and min(ans.values()) >= 0.28 * len(exs)
    dirs = Counter(ex["metadata"]["rotation_direction"] for ex in exs)
    assert min(dirs.values()) >= 0.45 * len(exs)
    lay = Counter(ex["metadata"]["layout"] for ex in exs)
    assert min(lay.values()) >= 0.45 * len(exs)
    for s in range(200):
        assert generate("9", s) == exs[s]


def test_prompt_contents():
    for ex in examples()[:600]:
        m = ex["metadata"]
        prompt = get_prompt("9", m, color_palette=LETTERS)
        assert m["rotation_direction"] in prompt
        if m["layout"] == "vertical":
            assert "top image" in prompt and "bottom image" in prompt
        else:
            assert "left image" in prompt and "right image" in prompt
        assert "90, 180, or 270" in prompt
        assert not re.search(r"color \d", prompt), prompt


def test_performance():
    times = []
    for s in range(500):
        t0 = time.perf_counter()
        generate("9", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 2.0, statistics.mean(times)
    assert np.percentile(times, 99) <= 20.0
