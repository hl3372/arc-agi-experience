"""Tests for task 11 (generators/global_transforms.py): reflect the whole grid.

Independent oracle: np.flipud / np.fliplr of the input equals the output and differs
from it.  Also: size envelope (2..30 plus bounded strips, 30 reached, squares by
coincidence), near-flat colours to 10, content families incl. library textures,
large canvases carrying real scenes, determinism, prompt contents, performance.
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

from common.backgrounds import texture_cells_isolated  # noqa: E402
from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("11", s) for s in range(SWEEP))
    return _CACHE


def test_output_is_the_flipped_input():
    for s, ex in enumerate(examples()):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        axis = ex["metadata"]["axis"]
        expected = np.flipud(inp) if axis == "horizontal" else np.fliplr(inp)
        assert np.array_equal(out, expected), f"seed {s}"
        assert not np.array_equal(inp, out), f"seed {s}: reflection invisible"
        assert ex["metadata"]["description"] == ("flip up-down" if axis == "horizontal" else "flip left-right")


def test_sizes():
    exs = examples()
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert hs.max() == 30 and ws.max() == 30
    assert not np.any((hs == 1) & (ws == 1))
    thin = np.mean((hs == 1) | (ws == 1))
    assert 0.02 <= thin <= 0.09, thin
    assert np.mean(hs == ws) <= 0.08
    non_thin_h = hs[(hs > 1) & (ws > 1)]
    counts = Counter(non_thin_h.tolist())
    assert min(counts.values()) >= 0.4 * max(counts.values())   # near-flat 2..30
    assert len(set(hs.tolist())) == 30 and len(set(ws.tolist())) == 30


def test_colours_near_flat_to_ten():
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in examples())
    total = sum(seen.values())
    assert max(seen) == 10
    assert sum(v for k, v in seen.items() if k >= 8) / total >= 0.12
    assert max(seen.values()) / total <= 0.35, seen


def test_content_families_and_textures():
    exs = examples()
    styles = Counter(ex["metadata"]["content_style"] for ex in exs)
    assert len(styles) >= 8, styles
    groups = Counter(ex["metadata"]["texture_group"] for ex in exs)
    assert groups["sparse"] >= 0.04 * len(exs) and groups["solid"] >= 0.03 * len(exs), groups
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
            assert key in m
        assert not m["texture_named"]
        if m["texture_group"] == "sparse":
            assert texture_cells_isolated(np.array(ex["input"]), m["texture_colors"]), f"seed {s}"
        if m["texture_group"] is not None:
            assert min(len(ex["input"]), len(ex["input"][0])) >= 8


def test_large_canvases_carry_scenes():
    occ = []
    for ex in examples():
        g = np.array(ex["input"])
        if g.size >= 600:
            vals, cnt = np.unique(g, return_counts=True)
            occ.append(1 - cnt.max() / g.size)
    assert len(occ) >= 100
    assert np.median(occ) >= 0.18, np.median(occ)


def test_determinism_and_prompt():
    exs = examples()
    for s in range(200):
        assert generate("11", s) == exs[s]
    for ex in exs[:300]:
        p = get_prompt("11", ex["metadata"], color_palette=list("ABCDEFGHIJ"))
        assert p.startswith("Reflect this grid across the ")
        assert not re.search(r"color \d", p) and "background" not in p


def test_performance():
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate("11", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 3.0, statistics.mean(times)
    assert np.percentile(times, 99) <= 20.0


def test_big_canvases_carry_content():
    """No canvas of 400+ cells is nearly empty: at least 3% of its cells differ from the
    majority colour (the single-object pattern style is limited to small canvases)."""
    checked = 0
    for s, ex in enumerate(examples()):
        g = np.array(ex["input"])
        if g.size >= 400:
            checked += 1
            vals, cnt = np.unique(g, return_counts=True)
            assert 1 - cnt.max() / g.size >= 0.03, f"seed {s}"
    assert checked >= 300
