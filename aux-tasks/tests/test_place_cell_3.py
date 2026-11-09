"""Tests for task 3 (``generators/construction_tiling.py``): place k listed cells.

Oracle (independent of the generator): the output must equal the input with
each listed (row, col) set to its listed colour, nothing else changed; every
placement flagged unchanged really repeats the input colour there.  Also
covered: k bands, colour / layout / order modes and their shares, per-placement
no-op share, identity-output share, structured shapes really form the named
shape, size envelope (1..30 each side, 30x30 reachable, squares only by
coincidence), 10 colours reachable in the output, determinism, prompt contents
under a letter palette, performance.
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

from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
LETTERS = list("ABCDEFGHIJ")
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("3", s) for s in range(SWEEP))
    return _CACHE


def _placements(m):
    return list(zip(m["placement_rows"], m["placement_cols"], m["placement_colors"]))


def test_output_is_input_with_placements_applied():
    for s, ex in enumerate(examples()):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        m = ex["metadata"]
        assert inp.shape == out.shape
        pl = _placements(m)
        assert len(pl) == m["n_placements"] >= 1
        assert len({(r, c) for r, c, _ in pl}) == len(pl), f"seed {s}: duplicate positions"
        expected = inp.copy()
        for r, c, col in pl:
            assert 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]
            expected[r, c] = col
        assert np.array_equal(out, expected), f"seed {s}"
        # changed flags and no-op count are honest
        for (r, c, col), changed in zip(pl, m["placement_changed"]):
            assert changed == (int(inp[r, c]) != col), f"seed {s}"
        assert m["n_noop"] == sum(1 for (r, c, col) in pl if int(inp[r, c]) == col)
        assert m["output_equals_input"] == (m["n_noop"] == len(pl)) == np.array_equal(inp, out)
        if len(pl) == 1:
            assert (m["target_row"], m["target_col"], m["target_color"]) == pl[0]
        else:
            assert "target_row" not in m


def test_k_bands_and_modes():
    exs = examples()
    ks = [ex["metadata"]["n_placements"] for ex in exs]
    n = len(exs)
    band = Counter("1" if k == 1 else "2-4" if k <= 4 else "5-8" if k <= 8 else "9-12" for k in ks)
    assert 0.20 <= band["1"] / n <= 0.32, band
    assert 0.28 <= band["2-4"] / n <= 0.42, band
    assert 0.18 <= band["5-8"] / n <= 0.32, band
    assert 0.08 <= band["9-12"] / n <= 0.20, band
    assert max(ks) == 12 and min(ks) == 1
    for ex in exs:
        assert ex["metadata"]["n_placements"] <= max(1, len(ex["input"]) * len(ex["input"][0]) // 3)
    multi = [ex["metadata"] for ex in exs if ex["metadata"]["n_placements"] >= 2]
    colour = Counter(m["colour_mode"] for m in multi)
    assert 0.40 <= colour["single"] / len(multi) <= 0.60, colour
    layout = Counter(m["layout_mode"] for m in multi)
    assert 0.20 <= layout["structured"] / len(multi) <= 0.36, layout
    shapes = Counter(m["shape"] for m in multi if m["layout_mode"] == "structured")
    for sh in ("h_line", "v_line", "diagonal", "L", "rect_outline"):
        assert shapes[sh] >= 5, shapes
    order = Counter(m["order_mode"] for m in multi)
    assert 0.32 <= order["shuffled"] / len(multi) <= 0.48, order
    for m in multi:
        pl = list(zip(m["placement_rows"], m["placement_cols"]))
        if m["order_mode"] == "row_major":
            assert pl == sorted(pl)
    mixed = [m for m in multi if m["colour_mode"] == "mixed"]
    assert sum(m["n_distinct_placed_colors"] >= 2 for m in mixed) / len(mixed) >= 0.7
    assert all(m["n_distinct_placed_colors"] == 1 for m in multi if m["colour_mode"] == "single")


def test_noop_and_identity_shares():
    exs = examples()
    total = sum(ex["metadata"]["n_placements"] for ex in exs)
    noop = sum(ex["metadata"]["n_noop"] for ex in exs)
    assert 0.10 <= noop / total <= 0.20, noop / total
    ident = sum(ex["metadata"]["output_equals_input"] for ex in exs) / len(exs)
    assert 0.02 <= ident <= 0.08, ident
    # identity outputs occur for k = 1 too (the original sub-task)
    assert any(ex["metadata"]["output_equals_input"] and ex["metadata"]["n_placements"] == 1 for ex in exs)
    assert any(ex["metadata"]["output_equals_input"] and ex["metadata"]["n_placements"] >= 2 for ex in exs)


def test_structured_shapes_are_real():
    for s, ex in enumerate(examples()):
        m = ex["metadata"]
        if m["layout_mode"] != "structured":
            continue
        cells = sorted(zip(m["placement_rows"], m["placement_cols"]))
        k = len(cells)
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        sh = m["shape"]
        if sh == "h_line":
            assert len(rs) == 1 and cs == set(range(min(cs), min(cs) + k)), f"seed {s}"
        elif sh == "v_line":
            assert len(cs) == 1 and rs == set(range(min(rs), min(rs) + k)), f"seed {s}"
        elif sh == "diagonal":
            assert len(rs) == k and len(cs) == k
            d1 = all(r - c == cells[0][0] - cells[0][1] for r, c in cells)
            d2 = all(r + c == cells[0][0] + cells[0][1] for r, c in cells)
            assert d1 or d2, f"seed {s}"
        elif sh == "L":
            h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            assert h >= 2 and w >= 2 and k == h + w - 1
            corner_ok = any(set(cells) == {(r, cc) for r in range(min(rs), max(rs) + 1)} | {(rr, c) for c in range(min(cs), max(cs) + 1)}
                            for rr in (min(rs), max(rs)) for cc in (min(cs), max(cs)))
            assert corner_ok, f"seed {s}"
        else:
            h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            assert h >= 3 and w >= 3 and k == 2 * (h + w) - 4
            assert all(r in (min(rs), max(rs)) or c in (min(cs), max(cs)) for r, c in cells), f"seed {s}"


def test_sizes_and_determinism():
    exs = examples()
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert hs.min() == 1 and ws.min() == 1 and hs.max() == 30 and ws.max() == 30
    assert (hs * ws).max() == 900
    assert np.mean(hs == ws) <= 0.08
    for s in range(200):
        assert generate("3", s) == exs[s]


def test_palette_reaches_ten_colours_in_output():
    seen = Counter(len(np.unique(np.array(ex["output"]))) for ex in examples())
    assert max(seen) == 10, seen
    assert seen[2] + seen[3] + seen[4] >= 0.25 * sum(seen.values()), seen


def test_prompt_contents():
    for s, ex in enumerate(examples()[:800]):
        m = ex["metadata"]
        prompt = get_prompt("3", m, color_palette=LETTERS)
        assert "0-indexed" in prompt and "Leave every other cell unchanged." in prompt
        assert not re.search(r"color \d", prompt), prompt
        for r, c, col in _placements(m):
            assert f"({r}, {c})" in prompt, f"seed {s}"
            assert f"color {LETTERS[col]}" in prompt, f"seed {s}"
        if m["n_placements"] == 1:
            assert prompt.startswith("Place a color ")
        elif m["colour_mode"] == "single":
            assert prompt.startswith("Place color ") and " and (" in prompt
        else:
            assert prompt.startswith("Place cells at the following positions: ")
            assert prompt.count(" in color ") == m["n_placements"]


def test_performance():
    times = []
    for s in range(500):
        t0 = time.perf_counter()
        generate("3", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 2.0, statistics.mean(times)
    assert np.percentile(times, 99) <= 15.0
