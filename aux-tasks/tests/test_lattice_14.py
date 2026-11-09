"""Tests for task 14 (generators/construction_tiling.py): build a lattice from a text spec.

Independent oracle: parse the text with the module's template regexes and rebuild the
grid with an INDEPENDENT builder written here from the wording semantics; also the
older sanity checks (every named colour appears, spacing consistent).  Also: form / kind
/ wording coverage, size envelope (any 1..30 canvas outside the structural full form,
30 reached, no forced squares), colours up to 5, determinism, text conventions,
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generators.construction_tiling import _parse_task14_text  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("14", s) for s in range(SWEEP))
    return _CACHE


def build_from_params(p: dict) -> np.ndarray:
    """Independent re-implementation of the wording semantics."""
    H, W = p["H"], p["W"]
    r0, c0, r1, c1 = p["region"]
    grid = np.full((H, W), p["Z"] if p["form"] == "region" else p["F"], dtype=np.uint8)
    grid[r0:r1 + 1, c0:c1 + 1] = p["F"]

    def paint(sr, sc, colour, kind):
        rows = [r for r in range(r0, r1 + 1) if (r - r0) % sr == 0]
        cols = [c for c in range(c0, c1 + 1) if (c - c0) % sc == 0]
        if kind == "thick":
            rows = sorted({x for r in rows for x in (r, r + 1) if x <= r1})
            cols = sorted({x for c in cols for x in (c, c + 1) if x <= c1})
        for r in rows:
            for c in range(c0, c1 + 1):
                if kind != "dotted" or (c - c0) % 2 == 0:
                    grid[r, c] = colour
        for c in cols:
            for r in range(r0, r1 + 1):
                if kind != "dotted" or (r - r0) % 2 == 0:
                    grid[r, c] = colour

    paint(p["sr"], p["sc"], p["L"], p["kind"])
    if p["form"] == "overlay":
        paint(p["sr2"], p["sc2"], p["L2"], "solid")
        if p.get("L3") is not None:
            paint(p["sr3"], p["sc3"], p["L3"], "solid")
    return grid


def test_text_rebuilds_the_grid():
    for s, ex in enumerate(examples()):
        p = _parse_task14_text(ex["input"])
        out = np.array(ex["output"], dtype=np.uint8)
        assert np.array_equal(build_from_params(p), out), f"seed {s}"
        m = ex["metadata"]
        assert (p["H"], p["W"]) == (m["grid_h"], m["grid_w"]) == out.shape
        assert p["form"] == m["form"] and p["kind"] == m["kind"]
        for colour in {p["L"], p["F"]} | ({p["Z"]} if p["form"] == "region" else set()):
            assert colour in out, f"seed {s}: colour {colour} named but absent"


def test_forms_kinds_wordings():
    exs = examples()
    forms = Counter(ex["metadata"]["form"] for ex in exs)
    for f in ("full", "partial", "region", "overlay"):
        assert forms[f] / len(exs) >= 0.10, forms
    kinds = Counter(ex["metadata"]["kind"] for ex in exs)
    assert kinds["thick"] / len(exs) >= 0.04 and kinds["dotted"] / len(exs) >= 0.03, kinds
    wordings = Counter(ex["metadata"]["wording"] for ex in exs)
    assert len(wordings) == 12, wordings
    assert any(ex["metadata"]["n_lattices"] == 3 for ex in exs)
    rect = sum(1 for ex in exs if ex["metadata"]["row_spacing"] != ex["metadata"]["col_spacing"])
    assert rect / len(exs) >= 0.5


def test_sizes():
    exs = examples()
    hs = np.array([ex["metadata"]["grid_h"] for ex in exs])
    ws = np.array([ex["metadata"]["grid_w"] for ex in exs])
    assert hs.max() == 30 and ws.max() == 30
    assert len(set(hs.tolist())) >= 24 and len(set(ws.tolist())) >= 24
    assert np.mean(hs == ws) <= 0.15, np.mean(hs == ws)
    sizes = {(int(h), int(w)) for h, w in zip(hs, ws)}
    assert len(sizes) >= 300


def test_colours_and_text_conventions():
    exs = examples()
    seen = Counter(len(np.unique(np.array(ex["output"]))) for ex in exs)
    assert max(seen) >= 4 and seen[2] >= 0.3 * len(exs), seen
    for ex in exs[:500]:
        text = ex["input"]
        assert isinstance(text, str) and text.count("color ") >= 2
        assert re.search(r"\d+ by \d+|\d+x\d+", text)


def test_determinism_and_performance():
    exs = examples()
    for s in range(200):
        assert generate("14", s) == exs[s]
    times = []
    for s in range(300):
        t0 = time.perf_counter()
        generate("14", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 3.0, statistics.mean(times)


def test_side_density_flat_and_dotted_wording():
    """Size-first sampling: every side value 4..30 is about as common as any other (H and W
    pooled, max/min <= 2), squares are rare, the dotted kind never carries a full-form
    wording (those assert a solid border), and the index-based wordings state 0-indexing."""
    exs = examples()
    sides = Counter(ex["metadata"]["grid_h"] for ex in exs) + Counter(ex["metadata"]["grid_w"] for ex in exs)
    assert set(sides) == set(range(4, 31)), sorted(sides)
    assert max(sides.values()) / min(sides.values()) <= 2.0, sides
    assert np.mean([ex["metadata"]["grid_h"] == ex["metadata"]["grid_w"] for ex in exs]) <= 0.10
    for s, ex in enumerate(exs):
        m, text = ex["metadata"], ex["input"]
        if m["kind"] == "dotted":
            assert m["base"] == "partial" and "outer border" not in text, f"seed {s}"
            assert "dotted" in text
        if m["wording"] in ("full_sq_1", "full_rect_1"):
            assert "indices starting at 0" in text, f"seed {s}"
        if m["form"] == "full":
            assert (m["grid_h"] - 1) % m["row_spacing"] == 0 and (m["grid_w"] - 1) % m["col_spacing"] == 0, f"seed {s}"
