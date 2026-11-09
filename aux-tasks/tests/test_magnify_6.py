"""Regression checks for task 6 (magnify by n), reopened in the Aug-2026 pass."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from registry import generate  # noqa: E402
from prompts import get_prompt  # noqa: E402
from common.backgrounds import texture_cells_isolated  # noqa: E402

N = 2000


def _exs(n=N):
    return [generate("6", s) for s in range(n)]


def test_output_is_exact_block_magnification():
    for seed in range(600):
        ex = generate("6", seed)
        inp = np.array(ex["input"], dtype=int)
        out = np.array(ex["output"], dtype=int)
        n = ex["metadata"]["scale"]
        assert np.array_equal(out, np.kron(inp, np.ones((n, n), dtype=int))), seed
        assert out.shape[0] <= 30 and out.shape[1] <= 30, seed
        assert len(np.unique(inp)) >= 2, seed
        assert inp.shape != (1, 1), seed
        assert f"factor of {n}" in get_prompt("6", ex["metadata"]), seed


def test_scale_and_size_coverage():
    exs = _exs()
    scales = Counter(e["metadata"]["scale"] for e in exs)
    assert set(range(2, 8)) <= set(scales), scales
    assert sum(scales[k] for k in range(8, 15)) >= 0.03 * N, scales
    hs = np.array([len(e["input"]) for e in exs])
    ws = np.array([len(e["input"][0]) for e in exs])
    # the feasible input box is [1, 30//n]; the upward lean must reach its ceiling
    assert hs.max() == 15 and ws.max() == 15
    assert np.mean(np.maximum(hs, ws) >= 11) >= 0.10  # the 11..15 zone is populated
    assert len({(h, w) for h, w in zip(hs.tolist(), ws.tolist())}) >= 120
    assert (np.minimum(hs, ws) == 1).any() and not ((hs == 1) & (ws == 1)).any()
    assert np.mean(hs == ws) <= 0.25  # small feasible boxes make coincidences likelier
    # 1xN strips are present but bounded (they were 16% by accident of the uniform side draw)
    strip = np.mean(np.minimum(hs, ws) == 1)
    assert 0.03 <= strip <= 0.08, strip


def test_content_families_and_colors():
    exs = _exs(1200)
    styles = Counter(e["metadata"]["content_style"] for e in exs)
    assert {"scene", "single_object", "dots", "noise"} <= set(styles), styles
    structured = {"stripes", "checker", "diagonal", "concentric", "mirror_lr", "mirror_ud", "mirror_both"}
    assert len(structured & set(styles)) >= 3, styles
    for e in exs:
        assert 2 <= e["metadata"]["n_colors"] <= 10
        assert e["metadata"]["n_colors"] == len(np.unique(np.array(e["input"])))


def test_colour_distribution_reaches_ten():
    """Flat colour draw over 2..10 (user rule): every count present, 10 reached, no bin
    overwhelmingly dominant.  The input is structurally small (<= 30 // scale per side)
    and 8% of inputs are single objects, so the realised shares lean low: 8-10 colours
    is ~12% here (task 10, with full-size canvases, sits near 20%)."""
    exs = _exs(4000)
    counts = Counter(len(np.unique(np.array(e["input"]))) for e in exs)
    n = len(exs)
    assert set(range(2, 11)) <= set(counts), counts
    assert counts[10] >= 0.01 * n, counts
    share_8_10 = (counts[8] + counts[9] + counts[10]) / n
    assert 0.08 <= share_8_10 <= 0.40, share_8_10
    assert max(counts.values()) / n <= 0.42, counts
    # on inputs that can show many colours the draw is close to flat
    big = [len(np.unique(np.array(e["input"]))) for e in exs if len(e["input"]) * len(e["input"][0]) >= 64]
    big_counts = Counter(big)
    assert (big_counts[8] + big_counts[9] + big_counts[10]) / len(big) >= 0.18, big_counts


def test_background_textures_on_inputs():
    exs = _exs(4000)
    tex = [e for e in exs if e["metadata"]["texture_group"] is not None]
    assert 0.04 <= len(tex) / len(exs) <= 0.20, len(tex) / len(exs)
    groups = Counter(e["metadata"]["texture_group"] for e in tex)
    assert groups["sparse"] > 0 and groups["solid"] > 0
    for e in exs:
        m = e["metadata"]
        for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
            assert key in m, key
        assert m["texture_named"] is False  # magnify prompts carry no object rule
        if m["texture_group"] is None:
            assert m["texture_kind"] is None and m["texture_colors"] == [] and m["texture_coverage"] == 0.0
            continue
        grid = np.array(e["input"])
        assert min(grid.shape) >= 8
        assert 0.0 < m["texture_coverage"] <= 0.60
        assert all(c in grid for c in m["texture_colors"])
        if m["texture_group"] == "sparse":
            assert texture_cells_isolated(grid, m["texture_colors"])


def test_determinism_and_siblings():
    for seed in (0, 5, 77):
        assert generate("6", seed) == generate("6", seed)
    path = os.path.join(os.path.dirname(__file__), "fixtures", "global_transforms_siblings_11_66.json")
    with open(path) as f:
        expected = json.load(f)
    for tid, want in expected.items():
        if tid == "11":
            continue   # task 11 was reworked in the September 2026 pass (tests/test_reflect_11.py)
        h = hashlib.sha1()
        for seed in range(500):
            h.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert h.hexdigest() == want, tid
