"""Tests for the reworked global-transform tasks 10 (rotate whole grid) and
55 (demagnify block-constant grid) in ``generators/global_transforms.py``.

  (a) an INDEPENDENT oracle re-solves every example from the visible input and
      the rendered prompt text (not from generator bookkeeping);
  (b) distribution assertions for the properties the review asked for
      (ARC-like sizes, square share, thin strips 1xY with Y >= 2 for task 10;
      many distinct heights / (H, W) pairs and a large small-grid share for 55);
  (c) determinism;
  (d) byte-identity of the sibling tasks 6 / 11 / 66 that share the file,
      against hashes captured from the ORIGINAL code
      (``tests/fixtures/global_transforms_siblings_6_11_66.json``);
  (e) area / token-budget caps and performance.
"""
from __future__ import annotations

import functools
from collections import Counter
import hashlib
import json
import os
import re
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402
from sft_wrapper import _MAX_TOKENS, GRID_FORMAT_COMPACT, render_input, render_output  # noqa: E402
from common.backgrounds import texture_cells_isolated  # noqa: E402

N_SWEEP = 5000
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "global_transforms_siblings_6_11_66.json")


@functools.lru_cache(maxsize=None)
def _sweep(task_id: str):
    return [generate(task_id, s) for s in range(N_SWEEP)]


def _est_tokens(task_id: str, ex: dict) -> float:
    """Same estimate as sft_wrapper.estimate_bundle_size (chars / 1.2) in the standard
    compact grid format.  Task sizes are NOT capped for the bundle (design is decoupled
    from the SFT budget); this only records that the largest example still fits."""
    chars = (len(render_input(task_id, ex["input"], fmt=GRID_FORMAT_COMPACT))
             + len(render_output(ex["output"], fmt=GRID_FORMAT_COMPACT))
             + len(get_prompt(task_id, ex["metadata"])) + 50)
    return chars / 1.2


# ---------------------------------------------------------------------------
# Task 10 — rotate whole grid
# ---------------------------------------------------------------------------

_ROT_RE = re.compile(r"^Rotate this grid (90|180|270) degrees(?: (clockwise|anticlockwise))?\.$")


def _solve_10(inp, prompt: str):
    """Oracle: parse angle/direction from the prompt and rotate with np.rot90."""
    m = _ROT_RE.match(prompt)
    assert m, prompt
    angle, direction = int(m.group(1)), m.group(2)
    steps_ccw = angle // 90
    if direction == "clockwise":
        steps_ccw = -steps_ccw
    elif direction is None:
        assert angle == 180, prompt
    return np.rot90(np.array(inp), steps_ccw).tolist()


def test_task10_oracle_matches_output():
    for s, ex in enumerate(_sweep("10")):
        assert _solve_10(ex["input"], get_prompt("10", ex["metadata"])) == ex["output"], f"seed {s}"


def test_task10_invariants():
    for s, ex in enumerate(_sweep("10")):
        a, b, m = np.array(ex["input"]), np.array(ex["output"]), ex["metadata"]
        h, w = a.shape
        assert a.min() >= 0 and a.max() <= 9 and b.min() >= 0 and b.max() <= 9
        assert (h, w) != (1, 1), f"seed {s}: 1x1 grid"
        assert h * w <= 900, f"seed {s}: area {h * w} > 900"
        assert ex["input"] != ex["output"], f"seed {s}: rotation is a no-op"
        k = m["k_cw"]
        assert b.shape == ((w, h) if k in (1, 3) else (h, w))
        if k in (1, 3):
            # direction word must be load-bearing: CW 90 != ACW 90
            assert not np.array_equal(np.rot90(a, 1), np.rot90(a, -1)), f"seed {s}: 180-symmetric input"
        expected_k = {("clockwise", 90): 1, ("clockwise", 270): 3,
                      ("anticlockwise", 90): 3, ("anticlockwise", 270): 1}.get((m.get("direction"), m["angle"]), 2)
        assert expected_k == k
        assert (m["angle"] == 180) == ("direction" not in m)
        assert m["thin"] == (min(h, w) == 1)
        assert 2 <= len(np.unique(a)) <= 10, f"seed {s}: {len(np.unique(a))} colours"
        assert m["n_colors"] == len(np.unique(a))
        assert (m["height"], m["width"]) == (h, w)


def test_task10_size_distribution_is_arc_like():
    exs = _sweep("10")
    H = [len(e["input"]) for e in exs]
    W = [len(e["input"][0]) for e in exs]
    sides = H + W
    assert 11 <= statistics.median(sides) <= 18, statistics.median(sides)  # near-flat bands
    assert np.percentile(sides, 25) <= 10  # flat 1..30 gives ~8; no area cap trims the top any more
    assert 14 <= np.percentile(sides, 75) <= 24  # near-flat bands
    square = sum(h == w for h, w in zip(H, W)) / len(exs)
    assert square <= 0.15, square  # no forced square share
    assert sum(min(h, w) <= 10 for h, w in zip(H, W)) / len(exs) >= 0.35
    assert max(sides) == 30 and min(sides) == 1
    assert len(set(H)) == 30 and len(set(W)) == 30


def test_task10_thin_strips_present_with_short_lengths():
    exs = _sweep("10")
    thin = [e for e in exs if e["metadata"]["thin"]]
    share = len(thin) / len(exs)
    assert 0.02 <= share <= 0.09, share
    lengths = [max(len(e["input"]), len(e["input"][0])) for e in thin]
    for y in (2, 3, 4):
        assert y in lengths, f"no thin strip of length {y}"
    assert min(lengths) >= 2
    # both orientations
    assert any(len(e["input"]) == 1 for e in thin) and any(len(e["input"][0]) == 1 for e in thin)
    for e in thin:
        v = np.array(e["input"]).ravel()
        assert len(set(v.tolist())) >= 2 and not np.array_equal(v, v[::-1])
        assert 2 <= len(set(v.tolist())) <= 6


def test_task10_content_and_angle_coverage():
    exs = _sweep("10")
    styles = {e["metadata"]["content_style"] for e in exs}
    assert {"objects_uniform", "objects_noise", "objects_striped", "objects_checker",
            "thin_runs", "thin_noise"} <= styles
    angles = [(e["metadata"]["angle"], e["metadata"].get("direction")) for e in exs]
    for key in [(90, "clockwise"), (90, "anticlockwise"), (270, "clockwise"),
                (270, "anticlockwise"), (180, None)]:
        assert angles.count(key) >= 0.1 * len(exs), key
    fams = set()
    for e in exs:
        fams.update(e["metadata"]["families"])
    assert len(fams) >= 10, fams
    mean_colors = statistics.mean(e["metadata"]["n_colors"] for e in exs)
    assert 4.0 <= mean_colors <= 6.5, mean_colors  # flat 2..10 draw (was capped at 6, mean ~3)
    assert {"objects_texture_sparse", "objects_texture_solid"} <= styles, styles


def test_task10_colour_distribution_reaches_ten():
    """Flat colour draw over 2..10 (user rule): 8-10 colours are a real share, no bin dominant."""
    exs = _sweep("10")
    counts = Counter(e["metadata"]["n_colors"] for e in exs)
    n = len(exs)
    assert set(range(2, 11)) <= set(counts), counts
    share_8_10 = (counts[8] + counts[9] + counts[10]) / n
    assert 0.15 <= share_8_10 <= 0.40, share_8_10
    assert max(counts.values()) / n <= 0.35, counts
    assert counts[10] >= 0.02 * n, counts


def test_task10_large_canvases_carry_large_scenes():
    """Non-background share holds with area (was 38% under 100 cells vs 11% at 700+)."""
    occ = {"small": [], "large": []}
    for e in _sweep("10"):
        a = np.array(e["input"])
        vals, cnt = np.unique(a, return_counts=True)
        share = 1 - cnt.max() / a.size
        if a.size < 100:
            occ["small"].append(share)
        elif a.size >= 700:
            occ["large"].append(share)
    assert len(occ["large"]) >= 200
    assert statistics.median(occ["large"]) >= 0.18, statistics.median(occ["large"])
    # object scenes alone (patterns excluded) also hold: the object budget's floor and
    # per-object cap rise with the area so the realised fill does not halve (10k: ~0.29)
    obj_large = []
    for e in _sweep("10"):
        a = np.array(e["input"])
        if a.size >= 700 and str(e["metadata"].get("content_style", "")).startswith("objects"):
            vals, cnt = np.unique(a, return_counts=True)
            obj_large.append(1 - cnt.max() / a.size)
    assert len(obj_large) >= 100
    assert statistics.median(obj_large) >= 0.25, statistics.median(obj_large)
    assert np.percentile(obj_large, 90) <= 0.70, np.percentile(obj_large, 90)
    assert statistics.median(occ["large"]) >= 0.45 * statistics.median(occ["small"])
    big = [e["metadata"]["n_objects"] for e in _sweep("10")
           if len(e["input"]) * len(e["input"][0]) >= 700 and e["metadata"]["n_objects"] > 0]
    assert max(big) >= 12 and statistics.median(big) >= 6, (max(big), statistics.median(big))


def test_task10_background_textures():
    exs = _sweep("10")
    tex = [e for e in exs if e["metadata"]["texture_group"] is not None]
    assert 0.12 <= len(tex) / len(exs) <= 0.30, len(tex) / len(exs)
    groups = Counter(e["metadata"]["texture_group"] for e in tex)
    assert groups["sparse"] > 0 and groups["solid"] > 0 and groups["structured"] > 0, groups
    for e in exs:
        m = e["metadata"]
        for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
            assert key in m, key
        assert m["texture_named"] is False  # rotate prompts carry no object rule
        if m["texture_group"] is None:
            assert m["texture_kind"] is None and m["texture_colors"] == [] and m["texture_coverage"] == 0.0
            continue
        a = np.array(e["input"])
        assert min(a.shape) >= 8 and m["content_style"] == "objects_texture_" + m["texture_group"]
        assert 0.0 < m["texture_coverage"] <= 0.60
        assert all(c in a for c in m["texture_colors"])
        if m["texture_group"] == "sparse":
            assert texture_cells_isolated(a, m["texture_colors"])


def test_task10_prompt_has_no_color_digits():
    for e in _sweep("10")[:500]:
        p = get_prompt("10", e["metadata"])
        assert set(re.findall(r"\d+", p)) == {str(e["metadata"]["angle"])}, p


# ---------------------------------------------------------------------------
# Task 55 — demagnify
# ---------------------------------------------------------------------------

_DEMAG_RE = re.compile(r"non-overlapping (\d+)x(\d+) blocks starting at the top-left corner")


def _solve_55(inp, prompt: str):
    """Oracle: parse n from the prompt, require monochrome blocks, take one cell per block."""
    m = _DEMAG_RE.search(prompt)
    assert m and m.group(1) == m.group(2), prompt
    n = int(m.group(1))
    a = np.array(inp)
    H, W = a.shape
    assert H % n == 0 and W % n == 0, (a.shape, n)
    out = []
    for r in range(0, H, n):
        row = []
        for c in range(0, W, n):
            block = a[r:r + n, c:c + n]
            assert (block == block[0, 0]).all(), "block not monochrome"
            row.append(int(block[0, 0]))
        out.append(row)
    return out


def test_task55_oracle_matches_output():
    for s, ex in enumerate(_sweep("55")):
        assert _solve_55(ex["input"], get_prompt("55", ex["metadata"])) == ex["output"], f"seed {s}"


def test_task55_invariants():
    for s, ex in enumerate(_sweep("55")):
        a, b, m = np.array(ex["input"]), np.array(ex["output"]), ex["metadata"]
        n = m["scale"]
        assert 2 <= n <= 15
        assert a.min() >= 0 and a.max() <= 9
        assert a.shape == (m["in_h"], m["in_w"]) and b.shape == (m["out_h"], m["out_w"])
        assert a.shape == (b.shape[0] * n, b.shape[1] * n)
        assert b.shape != (1, 1), f"seed {s}: 1x1 output"
        assert max(a.shape) <= 30
        assert 2 <= len(np.unique(b)) <= 10, f"seed {s}: {len(np.unique(b))} colours"
        assert m["n_colors"] == len(np.unique(b))


def test_task55_size_coverage():
    exs = _sweep("55")
    H = [len(e["input"]) for e in exs]
    W = [len(e["input"][0]) for e in exs]
    assert len(set(H)) >= 20, sorted(set(H))
    assert len(set(W)) >= 20, sorted(set(W))
    assert len(set(zip(H, W))) >= 250, len(set(zip(H, W)))
    # Small canvases thinned when task 55 took the MIN_SIDE = 5 floor on
    # 2026-09-15: both sides must now be 5..10 and stay compression-feasible.
    assert sum(max(h, w) <= 10 for h, w in zip(H, W)) / len(exs) >= 0.02
    assert statistics.median(H) <= 21
    # thin small grids (one side 1) exist, never both sides 1
    outs = [(e["metadata"]["out_h"], e["metadata"]["out_w"]) for e in exs]
    assert any(oh == 1 for oh, _ in outs) and any(ow == 1 for _, ow in outs)
    assert all((oh, ow) != (1, 1) for oh, ow in outs)
    thin_share = sum(1 in o for o in outs) / len(outs)
    assert 0.03 <= thin_share <= 0.20, thin_share
    square = sum(oh == ow for oh, ow in outs) / len(outs)
    assert square <= 0.25, square  # only the natural coincidences of independent draws


def test_task55_scale_and_content_coverage():
    exs = _sweep("55")
    scales = {e["metadata"]["scale"] for e in exs}
    assert set(range(2, 8)) <= scales and max(scales) >= 10
    styles = {e["metadata"]["content_style"] for e in exs}
    assert {"patchwork", "objects", "linear_bands", "mirror", "striped_panels"} <= styles
    assert statistics.mean(e["metadata"]["n_colors"] for e in exs) > 4.0
    assert statistics.mean(e["metadata"]["retains_blocks"] for e in exs) > .35



def test_task55_prompt_states_block_convention_and_no_color_digits():
    for e in _sweep("55")[:500]:
        p = get_prompt("55", e["metadata"])
        n = str(e["metadata"]["scale"])
        assert "non-overlapping" in p and "top-left" in p
        assert set(re.findall(r"\d+", p)) == {n}, p


# ---------------------------------------------------------------------------
# Shared: determinism, budget, performance, sibling byte-identity
# ---------------------------------------------------------------------------

def test_determinism():
    for tid in ("10", "55"):
        for s in range(100):
            assert generate(tid, s) == generate(tid, s)


def test_token_budget():
    for tid in ("10", "55"):
        worst = max(_est_tokens(tid, e) for e in _sweep(tid))
        assert worst <= _MAX_TOKENS, (tid, worst)


def test_performance():
    for tid in ("10", "55"):
        ms = []
        for s in range(2000):
            t0 = time.perf_counter()
            generate(tid, s)
            ms.append((time.perf_counter() - t0) * 1000)
        assert statistics.mean(ms) <= 5.0, (tid, statistics.mean(ms))
        assert np.percentile(ms, 99) <= 30.0, (tid, np.percentile(ms, 99))


def test_siblings_6_11_66_byte_identical_to_original():
    with open(FIXTURE) as fh:
        expected = json.load(fh)
    expected = {tid: h for tid, h in expected.items() if tid == "66"}  # 6 and 11 were reworked on review feedback
    assert set(expected) == {"66"}
    for tid, hashes in expected.items():
        assert len(hashes) == 500
        for s, want in enumerate(hashes):
            got = hashlib.sha1(json.dumps(generate(tid, s), sort_keys=True).encode()).hexdigest()
            assert got == want, f"task {tid} seed {s} changed"
