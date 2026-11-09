"""Tests for task 5a (``generators/background_restore.py``): remove every object, restore the
background.

Independent oracle: objects are the cells whose colour is one of the metadata object colours
(the analogue of the named colours the 4/5 tests use); the background under them is inferred
with a separate, compact implementation of the four hypothesis families (2-D periodic with the
smallest consistent period, border distance, axis / corner split enumerated exhaustively over
both colour roles with prefix sums, tiles), and the answer must be the unique fill.  Also
covered: the untouched cells, object validity, the size floor (short side >= 8, area >= 120,
no strips), the visibility guarantees (pattern classes witnessed, texture >= 60% visible, no
full-span object, occupancy <= 28%), identity share, colours, texture kinds and metadata,
occupancy by area, determinism, prompt contents under the letter palette, performance, and
the corner-split regression (the inside colour is enumerated, not taken from the nearest
visible cell).
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

from common.backgrounds import texture_cells_isolated  # noqa: E402
from generators.background_restore import (  # noqa: E402
    CLASS_WITNESSES,
    MAX_OCCUPANCY,
    MIN_AREA,
    MIN_SHORT_SIDE,
    SOLID_KINDS,
    SPARSE_KINDS,
    STRUCTURED_KINDS,
    TEXTURE_VISIBLE_MIN,
    _split_boundaries,
    infer_fill,
    pattern_classes,
)
from prompts import ISOLATED_CELL_RULE, get_prompt  # noqa: E402
from registry import generate  # noqa: E402

SWEEP = 2000
ORACLE_SEEDS = 1500
LETTERS = list("ABCDEFGHIJ")
_CACHE: list[dict] = []


def examples() -> list[dict]:
    if not _CACHE:
        _CACHE.extend(generate("5a", s) for s in range(SWEEP))
    return _CACHE


# ---------------------------------------------------------------------------
# Independent oracle
# ---------------------------------------------------------------------------

def _consistent_fill(v, vis, hidden, cls, n_cls):
    """Fill of hidden cells under a class map, or None (class conflict / unwitnessed)."""
    vals = {}
    for k, x in zip(cls[vis].tolist(), v[vis].tolist()):
        if vals.setdefault(k, x) != x:
            return None
    out = []
    for k in cls[hidden].tolist():
        if k not in vals:
            return None
        out.append(vals[k])
    return tuple(out)


def _split_fills(v, vis, hidden):
    """Every corner-rectangle / half-plane split consistent with the visible cells, both
    colour roles, via per-colour prefix sums (exhaustive and fast)."""
    H, W = v.shape
    fills = set()
    cols = sorted({int(c) for c in np.unique(v[vis])})
    if len(cols) < 2:
        return fills
    hid = np.argwhere(hidden)
    for fr in (False, True):
        for fc in (False, True):
            g = v[::-1] if fr else v
            vv = vis[::-1] if fr else vis
            if fc:
                g, vv = g[:, ::-1], vv[:, ::-1]
            stacks = []
            for c in cols:
                m = (vv & (g == c)).astype(np.int64)
                cum = np.zeros((H + 1, W + 1), dtype=np.int64)
                cum[1:, 1:] = m.cumsum(0).cumsum(1)
                stacks.append(cum)
            cum = np.stack(stacks, axis=-1)
            total = cum[H, W]
            inside = cum[1:, 1:]
            outside = total[None, None, :] - inside
            ok = ((inside > 0).sum(-1) == 1) & ((outside > 0).sum(-1) == 1)
            ai = inside.argmax(-1)
            bi = outside.argmax(-1)
            ok &= ai != bi
            hr = (H - 1 - hid[:, 0]) if fr else hid[:, 0]
            hc = (W - 1 - hid[:, 1]) if fc else hid[:, 1]
            for R0, C0 in np.argwhere(ok).tolist():
                R0 += 1
                C0 += 1
                if R0 == H and C0 == W:
                    continue
                if (R0 == H and fr) or (C0 == W and fc):
                    continue
                a, b = cols[ai[R0 - 1, C0 - 1]], cols[bi[R0 - 1, C0 - 1]]
                h_in = (hr < R0) & (hc < C0)
                fills.add(tuple(int(a) if x else int(b) for x in h_in.tolist()))
    return fills


def oracle_fills(grid: np.ndarray, hidden: np.ndarray) -> set:
    """Every distinct fill of the hidden cells the four families allow."""
    H, W = grid.shape
    v = grid.astype(int)
    vis = ~hidden
    rr, cc = np.indices((H, W))
    fills = set()
    if not hidden.any():
        return {()}
    valid = []
    for py in range(1, min(H, 18) + 1):
        for px in range(1, min(W, 18) + 1):
            if any(py % a == 0 and px % b == 0 for a, b in valid):
                continue
            f = _consistent_fill(v, vis, hidden, (rr % py) * px + (cc % px), py * px)
            if f is not None:
                valid.append((py, px))
                fills.add(f)
    depth = np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc))
    f = _consistent_fill(v, vis, hidden, depth, int(depth.max()) + 1)
    if f is not None:
        fills.add(f)
    fills |= _split_fills(v, vis, hidden)
    for b in (2, 3):
        for oy in range(b):
            for ox in range(b):
                ty, tx = (rr + oy) // b, (cc + ox) // b
                f = _consistent_fill(v, vis, hidden, ty * (int(tx.max()) + 1) + tx, (int(ty.max()) + 1) * (int(tx.max()) + 1))
                if f is not None:
                    fills.add(f)
    return fills


def _hidden(ex):
    m = ex["metadata"]
    inp = np.array(ex["input"])
    return inp, np.isin(inp, m["object_colors"]) if m["object_colors"] else np.zeros(inp.shape, bool)


def test_answer_is_the_unique_restorable_background():
    for s, ex in enumerate(examples()):
        inp, hidden = _hidden(ex)
        out = np.array(ex["output"])
        assert inp.shape == out.shape
        assert np.array_equal(inp[~hidden], out[~hidden]), f"seed {s}: a background cell changed"
        assert not np.isin(out, ex["metadata"]["object_colors"]).any(), f"seed {s}: an object colour survives"
        if s < ORACLE_SEEDS:
            fills = oracle_fills(inp, hidden)
            assert len(fills) == 1, f"seed {s}: {len(fills)} fills"
            assert next(iter(fills)) == tuple(int(x) for x in out[hidden]), f"seed {s}"
            fill, n, _fams = infer_fill(inp, hidden)
            assert n == 1 and np.array_equal(fill, out[hidden]), f"seed {s}: generator proof disagrees with the oracle"


def test_corner_split_enumerates_both_colour_roles():
    """A corner rectangle whose corner cell is hidden and whose nearest visible cell (by depth)
    carries the OUTSIDE colour: the inside colour must still be found (verifier finding)."""
    H, W = 9, 12
    grid = np.full((H, W), 5)
    grid[:1, :4] = 2                       # inside: rows < 1, cols < 4 in colour 2
    hidden = np.zeros((H, W), bool)
    hidden[0, :3] = True                   # the corner and its row neighbours
    hidden[1, :3] = True                   # depth-1 / depth-2 cells too
    # nearest visible cell by max(r, c) is (2, 0) in colour 5: the outside colour
    assert grid[2, 0] == 5 and not hidden[2, 0] and not hidden[0, 3] and grid[0, 3] == 2
    truth = tuple(int(x) for x in grid[hidden])
    rr, cc = np.indices((H, W))
    # the corner enumerator must offer the true boundary (rows < 1, cols < 4) for inside colour 2;
    # the old code took colour 5 from the nearest visible cell and never enumerated it
    bounds = _split_boundaries(grid, ~hidden, rr, cc, "corner", H, W)
    assert (1, 4) in bounds, bounds
    # the true fill is among the enumerated fills of both the generator and the oracle (this
    # synthetic scene also admits a periodic reading, so uniqueness is not claimed here)
    assert truth in oracle_fills(grid, hidden)
    _fill, n, fams = infer_fill(grid, hidden)
    assert "split" in fams and n >= 1
    # and the mirror image (inside rectangle at the bottom-right corner)
    g2 = grid[::-1, ::-1].copy()
    h2 = hidden[::-1, ::-1].copy()
    bounds2 = _split_boundaries(g2, ~h2, H - 1 - rr, W - 1 - cc, "corner", H, W)
    assert (1, 4) in bounds2, bounds2


def test_objects_are_valid_and_identity_share():
    exs = examples()
    ident = 0
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        inp = np.array(ex["input"])
        ident += m["identity"]
        if m["identity"]:
            assert ex["input"] == ex["output"] and m["n_objects"] == 0
            continue
        assert m["n_objects"] >= 1
        for c in m["object_colors"]:
            assert c not in m["background_colors"]
        obj = np.isin(inp, m["object_colors"])
        H, W = inp.shape
        pad = np.full((H + 2, W + 2), -1)
        pad[1:-1, 1:-1] = inp
        nb = np.zeros((H, W), bool)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                if dr == 1 and dc == 1:
                    continue
                nb |= pad[dr:dr + H, dc:dc + W] == inp
        assert np.all(nb[obj]), f"seed {s}: a single-cell object"
        # no object spans the full height or width of the canvas
        for c in m["object_colors"]:
            rows = np.any(inp == c, axis=1)
            cols = np.any(inp == c, axis=0)
            assert not rows.all() or not _one_component_spans(inp == c, axis=0), f"seed {s}"
            assert not cols.all() or not _one_component_spans(inp == c, axis=1), f"seed {s}"
    assert 0.02 <= ident / len(exs) <= 0.08, ident / len(exs)


def _one_component_spans(mask: np.ndarray, axis: int) -> bool:
    """Does one 8-connected component of ``mask`` reach both ends along ``axis``?"""
    H, W = mask.shape
    seen = np.zeros_like(mask)
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            lo, hi = 10 ** 6, -1
            while stack:
                a, b = stack.pop()
                k = a if axis == 0 else b
                lo, hi = min(lo, k), max(hi, k)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        x, y = a + dr, b + dc
                        if 0 <= x < H and 0 <= y < W and mask[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            stack.append((x, y))
            if lo == 0 and hi == (H - 1 if axis == 0 else W - 1):
                return True
    return False


def test_visibility_guarantees():
    """Every pattern class keeps min(3, size) visible witnesses, textures stay >= 60% visible,
    occupancy <= 28% -- recomputed here with the generator's class map but our own counting."""
    for s, ex in enumerate(examples()):
        m = ex["metadata"]
        inp, hidden = _hidden(ex)
        out = np.array(ex["output"])
        assert m["occupancy"] <= MAX_OCCUPANCY + 1e-9 and hidden.mean() <= MAX_OCCUPANCY + 1e-9, s
        kind = m["texture_kind"] or "plain"
        cls = pattern_classes(out, kind)
        total = Counter(cls.ravel().tolist())
        visible = Counter(cls[~hidden].ravel().tolist())
        for k, n in total.items():
            assert visible.get(k, 0) >= min(CLASS_WITNESSES, n), f"seed {s}: class {k} has {visible.get(k, 0)}/{n} witnesses"
        if m["texture_colors"]:
            tex = np.isin(out, m["texture_colors"])
            frac = (tex & ~hidden).sum() / tex.sum()
            assert frac >= TEXTURE_VISIBLE_MIN - 1e-9, f"seed {s}: texture {frac:.2f} visible"
            assert abs(frac - m["visibility"]["texture_visible"]) < 1e-3
        assert m["visibility"]["min_witnesses"] >= 1


def test_sizes():
    exs = examples()
    hs = np.array([len(ex["input"]) for ex in exs])
    ws = np.array([len(ex["input"][0]) for ex in exs])
    assert hs.min() == MIN_SHORT_SIDE == 8 and ws.min() == 8 and hs.max() == 30 and ws.max() == 30
    assert (hs * ws).min() >= MIN_AREA == 120
    assert (hs * ws).max() == 900
    assert np.mean(hs == ws) <= 0.08
    for side in (hs, ws):
        counts = Counter(side.tolist())
        vals = [counts[k] for k in range(8, 31)]
        # uniform over the feasible pairs: sides 8..14 have fewer partners under the area floor
        assert max(vals) / max(1, min(vals)) <= 2.6, counts
    assert len({(h, w) for h, w in zip(hs, ws)}) >= 400


def test_colours():
    seen = Counter(len(np.unique(np.array(ex["input"]))) for ex in examples())
    n = sum(seen.values())
    assert max(seen) == 10 and seen[1] == 0, seen
    assert 0.12 <= (seen[8] + seen[9] + seen[10]) / n <= 0.40, seen
    assert max(seen.values()) / n <= 0.35, seen


def test_texture_kinds_and_metadata():
    exs = examples()
    kinds = Counter(ex["metadata"]["texture_kind"] or "plain" for ex in exs)
    for k in ("random_dots", "multi_speckle", "mirror", "dotted_border"):
        assert kinds[k] == 0
    for k in SPARSE_KINDS + SOLID_KINDS:
        assert kinds[k] >= 0.01 * len(exs), (k, kinds)
    assert kinds["mosaic"] >= 0.003 * len(exs), kinds
    assert 0.15 <= kinds["plain"] / len(exs) <= 0.40, kinds
    for s, ex in enumerate(exs):
        m = ex["metadata"]
        out = np.array(ex["output"])
        assert m["texture_named"] is False
        for key in ("background", "background_colors", "texture_group", "texture_kind", "texture_colors",
                    "texture_coverage", "n_objects", "object_colors", "n_colors", "grid_h", "grid_w",
                    "identity", "determinability", "visibility", "connectivity"):
            assert key in m, key
        assert m["determinability"]["distinct_fills"] == 1
        assert set(m["background_colors"]) == {int(c) for c in np.unique(out)}
        if m["texture_kind"] is None:
            assert m["texture_colors"] == [] and len(np.unique(out)) == 1
        else:
            assert m["texture_colors"] and set(m["texture_colors"]) <= set(m["background_colors"])
            if m["texture_group"] == "sparse":
                assert texture_cells_isolated(out, m["texture_colors"]), s
            elif m["texture_group"] == "structured":
                assert min(out.shape) >= 12 and m["texture_kind"] in STRUCTURED_KINDS
            else:
                assert m["texture_group"] == "solid" and m["texture_kind"] in SOLID_KINDS


def test_occupancy_by_area():
    occ = {"<200": [], "200-399": [], "400-699": [], "700+": []}
    for ex in examples():
        m = ex["metadata"]
        if m["identity"]:
            continue
        a = m["grid_h"] * m["grid_w"]
        occ["<200" if a < 200 else "200-399" if a < 400 else "400-699" if a < 700 else "700+"].append(m["occupancy"])
    for k, v in occ.items():
        assert len(v) >= 50, k
        assert np.median(v) >= 0.15, (k, np.median(v))
        assert max(v) <= MAX_OCCUPANCY + 1e-9, (k, max(v))


def test_determinism_and_prompt():
    for s in range(200):
        assert generate("5a", s) == examples()[s]
    for ex in examples()[:300]:
        p = get_prompt("5a", ex["metadata"], color_palette=LETTERS)
        assert ISOLATED_CELL_RULE in p and "at least two cells" in p and "Remove every object" in p
        assert "Output the whole grid." in p
        assert not re.search(r"color \d", p) and "background is" not in p and "background color" not in p
        for c in ex["metadata"]["texture_colors"] + ex["metadata"]["background_colors"]:
            assert f"color {LETTERS[c]}" not in p


def test_performance():
    times = []
    for s in range(400):
        t0 = time.perf_counter()
        generate("5a", s)
        times.append((time.perf_counter() - t0) * 1000)
    assert statistics.mean(times) <= 4.5  # headroom for parallel load, statistics.mean(times)
    assert np.percentile(times, 99) <= 20.0
