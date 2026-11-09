"""Tests for task 20 (recolor every object with exactly n cells; connectivity 4 or 8 per example).

Task 21 (the 8-connected twin) was merged into task 20 on 2026-09-07: the connectivity is drawn
per example and stated in the prompt, and the scene is drawn without looking at it (the same
cluster feature -- same-colour pieces touching only at corners -- at the same rates under both
readings, see ``test_scene_independent_of_connectivity``).

An INDEPENDENT oracle re-solves every example from the displayed input alone, reading the
prompt literally: the background is the one colour a solver can identify without being told
(isolated single cells set aside, strict plurality by 1.25x, largest 4-connected region);
EVERY other same-colour component, texture cells included, is an object; the ones with exactly
n cells are recoloured and nothing else changes.  The texture policy follows from that reading:
no texture component may have exactly n cells, so for n == 1 there are no sparse textures at
all.  Distribution assertions encode the review targets; the sibling generators sharing
``generators/color_filtering.py`` (29, 70, 84, 85) are checked byte-for-byte against hashes
captured from the original code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import get_prompt  # noqa: E402
from registry import generate  # noqa: E402
from sft_wrapper import render_input, render_output  # noqa: E402

N_ORACLE = 2000
N_DIST = 1000
LETTERS = list("ABCDEFGHIJ")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sibling_hashes_color_filtering.json")

_D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_D8 = _D4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


# --------------------------------------------------------------------------- oracle

def _components(a: np.ndarray, skip, connectivity: int):
    """Same-colour components of every colour not in *skip* (own BFS)."""
    h, w = a.shape
    seen = np.zeros_like(a, dtype=bool)
    deltas = _D4 if connectivity == 4 else _D8
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or a[r, c] in skip:
                continue
            col = a[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in deltas:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and a[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((int(col), cells))
    return comps


def _isolated(a: np.ndarray) -> np.ndarray:
    """Cells with no 8-neighbour of the same colour."""
    h, w = a.shape
    iso = np.ones_like(a, dtype=bool)
    for r in range(h):
        for c in range(w):
            for dy, dx in _D8:
                ny, nx = r + dy, c + dx
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] == a[r, c]:
                    iso[r, c] = False
                    break
    return iso


def _infer_background(a: np.ndarray):
    """The background a solver told only 'ignore the background' can identify: isolated single
    cells set aside, the strict plurality colour by a 1.25x margin that also owns the largest
    4-connected region.  Returns (colour, margin)."""
    keep = ~_isolated(a)
    counts = Counter(int(v) for v in a[keep])
    assert counts, "every cell is isolated"
    ranked = counts.most_common()
    top, top_n = ranked[0]
    runner_n = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_n / runner_n if runner_n else float("inf")
    assert margin >= 1.25, f"background not identifiable (margin {margin:.3f})"
    masked = a.copy()
    masked[~keep] = -1
    regions = _components(masked, {-1}, 4)
    largest = max(regions, key=lambda t: len(t[1]))
    assert largest[0] == top, "plurality colour does not own the largest region"
    return top, margin


def _connectivity_from_prompt(prompt: str) -> int:
    got = re.findall(r"\b([48])-connected", prompt)
    assert len(set(got)) == 1, prompt
    return int(got[0])


def _solve(ex: dict):
    """Literal reading: every non-background same-colour component (under the connectivity the
    prompt states) is an object."""
    a = np.array(ex["input"])
    m = ex["metadata"]
    connectivity = _connectivity_from_prompt(get_prompt("20", m))
    assert connectivity == m["connectivity"]
    bg, margin = _infer_background(a)
    assert bg == m["background"], "metadata background differs from the identifiable one"
    assert margin >= 1.25
    n, target = m["n"], m["target_color"]
    assert target not in {int(v) for v in np.unique(a)}, "target colour must be absent from the input"
    comps = _components(a, {bg}, connectivity)
    matches = [cells for _, cells in comps if len(cells) == n]
    assert matches, "no object of size n"
    assert len(matches) < len(comps), "every object matches"
    out = a.copy()
    for cells in matches:
        for r, c in cells:
            out[r, c] = target
    return out.tolist(), connectivity


@pytest.fixture(scope="module")
def dist():
    exs = [generate("20", s) for s in range(N_DIST)]
    return exs, [e["metadata"] for e in exs]


# --------------------------------------------------------------------------- tests

def test_oracle_recomputes_every_answer():
    conns = Counter()
    for s in range(N_ORACLE):
        ex = generate("20", s)
        out, conn = _solve(ex)
        conns[conn] += 1
        assert out == ex["output"], f"seed {s}: oracle mismatch"
        assert ex["input"] != ex["output"]
        a = np.array(ex["input"])
        assert a.min() >= 0 and a.max() <= 9
        assert 4 <= a.shape[0] <= 30 and 4 <= a.shape[1] <= 30
        # the answer under the other reading is what the metadata says it is
        other = 12 - conn
        bg = ex["metadata"]["background"]
        tex = set(ex["metadata"]["texture_colors"])
        alt = np.array(ex["input"])
        for _, cells in _components(np.array(ex["input"]), {bg} | tex, other):
            if len(cells) == ex["metadata"]["n"]:
                for r, c in cells:
                    alt[r, c] = ex["metadata"]["target_color"]
        assert (alt.tolist() != ex["output"]) == ex["metadata"]["differs_under_other_connectivity"], s
    assert 0.42 <= conns[4] / N_ORACLE <= 0.58, conns


def test_retired_twin_raises():
    with pytest.raises(KeyError, match="merged into task 20"):
        generate("21", 0)


def test_texture_and_ground_never_form_an_n_cell_object():
    """The only texture rule on this task: no texture component (under the example's connectivity)
    has exactly n cells, so the literal reading and 'texture is background' agree.  For n == 1 a
    dot would be a 1-cell object, so there is no sparse texture at all.  The ground itself never
    forms a component of exactly n cells either, and texture colours are disjoint from the
    object colours and the recolour colour."""
    for s in range(N_ORACLE):
        ex = generate("20", s)
        m = ex["metadata"]
        conn = m["connectivity"]
        a = np.array(ex["input"])
        n, bg = m["n"], m["background"]
        tex = set(m["texture_colors"])
        assert m["texture_named"] is False
        assert bg not in tex and m["target_color"] not in tex
        if n == 1:
            assert m["texture_group"] != "sparse", f"seed {s}: sparse texture with n = 1"
            assert not (tex and _isolated(a)[np.isin(a, list(tex))].any()), "isolated texture cell with n = 1"
        if tex:
            assert m["texture_group"] in ("sparse", "solid")
            sizes = [len(cells) for col, cells in _components(a, set(range(10)) - tex, conn)]
            assert n not in sizes, f"seed {s}: texture component of exactly n cells"
            obj_cols = {int(a[r, c]) for r in range(a.shape[0]) for c in range(a.shape[1])
                        if a[r, c] != bg and a[r, c] not in tex}
            assert not (obj_cols & tex)
        else:
            assert m["texture_group"] is None and m["bg_style"] == "uniform"
        ground = [len(cells) for _, cells in _components(a, set(range(10)) - {bg}, conn)]
        assert n not in ground, f"seed {s}: ground component of exactly n cells"


def test_clusters_are_what_the_metadata_says():
    """Every cluster is one 8-component that splits into its pieces under 4-connectivity, with
    the piece sizes of the metadata; nothing else touches diagonally."""
    for s in range(600):
        ex = generate("20", s)
        m = ex["metadata"]
        a = np.array(ex["input"])
        bg = m["background"]
        skip = {bg} | set(m["texture_colors"])
        comps8 = _components(a, skip, 8)
        comps4 = _components(a, skip, 4)
        split = []
        for col, c8 in comps8:
            cells8 = set(c8)
            parts = sorted(len(c4) for _, c4 in comps4 if set(c4) <= cells8)
            if len(parts) > 1:
                split.append(parts)
        assert sorted(split) == sorted(sorted(p) for p in m["cluster_pieces"]), (s, split, m["cluster_pieces"])
        assert m["n_clusters"] == len(m["cluster_pieces"]) == m["diag_link"] and m["diag_pair"] == (m["n_clusters"] > 0)
        kinds = {("chain" if len(p) >= 3 else "pendant" if min(p) == 1 else "corner_pair") for p in m["cluster_pieces"]}
        assert kinds == set(m["kinds"]), (s, kinds, m["kinds"])
        for pieces, option in zip(m["cluster_pieces"], m["cluster_options"]):
            if option == "piece_n":
                assert m["n"] in pieces and sum(pieces) != m["n"]
            elif option == "sum_n":
                assert sum(pieces) == m["n"] and all(p < m["n"] for p in pieces)
            else:
                assert m["n"] not in pieces and sum(pieces) != m["n"]


def test_determinism():
    for s in range(40):
        assert generate("20", s) == generate("20", s)


def test_metadata_keys_and_prompt():
    for s in range(300):
        ex = generate("20", s)
        m = ex["metadata"]
        for key in ("n", "target_color", "background", "background_margin", "connectivity", "n_matches",
                    "n_objects", "object_sizes", "families", "bg_style", "decoy_sizes", "texture_group",
                    "texture_kind", "texture_colors", "texture_named", "texture_coverage", "palette_size",
                    "n_colors", "n_mode", "grid_h", "grid_w", "fg_fraction", "touch", "attempts",
                    "n_clusters", "cluster_pieces", "cluster_options", "kinds", "diag_pair", "diag_link",
                    "other_connectivity", "differs_under_other_connectivity"):
            assert key in m, f"seed {s}: missing metadata {key}"
        conn = m["connectivity"]
        assert conn in (4, 8) and m["other_connectivity"] == 12 - conn
        assert m["background_margin"] >= 1.25
        prompt = get_prompt("20", m)
        assert (f"exactly {m['n']} cell" + ("" if m['n'] == 1 else "s") + " to color") in prompt
        assert f"to color {m['target_color']}" in prompt
        assert prompt.endswith("Leave every other cell unchanged.")
        assert prompt.startswith("Ignore the background, whether it is a solid color or a pattern.")
        # policy: the background is never named or defined
        assert "background is" not in prompt and "background color" not in prompt
        assert re.findall(r"color (\d)", prompt) == [str(m["target_color"])]
        if conn == 4:
            assert "4-connected" in prompt and "corner" in prompt and "diagonal" not in prompt
        else:
            assert "8-connected" in prompt and "diagonally" in prompt
        letters = get_prompt("20", m, color_palette=LETTERS)
        assert not re.search(r"color \d", letters), letters
        assert f"to color {LETTERS[m['target_color']]}" in letters


def test_token_budget_compact_format():
    for s in range(600):
        ex = generate("20", s)
        chars = (len(render_input("20", ex["input"], fmt="compact"))
                 + len(render_output(ex["output"], task_id="20", fmt="compact"))
                 + len(get_prompt("20", ex["metadata"])) + 50)
        assert chars / 1.2 <= 2900, f"seed {s}: over the token budget"


def test_performance():
    t0 = time.perf_counter()
    for s in range(300):
        generate("20", s)
    assert (time.perf_counter() - t0) / 300 < 0.08, "too slow"


def test_distribution(dist):
    exs, metas = dist
    hs = [len(e["input"]) for e in exs]
    ws = [len(e["input"][0]) for e in exs]
    areas = [h * w for h, w in zip(hs, ws)]
    # sizes: H and W independent and near-flat over 4..30, no area cap, 30 reached
    for side in (hs, ws):
        assert min(side) == 4 and max(side) == 30
        assert 15.0 <= np.mean(side) <= 19.0
        for lo, hi in ((4, 10), (11, 20), (21, 30)):
            got = np.mean([lo <= v <= hi for v in side])
            expect = (hi - lo + 1) / 27
            assert abs(got - expect) <= 0.08, (lo, hi, got)
    assert np.mean([h == w for h, w in zip(hs, ws)]) <= 0.10   # no forced square share
    assert np.mean([a >= 700 for a in areas]) >= 0.05
    assert max(areas) >= 800
    assert abs(np.corrcoef(hs, ws)[0, 1]) <= 0.12
    # answer spread: every example has >= 1 match, never all objects, <= 60% matched; the number of
    # matches grows with the canvas under a footprint-based cap
    assert all(1 <= m["n_matches"] < m["n_objects"] for m in metas)
    assert all(m["n_matches"] <= 0.6 * m["n_objects"] for m in metas)
    mh = Counter(m["n_matches"] for m in metas)
    assert 0.38 <= mh[1] / N_DIST <= 0.60, mh
    assert mh[2] / N_DIST >= 0.18, mh
    assert sum(v for k, v in mh.items() if k >= 3) / N_DIST >= 0.12, mh
    assert sum(v for k, v in mh.items() if k >= 5) >= 8, mh
    assert max(mh) <= 12
    big = [m["n_matches"] for m, a in zip(metas, areas) if a >= 400]
    small = [m["n_matches"] for m, a in zip(metas, areas) if a < 100]
    assert np.mean(big) >= np.mean(small) + 0.6
    assert np.mean([v >= 3 for v in big]) >= 0.20
    # n: mode share of n = 1 kept near 7%, big n present
    assert 0.03 <= np.mean([m["n"] == 1 for m in metas]) <= 0.12
    assert np.mean([m["n"] >= 6 for m in metas]) >= 0.25
    assert max(m["n"] for m in metas) >= 20
    assert np.mean([bool(m["decoy_sizes"]) for m in metas]) >= 0.75
    # objects scale with the canvas
    assert 2 <= min(m["n_objects"] for m in metas) and max(m["n_objects"] for m in metas) <= 24
    n_obj_big = [m["n_objects"] for m, a in zip(metas, areas) if a >= 700]
    assert statistics.median(n_obj_big) >= 11
    # occupancy holds across canvas sizes (no thinning on large grids)
    for lo, hi in ((0, 100), (100, 200), (200, 400), (400, 700), (700, 901)):
        occ = [m["fg_fraction"] for m, a in zip(metas, areas) if lo <= a < hi]
        assert occ, (lo, hi)
        assert 0.12 <= statistics.median(occ) <= 0.26, (lo, hi, statistics.median(occ))
    # object sizes: real range, >= 8% of objects with >= 10 cells
    sizes = [s for m in metas for s in m["object_sizes"]]
    assert np.mean([s >= 10 for s in sizes]) >= 0.08
    assert max(sizes) >= 25 and min(sizes) == 1
    fams = Counter(f for m in metas for f in m["families"] if "+" not in f)
    assert len(fams) >= 8, fams
    # colours: 1..8 object colours (near-flat draw, capped by the object count), up to 9 colours in
    # the input and 10 in the output; the background stays identifiable by >= 1.25x
    pal = Counter(m["palette_size"] for m in metas)
    assert set(pal) == set(range(1, 9)), pal
    assert max(pal.values()) <= 0.30 * N_DIST, pal
    n_in = [len(set(v for row in e["input"] for v in row)) for e in exs]
    n_out = [len(set(v for row in e["output"] for v in row)) for e in exs]
    assert max(n_in) == 9 and max(n_out) == 10
    assert np.mean([v >= 7 for v in n_in]) >= 0.12
    assert all(m["background_margin"] >= 1.25 for m in metas)
    assert statistics.median(m["background_margin"] for m in metas) >= 3.0
    # textures: shared library, sparse or solid only, short side >= 8, moderate rate, several kinds
    mins = [min(h, w) for h, w in zip(hs, ws)]
    assert all(m["texture_group"] is None for m, mn in zip(metas, mins) if mn < 8)
    groups = Counter(m["texture_group"] for m in metas)
    assert set(groups) <= {None, "sparse", "solid"}, groups
    eligible = [m for m, mn in zip(metas, mins) if mn >= 8]
    tex_rate = np.mean([m["texture_group"] is not None for m in eligible])
    assert 0.12 <= tex_rate <= 0.35, tex_rate
    assert groups["sparse"] >= 40 and groups["solid"] >= 20, groups
    kinds = Counter(m["texture_kind"] for m in metas if m["texture_kind"])
    assert len(kinds) >= 6, kinds
    assert all(m["texture_coverage"] <= 0.55 for m in metas)
    assert all(m["texture_group"] != "sparse" for m in metas if m["n"] == 1)
    # deliberate different-colour touching stays rare
    assert np.mean([m["touch"] for m in metas]) <= 0.16
    # clusters: present in a good share of scenes, all three kinds, up to three per scene
    assert 0.40 <= np.mean([m["n_clusters"] > 0 for m in metas]) <= 0.75
    ck = Counter(k for m in metas for k in m["kinds"])
    assert {"chain", "pendant", "corner_pair"} <= set(ck), ck
    assert max(m["n_clusters"] for m in metas) == 3
    assert Counter(o for m in metas for o in m["cluster_options"]).keys() >= {"piece_n", "sum_n", "free"}


def _shares(rows, key_fn) -> dict:
    c = Counter(key_fn(m) for m in rows)
    n = len(rows)
    return {k: v / n for k, v in c.items()}


def test_scene_independent_of_connectivity(dist):
    """The picture never hints the connectivity: cluster counts, piece counts and kinds, textures,
    object counts, occupancy, n, matches and canvas sizes have the same rates under both readings;
    and the connectivity clause is load-bearing under each reading on its own."""
    exs, metas = dist
    by = {4: [m for m in metas if m["connectivity"] == 4], 8: [m for m in metas if m["connectivity"] == 8]}
    a, b = by[4], by[8]
    assert min(len(a), len(b)) >= 400
    for key_fn, tol in (
        (lambda m: m["n_clusters"], 0.08),
        (lambda m: m["texture_group"], 0.08),
        (lambda m: m["n"] == 1, 0.05),
        (lambda m: min(4, m["n_matches"]), 0.08),
        (lambda m: bool(m["touch"]), 0.06),
        (lambda m: min(5, m["palette_size"]), 0.08),
    ):
        sa, sb = _shares(a, key_fn), _shares(b, key_fn)
        for k in set(sa) | set(sb):
            assert abs(sa.get(k, 0) - sb.get(k, 0)) <= tol, (k, sa.get(k, 0), sb.get(k, 0))
    for key_fn, tol in (
        (lambda m: [len(p) for p in m["cluster_pieces"]], 0.10),
        (lambda m: m["kinds"], 0.10),
        (lambda m: m["cluster_options"], 0.10),
    ):
        ca = Counter(x for m in a for x in key_fn(m))
        cb = Counter(x for m in b for x in key_fn(m))
        na, nb = max(1, sum(ca.values())), max(1, sum(cb.values()))
        for k in set(ca) | set(cb):
            assert abs(ca[k] / na - cb[k] / nb) <= tol, (k, ca[k] / na, cb[k] / nb)
    # visible pieces (a cluster is one object under 8 but shows its pieces either way)
    def pieces(m):
        return m["n_objects"] + (sum(len(p) - 1 for p in m["cluster_pieces"]) if m["connectivity"] == 8 else 0)
    assert abs(np.mean([pieces(m) for m in a]) - np.mean([pieces(m) for m in b])) <= 1.0
    assert abs(np.mean([m["fg_fraction"] for m in a]) - np.mean([m["fg_fraction"] for m in b])) <= 0.03
    assert abs(np.mean([m["grid_h"] * m["grid_w"] for m in a]) - np.mean([m["grid_h"] * m["grid_w"] for m in b])) <= 40
    assert abs(np.mean([m["n"] for m in a]) - np.mean([m["n"] for m in b])) <= 1.0
    for rows in (a, b):
        assert np.mean([m["differs_under_other_connectivity"] for m in rows]) >= 0.40
        # the flip-inducing cluster options always flip the answer
        for m in rows:
            if any(o in ("piece_n", "sum_n") for o in m["cluster_options"]):
                assert m["differs_under_other_connectivity"], m


def test_siblings_byte_identical():
    expected = json.load(open(FIXTURE))
    # Task 29 was deliberately rebuilt later; its literal rule and distribution
    # now have their own regression suite. Keep the other historical pins.
    for tid in ("70", "84", "85"):
        for s, digest in enumerate(expected[tid]):
            got = hashlib.sha1(json.dumps(generate(tid, s), sort_keys=True).encode()).hexdigest()
            assert got == digest, f"sibling task {tid} changed at seed {s}"
