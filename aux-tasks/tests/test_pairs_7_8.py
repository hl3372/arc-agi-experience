"""Tests for the overhauled tasks 7 / 8 (shape-pair matching up to rotation).

Two answer modes: ``recolor`` (grid -> grid, every matched object painted with the
highlight colour) and ``pairs`` (grid -> text, objects labelled A, B, C ... in reading
order of their first cell, unordered pairs listed sorted). An INDEPENDENT oracle (own
BFS, own rotation canonicalisation, own labelling) re-solves every example from the
visible input; distribution assertions encode the review targets; sibling generators
in the same file are checked byte-for-byte against hashes from the original code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompts import ISOLATED_CELL_RULE, get_prompt  # noqa: E402
from registry import generate  # noqa: E402
from sft_wrapper import render_input, render_output  # noqa: E402

LETTERS = list("ABCDEFGHIJ")

N_ORACLE = 500
N_DIST = 800
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sibling_hashes_color_filtering.json")


# --------------------------------------------------------------------------- oracle

def _components(a: np.ndarray, bg: int, connectivity: int):
    h, w = a.shape
    seen = np.zeros_like(a, dtype=bool)
    d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    deltas = d4 if connectivity == 4 else d4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or a[r, c] == bg:
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


def _rotation_key(cells):
    pts = list(cells)
    best = None
    for _ in range(4):
        mr = min(r for r, _ in pts)
        mc = min(c for _, c in pts)
        key = tuple(sorted((r - mr, c - mc) for r, c in pts))
        if best is None or key < best:
            best = key
        pts = [(c, -r) for r, c in pts]      # rotate 90 degrees
    return best


def _mirror_key(cells):
    return _rotation_key([(r, -c) for r, c in cells])


def _isolated(a: np.ndarray) -> np.ndarray:
    """Own implementation of the isolated-cell rule: no 8-neighbour of the same colour."""
    h, w = a.shape
    iso = np.ones_like(a, dtype=bool)
    for r in range(h):
        for c in range(w):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w and a[r + dr, c + dc] == a[r, c]:
                        iso[r, c] = False
    return iso


def _object_view(a: np.ndarray, bg: int, bg_colors) -> np.ndarray:
    """What the prompt lets a solver do: named background colours -> bg, then isolated
    single cells of any colour -> bg."""
    view = np.where(np.isin(a, list(bg_colors)), bg, a)
    view = np.where(_isolated(view) & (view != bg), bg, view)
    return view




def _inferred_background(grid: np.ndarray, named: set, margin: float = 1.25):
    """Own implementation of the policy reading: set aside the named texture colours and
    every isolated single cell, take the most frequent remaining colour; it must beat the
    runner-up by ``margin`` and own the largest 4-connected region.  Returns (colour, ok)."""
    consider = ~_isolated(grid)
    if named:
        consider &= ~np.isin(grid, list(named))
    counts = np.bincount(grid[consider].ravel(), minlength=10)
    order = np.argsort(-counts, kind="stable")
    color, runner = int(order[0]), int(order[1])
    # named texture cells are background in the prompt's reading, so for the region test
    # they join the candidate colour and bridge regions split by bands / grid lines
    if named:
        grid = np.where(np.isin(grid, list(named)), color, grid)
        consider = consider | np.isin(grid, [color])
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    best_size, best_color = 0, -1
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not consider[r, c]:
                continue
            col = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            n = 0
            while stack:
                a, b = stack.pop()
                n += 1
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < H and 0 <= nb < W and not seen[na, nb] and consider[na, nb] and grid[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            if n > best_size:
                best_size, best_color = n, col
    ok = counts[color] > counts[runner] and counts[color] >= margin * counts[runner] and best_color == color
    return color, bool(ok)


def _solve(ex: dict, by_color: bool, structural: bool = True):
    a = np.array(ex["input"])
    m = ex["metadata"]
    bg = m["background_color"]
    # structural: only the NAMED texture colours are background (sparse ones must be
    # inferred through the isolated-cell rule); colour-based: every texture colour is.
    if structural:
        bg_colors = {bg} | (set(m["texture_colors"]) if m["texture_named"] else set())
    else:
        bg_colors = {bg} | set(m["texture_colors"])
    view = _object_view(a, bg, bg_colors)
    comps = _components(view, bg, m["connectivity"])
    assert all(len(cells) >= 2 for _, cells in comps)
    assert 2 <= len(comps) <= 26
    order = sorted(range(len(comps)), key=lambda i: min(comps[i][1]))   # reading order of first cell
    label = {idx: chr(65 + rank) for rank, idx in enumerate(order)}
    groups = {}
    for idx, (col, cells) in enumerate(comps):
        key = (_rotation_key(cells), col) if by_color else _rotation_key(cells)
        groups.setdefault(key, []).append(idx)
    pairs, matched = [], set()
    for g in groups.values():
        if len(g) >= 2:
            pairs += list(combinations(sorted(label[i] for i in g), 2))
            for i in g:
                matched.update(comps[i][1])
    pairs.sort()
    if m["answer_mode"] == "pairs":
        answer = "[" + ",".join(f"({x},{y})" for x, y in pairs) + "]"
        assert answer == ex["output"], (answer, ex["output"])
    else:
        assert m["highlight_color"] not in {int(v) for v in np.unique(a)}
        out = a.copy()
        for r, c in matched:
            out[r, c] = m["highlight_color"]
        assert out.tolist() == ex["output"]
    return comps, groups, pairs


# --------------------------------------------------------------------------- tests

def test_oracle_recomputes_every_answer():
    for tid, by_color in (("7", False), ("8", True)):
        for s in range(N_ORACLE):
            ex = generate(tid, s)
            comps, groups, pairs = _solve(ex, by_color)
            # the colour-based reading (every texture colour is background) gives the same answer
            comps_c, _groups_c, pairs_c = _solve(ex, by_color, structural=False)
            assert pairs_c == pairs and len(comps_c) == len(comps), f"task {tid} seed {s}"
            m = ex["metadata"]
            assert m["n_pairs"] == len(pairs) and m["has_pair"] == bool(pairs)
            a = np.array(ex["input"])
            assert a.min() >= 0 and a.max() <= 9
            if m["answer_mode"] == "recolor":
                assert a.shape[0] * a.shape[1] <= 900
                assert ex["input"] != ex["output"] or not pairs
            else:
                assert isinstance(ex["output"], str)


def test_mirror_images_never_match_and_teach_pairs_show_two_rotations():
    """The prompt says mirrors do not count: hard-negative mirror pairs must stay unmatched,
    and every scene with pairs contains a matched group whose members differ in orientation."""
    mirror_scenes = 0
    for tid, by_color in (("7", False), ("8", True)):
        for s in range(N_DIST):
            ex = generate(tid, s)
            m = ex["metadata"]
            comps, groups, pairs = _solve(ex, by_color)
            if m["hard_negative"] == "mirror":
                mirror_scenes += 1
                keys = [_rotation_key(cells) for _, cells in comps]
                # some object has its mirror image present as a different rotation class
                assert any(_mirror_key(cells) in keys and _mirror_key(cells) != _rotation_key(cells)
                           for _, cells in comps)
            if pairs:
                teach = False
                for g in groups.values():
                    if len(g) >= 2:
                        norms = set()
                        for i in g:
                            cells = comps[i][1]
                            mr = min(r for r, _ in cells)
                            mc = min(c for _, c in cells)
                            norms.add(tuple(sorted((r - mr, c - mc) for r, c in cells)))
                        teach |= len(norms) >= 2
                assert teach, f"task {tid} seed {s}: no pair shows two different rotations"
            if m["connectivity"] == 8 and pairs:
                diag_groups = 0
                shape = np.array(ex["input"]).shape
                for g in groups.values():
                    if len(g) >= 2:
                        needs_diag = False
                        for i in g:      # one mask per member: touching neighbours must not merge
                            sub = np.zeros(shape, dtype=int)
                            for r, c in comps[i][1]:
                                sub[r, c] = 1
                            needs_diag |= len(_components(sub, 0, 4)) > 1
                        diag_groups += int(needs_diag)
                assert diag_groups == 1, f"task {tid} seed {s}: expected exactly one diagonal-link pair"
    assert mirror_scenes >= 20


def test_determinism():
    for tid in ("7", "8"):
        for s in range(40):
            assert generate(tid, s) == generate(tid, s)


def test_prompts_state_the_conventions():
    for tid in ("7", "8"):
        for s in range(80):
            ex = generate(tid, s)
            m = ex["metadata"]
            prompt = get_prompt(tid, m)
            assert "mirror images do not count" in prompt
            assert "at least two cells" in prompt and ISOLATED_CELL_RULE in prompt
            assert ("horizontally, vertically, or diagonally" in prompt) == (m["connectivity"] == 8)
            if m["answer_mode"] == "pairs":
                assert "reading order" in prompt and "[]" in prompt and "alphabetical" in prompt
                assert "highlight_color" not in m
            else:
                assert str(m["highlight_color"]) in prompt and "Recolor" in prompt
            if tid == "8":
                assert "same color" in prompt
            else:
                assert "colors do not matter" in prompt


def test_token_budget_compact_format():
    """Sizes are not capped for the SFT bundle (task design is decoupled from the budget);
    this records that in the standard compact grid format every example still fits 2900 tokens."""
    for tid in ("7", "8"):
        for s in range(N_DIST):
            ex = generate(tid, s)
            chars = (len(render_input(tid, ex["input"], fmt="compact"))
                     + len(render_output(ex["output"], task_id=tid, fmt="compact"))
                     + len(get_prompt(tid, ex["metadata"])) + 50)
            assert chars / 1.2 <= 2900, f"task {tid} seed {s}: over the token budget"


def test_textures_follow_the_policy():
    """Sparse textures are inferred (never named, isolated cells, <= 20% of free cells);
    solid / structured textures are named; texture colours never collide with the
    background, an object colour or the highlight; textures never touch objects; the
    palette / colour distributions reach the agreed ranges."""
    for tid in ("7", "8"):
        exs = [generate(tid, s) for s in range(N_DIST)]
        groups = Counter(e["metadata"]["texture_group"] for e in exs)
        assert 0.18 <= 1 - groups[None] / N_DIST <= 0.35, groups
        assert groups["sparse"] >= 0.08 * N_DIST and groups["solid"] >= 0.05 * N_DIST and groups["structured"] >= 5, groups
        # colours: near-flat palette draw over 2..8; the INPUT shows up to 9 colours (the
        # highlight must be absent) and the output up to 10.  The recolour removes the
        # matched objects' colours, so the output count is the lower of the two.
        n_in = Counter(e["metadata"]["n_colors_input"] for e in exs)
        n_out = Counter(e["metadata"]["n_colors_output"] for e in exs)
        assert sum(v for k, v in n_in.items() if k >= 8) >= 0.05 * N_DIST, n_in
        assert sum(v for k, v in n_in.items() if k >= 7) >= 0.12 * N_DIST, n_in
        assert sum(v for k, v in n_in.items() if k <= 4) >= 0.20 * N_DIST, n_in
        assert max(v for k, v in n_in.items()) <= 0.40 * N_DIST, n_in   # no dominant colour count
        assert max(n_in) >= 9 and max(n_out) >= 9, (n_in, n_out)
        for s, ex in enumerate(exs):
            m = ex["metadata"]
            a = np.array(ex["input"])
            prompt = get_prompt(tid, m, color_palette=LETTERS)
            assert not re.search(r"color \d", prompt), prompt
            # policy: the background is never named or defined; the prompt only says to ignore it
            assert "background is color" not in prompt and "background color" not in prompt, prompt
            assert prompt.count("Ignore the background, whether it is a solid color or a pattern.") == 1, prompt
            # identifiability: the most frequent colour (named textures and isolated cells set
            # aside) is the generator's background, by a clear margin, as the largest region
            named = set(m["texture_colors"]) if m["texture_named"] else set()
            color, ok = _inferred_background(a, named)
            assert ok and color == m["background_color"], f"task {tid} seed {s}: background not identifiable"
            assert m["background_margin"] >= 1.25, (tid, s)
            for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
                assert key in m, key
            tex = m["texture_colors"]
            if m["texture_group"] is None:
                assert tex == [] and not m["texture_named"] and m["texture_coverage"] == 0
                assert "background texture, not objects" not in prompt
                continue
            assert min(a.shape) >= 8 and tex
            assert m["background_color"] not in tex and m["highlight_color"] not in tex
            objects = {int(v) for v in np.unique(_object_view(a, m["background_color"], {m["background_color"]} | set(tex)))} - {m["background_color"]}
            assert not (objects & set(tex)), f"task {tid} seed {s}: texture colour used by an object"
            if m["texture_group"] == "sparse":
                assert not m["texture_named"]
                assert "background texture, not objects" not in prompt
                mask = np.isin(a, tex)
                assert _isolated(np.where(mask, a, -1))[mask].all(), f"task {tid} seed {s}: sparse texture not isolated"
                free = (a == m["background_color"]).sum() + mask.sum()
                assert mask.sum() <= 0.20 * free + 1e-9
            else:
                # solid / structured kinds are still generated, but nothing is named in the prompt
                for t in tex:
                    assert f"color {LETTERS[t]}" not in prompt
                assert ("background texture, not objects" not in prompt
                        and "Ignore the background, whether it is a solid color or a pattern. Treat" in prompt)
                if m["texture_group"] == "structured":
                    assert min(a.shape) >= 12 and m["texture_kind"] == "mosaic"


def _dist_checks(tid: str):
    exs = [generate(tid, s) for s in range(N_DIST)]
    metas = [e["metadata"] for e in exs]
    hs = [len(e["input"]) for e in exs]
    ws = [len(e["input"][0]) for e in exs]
    mins = [min(h, w) for h, w in zip(hs, ws)]
    # answer modes
    modes = Counter(m["answer_mode"] for m in metas)
    assert modes["recolor"] == N_DIST and not modes["pairs"]  # pair-list mode removed on review feedback
    # sizes: independent H, W, short side >= 5, both tall-narrow and short-wide grids present
    assert min(mins) >= 5 and max(hs) == 30 and max(ws) == 30
    assert np.mean([m <= 10 for m in mins]) >= 0.40
    assert np.percentile(hs, 25) <= 12 and np.percentile(hs, 75) >= 18
    assert np.percentile(ws, 25) <= 10 and np.percentile(ws, 75) >= 18
    assert np.mean([h >= 23 and w <= 12 for h, w in zip(hs, ws)]) >= 0.03
    assert np.mean([w >= 23 and h <= 12 for h, w in zip(hs, ws)]) >= 0.03
    assert np.mean([h == w for h, w in zip(hs, ws)]) <= 0.15  # no forced square share
    # answer spread
    n_pairs = [m["n_pairs"] for m in metas]
    assert 0.06 <= np.mean([p == 0 for p in n_pairs]) <= 0.18
    assert 0.22 <= np.mean([p == 1 for p in n_pairs]) <= 0.38
    assert 0.20 <= np.mean([p == 2 for p in n_pairs]) <= 0.36
    assert 0.22 <= np.mean([p >= 3 for p in n_pairs]) <= 0.38
    assert np.mean([3 in m["group_sizes"] for m in metas]) >= 0.10
    assert np.mean([m["n_matched"] <= 0.6 * m["n_objects"] for m in metas]) >= 0.70
    # object counts: the cap grows with the canvas (8 on a short side <= 8, 12 under 400
    # cells, 16 under 700, 20 at 700+; the solver labels objects A..Z so never above 26)
    areas = [h * w for h, w in zip(hs, ws)]
    assert all(2 <= m["n_objects"] <= 26 for m in metas)
    assert all(m["n_objects"] <= 8 for m, mn in zip(metas, mins) if mn <= 8)
    assert all(m["n_objects"] <= 12 for m, a in zip(metas, areas) if a < 400)
    assert all(m["n_objects"] <= 16 for m, a in zip(metas, areas) if a < 700)
    assert all(m["n_objects"] <= 20 for m in metas)
    big = [m["n_objects"] for m, a in zip(metas, areas) if a >= 700]
    # ~5% of the 800-seed sweep lands on 700+ cells
    assert len(big) >= 25 and np.median(big) >= 10 and max(big) >= 16, (len(big), np.median(big), max(big))
    # large canvases carry large scenes: object occupancy (texture excluded) holds with area
    occ = {"small": [], "large": []}
    for e, m, a in zip(exs, metas, areas):
        g = np.array(e["input"])
        bgset = {m["background_color"]} | set(m["texture_colors"])
        cells = int(sum((g == c).sum() for c in range(10) if c not in bgset))
        (occ["small"] if a < 100 else occ["large"] if a >= 700 else []).append(cells / a)
    assert np.median(occ["large"]) >= 0.22, np.median(occ["large"])   # 10k-seed figure ~0.27 (was 0.13)
    assert np.percentile(occ["large"], 90) <= 0.50, np.percentile(occ["large"], 90)
    assert np.median(occ["small"]) >= 0.24, np.median(occ["small"])   # small grids unchanged (~0.30)
    # objects: real size range and family variety
    sizes = [s for m in metas for s in m["object_sizes"]]
    assert np.percentile(sizes, 90) >= 12 and max(sizes) >= 25
    assert np.mean([s >= 10 for s in sizes]) >= 0.15
    fams = Counter(f for m in metas for f in m["families"])
    assert len([f for f in fams if not f.endswith("+diag")]) >= 10, fams
    # 8-connected scenes: diagonal links on roughly a third of the objects, never in 4-conn scenes
    m8 = [m for m in metas if m["connectivity"] == 8]
    m4 = [m for m in metas if m["connectivity"] == 4]
    assert all(m["diag_link_objects"] == 0 for m in m4)
    share = sum(m["diag_link_objects"] for m in m8) / sum(m["n_objects"] for m in m8)
    assert 0.25 <= share <= 0.48, share
    # hard negatives (mirror / near-miss look-alikes) are a regular feature
    assert np.mean([m["hard_negative"] != "none" for m in metas]) >= 0.20
    assert Counter(m["hard_negative"] for m in metas)["mirror"] >= 20
    # every object has at least two cells (the prompt's rule)
    assert all(min(m["object_sizes"]) >= 2 for m in metas)
    return metas


def test_distribution_task_7():
    metas = _dist_checks("7")
    assert all(m["mismatch_copies"] == 0 for m in metas)


def test_distribution_task_8():
    metas = _dist_checks("8")
    with_groups = [m for m in metas if m["group_sizes"]]
    assert np.mean([m["mismatch_copies"] >= 1 for m in with_groups]) >= 0.60
    no_groups = [m for m in metas if not m["group_sizes"]]
    assert np.mean([m["mismatch_copies"] >= 2 for m in no_groups]) >= 0.50
    # the realised colour count follows the palette draw (singleton objects fill the palette
    # when matched groups and copies would otherwise reuse colours): 8-9 input colours >= 12%
    n_in = Counter(m["n_colors_input"] for m in metas)
    assert sum(v for k, v in n_in.items() if k >= 8) >= 0.12 * len(metas), n_in
    assert max(n_in.values()) <= 0.35 * len(metas), n_in


def test_siblings_byte_identical():
    expected = json.load(open(FIXTURE))
    # Task 29 was deliberately rebuilt later and has its own semantic suite.
    for tid in ("70", "84", "85"):
        for s, digest in enumerate(expected[tid]):
            got = hashlib.sha1(json.dumps(generate(tid, s), sort_keys=True).encode()).hexdigest()
            assert got == digest, f"sibling task {tid} changed at seed {s}"
