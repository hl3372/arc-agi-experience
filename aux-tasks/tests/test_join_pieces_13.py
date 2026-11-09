"""Tests for task 13 (``generators/join_pieces.py`` — join the matching jagged pieces).

The core check is an INDEPENDENT brute-force cell-set solver that uses only the visible input
grid plus the conventions the prompt states (which pieces are fixed).  The background is the
most frequent colour; pieces are 4-connected same-colour groups of at least two cells (isolated
single cells are background texture and are ignored).  Under the prompt's strict, side-agnostic
reading — a loose piece arrives from the left, the right, above or below; every facing cell
(no cell of the piece directly beyond it in the direction of approach) has a cell of the fixed
piece directly beyond it; the seam is jagged; pure translation; no overlap — it enumerates every
seating of every loose piece in every direction against every fixed piece and every global
assembly, and requires exactly one assembly that equals the generator's output (texture cells
untouched, vacated cells plain background).

Multi-side docking (the September-2026 follow-up): a fixed piece may carry cavities on two,
three or all four of its sides; the tests pin the shares by canvas bucket, the per-site
direction consistency, corner clearance (docked pieces never touch, not even around a corner),
all-direction uniqueness on every example, the fixed-piece rule (touching any border / the
named colour) identifying the fixed pieces on every example, and the exact wording.

Also covered (September-2026 campaign): near-flat sizes over 6..30, colours 2..10, sparse
textures from the shared library (never named, never touching a piece, identical in input and
output), piece bulk scaling with the canvas, the prompt policy (nothing about the background
named), the generator's own prompt-side re-solver, determinism, performance, sibling pins.

Leftovers (2026-09-07): about 30% of the examples carry extra loose pieces that seat nowhere
(they stay where they are) and about 30% carry a cavity that no loose piece fits (it stays
open).  The solver treats a piece with no seat as staying put; the tests prove the extras seat
nowhere and the open cavities admit nothing, that the prompt never reveals either, and pin the
shares.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backgrounds import SPARSE_KINDS, texture_cells_isolated  # noqa: E402
from common.objects import connected_components  # noqa: E402
from common.backgrounds import isolated_cell_mask  # noqa: E402
from generators.join_pieces import BG_MARGIN, FIXED_SHARE_CAP, resolve_from_prompt  # noqa: E402
from prompts import ISOLATED_CELL_RULE, get_prompt  # noqa: E402
from registry import generate  # noqa: E402

N_SOLVE = 2000
N_DIST = 2000
LETTERS = list("ABCDEFGHIJ")
_CACHE: dict[int, dict] = {}


def _ex(seed: int) -> dict:
    if seed not in _CACHE:
        _CACHE[seed] = generate("13", seed)
    return _CACHE[seed]


# ---------------------------------------------------------------------------
# independent solver
# ---------------------------------------------------------------------------

# direction of approach -> the offset to the cell "directly beyond" a facing cell
BEYOND = {"left": (0, 1), "right": (0, -1), "top": (1, 0), "bottom": (-1, 0)}


def _touches_border(cells, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _pieces(grid: np.ndarray, bg: int) -> list[tuple[int, frozenset]]:
    """Pieces = 4-connected same-colour groups of >= 2 cells; isolated cells are texture."""
    return [(c["color"], frozenset(c["cells"]))
            for c in connected_components(grid, connectivity=4, background=bg) if len(c["cells"]) >= 2]


def _fits_from(loose: frozenset, anchor: frozenset, all_anchor: frozenset, h: int, w: int,
               direction: str) -> list[frozenset]:
    """Every translation of *loose* that seats it flush against *anchor* arriving from
    *direction*: each facing cell (no cell of the piece directly beyond it) has a cell of
    *anchor* directly beyond it, the facing cells do not all lie on one line perpendicular to
    the approach (a jagged seam), the piece stays inside the grid and overlaps no fixed cell.
    Exhaustive: the first facing cell must sit next to some anchor cell, which fixes the shift."""
    dr, dc = BEYOND[direction]
    facing = [(r, c) for (r, c) in loose if (r + dr, c + dc) not in loose]
    if direction in ("left", "right"):
        if len({c for _, c in facing}) < 2:
            return []
    elif len({r for r, _ in facing}) < 2:
        return []
    r0, c0 = facing[0]
    out: dict[tuple[int, int], frozenset] = {}
    for ar, ac in anchor:
        dy, dx = ar - dr - r0, ac - dc - c0
        if (dy, dx) in out:
            continue
        if not all((r + dy + dr, c + dx + dc) in anchor for r, c in facing):
            continue
        placed = frozenset((r + dy, c + dx) for r, c in loose)
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in placed):
            continue
        if placed & all_anchor:
            continue
        out[(dy, dx)] = placed
    return list(out.values())


def _fits(loose: frozenset, anchor: frozenset, all_anchor: frozenset, h: int, w: int) -> list[tuple[str, frozenset]]:
    """All distinct seatings (cell sets) of *loose* against *anchor* over the four directions,
    each with the set of directions it can be reached from (a piece deep in a cavity may be
    flush from two directions at once; that is one seating, not two)."""
    seen: dict[frozenset, set] = {}
    for direction in ("left", "right", "top", "bottom"):
        for placed in _fits_from(loose, anchor, all_anchor, h, w, direction):
            seen.setdefault(placed, set()).add(direction)
    return [(frozenset(d), placed) for placed, d in seen.items()]


def _count_assemblies(fits_per_loose: list[list[frozenset]], blocked: frozenset = frozenset(),
                      cap: int = 3) -> list[list[frozenset | None]]:
    """Every global assembly (at most *cap*): a piece with no seat stays put (None); seats may
    not overlap each other or *blocked* (the staying pieces' cells)."""
    found: list[list[frozenset | None]] = []

    def rec(i: int, chosen: list[frozenset | None]) -> None:
        if len(found) >= cap:
            return
        if i == len(fits_per_loose):
            found.append(list(chosen))
            return
        for placed in (fits_per_loose[i] or [None]):
            if placed is not None and (placed & blocked or any(p is not None and placed & p for p in chosen)):
                continue
            chosen.append(placed)
            rec(i + 1, chosen)
            chosen.pop()

    rec(0, [])
    return found


def _solve(ex: dict) -> dict:
    m = ex["metadata"]
    inp = np.array(ex["input"])
    out = np.array(ex["output"])
    bg = _background(inp)
    h, w = inp.shape
    pieces = _pieces(inp, bg)
    if m["anchor_rule"] == "border":
        anchors = [(col, cells) for col, cells in pieces if _touches_border(cells, h, w)]
    else:
        anchors = [(col, cells) for col, cells in pieces if col == int(m["anchor_color"])]
    anchor_ids = {id(cells) for _, cells in anchors}
    loose = [(col, cells) for col, cells in pieces if id(cells) not in anchor_ids]
    all_anchor = frozenset().union(*[cells for _, cells in anchors]) if anchors else frozenset()
    fits_per_loose = []
    seatings = []          # per loose piece: [(anchor index, direction, placed cells)]
    for _, cells in loose:
        fl: list[frozenset] = []
        st = []
        for ai, (_, acells) in enumerate(anchors):
            for dirs, placed in _fits(cells, acells, all_anchor, h, w):
                fl.append(placed)
                st.append((ai, dirs, placed))
        fits_per_loose.append(fl)
        seatings.append(st)
    stay_cells = frozenset().union(*[cells for (_, cells), fl in zip(loose, fits_per_loose) if not fl]) if loose else frozenset()
    assemblies = _count_assemblies(fits_per_loose, stay_cells)
    matches = 0
    for asm in assemblies:
        g = inp.copy()
        for (_, cells), placed in zip(loose, asm):     # vacated cells become plain background
            if placed is not None:
                for r, c in cells:
                    g[r, c] = bg
        for (col, _), placed in zip(loose, asm):
            if placed is not None:
                for r, c in placed:
                    g[r, c] = col
        if np.array_equal(g, out):
            matches += 1
    return {
        "bg": bg, "n_anchors": len(anchors), "n_loose": len(loose), "fits": [len(f) for f in fits_per_loose],
        "n_stay": sum(1 for f in fits_per_loose if not f),
        "n_assemblies": len(assemblies), "output_matches": matches, "seatings": seatings,
        "anchors": anchors, "loose": loose,
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_independent_solver_unique_answer():
    for seed in range(N_SOLVE):
        ex = _ex(seed)
        res = _solve(ex)
        m = ex["metadata"]
        assert res["bg"] == m["bg_color"], (seed, res)
        assert res["n_anchors"] == m["n_anchors"] and res["n_loose"] == m["n_loose"], (seed, res)
        assert res["n_loose"] >= 1, seed
        # every loose piece seats in exactly one way over all four directions and all fixed
        # pieces, except the extras, which seat nowhere (and are exactly the metadata's extras)
        assert all(f in (0, 1) for f in res["fits"]), (seed, res["fits"])
        assert res["n_stay"] == m["n_extra"] and res["n_loose"] - res["n_stay"] == m["n_docked"] >= 1, seed
        assert res["n_assemblies"] == 1 and res["output_matches"] == 1, (seed, res["n_assemblies"])


def test_prompt_side_resolver_agrees_on_oriented_grids():
    for seed in range(600):
        ex = _ex(seed)
        m = ex["metadata"]
        assert resolve_from_prompt(np.array(ex["input"]), np.array(ex["output"]), m["dock_edge"],
                                   m["anchor_rule"], m.get("anchor_color")), seed
        # a wrong answer is rejected: move one loose cell of the output
        out = np.array(ex["output"])
        inp = np.array(ex["input"])
        changed = np.argwhere(out != inp)
        r, c = changed[0]
        bad = out.copy()
        bad[r, c] = m["bg_color"] if bad[r, c] != m["bg_color"] else (m["bg_color"] + 1) % 10
        assert not resolve_from_prompt(inp, bad, m["dock_edge"], m["anchor_rule"], m.get("anchor_color")), seed


def test_determinism():
    for seed in range(60):
        a, b = generate("13", seed), generate("13", seed)
        assert a["input"] == b["input"] and a["output"] == b["output"] and a["metadata"] == b["metadata"]


def test_grid_bounds_values_and_background():
    for seed in range(N_DIST):
        ex = _ex(seed)
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        m = ex["metadata"]
        h, w = inp.shape
        assert inp.shape == out.shape and 6 <= h <= 30 and 6 <= w <= 30 and h * w <= 900, (seed, inp.shape)
        assert list(m["grid_size"]) == [h, w]
        assert inp.min() >= 0 and inp.max() <= 9 and out.min() >= 0 and out.max() <= 9
        assert _background(inp) == m["bg_color"], seed
        assert m["bg_style"] == (m["texture_kind"] or "uniform")
        assert m["n_colors"] == len(np.unique(inp))


def test_pure_translation_fixed_pieces_unchanged_no_touching():
    for seed in range(600):
        ex = _ex(seed)
        m = ex["metadata"]
        bg = m["bg_color"]
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        pin, pout = _pieces(inp, bg), _pieces(out, bg)
        # every piece is a translation-invariant match between input and output (same colour)
        def norm(cells):
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            return frozenset((r - r0, c - c0) for r, c in cells)
        assert Counter((col, norm(c)) for col, c in pin) == Counter((col, norm(c)) for col, c in pout), seed
        assert len(pin) == m["n_anchors"] + m["n_loose"]
        # fixed pieces are identical in input and output; loose pieces moved and left background
        h, w = inp.shape
        if m["anchor_rule"] == "border":
            anchors = [c for col, c in pin if _touches_border(c, h, w)]
            loose = [c for col, c in pin if not _touches_border(c, h, w)]
        else:
            anchors = [c for col, c in pin if col == m["anchor_color"]]
            loose = [c for col, c in pin if col != m["anchor_color"]]
        assert len(anchors) == m["n_anchors"] and len(loose) == m["n_loose"], seed
        for cells in anchors:
            assert all(out[r, c] == inp[r, c] for r, c in cells), seed
        # the extras stay exactly where they are; every other loose piece moved
        extras = [c for c in loose if all(out[r, cc] == inp[r, cc] for r, cc in c)]
        assert len(extras) == m["n_extra"], seed
        moved = frozenset().union(*[c for c in loose if not any(c is e for e in extras)]) if len(loose) > len(extras) else frozenset()
        stay = frozenset().union(*anchors) | (frozenset().union(*extras) if extras else frozenset())
        assert moved, seed
        assert all(out[r, c] == bg or out[r, c] != inp[r, c] or (r, c) in stay for r, c in moved), seed
        # every cell that is not a piece in either grid is identical (texture stays put)
        piece_cells = moved | stay | frozenset().union(*[c for _, c in pout])
        rest = np.ones(inp.shape, dtype=bool)
        for r, c in piece_cells:
            rest[r, c] = False
        assert np.array_equal(inp[rest], out[rest]), seed
        # no two pieces touch (8-neighbourhood) in the input
        ids = np.full(inp.shape, -1)
        for i, (_, cells) in enumerate(pin):
            for r, c in cells:
                ids[r, c] = i
        for r in range(h):
            for c in range(w):
                if ids[r, c] < 0:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and ids[rr, cc] >= 0:
                            assert ids[rr, cc] == ids[r, c], (seed, r, c)


def test_anchor_rule_identifies_the_fixed_pieces_on_every_example():
    """Border rule: exactly the fixed pieces touch a border of the grid (any border), so no
    loose piece ever does; colour rule: exactly the fixed pieces carry the named colour."""
    seen_rules = set()
    for seed in range(N_DIST):
        ex = _ex(seed)
        m = ex["metadata"]
        seen_rules.add(m["anchor_rule"])
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        pin = _pieces(inp, m["bg_color"])
        h, w = inp.shape
        side = m["fixed_side"]
        assert side == {"right": "left", "left": "right", "top": "bottom", "bottom": "top"}[m["dock_edge"]]
        assert m["loose_edge"] == side
        moved = frozenset().union(*[cells for _, cells in pin]) - frozenset(map(tuple, np.argwhere(inp == out).tolist()))
        if m["anchor_rule"] == "border":
            touching = [cells for _, cells in pin if _touches_border(cells, h, w)]
            assert len(touching) == m["n_anchors"], seed
            assert all(not (cells & moved) for cells in touching), seed       # touching pieces stay
            unmoved_loose = [cells for _, cells in pin if not _touches_border(cells, h, w) and not (cells & moved)]
            assert len(unmoved_loose) == m["n_extra"], seed                   # only the extras stay
            assert "anchor_color" not in m
        else:
            cols = Counter(col == m["anchor_color"] for col, _ in pin)
            assert cols[True] == m["n_anchors"] and cols[False] == m["n_loose"], seed
            assert m["anchor_color"] != m["bg_color"]
            assert all(not (cells & moved) for col, cells in pin if col == m["anchor_color"]), seed
    assert seen_rules == {"border", "color"}


def test_textures_follow_the_policy():
    """Sparse textures only, never named, never touching a piece, identical in input and output."""
    textured = 0
    kinds = Counter()
    for seed in range(N_DIST):
        ex = _ex(seed)
        m = ex["metadata"]
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        for key in ("texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage"):
            assert key in m, key
        assert not m["texture_named"]
        if m["texture_group"] is None:
            assert m["texture_kind"] is None and m["texture_colors"] == [] and m["texture_coverage"] == 0.0
            assert len(np.unique(inp)) == len(set(m["piece_colors"])) + 1
            continue
        textured += 1
        kinds[m["texture_kind"]] += 1
        assert m["texture_group"] == "sparse" and m["texture_kind"] in SPARSE_KINDS
        assert min(inp.shape) >= 8
        tex = set(m["texture_colors"])
        assert tex and not (tex & set(m["piece_colors"])) and m["bg_color"] not in tex
        assert texture_cells_isolated(inp, m["texture_colors"])
        tmask = np.isin(inp, m["texture_colors"])
        assert tmask.any() and np.array_equal(inp[tmask], out[tmask])
        assert 0.0 < m["texture_coverage"] <= 0.20 + 1e-9
        # never 8-adjacent to a piece, in the input or in the output
        for grid in (inp, out):
            piece = (grid != m["bg_color"]) & ~np.isin(grid, m["texture_colors"])
            dil = piece.copy()
            dil[1:, :] |= piece[:-1, :]; dil[:-1, :] |= piece[1:, :]
            dil[:, 1:] |= piece[:, :-1]; dil[:, :-1] |= piece[:, 1:]
            dil[1:, 1:] |= piece[:-1, :-1]; dil[1:, :-1] |= piece[:-1, 1:]
            dil[:-1, 1:] |= piece[1:, :-1]; dil[:-1, :-1] |= piece[1:, 1:]
            assert not (dil & tmask).any(), seed
    assert 0.18 <= textured / N_DIST <= 0.36, textured / N_DIST
    assert len(kinds) >= 6, kinds


def test_prompt_policy_and_specificity():
    for seed in range(300):
        ex = _ex(seed)
        m = ex["metadata"]
        p = get_prompt("13", m, color_palette=LETTERS)
        assert not re.search(r"color \d", p), p
        assert "background of color" not in p and "background is" not in p and "background color" not in p
        assert "at least two same-colored cells" in p and ISOLATED_CELL_RULE in p
        # side-agnostic but exact: the four directions of approach and the flush + jagged seam
        assert "arriving from the left, from the right, from above or from below" in p
        assert "every cell of the loose piece that faces the fixed piece" in p
        assert "has a cell of that fixed piece directly beyond it" in p
        assert "the seam is jagged" in p and "do not all lie in one row when arriving from above or below, or in one column when arriving from the left or right" in p
        assert f"its {m['loose_edge']} edge" not in p and f"the {m['dock_edge']} edge" not in p
        assert " edge" not in p and "border" not in p.replace("touch the border of the grid", "")
        if m["anchor_rule"] == "color":
            assert f"pieces of color {LETTERS[m['anchor_color']]} are fixed" in p
        else:
            assert "The pieces that touch the border of the grid are fixed and never move; every other piece is loose." in p
        words = {1: "one of its sides", 2: "two of its sides", 3: "three of its sides", 4: "all four of its sides"}
        if m["variant"] == "host":
            assert f"The fixed piece has {m['n_sites']} jagged cavities in total, spread over {words[m['max_sides']]}, and each cavity receives at most one loose piece." in p
        elif m["variant"] == "mixed":
            assert "A fixed piece may have several jagged cavities, possibly on different sides, and each cavity receives at most one loose piece." in p
        else:
            assert "Each fixed piece receives at most one loose piece." in p
        # leftovers: never "exactly one loose piece", never "every loose piece is used"; the
        # standing sentence covers extras and open cavities on every example
        assert "exactly one loose piece" not in p and "used exactly once" not in p and "at least one" not in p
        assert "A loose piece that seats nowhere stays exactly where it is, and a cavity that no loose piece fits stays open." in p
        assert "do not rotate or reflect" in p and "become plain background" in p
        assert "A moved piece may not overlap any other piece." in p
        assert "one connected block of a single color" in p and "solid block" not in p
        assert " - " not in p                      # no spaced-hyphen parentheticals
        for c in m["texture_colors"]:
            assert f"color {LETTERS[c]}" not in p


def test_size_and_colour_distributions():
    sides: list[int] = []
    colours = Counter()
    squares = 0
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        h, w = m["grid_size"]
        sides += [h, w]
        squares += h == w
        colours[m["n_colors"]] += 1
    per_value = Counter(sides)
    assert set(per_value) == set(range(6, 31))
    dens = np.array([per_value[v] for v in range(6, 31)], dtype=float)
    assert dens.max() / dens.min() <= 1.9, per_value          # near-flat: no band spike
    assert squares / N_DIST <= 0.10
    assert max(colours) >= 9 and colours[2] == 0 and colours[3] > 0
    assert sum(v for k, v in colours.items() if k >= 6) / N_DIST >= 0.12, colours
    assert max(colours.values()) / N_DIST <= 0.42, colours


def test_pieces_scale_with_the_canvas():
    small: dict[str, list[int]] = {"t": [], "s": [], "r": []}
    large: dict[str, list[int]] = {"t": [], "s": [], "r": []}
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        h, w = m["grid_size"]
        tgt = small if h * w < 150 else large if h * w >= 600 else None
        if tgt is None:
            continue
        tgt["t"] += m["loose_thickness"]
        tgt["s"] += m["spans"]
        tgt["r"] += m["reliefs"]
    assert len(large["t"]) >= 200
    assert np.median(large["t"]) >= 2.5 and np.percentile(large["t"], 90) >= 4, np.median(large["t"])
    assert np.percentile(large["s"], 90) >= 7 and max(large["s"]) >= 12
    assert np.percentile(large["r"], 90) >= 5 and max(large["r"]) >= 7
    assert np.median(small["t"]) <= 2 and np.median(small["s"]) <= 4     # small canvases stay feasible
    # overall: single-cell-thick loose pieces are no longer the majority
    all_t = small["t"] + large["t"]
    assert Counter(all_t)[1] / len(all_t) <= 0.45


def test_structure_distribution_targets():
    n_sites = Counter()
    styles = Counter()
    reliefs = Counter()
    variants = Counter()
    docks = Counter()
    schemes = Counter()
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        n_sites[m["n_sites"]] += 1
        styles.update(m["styles"])
        reliefs.update(m["reliefs"])
        variants[m["variant"]] += 1
        docks[m["dock_edge"]] += 1
        schemes[m["color_scheme"]] += 1
    assert 0.10 <= n_sites[1] / N_DIST <= 0.25, n_sites
    assert max(n_sites) >= 5
    total_sites = sum(styles.values())
    for style in ("random", "teeth", "staircase", "notch", "vshape", "plateau", "walk"):
        assert styles[style] / total_sites >= 0.08, (style, styles)
    assert sum(v for k, v in reliefs.items() if k >= 3) / total_sites >= 0.35, reliefs
    assert set(variants) == {"pairs", "host", "mixed"} and min(variants.values()) / N_DIST >= 0.08, variants
    assert set(docks) == {"left", "right", "top", "bottom"}
    assert set(schemes) == {"distinct", "two_tone", "uniform_loose"}


def test_metadata_keys():
    required = {"variant", "anchor_rule", "bg_color", "dock_edge", "loose_edge", "fixed_side", "orientation",
                "n_anchors", "n_loose", "n_sites", "sites_per_anchor", "spans", "reliefs", "styles",
                "rect_union", "color_scheme", "grid_size", "bg_style", "n_colors", "piece_colors",
                "texture_group", "texture_kind", "texture_colors", "texture_named", "texture_coverage",
                "size_draw", "caps", "dock_sides_per_anchor", "n_sides_per_anchor", "max_sides",
                "multi_side", "piece_sides", "n_docked", "n_open_cavities", "site_open", "open_sites",
                "open_cavity_sides", "open_cavity_cells", "n_extra", "extra_kinds", "extra_sizes",
                "extra_piece_colors"}
    for seed in range(100):
        m = _ex(seed)["metadata"]
        assert required <= set(m), required - set(m)
        # the per-site lists describe every cavity (open ones included); n_loose counts the
        # loose pieces visible in the input: the docked ones plus the extras
        assert len(m["styles"]) == len(m["spans"]) == len(m["reliefs"]) == len(m["site_open"]) == m["n_sites"]
        assert m["n_docked"] == m["n_sites"] - m["n_open_cavities"] >= 1
        assert m["n_loose"] == m["n_docked"] + m["n_extra"]
        assert sum(m["site_open"]) == m["n_open_cavities"] == len(m["open_sites"]) == len(m["open_cavity_sides"])
        assert len(m["extra_kinds"]) == len(m["extra_sizes"]) == len(m["extra_piece_colors"]) == m["n_extra"] <= 2
        assert set(m["extra_kinds"]) <= {"near_miss", "jagged"} and all(s >= 4 for s in m["extra_sizes"])
        assert (len(m["open_cavity_cells"]) > 0) == (m["n_open_cavities"] > 0)
        assert sum(m["sites_per_anchor"]) == m["n_sites"] and len(m["sites_per_anchor"]) == m["n_anchors"]
        assert len(m["piece_sides"]) == m["n_sites"] and len(m["dock_sides_per_anchor"]) == m["n_anchors"]
        assert m["n_sides_per_anchor"] == [len(s) for s in m["dock_sides_per_anchor"]]
        assert m["max_sides"] == max(m["n_sides_per_anchor"]) and m["multi_side"] == (m["max_sides"] >= 2)
        for sides, k in zip(m["dock_sides_per_anchor"], m["sites_per_anchor"]):
            assert 1 <= len(sides) <= min(4, k) and len(set(sides)) == len(sides)
            assert set(sides) <= {"left", "right", "top", "bottom"}


def test_performance():
    for seed in range(20):  # warm-up
        generate("13", seed)
    times = []
    for seed in range(1000, 1400):
        t0 = time.perf_counter()
        generate("13", seed)
        times.append((time.perf_counter() - t0) * 1000)
    times_a = np.array(times)
    assert times_a.mean() <= 6.0, times_a.mean()
    assert np.percentile(times_a, 99) <= 40.0, np.percentile(times_a, 99)


def test_other_misc_tasks_byte_identical():
    """Every other generator registered in generators/misc.py is unchanged by the task-13 move."""
    path = os.path.join(os.path.dirname(__file__), "fixtures", "misc_sibling_hashes_task13.json")
    with open(path) as f:
        fixture = json.load(f)
    n = int(fixture["n_seeds"])
    for tid, expected in sorted(fixture["sha1"].items()):
        # The later 26–28 rebuild replaces these historical snapshots with
        # independent semantic and coverage tests for those tasks.
        # The 31/35 entries now pin their approved September rebuilds, and the
        # 91/92 entries their approved 2026-09-15 rebuilds; the 92b entry is
        # gone with the id, which is merged into 92.  The 83 entry pins its
        # approved 2026-09-16 background pass (sparse textures).
        if tid in {"26", "27", "28"}:
            continue
        h = hashlib.sha1()
        for seed in range(n):
            h.update(json.dumps(generate(tid, seed), sort_keys=True).encode())
        assert h.hexdigest() == expected, tid


# ---------------------------------------------------------------------------
# follow-up after the independent verification: identifiability, bulk, spans
# ---------------------------------------------------------------------------

def _largest_region(grid: np.ndarray, color: int) -> int:
    return max((len(c["cells"]) for c in connected_components(grid, connectivity=4, background=None)
                if c["color"] == color), default=0)


def test_background_identifiable_without_being_named():
    """Nothing in the prompt names the background: after the isolated texture cells are set
    aside it must be the strict plurality by a clear margin and own the largest region."""
    for seed in range(N_DIST):
        ex = _ex(seed)
        m = ex["metadata"]
        inp = np.array(ex["input"])
        keep = ~isolated_cell_mask(inp)
        vals, counts = np.unique(inp[keep], return_counts=True)
        table = dict(zip(vals.tolist(), counts.tolist()))
        bg = m["bg_color"]
        runner = max((c for v, c in table.items() if v != bg), default=0)
        assert table[bg] >= BG_MARGIN * runner, (seed, table)
        assert m["background_margin"] >= BG_MARGIN, seed
        largest_other = max((len(c["cells"]) for c in connected_components(inp, connectivity=4, background=bg)),
                            default=0)
        assert _largest_region(inp, bg) > largest_other, seed


def test_fixed_pieces_are_a_minority_of_the_canvas():
    shares = []
    for seed in range(N_DIST):
        ex = _ex(seed)
        m = ex["metadata"]
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        keep = ~isolated_cell_mask(inp)
        # unchanged non-background cells are the fixed pieces plus the extras (which also stay)
        fixed = int(((inp != m["bg_color"]) & (inp == out) & keep).sum()) - sum(m["extra_sizes"])
        shares.append(fixed / inp.size)
    shares = np.array(shares)
    assert shares.max() <= FIXED_SHARE_CAP + 1e-9, shares.max()
    assert np.percentile(shares, 90) <= 0.30
    assert (shares >= 0.40).sum() == 0


def test_multi_site_hosts_have_wide_sites():
    """Sites of a host with three or more cavities are not all the minimum width, and long
    seams exist overall (the verifier found spans collapsing to 3 and spans >= 12 at 0.3%)."""
    spans_all: list[int] = []
    multi: list[int] = []
    shrunk = 0
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        spans_all += m["spans"]
        idx = 0
        for k in m["sites_per_anchor"]:
            for _ in range(k):
                if k >= 3:
                    multi.append(m["spans"][idx])
                idx += 1
        shrunk += m["size_draw"] > 0
    assert sum(sp >= 10 for sp in spans_all) / len(spans_all) >= 0.02
    assert len(multi) >= 60
    assert multi.count(3) / len(multi) <= 0.45, Counter(multi)
    assert shrunk / N_DIST <= 0.55                      # the first draw usually fits now


# ---------------------------------------------------------------------------
# multi-side docking (September-2026 follow-up)
# ---------------------------------------------------------------------------

def _bucket(area: int) -> str:
    return "<200" if area < 200 else "200-499" if area < 500 else "500+"


def test_multi_side_shares_by_canvas_bucket():
    """Single-side only under 200 cells; two-side anchors on roughly a quarter of the examples
    from 200 cells; three- and four-side anchors on roughly a tenth from 500 cells; four-side
    anchors only under the colour rule (a four-side anchor has no free side for the border)."""
    by_bucket: dict[str, Counter] = {"<200": Counter(), "200-499": Counter(), "500+": Counter()}
    combos = Counter()
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        h, w = m["grid_size"]
        by_bucket[_bucket(h * w)][m["max_sides"]] += 1
        for sides in m["dock_sides_per_anchor"]:
            if len(sides) >= 2:
                combos[tuple(sorted(sides))] += 1
            if len(sides) == 4:
                assert m["anchor_rule"] == "color", seed
    small = by_bucket["<200"]
    assert set(small) == {1}, small
    mid = by_bucket["200-499"]
    mid_n = sum(mid.values())
    assert 0.18 <= mid[2] / mid_n <= 0.38, mid
    assert mid[3] + mid[4] == 0, mid
    large = by_bucket["500+"]
    large_n = sum(large.values())
    assert 0.15 <= large[2] / large_n <= 0.40, large
    assert 0.06 <= (large[3] + large[4]) / large_n <= 0.20, large
    assert large[4] >= 1 and large[3] >= 4, large
    # both opposite and adjacent pairs occur, and every four-side combination is the full set
    assert combos[("left", "right")] >= 10 and combos[("bottom", "top")] >= 10, combos
    assert sum(v for k, v in combos.items() if len(k) == 2 and k not in (("left", "right"), ("bottom", "top"))) >= 5, combos
    assert all(k == ("bottom", "left", "right", "top") for k in combos if len(k) == 4)


def _match(pieces: list[tuple[int, frozenset]], meta: list[tuple[int, str]]) -> bool:
    """Is there a one-to-one matching of solver pieces (anchor index, valid directions) to
    metadata entries (anchor index, recorded direction) with the recorded direction valid?"""
    used = [False] * len(meta)

    def rec(i: int) -> bool:
        if i == len(pieces):
            return True
        ai, dirs = pieces[i]
        for j, (aj, d) in enumerate(meta):
            if not used[j] and aj == ai and d in dirs:
                used[j] = True
                if rec(i + 1):
                    return True
                used[j] = False
        return False

    return rec(0)


def test_per_site_direction_consistency():
    """Piece by piece, the direction the metadata records for a loose piece is a direction
    from which the independent solver finds its unique seating flush, on the fixed piece the
    metadata assigns it; and every fixed piece's recorded docking sides are exactly the
    directions of the pieces seated on it."""
    checked_multi = 0
    for seed in range(N_SOLVE):
        ex = _ex(seed)
        m = ex["metadata"]
        res = _solve(ex)
        assert all(len(st) in (0, 1) for st in res["seatings"]), seed
        assert sum(len(st) == 0 for st in res["seatings"]) == m["n_extra"], seed
        # solver anchors are components in scan order; metadata anchors are units in unit
        # order: match them through their docked pieces' directions, anchor by anchor (open
        # cavities have no docked piece, so their sides are left out; an anchor whose every
        # cavity is open receives nothing and is left out too)
        per_anchor: dict[int, list[frozenset]] = {}
        for st in res["seatings"]:
            if st:
                per_anchor.setdefault(st[0][0], []).append(st[0][1])
        meta_anchor: list[list[str]] = []
        idx = 0
        for k in m["sites_per_anchor"]:
            sides = [s for s, is_open in zip(m["piece_sides"][idx: idx + k], m["site_open"][idx: idx + k]) if not is_open]
            if sides:
                meta_anchor.append(sides)
            idx += k
        assert len(per_anchor) == len(meta_anchor) <= m["n_anchors"], seed
        # a matching of solver anchors to metadata anchors such that each anchor's pieces match
        solver_groups = list(per_anchor.values())
        taken = [False] * len(meta_anchor)

        def rec(i: int) -> bool:
            if i == len(solver_groups):
                return True
            for j, sides in enumerate(meta_anchor):
                if taken[j] or len(sides) != len(solver_groups[i]):
                    continue
                if _match([(0, d) for d in solver_groups[i]], [(0, x) for x in sides]):
                    taken[j] = True
                    if rec(i + 1):
                        return True
                    taken[j] = False
            return False

        assert rec(0), (seed, [sorted(map(sorted, g)) for g in solver_groups], meta_anchor)
        idx = 0
        for k, dock in zip(m["sites_per_anchor"], m["dock_sides_per_anchor"]):
            own = m["piece_sides"][idx: idx + k]
            docked = [s for s, is_open in zip(own, m["site_open"][idx: idx + k]) if not is_open]
            idx += k
            assert sorted(set(own)) == sorted(dock), seed          # the cavities' sides
            assert set(docked) <= set(dock), seed                  # the docked pieces' directions
        checked_multi += m["max_sides"] >= 2
    assert checked_multi >= 150


def test_corner_clearance_docked_pieces_never_touch():
    """In the answer, no two loose pieces touch each other (8-neighbourhood), also not around a
    corner of a multi-side anchor; every docked piece touches only its own fixed piece."""
    multi_seen = 0
    for seed in range(N_SOLVE):
        ex = _ex(seed)
        m = ex["metadata"]
        res = _solve(ex)
        placed = [st[0][2] for st in res["seatings"] if st]
        owners = [st[0][0] for st in res["seatings"] if st]
        extras = [cells for (_, cells), st in zip(res["loose"], res["seatings"]) if not st]
        for i in range(len(placed)):
            grown = frozenset((r + dr, c + dc) for r, c in placed[i] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            for j in range(len(placed)):
                if i != j:
                    assert not (grown & placed[j]), (seed, i, j)
            for ai, (_, acells) in enumerate(res["anchors"]):
                if ai != owners[i]:
                    assert not (grown & acells), (seed, i, ai)
            for e in extras:                                   # nor a staying extra
                assert not (grown & e), (seed, i)
        multi_seen += m["max_sides"] >= 2
    assert multi_seen >= 150


def test_all_direction_uniqueness_on_every_example():
    """Over all four directions and all fixed pieces, every loose piece has exactly one seating
    in every example, and no seating exists in any other direction (a side-agnostic rule must
    not open a second reading)."""
    for seed in range(N_SOLVE):
        ex = _ex(seed)
        res = _solve(ex)
        for st in res["seatings"]:
            assert len(st) <= 1, (seed, [(a, d) for a, d, _ in st])
        assert sum(len(st) == 0 for st in res["seatings"]) == ex["metadata"]["n_extra"], seed
        assert res["n_assemblies"] == 1 and res["output_matches"] == 1, seed


def test_multi_side_anchors_are_hosts_or_mixed_and_sane():
    """A multi-side anchor carries at least as many cavities as sides, its docking sides are
    distinct, and its docked pieces' directions are exactly its sides."""
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        idx = 0
        for sides, k in zip(m["dock_sides_per_anchor"], m["sites_per_anchor"]):
            own = m["piece_sides"][idx: idx + k]
            idx += k
            assert set(own) == set(sides), (seed, own, sides)
            assert k >= len(sides)
        if m["multi_side"]:
            assert m["variant"] in ("host", "mixed"), (seed, m["variant"])


# ---------------------------------------------------------------------------
# leftovers (2026-09-07): extra pieces that seat nowhere, cavities that stay open
# ---------------------------------------------------------------------------

def test_leftovers_shares_and_proofs():
    """About 30% of the examples carry an extra loose piece and about 30% an open cavity (drawn
    independently); every extra is provably unseatable (the independent solver finds no seat
    against any fixed piece from any direction), stays exactly where it is, and every open
    cavity admits no loose piece (no seat of any loose piece covers a cell of its footprint,
    which is background in both grids); at least one piece always docks."""
    n = 400
    with_extra = with_open = both = 0
    kinds = Counter()
    extras_seen = 0
    for seed in range(n):
        ex = _ex(seed)
        m = ex["metadata"]
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        res = _solve(ex)
        with_extra += m["n_extra"] > 0
        with_open += m["n_open_cavities"] > 0
        both += m["n_extra"] > 0 and m["n_open_cavities"] > 0
        kinds.update(m["extra_kinds"])
        assert m["n_docked"] >= 1 and res["n_loose"] - res["n_stay"] >= 1, seed
        # extras: no seat anywhere; unchanged between input and output; their colours as recorded
        stay = [(col, cells) for (col, cells), st in zip(res["loose"], res["seatings"]) if not st]
        assert len(stay) == m["n_extra"], seed
        assert sorted(col for col, _ in stay) == sorted(m["extra_piece_colors"]), seed
        assert sorted(len(c) for _, c in stay) == sorted(m["extra_sizes"]), seed
        for _, cells in stay:
            assert all(out[r, c] == inp[r, c] for r, c in cells), seed
        extras_seen += len(stay)
        # open cavities: footprint background in both grids and covered by no seat of any piece
        foot = {tuple(rc) for rc in m["open_cavity_cells"]}
        assert (len(foot) > 0) == (m["n_open_cavities"] > 0), seed
        for r, c in foot:
            assert inp[r, c] == m["bg_color"] and out[r, c] == m["bg_color"], (seed, r, c)
        for st in res["seatings"]:
            for _, _, placed in st:
                assert not (placed & foot), seed
        # an open cavity is a real cavity: its footprint touches the fixed piece's cells
        if foot:
            stayset = frozenset().union(*[cells for _, cells in res["anchors"]])
            grown = {(r + dr, c + dc) for r, c in foot for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))}
            assert grown & stayset, seed
    assert 0.20 <= with_extra / n <= 0.42, with_extra / n
    assert 0.20 <= with_open / n <= 0.42, with_open / n
    assert both >= 8, both
    assert set(kinds) == {"near_miss", "jagged"} and min(kinds.values()) >= 8, kinds
    assert extras_seen >= with_extra


def test_prompt_never_reveals_leftovers():
    """Within one canonical wording bucket (variant, rule, anchor colour, cavity count, sides)
    the prompt is identical whether or not the example has extras or open cavities, and it
    carries the standing leftover sentence on every example."""
    buckets: dict[tuple, dict[str, set]] = {}
    for seed in range(N_DIST):
        m = _ex(seed)["metadata"]
        p = get_prompt("13", m, color_palette=LETTERS)
        assert "A loose piece that seats nowhere stays exactly where it is, and a cavity that no loose piece fits stays open." in p
        assert "at most one loose piece" in p
        for word in ("extra", "leftover", "spare", "unused", "no cavity", "n_extra"):
            assert word not in p.lower(), (seed, word)
        key = (m["variant"], m["anchor_rule"], m.get("anchor_color"), m["n_sites"], m["max_sides"])
        b = buckets.setdefault(key, {"prompts": set(), "flags": set()})
        b["prompts"].add(p)
        b["flags"].add((m["n_extra"] > 0, m["n_open_cavities"] > 0))
    mixed_buckets = 0
    for key, b in buckets.items():
        assert len(b["prompts"]) == 1, (key, len(b["prompts"]))
        mixed_buckets += len(b["flags"]) >= 3
    assert mixed_buckets >= 20, mixed_buckets      # plenty of buckets hold examples with and without leftovers



def test_extras_look_like_candidates():
    """No extra is a plain rectangle (a rectangle can never seat under the jaggedness rule, so it
    would give itself away) and every extra has at least four cells."""
    seen = 0
    for seed in range(400):
        m = generate("13", seed)["metadata"]
        if not m["n_extra"]:
            continue
        inp = np.asarray(generate("13", seed)["input"])
        out = np.asarray(generate("13", seed)["output"])
        for size in m["extra_sizes"]:
            assert size >= 4, seed
        # the staying loose pieces are exactly the extras; none fills its bounding box
        for colour in m["extra_piece_colors"]:
            cells = np.argwhere((inp == colour) & (out == colour))
            if len(cells) == 0:
                continue
            r0, c0 = cells.min(axis=0); r1, c1 = cells.max(axis=0)
            assert len(cells) < (r1 - r0 + 1) * (c1 - c0 + 1), seed
            seen += 1
    assert seen >= 60
