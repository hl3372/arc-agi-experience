"""Property tests for the shared background library (``common/backgrounds.py``).

Every kind: produced where the size rule allows and refused below it; painted
cells never touch object cells; sparse kinds are isolated single cells (any
colour combination); coverage limits hold; solid kinds are contiguous on an
empty canvas; texture colours avoid the background / excluded / present
colours; the structural helpers agree; the prompt sentence follows the
named-vs-inferred policy; determinism; performance.
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.backgrounds import (  # noqa: E402
    COVERAGE_LIMIT,
    GROUPS,
    KIND_GROUP,
    MIN_SHORT_SIDE,
    SOLID_KINDS,
    SPARSE_KINDS,
    STRUCTURED_KINDS,
    allowed_groups,
    apply_background,
    isolated_cell_mask,
    pick_texture_colors,
    solid_layers,
    sparse_mask,
    structured_layers,
    texture_cells_isolated,
)
from prompts import ISOLATED_CELL_RULE, background_sentence  # noqa: E402
from symbol_remap import COLOR_PALETTE_META_KEY  # noqa: E402

ALL_KINDS = SPARSE_KINDS + SOLID_KINDS + STRUCTURED_KINDS
LETTERS = list("ABCDEFGHIJ")
SIZES = [(1, 1), (1, 30), (30, 1), (7, 30), (8, 8), (8, 30), (11, 12), (12, 12), (15, 9), (20, 25), (30, 30)]


def _scene(rng, H, W, bg=0):
    """A plain grid with a few random rectangles of object colours and its occupancy mask."""
    grid = np.full((H, W), bg, dtype=np.uint8)
    occ = np.zeros((H, W), dtype=bool)
    for _ in range(int(rng.integers(1, 5))):
        h = int(rng.integers(1, max(2, min(6, H)) + 1))
        w = int(rng.integers(1, max(2, min(6, W)) + 1))
        r = int(rng.integers(0, H - h + 1))
        c = int(rng.integers(0, W - w + 1))
        grid[r:r + h, c:c + w] = int(rng.integers(1, 4))
        occ[r:r + h, c:c + w] = True
    return grid, occ


def _adjacent_pairs(mask):
    H, W = mask.shape
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = mask[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
        b = mask[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)]
        if np.any(a & b):
            return True
    return False


# ---------------------------------------------------------------------------
# size rules
# ---------------------------------------------------------------------------

def test_size_rules_per_kind():
    for kind in ALL_KINDS:
        group = KIND_GROUP[kind]
        floor = MIN_SHORT_SIDE[group]
        for H, W in SIZES:
            rng = np.random.default_rng(ALL_KINDS.index(kind) * 1000 + H * 31 + W)
            grid = np.zeros((H, W), dtype=np.uint8)
            occ = np.zeros((H, W), dtype=bool)
            got = apply_background(rng, grid, 0, occ, kind=kind)
            if min(H, W) < floor:
                assert got is None, (kind, H, W)
            else:
                assert got is not None, (kind, H, W)
                display, info = got
                assert info["kind"] == kind and info["group"] == group
                assert (display != 0).any()


def test_allowed_groups():
    assert allowed_groups(7, 30) == []
    assert allowed_groups(8, 30) == ["sparse", "solid"]
    assert allowed_groups(12, 12) == list(GROUPS)


# ---------------------------------------------------------------------------
# painting properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ALL_KINDS)
def test_never_touches_objects_and_respects_coverage(kind):
    group = KIND_GROUP[kind]
    floor = MIN_SHORT_SIDE[group]
    n = 0
    for seed in range(120):
        rng = np.random.default_rng(seed)
        H = int(rng.integers(floor, 31))
        W = int(rng.integers(floor, 31))
        grid, occ = _scene(rng, H, W, bg=0)
        got = apply_background(rng, grid, 0, occ, kind=kind, exclude=[9])
        if got is None:
            continue
        n += 1
        display, info = got
        assert np.array_equal(display[occ], grid[occ]), (kind, seed)
        painted = display != grid
        assert not (painted & occ).any()
        free = (~occ).sum()
        assert painted.sum() / free <= COVERAGE_LIMIT[group] + 1e-9, (kind, seed, painted.sum() / free)
        assert abs(info["coverage"] - painted.sum() / free) < 1e-9
        present = {int(v) for v in np.unique(grid)}
        for c in info["texture_colors"]:
            assert c != 0 and c != 9 and c not in present, (kind, seed, c)
        assert set(np.unique(display[painted]).tolist()) == set(info["texture_colors"])
        assert info["named"] == (group != "sparse")
    assert n >= 100, (kind, n)


@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_sparse_isolation(kind):
    multi = 0
    for seed in range(150):
        rng = np.random.default_rng(seed)
        H = int(rng.integers(8, 31))
        W = int(rng.integers(8, 31))
        grid, occ = _scene(rng, H, W, bg=0)
        got = apply_background(rng, grid, 0, occ, kind=kind)
        if got is None:
            continue
        display, info = got
        mask = np.isin(display, info["texture_colors"])
        assert not _adjacent_pairs(mask), (kind, seed)
        assert texture_cells_isolated(display, info["texture_colors"])
        # every texture cell is an isolated cell in the structural sense
        assert isolated_cell_mask(display)[mask].all(), (kind, seed)
        if kind == "multi_speckle" and len(info["texture_colors"]) >= 2:
            multi += 1
    if kind == "multi_speckle":
        assert multi >= 100, multi  # several colours at once in nearly every example


@pytest.mark.parametrize("kind", SOLID_KINDS)
def test_solid_kinds_are_contiguous_on_an_empty_canvas(kind):
    """Every texture cell has an 8-neighbour of the same texture colour (lines, bands,
    blocks, rings, split regions; a checker of 1x1 blocks connects diagonally)."""
    for seed in range(80):
        rng = np.random.default_rng(seed)
        H = int(rng.integers(8, 31))
        W = int(rng.integers(8, 31))
        grid = np.zeros((H, W), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)
        got = apply_background(rng, grid, 0, occ, kind=kind)
        assert got is not None, (kind, seed)
        display, info = got
        for col in info["texture_colors"]:
            m = display == col
            H2, W2 = m.shape
            has_nb = np.zeros_like(m)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    a = m[max(0, -dr):H2 - max(0, dr), max(0, -dc):W2 - max(0, dc)]
                    b = m[max(0, dr):H2 - max(0, -dr), max(0, dc):W2 - max(0, -dc)]
                    has_nb[max(0, -dr):H2 - max(0, dr), max(0, -dc):W2 - max(0, dc)] |= a & b
            assert has_nb[m].all(), (kind, seed, col)


def test_thinning_keeps_solid_kinds_under_the_limit_when_objects_crowd_the_canvas():
    """Objects covering most of the background-coloured cells push the free-cell coverage up;
    the painter thins the pattern instead of subsampling."""
    for kind in SOLID_KINDS:
        for seed in range(40):
            rng = np.random.default_rng(seed)
            H, W = 20, 20
            base = solid_layers(np.random.default_rng(seed), H, W, kind)
            occ = (base == 0) & (np.random.default_rng(seed + 1).random((H, W)) < 0.7)
            grid = np.zeros((H, W), dtype=np.uint8)
            grid[occ] = 1
            got = apply_background(rng, grid, 0, occ, kind=kind)
            if got is None:
                continue
            display, info = got
            assert info["coverage"] <= COVERAGE_LIMIT["solid"] + 1e-9, (kind, seed, info["coverage"])


def test_colour_pool_rules():
    rng = np.random.default_rng(0)
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[2:4, 2:4] = 5
    occ = grid == 5
    for seed in range(200):
        got = apply_background(np.random.default_rng(seed), grid, 0, occ, exclude=[7, 8])
        if got is None:
            continue
        for c in got[1]["texture_colors"]:
            assert c not in (0, 5, 7, 8)
    # explicit colours are honoured; the background colour is refused
    got = apply_background(rng, grid, 0, occ, kind="bands", colors=[6])
    assert got is not None and got[1]["texture_colors"] == [6]
    assert apply_background(rng, grid, 0, occ, kind="bands", colors=[0]) is None
    # an exhausted pool returns None
    full = np.arange(9).reshape(3, 3).astype(np.uint8)
    full = np.tile(full, (4, 4))[:12, :12]
    assert apply_background(rng, full, 9, np.zeros((12, 12), bool), kind="bands") is None
    assert pick_texture_colors(rng, 3, range(10)) == []


def test_structural_helpers_agree():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        H = int(rng.integers(8, 31))
        W = int(rng.integers(8, 31))
        grid, occ = _scene(rng, H, W, bg=0)
        got = apply_background(rng, grid, 0, occ, group="sparse")
        if got is None:
            continue
        display, info = got
        iso = isolated_cell_mask(display)
        tex = np.isin(display, info["texture_colors"])
        assert iso[tex].all()
        # a 2-cell object is never "isolated"
        big = np.zeros((10, 10), dtype=np.uint8)
        big[3, 3:5] = 4
        assert not isolated_cell_mask(big)[3, 3] and not isolated_cell_mask(big)[3, 4]
        assert texture_cells_isolated(big, [4]) is False
        assert texture_cells_isolated(big, [7]) is True


def test_group_weights_and_defaults():
    counts = {g: 0 for g in GROUPS}
    for seed in range(600):
        rng = np.random.default_rng(seed)
        grid = np.zeros((16, 16), dtype=np.uint8)
        occ = np.zeros((16, 16), dtype=bool)
        got = apply_background(rng, grid, 0, occ)
        assert got is not None
        counts[got[1]["group"]] += 1
    assert counts["sparse"] > counts["solid"] > counts["structured"] > 0
    # structured is never chosen under a short side of 12
    for seed in range(200):
        got = apply_background(np.random.default_rng(seed), np.zeros((10, 30), np.uint8), 0, np.zeros((10, 30), bool))
        assert got is not None and got[1]["group"] != "structured"


def test_background_sentence_policy():
    meta = {"texture_group": "sparse", "texture_colors": [4], "texture_named": False}
    assert background_sentence(meta) == ""
    assert background_sentence({}) == ""
    meta = {"texture_group": "solid", "texture_colors": [4], "texture_named": True}
    assert background_sentence(meta) == "cells of color 4 are background texture, not objects"
    meta = {"texture_group": "solid", "texture_colors": [4, 7], "texture_named": True, COLOR_PALETTE_META_KEY: LETTERS}
    s = background_sentence(meta)
    assert s == "cells of colors E and H are background texture, not objects", s
    assert not re.search(r"color \d", s)
    meta = {"texture_group": "structured", "texture_colors": [1, 2, 3], COLOR_PALETTE_META_KEY: LETTERS}
    assert background_sentence(meta) == "cells of colors B, C and D are background texture, not objects"
    assert "isolated single cells" in ISOLATED_CELL_RULE and "background texture" in ISOLATED_CELL_RULE


def test_determinism():
    for seed in range(50):
        grid, occ = _scene(np.random.default_rng(seed), 20, 24, bg=0)
        a = apply_background(np.random.default_rng(seed), grid, 0, occ)
        b = apply_background(np.random.default_rng(seed), grid, 0, occ)
        assert (a is None and b is None) or (np.array_equal(a[0], b[0]) and a[1] == b[1])


def test_layer_builders_shapes():
    rng = np.random.default_rng(3)
    for kind in SPARSE_KINDS:
        m = sparse_mask(rng, 14, 21, kind)
        assert m.shape == (14, 21) and m.dtype == bool and m.any()
        assert not _adjacent_pairs(m), kind
    for kind in SOLID_KINDS:
        for thin in range(5):
            L = solid_layers(rng, 14, 21, kind, thin)
            assert L.shape == (14, 21) and L.max() >= 1
    for kind in STRUCTURED_KINDS:
        L = structured_layers(rng, 14, 21, kind)
        assert L.shape == (14, 21) and set(np.unique(L).tolist()) <= {0, 1, 2}


def test_performance():
    rng = np.random.default_rng(0)
    grid, occ = _scene(rng, 30, 30, bg=0)
    times = []
    for seed in range(300):
        t0 = time.perf_counter()
        apply_background(np.random.default_rng(seed), grid, 0, occ)
        times.append(time.perf_counter() - t0)
    assert np.mean(times) * 1000 <= 0.3, np.mean(times) * 1000
