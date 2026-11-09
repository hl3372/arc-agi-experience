"""Checks for the shared shape vocabulary (common/shapes.py) and topology library."""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import shapes as S  # noqa: E402
from common import topology as T  # noqa: E402
from common.objects import bounding_box  # noqa: E402


def _mask(shape):
    r0, c0, r1, c1 = bounding_box(shape)
    m = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in shape:
        m[r, c] = True
    return m


def test_sample_shape_respects_bounds_and_connectivity():
    rng = np.random.default_rng(0)
    fams = Counter()
    for i in range(2000):
        conn = 8 if i % 4 == 0 else 4
        max_cells = int(rng.integers(1, 60))
        max_h = int(rng.integers(1, 15))
        max_w = int(rng.integers(1, 15))
        shape, fam = S.sample_shape(rng, max_cells, max_h, max_w, connectivity=conn)
        fams[fam] += 1
        assert len(shape) <= max_cells
        h, w = S.shape_size(shape)
        assert h <= max_h and w <= max_w
        assert min(r for r, _ in shape) == 0 and min(c for _, c in shape) == 0
        assert T.is_connected(_mask(shape), 8 if fam == "diagonal" else 4), fam
    assert len(fams) >= 12, fams


def test_free_polyominoes_are_distinct_under_symmetry():
    canon = set()
    for name, shape in S.FREE_POLYOMINOES.items():
        key = min(tuple(sorted(o)) for o in __import__("common.objects", fromlist=["all_orientations"]).all_orientations(shape))
        assert key not in canon, name
        canon.add(key)
    assert len(canon) == 21


def test_topology_oracles_on_known_masks():
    ring = np.zeros((5, 5), dtype=bool)
    ring[1:4, 1:4] = True
    ring[2, 2] = False
    assert T.count_holes(ring) == 1
    assert T.min_wall_thickness(ring) == 1
    opened = ring.copy()
    opened[1, 2] = False
    assert T.count_holes(opened) == 0
    assert T.gap_is_unique_repair(ring, [(1, 2)])
    assert not T.gap_is_unique_repair(ring, [(1, 1)])  # corner: no enclosed neighbour
    # a diagonal staircase still encloses under 4-connectivity of empty cells
    diamond = T.diamond_mask(7, 7)
    wall, inner = T.outline(diamond, 1)
    assert inner.any() and T.count_holes(wall) == 1


def test_silhouettes_and_carving_are_valid():
    for seed in range(200):
        rng = np.random.default_rng(seed)
        res = T.random_silhouette(rng, int(rng.integers(4, 20)), int(rng.integers(4, 20)))
        if res is None:
            continue
        mask, fam = res
        assert T.is_connected(mask, 4) and T.count_holes(mask) == 0, (seed, fam)
        budget = int(T.carveable_cells(mask, np.zeros_like(mask), 1).sum())
        if budget < 3:
            continue
        plan, _ = T.hole_plan(rng, 2, budget)
        out = T.carve_holes(rng, mask, plan, strict=False)
        assert out is not None
        carved, holes = out
        assert T.is_connected(carved, 4)
        assert T.count_holes(carved) == len(holes)
