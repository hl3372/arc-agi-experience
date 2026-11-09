"""Small fixed details for tasks whose moving shapes may be multicolored."""
from __future__ import annotations

import numpy as np

from common.backgrounds import sparse_mask

KINDS = ("lattice", "dotted_stripes", "sparse_checker", "dotted_border", "dotted_diagonal", "random_dots")


def moving_cells(grid, ground, excluded=()):
    """Cells in non-field edge-connected groups of at least two cells.

    Adjacency is independent of color. A clipped fragment is never classified
    again: callers use this on the source image only.
    """
    a = np.asarray(grid)
    occupied = (a != ground) & ~np.isin(a, excluded)
    adjacent = np.zeros(a.shape, bool)
    adjacent[1:] |= occupied[:-1]
    adjacent[:-1] |= occupied[1:]
    adjacent[:, 1:] |= occupied[:, :-1]
    adjacent[:, :-1] |= occupied[:, 1:]
    return occupied & adjacent


def add_details(rng, grid, ground, occupied, colors, kind):
    """Paint a sparse pattern before the motion is selected; no destination mask."""
    a = np.asarray(grid)
    h, w = a.shape
    padded = np.pad(occupied, 1)
    protected = np.logical_or.reduce([padded[r:r + h, c:c + w] for r in range(3) for c in range(3)])
    for _ in range(12):
        dots = sparse_mask(rng, h, w, kind) & ~protected
        points = np.argwhere(dots)
        # Keep the pattern legible by retaining whole alternating rows/columns
        # when it is dense, rather than adding large clouds of individual dots.
        if len(points) > .06 * a.size:
            rr, cc = np.indices(a.shape)
            axis = rr if rng.integers(2) else cc
            selected = np.unique(axis[dots])
            dots &= np.isin(axis, selected[int(rng.integers(2))::2])
            points = np.argwhere(dots)
        if not len(points):
            continue
        if len(points) > max(1, int(.06 * a.size)):
            points = points[rng.choice(len(points), max(1, int(.06 * a.size)), replace=False)]
            dots[:] = False
            dots[points[:, 0], points[:, 1]] = True
        out = a.copy()
        for i, (r, c) in enumerate(points):
            out[r, c] = colors[i % len(colors)]
        return out, dots
    return None
