"""Infer a unique, possibly clipped translation of a clean multicolor scene."""
from __future__ import annotations

import numpy as np

from generators.crop_frame_26 import _render_scene
from generators.rigid_scene import capacity, draw_objects, infer_ground
from generators.fixed_details import KINDS as DETAIL_KINDS, add_details, moving_cells


PROMPT41 = (
    "The input is a pair of grids named before and after. "
    "Isolated single cells in before are decorations, whatever their color, "
    "and stay fixed unless covered by a moved shape. One or more shapes in before move "
    "by the same signed offset to make after, keeping their colors and orientation. "
    "Each shape contains at least two edge-connected cells and may have several colors. "
    "Cells moved outside the grid are clipped. The offset may be zero. "
    "Return that offset as (dr,dc), where negative dr means up, positive dr means down, "
    "negative dc means left, and positive dc means right."
)


def matching_offsets(before, after):
    """Enumerate all offsets compatible with the visible source and fixed details.

    A non-static destination cell provides a complete candidate set. Every
    candidate is then checked against the entire image using shifted windows.
    Every visible color is considered as the field, without construction labels.
    """
    a, b = np.asarray(before), np.asarray(after)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Translation requires equal rectangular canvases")
    h, w = a.shape
    matches = set()
    for color in np.unique(np.concatenate([a.ravel(), b.ravel()])):
        moving = moving_cells(a, color)
        sources = np.argwhere(moving)
        if not len(sources):
            continue
        base = a.copy()
        base[moving] = color
        if np.array_equal(b, base):
            # This also catches a surviving cell hidden by an equal-colored
            # fixed dot: arbitrarily large fully clipped shifts look identical.
            raise ValueError("Full erasure cannot determine a unique offset")
        evidence = np.argwhere((b != base) & (b != color))
        if not len(evidence):
            continue
        rr, cc = min(evidence, key=lambda p: int(np.count_nonzero(a[moving] == b[tuple(p)])))
        for r, c in sources[a[tuple(sources.T)] == b[rr, cc]]:
            dr, dc = int(rr - r), int(cc - c)
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            out = base.copy()
            if r0 < r1 and c0 < c1:
                dest = out[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                take = moving[r0:r1, c0:c1]
                dest[take] = a[r0:r1, c0:c1][take]
            if np.array_equal(out, b):
                matches.add((dr, dc))
    return sorted(matches)


def validate(before, after, offset):
    if matching_offsets(before, after) != [tuple(offset)]:
        raise ValueError("The visible pair does not determine the planned translation")


def _move(grid, ground, dr, dc, source):
    # Constructor moves labeled points; the re-solver compares array windows.
    out = grid.copy()
    out[source] = ground
    h, w = grid.shape
    for r, c in np.argwhere(source):
        rr, cc = int(r) + dr, int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = grid[r, c]
    return out


def generate(rng):
    while True:
        h, w = map(int, rng.integers(1, 31, size=2))
        if h * w >= 6:
            break
    ncolors = int(rng.integers(2, min(10, 1 + h * w // 3) + 1))
    palette = list(map(int, rng.choice(10, ncolors, replace=False)))
    ground, colors = palette[0], palette[1:]
    count = int(rng.integers(1, min(12, capacity(h, w), h * w // 55 + 2, h * w // 6) + 1))
    fill = str(rng.choice(["bands", "diagonal_bands", "stair_bands", "patchwork", "striped_panels"]))
    textured = min(h, w) >= 8 and rng.random() < .50
    background = str(rng.choice(DETAIL_KINDS)) if textured else "solid"
    detail_colors = [int(rng.choice(colors))]
    zero = bool(rng.random() < .06)
    wide = bool(rng.random() < .40)
    for attempt in range(1, 301):
        try:
            objects, kinds = draw_objects(rng, h, w, [colors[0]], count, scale=(.48, .88))
        except ValueError:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for cells, _ in objects:
            for p in cells:
                mask[p] = True
        if mask.sum() < len(colors):
            continue
        painted = _render_scene(rng, mask, colors, fill)
        if painted is None:
            continue
        a = np.full((h, w), ground, dtype=np.uint8)
        a[mask] = painted[0][mask]
        details = np.zeros_like(mask)
        if textured:
            painted = add_details(rng, a, ground, mask, detail_colors, background)
            if painted is None:
                continue
            a, details = painted
        if not np.array_equal(moving_cells(a, ground), mask):
            continue
        if len(np.unique(a)) != ncolors:
            continue
        try:
            if infer_ground(a) != ground:
                continue
        except ValueError:
            continue
        for motion_attempt in range(1, 121):
            limits = (h - 1, w - 1) if wide else (min(5, h - 1), min(5, w - 1))
            dr, dc = (0, 0) if zero else tuple(int(rng.integers(-v, v + 1)) for v in limits)
            if not zero and (dr, dc) == (0, 0):
                continue
            b = _move(a, ground, dr, dc, mask)
            try:
                validate(a, b, (dr, dc))
            except ValueError:
                continue
            counts = sorted(np.unique(a, return_counts=True)[1])
            coords = np.argwhere(mask) + np.asarray((dr, dc))
            survived = int(np.count_nonzero((coords[:, 0] >= 0) & (coords[:, 0] < h) &
                                            (coords[:, 1] >= 0) & (coords[:, 1] < w)))
            return {
                "input": {"before": a.tolist(), "after": b.tolist()},
                "output": f"({dr},{dc})",
                "metadata": {
                    "dr": dr, "dc": dc, "bg_color": ground,
                    "n_colors": ncolors, "object_count": count, "shape_families": kinds,
                    "fill_family": fill, "background_kind": background,
                    "texture_cells": int(details.sum()),
                    "background_margin": float(counts[-1] / counts[-2]),
                    "foreground_cells": int(mask.sum()),
                    "surviving_cells": survived,
                    "clipped_cells": int(mask.sum() - survived),
                    "offset_candidates": 1, "motion_range": "wide" if wide else "short",
                    "attempts": attempt, "motion_attempts": motion_attempt,
                },
            }
    raise RuntimeError(f"Cannot construct a unique translation on {h}x{w} with {ncolors} colors")
