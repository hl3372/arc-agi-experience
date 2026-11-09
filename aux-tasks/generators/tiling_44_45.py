"""Coherent tile content, with distinct whole-grid and minimal-tile rules."""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from generators.crop_frame_26 import _partition, _render_scene


def smallest_tile(grid):
    """Find the minimum rectangular tile by independently checking all divisors.

    Each valid rectangle supplies an independent row and column period, so the
    least-area solution is unique. This checks displayed cells, not the base
    tile or repeat counts retained by construction.
    """
    a = np.asarray(grid)
    if a.ndim != 2 or not a.size:
        raise ValueError("Expected a nonempty rectangular grid")
    h, w = a.shape
    matches = []
    for th in range(1, h + 1):
        if h % th:
            continue
        for tw in range(1, w + 1):
            if w % tw:
                continue
            if all(np.array_equal(a[r:r+th, c:c+tw], a[:th, :tw])
                   for r in range(0, h, th) for c in range(0, w, tw)):
                matches.append((th, tw))
    area = min(hh * ww for hh, ww in matches)
    winners = [(hh, ww) for hh, ww in matches if hh * ww == area]
    if len(winners) != 1:
        raise ValueError("Ambiguous minimum tile")
    return winners[0]


def validate_repeat(grid, output, repeats):
    a, b = np.asarray(grid), np.asarray(output)
    h, w = a.shape
    m, n = repeats
    if min(m, n) < 1 or b.shape != (h * m, w * n):
        raise ValueError("Incorrect whole-grid repeat dimensions")
    for r in range(b.shape[0]):
        for c in range(b.shape[1]):
            if b[r, c] != a[r % h, c % w]:
                raise ValueError("Incorrect whole-grid repeat content")


def validate_scale(grid, repeats):
    h, w = np.asarray(grid).shape
    th, tw = smallest_tile(grid)
    if (h // th, w // tw) != tuple(repeats):
        raise ValueError("Planned scale is not the visible minimum-tile scale")


def _content(rng, h, w, palette, primitive):
    # Preserve palette and family through retries. There is no corner-pixel
    # repair or canned fallback to force a primitive tile.
    if min(h, w) == 1:
        family = "linear_bands"
    else:
        choices = ["patchwork", "diagonal_bands", "stair_bands", "objects", "striped_panels"]
        if primitive and h == w == 2:
            choices.remove("stair_bands")
        if ((h + 1) // 2) * ((w + 1) // 2) >= len(palette):
            choices.append("mirror")
        family = str(rng.choice(choices))
    for attempt in range(1, 401):
        if family == "linear_bands":
            length = max(h, w)
            cuts = sorted(map(int, rng.choice(np.arange(1, length), len(palette) - 1, replace=False)))
            sizes = np.diff([0, *cuts, length])
            grid = np.repeat(palette, sizes).reshape(h, w)
        elif family == "patchwork":
            count = min(h * w, max(3, len(palette)) + int(rng.integers(0, 5)))
            boxes = _partition(rng, np.ones((h, w), dtype=bool), count)
            rng.shuffle(boxes)
            grid = np.empty((h, w), dtype=np.uint8)
            for i, (r, c, bh, bw) in enumerate(boxes):
                grid[r:r+bh, c:c+bw] = palette[i % len(palette)]
        else:
            rendered = _render_scene(rng, np.ones((h, w), dtype=bool), palette, family)
            if rendered is None:
                continue
            grid = rendered[0]
        if len(np.unique(grid)) != len(palette):
            continue
        if primitive and smallest_tile(grid) != (h, w):
            continue
        return grid, family, attempt
    raise RuntimeError(f"Cannot draw {family} on {h}x{w} with {len(palette)} colors")


@lru_cache(maxsize=None)
def _subtiles(h, w):
    return tuple((th, tw) for th in range(1, h + 1) if h % th == 0
                 for tw in range(1, w + 1) if w % tw == 0
                 if (th, tw) != (h, w) and th * tw >= 2)


def generate_repeat(rng):
    identity = bool(rng.random() < .06)
    # Independent uniform sides, conditioned only on fitting a nonidentity
    # repeat under the ARC cap. An unrepeated side can use all 30 cells.
    while True:
        h, w = map(int, rng.integers(1, 31, 2))
        if identity or min(h, w) <= 15:
            break
    options = [(m, n) for m in range(1, 30 // h + 1) for n in range(1, 30 // w + 1)
               if (m, n) != (1, 1)]
    m, n = (1, 1) if identity else options[int(rng.integers(len(options)))]
    palette_n = 1 if h * w == 1 else int(rng.integers(2, min(10, h * w) + 1))
    palette = list(map(int, rng.choice(10, palette_n, replace=False)))
    periodic_options = [shape for shape in _subtiles(h, w) if shape[0] * shape[1] >= palette_n]
    planned_periodic = bool(periodic_options and rng.random() < .35)
    th, tw = periodic_options[int(rng.integers(len(periodic_options)))] if planned_periodic else (h, w)
    base, family, attempts = _content(rng, th, tw, palette, primitive=False)
    grid = np.tile(base, (h // th, w // tw))
    output = np.tile(grid, (m, n))
    validate_repeat(grid, output, (m, n))
    ph, pw = smallest_tile(grid)
    return {"input": grid.tolist(), "output": output.tolist(),
            "metadata": {"tile_rows": m, "tile_cols": n, "n_colors": palette_n,
                         "content_family": family, "input_periodic": (ph, pw) != (h, w),
                         "smallest_input_tile": [ph, pw], "planned_periodic": planned_periodic,
                         "attempts": attempts}}


def generate_scale(rng):
    # Size first; remove the old composite-repeat branch. Prime dimensions are
    # valid on a once-only axis or when a one-cell-wide motif repeats there.
    thin_motif = bool(rng.random() < .08)
    while True:
        h, w = map(int, rng.integers(1, 31, 2))
        options = [shape for shape in _subtiles(h, w)
                   if (min(shape) == 1) == thin_motif]
        if options:
            break
    max_area = max(th * tw for th, tw in options)
    palette_n = int(rng.integers(2, min(10, max_area) + 1))
    palette = list(map(int, rng.choice(10, palette_n, replace=False)))
    options = [shape for shape in options if shape[0] * shape[1] >= palette_n]
    both = [(th, tw) for th, tw in options if th < h and tw < w]
    once = [(th, tw) for th, tw in options if th == h or tw == w]
    if both and once:
        options = once if rng.random() < .25 else both
    # Give larger motifs meaningful weight without dropping the high-count
    # tail. The palette's physical capacity is the only color/size coupling.
    weights = np.asarray([(th * tw) ** .45 for th, tw in options])
    th, tw = options[int(rng.choice(len(options), p=weights / weights.sum()))]
    base, family, attempts = _content(rng, th, tw, palette, primitive=True)
    m, n = h // th, w // tw
    grid = np.tile(base, (m, n))
    validate_scale(grid, (m, n))
    return {"input": grid.tolist(), "output": f"{m}x{n}",
            "metadata": {"M": m, "N": n, "tile_h": th, "tile_w": tw,
                         "n_colors": palette_n, "content_family": family,
                         "minimal_tile_candidates": 1, "attempts": attempts}}
