"""Count/remove isolated cells; the scene is independent of connectivity.

Single cells are foreground in this family, never background decoration.  The
plain field is inferred from a strict visible plurality.  Shapes, diagonal
chains and separated point arrangements share the same generator for both
rules and both output formats.
"""
from __future__ import annotations

import numpy as np

from common import shapes as S
from generators.crop_frame_26 import _geometric_shape
from generators.rigid_scene import capacity, slots, infer_ground


def choose_connectivity(rng):
    return int(rng.choice((4, 8)))


def isolation_mask(grid, background, connectivity):
    """Vectorized construction calculation, independent of the scalar oracle."""
    a = np.asarray(grid)
    h, w = a.shape
    supported = np.zeros_like(a, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx) or (connectivity == 4 and abs(dy) + abs(dx) != 1):
                continue
            ys, xs = slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx))
            yt, xt = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
            supported[ys, xs] |= a[ys, xs] == a[yt, xt]
    return (a != background) & ~supported


def solve(grid, connectivity, *, remove=False):
    """Literal visible-grid re-solver; works with arbitrary color symbols."""
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    a = np.asarray(grid)
    bg = infer_ground(a)
    h, w = a.shape
    selected = []
    for r in range(h):
        for c in range(w):
            if a[r, c] == bg:
                continue
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if connectivity == 8:
                neighbors += [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
            if not any(0 <= y < h and 0 <= x < w and a[y, x] == a[r, c] for y, x in neighbors):
                selected.append((r, c))
    if not remove:
        return str(len(selected))
    out = a.copy()
    for pos in selected:
        out[pos] = bg
    return out.tolist()


def validate_example(ex, *, remove=False):
    a = np.asarray(ex["input"])
    m = ex["metadata"]
    if a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape) or a.size < 8:
        raise ValueError("Invalid isolation canvas")
    if len(np.unique(a)) != m["n_colors"] or infer_ground(a) != m["bg_color"]:
        raise ValueError("Isolation palette or background is not recoverable")
    if solve(a, m["connectivity"], remove=remove) != ex["output"]:
        raise ValueError("Isolation output differs from the independent re-solver")
    for conn in (4, 8):
        if int(solve(a, conn)) != m[f"count_{conn}"]:
            raise ValueError("Isolation counts do not describe the final grid")


def _shape(rng, sh, sw):
    hh = max(1, int(round(sh * rng.uniform(*((.20, .44) if sw == 1 else (.65, 1.0))))))
    ww = max(1, int(round(sw * rng.uniform(*((.20, .44) if sh == 1 else (.65, 1.0))))))
    if min(sh, sw) >= 3:
        hh, ww = max(3, hh), max(3, ww)
    if hh * ww < 2:
        if sh >= 2:
            hh = 2
        else:
            ww = 2
    if min(hh, ww) >= 3 and rng.random() < .15:
        cells, family = S.frame(rng, hh, ww), "frame"
    else:
        cells, family = _geometric_shape(rng, hh, ww)
    y0, x0 = min(y for y, _ in cells), min(x for _, x in cells)
    cells = {(int(y - y0), int(x - x0)) for y, x in cells}
    return cells, family


def _render(rng, h, w, bg, colors, n, zero, contrast, density):
    feasible = [m for m in range(2, 7) if ((h + 1) // (m + 1)) * ((w + 1) // (m + 1)) >= n]
    boxes = slots(rng, h, w, n, minimum=max(feasible, default=1))
    rng.shuffle(boxes)
    grid = np.full((h, w), bg, dtype=np.uint8)
    families, areas = [], []
    for i, (r, c, sh, sw) in enumerate(boxes):
        color = colors[i % len(colors)]
        if zero or i == 0:
            kind = "shape"
        elif i == 1:
            kind = "points"
        elif i == 2 and contrast and min(sh, sw) >= 2:
            kind = "diagonal"
        else:
            kind = str(rng.choice(("shape", "points", "diagonal"), p=(.55, .25, .20)))
        if kind == "diagonal" and min(sh, sw) < 2:
            kind = "points"
        if kind == "shape":
            if zero and h * w <= 12:
                # A larger body cannot leave a strict field plurality on the
                # smallest canvases; a domino is still a genuine zero case.
                cells = {(0, 0), (0, 1)} if sw >= 2 else {(0, 0), (1, 0)}
                family = "bar"
            else:
                cells, family = _shape(rng, sh, sw)
            bh, bw = max(y for y, _ in cells) + 1, max(x for _, x in cells) + 1
            rr, cc = r + int(rng.integers(sh - bh + 1)), c + int(rng.integers(sw - bw + 1))
            for y, x in cells:
                grid[rr + y, cc + x] = color
        elif kind == "diagonal":
            length = int(rng.integers(2, min(sh, sw) + 1))
            rr, cc = r + int(rng.integers(sh - length + 1)), c + int(rng.integers(sw - length + 1))
            reverse = bool(rng.integers(2))
            cells = {(j, length - 1 - j if reverse else j) for j in range(length)}
            for y, x in cells:
                grid[rr + y, cc + x] = color
            family = "diagonal"
        else:
            paired = len(colors) > 1 and sw >= 2 and rng.random() < .45
            stride = 3 if paired else 2
            yy, xx = int(rng.integers(min(2, sh))), int(rng.integers(min(stride, max(1, sw - int(paired)))))
            pool = [(y, x) for y in range(yy, sh, 2) for x in range(xx, sw - int(paired), stride)]
            rng.shuffle(pool)
            cap = len(pool)
            if density == "small":
                count = int(rng.integers(1, min(5, cap) + 1))
            elif density == "dense":
                count = int(rng.integers(max(1, cap * 2 // 3), cap + 1))
            else:
                count = int(rng.integers(1, cap + 1))
            cells = set(pool[:count])
            other = colors[(i + 1) % len(colors)]
            for y, x in list(cells):
                grid[r + y, c + x] = color
                if paired:
                    grid[r + y, c + x + 1] = other
                    cells.add((y, x + 1))
            family = "mixed_color_pairs" if paired else "separated_points"
        families.append(family)
        areas.append(len(cells))
    return grid, families, areas


def generate(rng, *, remove=False):
    while True:
        h, w = map(int, rng.integers(1, 31, size=2))
        if h * w >= 8:
            break
    zero = bool(rng.random() < .06)
    max_colors = min(10, capacity(h, w) + 1, (h * w - 3) // 2 + 1)
    n_colors = int(rng.integers(2, max_colors + 1))
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    bg, colors = palette[0], palette[1:]
    budget = int(h * w * len(colors) / (len(colors) + 1.3))
    cap = min(capacity(h, w), budget // 2 if zero else budget - 1)
    lower = max(len(colors), min(1 if zero else 3, cap))
    upper = min(cap, max(lower, len(colors) + 2 + h * w // 150))
    choices = [k for k in range(lower, upper + 1)
               if not zero or h * w - 2 * k > 2.5 * ((k + len(colors) - 1) // len(colors))]
    if not choices:
        choices = [len(colors)]
    n = int(rng.choice(choices))
    contrast = bool(rng.random() < .85)
    density = str(rng.choice(("small", "any", "dense"), p=(.25, .45, .30)))
    for attempt in range(1, 101):
        grid, families, areas = _render(rng, h, w, bg, colors, n, zero, contrast, density)
        try:
            if infer_ground(grid) != bg:
                continue
        except ValueError:
            continue
        four, eight = isolation_mask(grid, bg, 4), isolation_mask(grid, bg, 8)
        count4, count8 = int(four.sum()), int(eight.sum())
        if len(np.unique(grid)) != n_colors or (zero and count4) or (not zero and not count8):
            continue
        # Rule selection is last, after all scene-dependent rejection.
        conn = choose_connectivity(rng)
        selected = four if conn == 4 else eight
        output = grid.copy()
        output[selected] = bg
        counts = sorted(np.unique(grid, return_counts=True)[1])
        ex = {"input": grid.tolist(), "output": output.tolist() if remove else str(int(selected.sum())),
              "metadata": {"connectivity": conn, "bg_color": bg, "background": "solid",
                           "background_margin": float(counts[-1] / counts[-2]),
                           "n_colors": n_colors, "grid_h": h, "grid_w": w,
                           "count_4": count4, "count_8": count8,
                           "connectivity_matters": count4 != count8,
                           "foreground_cells": int((grid != bg).sum()),
                           "shape_families": families, "shape_areas": areas,
                           "point_density": density, "zero": zero, "attempts": attempt}}
        validate_example(ex, remove=remove)
        return ex
    raise RuntimeError("Could not realize the fixed isolation scene and palette")
