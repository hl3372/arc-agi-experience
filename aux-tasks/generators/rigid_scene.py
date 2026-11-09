"""Small geometry helpers shared by tasks 31 and 34; no task registration."""
from __future__ import annotations

from collections import deque
import numpy as np

from generators.crop_frame_26 import _geometric_shape


def capacity(h, w):
    return max(((h + 1) // 3) * ((w + 1) // 2), ((h + 1) // 2) * ((w + 1) // 3))


def slots(rng, h, w, n, r=0, c=0, minimum=1):
    """Random guillotine layout with one-cell gutters and two-cell minimums."""
    if n == 1:
        return [(r, c, h, w)]
    choices = []
    def cap(a, b):
        return capacity(a, b) if minimum == 1 else ((a + 1) // (minimum + 1)) * ((b + 1) // (minimum + 1))
    for axis, extent in enumerate((h, w)):
        for cut in range(1, extent - 1):
            ca = cap(cut, w) if axis == 0 else cap(h, cut)
            cb = cap(h - cut - 1, w) if axis == 0 else cap(h, w - cut - 1)
            low, high = max(1, n - cb), min(n - 1, ca)
            if low <= high:
                choices.append((axis, cut, low, high))
    if not choices:
        raise ValueError("Cannot partition this slot budget")
    axis, cut, low, high = choices[int(rng.integers(len(choices)))]
    left = int(rng.integers(low, high + 1))
    if axis == 0:
        return slots(rng, cut, w, left, r, c, minimum) + slots(rng, h - cut - 1, w, n - left, r + cut + 1, c, minimum)
    return slots(rng, h, cut, left, r, c, minimum) + slots(rng, h, w - cut - 1, n - left, r, c + cut + 1, minimum)


def draw_objects(rng, h, w, colors, n=None, scale=(.38, .85), max_h=None, max_w=None):
    n = len(colors) if n is None else n
    objects, families = [], []
    # Reserve genuinely two-dimensional slots whenever the scene can hold
    # them. Otherwise unconstrained guillotine cuts overwhelmingly make bars,
    # even on large canvases with ample room for letters and geometric bodies.
    feasible = [m for m in range(2, 7) if ((h + 1) // (m + 1)) * ((w + 1) // (m + 1)) >= n]
    minimum = max(feasible, default=1)
    for index, (r, c, sh, sw) in enumerate(slots(rng, h, w, n, minimum=minimum)):
        hh = max(1, int(round(sh * float(rng.uniform(.20, .43) if sw == 1 else rng.uniform(*scale)))))
        ww = max(1, int(round(sw * float(rng.uniform(.20, .43) if sh == 1 else rng.uniform(*scale)))))
        if min(sh, sw) >= 3 and rng.random() < .85:
            hh, ww = max(3, hh), max(3, ww)
        if max_h is not None:
            hh = min(hh, max_h)
        if max_w is not None:
            ww = min(ww, max_w)
        if hh * ww < 2:
            if sh >= 2:
                hh = 2
            else:
                ww = 2
        cells, family = _geometric_shape(rng, hh, ww)
        cells = {(int(y), int(x)) for y, x in cells}
        if len(cells) < 2:
            raise ValueError("Shape is too small")
        y0, x0 = min(y for y, _ in cells), min(x for _, x in cells)
        bh, bw = max(y for y, _ in cells) - y0 + 1, max(x for _, x in cells) - x0 + 1
        rr, cc = r + int(rng.integers(sh - bh + 1)), c + int(rng.integers(sw - bw + 1))
        objects.append(({(y - y0 + rr, x - x0 + cc) for y, x in cells}, colors[index % len(colors)]))
        families.append(family)
    return objects, families


def visible_objects(grid, *, excluded=(), minimum=2):
    """Independent flood fill of same-colour 4-connected components."""
    a = np.asarray(grid)
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    result, dots = [], set()
    for r in range(h):
        for c in range(w):
            if seen[r, c] or a[r, c] in excluded:
                continue
            color = a[r, c].item()
            q = deque([(r, c)])
            cells = {(r, c)}
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and a[yy, xx] == color:
                        seen[yy, xx] = True
                        cells.add((yy, xx))
                        q.append((yy, xx))
            if len(cells) >= minimum:
                result.append((cells, color))
            else:
                dots |= cells
    return sorted(result, key=lambda item: min(item[0])), dots


def infer_ground(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    if len(order) < 2 or counts[order[-1]] <= 1.25 * counts[order[-2]]:
        raise ValueError("Background is not identifiable")
    return colors[order[-1]].item()
