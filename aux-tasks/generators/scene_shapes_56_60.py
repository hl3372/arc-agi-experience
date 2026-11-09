"""Geometry and decoration for the rebuilt contact, search and path tasks.

No task registration or answer logic lives here. Shapes have a small, familiar
vocabulary; their dimensions and wall widths vary with the available space.
"""
from __future__ import annotations

import numpy as np


def crop(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError("Empty shape")
    return mask[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()


def shape(rng, h, w, family=None, *, thin=False):
    choices = ["rectangle", "bar"]
    if min(h, w) >= 2:
        choices += ["L", "staircase"]
    if min(h, w) >= 3:
        choices += ["T", "U", "H", "plus", "frame"]
    if min(h, w) >= 5:
        choices += ["chamfer", "diamond"]
    family = str(rng.choice(choices)) if family is None else family
    if family not in choices and family != "square":
        raise ValueError("Shape family does not fit its dimensions")
    if family == "square":
        h = w = min(h, w)
    if family == "bar":
        if h > w or (h == w and rng.integers(2)):
            w = 1
        else:
            h = 1
    yy, xx = np.indices((h, w))
    t = 1 if thin else int(rng.integers(1, max(2, min(h, w)//3+1)))
    mask = np.ones((h, w), bool)
    if family == "L":
        mask = (xx < t) | (yy >= h-t)
    elif family == "T":
        mask = (yy < t) | ((xx >= (w-t)//2) & (xx < (w+t)//2))
    elif family == "U":
        mask = (xx < t) | (xx >= w-t) | (yy >= h-t)
    elif family == "H":
        mask = (xx < t) | (xx >= w-t) | ((yy >= (h-t)//2) & (yy < (h+t)//2))
    elif family == "plus":
        mask = ((xx >= (w-t)//2) & (xx < (w+t)//2)) | ((yy >= (h-t)//2) & (yy < (h+t)//2))
    elif family == "frame":
        t = min(t, (min(h, w)-1)//2)
        mask = np.minimum.reduce([yy, xx, h-1-yy, w-1-xx]) < t
    elif family == "staircase":
        steps = int(rng.integers(2, min(5, h, w)+1))
        mask = xx < np.ceil((np.minimum(steps-1, yy*steps//h)+1)*w/steps)
    elif family == "chamfer":
        k = int(rng.integers(1, max(2, min(h, w)//3+1)))
        mask = (yy+xx >= k) & (yy+w-1-xx >= k) & (h-1-yy+xx >= k) & (h-1-yy+w-1-xx >= k)
    elif family == "diamond":
        # Every row contains the center column, including elongated diamonds.
        radius = np.maximum(0, np.floor((1-abs(yy-(h-1)/2)/max(.5, (h-1)/2))*(w-1)/2))
        mask = abs(xx-(w-1)//2) <= radius
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return crop(mask), family


def halo(mask, diagonal=True):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ()):
        result |= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
    return result


def placement_map(mask, free):
    h, w = free.shape
    sh, sw = mask.shape
    if sh > h or sw > w:
        return None
    ok = np.ones((h-sh+1, w-sw+1), bool)
    for y, x in np.argwhere(mask):
        ok &= free[y:y+ok.shape[0], x:x+ok.shape[1]]
    return ok


def place(rng, grid, mask, color, free):
    ok = placement_map(mask, free)
    if ok is None:
        return None
    spots = np.argwhere(ok)
    if not len(spots):
        return None
    r, c = map(int, spots[int(rng.integers(len(spots)))])
    occupied = np.zeros(grid.shape, bool)
    occupied[r:r+mask.shape[0], c:c+mask.shape[1]] = mask
    grid[occupied] = color
    return occupied


def decorate(rng, grid, protected, bg, colors, style, *, density=.18, max_pattern_fraction=None):
    """Paint only nonsemantic colors; return actual decoration metadata.

    Every requested accent is represented by a small clean shape if the
    pattern did not already use it. A palette failure rejects the scene.
    """
    colors = list(colors)
    if not colors:
        return {"background_style": "solid", "decoration_families": []}
    h, w = grid.shape
    yy, xx = np.indices((h, w))
    period = int(rng.integers(3, 8))
    offset = int(rng.integers(period))
    axis = yy if rng.integers(2) else xx
    if style == "lattice":
        marked = (yy % period == offset) & (xx % period == offset)
        slots = yy//period + xx//period
    elif style == "stripes":
        marked = (axis+offset) % period == 0
        slots = axis//period
    elif style == "diagonal":
        marked = (yy+xx+offset) % period == 0
        slots = (yy+xx)//period
    elif style == "checker":
        marked = (yy//period+xx//period) % 2 == 0
        slots = yy//period+2*(xx//period)
    elif style == "split":
        extent = h if np.array_equal(axis, yy) else w
        marked = axis >= max(1, int(round(extent*.70)))
        slots = yy//period+xx//period
    else:
        marked = np.zeros(grid.shape, bool)
        slots = yy+xx
    marked &= ~protected & (grid == bg)
    available = np.argwhere((grid == bg) & ~protected)
    if not len(available):
        raise ValueError("No background cell remains")
    reserve = tuple(available[int(rng.integers(len(available)))])
    marked[reserve] = False
    if max_pattern_fraction is not None and marked.sum() > grid.size*max_pattern_fraction:
        cells = np.argwhere(marked)
        # Crop a coherent part of the pattern instead of turning it into noise.
        order = np.argsort(cells[:, int(rng.integers(2))], kind="stable")
        keep = cells[order[:int(grid.size*max_pattern_fraction)]]
        marked[:] = False
        marked[keep[:, 0], keep[:, 1]] = True
    grid[marked] = np.asarray(colors)[slots[marked] % len(colors)]
    families = []
    used = protected | (grid != bg)
    used[reserve] = True
    # Reserve one patch for every missing palette color, then optional clutter.
    required = [c for c in colors if c not in grid]
    extra = int(rng.integers(0, max(2, min(10, grid.size//45)+1)))
    for col in required + [int(rng.choice(colors)) for _ in range(extra)]:
        for trial in range(12):
            max_h = max(1, min(h, int(round(h*np.sqrt(density/max(1, len(colors)))))))
            max_w = max(1, min(w, int(round(w*np.sqrt(density/max(1, len(colors)))))))
            sh = int(rng.integers(1 if max_h < 3 else 2, max_h+1))
            sw = int(rng.integers(1 if max_w < 3 else 2, max_w+1))
            mask, fam = shape(rng, sh, sw)
            if mask.sum() < 2 and grid.size >= 30 and trial < 10:
                continue
            stamped = place(rng, grid, mask, col, ~used)
            if stamped is not None:
                used |= stamped
                families.append(fam)
                break
        if col not in grid:
            free = np.argwhere(~used)
            if not len(free):
                raise ValueError("Not enough space for the requested palette")
            r, c = free[int(rng.integers(len(free)))]
            grid[r, c] = col
            used[r, c] = True
            families.append("cell")
    return {"background_style": style if marked.any() else "solid", "decoration_families": families}
