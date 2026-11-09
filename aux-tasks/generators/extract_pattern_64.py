"""Extract a drawn foreground from a recoverable periodic or symmetric field.

The reader searches color partitions and background models from the final grid.
It neither knows the planted shape nor chooses a foreground by color frequency.
Every equality class needs visible witnesses; a hidden class is not evidence.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from generators.scene_shapes_56_60 import shape


SYMMETRIES = ("horizontal_mirror", "vertical_mirror", "both_axis_mirror", "rotational_180")
SHAPES = ("rectangle", "L", "T", "U", "H", "plus", "frame", "staircase", "chamfer", "ellipse", "bar")


@lru_cache(maxsize=256)
def _classes(h, w, family, p=1, q=1):
    """Reader labels, independently of the reflection/repetition constructor."""
    yy, xx = np.indices((h, w))
    if family == "periodic":
        labels = (yy % p)*q + xx % q
    elif family == "horizontal_mirror":
        labels = np.minimum(yy, h-1-yy)*w + xx
    elif family == "vertical_mirror":
        labels = yy*w + np.minimum(xx, w-1-xx)
    elif family == "both_axis_mirror":
        labels = np.minimum(yy, h-1-yy)*w + np.minimum(xx, w-1-xx)
    elif family == "rotational_180":
        labels = np.minimum(yy*w+xx, (h-1-yy)*w+w-1-xx)
    else:
        raise ValueError(f"Unknown pattern family {family}")
    return np.unique(labels, return_inverse=True)[1].reshape(h, w)


def _witnesses(indices, visible, labels, minimum):
    used = labels[visible]
    counts = np.bincount(used, minlength=int(labels.max())+1)
    if counts.min() < minimum:
        return False
    colors = np.zeros(len(counts), dtype=np.int64)
    np.bitwise_or.at(colors, used, 1 << indices[visible])
    return bool(np.all((colors & (colors-1)) == 0))


def pattern_model(grid, foreground):
    """Find a visible certificate; no proposed period or symmetry is supplied."""
    grid = np.asarray(grid)
    h, w = grid.shape
    visible = ~foreground
    _, indices = np.unique(grid, return_inverse=True)
    indices = indices.reshape(h, w)
    if len(np.unique(indices[visible])) < 2:
        return None
    # A shift is only a prefilter. Full residue classes below must agree and
    # have at least two witnesses, including across the occluded portion.
    rows = [p for p in range(1, h//2+1)
            if np.all((indices[p:] == indices[:-p]) | ~visible[p:] | ~visible[:-p])]
    cols = [q for q in range(1, w//2+1)
            if np.all((indices[:, q:] == indices[:, :-q]) | ~visible[:, q:] | ~visible[:, :-q])]
    for p in rows:
        for q in cols:
            if _witnesses(indices, visible, _classes(h, w, "periodic", p, q), 2):
                return {"family": "periodic", "period": [p, q]}
    for family in SYMMETRIES:
        if _witnesses(indices, visible, _classes(h, w, family), 1):
            return {"family": family}
    return None


def _component_graph(indices):
    """Monochrome components permit fast exact union-connectivity checks."""
    h, w = indices.shape
    labels = np.full((h, w), -1, int)
    groups = [0]*(int(indices.max())+1)
    k = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] >= 0:
                continue
            col = indices[r, c]
            groups[col] |= 1 << k
            labels[r, c] = k
            stack = [(r, c)]
            while stack:
                y, x = stack.pop()
                for a, b in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= a < h and 0 <= b < w and labels[a, b] < 0 and indices[a, b] == col:
                        labels[a, b] = k
                        stack.append((a, b))
            k += 1
    neighbors = [0]*k
    for left, right in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
        for a, b in zip(left.ravel(), right.ravel()):
            a, b = int(a), int(b)
            if a != b:
                neighbors[a] |= 1 << b
                neighbors[b] |= 1 << a
    return groups, neighbors


def candidates(grid):
    """All connected color unions with a fully witnessed background reading.

    Search every color subset, including unions larger than the generator's
    three-color objects. Require background context along both canvas axes:
    unions of full-width bands are fields, not localized foreground drawings.
    Do not prune by rarity, area or the planted family.
    """
    grid = np.asarray(grid)
    colors, indices = np.unique(grid, return_inverse=True)
    indices = indices.reshape(grid.shape)
    n = len(colors)
    eligible = 0
    for i in range(n):
        ys, xs = np.nonzero(indices == i)
        if ys.max()-ys.min()+1 < grid.shape[0] and xs.max()-xs.min()+1 < grid.shape[1]:
            eligible |= 1 << i
    groups, neighbors = _component_graph(indices)
    unions = [0]*(1 << n)
    pixel_bits = 1 << indices
    for subset in range(1, 1 << n):
        low = subset & -subset
        active = unions[subset ^ low] | groups[low.bit_length()-1]
        unions[subset] = active
        if subset.bit_count() > n-2 or subset & ~eligible:
            continue
        seen = todo = active & -active
        while todo:
            first = todo & -todo
            todo ^= first
            new = neighbors[first.bit_length()-1] & active & ~seen
            seen |= new
            todo |= new
        if seen != active:
            continue
        mask = (pixel_bits & subset) != 0
        ys, xs = np.nonzero(mask)
        if ys.max()-ys.min()+1 == grid.shape[0] or xs.max()-xs.min()+1 == grid.shape[1]:
            continue
        model = pattern_model(grid, mask)
        if model is not None:
            yield mask, model


def solve(grid, canvas_color):
    grid = np.asarray(grid)
    answer = None
    for mask, _ in candidates(grid):
        output = np.full_like(grid, canvas_color)
        output[mask] = grid[mask]
        if answer is not None and not np.array_equal(output, answer):
            raise ValueError("Ambiguous foreground: different witnessed extractions")
        answer = output
    if answer is None:
        raise ValueError("No fully witnessed connected foreground")
    return answer


def validate(example):
    grid = np.asarray(example["input"])
    if grid.ndim != 2 or not all(5 <= s <= 30 for s in grid.shape):
        raise ValueError("Task 64 sides must be in 5..30")
    answer = solve(grid, example["metadata"]["canvas_color"])
    if not np.array_equal(answer, example["output"]):
        raise ValueError("Task 64 output disagrees with visible extraction")


def _parts(rng, h, w, n):
    boxes = [(0, 0, h, w)]
    while len(boxes) < n:
        choices = [i for i, (y, x, a, b) in enumerate(boxes) if a*b >= 2]
        i = int(rng.choice(choices))
        y, x, a, b = boxes.pop(i)
        if a > 1 and (b == 1 or rng.integers(2)):
            cut = int(rng.integers(1, a))
            boxes += [(y, x, cut, b), (y+cut, x, a-cut, b)]
        else:
            cut = int(rng.integers(1, b))
            boxes += [(y, x, a, cut), (y, x+cut, a, b-cut)]
    grid = np.empty((h, w), int)
    for color, (y, x, a, b) in enumerate(boxes):
        grid[y:y+a, x:x+b] = color
    return grid


def _content(rng, h, w, colors, style):
    n = len(colors)
    yy, xx = np.indices((h, w))
    if style == "bands":
        along_rows = h >= n and (w < n or bool(rng.integers(2)))
        length = h if along_rows else w
        cuts = sorted(rng.choice(np.arange(1, length), n-1, replace=False))
        labels = np.searchsorted(cuts, yy if along_rows else xx, side="right")
    elif style == "diagonal_bands":
        labels = np.minimum(n-1, (yy+xx)*n//(h+w-1))
    elif style == "checks":
        # Alternating equal blocks with cycling colors. All phases appear.
        labels = (yy+xx) % n
    elif style == "dots":
        labels = np.zeros((h, w), int)
        locations = [(r, c) for r in range(1, h, 2) for c in range(1, w, 2)]
        rng.shuffle(locations)
        for i, (r, c) in enumerate(locations[:max(n-1, len(locations)//2)]):
            labels[r, c] = 1+i % (n-1)
    elif style == "frames":
        d = np.minimum.reduce([yy, xx, h-1-yy, w-1-xx])
        labels = d % n
    elif style == "motif":
        labels = np.zeros((h, w), int)
        mh, mw = h-2, w-2
        mask, _ = shape(rng, mh, mw, str(rng.choice(["rectangle", "L", "T", "plus", "frame"])))
        ys, xs = np.nonzero(mask)
        order = np.lexsort((xs, ys))
        labels[ys[order]+1, xs[order]+1] = 1 + np.minimum(n-2, np.arange(len(ys))*(n-1)//len(ys))
    elif style == "patchwork":
        labels = _parts(rng, h, w, n)
    else:
        raise ValueError(f"Unknown content {style}")
    return np.asarray(colors)[labels]


def _styles(h, w, n):
    styles = ["patchwork"]
    if max(h, w) >= n:
        styles += ["bands"]
    if min(h, w) >= 2 and h+w-1 >= n:
        styles += ["diagonal_bands", "checks"]
    if ((h+1)//2) >= n and ((w+1)//2) >= n:
        styles += ["frames"]
    if (h//2)*(w//2) >= n-1 and min(h, w) >= 3:
        styles += ["dots"]
    if min(h, w) >= 5 and n <= min(4, h+w-5):
        styles += ["motif"]
    return styles


def _reflect(grid, family):
    """Copy actual array halves; the reader uses separate class formulas."""
    grid = grid.copy()
    h, w = grid.shape
    if family in ("horizontal_mirror", "both_axis_mirror"):
        grid[(h+1)//2:] = grid[:h//2][::-1]
    if family in ("vertical_mirror", "both_axis_mirror"):
        grid[:, (w+1)//2:] = grid[:, :w//2][:, ::-1]
    if family == "rotational_180":
        flat = grid.ravel()
        flat[(h*w+1)//2:] = flat[:h*w//2][::-1]
    return grid


def _foreground(rng, h, w, family):
    if family == "ellipse":
        yy, xx = np.indices((h, w))
        return ((yy-(h-1)/2)/(h/2))**2 + ((xx-(w-1)/2)/(w/2))**2 <= 1
    return shape(rng, h, w, family)[0]


def generate(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    family = "periodic" if rng.random() < .67 else str(rng.choice(SYMMETRIES))
    n_obj = int(rng.choice([1, 2, 3], p=[.55, .30, .15]))
    if family == "periodic":
        p = int(rng.integers(1, min(9, h//2)+1))
        q = int(rng.integers(1, min(9, w//2)+1))
        if p*q == 1:
            if rng.integers(2):
                p = 2
            else:
                q = 2
        capacity = p*q
    else:
        p = h if family == "vertical_mirror" else (h+1)//2
        q = (w+1)//2 if family in ("vertical_mirror", "both_axis_mirror") else w
        capacity = p*q
    n_bg = int(rng.integers(2, min(10-n_obj, capacity)+1))
    colors = list(map(int, rng.permutation(10)[:n_obj+n_bg]))
    foreground_colors, background_colors = colors[:n_obj], colors[n_obj:]
    canvas = int(rng.choice([c for c in range(10) if c not in foreground_colors]))
    # Draw the whole independent half/quadrant before reflection. Single-axis
    # fields keep their full perpendicular extent, without stretched edge bands.
    style = str(rng.choice(_styles(p, q, n_bg)))
    cap_h, cap_w = h-1, w-1
    axis = None
    if family == "periodic":
        # At most k-1 of the k complete repeats can be covered on this axis;
        # a whole visible phase-row/column supplies two witnesses per class.
        if rng.integers(2):
            cap_h = (h//p-1)*p
        else:
            cap_w = (w//q-1)*q
    else:
        axis = (0 if family == "horizontal_mirror" else 1 if family == "vertical_mirror"
                else int(rng.integers(2)))
        if axis == 0:
            cap_h = (h-1)//2
        else:
            cap_w = (w-1)//2
    if n_obj > 1:
        # A partial color union must not erase the mirror partner of another
        # object color and turn that surviving piece into a background motif.
        cap_h, cap_w = min(cap_h, (h-1)//2), min(cap_w, (w-1)//2)
    available = [f for f in SHAPES if min(cap_h, cap_w) >=
                 (5 if f == "chamfer" else 3 if f in ("T", "U", "H", "plus", "frame", "ellipse") else 2)
                 and (f != "bar" or max(cap_h, cap_w) >= 3)]
    weights = np.array([2.2 if f == "rectangle" else 1.4 if f in ("ellipse", "chamfer")
                        else .55 if f == "bar" else 1. for f in available])
    obj_family = str(rng.choice(available, p=weights/weights.sum()))
    minimum = 5 if obj_family == "chamfer" else 3 if obj_family in ("T", "U", "H", "plus", "frame", "ellipse") else 2
    sh = int(rng.integers(max(minimum, int(.52*cap_h)), cap_h+1))
    sw = int(rng.integers(max(minimum, int(.52*cap_w)), cap_w+1))
    if obj_family == "bar" and max(sh, sw) < 3:
        if cap_h >= 3:
            sh = 3
        else:
            sw = 3
    # Occupancy scales with the canvas. Large filled bodies leave ample context.
    while sh*sw > .42*h*w and (sh > minimum or sw > minimum):
        if sh > minimum and (sw == minimum or sh/h > sw/w):
            sh -= 1
        else:
            sw -= 1
    for attempt in range(1, 161):
        tile = _content(rng, p, q, background_colors, style)
        if family == "rotational_180" and h % 2 and style in ("diagonal_bands", "checks"):
            # Only the left part of the center row is independent. Keep both
            # extrema of a diagonal ramp in the retained half, not its padding.
            tile = tile[:, ::-1]
        if family == "periodic":
            field = np.tile(tile, ((h+p-1)//p, (w+q-1)//q))[:h, :w]
        else:
            # Values outside the independent half/quadrant are overwritten by
            # reflection, so padding cannot become an extra content family.
            expanded = tile[np.minimum(np.arange(h), p-1)[:, None],
                            np.minimum(np.arange(w), q-1)[None, :]]
            field = _reflect(expanded, family)
        mask = _foreground(rng, sh, sw, obj_family)
        mh, mw = mask.shape
        if mask.sum() < max(3, n_obj):
            continue
        r = int(rng.integers(0, h-mh+1))
        c = int(rng.integers(0, w-mw+1))
        if axis == 0 or n_obj > 1:
            r = int(rng.integers(0, (h-1)//2-mh+1))
            if rng.integers(2):
                r = h-mh-r
        if axis == 1 or n_obj > 1:
            c = int(rng.integers(0, (w-1)//2-mw+1))
            if rng.integers(2):
                c = w-mw-c
        occupied = np.zeros((h, w), bool)
        occupied[r:r+mh, c:c+mw] = mask
        if len(np.unique(field[~occupied])) != n_bg:
            continue
        grid = field.copy()
        yy, xx = np.nonzero(occupied)
        order = np.lexsort((xx, yy)) if rng.integers(2) else np.lexsort((yy, xx))
        fill = np.asarray(foreground_colors)[np.minimum(n_obj-1, np.arange(len(yy))*n_obj//len(yy))]
        grid[yy[order], xx[order]] = fill
        output = np.full((h, w), canvas, int)
        output[occupied] = grid[occupied]
        # Every attempt keeps size, content family, palette and shape dimensions.
        try:
            actual = solve(grid, canvas)
        except ValueError:
            continue
        if not np.array_equal(actual, output):
            continue
        model = pattern_model(grid, occupied)
        return {"input": grid.tolist(), "output": output.tolist(), "metadata": {
            "canvas_color": canvas, "background_family": family,
            "background_style": style, "background_colors": background_colors,
            "object_colors": foreground_colors, "object_family": obj_family,
            "object_cells": int(occupied.sum()), "object_bbox": [r, c, mh, mw],
            "object_occupancy": float(occupied.mean()), "connectivity": 4,
            "planned_color_count": n_obj+n_bg, "input_color_count": len(np.unique(grid)),
            "planned_period": [p, q] if family == "periodic" else None,
            "visible_model": model, "unique_output": True, "attempts": attempt,
        }}
    raise RuntimeError(f"Task 64 could not realize fixed plan: {(h, w, family, style, obj_family, sh, sw, n_bg, n_obj, p, q)}")
