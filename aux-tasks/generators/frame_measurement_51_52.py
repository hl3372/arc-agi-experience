"""Visible rectangular-frame measurements, with the question drawn last.

Frame geometry, interior content, decoration, and palette are shared by every
largest/smallest and total/color-count question. Recognition checks complete
same-color components, including possible accidental frames in the field.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from common.backgrounds import sparse_mask
from generators.rigid_scene import slots, infer_ground, visible_objects
from generators.rectangle_edits_49_50 import _distractor, _halo, _ring, _allowance, BACKGROUND_KINDS
from generators.crop_frame_26 import _partition, _geometric_shape

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5


def frames(grid):
    """Recover exact frame components without being told any of their colors."""
    a = np.asarray(grid)
    result = []
    groups, _ = visible_objects(a)
    for cells, color in groups:
        r, c = min(y for y, _ in cells), min(x for _, x in cells)
        h = max(y for y, _ in cells) - r + 1
        w = max(x for _, x in cells) - c + 1
        if min(h, w) < 3:
            continue
        # Infer depth from component coordinates, independently of stamping.
        t = 1 + max(min(y-r, x-c, r+h-1-y, c+w-1-x) for y, x in cells)
        if 2*t >= min(h, w):
            continue
        expected = {(y, x) for y in range(r, r+h) for x in range(c, c+w)
                    if y < r+t or y >= r+h-t or x < c+t or x >= c+w-t}
        if cells == expected:
            result.append((r, c, h, w, color, t))
    return sorted(result)


def solve(grid, *, extreme=None, count_mode="all", target_color=None):
    a = np.asarray(grid)
    found = frames(a)
    if not found:
        raise ValueError("No rectangular frame")
    if extreme is None:
        return max(f[5] for f in found)
    if extreme not in ("largest", "smallest") or count_mode not in ("all", "color"):
        raise ValueError("Unknown rectangle measurement")
    area = (max if extreme == "largest" else min)(h*w for _, _, h, w, _, _ in found)
    selected = [f for f in found if f[2]*f[3] == area]
    if len(selected) != 1:
        raise ValueError("The extreme frame is not unique by outer area")
    r, c, h, w, _, t = selected[0]
    inner = a[r+t:r+h-t, c+t:c+w-t]
    return int(inner.size if count_mode == "all" else (inner == target_color).sum())


def validate_example(ex, *, thickness):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(3 <= d <= 30 for d in a.shape):
        raise ValueError("Invalid frame canvas")
    if infer_ground(a) != m["bg_color"] or len(np.unique(a)) != m["n_colors"]:
        raise ValueError("Incorrect palette or ambiguous background")
    if [list(f) for f in frames(a)] != m["rectangles"]:
        raise ValueError("Visible frames differ from the planned frames")
    if thickness:
        answer = solve(a)
    else:
        # Both questions must be valid before either one can be selected.
        for extreme in ("largest", "smallest"):
            solve(a, extreme=extreme)
        answer = solve(a, extreme=m["extreme_kind"], count_mode=m["count_mode"],
                       target_color=m.get("target_color"))
    if str(answer) != ex["output"]:
        raise ValueError("Frame answer differs from the visible-grid solution")


@lru_cache(maxsize=None)
def _contrasts(h, w):
    """Feasible large-thin/small-thick pairs, including wall-cell reversals."""
    result = []
    for axis in (0, 1):
        height, width = (h, w) if axis == 0 else (w, h)
        for thick in range(2, 7):
            side = 2*thick + 2
            if width < side:
                continue
            for thin in range(1, thick):
                for ah in range(2*thin+2, height-side):
                    aw = width
                    wall_a, wall_b = 2*thin*(ah+aw)-4*thin*thin, side*side-4
                    if ah*aw > side*side and wall_a > wall_b and wall_a+wall_b < .39*h*w:
                        result.append((axis, ah, aw, thin, side, thick))
    return tuple(result)


def _contrast_rects(rng, h, w):
    choices = _contrasts(h, w)
    axis, ah, aw, thin, side, thick = choices[int(rng.integers(len(choices)))]
    height, width = (h, w) if axis == 0 else (w, h)
    rr = int(rng.integers(ah+1, height-side+1))
    cc = int(rng.integers(width-side+1))
    shapes = [(0, 0, ah, aw, thin), (rr, cc, side, side, thick)]
    if axis:
        shapes = [(c, r, w0, h0, t) for r, c, h0, w0, t in shapes]
    fy, fx = rng.integers(2, size=2)
    return [(h-r-hh if fy else r, w-c-ww if fx else c, hh, ww, t)
            for r, c, hh, ww, t in shapes]


@lru_cache(maxsize=None)
def _aspect_contrasts(h, w):
    result = []
    for axis in (0, 1):
        height, width = (h, w) if axis == 0 else (w, h)
        for ah in range(4, height-3):
            for aw in range(max(4, ah-2), min(width, ah+2)+1):
                for bh in range(3, min(6, height-ah-1)+1):
                    for bw in range(max(ah, aw)+1, width+1):
                        if ah*aw > bh*bw and ah+aw < bh+bw and 2*(ah+aw+bh+bw)-8 < .35*h*w:
                            result.append((axis, ah, aw, bh, bw))
    return tuple(result)


def _aspect_rects(rng, h, w):
    options = _aspect_contrasts(h, w)
    axis, ah, aw, bh, bw = options[int(rng.integers(len(options)))]
    height, width = (h, w) if axis == 0 else (w, h)
    top = int(rng.integers(height-ah-bh))
    left_a, left_b = int(rng.integers(width-aw+1)), int(rng.integers(width-bw+1))
    t = 2 if min(ah, aw) >= 6 and rng.random() < .35 else 1
    shapes = [(top, left_a, ah, aw, t), (top+ah+1, left_b, bh, bw, 1)]
    if axis:
        shapes = [(c, r, ww, hh, t) for r, c, hh, ww, t in shapes]
    fy, fx = rng.integers(2, size=2)
    return [(h-r-hh if fy else r, w-c-ww if fx else c, hh, ww, t) for r, c, hh, ww, t in shapes]


def _thickness_geometry(rng, h, w, boxes, palette_n):
    """Plan a realizable width before choosing its frame's outer dimensions.

    A thick frame can be compact; it need not inherit the largest random box.
    Reserve the other walls and palette before spending the wall-cell budget.
    """
    n = len(boxes)
    budget = int((h*w-2*max(0, palette_n-2)-2)//2.25)
    r, c, sh, sw = boxes[0]
    feasible = []
    for t in range(1, 9):
        side = 3 if t == 1 else 2*t+2
        if side <= min(sh, sw) and 2*t*(2*side)-4*t*t <= budget-8*(n-1):
            feasible.append(t)
    weights = np.asarray(feasible, float)**1.5
    if len(feasible) > 1:
        weights[0] *= .20
    answer = int(rng.choice(feasible, p=weights/weights.sum()))
    result = []
    for i, (r, c, sh, sw) in enumerate(boxes):
        remaining = budget-8*(n-i-1)
        widths = [answer] if i == 0 else range(1, min(answer, 4)+1)
        options = []
        for t in widths:
            side = 3 if t == 1 else 2*t+2
            for hh in range(side, sh+1):
                for ww in range(side, sw+1):
                    wall = 2*t*(hh+ww)-4*t*t
                    if wall <= remaining:
                        options.append((hh, ww, t, wall))
        if i == 0 and rng.random() < .45:
            compact = [p for p in options if min(p[:2]) <= 2*answer+4]
            options = compact or options
        hh, ww, t, wall = options[int(rng.integers(len(options)))]
        result.append((r+int(rng.integers(sh-hh+1)), c+int(rng.integers(sw-ww+1)), hh, ww, t))
        budget -= wall
    fy, fx = rng.integers(2, size=2)
    return [(h-r-hh if fy else r, w-c-ww if fx else c, hh, ww, t) for r, c, hh, ww, t in result]


def _geometries(rng, h, w, n, mode, thickness, palette_n):
    if mode == "contrast":
        return _contrast_rects(rng, h, w)
    if mode == "aspect_contrast":
        return _aspect_rects(rng, h, w)
    floor = 4 if not thickness and min(h, w) >= 4 else 3
    possible = [m for m in range(floor, 10) if ((h+1)//(m+1))*((w+1)//(m+1)) >= n]
    minimum = max(possible) if mode == "comparable" else int(rng.choice(possible))
    boxes = slots(rng, h, w, n, minimum=minimum)
    anchor = False
    if thickness and mode == "varied" and min(h, w) >= 6 and h*w >= 18*n+48:
        options = []
        for axis, extent in enumerate((h, w)):
            for cut in range(6, extent-3):
                ph, pw = (cut, w) if axis == 0 else (h, cut)
                rh, rw = (h-cut-1, w) if axis == 0 else (h, w-cut-1)
                if ((rh+1)//4)*((rw+1)//4) >= n-1 and ph*pw >= .35*h*w:
                    options.append((axis, cut, ph, pw, rh, rw))
        if n == 1:
            boxes, anchor = [(0, 0, h, w)], True
        elif options:
            weights = np.asarray([min(p[2:4])**2 for p in options], float)
            axis, cut, ph, pw, rh, rw = options[int(rng.choice(len(options), p=weights/weights.sum()))]
            rr, cc = (cut+1, 0) if axis == 0 else (0, cut+1)
            boxes = [(0, 0, ph, pw)] + slots(rng, rh, rw, n-1, rr, cc, minimum=3)
            anchor = True
    if anchor:
        return _thickness_geometry(rng, h, w, boxes, palette_n)
    sizes = []
    compact = rng.random() < .25
    tiny = rng.random() < .12
    for i, (r, c, sh, sw) in enumerate(boxes):
        low = .0 if compact else (.25 if min(sh, sw) < 5 else (.72 if mode == "comparable" else .40))
        minimum_side = 3 if thickness or tiny else min(4, sh, sw)
        if (thickness and mode == "comparable" and min(sh, sw) >= 6
                and h*w > 45*n+2*max(0, palette_n-2)+4):
            minimum_side = 6
        hh = int(rng.integers(max(minimum_side, int(sh*low)), sh+1))
        ww = int(rng.integers(max(minimum_side, int(sw*low)), sw+1))
        sizes.append((r+int(rng.integers(sh-hh+1)), c+int(rng.integers(sw-ww+1)), hh, ww))
    caps = [min(8 if thickness else 5, max(1, (min(hh, ww)-2)//2)) for _, _, hh, ww in sizes]
    if thickness and mode == "equal":
        ts = [int(rng.integers(1, min(caps)+1))]*n
    elif thickness:
        values = np.arange(1, max(caps)+1)
        weights = values.astype(float)**1.6
        answer = int(rng.choice(values, p=weights/weights.sum()))
        winner = int(rng.choice([i for i, cap in enumerate(caps) if cap >= answer]))
        ts = [int(rng.integers(1, min(cap, answer, 2 if rng.random() < .75 else answer)+1)) for cap in caps]
        ts[winner] = answer
    else:
        # Varied walls, but retain substantial, readable interiors.
        ts = [int(rng.integers(1, min(cap, max(1, min(hh, ww)//5))+1))
              for (_, _, hh, ww), cap in zip(sizes, caps)]
    return [(*box, t) for box, t in zip(sizes, ts)]


def _interior(rng, h, w, bg, colors):
    a = np.full((h, w), bg, dtype=np.uint8)
    if not colors:
        return a, "empty"
    family = str(rng.choice(("empty", "solid", "bands", "blocks", "objects", "checker"),
                            p=(.12, .12, .23, .21, .24, .08)))
    pool = [bg, *colors]
    rng.shuffle(pool)
    if family == "solid":
        a[:] = int(rng.choice(pool))
    elif family == "bands":
        axis = int(rng.integers(2))
        extent = (h, w)[axis]
        n = int(rng.integers(1, min(extent, len(pool)+2)+1))
        cuts = sorted(map(int, rng.choice(np.arange(1, extent), n-1, replace=False)))
        for i, (lo, hi) in enumerate(zip([0, *cuts], [*cuts, extent])):
            if axis == 0:
                a[lo:hi, :] = pool[i % len(pool)]
            else:
                a[:, lo:hi] = pool[i % len(pool)]
    elif family == "blocks":
        n = int(rng.integers(1, min(h*w, max(3, len(pool)))+1))
        boxes = _partition(rng, np.ones((h, w), bool), n)
        rng.shuffle(boxes)
        for i, (r, c, hh, ww) in enumerate(boxes):
            a[r:r+hh, c:c+ww] = pool[i % len(pool)]
    elif family == "objects":
        for _ in range(int(rng.integers(1, 4))):
            hh, ww = int(rng.integers(1, h+1)), int(rng.integers(1, w+1))
            cells, _ = _geometric_shape(rng, hh, ww)
            r, c = int(rng.integers(h-hh+1)), int(rng.integers(w-ww+1))
            col = int(rng.choice(colors))
            for y, x in cells:
                a[r+y, c+x] = col
    elif family == "checker":
        rr, cc = np.indices((h, w))
        step = int(rng.integers(1, max(2, min(h, w)//3+1)))
        a = np.asarray(pool, dtype=np.uint8)[(rr//step + cc//step) % len(pool)]
    return a, family


def _decorate(rng, a, occupied, palette, bg, patterned):
    """Use clean nonframe objects for extra colors; texture remains optional."""
    h, w = a.shape
    missing = [co for co in palette[1:] if not np.any(a == co)]
    extras = missing + [int(rng.choice(palette[1:])) for _ in range(int(rng.integers(0, 3)))]
    families = []
    for i, color in enumerate(extras):
        current = np.bincount(a[a != bg], minlength=10)
        reserved = np.zeros(10, int)
        for co in missing[i+1:]:
            reserved[co] += 2
        allowance = _allowance(a.size, current, color, reserved)
        if allowance < 2:
            if i < len(missing):
                return None
            continue
        free = ~_halo(occupied)
        placed = False
        for trial in range(80):
            if trial < 45:
                hh = int(rng.integers(1, min(h, 9)+1))
                ww = int(rng.integers(1, min(w, 9)+1))
                if hh*ww < 2:
                    continue
                mask, family = _distractor(rng, hh, ww, True)
            else:
                hh, ww = (1, 2) if rng.integers(2) else (2, 1)
                mask, family = np.ones((hh, ww), bool), "bar"
            if int(mask.sum()) > allowance:
                continue
            points = np.argwhere(free)
            if not len(points):
                break
            r, c = map(int, points[int(rng.integers(len(points)))])
            if r+hh <= h and c+ww <= w and np.all(free[r:r+hh, c:c+ww][mask]):
                a[r:r+hh, c:c+ww][mask] = color
                occupied[r:r+hh, c:c+ww] |= mask
                families.append(family)
                placed = True
                break
        if not placed and i < len(missing):
            # On thin canvases the remaining room is often inside a frame or
            # in a narrow gutter. A short colored run there is still a clean
            # decoration; it cannot join a wall because this color is absent.
            candidates = []
            for r, c in np.argwhere(a == bg):
                for dr, dc in ((0, 1), (1, 0)):
                    if r+dr < h and c+dc < w and a[r+dr, c+dc] == bg:
                        candidates.append((int(r), int(c), dr, dc))
            if not candidates:
                return None
            r, c, dr, dc = candidates[int(rng.integers(len(candidates)))]
            a[r, c] = a[r+dr, c+dc] = color
            occupied[r, c] = occupied[r+dr, c+dc] = True
            families.append("short_run")
    kind, texture_cells = "plain", 0
    if patterned:
        kind = str(rng.choice(BACKGROUND_KINDS))
        mask = sparse_mask(rng, h, w, kind) & ~_halo(occupied)
        for i, (r, c) in enumerate(np.argwhere(mask)):
            color = palette[1+i % (len(palette)-1)]
            current = np.bincount(a[a != bg], minlength=10)
            if _allowance(a.size, current, color, np.zeros(10, int)) >= 1:
                a[r, c] = color
                texture_cells += 1
        if not texture_cells:
            kind = "plain"
    return families, kind, texture_cells


def choose_question(rng):
    return str(rng.choice(("largest", "smallest"))), str(rng.choice(("all", "color")))


def generate(rng, *, thickness):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, 2))
        if h*w >= 21:
            break
    footprint = 5 if not thickness and min(h, w) >= 4 else 4
    # Counting interiors needs room for two distinct extremes and useful
    # interiors between them, in addition to walls, gutters and clear field.
    cell_budget = 24 if thickness else 42
    cap = max(1, min(12, ((h+1)//footprint)*((w+1)//footprint), h*w//cell_budget))
    if cap == 1 or rng.random() < .08:
        n = 1
    elif cap <= 4 or rng.random() < .72:
        n = int(rng.integers(2, min(4, cap)+1))
    else:
        n = int(rng.integers(5, cap+1))
    modes = ["varied", "comparable", "equal"] if thickness else ["varied", "comparable"]
    mode = str(rng.choice(modes, p=(.65, .25, .10) if thickness else (.45, .55)))
    if thickness and _contrasts(h, w) and rng.random() < .32:
        mode, n = "contrast", 2
    if not thickness and _aspect_contrasts(h, w) and rng.random() < .30:
        mode, n = "aspect_contrast", 2
    palette_cap = min(10, max(2, 1+(h*w-18*n)//4))
    n_colors = int(rng.integers(2, palette_cap+1))
    palette = list(map(int, rng.permutation(10)[:n_colors]))
    bg = palette[0]
    # Keep some colors available for meaningful interior content even when
    # many frames share the same wall color.
    wall_cap = min(n, n_colors-1 if thickness else max(1, n_colors-2))
    nw = int(rng.integers(1, wall_cap+1))
    wall_pool, fill_pool = palette[1:1+nw], palette[1+nw:]
    patterned = min(h, w) >= 8 and rng.random() < .38
    for attempt in range(1, 301):
        geometry = _geometries(rng, h, w, n, mode, thickness, n_colors)
        areas = [hh*ww for _, _, hh, ww, _ in geometry]
        if not thickness and (areas.count(min(areas)) != 1 or areas.count(max(areas)) != 1):
            continue
        a = np.full((h, w), bg, dtype=np.uint8)
        occupied = np.zeros((h, w), bool)
        planned, fills, islands = [], [], 0
        for i, (r, c, hh, ww, t) in enumerate(geometry):
            color = int(rng.choice(wall_pool))
            sub = a[r:r+hh, c:c+ww]
            sub[_ring(hh, ww, t)] = color
            if not thickness:
                interior, family = _interior(rng, hh-2*t, ww-2*t, bg, fill_pool)
                ih, iw = interior.shape
                if min(ih, iw) >= 3 and rng.random() < .25:
                    # An interior island can share the wall color while being
                    # separated from the wall. Thus querying a wall color does
                    # not automatically imply a zero interior count.
                    bh = int(rng.integers(1, ih-1))
                    bw = int(rng.integers(1, iw-1))
                    yy, xx = int(rng.integers(1, ih-bh)), int(rng.integers(1, iw-bw))
                    island = np.ones((bh, bw), bool)
                    if min(bh, bw) >= 2 and rng.integers(2):
                        island[:-1, 1:] = False  # familiar L
                    interior[yy:yy+bh, xx:xx+bw][island] = color
                    islands += 1
                sub[t:hh-t, t:ww-t] = interior
                fills.append(family)
            occupied[r:r+hh, c:c+ww] = True
            planned.append((r, c, hh, ww, color, t))
        decoration = _decorate(rng, a, occupied, palette, bg, patterned)
        if decoration is None:
            continue
        try:
            if infer_ground(a) != bg:
                continue
        except ValueError:
            continue
        found = frames(a)
        if found != sorted(planned) or len(np.unique(a)) != n_colors:
            continue
        families, kind, texture_cells = decoration
        counts = np.bincount(a.ravel(), minlength=10)
        m = {"grid_h": h, "grid_w": w, "n_colors": n_colors, "bg_color": bg,
             "background_margin": float(counts[bg]/max(counts[c] for c in palette[1:])),
             "bg_style": kind, "texture_cells": texture_cells, "distractor_families": families,
             "rectangles": [list(f) for f in found], "n_rectangles": n, "n_requested": n,
             "rect_colors": [f[4] for f in found], "rect_sizes": [[f[2], f[3]] for f in found],
             "thicknesses": [f[5] for f in found], "interior_families": fills,
             "wall_color_islands": islands, "layout_mode": mode, "attempts": attempt}
        if thickness:
            answer = max(t for _, _, _, _, t in geometry)
            m.update(max_thickness=answer, n_at_max=sum(f[5] == answer for f in found),
                     all_equal_thickness=len({f[5] for f in found}) == 1)
        else:
            # No question-dependent retry: the complete input is already valid.
            extreme, count_mode = choose_question(rng)
            selected = (max if extreme == "largest" else min)(found, key=lambda f: f[2]*f[3])
            r, c, hh, ww, _, t = selected
            inner = a[r+t:r+hh-t, c+t:c+ww-t]
            # The field colour is never the counted colour (sweep 2026-09-17): the prompt opens
            # "Ignoring the background", so asking how many background-coloured cells lie inside the
            # frame asks for exactly the cells it just said to ignore.  The scene is untouched; only
            # the question is drawn differently, and a frame whose interior shows no other colour
            # gets the colour-free question instead of a guaranteed zero.
            present = [co for co in map(int, np.unique(inner)) if co != bg]
            if not present:
                count_mode = "all"
            m.update(extreme_kind=extreme, count_mode=count_mode, size_metric="outer_area",
                     extreme_rect=list(selected), interiors=[(f[2]-2*f[5])*(f[3]-2*f[5]) for f in found])
            if count_mode == "color":
                absent = [co for co in palette if co not in present and co != bg]
                target = int(rng.choice(absent if absent and rng.random() < .09 else present))
                m["target_color"] = target
                answer = int((inner == target).sum())
            else:
                answer = int(inner.size)
        m["answer"] = answer
        ex = {"input": a.tolist(), "output": str(answer), "metadata": m}
        validate_example(ex, thickness=thickness)
        return ex
    raise RuntimeError(f"Could not realize fixed {h}x{w} {mode} frame scene with {n} frames and {n_colors} colors")
