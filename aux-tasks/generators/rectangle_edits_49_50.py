"""Rectangle editing with repeated colors, distractors and sparse backgrounds.

Construction records geometry.  Validation instead recognizes complete visible
components and reconstructs the entire output.  Task 49 chooses the requested
width only after the input is finished; small rectangles with no remaining
interior stay unchanged, as stated in the prompt.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from common.backgrounds import sparse_mask
from generators.rigid_scene import capacity, slots, infer_ground
from generators.exterior_field import infer_exterior_field

BACKGROUND_KINDS = ("lattice", "dotted_stripes", "sparse_checker", "dotted_border", "dotted_diagonal")

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5


def choose_thickness(rng):
    return int(rng.choice((1, 2, 3), p=(.6, .25, .15)))


def components(grid, background):
    a = np.asarray(grid)
    h, w = a.shape
    seen = set()
    groups = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or a[r, c] == background:
                continue
            color = a[r, c].item()
            group = {(r, c)}
            seen.add((r, c))
            queue = deque([(r, c)])
            while queue:
                y, x = queue.popleft()
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and (yy, xx) not in seen and a[yy, xx] == color:
                        group.add((yy, xx))
                        seen.add((yy, xx))
                        queue.append((yy, xx))
            groups.append((color, group))
    return groups


def rectangles(grid, *, hollow):
    """Recognize exact monochrome component geometry, not color bounding boxes."""
    a = np.asarray(grid)
    bg = infer_ground(a) if hollow else infer_exterior_field(a)
    result = []
    for color, cells in components(a, bg):
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < (3 if hollow else 2):
            continue
        if not hollow:
            if len(cells) == h * w:
                result.append((r0, c0, h, w, color, 0))
            continue
        for t in range(1, (min(h, w) - 1) // 2 + 1):
            expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                        if min(r - r0, r1 - r, c - c0, c1 - c) < t}
            if cells == expected:
                result.append((r0, c0, h, w, color, t))
                break
    return sorted(result)


def solve(grid, *, fill, thickness=1):
    a = np.asarray(grid)
    bg = infer_ground(a) if fill else infer_exterior_field(a)
    out = a.copy()
    for r, c, h, w, color, wall in rectangles(a, hollow=fill):
        t = wall if fill else thickness
        # Explicitly empty interiors avoid Python's negative-slice semantics.
        if h > 2 * t and w > 2 * t:
            for y in range(r + t, r + h - t):
                for x in range(c + t, c + w - t):
                    out[y, x] = color if fill else bg
    return out.tolist()


def validate_example(ex, *, fill):
    a = np.asarray(ex["input"])
    m = ex["metadata"]
    if a.ndim != 2 or not all((3 if fill else 4) <= x <= 30 for x in a.shape) or a.size < 21:
        raise ValueError("Invalid rectangle-edit canvas")
    field = infer_ground(a) if fill else infer_exterior_field(a)
    if len(np.unique(a)) != m["n_colors"] or field != m["bg_color"]:
        raise ValueError("Rectangle-edit palette or field is not identifiable")
    if solve(a, fill=fill, thickness=m.get("border_thickness", 1)) != ex["output"]:
        raise ValueError("Rectangle edit differs from the independent visible-grid solution")
    found = rectangles(a, hollow=fill)
    if [list(x) for x in found] != m["rectangles"]:
        raise ValueError("Planned rectangles differ from visible targets")


def _layout_choices(h, w, n):
    choices = []
    for axis, extent in enumerate((h, w)):
        for cut in range(3, extent - 1):
            ph, pw = (cut, w) if axis == 0 else (h, cut)
            sh, sw = (h - cut - 1, w) if axis == 0 else (h, w - cut - 1)
            if capacity(sh, sw) >= n - 1:
                choices.append((axis, cut, ph, pw, sh, sw))
    return choices


def _boxes(rng, h, w, n, large):
    if n == 1:
        return [(0, 0, h, w)]
    choices = _layout_choices(h, w, n)
    if large:
        roomy = [v for v in choices if min(v[2:4]) >= 7]
        if roomy:
            choices = roomy
    spacious = [v for v in choices if ((v[4] + 1) // 4) * ((v[5] + 1) // 4) >= n - 1]
    if spacious:
        choices = spacious
    axis, cut, ph, pw, sh, sw = choices[int(rng.integers(len(choices)))]
    sr, sc = (cut + 1, 0) if axis == 0 else (0, cut + 1)
    feasible = [m for m in range(2, 6) if ((sh + 1) // (m + 1)) * ((sw + 1) // (m + 1)) >= n - 1]
    rest = slots(rng, sh, sw, n - 1, sr, sc, minimum=max(feasible, default=1))
    result = [(0, 0, ph, pw)] + rest
    # Reflections distribute the primary object over all borders/corners.
    flip_y, flip_x = bool(rng.integers(2)), bool(rng.integers(2))
    return [(h - r - hh if flip_y else r, w - c - ww if flip_x else c, hh, ww)
            for r, c, hh, ww in result]


def _ring(h, w, t):
    yy, xx = np.mgrid[:h, :w]
    return np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]) < t


def _distractor(rng, h, w, fill):
    if min(h, w) < 3:
        mask = np.zeros((h, w), bool)
        if h >= w:
            mask[:, 0] = True
        else:
            mask[0, :] = True
        return mask, "bar"
    families = ["L", "T", "U", "plus", "open_frame", "filled_rect" if fill else "frame"]
    if min(h, w) >= 5:
        families.append("chamfered_ring")
    kind = str(rng.choice(families))
    mask = np.zeros((h, w), bool)
    t = int(rng.integers(1, max(2, min(h, w) // 3 + 1)))
    if kind in ("frame", "open_frame"):
        t = min(t, (min(h, w) - 1) // 2)
        mask = _ring(h, w, t)
        if kind == "open_frame":
            gap = int(rng.integers(1, max(2, min(4, w - 2 * t + 1))))
            start = int(rng.integers(t, w - t - gap + 1))
            mask[:t, start:start + gap] = False
    elif kind == "filled_rect":
        mask[:] = True
    elif kind == "L":
        mask[:, :t] = mask[-t:, :] = True
    elif kind == "T":
        mask[:t, :] = True
        mask[:, (w - t) // 2:(w - t) // 2 + t] = True
    elif kind == "U":
        mask[:, :t] = mask[:, -t:] = mask[-t:, :] = True
    elif kind == "plus":
        mask[(h - t) // 2:(h - t) // 2 + t, :] = True
        mask[:, (w - t) // 2:(w - t) // 2 + t] = True
    else:
        t = max(2, min(4, (min(h, w) - 1) // 2))
        mask = _ring(h, w, t)
        k = max(1, t - 1)
        mask[:k, :k] = mask[:k, -k:] = mask[-k:, :k] = mask[-k:, -k:] = False
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask.copy(), kind


def _halo(mask):
    h, w = mask.shape
    out = np.zeros_like(mask)
    for y in range(-1, 2):
        for x in range(-1, 2):
            out[max(0, y):min(h, h + y), max(0, x):min(w, w + x)] |= mask[max(0, -y):min(h, h - y), max(0, -x):min(w, w - x)]
    return out


def _allowance(area, current, color, reserved):
    """Exact integer budget preserving a >1.25 field margin after reservations."""
    base = np.asarray(current) + np.asarray(reserved)
    free = area - int(base.sum())
    return min((4 * free - 5 * int(v) - 1) // (9 if co == color else 4)
               for co, v in enumerate(base))


def generate(rng, *, fill):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
        if h * w >= 21:
            break
    # Reserve actual exterior context before allocating objects. The reserved
    # slack can lie on either side, so clear edge and corner scenes remain.
    lh, lw = (h, w) if fill else (h - (2 if h >= 5 else 1), w - (2 if w >= 5 else 1))
    max_objects = max(n for n in range(1, min(16, capacity(lh, lw), max(1, (h * w - 19) // 2)) + 1)
                      if n == 1 or _layout_choices(lh, lw, n))
    n_colors = int(rng.integers(2, min(10, max_objects + 1) + 1))
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    bg, colors = palette[0], palette[1:]
    extra = 2 if not fill and len(colors) == 1 else 5
    choices = [k for k in range(len(colors), min(max_objects, len(colors) + extra) + 1)
               if h * w - 9 - 2 * (k - 1) > 1.25 * (9 + 2 * ((k + len(colors) - 1) // len(colors) - 1))]
    n = int(rng.choice(choices))
    zero = bool(rng.random() < (.05 if fill else .015))
    mode = str(rng.choice(("single", "mixed", "many"), p=(.18, .42, .40)))
    large = bool(rng.random() < .85)
    patterned = min(h, w) >= 8 and h * w >= 160 and rng.random() < .40
    kind = str(rng.choice(BACKGROUND_KINDS)) if patterned else "solid"
    for attempt in range(1, 151):
        boxes = _boxes(rng, lh, lw, n, large)
        if not fill:
            # Interior placements lead; both orientations of each boundary
            # placement are available without changing the object budget.
            ro = h - lh
            co = w - lw
            dy = 1 if ro == 2 and rng.random() < .70 else int(rng.integers(ro + 1))
            dx = 1 if co == 2 and rng.random() < .70 else int(rng.integers(co + 1))
            boxes = [(r + dy, c + dx, hh, ww) for r, c, hh, ww in boxes]
        object_colors = (colors * ((n + len(colors) - 1) // len(colors)))[:n]
        rng.shuffle(object_colors)
        inp = np.full((h, w), bg, dtype=np.uint8)
        target_records, all_objects, families = [], [], []
        target_object_indices = {}
        reserved = np.zeros(10, dtype=int)
        for color in object_colors[1:]:
            reserved[color] += 2
        primary_limit = _allowance(h * w, np.zeros(10, int), object_colors[0], reserved)
        primary_side = min(7, boxes[0][2], boxes[0][3], int(np.sqrt(primary_limit)))
        complete = True
        for i in list(range(1, n)) + [0]:
            r, c, sh, sw = boxes[i]
            color = object_colors[i]
            target = not zero and min(sh, sw) >= 3 and (i == 0 or (mode != "single" and rng.random() < (.85 if mode == "many" else .55)))
            if i == 0:
                other_area = int((inp != bg).sum())
                same_color_extra = int((inp == color).sum())
                largest_other = max((int((inp == co).sum()) for co in colors if co != color), default=0)
                remaining = h * w - other_area
                limit = min((4 * remaining - 5 * same_color_extra - 1) // 9,
                            (4 * remaining - 5 * largest_other - 1) // 4)
                # A target must leave a visible route around it on both axes.
                # In particular, the old large-object preference forced full
                # canvas bands on narrow grids before the spatial guard ran.
                max_h, max_w = (sh, sw) if fill else (min(sh, h - 1), min(sw, w - 1))
                sizes = [(hh, ww) for hh in range(3, max_h + 1) for ww in range(3, max_w + 1)
                         if (2 * (hh + ww) - 4 if fill and target else hh * ww) <= limit]
                if not sizes:
                    complete = False
                    break
                if large:
                    minimum = min(7, max(min(v) for v in sizes))
                    sizes = [v for v in sizes if min(v) >= minimum]
                hh, ww = sizes[int(rng.integers(len(sizes)))]
            else:
                if not fill and large:
                    reserved = np.zeros(10, dtype=int)
                    reserved[object_colors[0]] = primary_side ** 2
                    for co in object_colors[i + 1:]:
                        reserved[co] += 2
                    current = np.bincount(inp[inp != bg], minlength=10)
                    allowance = _allowance(h * w, current, color, reserved)
                    sizes = [(hh, ww) for hh in range(3, min(sh, h - 1) + 1)
                             for ww in range(3, min(sw, w - 1) + 1) if hh * ww <= allowance]
                    if not sizes:
                        target = False
                        sizes = [(hh, ww) for hh in range(1, min(sh, h - 1) + 1)
                                 for ww in range(1, min(sw, w - 1) + 1) if 2 <= hh * ww <= allowance]
                    if not sizes:
                        complete = False
                        break
                    hh, ww = sizes[int(rng.integers(len(sizes)))]
                else:
                    hh = max(1, int(round(sh * rng.uniform(.45, .95))))
                    ww = max(1, int(round(sw * rng.uniform(.45, .95))))
                    if target:
                        hh, ww = max(3, hh), max(3, ww)
                    elif hh * ww < 2:
                        hh, ww = (2, 1) if sh >= 2 else (1, 2)
                    if not fill:
                        hh, ww = min(hh, h - 1), min(ww, w - 1)
            rr, cc = r + int(rng.integers(sh - hh + 1)), c + int(rng.integers(sw - ww + 1))
            if target:
                widths = [t for t in range(1, min(5, (min(hh, ww) - 1) // 2) + 1)
                          if i != 0 or 2 * t * (hh + ww) - 4 * t * t <= limit]
                wall = int(rng.choice(widths)) if fill else 0
                mask = _ring(hh, ww, wall) if fill else np.ones((hh, ww), bool)
                family = "frame" if fill else "filled_rect"
                target_records.append((rr, cc, hh, ww, color, wall))
                target_object_indices[target_records[-1]] = len(all_objects)
            else:
                mask, family = _distractor(rng, hh, ww, fill)
            inp[rr:rr + hh, cc:cc + ww][mask] = color
            all_objects.append(int(mask.sum()))
            families.append(family)
        if not complete:
            continue
        if fill and len(target_records) > 1:
            # Removing the outer corners of a thick frame leaves a simple,
            # closed nonrectangular outline. Keep another actual target and
            # prefer a conversion that also preserves mixed target widths.
            candidates = [v for v in target_records if v[5] >= 2]
            varied = [v for v in candidates if len({x[5] for x in target_records if x != v}) > 1]
            if candidates and rng.random() < .5:
                choices = varied or candidates
                record = choices[int(rng.integers(len(choices)))]
                rr, cc, hh, ww, color, wall = record
                k = wall - 1
                sub = inp[rr:rr + hh, cc:cc + ww]
                sub[:k, :k] = sub[:k, -k:] = sub[-k:, :k] = sub[-k:, -k:] = bg
                index = target_object_indices[record]
                all_objects[index] -= 4 * k * k
                families[index] = "chamfered_ring"
                target_records.remove(record)
        texture_cells = 0
        if patterned:
            dots = sparse_mask(rng, h, w, kind) & ~_halo(inp != bg)
            for j, (y, x) in enumerate(np.argwhere(dots)):
                inp[y, x] = colors[j % len(colors)]
            texture_cells = int(dots.sum())
            if not texture_cells:
                continue
        try:
            field = infer_ground(inp) if fill else infer_exterior_field(inp)
            if field != bg or len(np.unique(inp)) != n_colors:
                continue
        except ValueError:
            continue
        # Reject phantom candidates, collapsed distractors and accidental joins.
        found = rectangles(inp, hollow=fill)
        if found != sorted(target_records):
            continue
        # No requested-thickness-dependent retry or placement.
        requested = choose_thickness(rng) if not fill else None
        out = inp.copy()
        for rr, cc, hh, ww, color, wall in target_records:
            t = wall if fill else requested
            if hh > 2 * t and ww > 2 * t:
                out[rr + t:rr + hh - t, cc + t:cc + ww - t] = color if fill else bg
        values, counts = np.unique(inp, return_counts=True)
        field_count = int(counts[values == bg][0])
        ex = {"input": inp.tolist(), "output": out.tolist(), "metadata": {
            "bg_color": bg, "background": kind, "background_margin": field_count / int(max(counts[values != bg])),
            "n_colors": n_colors, "grid_h": h, "grid_w": w, "rectangles": [list(x) for x in found],
            "rectangle_count": len(found), "object_count": n, "shape_families": families,
            "object_areas": all_objects, "foreground_cells": int(sum(all_objects)),
            "texture_cells": texture_cells, "target_mode": mode, "planned_zero": zero,
            "identity": bool(np.array_equal(inp, out)), "changed_cells": int((inp != out).sum()),
            "attempts": attempt}}
        if not fill:
            ex["metadata"]["border_thickness"] = requested
            ex["metadata"]["background_certificate"] = "connected_exterior"
        validate_example(ex, fill=fill)
        return ex
    raise RuntimeError("Could not realize the fixed rectangle scene and palette")
