"""Cell-centered diamonds with input-based validation (tasks 71 and 72)."""
from __future__ import annotations

import numpy as np

from generators.geometry_fields_71_74 import field_metadata, field_plan, render_field


def _mask(h, w, center, radius, style="filled"):
    rr, cc = np.indices((h, w))
    d = abs(rr - center[0]) + abs(cc - center[1])
    return d <= radius if style == "filled" else d == radius


def solve_draw_diamonds(grid, marker_color, diamond_color, span):
    """Literal row-interval painting, independent of the builder's masks."""
    a = np.asarray(grid)
    if span < 3 or span % 2 != 1 or marker_color == diamond_color:
        raise ValueError("A diamond needs an odd span of at least three and distinct colors")
    radius = span // 2
    anchors = np.argwhere(a == marker_color)
    if not len(anchors):
        raise ValueError("No marked diamond centers")
    out = a.copy()
    for cr, cc in anchors:
        for row in range(max(0, int(cr) - radius), min(a.shape[0], int(cr) + radius + 1)):
            reach = radius - abs(row - int(cr))
            out[row, max(0, int(cc) - reach):min(a.shape[1], int(cc) + reach + 1)] = diamond_color
    return out.tolist()


def validate_draw_diamonds(grid, output, marker_color, diamond_color, span):
    if solve_draw_diamonds(grid, marker_color, diamond_color, span) != output:
        raise ValueError("Task 71: visible markers and span do not reproduce the output")


def _nearest(mask):
    """Exact Manhattan distance to a true cell, using four 1-D min passes."""
    a = np.where(mask, 0, 10000)
    for axis in (0, 1):
        pos = np.arange(a.shape[axis]).reshape((-1, 1) if axis == 0 else (1, -1))
        a = np.minimum.accumulate(a - pos, axis=axis) + pos
        a = np.flip(np.minimum.accumulate(np.flip(a + pos, axis=axis), axis=axis), axis=axis) - pos
    return a


def diamond_readings(grid):
    """All visible filled/one-cell-outline diamond readings in every color.

    A center outside the canvas clamps to its boundary while subtracting the
    clamping distance from its radius, preserving the visible mask. Consequently
    every off-grid ambiguity has a boundary-center witness. Such a witness can
    be moved one cell outward again to obtain a different full span.
    """
    a = np.asarray(grid)
    h, w = a.shape
    rr, cc = np.indices((h, w))
    u, v = rr + cc, rr - cc
    boundary = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
    readings = {}
    unbounded = False
    for color in np.unique(a):
        target = a == color
        if target.all():
            return {}, True
        rows, cols = np.nonzero(target)
        tu, tv = rows + cols, rows - cols
        low = np.maximum.reduce([tu.max() - u, u - tu.min(), tv.max() - v, v - tv.min()])
        high = _nearest(~target) - 1
        filled = (np.maximum(low, 1) <= high)
        # An outline requires all colored cells at exactly one radius, and no
        # missing visible cell at that radius. Count its whole visible perimeter.
        radius = low
        perimeter = np.zeros((h, w), int)
        for row in range(h):
            spread = radius - abs(row - rr)
            perimeter += (spread >= 0) & (cc - spread >= 0) & (cc - spread < w)
            perimeter += (spread > 0) & (cc + spread >= 0) & (cc + spread < w)
        outlined = (radius >= 1) & (_nearest(target) == radius) & (perimeter == int(target.sum()))
        # A radius-zero point on the boundary is also a clamped witness: one
        # outward step turns it into a legitimate radius-one clipped diamond.
        if (((low <= high) | ((_nearest(target) == radius) & (perimeter == int(target.sum())))) & boundary).any():
            unbounded = True
        for cr, c in np.argwhere(filled):
            for r in range(max(1, int(low[cr, c])), int(high[cr, c]) + 1):
                readings.setdefault(2 * r + 1, []).append((int(color) if np.issubdtype(a.dtype, np.integer) else str(color), int(cr), int(c), r, "filled"))
        for cr, c in np.argwhere(outlined):
            r = int(radius[cr, c])
            readings.setdefault(2 * r + 1, []).append((int(color) if np.issubdtype(a.dtype, np.integer) else str(color), int(cr), int(c), r, "outline"))
    return readings, unbounded


def solve_diamond_length(grid):
    readings, unbounded = diamond_readings(grid)
    if unbounded or len(readings) != 1:
        raise ValueError("Task 72: the visible diamond does not have one inferable full span")
    return str(next(iter(readings)))


def validate_diamond_length(grid, output):
    if solve_diamond_length(grid) != str(output):
        raise ValueError("Task 72: visible diamond span disagrees with the answer")


def _sample_radius(rng, cap, minimum=1):
    if cap < minimum:
        raise ValueError("No radius capacity")
    if rng.random() < .62:
        return int(rng.integers(minimum, cap + 1))
    return int(rng.integers(max(minimum, (cap + 1) // 2), cap + 1))


def _pack_centers(rng, h, w, radius, clipped, overlap):
    rr, cc = np.indices((h, w))
    interior = (rr >= radius) & (rr + radius < h) & (cc >= radius) & (cc + radius < w)
    available = np.ones((h, w), bool) if clipped else interior.copy()
    first_options = np.argwhere(available & (~interior if clipped else interior))
    if overlap:
        rlo, rhi = (0, h - 1) if clipped else (radius, h - radius - 1)
        clo, chi = (0, w - 1) if clipped else (radius, w - radius - 1)
        first_options = np.asarray([
            (r, c) for r, c in first_options
            if max(abs(r - rlo), abs(r - rhi)) + max(abs(c - clo), abs(c - chi)) > radius
        ])
    first = tuple(map(int, first_options[int(rng.integers(len(first_options)))]))
    selected = [first]
    mandatory = 1
    if overlap:
        distance = abs(rr - first[0]) + abs(cc - first[1])
        partners = np.argwhere(available & (distance > radius) & (distance <= 2 * radius))
        if not len(partners):
            raise RuntimeError("Task 71: overlap-capacity witness was invalid")
        second = tuple(map(int, partners[int(rng.integers(len(partners)))]))
        selected.append(second)
        mandatory = 2
    for cr, c in selected:
        available &= abs(rr - cr) + abs(cc - c) >= 2 * radius + 2
    while available.any() and len(selected) < 10:
        options = np.argwhere(available)
        cr, c = map(int, options[int(rng.integers(len(options)))])
        selected.append((cr, c))
        available &= abs(rr - cr) + abs(cc - c) >= 2 * radius + 2
    return selected, mandatory


def generate_draw_diamonds(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    clipped = bool(rng.random() < .58)
    cap = min(h, w) // 2 + 1 if clipped else (min(h, w) - 1) // 2
    radius = _sample_radius(rng, cap)
    span = 2 * radius + 1
    overlap_capable = h + w - 2 - (0 if clipped else 4 * radius) > radius
    overlap_requested = bool(overlap_capable and rng.random() < .20)
    pool, mandatory = _pack_centers(rng, h, w, radius, clipped, overlap_requested)
    count_cap = len(pool)
    if count_cap == mandatory:
        count = mandatory
    elif rng.random() < .35 and mandatory == 1:
        count = 1
    elif count_cap >= 5 and rng.random() < .35:
        count = int(rng.integers(max(5, mandatory), count_cap + 1))
    else:
        count = int(rng.integers(mandatory, min(4, count_cap) + 1))
    tail = list(range(mandatory, len(pool)))
    rng.shuffle(tail)
    centers = pool[:mandatory] + [pool[i] for i in tail[:count - mandatory]]
    marked = np.zeros((h, w), bool)
    for center in centers:
        marked[center] = True
    kind, n_accents = field_plan(rng, h, w, drawing_diamonds=True, free=~marked)
    colors = list(map(int, rng.choice(10, size=3 + n_accents, replace=False)))
    bg, marker, diamond = colors[:3]
    accents = colors[3:]
    masks = [_mask(h, w, center, radius) for center in centers]
    cover = np.sum(masks, axis=0)
    for attempts in range(1, 81):
        field = render_field(rng, h, w, bg, accents, kind, avoid=marked)
        if field is None:
            continue
        inp = field.copy()
        for cr, c in centers:
            inp[cr, c] = marker
        if set(np.unique(inp)) != {bg, marker, *accents}:
            continue
        out = field.copy()
        out[cover > 0] = diamond
        output = out.tolist()
        validate_draw_diamonds(inp, output, marker, diamond, span)
        full_area = 1 + 2 * radius * (radius + 1)
        clipped_count = sum(int(mask.sum()) < full_area for mask in masks)
        return {
            "input": inp.tolist(), "output": output,
            "metadata": {
                "grid_h": h, "grid_w": w, "n_colors": len(np.unique(inp)),
                "planned_n_colors": 2 + n_accents,
                "length": span, "tip_to_tip_span": span, "radius": radius,
                "bg_color": bg, "marker_color": marker, "diamond_color": diamond,
                "n_diamonds": count, "n_objects": count, "requested_count": count,
                "count_cap": count_cap, "span_cap": 2 * cap + 1,
                "centers": [list(c) for c in centers], "attempts": attempts,
                "family": "filled_diamonds", "placement": "free",
                "clip_at_edges_if_applicable": True,
                "is_clipped_example": bool(clipped_count), "clipped_count": clipped_count,
                "op_clipped": clipped, "op_overlap": overlap_requested,
                "overlap_capable": overlap_capable,
                "overlap_cells": int((cover > 1).sum()),
                "visible_diamond_cells": int((cover > 0).sum()),
                "foreground_fraction": float(np.mean(cover > 0)),
                "accent_overwrites": int(((cover > 0) & (inp != bg) & (inp != marker)).sum()),
                "background_outside_preserved": bool(np.array_equal(inp[cover == 0], out[cover == 0])),
                **field_metadata(inp, bg, kind, accents),
            },
        }
    raise RuntimeError("Task 71: the fixed field/color plan could not be rendered")


def _length_plans(h, w, mode):
    rr, cc = np.indices((h, w))
    interior = (rr > 0) & (rr < h - 1) & (cc > 0) & (cc < w - 1)
    plans = {}
    radius_cap = (min(h, w) - 1) // 2 if mode == "complete" else max(h, w) // 2 if mode == "one_axis" else min(h, w) - 1
    for radius in range(1 if mode == "complete" else 2, radius_cap + 1):
        vr = (rr < radius) | (rr + radius >= h)
        hc = (cc < radius) | (cc + radius >= w)
        eligible = interior & ((~vr & ~hc) if mode == "complete" else ((vr ^ hc) if mode == "one_axis" else (vr & hc)))
        # All four diagonal faces have visible witnesses. Together they fix both
        # center and radius, including when neither full corner span is visible.
        eligible &= (rr + cc >= radius) & (h + w - 2 - rr - cc >= radius)
        eligible &= (w - 1 + rr - cc >= radius) & (h - 1 - rr + cc >= radius)
        # Merely reaching a face at a canvas corner is insufficient: that face
        # may continue outside the canvas without changing any visible cell.
        # At least one opposite pair must also have in-grid exterior witnesses.
        eligible &= (((rr + cc > radius) & (h + w - 2 - rr - cc > radius))
                     | ((w - 1 + rr - cc > radius) & (h - 1 - rr + cc > radius)))
        if mode == "both_axes" and 1 + 2 * radius * (radius + 1) > .70 * h * w:
            continue
        if eligible.any():
            plans[radius] = np.argwhere(eligible)
    return plans


def generate_measure_diamond(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    mode = str(rng.choice(["complete", "one_axis", "both_axes"], p=[.42, .33, .25]))
    plans = _length_plans(h, w, mode)
    if not plans:
        raise RuntimeError("Task 72: missing size/mode capacity")
    radii = list(plans)
    if rng.random() < .38:
        radii = radii[len(radii) // 2:]
    radius = int(rng.choice(radii))
    style = "outline" if radius >= 2 and rng.random() < .28 else "filled"
    centers = plans[radius]
    center = tuple(map(int, centers[int(rng.integers(len(centers)))]))
    mask = _mask(h, w, center, radius, style)
    kind, n_accents = field_plan(rng, h, w, free=~mask)
    colors = list(map(int, rng.choice(10, size=2 + n_accents, replace=False)))
    bg, diamond = colors[:2]
    accents = colors[2:]
    for attempts in range(1, 121):
        field = render_field(rng, h, w, bg, accents, kind, avoid=mask)
        if field is None:
            continue
        grid = field.copy()
        grid[mask] = diamond
        if set(np.unique(grid)) != {bg, diamond, *accents}:
            continue
        readings, unbounded = diamond_readings(grid)
        if unbounded or set(readings) != {2 * radius + 1}:
            continue
        answer = str(2 * radius + 1)
        validate_diamond_length(grid, answer)
        rows, cols = np.nonzero(mask)
        bh, bw = int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1
        return {
            "input": grid.tolist(), "output": answer,
            "metadata": {
                "grid_h": h, "grid_w": w, "n_colors": len(np.unique(grid)),
                "planned_n_colors": 2 + n_accents,
                "bg_color": bg, "diamond_color": diamond, "family": style,
                "clip_axis": "none" if mode == "complete" else "both" if mode == "both_axes" else "vertical" if bh < int(answer) else "horizontal",
                "clip_mode": mode, "is_clipped_example": mode != "complete",
                "center": list(center), "radius": radius, "length": int(answer),
                "span_cap": 2 * max(plans) + 1, "bbox_h": bh, "bbox_w": bw,
                "requires_span_inference": max(bh, bw) < int(answer),
                "visible_diamond_cells": int(mask.sum()), "foreground_fraction": float(mask.mean()),
                "unique_full_span": True, "off_grid_alternatives": 0,
                "attempts": attempts, **field_metadata(grid, bg, kind, accents),
            },
        }
    raise RuntimeError("Task 72: the fixed size/radius/style/field plan is not uniquely readable")
