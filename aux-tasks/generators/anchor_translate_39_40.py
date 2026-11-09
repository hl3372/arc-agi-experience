"""Varied rigid transformations. Scene construction precedes the motion draw."""
from __future__ import annotations

import numpy as np
from common.backgrounds import apply_background, SPARSE_KINDS
from generators.rigid_scene import capacity, draw_objects, visible_objects
from generators.crop_frame_26 import _render_scene
from generators.exterior_field import infer_exterior_field
from generators.fixed_details import KINDS as DETAIL_KINDS, add_details, moving_cells

ROTATIONS = (0, 90, 180, 270)
ROTATION_WEIGHTS = (.05, .31666666666666665, .31666666666666665, .31666666666666665)
PROMPT39 = (
    "Ignoring the background and the color {anchor} anchor cell, the single object is an "
    "edge-connected group of at least two cells and may contain several colors. "
    "Isolated single cells, whatever their color, are fixed detail. Rotate the object {angle} degrees clockwise "
    "around the center of the anchor cell. Keep the anchor in place and preserve the color of every moved cell. "
    "Keep the details in place unless covered by the object. Fill vacated cells with the plain "
    "background color and clip anything outside the grid. Leave every other cell unchanged."
)
PROMPT40 = (
    "Ignoring the background, treat each 4-connected same-color group of at least two cells in the input "
    "(connected horizontally or vertically; cells that touch only at a corner are not connected) as an "
    "object. Isolated single cells, whatever their color, are background texture and are ignored; different "
    "colors are separate even where they touch. "
    "Translate every object {direction}, preserving its colors, and clip anything moved outside the grid. "
    "Keep background texture in place; moved objects may cover it. Fill vacated object cells with "
    "the plain background color. Leave every other cell unchanged."
)
CONTENT = ("bands", "diagonal_bands", "stair_bands", "patchwork", "striped_panels")


def _halo(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[r:r + h, c:c + w] for r in range(3) for c in range(3)])


def solve_rotation(grid, anchor_color, angle):
    a = np.asarray(grid)
    ground = infer_exterior_field(a, allow_strips=True, exterior_mass=False)
    anchor = np.argwhere(a == anchor_color)
    if len(anchor) != 1 or anchor_color == ground or angle not in ROTATIONS:
        raise ValueError("Invalid rotation anchor or angle")
    ar, ac = map(int, anchor[0])
    objects = moving_cells(a, ground, excluded=(anchor_color,))
    if objects.sum() < 2 or not _connected(objects):
        raise ValueError("There must be exactly one movable object")
    # Inverse-map each destination. The constructor forward-maps object cells.
    out = a.copy()
    out[objects] = ground
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            y, x = r - ar, c - ac
            for _ in range((angle // 90) % 4):
                y, x = -x, y
            rr, cc = y + ar, x + ac
            if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and objects[rr, cc]:
                out[r, c] = a[rr, cc]
    out[ar, ac] = anchor_color
    return out


def solve_translation(grid, dr, dc):
    a = np.asarray(grid)
    ground = infer_exterior_field(a)
    objects, _ = visible_objects(a, excluded=(ground,))
    if not objects:
        raise ValueError("No movable objects")
    out = a.copy()
    points = {}
    for cells, color in objects:
        for r, c in cells:
            out[r, c] = ground
            points[r, c] = color
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if (r - dr, c - dc) in points:
                out[r, c] = points[r - dr, c - dc]
    return out


def validate_example(ex, task):
    m = ex["metadata"]
    if task == 39 and min(np.asarray(ex["input"]).shape) < 5:
        raise ValueError("Rotation canvases require at least five cells on each side")
    if task == 40 and min(np.asarray(ex["input"]).shape) < 4:
        raise ValueError("Translation canvases require at least four cells on each side")
    out = (solve_rotation(ex["input"], m["anchor_color"], m["angle"]) if task == 39 else
           solve_translation(ex["input"], m["dr"], m["dc"]))
    if out.tolist() != ex["output"]:
        raise ValueError("Visible rigid-transform answer differs from construction")
    if len(np.unique(ex["input"])) != m["n_colors"]:
        raise ValueError("Rigid-transform palette budget was lost")


def _sides(rng):
    return tuple(map(int, rng.integers(5, 31, size=2)))


def _rotation_angle(rng):
    return int(rng.choice(ROTATIONS, p=ROTATION_WEIGHTS))


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo, seen = [next(iter(cells))], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        r, c = p
        todo.extend(q for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if q in cells and q not in seen)
    return seen == cells


def _strip_rotation(rng, h, w, minimum, anchor_mode):
    """Construct the narrow family with field-colored endpoints, without retries."""
    length = max(h, w)
    count = int(rng.integers(max(2, minimum), max(max(2, minimum), int(.42 * length)) + 1))
    inside = anchor_mode == "inside"
    span = count + int(inside)
    start = int(rng.integers(1, length - span))
    line = np.zeros(length, bool)
    line[start:start + span] = True
    if inside:
        anchor = start if rng.integers(2) else start + span - 1
        line[anchor] = False
    else:
        options = [p for p in range(1, length - 1) if not line[p]]
        adjacent = [p for p in options if p in (start - 1, start + span)]
        anchor = int(rng.choice(adjacent if anchor_mode == "near" and adjacent else options))
    mask = line.reshape(h, w)
    position = (0, anchor) if h == 1 else (anchor, 0)
    points = np.argwhere(mask)
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0))
    return mask, position, (r0, c0, r1, c1)


def generate_rotate_around_anchor(rng):
    h, w = _sides(rng)
    cap = min(8, max(1, h * w // 5))
    modes = ["monochrome"] + (["few_colors"] if cap >= 2 else []) + (["rich"] if cap >= 4 else [])
    weights = np.asarray([.40, .40, .20][:len(modes)])
    fill_mode = str(rng.choice(modes, p=weights / weights.sum()))
    object_colors = (1 if fill_mode == "monochrome" else
                     int(rng.integers(2, min(3, cap) + 1)) if fill_mode == "few_colors" else
                     int(rng.integers(4, cap + 1)))
    texture = min(h, w) >= 8 and rng.random() < .50
    extra_color = int(texture and object_colors < 8 and rng.random() < .35)
    n_colors = 2 + object_colors + extra_color
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    ground, anchor_color = palette[:2]
    colors = palette[2:2 + object_colors]
    detail_colors = palette[-1:] if extra_color else [int(rng.choice(colors))]
    background = str(rng.choice(DETAIL_KINDS)) if texture else "solid"
    content = "solid" if object_colors == 1 else str(rng.choice(CONTENT))
    anchor_mode = str(rng.choice(("inside", "near", "anywhere", "edge"), p=(.55, .35, .05, .05)))
    # Keep ordinary bar objects after removing strip-shaped canvases. The bar
    # must hold the fixed palette and an optional anchor at one endpoint.
    bar_axes = [axis for axis, extent in enumerate((h, w))
                if extent - 1 >= max(2, len(colors)) + int(anchor_mode == "inside")]
    bar_axis = int(rng.choice(bar_axes)) if bar_axes and rng.random() < .10 else None
    for attempt in range(1, 301):
        if min(h, w) == 1:
            mask, (ar, ac), (r0, c0, r1, c1) = _strip_rotation(rng, h, w, len(colors), anchor_mode)
            families = ["bar"]
        else:
            objects, families = draw_objects(rng, h, w, [colors[0]], scale=(.45, .90),
                                             max_h=1 if bar_axis == 1 else h - 1,
                                             max_w=1 if bar_axis == 0 else w - 1)
            cells = objects[0][0]
            mask = np.zeros((h, w), dtype=bool)
            for p in cells:
                mask[p] = True
            r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
            r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
            rr, cc = np.mgrid[:h, :w]
            if anchor_mode == "inside":
                candidates = (np.asarray([(r0, c0), (r1, c1)]) if bar_axis is not None else
                              np.argwhere((rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)))
            elif anchor_mode == "near":
                distance = np.maximum(r0 - rr, np.maximum(rr - r1, 0)) + np.maximum(c0 - cc, np.maximum(cc - c1, 0))
                candidates = np.argwhere((distance <= max(2, min(h, w) // 4)) & ~mask)
            elif anchor_mode == "edge":
                candidates = np.argwhere(((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)) & ~mask)
            else:
                candidates = np.argwhere(~mask)
            if not len(candidates):
                continue
            ar, ac = map(int, candidates[int(rng.integers(len(candidates)))])
            mask[ar, ac] = False
        if mask.sum() < max(2, len(colors)) or not _connected(mask):
            continue
        rendered = (np.full((h, w), colors[0], dtype=np.uint8), {}) if object_colors == 1 else _render_scene(rng, mask, colors, content)
        if rendered is None:
            continue
        a = np.full((h, w), ground, dtype=np.uint8)
        a[mask] = rendered[0][mask]
        a[ar, ac] = anchor_color
        detail = np.zeros_like(mask)
        if texture:
            painted = add_details(rng, a, ground, mask | (a == anchor_color), detail_colors, background)
            if painted is None:
                continue
            a, detail = painted
        if len(np.unique(a)) != n_colors:
            continue
        try:
            if infer_exterior_field(a, allow_strips=True, exterior_mass=False) != ground:
                continue
        except ValueError:
            continue
        if not np.array_equal(moving_cells(a, ground, excluded=(anchor_color,)), mask):
            continue
        # All angles, including total clipping, are legal on this same input.
        angle = _rotation_angle(rng)
        out = a.copy()
        out[mask] = ground
        moved = 0
        for r, c in np.argwhere(mask):
            y, x = int(r) - ar, int(c) - ac
            for _ in range(angle // 90):
                y, x = x, -y
            y, x = y + ar, x + ac
            if 0 <= y < h and 0 <= x < w:
                out[y, x] = a[r, c]
                moved += 1
        out[ar, ac] = anchor_color
        ex = {"input": a.tolist(), "output": out.tolist(), "metadata": {
            "anchor_color": anchor_color, "angle": angle, "anchor_position": [ar, ac],
            "anchor_mode": anchor_mode, "background_color": ground, "n_colors": n_colors,
            "object_color_count": object_colors, "fill_mode": fill_mode,
            "background": background, "texture_cells": int(detail.sum()),
            "shape_family": families[0], "content_family": content, "object_area": int(mask.sum()),
            "object_bbox": [r0, c0, r1, c1], "surviving_cells": moved,
            "fully_clipped": moved == 0, "attempts": attempt}}
        validate_example(ex, 39)
        return ex
    raise RuntimeError("Task 39 could not realize its fixed scene budgets")


def _translation(rng, h, w):
    mode = str(rng.choice(("identity", "short", "wide", "clear"), p=(.05, .55, .37, .03)))
    if mode == "identity":
        return 0, 0, mode
    axis = int(rng.choice((0, 1, 2), p=(.225, .225, .55)))
    max_r = max(1, min(5, h - 1) if mode == "short" else h - 1)
    max_c = max(1, min(5, w - 1) if mode == "short" else w - 1)
    dr = int(rng.integers(1, max_r + 1)) * int(rng.choice((-1, 1))) if axis != 1 else 0
    dc = int(rng.integers(1, max_c + 1)) * int(rng.choice((-1, 1))) if axis != 0 else 0
    if mode == "clear":
        if axis != 1:
            dr = h * (1 if dr > 0 else -1)
        else:
            dc = w * (1 if dc > 0 else -1)
    return dr, dc, mode


def _direction(dr, dc):
    if dr == dc == 0:
        return "by zero cells (keep it in place)"
    parts = []
    if dr:
        parts.append(f"{'down' if dr > 0 else 'up'} {abs(dr)} {'cell' if abs(dr) == 1 else 'cells'}")
    if dc:
        parts.append(f"{'right' if dc > 0 else 'left'} {abs(dc)} {'cell' if abs(dc) == 1 else 'cells'}")
    return " and ".join(parts)


def generate_translate_all(rng):
    h, w = map(int, rng.integers(4, 31, size=2))
    texture = min(h, w) >= 8 and rng.random() < .48
    rh, rw = h - 2 - (2 if texture else 0), w - 2
    cap = max(1, min(12, capacity(rh, rw), (h * w - 3) // 2))
    n_colors = int(rng.integers(3 if texture else 2, min(10, cap + 1 + int(texture)) + 1))
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    ground = palette[0]
    colors = palette[1:-1] if texture else palette[1:]
    texture_color = palette[-1] if texture else None
    feasible_counts = [n for n in range(len(colors), min(cap, max(len(colors), h * w // 55 + 2)) + 1)
                       if h * w - 2 * n > 2.5 * int(np.ceil(n / len(colors)))]
    n_objects = int(rng.choice(feasible_counts))
    kind = str(rng.choice(SPARSE_KINDS[:-1])) if texture else "solid"
    for attempt in range(1, 201):
        objects, families = draw_objects(rng, rh, rw, colors, n_objects, scale=(.50, .95))
        dy = 1 if rng.random() < .70 else int(rng.integers(h - rh + 1))
        dx = 1 if rng.random() < .70 else int(rng.integers(w - rw + 1))
        a = np.full((h, w), ground, dtype=np.uint8)
        source = np.zeros((h, w), dtype=bool)
        for cells, color in objects:
            for r, c in cells:
                a[r + dy, c + dx], source[r + dy, c + dx] = color, True
        if texture:
            painted = apply_background(rng, a, ground, _halo(source), kind=kind, colors=[texture_color])
            if painted is None:
                continue
            a = painted[0]
        if rng.integers(2):
            a, source = a[::-1].copy(), source[::-1].copy()
        if rng.integers(2):
            a, source = a[:, ::-1].copy(), source[:, ::-1].copy()
        try:
            if infer_exterior_field(a) != ground:
                continue
        except ValueError:
            continue
        if len(np.unique(a)) != n_colors:
            continue
        dr, dc, mode = _translation(rng, h, w)
        out = a.copy()
        out[source] = ground
        moved, covered = 0, 0
        for r, c in np.argwhere(source):
            y, x = int(r) + dr, int(c) + dc
            if 0 <= y < h and 0 <= x < w:
                covered += int(texture and a[y, x] == texture_color)
                out[y, x] = a[r, c]
                moved += 1
        ex = {"input": a.tolist(), "output": out.tolist(), "metadata": {
            "dr": dr, "dc": dc, "direction": _direction(dr, dc), "motion_mode": mode,
            "background_color": ground, "background": kind, "n_colors": n_colors,
            "texture_cells": int((a == texture_color).sum()) if texture else 0,
            "covered_background_cells": covered, "object_count": n_objects,
            "shape_families": families, "object_areas": [len(cells) for cells, _ in objects],
            "foreground_cells": int(source.sum()), "surviving_cells": moved,
            "fully_clipped": moved == 0, "attempts": attempt}}
        validate_example(ex, 40)
        return ex
    raise RuntimeError("Task 40 could not realize its fixed scene and background budgets")
