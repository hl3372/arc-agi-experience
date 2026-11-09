"""Reflect objects across an inferred divider; preserve all background details."""
from __future__ import annotations

import numpy as np

from common.backgrounds import apply_background, SPARSE_KINDS
from generators.rigid_scene import capacity, draw_objects, infer_ground, visible_objects

PROMPT34 = (
    "The divider is the single non-background line spanning a whole row or column. "
    "Ignoring the background and divider, treat each 4-connected same-color group of at least two cells "
    "(connected horizontally or vertically; cells that touch only at a corner are not connected) as one "
    "shape. Isolated single cells, whatever their color, are background texture and are ignored; different "
    "colors are separate even where they touch. "
    "Shapes may lie on either or both sides of the divider. Keep each original shape where it is and add its mirror image "
    "on the opposite side, at the same distance from the divider. Leave every other cell unchanged."
)


def _reflect(r, c, orientation, divider):
    return (2 * divider - r, c) if orientation == "horizontal" else (r, 2 * divider - c)


def _lines(a, ground):
    result = []
    for axis, name in ((0, "horizontal"), (1, "vertical")):
        for i, line in enumerate(a if axis == 0 else a.T):
            if line[0] != ground and np.all(line == line[0]):
                result.append((name, i, line[0].item()))
    return result


def solve_mirror(grid):
    """Infer ground, divider, objects and destinations from visible cells only."""
    a = np.asarray(grid)
    h, w = a.shape
    ground = infer_ground(a)
    lines = _lines(a, ground)
    if len(lines) != 1:
        raise ValueError("Task 34 requires exactly one visible divider")
    orientation, divider, color = lines[0]
    objects, dots = visible_objects(a, excluded=(ground, color))
    if not objects:
        raise ValueError("Task 34 has no shapes")
    for cells, _ in objects:
        sides = {r < divider if orientation == "horizontal" else c < divider for r, c in cells}
        if len(sides) != 1:
            raise ValueError("Task 34 has a shape crossing the divider")
    out = a.copy()
    # Recover the source mask independently, without the constructor's objects
    # or its coordinate-reflection helper.
    source = np.zeros_like(a, dtype=bool)
    for cells, _ in objects:
        for r, c in cells:
            source[r, c] = True
    for r, c in np.argwhere(source):
        if orientation == "horizontal":
            rr, cc = divider + (divider - int(r)), int(c)
        else:
            rr, cc = int(r), divider + (divider - int(c))
        if not (0 <= rr < h and 0 <= cc < w) or a[rr, cc] != ground:
            raise ValueError("A reflected shape would clip or overwrite background detail")
        out[rr, cc] = a[r, c]
    if any(out[r, c] != a[r, c] for r, c in dots):
        raise ValueError("Background detail changed")
    return out


def validate_example(ex):
    if solve_mirror(ex["input"]).tolist() != ex["output"]:
        raise ValueError("Task 34 visible reconstruction differs from output")
    if len(np.unique(ex["input"])) != ex["metadata"]["n_colors"]:
        raise ValueError("Task 34 palette budget was lost")


def _halo(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[y:y + mask.shape[0], x:x + mask.shape[1]]
                                 for y in range(3) for x in range(3)])


def _texture(rng, a, out, ground, divider_color, texture_color, orientation, divider, symmetry, kind):
    # The divider is explicitly excluded from the object rule. A texture dot
    # may sit beside it, but never beside a shape or a reflected destination.
    occupied = (_halo(((a != ground) & (a != divider_color)) |
                      ((out != ground) & (out != divider_color))) | (a == divider_color))
    painted = apply_background(rng, a, ground, occupied, kind=kind,
                               colors=[texture_color], exclude=[int(c) for c in np.unique(a)])
    if painted is None:
        return None
    texture = painted[0] == texture_color
    h, w = a.shape
    if symmetry == "asymmetric":
        axis = np.indices((h, w))[0 if orientation == "horizontal" else 1]
        first_side = axis < divider
        if rng.random() < .35:
            texture &= first_side if rng.integers(2) else ~first_side
        else:
            other = apply_background(rng, a, ground, occupied, kind=kind,
                                     colors=[texture_color], exclude=[int(c) for c in np.unique(a)])
            if other is None:
                return None
            texture = np.where(first_side, texture, other[0] == texture_color)
    if symmetry == "symmetric":
        chosen = np.zeros_like(texture)
        for r, c in np.argwhere(texture):
            rr, cc = _reflect(int(r), int(c), orientation, divider)
            if not (0 <= rr < h and 0 <= cc < w) or occupied[rr, cc]:
                continue
            if _halo(chosen)[r, c] or _halo(chosen)[rr, cc]:
                continue
            chosen[r, c] = chosen[rr, cc] = True
        texture = chosen
    if not texture.any():
        return None
    asymmetric_pairs = 0
    for r in range(h):
        for c in range(w):
            rr, cc = _reflect(r, c, orientation, divider)
            if 0 <= rr < h and 0 <= cc < w and texture[r, c] != texture[rr, cc]:
                asymmetric_pairs += 1
    if (symmetry == "asymmetric") != bool(asymmetric_pairs):
        return None
    b, z = a.copy(), out.copy()
    b[texture] = z[texture] = texture_color
    return b, z, int(texture.sum()), asymmetric_pairs // 2


def generate_mirror_across_divider(rng):
    h, w = map(int, rng.integers(3, 31, size=2))
    orientation = str(rng.choice(("horizontal", "vertical")))
    extent = h if orientation == "horizontal" else w
    low, high = max(1, extent // 4), min(extent - 2, (3 * extent) // 4)
    divider = (extent - 1) // 2 if rng.random() < .35 else int(rng.integers(low, high + 1))
    side = int(rng.integers(2))
    depth = min(divider, extent - 1 - divider)
    rh, rw = (depth, w) if orientation == "horizontal" else (h, depth)
    mode = str(rng.choice(("solid", "symmetric", "asymmetric"), p=(.10, .30, .60))) if min(h, w) >= 8 else "solid"
    texture_slots = int(mode != "solid")
    if texture_slots:
        # Reserve a narrow clear region on both sides. Even on a shallow
        # off-centre divider, background detail then has a visible counterpart
        # instead of existing only in the unmatched tail of the larger side.
        if orientation == "horizontal":
            rw -= 3
        else:
            rh -= 3
    object_cap = min(10, capacity(rh, rw), max(1, (h * w) // 10))
    minimum_colors = 3 + texture_slots
    n_colors = int(rng.integers(minimum_colors, min(10, object_cap + 2 + texture_slots) + 1))
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    ground, divider_color = palette[:2]
    object_colors = palette[2:-1] if texture_slots else palette[2:]
    texture_color = palette[-1] if texture_slots else None
    n_objects = int(rng.integers(len(object_colors), max(len(object_colors), min(object_cap, len(object_colors) + 2)) + 1))
    # Plan one visible original per reserved reflection pair. Mixed scenes do
    # not need overwrite rules: every counterpart is still empty in the input.
    mixed = n_objects >= 2 and rng.random() < .5
    source_sides = np.full(n_objects, side, dtype=int)
    if mixed:
        source_sides[rng.permutation(n_objects)[:int(rng.integers(1, n_objects))]] = 1 - side
    kind = str(rng.choice(SPARSE_KINDS[:-1])) if texture_slots else "solid"
    for attempt in range(1, 201):
        try:
            objects, families = draw_objects(rng, rh, rw, object_colors, n_objects,
                                            scale=(.50, .95) if h * w >= 150 else (.38, .85),
                                            max_h=h - 1 if orientation == "vertical" else None,
                                            max_w=w - 1 if orientation == "horizontal" else None)
        except ValueError:
            continue
        a = np.full((h, w), ground, dtype=np.uint8)
        if orientation == "horizontal":
            a[divider, :] = divider_color
            offset_r, offset_c = divider - depth, 0
        else:
            a[:, divider] = divider_color
            offset_r, offset_c = 0, divider - depth
        out = a.copy()
        for object_side, (cells, color) in zip(source_sides, objects):
            for r, c in cells:
                r, c = r + offset_r, c + offset_c
                if object_side:
                    r, c = _reflect(r, c, orientation, divider)
                a[r, c] = out[r, c] = color
                rr, cc = _reflect(r, c, orientation, divider)
                out[rr, cc] = color
        texture_cells = asymmetric_pairs = 0
        if texture_slots:
            result = _texture(rng, a, out, ground, divider_color, texture_color, orientation, divider, mode, kind)
            if result is None:
                continue
            a, out, texture_cells, asymmetric_pairs = result
        try:
            if infer_ground(a) != ground or len(_lines(a, ground)) != 1:
                continue
        except ValueError:
            continue
        ex = {"input": a.tolist(), "output": out.tolist(), "metadata": {
            "divider_orientation": orientation, "divider_position": divider,
            "divider_color": divider_color, "background_color": ground,
            "background": kind, "background_symmetry": mode, "texture_cells": texture_cells,
            "asymmetric_background_pairs": asymmetric_pairs, "n_colors": n_colors,
            "object_count": n_objects, "shape_families": families,
            "object_areas": [len(cells) for cells, _ in objects],
            "foreground_cells": sum(len(cells) for cells, _ in objects),
            "source_side_counts": [int(np.sum(source_sides == i)) for i in (0, 1)],
            "mixed_source_sides": bool(mixed),
            "scene_mode": "single_object" if n_objects == 1 else "multi_object", "attempts": attempt}}
        validate_example(ex)
        return ex
    raise RuntimeError("Task 34 could not realize its fixed scene and background budgets")
