"""Task 83: recolor cells that do not touch the outside of their object.

The completed drawing precedes the independent edge/corner rule draw.  A
partition witnesses space for the exact bodies; repeated free relocation then
varies their positions without changing the count, palette or silhouettes.

Backgrounds (September 2026).  The ground used to be a plain color on every
grid.  Sparse library textures now cover about a quarter of the eligible grids
(short side >= 8): isolated dots that keep one clear cell from every object and
never reuse the ground, the fill or an object color.  They are inert by
construction, not by exception -- a dot is a one-cell group, every one of its
cells touches something outside it through a shared edge, so it has no interior
cell and the fill reaches it under neither reading of the rule.  The re-solver
therefore treats a dot as the one-cell object it is, with no special case, and
the prompt keeps the house sentence "Ignore the background, whether it is a
solid color or a pattern." on plain and textured grids alike.

The ground stays readable as the ground: objects never reach the border ring,
and the dots on it are thinned until the ground owns ``RING_GROUND`` of the ring
with no other color over ``RING_RIVAL``, well inside the figure/ground
certificate the validator re-runs on the finished picture.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np

from common.backgrounds import SPARSE_KINDS, apply_background, texture_cells_isolated
from common.grid import NUM_COLORS
from generators.exterior_field import figure_ground_field
from generators.scene_shapes_56_60 import halo, placement_map


FAMILIES = ("rectangle", "chamfer", "diamond", "plus", "L", "T", "U", "staircase")
FAMILY_WEIGHTS = (.25, .16, .15, .12, .10, .08, .06, .08)
STEPS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
STEPS_8 = STEPS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

TEXTURE_MIN_SIDE = 8      # the library's size floor for a sparse texture
TEXTURE_P = .30           # attempt rate on eligible grids -> ~27% realised
TEXTURE_TRIES = 3
MIN_TEXTURE_CELLS = 4     # fewer dots than this is a speck, not a background
RING_GROUND = .70         # the ground's share of the border ring
RING_RIVAL = .20          # every other color's share of it
MID_AREA, BIG_AREA = 150, 400   # canvases that carry more bodies, and so more inks
COUNT_WEIGHTS = ((.32, .49, .19), (.25, .47, .28), (.18, .44, .38))
COLOR_WEIGHTS = ((.16, .52, .32), (.12, .46, .42), (.08, .42, .50))
_NO_TEXTURE = {"background_style": "plain", "texture_group": None, "texture_kind": None,
               "texture_colors": [], "texture_cells": 0, "texture_coverage": .0,
               "texture_named": False}


def _inner(mask, rule):
    """Writer: intersect shifted copies of a binary object's footprint."""
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in STEPS_4 if rule == 4 else STEPS_8:
        result &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


@lru_cache(maxsize=8192)
def _body(family, h, w):
    y, x = np.indices((h, w))
    t = max(3, min(h, w) // 3)
    if family == "rectangle":
        mask = np.ones((h, w), dtype=bool)
    elif family == "chamfer":
        cut = max(1, min(h, w) // 4)
        mask = (y + x >= cut) & (y + w - 1 - x >= cut)
        mask &= (h - 1 - y + x >= cut) & (h - 1 - y + w - 1 - x >= cut)
    elif family == "diamond":
        if min(h, w) < 5 or h % 2 == 0 or w % 2 == 0 or max(h, w) > 2 * min(h, w):
            return None
        mask = abs(y - h // 2) * (w // 2) + abs(x - w // 2) * (h // 2) <= (h // 2) * (w // 2)
    elif family in ("L", "T", "plus", "U"):
        if min(h, w) < 5:
            return None
        stem = (x >= (w - t) // 2) & (x < (w - t) // 2 + t)
        bar = (y >= (h - t) // 2) & (y < (h - t) // 2 + t)
        if family == "L":
            mask = (x < t) | (y >= h - t)
        elif family == "T":
            mask = (y < t) | stem
        elif family == "plus":
            mask = stem | bar
        else:
            arm = min(t, (w - 1) // 2)
            mask = (x < arm) | (x >= w - arm) | (y >= h - t)
    elif family == "staircase":
        steps = 3 if min(h, w) >= 8 else 2
        mask = x < ((y * steps // h + 1) * w + steps - 1) // steps
    else:
        raise ValueError("Unknown interior shape family")
    if not mask.any() or (family != "rectangle" and mask.all()) or not _inner(mask, 8).any():
        return None
    mask.flags.writeable = False
    return mask


def _capacity(h, w):
    return ((h + 1) // 4) * ((w + 1) // 4)


def _regions(rng, h, w, count, row=1, col=1):
    if count == 1:
        return [(row, col, h, w)]
    cuts = []
    for axis, extent in enumerate((h, w)):
        for cut in range(3, extent - 3):
            ca = _capacity(cut, w) if axis == 0 else _capacity(h, cut)
            cb = _capacity(h - cut - 1, w) if axis == 0 else _capacity(h, w - cut - 1)
            low, high = max(1, count - cb), min(count - 1, ca)
            if low <= high:
                cuts.append((axis, cut, low, high))
    if not cuts:
        raise ValueError("No witness for the selected object count")
    axis, cut, low, high = cuts[int(rng.integers(len(cuts)))]
    left = int(rng.integers(low, high + 1))
    if axis == 0:
        return (_regions(rng, cut, w, left, row, col)
                + _regions(rng, h - cut - 1, w, count - left, row + cut + 1, col))
    return (_regions(rng, h, cut, left, row, col)
            + _regions(rng, h, w - cut - 1, count - left, row, col + cut + 1))


def _budgets(rng, regions, total):
    budgets = np.full(len(regions), 9, dtype=int)
    spare = np.array([h * w - 9 for _, _, h, w in regions])
    remaining = min(int(spare.sum()), max(0, total - int(budgets.sum())))
    weights = rng.uniform(.65, 1.35, size=len(regions))
    while remaining:
        probabilities = spare * weights
        probabilities /= probabilities.sum()
        amount = np.minimum(spare, np.floor(probabilities * remaining).astype(int))
        if not amount.any():
            amount[int(rng.choice(len(regions), p=probabilities))] = 1
        budgets += amount
        spare -= amount
        remaining -= int(amount.sum())
    return budgets.tolist()


def _planned_body(rng, h, w, budget, thin):
    if thin:
        vertical = h > w or (h == w and bool(rng.integers(2)))
        extent = h if vertical else w
        length = int(rng.integers(2, min(extent, budget) + 1))
        return np.ones((length, 1) if vertical else (1, length), dtype=bool), "bar"
    sizes = [(a, b) for a in range(3, h + 1) for b in range(3, w + 1) if a * b <= budget]
    possibilities = []
    for family, weight in zip(FAMILIES, FAMILY_WEIGHTS):
        candidates = [(a, b) for a, b in sizes if _body(family, a, b) is not None]
        if candidates:
            large = [(a, b) for a, b in candidates if a * b >= .70 * budget]
            possibilities.append((family, weight, large or candidates))
    weights = np.array([weight for _, weight, _ in possibilities])
    weights /= weights.sum()
    family, _, candidates = possibilities[int(rng.choice(len(possibilities), p=weights))]
    a, b = candidates[int(rng.integers(len(candidates)))]
    mask = _body(family, a, b)
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask.copy(), family


def _plan(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    # A canvas with room for more bodies carries more of them, and with them more
    # inks: distinct object colors can never outnumber the objects, so a palette
    # that reaches 9 has to be drawn here and not in the color draw below.
    tier = (h * w >= MID_AREA) + (h * w >= BIG_AREA)
    cap = min(10, _capacity(h - 2, w - 2), max(1, int(.32 * h * w / 9)))
    count_class = str(rng.choice(("one", "few", "many"), p=COUNT_WEIGHTS[tier]))
    if count_class == "one" or cap == 1:
        count = 1
    elif count_class == "few":
        count = int(rng.integers(2, min(4, cap) + 1))
    else:
        count = int(rng.integers(min(5, cap), cap + 1))
    regions = _regions(rng, h - 2, w - 2, count)
    budgets = _budgets(rng, regions, int(.49 * h * w))
    thin_index = int(rng.integers(1, count)) if count > 1 and rng.random() < .18 else -1
    shapes = [_planned_body(rng, a, b, budget, i == thin_index)
              for i, ((_, _, a, b), budget) in enumerate(zip(regions, budgets))]
    # The texture intent is drawn here so the palette can keep a color free for it;
    # it says nothing about the rule, which is drawn after the whole drawing.
    texture = min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P
    color_cap = min(count, 7 if texture else 8)
    color_class = str(rng.choice(("mono", "few", "rich"), p=COLOR_WEIGHTS[tier]))
    if color_class == "mono":
        inks = 1
    elif color_class == "few":
        inks = int(rng.integers(min(2, color_cap), min(3, color_cap) + 1))
    else:
        # a rich scene leans on the upper half of what the object count allows
        # (the best of two draws), so a canvas full of bodies is rarely a duet
        inks = int(max(rng.integers(min(4, color_cap), color_cap + 1, size=2)))
    palette = list(map(int, rng.choice(NUM_COLORS, size=inks + 2, replace=False)))
    colors = palette[1:-1]
    assignment = colors + list(map(int, rng.choice(colors, size=count - inks)))
    rng.shuffle(assignment)
    return {"h": h, "w": w, "count": count, "count_cap": cap,
            "count_class": count_class, "color_class": color_class,
            "regions": regions, "budgets": budgets, "shapes": shapes,
            "background": palette[0], "fill": palette[-1], "colors": assignment,
            "palette": palette[:-1], "thin_index": thin_index, "texture": texture}


def _draw(rng, plan):
    h, w = plan["h"], plan["w"]
    footprints, origins = [], []
    for (r, c, a, b), (mask, _) in zip(plan["regions"], plan["shapes"]):
        sh, sw = mask.shape
        r += int(rng.integers(a - sh + 1))
        c += int(rng.integers(b - sw + 1))
        placed = np.zeros((h, w), dtype=bool)
        placed[r:r + sh, c:c + sw] = mask
        footprints.append(placed)
        origins.append((r, c))
    positions = list(origins)
    moved = set()
    for _ in range(4):
        for index in rng.permutation(plan["count"]):
            index = int(index)
            other = np.zeros((h, w), dtype=bool)
            for j, footprint in enumerate(footprints):
                if j != index:
                    other |= footprint
            free = ~halo(other)
            free[[0, -1], :] = False
            free[:, [0, -1]] = False
            mask = plan["shapes"][index][0]
            candidates = np.argwhere(placement_map(mask, free))
            if not len(candidates):
                raise ValueError("Free relocation lost its current valid position")
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            if (r, c) != positions[index]:
                moved.add(index)
            sh, sw = mask.shape
            footprints[index][:] = False
            footprints[index][r:r + sh, c:c + sw] = mask
            positions[index] = r, c
    grid = np.full((h, w), plan["background"], dtype=int)
    for footprint, color in zip(footprints, plan["colors"]):
        grid[footprint] = color
    return grid, footprints, positions, len(moved)


def _ring(h, w):
    ring = np.zeros((h, w), dtype=bool)
    ring[[0, -1], :] = True
    ring[:, [0, -1]] = True
    return ring


def ring_shares(grid, ground):
    """(the ground's share of the border ring, the largest share of any other color).

    Objects are drawn strictly inside the canvas, so the ring carries the ground
    and the texture only; this is what keeps the ground the plain reading of the
    picture once dots sit on it.
    """
    a = np.asarray(grid)
    ring = _ring(*a.shape)
    total = int(ring.sum())
    values, counts = np.unique(a[ring], return_counts=True)
    tally = dict(zip(values.tolist(), counts.tolist()))
    rivals = [k for v, k in tally.items() if v != ground]
    return tally.get(ground, 0) / total, (max(rivals) / total if rivals else .0)


def _thin_ring(rng, display, grid, ground, colors):
    """Drop texture cells from the border ring until the ground plainly owns it.

    Removing cells never breaks the isolation the library guarantees.
    """
    ring = _ring(*display.shape)
    while True:
        share, rival = ring_shares(display, ground)
        if share >= RING_GROUND and rival <= RING_RIVAL:
            return display
        idx = np.argwhere(ring & np.isin(display, colors))
        if not len(idx):
            return display
        r, c = idx[int(rng.integers(len(idx)))]
        display[r, c] = grid[r, c]


def _texture(rng, grid, plan, occupied):
    """Paint one sparse library texture on the free cells of a finished drawing.

    Isolated dots one cell clear of every object: a dot is a one-cell object with
    no interior cell, so the answer is the same picture whether a reader calls the
    dots background or objects.  Returns ``(display, metadata)`` and falls back to
    the plain ground when no attempt produces a readable picture.
    """
    if not plan["texture"]:
        return grid, dict(_NO_TEXTURE)
    ground, fill = plan["background"], plan["fill"]
    free = NUM_COLORS - len(np.unique(grid)) - 1          # minus the fill color
    kinds = [k for k in SPARSE_KINDS if free >= 2 or k != "multi_speckle"]
    room = int((~halo(occupied)).sum())
    for _ in range(TEXTURE_TRIES):
        kind = kinds[int(rng.integers(len(kinds)))] if kinds else None
        painted = None if kind is None else apply_background(
            rng, grid, ground, halo(occupied), group="sparse", kind=kind, exclude=[fill])
        if painted is None:
            continue
        display, info = painted
        display = _thin_ring(rng, display.copy(), grid, ground, info["texture_colors"])
        cells = display != grid
        if int(cells.sum()) < MIN_TEXTURE_CELLS:
            continue      # a couple of dots left over from thinning is not a texture
        if not texture_cells_isolated(display, info["texture_colors"]):
            continue
        try:
            if figure_ground_field(display) != ground:
                continue
        except ValueError:
            continue
        return display, {
            "background_style": info["kind"], "texture_group": info["group"],
            "texture_kind": info["kind"],
            "texture_colors": sorted({int(c) for c in np.unique(display[cells])}),
            "texture_cells": int(cells.sum()),
            "texture_coverage": float(int(cells.sum()) / max(1, room)),
            "texture_named": False,
        }
    return grid, dict(_NO_TEXTURE)


def literal_solution(grid, fill_color, rule):
    """Validator: inspect raw neighbors of each original cell, without masks."""
    if rule not in (4, 8):
        raise ValueError("The exterior-contact rule must be 4 or 8")
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    ground = figure_ground_field(rows)
    result = [list(row) for row in rows]
    for r, row in enumerate(rows):
        for c, color in enumerate(row):
            if color == ground:
                continue
            interior = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (not dr and not dc) or (rule == 4 and dr and dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w) or rows[rr][cc] != color:
                        interior = False
            if interior:
                result[r][c] = fill_color
    return result


def _visible_objects(grid, background):
    remaining = {(r, c) for r, row in enumerate(grid) for c, color in enumerate(row) if color != background}
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = grid[start[0]][start[1]]
        part, pending = {start}, [start]
        for r, c in pending:
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining and grid[q[0]][q[1]] == color:
                    remaining.remove(q)
                    part.add(q)
                    pending.append(q)
        parts.append((color, part))
    return parts


def validate_example(example):
    grid, output, meta = example["input"], example["output"], example["metadata"]
    ground = figure_ground_field(grid)
    if ground != meta["background_color"]:
        raise ValueError("The visible field disagrees with the plan")
    # The re-solver reads the picture as the prompt does and knows nothing about
    # textures: a dot is a one-cell group whose only cell touches the outside.
    if literal_solution(grid, meta["fill_color"], meta["connectivity"]) != output:
        raise ValueError("Interior recoloring disagrees with the literal exterior-contact rule")
    texture = sorted(int(c) for c in meta["texture_colors"])
    parts = _visible_objects(grid, ground)
    dots = [(color, cells) for color, cells in parts if len(cells) == 1]
    bodies = [(color, cells) for color, cells in parts if len(cells) > 1]
    sizes = sorted(len(cells) for _, cells in bodies)
    if len(bodies) != meta["planned_object_count"] or len(bodies) != meta["n_objects"]:
        raise ValueError("The object count changed during placement")
    if sizes != sorted(meta["object_sizes"]) or (sizes and min(sizes) < 2):
        raise ValueError("The object footprints disagree with the plan")
    colors = set(value for row in grid for value in row)
    if (colors != set(meta["planned_palette"]) | set(texture) or len(colors) != meta["n_colors"]
            or meta["fill_color"] in colors):
        raise ValueError("The planned palette was not preserved")
    _validate_texture(grid, output, meta, ground, texture, dots, bodies)
    for rule in (4, 8):
        answer = literal_solution(grid, meta["fill_color"], rule)
        actual = sum(a != b for row, target in zip(grid, answer) for a, b in zip(row, target))
        if actual != meta[f"interior_cells_{rule}"] or actual == 0:
            raise ValueError("The interior-area certificate is incorrect")


def _validate_texture(grid, output, meta, ground, texture, dots, bodies):
    """Everything the texture has to be, recomputed from the emitted picture."""
    a = np.asarray(grid)
    style, kind = meta["background_style"], meta["texture_kind"]
    if (style == "plain") != (kind is None) or (kind is not None and style != kind):
        raise ValueError("The background style disagrees with the texture kind")
    if meta["texture_named"] or bool(texture) != bool(kind):
        raise ValueError("The texture metadata disagrees with the picture")
    forbidden = {int(ground), int(meta["fill_color"])} | {int(c) for c in meta["object_colors"]}
    if set(texture) & forbidden or len(set(texture)) != len(texture):
        raise ValueError("A texture color repeats the ground, the fill or an object color")
    cells = np.isin(a, texture) if texture else np.zeros(a.shape, dtype=bool)
    if int(cells.sum()) != meta["texture_cells"] or len(dots) != int(cells.sum()):
        raise ValueError("The texture cells are not the isolated single cells of the picture")
    if texture and not texture_cells_isolated(a, texture):
        raise ValueError("Two texture cells touch: the dots must stay isolated")
    objects = np.zeros(a.shape, dtype=bool)
    for _, body in bodies:
        for r, c in body:
            objects[r, c] = True
    if (halo(objects) & cells).any():
        raise ValueError("A texture cell touches an object: the dots must stay isolated from them")
    if (np.asarray(output)[cells] != a[cells]).any():
        raise ValueError("A texture cell changed in the output")
    share, rival = ring_shares(a, ground)
    if share < RING_GROUND or rival > RING_RIVAL:
        raise ValueError("The ground no longer owns the border ring")


def generate_fill_leaving_shell(rng):
    plan = _plan(rng)
    grid, footprints, positions, moved = _draw(rng, plan)
    occupied = np.zeros(grid.shape, dtype=bool)
    for footprint in footprints:
        occupied |= footprint
    # The texture is part of the drawing, painted before the rule is drawn and
    # one clear cell away from every body, so it moves no interior either way.
    grid, texture = _texture(rng, grid, plan, occupied)
    # Both requests are defined and have a nonempty interior on this drawing.
    # Nothing in the input construction has access to the selected rule.
    interiors = {rule: [_inner(mask, rule) for mask in footprints] for rule in (4, 8)}
    rule = int(rng.choice((4, 8)))
    output = grid.copy()
    for mask in interiors[rule]:
        output[mask] = plan["fill"]
    counts = Counter(grid.ravel().tolist())
    sizes = [int(mask.sum()) for mask in footprints]
    areas = {key: [int(mask.sum()) for mask in value] for key, value in interiors.items()}
    meta = {
        "grid_h": plan["h"], "grid_w": plan["w"], "connectivity": rule,
        "object_connectivity": 4, "shell_thickness": 1, "fill_color": plan["fill"],
        "background_color": plan["background"], **texture,
        "background_margin": counts[plan["background"]] / max(n for color, n in counts.items() if color != plan["background"]),
        "n_colors": len(counts), "planned_palette": plan["palette"],
        "n_objects": plan["count"], "planned_object_count": plan["count"],
        "object_count_cap": plan["count_cap"], "planned_count_class": plan["count_class"],
        "planned_color_class": plan["color_class"], "object_colors": plan["colors"],
        "object_kinds": [kind for _, kind in plan["shapes"]], "object_sizes": sizes,
        "object_bounds": [[r, c, *mask.shape] for (r, c), (mask, _) in zip(positions, plan["shapes"])],
        "planned_bbox_budgets": plan["budgets"], "foreground_cells": sum(sizes),
        "occupancy": sum(sizes) / grid.size, "interior_areas_4": areas[4], "interior_areas_8": areas[8],
        "interior_cells_4": sum(areas[4]), "interior_cells_8": sum(areas[8]),
        "connectivity_matters": areas[4] != areas[8], "unchanged_objects": areas[rule].count(0),
        "attempts": 1, "placement": "free_relocation", "relocation_rounds": 4, "relocated_objects": moved,
    }
    example = {"input": grid.tolist(), "output": output.tolist(), "metadata": meta}
    validate_example(example)
    return example
