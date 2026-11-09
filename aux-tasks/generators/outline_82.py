"""Readable outlines of freely placed, scaled drawings on a plain field.

A finite partition witnesses space for the chosen count.  Bodies, colors and
count are then fixed while every body is freely relocated over the canvas.
Only after that complete input is drawn do we sample the outline thickness.
All three widths are eligible on every input; clipping is ordinary geometry.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from generators.scene_shapes_56_60 import shape as familiar_shape


WIDTHS = (1, 2, 3)
WIDTH_WEIGHTS = (.56, .34, .10)
MAX_OBJECTS = 6
FREE_PLACEMENT_PASSES = 3


def _capacity(h, w, gap):
    return min(MAX_OBJECTS, ((h + gap) // (3 + gap)) * ((w + gap) // (3 + gap)))


def _partition(rng, box, count, gap):
    """Uneven guillotine boxes are a capacity witness, not visible slots."""
    if count == 1:
        return [box]
    r, c, h, w = box
    options = []
    for axis, (length, cross) in enumerate(((h, w), (w, h))):
        for cut in range(3, length - gap - 2):
            left = _capacity(cut, cross, gap)
            right = _capacity(length - gap - cut, cross, gap)
            for n in range(max(1, count - right), min(count - 1, left) + 1):
                options.append((axis, cut, n))
    if not options:
        raise RuntimeError("Task 82: the count has no partition witness")
    axis, cut, first = options[int(rng.integers(len(options)))]
    if axis == 0:
        a, b = (r, c, cut, w), (r + cut + gap, c, h - cut - gap, w)
    else:
        a, b = (r, c, h, cut), (r, c + cut + gap, h, w - cut - gap)
    return _partition(rng, a, first, gap) + _partition(rng, b, count - first, gap)


def _body(rng, h, w):
    choices = ["rectangle", "rectangle", "rectangle", "L", "staircase"]
    if min(h, w) >= 3:
        choices += ["T", "U", "H", "plus", "frame"]
    if min(h, w) >= 5:
        choices += ["chamfer", "diamond"]
    # Bars have an intentional long axis.  Large bars grow in both dimensions.
    if max(h, w) >= 2 * min(h, w):
        choices += ["bar", "bar"]
    family = str(rng.choice(choices))
    if family == "bar":
        return np.ones((h, w), dtype=bool), family, 1
    scale = 3 if min(h, w) >= 15 else 2 if min(h, w) >= 7 else 1
    # Families requiring five base cells retain that resolution before scaling.
    if family in ("chamfer", "diamond") and min(h, w) // scale < 5:
        scale = 1
    mask, actual = familiar_shape(rng, h // scale, w // scale, family)
    if actual != family:
        raise RuntimeError("Task 82: a planned shape family changed")
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1), family, scale


def _dilate(mask, width):
    """Construction: grow the mask by one full 3x3 neighborhood per step."""
    result = mask.copy()
    h, w = mask.shape
    for _ in range(width):
        padded = np.pad(result, 1)
        result = np.logical_or.reduce([
            padded[dr:dr + h, dc:dc + w] for dr in range(3) for dc in range(3)
        ])
    return result


def _put(mask, position, canvas_shape):
    result = np.zeros(canvas_shape, dtype=bool)
    r, c = position
    result[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
    return result


def _free_positions(mask, blocked, margins, anchor):
    """Enumerate every valid translation, not a bounded list of lucky guesses."""
    h, w = blocked.shape
    sh, sw = mask.shape
    top, bottom, left, right = margins
    nr, nc = h - sh + 1, w - sw + 1
    ok = np.ones((nr, nc), dtype=bool)
    for r, c in np.argwhere(mask):
        ok &= ~blocked[r:r + nr, c:c + nc]
    rr, cc = np.indices(ok.shape)
    ok &= (rr >= top) & (rr <= h - sh - bottom)
    ok &= (cc >= left) & (cc <= w - sw - right)
    if anchor == 0:
        ok &= rr == top
    elif anchor == 1:
        ok &= rr == h - sh - bottom
    elif anchor == 2:
        ok &= cc == left
    elif anchor == 3:
        ok &= cc == w - sw - right
    return np.argwhere(ok)


def _plan_and_draw(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    mh, mw = min(3, max(1, (h - 3) // 2)), min(3, max(1, (w - 3) // 2))
    margins = [mh, mh, mw, mw]
    border_near = bool(rng.random() < .14)
    edge = int(rng.integers(4)) if border_near else None
    if edge is not None:
        margins[edge] = int(rng.integers(2))
    close = bool(rng.random() < .20)
    gap = int(rng.integers(1, 3)) if close else 4
    top, bottom, left, right = margins
    inner = (top, left, h - top - bottom, w - left - right)
    capacity = _capacity(inner[2], inner[3], gap)
    weights = np.asarray((.28, .32, .20, .10, .07, .03))[:capacity]
    count = int(rng.choice(np.arange(1, capacity + 1), p=weights / weights.sum()))
    boxes = _partition(rng, inner, count, gap)
    box_area = sum(bh * bw for _, _, bh, bw in boxes)
    colors = list(map(int, rng.permutation(10)))
    bg, outline = colors.pop(), colors.pop()
    # Coherent monochrome bodies keep the basic scene easy to read.  Distinct
    # colors are used on some multi-object scenes; no palette-filling clutter.
    palette_count = int(rng.integers(2, count + 1)) if count > 1 and rng.random() < .80 else 1
    palette = colors[:palette_count]
    inks = (palette * count)[:count]
    rng.shuffle(inks)
    bodies, positions, families, scales = [], [], [], []
    anchor_index = None
    for index, (r, c, bh, bw) in enumerate(boxes):
        budget = max(4, int(.46 * h * w * bh * bw / box_area))
        candidates = [(sh, sw) for sh in range(2, bh + 1) for sw in range(2, bw + 1)
                      if sh * sw <= budget]
        minimum_area = .60 * max(sh * sw for sh, sw in candidates)
        candidates = [(sh, sw) for sh, sw in candidates if sh * sw >= minimum_area]
        # Bodies visibly use their allotted space, with aspect-ratio variation.
        # The total bounding-box area is <= 46%; a large canvas never receives
        # an accidentally tiny body from the low end of a weighted size draw.
        scores = np.asarray([sh * sw for sh, sw in candidates], dtype=float) ** 3
        sh, sw = candidates[int(rng.choice(len(candidates), p=scores / scores.sum()))]
        mask, family, scale = _body(rng, sh, sw)
        sh, sw = mask.shape
        pr, pc = int(rng.integers(r, r + bh - sh + 1)), int(rng.integers(c, c + bw - sw + 1))
        eligible_edge = (edge == 0 and r == top or edge == 1 and r + bh == h - bottom
                         or edge == 2 and c == left or edge == 3 and c + bw == w - right)
        if anchor_index is None and eligible_edge:
            anchor_index = index
            if edge == 0:
                pr = top
            elif edge == 1:
                pr = h - bottom - sh
            elif edge == 2:
                pc = left
            else:
                pc = w - right - sw
        bodies.append(mask)
        positions.append((pr, pc))
        families.append(family)
        scales.append(scale)
    if edge is not None and anchor_index is None:
        raise RuntimeError("Task 82: border plan lost its capacity witness")
    witness_positions = list(positions)
    moved = 0
    for _ in range(FREE_PLACEMENT_PASSES):
        for i in map(int, rng.permutation(count)):
            others = np.zeros((h, w), dtype=bool)
            for j in range(count):
                if j != i:
                    others |= _put(bodies[j], positions[j], (h, w))
            candidates = _free_positions(bodies[i], _dilate(others, gap), margins,
                                         edge if i == anchor_index else None)
            if not len(candidates):
                raise RuntimeError("Task 82: a fixed body lost all valid placements")
            position = tuple(map(int, candidates[int(rng.integers(len(candidates)))]))
            moved += position != positions[i]
            positions[i] = position
    grid = np.full((h, w), bg, dtype=int)
    records, masks = [], []
    for index, (body, position, family, scale, color) in enumerate(zip(bodies, positions, families, scales, inks)):
        mask = _put(body, position, (h, w))
        grid[mask] = color
        masks.append(mask)
        records.append({"plan_index": index, "color": color, "family": family, "scale": scale,
                        "bounds": [*position, *body.shape], "area": int(body.sum()),
                        "first_cell": np.argwhere(mask)[0].tolist()})
    records.sort(key=lambda obj: obj["first_cell"])
    return grid, masks, {
        "grid_h": h, "grid_w": w, "background_color": bg,
        "background_style": "solid", "outline_color": outline,
        "planned_n_objects": count, "witnessed_capacity": capacity,
        "planned_families": sorted(families), "planned_object_colors": sorted(inks),
        "planned_object_areas": sorted(int(body.sum()) for body in bodies),
        "objects": records, "placement_gap": gap, "placement_margins": margins,
        "border_near_plan": border_near, "border_anchor_edge": edge, "close_spacing_plan": close,
        "capacity_boxes": [list(box) for box in boxes],
        "capacity_witness_positions": [list(p) for p in witness_positions],
        "free_placement_passes": FREE_PLACEMENT_PASSES, "free_relocations": moved,
        "attempts": 1, "query_drawn_after_input": True,
        "eligible_outline_thicknesses": list(WIDTHS),
    }


def _ground(grid):
    """Read the surrounding field; its identity is never supplied to a solver."""
    ring = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    counts = Counter(map(int, ring))
    bg, mass = counts.most_common(1)[0]
    if mass <= len(ring) / 2 or np.count_nonzero(grid == bg) <= grid.size / 2:
        raise ValueError("Task 82: no clear surrounding background")
    return bg, mass / len(ring)


def _objects(grid, bg):
    remaining = set(map(tuple, np.argwhere(grid != bg).tolist()))
    objects = []
    while remaining:
        first = min(remaining)
        color = int(grid[first])
        remaining.remove(first)
        cells, queue = [first], deque([first])
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = (r + dr, c + dc)
                    if point in remaining and int(grid[point]) == color:
                        remaining.remove(point)
                        cells.append(point)
                        queue.append(point)
        if len(cells) < 2:
            raise ValueError("Task 82: an isolated cell is not a planned body")
        mask = np.zeros(grid.shape, dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        objects.append((color, cells, mask))
    if not objects:
        raise ValueError("Task 82: no visible objects")
    return objects


def _distances(mask):
    """Independent exact distance: two raster passes, no repeated dilation."""
    h, w = mask.shape
    far = h + w + 1
    distance = np.where(mask, 0, far).astype(int)
    for r in range(h):
        for c in range(w):
            if r:
                distance[r, c] = min(distance[r, c], 1 + int(distance[r - 1, max(0, c - 1):min(w, c + 2)].min()))
            if c:
                distance[r, c] = min(distance[r, c], distance[r, c - 1] + 1)
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if r + 1 < h:
                distance[r, c] = min(distance[r, c], 1 + int(distance[r + 1, max(0, c - 1):min(w, c + 2)].min()))
            if c + 1 < w:
                distance[r, c] = min(distance[r, c], distance[r, c + 1] + 1)
    return distance


def solve_outline(grid, thickness, color):
    """Re-solve from the visible field and explicit prompt givens only."""
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or min(grid.shape) < 1 or thickness not in WIDTHS or not 0 <= color <= 9:
        raise ValueError("Task 82: invalid grid or outline width")
    bg, _ = _ground(grid)
    if color in grid:
        raise ValueError("Task 82: outline color must be reserved")
    _objects(grid, bg)
    distances = _distances(grid != bg)
    out = grid.copy()
    out[(grid == bg) & (distances <= thickness)] = color
    return out


def _facts(grid, objects, thickness):
    bg, border_share = _ground(grid)
    bands = []
    clipped = 0
    object_gap = max(grid.shape)
    for _, cells, mask in objects:
        coordinates = np.asarray(cells)
        clearance = min(int(coordinates[:, 0].min()), int(coordinates[:, 1].min()),
                        grid.shape[0] - 1 - int(coordinates[:, 0].max()),
                        grid.shape[1] - 1 - int(coordinates[:, 1].max()))
        clipped += clearance < thickness
        distance = _distances(mask)
        bands.append((distance <= thickness) & (grid == bg))
        other_cells = (grid != bg) & ~mask
        if np.any(other_cells):
            object_gap = min(object_gap, int(distance[other_cells].min()) - 1)
    multiplicity = np.sum(bands, axis=0)
    overlap_pairs = meeting_pairs = 0
    for i, a in enumerate(bands):
        for b in bands[i + 1:]:
            overlaps = bool(np.any(a & b))
            overlap_pairs += overlaps
            # This checker uses direct adjacent-cell comparisons, not dilation.
            touches = overlaps
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    ar = slice(max(0, dr), min(grid.shape[0], grid.shape[0] + dr))
                    ac = slice(max(0, dc), min(grid.shape[1], grid.shape[1] + dc))
                    br = slice(max(0, -dr), min(grid.shape[0], grid.shape[0] - dr))
                    bc = slice(max(0, -dc), min(grid.shape[1], grid.shape[1] - dc))
                    touches |= bool(np.any(a[ar, ac] & b[br, bc]))
            meeting_pairs += touches
    distances = _distances(grid != bg)
    return {
        "n_objects": len(objects), "n_colors": int(len(np.unique(grid))),
        "object_cells": int(np.count_nonzero(grid != bg)),
        "minimum_object_gap": object_gap if len(objects) > 1 else None,
        "background_border_share": border_share,
        "background_share": float(np.mean(grid == bg)),
        "n_clipped_outlines": clipped, "overlapping_band_pairs": overlap_pairs,
        "meeting_or_overlapping_band_pairs": meeting_pairs,
        "overlap_cells": int(np.count_nonzero(multiplicity > 1)),
        "output_changed_cells": int(np.count_nonzero(multiplicity)),
        "distinct_width_outputs": len({(distances <= t).tobytes() for t in WIDTHS}),
        "original_objects_preserved": True,
    }


def validate_outline(example):
    """Reject answer and certificate drift using a final-visible-grid re-solve."""
    grid, out = np.asarray(example["input"]), np.asarray(example["output"])
    m = example["metadata"]
    if grid.ndim != 2 or grid.shape != out.shape or tuple(grid.shape) != (m["grid_h"], m["grid_w"]):
        raise ValueError("Task 82: grid dimensions changed")
    if (min(grid.shape) < 5 or max(grid.shape) > 30 or np.any((grid < 0) | (grid > 9))
            or np.any((out < 0) | (out > 9))):
        raise ValueError("Task 82: invalid canvas")
    bg, _ = _ground(grid)
    if bg != m["background_color"] or m["background_style"] != "solid":
        raise ValueError("Task 82: background certificate changed")
    t = m["outline_thickness"]
    if m["eligible_outline_thicknesses"] != list(WIDTHS) or not m["query_drawn_after_input"]:
        raise ValueError("Task 82: the input-dependent query certificate changed")
    expected = solve_outline(grid, t, m["outline_color"])
    if not np.array_equal(out, expected):
        raise ValueError("Task 82: output disagrees with the visible-grid rule")
    objects = _objects(grid, bg)
    if not (len(objects) == m["planned_n_objects"] <= m["witnessed_capacity"] <= MAX_OBJECTS):
        raise ValueError("Task 82: object count drift")
    records = m["objects"]
    if len(records) != len(objects):
        raise ValueError("Task 82: object records changed")
    for record, (color, cells, _) in zip(records, objects):
        rr, cc = zip(*cells)
        bounds = [min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1]
        if (record["color"] != color or record["area"] != len(cells)
                or record["bounds"] != bounds or record["first_cell"] != list(min(cells))):
            raise ValueError("Task 82: visible object record changed")
    if (sorted(record["family"] for record in records) != m["planned_families"]
            or sorted(color for color, _, _ in objects) != m["planned_object_colors"]
            or sorted(len(cells) for _, cells, _ in objects) != m["planned_object_areas"]):
        raise ValueError("Task 82: shape or palette plan changed")
    facts = _facts(grid, objects, t)
    if any(m[key] != value for key, value in facts.items()):
        raise ValueError("Task 82: visible geometry certificate changed")
    if not facts["output_changed_cells"] or m["attempts"] != 1:
        raise ValueError("Task 82: missing outline or a rerolled plan")
    _validate_capacity(m, facts)


def _validate_capacity(m, facts):
    h, w, gap = m["grid_h"], m["grid_w"], m["placement_gap"]
    margins = m["placement_margins"]
    top, bottom, left, right = margins
    usual_h, usual_w = min(3, max(1, (h - 3) // 2)), min(3, max(1, (w - 3) // 2))
    usual = [usual_h, usual_h, usual_w, usual_w]
    edge = m["border_anchor_edge"]
    if m["border_near_plan"]:
        if edge not in range(4) or margins[edge] not in (0, 1):
            raise ValueError("Task 82: invalid border plan")
        usual[edge] = margins[edge]
    elif edge is not None:
        raise ValueError("Task 82: unexpected border anchor")
    if margins != usual or gap not in ((1, 2) if m["close_spacing_plan"] else (4,)):
        raise ValueError("Task 82: placement plan drift")
    if m["witnessed_capacity"] != _capacity(h - top - bottom, w - left - right, gap):
        raise ValueError("Task 82: invalid witnessed capacity")
    if facts["minimum_object_gap"] is not None and facts["minimum_object_gap"] < gap:
        raise ValueError("Task 82: visible bodies violate their separation plan")
    boxes, positions = m["capacity_boxes"], m["capacity_witness_positions"]
    records = sorted(m["objects"], key=lambda record: record["plan_index"])
    if (len(boxes) != len(records) or len(positions) != len(records)
            or [obj["plan_index"] for obj in records] != list(range(len(records)))):
        raise ValueError("Task 82: capacity witness lost a body")
    for box, position, record in zip(boxes, positions, records):
        r, c, bh, bw = box
        pr, pc = position
        rr, cc, sh, sw = record["bounds"]
        if (min(bh, bw) < 3 or r < top or c < left or r + bh > h - bottom or c + bw > w - right
                or not (r <= pr <= r + bh - sh and c <= pc <= c + bw - sw)
                or not (top <= rr <= h - bottom - sh and left <= cc <= w - right - sw)):
            raise ValueError("Task 82: invalid capacity placement")
    for i, (r, c, bh, bw) in enumerate(boxes):
        for rr, cc, sh, sw in boxes[i + 1:]:
            if max(rr - (r + bh - 1), r - (rr + sh - 1), cc - (c + bw - 1), c - (cc + sw - 1)) <= gap:
                raise ValueError("Task 82: capacity boxes are too close")
    if (m["free_placement_passes"] != FREE_PLACEMENT_PASSES
            or not 0 <= m["free_relocations"] <= FREE_PLACEMENT_PASSES * len(records)):
        raise ValueError("Task 82: invalid free-placement trace")


def generate(rng):
    grid, masks, metadata = _plan_and_draw(rng)
    thickness = int(rng.choice(WIDTHS, p=WIDTH_WEIGHTS))
    metadata["outline_thickness"] = thickness
    body = np.logical_or.reduce(masks)
    out = grid.copy()
    out[_dilate(body, thickness) & ~body] = metadata["outline_color"]
    metadata.update(_facts(grid, _objects(grid, metadata["background_color"]), thickness))
    example = {"input": grid.tolist(), "output": out.tolist(), "metadata": metadata}
    validate_outline(example)
    return example
