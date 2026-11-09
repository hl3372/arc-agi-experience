"""Familiar, freely positioned monochrome drawings for tasks 84 and 85.

The complete collection of masks, inks and a capacity witness is fixed before
placement. Moving a mask away from its witness never loses an object: its old
position remains a legal position. The two answer builders use planned masks;
the validator independently flood-fills the finished input and uses coordinate
rotations. No helper from a passed task is changed or used for answer logic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np


FAMILIES = ("bar", "rectangle", "L", "T", "U", "H", "plus", "frame",
            "staircase", "chamfer", "diamond")
PALETTE_WEIGHTS = np.asarray((.17, .14, .12, .10, .09, .09, .10, .10, .09))


def _capacity(h, w):
    """A conservative packing certificate for 3-cell bars or 2x2 hosts."""
    return max(((h + 1) // 2) * ((w + 1) // 4),
               ((h + 1) // 4) * ((w + 1) // 2),
               ((h + 1) // 3) * ((w + 1) // 3))


def _rooms(rng, h, w, count, margin):
    """Uneven recursive free areas, separated by one cell, certify capacity."""
    pending = [(margin, margin, h - 2 * margin, w - 2 * margin, count)]
    rooms = []
    while pending:
        r, c, a, b, n = pending.pop()
        if n == 1:
            rooms.append((r, c, a, b))
            continue
        choices, weights = [], []
        for axis, side in enumerate((a, b)):
            for cut in range(1, side - 1):
                aa, bb = (cut, b) if axis == 0 else (a, cut)
                cc, dd = (a - cut - 1, b) if axis == 0 else (a, b - cut - 1)
                cap_a, cap_b = _capacity(aa, bb), _capacity(cc, dd)
                for first in range(max(1, n - cap_b), min(n - 1, cap_a) + 1):
                    fraction = aa * bb / (aa * bb + cc * dd)
                    # Count and area track one another, with ample unevenness.
                    weight = np.exp(-14 * abs(fraction - first / n))
                    weight *= ((min(aa, bb) / max(aa, bb)) *
                               (min(cc, dd) / max(cc, dd))) ** .35
                    choices.append((axis, cut, first))
                    weights.append(weight)
        if not choices:
            raise RuntimeError("A certified object-room plan could not be divided")
        p = np.asarray(weights) / sum(weights)
        axis, cut, first = choices[int(rng.choice(len(choices), p=p))]
        if axis == 0:
            pending.extend(((r, c, cut, b, first),
                            (r + cut + 1, c, a - cut - 1, b, n - first)))
        else:
            pending.extend(((r, c, a, cut, first),
                            (r, c + cut + 1, a, b - cut - 1, n - first)))
    rng.shuffle(rooms)
    return rooms


def _mask_key(mask):
    return min((x.shape, x.tobytes()) for x in (np.rot90(mask, k).copy() for k in range(4)))


def _figure(family, h, w, t):
    y, x = np.indices((h, w))
    if family in ("rectangle", "bar"):
        mask = np.ones((h, w), bool)
    elif family == "L":
        mask = (x < t) | (y >= h - t)
    elif family == "T":
        mask = (y < t) | ((x >= (w - t) // 2) & (x < (w - t) // 2 + t))
    elif family == "U":
        mask = (x < t) | (x >= w - t) | (y >= h - t)
    elif family == "H":
        mask = (x < t) | (x >= w - t) | ((y >= (h - t) // 2) & (y < (h - t) // 2 + t))
    elif family == "plus":
        mask = ((x >= (w - t) // 2) & (x < (w - t) // 2 + t)) | ((y >= (h - t) // 2) & (y < (h - t) // 2 + t))
    elif family == "frame":
        mask = np.minimum.reduce((y, x, h - 1 - y, w - 1 - x)) < t
    elif family == "staircase":
        steps = min(4, h, w)
        mask = x < np.ceil((np.minimum(steps - 1, y * steps // h) + 1) * w / steps)
    elif family == "chamfer":
        mask = (y + x >= t) & (y + w - 1 - x >= t) & (h - 1 - y + x >= t) & (h - 1 - y + w - 1 - x >= t)
    elif family == "diamond":
        # Odd spans produce one continuous, symmetric filled diamond.
        mask = abs(y - h // 2) / max(1, h // 2) + abs(x - w // 2) / max(1, w // 2) <= 1 + 1e-9
    else:
        raise ValueError(family)
    return np.asarray(mask, bool)


def _catalog(rng, short, long, budget, *, chiral=False):
    """Feasible drawn candidates, including small witnesses, with no blobs."""
    candidates = []
    families = ("L", "staircase") if chiral else FAMILIES
    for family in families:
        low = 1 if family == "bar" else 2 if family in ("rectangle", "L", "staircase") else 5 if family in ("chamfer", "diamond") else 3
        if short < low or long < (2 if family == "bar" else low):
            continue
        dimensions = []
        for _ in range(14):
            a = 1 if family == "bar" else int(rng.integers(low, short + 1))
            b = int(rng.integers(max(2, low), long + 1))
            dimensions.append((a, b))
        dimensions.extend(((low, max(2, low)), (1 if family == "bar" else short, long)))
        if family == "bar":
            dimensions.extend((1, n) for n in range(2, long + 1))
        for a, b in dimensions:
            if family == "diamond":
                a -= not a % 2
                b -= not b % 2
            max_t = max(1, (min(a, b) - 1) // 2)
            t = 1 if family == "bar" else int(rng.integers(1, max_t + 1))
            mask = _figure(family, a, b, t)
            cells = int(mask.sum())
            if not 2 <= cells <= budget:
                continue
            if chiral and _mask_key(mask) == _mask_key(mask[:, ::-1]):
                continue
            candidates.append((mask, family))
    return candidates


def _choose_shape(rng, rooms, budgets, indices, forbidden=(), *, chiral=False):
    short = min(min(rooms[i][2:]) for i in indices)
    long = min(max(rooms[i][2:]) for i in indices)
    budget = min(budgets[i] for i in indices)
    candidates = [(mask, kind) for mask, kind in _catalog(rng, short, long, budget, chiral=chiral)
                  if _mask_key(mask) not in forbidden]
    if not candidates:
        return None
    families = sorted({kind for _, kind in candidates})
    family = str(rng.choice(families))
    choices = [(mask, kind) for mask, kind in candidates if kind == family]
    # Choose a substantial body of the selected family within its cell budget.
    top = max(int(mask.sum()) for mask, _ in choices)
    choices = [(mask, kind) for mask, kind in choices if mask.sum() >= .72 * top]
    return choices[int(rng.integers(len(choices)))]


def _orient(rng, mask, room, preferred=None):
    a, b = room[2:]
    options = [k for k in range(4) if np.rot90(mask, k).shape[0] <= a and np.rot90(mask, k).shape[1] <= b]
    if not options:
        raise ValueError("An object does not fit its planned room")
    k = preferred if preferred in options else int(rng.choice(options))
    return np.rot90(mask, k).copy(), k


def _group_indices(rng, rooms, available, count):
    """Similar host sizes avoid forcing large copies into a tiny leftover area."""
    ordered = sorted(available, key=lambda i: (min(rooms[i][2:]), max(rooms[i][2:])))
    windows = [ordered[j:j + count] for j in range(len(ordered) - count + 1)]
    scores = []
    for group in windows:
        areas = [rooms[i][2] * rooms[i][3] for i in group]
        scores.append((min(areas) / max(areas)) ** 2)
    return windows[int(rng.choice(len(windows), p=np.asarray(scores) / sum(scores)))]


def _plan(rng, task):
    h, w = map(int, rng.integers(5, 31, size=2))
    margin = 1 if _capacity(h - 2, w - 2) >= 3 else 0
    cap = min(14, _capacity(h - 2 * margin, w - 2 * margin), max(3, h * w // 12))
    mode = str(rng.choice(("none", "some", "all"), p=(.04, .92, .04))) if task == "84" else "scene"
    if mode == "none":
        # Plain, simple no-match examples are a small intentional minority.
        # A 3x5 interior holds three bars, but only two distinct bar shapes.
        distinct_cap = 2 if (h - 2 * margin) * (w - 2 * margin) <= 15 else 3
        cap = min(cap, distinct_cap)
    palette_cap = min(10, cap + 1)
    weights = PALETTE_WEIGHTS[:palette_cap - 1]
    n_colors = int(rng.choice(np.arange(2, palette_cap + 1), p=weights / weights.sum()))
    minimum = max(n_colors - 1, 3 if mode == "some" else 2 if mode in ("all", "scene") else 1)
    count = minimum if rng.random() < .50 else int(rng.integers(minimum, cap + 1))
    rooms = _rooms(rng, h, w, count, margin)
    density = str(rng.choice(("open", "full"), p=(.25, .75)))
    target = float(rng.uniform(.26, .34) if density == "open" else rng.uniform(.39, .44))
    total_room_area = sum(a * b for _, _, a, b in rooms)
    budgets = [max(3, int(target * h * w * a * b / total_room_area)) for _, _, a, b in rooms]
    # On the few unframed tiny canvases, no object spans a complete grid axis.
    rooms = [(r, c, min(a, h - 1), min(b, w - 1)) for r, c, a, b in rooms]
    palette = list(map(int, rng.choice(10, size=n_colors, replace=False)))
    colors = palette[1:] + list(map(int, rng.choice(palette[1:], size=count - n_colors + 1)))
    rng.shuffle(colors)
    specs = [None] * count
    mirror = False

    if task == "85":
        for i in range(count):
            chosen = _choose_shape(rng, rooms, budgets, [i])
            if chosen is None:
                raise RuntimeError("No familiar shape fits a certified room")
            specs[i] = (*chosen, None)
    elif mode == "none":
        used = set()
        # Restrictive rooms get their choice before roomier rooms.
        for i in sorted(range(count), key=lambda i: rooms[i][2] * rooms[i][3]):
            chosen = _choose_shape(rng, rooms, budgets, [i], used)
            if chosen is None:
                # The count was capacity-planned; extend the local cell budget,
                # not the count or family, to a fitting distinct drawn witness.
                limits = budgets.copy()
                limits[i] = rooms[i][2] * rooms[i][3]
                chosen = _choose_shape(rng, rooms, limits, [i], used)
            if chosen is None:
                raise RuntimeError("No distinct familiar shape fits a no-match room")
            used.add(_mask_key(chosen[0]))
            specs[i] = (*chosen, None)
    else:
        available = list(range(count))
        unique = int(rng.choice(available)) if mode == "some" else None
        if unique is not None:
            available.remove(unique)
        groups = []
        while len(available) >= 2:
            maximum = min(5, len(available))
            if mode == "all" and len(available) == 3:
                members = 3
            else:
                members = int(rng.choice(range(2, maximum + 1))) if rng.random() < .45 else 2
                if mode == "all" and len(available) - members == 1:
                    members += 1
            group = _group_indices(rng, rooms, available, members)
            groups.append(group)
            available = [i for i in available if i not in group]
            if mode == "some" and (rng.random() < .55 or len(groups) == 3):
                break
        forbidden = set()
        if unique is not None:
            if rng.random() < .30:
                chosen = _choose_shape(rng, rooms, budgets, groups[0] + [unique], chiral=True)
                if chosen is not None:
                    mask, kind = chosen
                    specs[unique] = (mask[:, ::-1].copy(), kind, None)
                    for i in groups[0]:
                        specs[i] = (mask, kind, None)
                    mirror = True
            if specs[unique] is None:
                forced_keys = set()
                for group in groups:
                    short = min(min(rooms[i][2:]) for i in group)
                    long = min(max(rooms[i][2:]) for i in group)
                    if short == 1 and long == 2:
                        forced_keys.add(_mask_key(np.ones((1, 2), bool)))
                chosen = _choose_shape(rng, rooms, budgets, [unique], forced_keys)
                if chosen is None:
                    raise RuntimeError("No unique shape fits its room")
                specs[unique] = (*chosen, None)
            forbidden.add(_mask_key(specs[unique][0]))
        for group in groups:
            chosen = None if specs[group[0]] is None else specs[group[0]][:2]
            if chosen is None:
                chosen = _choose_shape(rng, rooms, budgets, group, forbidden)
            if chosen is None:
                raise RuntimeError("No matching shape fits its certified rooms")
            preferred = int(rng.integers(4)) if rng.random() < .35 else None
            for i in group:
                specs[i] = (*chosen, preferred)
        for i in available:
            chosen = _choose_shape(rng, rooms, budgets, [i], forbidden)
            if chosen is None:
                raise RuntimeError("No remaining familiar shape fits its room")
            specs[i] = (*chosen, None)

    bodies = []
    for i, (mask, kind, preferred) in enumerate(specs):
        mask, angle = _orient(rng, mask, rooms[i], preferred)
        r, c, a, b = rooms[i]
        r += int(rng.integers(a - mask.shape[0] + 1))
        c += int(rng.integers(b - mask.shape[1] + 1))
        bodies.append({"mask": mask, "kind": kind, "color": colors[i], "r": r, "c": c})
    return {"h": h, "w": w, "margin": margin, "palette": palette, "bodies": bodies,
            "count": count, "count_cap": cap, "density": density, "outcome_plan": mode,
            "mirror_plan": mirror, "contact_plan": bool(rng.random() < .30)}


def _dilate(mask, diagonal):
    padded = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)) + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ()):
        result |= padded[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]]
    return result


def _locations(mask, free):
    a, b = mask.shape
    if a > free.shape[0] or b > free.shape[1]:
        return np.empty((0, 2), int)
    windows = np.lib.stride_tricks.sliding_window_view(free, (a, b))
    return np.argwhere(np.all(windows | ~mask, axis=(-2, -1)))


def _place(rng, plan):
    h, w, bodies = plan["h"], plan["w"], plan["bodies"]
    owner = np.full((h, w), -1, int)
    for i, body in enumerate(bodies):
        r, c, mask = body["r"], body["c"], body["mask"]
        owner[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = i
    moved = set()
    for _ in range(2):
        for index in rng.permutation(len(bodies)):
            i = int(index)
            body = bodies[i]
            owner[owner == i] = -1
            occupied = owner >= 0
            if plan["contact_plan"]:
                same = np.zeros((h, w), bool)
                for j, other in enumerate(bodies):
                    if j != i and other["color"] == body["color"]:
                        same |= owner == j
                free = ~(occupied | _dilate(same, False))
            else:
                free = ~_dilate(occupied, True)
            if plan["margin"]:
                free[0, :] = free[-1, :] = False
                free[:, 0] = free[:, -1] = False
            spots = _locations(body["mask"], free)
            if not len(spots):
                raise RuntimeError("A complete placement plan lost its legal witness")
            r, c = map(int, spots[int(rng.integers(len(spots)))])
            if (r, c) != (body["r"], body["c"]):
                moved.add(i)
            body["r"], body["c"] = r, c
            a, b = body["mask"].shape
            owner[r:r + a, c:c + b][body["mask"]] = i
    grid = np.full((h, w), plan["palette"][0], int)
    for i, body in enumerate(bodies):
        grid[owner == i] = body["color"]
    return grid, owner, len(moved)


def choose_filter_kind(rng):
    """Called only after the complete task-85 input has been drawn."""
    return str(rng.choice(("color", "size=", "size<", "size>")))


def _filter(rng, bodies, bg, kind):
    sizes = sorted({int(b["mask"].sum()) for b in bodies})
    colors = sorted({b["color"] for b in bodies})
    if kind == "color":
        some = colors if len(colors) > 1 else []
        all_values = colors if len(colors) == 1 else []
        # the field colour is never a filter value: the prompt would name the background (sweep 2026-09-17)
        none = [c for c in range(10) if c not in colors and c != bg]
    elif kind == "size=":
        some = sizes if len(sizes) > 1 else []
        all_values = sizes if len(sizes) == 1 else []
        none = [n for n in range(1, max(sizes) + 3) if n not in sizes]
    elif kind == "size<":
        some = list(range(min(sizes) + 1, max(sizes) + 1))
        all_values = [max(sizes) + 1, max(sizes) + 2]
        none = [min(sizes), max(1, min(sizes) - 1)]
    elif kind == "size>":
        some = list(range(min(sizes), max(sizes)))
        all_values = [min(sizes) - 1, max(0, min(sizes) - 2)]
        none = [max(sizes), max(sizes) + 1]
    else:
        raise ValueError(kind)
    pools = {"none": none, "some": some, "all": all_values}
    labels = [name for name, values in pools.items() if values]
    weights = np.asarray([{"none": .02, "some": .96, "all": .02}[name] for name in labels])
    mode = str(rng.choice(labels, p=weights / weights.sum()))
    values = pools[mode]
    if kind in ("size<", "size>") and mode == "some":
        observed = [n for n in values if n in sizes]
        between = [n for n in values if n not in sizes]
        # Both strict boundaries and values between observed sizes matter.
        if observed and between:
            values = observed if rng.random() < .5 else between
    value = int(rng.choice(values))
    selected = []
    for body in bodies:
        n = int(body["mask"].sum())
        selected.append(body["color"] == value if kind == "color" else n == value if kind == "size=" else n < value if kind == "size<" else n > value)
    return value, selected


def _visible(grid):
    """Independent literal reader: infer a plain field and flood-fill colors."""
    h, w = grid.shape
    counts = Counter(map(int, grid.flat))
    ranked = counts.most_common()
    if len(ranked) < 2 or ranked[0][1] < 1.25 * ranked[1][1]:
        raise ValueError("The plain field is not independently identifiable")
    bg = ranked[0][0]
    seen = set()
    groups = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = y + dy, x + dx
                    if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and grid[q] == color:
                        seen.add(q)
                        stack.append(q)
            variants = []
            current = cells
            for _ in range(4):
                y0, x0 = min(y for y, x in current), min(x for y, x in current)
                variants.append(tuple(sorted((y - y0, x - x0) for y, x in current)))
                current = [(x, -y) for y, x in current]
            groups.append({"color": color, "cells": cells, "key": min(variants)})
    return bg, groups


def validate(example, task):
    """Recompute the complete edit from the final visible input, then compare."""
    grid = np.asarray(example["input"])
    out = np.asarray(example["output"])
    m = example["metadata"]
    bg, groups = _visible(grid)
    if bg != m["background"] or len(groups) != m["planned_object_count"]:
        raise ValueError("The visible field or object count differs from the complete plan")
    if Counter(len(g["cells"]) for g in groups) != Counter(m["object_sizes"]):
        raise ValueError("The visible object sizes differ from the complete plan")
    if len(np.unique(grid)) != m["planned_n_colors"]:
        raise ValueError("The visible palette differs from the complete plan")
    visible_records = []
    for group in groups:
        rr = [r for r, c in group["cells"]]
        cc = [c for r, c in group["cells"]]
        visible_records.append((group["color"], min(rr), min(cc), max(rr) - min(rr) + 1,
                                max(cc) - min(cc) + 1, len(rr)))
    planned_records = [(b["color"], *b["bounds"], b["cells"]) for b in m["object_records"]]
    if sorted(visible_records) != sorted(planned_records):
        raise ValueError("The visible body records differ from the complete plan")
    if task == "85":
        kind, value = m["filter_kind"], m["filter_value"]
        description = "color" if kind == "color" else f"{kind}{value}"
        if m["filter_description"] != description or (kind == "color" and m["filter_color"] != value):
            raise ValueError("The rendered filter disagrees with the validated filter")
    class_sizes = Counter(g["key"] for g in groups)
    expected = grid.copy()
    removed = 0
    for group in groups:
        if task == "84":
            remove = class_sizes[group["key"]] >= 2
        else:
            kind, value = m["filter_kind"], m["filter_value"]
            size = len(group["cells"])
            remove = group["color"] == value if kind == "color" else size == value if kind == "size=" else size < value if kind == "size<" else size > value
        if remove:
            removed += 1
            for r, c in group["cells"]:
                expected[r, c] = bg
    if out.shape != expected.shape or not np.array_equal(out, expected):
        raise ValueError("The planned erasure disagrees with the visible-grid rule")
    if removed != m["n_removed"]:
        raise ValueError("The visible removed-object count differs from the metadata")
    if task == "84" and sorted(class_sizes.values()) != m["class_sizes"]:
        raise ValueError("The visible matching classes differ from the plan")
    return True


def generate_erasure(rng, task):
    plan = _plan(rng, task)
    grid, owner, moved = _place(rng, plan)
    bodies = plan["bodies"]
    keys = [_mask_key(b["mask"]) for b in bodies]
    classes = Counter(keys)
    if task == "84":
        selected = [classes[key] >= 2 for key in keys]
        extra = {"class_sizes": sorted(classes.values())}
    else:
        kind = choose_filter_kind(rng)
        value, selected = _filter(rng, bodies, plan["palette"][0], kind)
        extra = {"filter_kind": kind, "filter_value": value,
                 "filter_description": "color" if kind == "color" else f"{kind}{value}"}
        if kind == "color":
            extra["filter_color"] = value
    out = grid.copy()
    for i, remove in enumerate(selected):
        if remove:
            out[owner == i] = plan["palette"][0]
    sizes = [int(b["mask"].sum()) for b in bodies]
    removed = sum(selected)
    actual_outcome = "none" if not removed else "all" if removed == len(bodies) else "some"
    if task == "84" and actual_outcome != plan["outcome_plan"]:
        raise ValueError("The planned matching outcome changed")
    colors, populations = np.unique(grid, return_counts=True)
    bg_count = int(np.sum(grid == plan["palette"][0]))
    margin = bg_count / max(int(n) for color, n in zip(colors, populations) if color != plan["palette"][0])
    m = {"grid_h": plan["h"], "grid_w": plan["w"], "background": plan["palette"][0],
         "background_style": "uniform", "background_margin": margin, "texture_named": False,
         "connectivity": 4, "n_objects": len(bodies), "planned_object_count": plan["count"],
         "object_count_cap": plan["count_cap"], "n_colors": len(colors),
         "planned_n_colors": len(plan["palette"]), "object_sizes": sizes,
         "object_kinds": [b["kind"] for b in bodies], "largest_object_cells": max(sizes),
         "foreground_occupancy": sum(sizes) / grid.size, "n_removed": removed,
         "n_kept": len(bodies) - removed, "removed_cells": sum(n for n, yes in zip(sizes, selected) if yes),
         "outcome": actual_outcome, "planned_density": plan["density"],
         "contact_plan": plan["contact_plan"], "mirror_negative_plan": plan["mirror_plan"],
         "placement_method": "free_moves_from_capacity_witness", "freely_moved_objects": moved,
         "attempts": 1, "object_records": [{"kind": b["kind"], "color": b["color"],
               "bounds": [b["r"], b["c"], *map(int, b["mask"].shape)], "cells": sizes[i]}
              for i, b in enumerate(bodies)], **extra}
    example = {"input": grid.tolist(), "output": out.tolist(), "metadata": m}
    validate(example, task)
    return example
