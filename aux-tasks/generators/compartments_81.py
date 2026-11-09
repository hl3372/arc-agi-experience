"""Readable floor plans for the largest / smallest compartment question.

A scene is drawn as an explicit list of rooms.  A room is a rectangle with one
to four rectangular blocks taken out of it (L, U, T, plus, stepped, alcove and
block-shaped irregular rooms), a staircase half of a box, the ring left around
a nested room, or a plain rectangle.  Two devices ride on top of that plan: a
nested room is a rectangular wall ring dropped inside a room, so the ring and
the room inside it are separate compartments, and a doorway is a one-cell-wide
passage cut through a partition, so two rooms read as one compartment and the
solver has to trace connectivity instead of counting boxes.

Every chamber is at least two cells wide in both directions on every canvas, so
a narrow grid reads as a floor plan rather than a barcode of one-cell slots;
only a nested room's wall ring is one cell thick, and that is a wall.  The
outer frame is equally deep (one to three cells) on all four sides.

The answer is taken from the plan with a union-find over room adjacency under
each rule; ``validate_compartments`` re-solves the finished picture by flooding
the visible grid under both rules and rejects any disagreement.  The complete,
coloured input is finished before the connectivity coin is drawn, and nothing
below that draw may edit it.
"""
from __future__ import annotations

from collections import deque

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
N8 = N4 + DIAGONALS

FAMILIES = ("rectangular_partition", "cutout_rooms", "corner_suite",
            "double_corner", "stepped_partition", "paired_rooms")
FAMILY_WEIGHTS = (.11, .09, .47, .29, .08, .07)

ROOM_KINDS = ("rectangle", "L", "stepped", "U", "T", "plus", "alcove",
              "irregular", "ring")
CARVED_KINDS = ("L", "stepped", "U", "T", "plus", "alcove", "irregular")
KIND_WEIGHTS = {"L": .18, "stepped": .13, "U": .14, "alcove": .14,
                "T": .13, "plus": .12, "irregular": .16}
CARVE_RATE = {"rectangular_partition": .68, "cutout_rooms": 1.0,
              "corner_suite": .74, "double_corner": .74,
              "stepped_partition": .82, "paired_rooms": .72}

MAX_ATTEMPTS = 40


class _Infeasible(ValueError):
    """A drawn plan that cannot be completed; the caller draws another one."""


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------


def _cells(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _sides(box):
    r0, c0, r1, c1 = box
    return r1 - r0, c1 - c0


def _bounds(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _connected(cells, steps):
    if not cells:
        return False
    first = min(cells)
    seen, queue = {first}, deque([first])
    while queue:
        r, c = queue.popleft()
        for dr, dc in steps:
            cell = (r + dr, c + dc)
            if cell in cells and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen) == len(cells)


def _shallow(side):
    """The shallowest bite a side may take: two cells, or one on a thin box."""
    return 2 if side >= 4 else 1


def _small(rng, low, high):
    """A value in ``[low, high]`` pulled towards the low end."""
    return int(min(rng.integers(low, high + 1), rng.integers(low, high + 1)))


def outline_corners(cells):
    """Corners of a region outline: 4 for a rectangle, 6 for an L, 12 for a plus."""
    if not cells:
        return 0
    r0, c0, r1, c1 = _bounds(cells)
    corners = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            quad = ((r - 1, c - 1) in cells, (r - 1, c) in cells,
                    (r, c - 1) in cells, (r, c) in cells)
            filled = sum(quad)
            if filled in (1, 3):
                corners += 1
            elif filled == 2 and quad[0] == quad[3]:
                corners += 2
    return corners


# ---------------------------------------------------------------------------
# room shapes
# ---------------------------------------------------------------------------


def _carve_blocks(rng, h, w, kind):
    """Blocks to remove from an ``h`` by ``w`` box, in box-local coordinates.

    Every block reaches the box border, so the wall around the room stays one
    piece, and every surviving arm is at least two cells wide: the rooms are
    unions of a few rectangles rather than carved noise.
    """
    if kind == "L":
        if h < 3 or w < 3 or max(h, w) < 4:
            return None
        return [(0, 0, _small(rng, _shallow(h), h - 2), _small(rng, _shallow(w), w - 2))], "L"
    if kind == "stepped":
        if h < 5 or w < 5:
            return None
        deep = int(rng.integers(3, h - 1))
        shallow = int(rng.integers(2, deep))
        narrow = int(rng.integers(2, w - 2))
        broad = int(rng.integers(narrow + 1, w - 1))
        return [(0, 0, deep, narrow), (0, 0, shallow, broad)], "stepped"
    if kind in ("U", "alcove"):
        if h < 3 or w < 6:
            return None
        width = _small(rng, 2, w - 4)
        start = int(rng.integers(2, w - 1 - width))
        half, floor = (h + 1) // 2, _shallow(h)
        low, high = (half, h - 2) if kind == "U" else (floor, max(floor, half - 1))
        low, high = max(floor, min(low, h - 2)), max(floor, min(high, h - 2))
        depth = int(rng.integers(min(low, high), high + 1))
        return [(0, start, depth, start + width)], ("U" if 2 * depth >= h else "alcove")
    if kind == "T":
        if h < 3 or w < 6:
            return None
        left = _small(rng, 2, w - 4)
        right = _small(rng, 2, w - 2 - left)
        depth = int(rng.integers(_shallow(h), h - 1))
        return [(0, 0, depth, left), (0, w - right, depth, w)], "T"
    if kind == "plus":
        if h < 6 or w < 6:
            return None
        top = _small(rng, 2, h - 4)
        bar = int(rng.integers(2, h - top - 1))
        left = _small(rng, 2, w - 4)
        stem = int(rng.integers(2, w - left - 1))
        return [(0, 0, top, left), (0, left + stem, top, w),
                (top + bar, 0, h, left), (top + bar, left + stem, h, w)], "plus"
    if kind == "irregular":
        if h < 5 or w < 5:
            return None
        picks = int(rng.integers(2, 4))
        blocks = []
        for index in rng.permutation(4)[:picks]:
            down, across = divmod(int(index), 2)
            tall, wide = _small(rng, 2, h - 3), _small(rng, 2, w - 3)
            top = h - tall if down else 0
            left = w - wide if across else 0
            blocks.append((top, left, top + tall, left + wide))
        return blocks, "irregular"
    raise ValueError(f"Task 81 has no room shape named {kind!r}")


def _carve(rng, box, kind):
    """Place one room shape inside ``box`` at a random orientation."""
    r0, c0, r1, c1 = box
    h, w = r1 - r0, c1 - c0
    transpose = bool(rng.integers(2))
    made = _carve_blocks(rng, w if transpose else h, h if transpose else w, kind)
    if made is None:
        return None
    blocks, resolved = made
    flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
    removed = set()
    for a0, b0, a1, b1 in blocks:
        for a in range(a0, a1):
            for b in range(b0, b1):
                r, c = (b, a) if transpose else (a, b)
                removed.add((r0 + (h - 1 - r if flip_r else r),
                             c0 + (w - 1 - c if flip_c else c)))
    room = _cells(box) - removed
    if not room or not _connected(room, N4):
        return None
    return room, resolved


def _carve_room(rng, box, protected, rate, force=False):
    """A carved room when the box allows one, otherwise the plain rectangle."""
    if not force and rng.random() >= rate:
        return _cells(box), "rectangle"
    weights = np.array([KIND_WEIGHTS[kind] for kind in CARVED_KINDS])
    order = rng.choice(len(CARVED_KINDS), size=len(CARVED_KINDS),
                       replace=False, p=weights / weights.sum())
    for index in order:
        for _ in range(3):
            made = _carve(rng, box, CARVED_KINDS[int(index)])
            if made is not None and set(protected) <= made[0]:
                return made
    return _cells(box), "rectangle"


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------


def _cut_options(box, minimum, thickness):
    """Straight cuts that leave two chambers of at least two cells each."""
    r0, c0, r1, c1 = box
    out = []
    for row in range(r0 + minimum, r1 - thickness - minimum + 1):
        out.append(((r0, c0, row, c1), (row + thickness, c0, r1, c1)))
    for col in range(c0 + minimum, c1 - thickness - minimum + 1):
        out.append(((r0, c0, r1, col), (r0, col + thickness, r1, c1)))
    return [cut for cut in out if all(np.prod(_sides(part)) >= 2 for part in cut)]


def _guillotine(rng, boxes, target, minimum, max_thickness, distinct=False):
    """Straight cuts of independently drawn thickness until ``target`` boxes.

    With ``distinct`` the cutter prefers a chamber size it has not used yet.
    On a strip-shaped interior every chamber shares the same height, so plain
    cutting piles several chambers on the same small area and the smallest one
    is a tie far more often than on a roomy canvas.
    """
    boxes = list(boxes)
    thicknesses = []
    while len(boxes) < target:
        options = []
        for index, box in enumerate(boxes):
            for thickness in range(1, max_thickness + 1):
                cuts = _cut_options(box, minimum, thickness)
                if cuts:
                    options.append((index, thickness, cuts))
        if not options:
            break
        indices = sorted({option[0] for option in options})
        weight = np.array([float(np.prod(_sides(boxes[i]))) for i in indices])
        if rng.random() < .45:
            weight = np.ones(len(indices))
        index = indices[int(rng.choice(len(indices), p=weight / weight.sum()))]
        local = [option for option in options if option[0] == index]
        prefer = np.array([(.62, .27, .11)[option[1] - 1] for option in local])
        _, thickness, cuts = local[int(rng.choice(len(local), p=prefer / prefer.sum()))]
        roomy = [cut for cut in cuts if all(min(_sides(part)) >= 4 for part in cut)]
        pool = roomy if roomy and rng.random() < .80 else cuts
        if distinct:
            taken = {int(np.prod(_sides(box))) for i, box in enumerate(boxes) if i != index}
            fresh = [cut for cut in pool
                     if len({int(np.prod(_sides(part))) for part in cut} | taken)
                     == len(taken) + 2]
            if fresh and rng.random() < .85:
                pool = fresh
        boxes[index:index + 1] = list(pool[int(rng.integers(len(pool)))])
        thicknesses.append(thickness)
    return boxes, thicknesses


def _corner_split(rng, box, minimum):
    """Four chambers where two of them meet at a single shared corner.

    The two diagonal chambers have no wall between their touching corners, so
    the connectivity clause alone decides whether they are one compartment.
    """
    r0, c0, r1, c1 = box
    if min(r1 - r0, c1 - c0) < 2 * minimum + 2:
        return None
    for _ in range(10):
        row = int(rng.integers(r0 + minimum + 1, r1 - minimum))
        col = int(rng.integers(c0 + minimum + 1, c1 - minimum))
        boxes = [(r0, c0, row, col), (row, col, r1, c1),
                 (r0, col + 1, row - 1, c1), (row + 1, c0, r1, col - 1)]
        if all(np.prod(_sides(part)) >= 2 for part in boxes):
            return boxes, {0: [(row - 1, col - 1)], 1: [(row, col)]}
    return None


def _staircase(rng, box, minimum):
    """Halve a box with a two-cell-thick staircase wall of one to three steps."""
    r0, c0, r1, c1 = box
    thickness = 2
    for transpose in sorted((False, True), key=lambda _: rng.random()):
        h, w = (c1 - c0, r1 - r0) if transpose else (r1 - r0, c1 - c0)
        drift = w - thickness - 2 * minimum
        if drift < 1 or h < 6:
            continue
        steps = int(rng.integers(1, min(3, drift, h // 2 - 1) + 1))
        rows = None
        for _ in range(12):
            picked = sorted(int(x) for x in
                            rng.choice(np.arange(2, h - 1), size=steps, replace=False))
            if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
                rows = picked
                break
        if rows is None:
            continue
        if rng.random() < .5:
            direction, start = 1, int(rng.integers(minimum, w - thickness - minimum - steps + 1))
        else:
            direction, start = -1, int(rng.integers(minimum + steps, w - thickness - minimum + 1))
        near, far = set(), set()
        column, seen = start, 0
        for r in range(h):
            if seen < steps and r == rows[seen]:
                column += direction
                seen += 1
            for c in range(w):
                if c < column:
                    near.add((r, c))
                elif c >= column + thickness:
                    far.add((r, c))

        def place(cells):
            return {(r0 + (c if transpose else r), c0 + (r if transpose else c))
                    for r, c in cells}

        return place(near), place(far), thickness
    return None


# ---------------------------------------------------------------------------
# the drawn plan
# ---------------------------------------------------------------------------


def _plan(rng):
    h, w = (int(x) for x in rng.integers(5, 31, size=2))
    short = min(h, w)
    if short >= 16:
        frame = int(rng.choice((1, 2, 3), p=(.56, .29, .15)))
    elif short >= 11:
        frame = int(rng.choice((1, 2), p=(.70, .30)))
    else:
        frame = 1
    ih, iw = h - 2 * frame, w - 2 * frame
    inner = min(ih, iw)
    # Two is the floor everywhere: a chamber is at least two cells wide in both
    # directions.  A floor of one used to be allowed on the narrow canvases
    # (interior five or less), where it turned the plan into a row of one-cell
    # slots -- a barcode rather than a floor plan.
    if inner <= 8:
        minimum = 2
    elif inner <= 12:
        minimum = int(rng.choice((2, 3), p=(.30, .70)))
    else:
        minimum = int(rng.choice((3, 4, 5), p=(.36, .44, .20)))
    pitch = minimum + 1
    cap = max(2, min(24, (ih * iw) // (2 * pitch * pitch) + 1))
    choices = np.arange(2, cap + 1)
    weights = 1.0 / np.sqrt(choices - 1)
    plan = {"h": h, "w": w, "frame": frame, "minimum": minimum,
            "max_thickness": 1 if inner < 7 else 2 if inner < 13 else 3,
            "target": int(rng.choice(choices, p=weights / weights.sum())),
            "cap": cap, "interior": (frame, frame, h - frame, w - frame),
            "strip": inner <= 2 * minimum + 1}
    available = np.array([FAMILY_WEIGHTS[i] if _family_fits(name, plan) else 0.0
                          for i, name in enumerate(FAMILIES)])
    if not available.sum():
        raise _Infeasible("Task 81 has no layout family for this canvas")
    plan["family"] = str(rng.choice(FAMILIES, p=available / available.sum()))
    if plan["family"] == "paired_rooms":
        plan["target"] = 3
    plan["mirror_rows"] = bool(rng.integers(2))
    plan["mirror_cols"] = bool(rng.integers(2))
    plan["palette"] = [int(x) for x in rng.choice(10, 4, replace=False)]
    return plan


def _family_fits(family, plan):
    ih, iw = plan["h"] - 2 * plan["frame"], plan["w"] - 2 * plan["frame"]
    minimum = plan["minimum"]
    if family in ("rectangular_partition", "cutout_rooms"):
        return plan["cap"] >= 2
    if family == "corner_suite":
        return min(ih, iw) >= 2 * minimum + 2
    if family == "double_corner":
        return min(ih, iw) >= 3 * minimum + 4
    if family == "stepped_partition":
        return min(ih, iw) >= 2 * minimum + 3 and max(ih, iw) >= 6
    if family == "paired_rooms":
        return min(ih, iw) >= 2 * minimum + 2 and min(plan["h"], plan["w"]) >= 7
    raise ValueError(f"Task 81 has no layout family named {family!r}")


def _paired_boxes(rng, plan):
    """A larger chamber beside two equally sized, separated chambers."""
    h, w, f = plan["h"], plan["w"], plan["frame"]
    transpose = bool(rng.integers(2))
    if transpose:
        h, w = w, h
    ih, iw = h - 2 * f, w - 2 * f
    twin = (ih - 1) // 2
    # Both the single chamber and the twins keep the plan's two-cell floor.
    floor = 2
    widths = [x for x in range(floor, iw - 1 - floor + 1)
              if ih * x > twin * (iw - 1 - x) and .25 <= x / iw <= .7
              and ih * x >= 2 and twin * (iw - 1 - x) >= 2]
    if not widths or twin < floor:
        raise _Infeasible("Task 81 paired-room plan is infeasible")
    left = widths[int(rng.integers(len(widths)))]
    boxes = [(f, f, h - f, f + left), (f, f + left + 1, f + twin, w - f),
             (h - f - twin, f + left + 1, h - f, w - f)]
    if transpose:
        boxes = [(c0, r0, c1, r1) for r0, c0, r1, c1 in boxes]
    return boxes, [1]


def _double_corner_boxes(rng, plan):
    """A corner suite whose side chamber carries a second corner join."""
    minimum = plan["minimum"]
    for _ in range(8):
        split = _corner_split(rng, plan["interior"], minimum)
        if split is None:
            break
        boxes, protected = split
        for index in sorted((2, 3), key=lambda _: rng.random()):
            inner = _corner_split(rng, boxes[index], minimum)
            if inner is None:
                continue
            more, extra = inner
            grown = boxes[:index] + more + boxes[index + 1:]
            keep = dict(protected)
            for key, cells in extra.items():
                keep[index + key] = cells
            return grown, keep, [1]
    raise _Infeasible("Task 81 could not fit a second corner join")


def _grow(rng, boxes, protected, target, minimum, max_thickness):
    """Subdivide further, leaving the chambers that carry a corner join whole."""
    thicknesses = []
    slots = [[box] for box in boxes]
    while sum(map(len, slots)) < target:
        plain = [i for i in range(len(slots)) if not protected.get(i)]
        marked = [i for i in range(len(slots)) if protected.get(i)]
        order = ([plain[int(i)] for i in rng.permutation(len(plain))]
                 + [marked[int(i)] for i in rng.permutation(len(marked))])
        grew = False
        for index in order:
            group, used = _guillotine(rng, slots[index], len(slots[index]) + 1,
                                      minimum, max_thickness)
            if len(group) > len(slots[index]):
                slots[index] = group
                thicknesses.extend(used)
                grew = True
                break
        if not grew:
            break
    out, keep = [], {}
    for index, group in enumerate(slots):
        for box in group:
            r0, c0, r1, c1 = box
            keep[len(out)] = [cell for cell in protected.get(index, ())
                              if r0 <= cell[0] < r1 and c0 <= cell[1] < c1]
            out.append(box)
    return out, keep, thicknesses


def _layout(rng, plan):
    """Base boxes, finished rooms, protected corners and partition thicknesses."""
    family = plan["family"]
    minimum, max_thickness = plan["minimum"], plan["max_thickness"]
    if family in ("rectangular_partition", "cutout_rooms"):
        boxes, thicknesses = _guillotine(rng, [plan["interior"]], plan["target"],
                                         minimum, max_thickness, plan["strip"])
        return boxes, [], {}, thicknesses
    if family == "paired_rooms":
        boxes, thicknesses = _paired_boxes(rng, plan)
        return boxes, [], {}, thicknesses
    if family == "stepped_partition":
        boxes, thicknesses = _guillotine(rng, [plan["interior"]],
                                         max(1, plan["target"] - 1), minimum, max_thickness)
        for index in sorted(range(len(boxes)), key=lambda i: -np.prod(_sides(boxes[i]))):
            split = _staircase(rng, boxes[index], minimum)
            if split is not None:
                near, far, thickness = split
                rest = boxes[:index] + boxes[index + 1:]
                return rest, [(near, "stepped"), (far, "stepped")], {}, thicknesses + [thickness]
        raise _Infeasible("Task 81 could not fit a staircase wall")
    if family == "corner_suite":
        split = _corner_split(rng, plan["interior"], minimum)
        if split is None:
            raise _Infeasible("Task 81 could not fit a corner join")
        base, protected = split
        thicknesses = [1]
    else:
        base, protected, thicknesses = _double_corner_boxes(rng, plan)
    boxes, keep, used = _grow(rng, base, protected, plan["target"], minimum, max_thickness)
    return boxes, [], keep, thicknesses + used


# ---------------------------------------------------------------------------
# nested rooms and doorways
# ---------------------------------------------------------------------------


def _nest(rng, room):
    """A rectangular wall ring with a room inside it, both inside ``room``."""
    top0, left0, bottom, right = _bounds(room)
    top0, left0, bottom, right = top0 + 1, left0 + 1, bottom - 1, right - 1
    for _ in range(12):
        thickness = 1 if rng.random() < .75 else 2
        span = 2 * thickness + 2
        if bottom - top0 < span or right - left0 < span:
            return None
        height = int(rng.integers(span, bottom - top0 + 1))
        width = int(rng.integers(span, right - left0 + 1))
        top = int(rng.integers(top0, bottom - height + 1))
        left = int(rng.integers(left0, right - width + 1))
        if not _cells((top - 1, left - 1, top + height + 1, left + width + 1)) <= room:
            continue
        inside = _cells((top + thickness, left + thickness,
                         top + height - thickness, left + width - thickness))
        ring = _cells((top, left, top + height, left + width)) - inside
        rest = room - _cells((top, left, top + height, left + width))
        if rest and _connected(rest, N4) and _connected(inside, N4):
            return rest, inside, ring
    return None


def _gates(rooms, height, width):
    """One-cell-wide passages that would merge two rooms into one compartment."""
    owner = {}
    for index, cells in enumerate(rooms):
        for cell in cells:
            owner[cell] = index
    found = []
    for index, cells in enumerate(rooms):
        for r, c in cells:
            for dr, dc in N4:
                run = []
                for step in range(1, 4):
                    cell = (r + dr * step, c + dc * step)
                    if not (0 <= cell[0] < height and 0 <= cell[1] < width):
                        run = []
                        break
                    if cell in owner:
                        break
                    run.append(cell)
                else:
                    run = []
                if not run:
                    continue
                other = owner.get((r + dr * (len(run) + 1), c + dc * (len(run) + 1)))
                if other is None or other <= index:
                    continue
                if all((cell[0] + dc, cell[1] + dr) not in owner
                       and (cell[0] - dc, cell[1] - dr) not in owner for cell in run):
                    found.append((index, other, tuple(run)))
    return found


# ---------------------------------------------------------------------------
# scene assembly
# ---------------------------------------------------------------------------


def _draw_scene(rng, plan):
    h, w, family = plan["h"], plan["w"], plan["family"]
    boxes, prerooms, protected, thicknesses = _layout(rng, plan)
    rooms = [cells for cells, _ in prerooms]
    kinds = [kind for _, kind in prerooms]
    forced = -1
    if family == "cutout_rooms":
        roomy = [i for i, box in enumerate(boxes) if min(_sides(box)) >= 4]
        if not roomy:
            raise _Infeasible("Task 81 cut-out layout has no carvable chamber")
        forced = int(rng.choice(roomy))
    for index, box in enumerate(boxes):
        cells, kind = _carve_room(rng, box, protected.get(index, ()),
                                  CARVE_RATE[family], force=index == forced)
        rooms.append(cells)
        kinds.append(kind)
    rings, nested = set(), 0
    hosts = [i for i in range(len(rooms)) if min(_sides(_bounds(rooms[i]))) >= 6]
    if len(rooms) >= 2 and hosts and rng.random() < .85:
        rng.shuffle(hosts)
        for host in hosts[:1 + int(rng.random() < .18)]:
            made = _nest(rng, rooms[host])
            if made is None:
                continue
            rest, inside, ring = made
            rooms[host], kinds[host] = rest, "ring"
            rooms.append(inside)
            kinds.append("rectangle")
            rings |= ring
            nested += 1
    doorways = []
    if len(rooms) >= 3 and rng.random() < .30:
        options = [gate for gate in _gates(rooms, h, w) if not rings & set(gate[2])]
        rng.shuffle(options)
        wanted = 1 + int(rng.random() < .35) + int(rng.random() < .12)
        merged = list(range(len(rooms)))

        def root(x):
            while merged[x] != x:
                merged[x] = merged[merged[x]]
                x = merged[x]
            return x

        for near, far, run in options:
            if len(doorways) >= wanted:
                break
            if root(near) == root(far):
                continue
            merged[root(near)] = root(far)
            rooms[near] = rooms[near] | set(run)
            doorways.append(run)
    if plan["mirror_rows"]:
        rooms = [{(h - 1 - r, c) for r, c in cells} for cells in rooms]
        doorways = [tuple((h - 1 - r, c) for r, c in run) for run in doorways]
    if plan["mirror_cols"]:
        rooms = [{(r, w - 1 - c) for r, c in cells} for cells in rooms]
        doorways = [tuple((r, w - 1 - c) for r, c in run) for run in doorways]
    background, wall = plan["palette"][0], plan["palette"][1]
    grid = np.full((h, w), wall, dtype=np.int64)
    for cells in rooms:
        for r, c in cells:
            grid[r, c] = background
    if int(np.count_nonzero(grid != wall)) != sum(map(len, rooms)):
        raise _Infeasible("Task 81 rooms overlap")
    door_cells = {cell for run in doorways for cell in run}
    if _wall_fault(grid, wall, door_cells, nested):
        raise _Infeasible("Task 81 wall structure is not readable")
    if set(_side_depths(grid, wall)) != {plan["frame"]}:
        raise _Infeasible("Task 81 outer frame is deeper on one side than another")
    counts = {rule: len(_visible_regions(grid, rule)[1]) for rule in (4, 8)}
    for run in doorways:
        closed = grid.copy()
        for r, c in run:
            closed[r, c] = wall
        if any(len(_visible_regions(closed, rule)[1]) != counts[rule] + 1 for rule in (4, 8)):
            raise _Infeasible("Task 81 doorway does not split one compartment under both rules")
    info = {"thicknesses": sorted({int(t) for t in thicknesses}), "nested": nested,
            "doorways": [sorted(run) for run in doorways]}
    return grid, rooms, kinds, info


def _plan_regions(rooms, connectivity):
    """Compartments of the plan: rooms joined by adjacency under one rule."""
    steps = N8 if connectivity == 8 else N4
    owner = {}
    for index, cells in enumerate(rooms):
        for cell in cells:
            owner[cell] = index
    parent = list(range(len(rooms)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for index, cells in enumerate(rooms):
        for r, c in cells:
            for dr, dc in steps:
                other = owner.get((r + dr, c + dc))
                if other is not None and root(other) != root(index):
                    parent[root(index)] = root(other)
    groups = {}
    for index, cells in enumerate(rooms):
        groups.setdefault(root(index), set()).update(cells)
    return list(groups.values())


def _targets(regions):
    largest = min(regions, key=lambda cells: (-len(cells), min(cells)))
    smallest = min(regions, key=lambda cells: (len(cells), min(cells)))
    if len(largest) == len(smallest):
        raise _Infeasible("Task 81 requires different largest and smallest compartments")
    return largest, smallest


def _corner_contacts(rooms):
    """Room pairs that touch only at a corner, so the rule decides their fate."""
    owner = {}
    for index, cells in enumerate(rooms):
        for cell in cells:
            owner[cell] = index
    edge, diagonal = set(), set()
    for index, cells in enumerate(rooms):
        for r, c in cells:
            for dr, dc in N4:
                other = owner.get((r + dr, c + dc))
                if other is not None and other != index:
                    edge.add((min(index, other), max(index, other)))
            for dr, dc in DIAGONALS:
                other = owner.get((r + dr, c + dc))
                if other is not None and other != index:
                    diagonal.add((min(index, other), max(index, other)))
    return len(diagonal - edge)


# ---------------------------------------------------------------------------
# independent reading of the finished picture
# ---------------------------------------------------------------------------


def _visible_regions(grid, connectivity):
    """Literal independent reader: infer walls from the complete outer frame."""
    if connectivity not in (4, 8):
        raise ValueError("Task 81 connectivity must be four or eight")
    colors = set(grid.ravel().tolist())
    border = set(grid[0].tolist() + grid[-1].tolist() + grid[:, 0].tolist() + grid[:, -1].tolist())
    if len(colors) != 2 or len(border) != 1:
        raise ValueError("Task 81 requires two colors and a uniform wall frame")
    wall = next(iter(border))
    steps = N8 if connectivity == 8 else N4
    remaining = {tuple(map(int, rc)) for rc in np.argwhere(grid != wall)}
    regions = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = deque([first])
        region = {first}
        while queue:
            r, c = queue.popleft()
            for dr, dc in steps:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    region.add(neighbor)
        regions.append(region)
    return wall, regions


def _side_depths(grid, wall):
    """How deep the solid wall runs in from each of the four sides.

    A uniform outer frame reads as a frame only when the four depths agree; a
    room whose carved-out face lies along one edge would otherwise pile extra
    wall rows on that side alone.
    """
    out = []
    for view in (grid, grid[::-1], grid.T, grid.T[::-1]):
        depth = 0
        while depth < view.shape[0] and bool(np.all(view[depth] == wall)):
            depth += 1
        out.append(depth)
    return out


def _frame_depth(grid, wall):
    h, w = grid.shape
    depth = 0
    while 2 * depth < min(h, w):
        k = depth
        ring = np.concatenate([grid[k, k:w - k], grid[h - 1 - k, k:w - k],
                               grid[k:h - k, k], grid[k:h - k, w - 1 - k]])
        if not np.all(ring == wall):
            break
        depth += 1
    return depth


def _wall_pieces(grid, wall):
    remaining = {tuple(map(int, rc)) for rc in np.argwhere(grid == wall)}
    pieces = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = deque([first])
        piece = {first}
        while queue:
            r, c = queue.popleft()
            for dr, dc in N4:
                cell = (r + dr, c + dc)
                if cell in remaining:
                    remaining.remove(cell)
                    queue.append(cell)
                    piece.add(cell)
        pieces.append(piece)
    return pieces


def _ring_thickness(piece):
    """The uniform thickness of a rectangular ring, or zero if it is not one."""
    r0, c0, r1, c1 = _bounds(piece)
    for thickness in (1, 2, 3):
        if r1 - r0 <= 2 * thickness or c1 - c0 <= 2 * thickness:
            break
        inside = _cells((r0 + thickness, c0 + thickness, r1 - thickness, c1 - thickness))
        if piece == _cells((r0, c0, r1, c1)) - inside:
            return thickness
    return 0


def _wall_fault(grid, wall, door_cells, nested):
    """Why the walls would not read as walls, or ``None`` when they do.

    Every wall belongs to the frame structure or is a closed rectangular ring
    around a nested room, and a wall only ends where a doorway opens it.
    """
    pieces = _wall_pieces(grid, wall)
    outer = [piece for piece in pieces if (0, 0) in piece]
    if len(outer) != 1:
        return "the outer wall structure is broken"
    rings = [piece for piece in pieces if piece is not outer[0]]
    if len(rings) != nested or any(not _ring_thickness(piece) for piece in rings):
        return "a free-standing wall is not a closed rectangular ring"
    cells = set().union(*pieces)
    for r, c in cells:
        if sum((r + dr, c + dc) in cells for dr, dc in N4) >= 2:
            continue
        if not any((r + dr, c + dc) in door_cells for dr, dc in N4):
            return "a wall ends in a loose stub"
    return None


def validate_compartments(example):
    """Re-solve both rules from visible cells; reject any incorrect output."""
    grid = np.asarray(example["input"])
    if grid.ndim != 2 or not all(5 <= n <= 30 for n in grid.shape):
        raise ValueError("Task 81 requires grid sides from five to thirty")
    if not np.issubdtype(grid.dtype, np.integer) or grid.min() < 0 or grid.max() > 9:
        raise ValueError("Task 81 requires integer colors from zero to nine")
    meta = example["metadata"]
    if (meta["grid_h"], meta["grid_w"]) != grid.shape or meta["n_colors"] != 2:
        raise ValueError("Task 81 canvas metadata disagrees with the visible grid")
    if meta["planned_family"] != meta["layout_family"] or meta["layout_family"] not in FAMILIES:
        raise ValueError("Task 81 did not preserve its layout plan")
    if meta["connectivity"] not in (4, 8):
        raise ValueError("Task 81 connectivity must be four or eight")
    paints = [meta["color_large"], meta["color_small"]]
    if len(set(paints)) != 2 or any(c in grid for c in paints) or not all(0 <= c <= 9 for c in paints):
        raise ValueError("Task 81 paint colors must be distinct new colors")
    if set(meta["room_kinds"]) - set(ROOM_KINDS) or len(meta["room_kinds"]) != meta["n_rooms"]:
        raise ValueError("Task 81 chamber metadata disagrees with its plan")
    if (sorted(set(meta["wall_thicknesses"])) != list(meta["wall_thicknesses"])
            or not all(1 <= t <= 3 for t in meta["wall_thicknesses"])
            or meta["varying_wall_thickness"] != (len(meta["wall_thicknesses"]) > 1)):
        raise ValueError("Task 81 wall-thickness metadata is malformed")
    interpretations = {}
    for rule in (4, 8):
        wall, regions = _visible_regions(grid, rule)
        if meta["wall_color"] != wall or set(grid.ravel()) != {wall, meta["background_color"]}:
            raise ValueError("Task 81 color metadata disagrees with the visible grid")
        if len(regions) < 2 or len({len(region) for region in regions}) < 2:
            raise ValueError("Task 81 needs distinct extrema under both connectivity rules")
        # Explicit scan order rather than the construction's target helper.
        largest = smallest = regions[0]
        for region in regions[1:]:
            if len(region) > len(largest):
                largest = region
            if len(region) < len(smallest):
                smallest = region
        expected = grid.copy()
        for cells, color in ((largest, paints[0]), (smallest, paints[1])):
            for r, c in cells:
                expected[r, c] = color
        interpretations[rule] = expected
        if meta[f"room_count_{rule}"] != len(regions):
            raise ValueError("Task 81 room-count metadata disagrees with visible cells")
        actual_sizes = sorted(len(region) for region in regions)
        if meta[f"areas_{rule}"] != actual_sizes:
            raise ValueError("Task 81 area metadata disagrees with visible cells")
        if meta[f"ties_large_{rule}"] != (actual_sizes.count(actual_sizes[-1]) > 1):
            raise ValueError("Task 81 largest-tie metadata disagrees with visible cells")
        if meta[f"ties_small_{rule}"] != (actual_sizes.count(actual_sizes[0]) > 1):
            raise ValueError("Task 81 smallest-tie metadata disagrees with visible cells")
        if rule == meta["connectivity"]:
            if (meta["room_count"], meta["smallest_area"], meta["largest_area"]) != (len(regions), actual_sizes[0], actual_sizes[-1]):
                raise ValueError("Task 81 selected-room metadata disagrees with visible cells")
            if meta["ties_large"] != meta[f"ties_large_{rule}"] or meta["ties_small"] != meta[f"ties_small_{rule}"]:
                raise ValueError("Task 81 selected-tie metadata disagrees with visible cells")
            if (meta["largest_is_first"], meta["smallest_is_first"]) != (
                    largest is regions[0], smallest is regions[0]):
                raise ValueError("Task 81 extremum-position metadata disagrees with visible cells")
    if not np.array_equal(example["output"], interpretations[meta["connectivity"]]):
        raise ValueError("Task 81 output disagrees with the visible compartments")
    if meta["connectivity_matters"] != (not np.array_equal(interpretations[4], interpretations[8])):
        raise ValueError("Task 81 connectivity metadata disagrees with visible cells")
    if meta["frame_thickness"] != _frame_depth(grid, meta["wall_color"]):
        raise ValueError("Task 81 frame-thickness metadata disagrees with visible cells")
    if (set(_side_depths(grid, meta["wall_color"])) != {meta["frame_thickness"]}
            or not 1 <= meta["frame_thickness"] <= 3):
        raise ValueError("Task 81 outer frame must be one to three cells deep on all four sides")
    doorways = [tuple(map(tuple, run)) for run in meta["doorway_cells"]]
    door_cells = {cell for run in doorways for cell in run}
    if len(doorways) != meta["n_doorways"] or len(door_cells) != sum(map(len, doorways)):
        raise ValueError("Task 81 doorway metadata disagrees with its plan")
    fault = _wall_fault(grid, meta["wall_color"], door_cells, meta["n_nested"])
    if fault:
        raise ValueError(f"Task 81 walls must read as walls: {fault}")
    # A doorway is a gap through a wall, and closing it splits one compartment
    # under either reading of the connectivity clause.
    height, width = grid.shape
    base = {rule: len(_visible_regions(grid, rule)[1]) for rule in (4, 8)}
    for run in doorways:
        for r, c in run:
            if grid[r, c] == meta["wall_color"]:
                raise ValueError("Task 81 doorway cells must be open")
            across = sum(grid[r, c + dc] == meta["wall_color"]
                         for dc in (-1, 1) if 0 <= c + dc < width) == 2
            down = sum(grid[r + dr, c] == meta["wall_color"]
                       for dr in (-1, 1) if 0 <= r + dr < height) == 2
            if not (across or down):
                raise ValueError("Task 81 doorway cells must lie inside a wall")
        closed = grid.copy()
        for r, c in run:
            closed[r, c] = meta["wall_color"]
        for rule in (4, 8):
            if len(_visible_regions(closed, rule)[1]) != base[rule] + 1:
                raise ValueError("Task 81 doorways must merge exactly two compartments")
    if meta["planned_room_count"] != meta["room_count_4"]:
        raise ValueError("Task 81 did not preserve its planned chamber count")
    if meta["compartment_cells"] != int(np.count_nonzero(grid != meta["wall_color"])):
        raise ValueError("Task 81 chamber metadata disagrees with visible cells")


def generate_color_compartments(rng):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            plan = _plan(rng)
            grid, rooms, kinds, info = _draw_scene(rng, plan)
            regions = {rule: _plan_regions(rooms, rule) for rule in (4, 8)}
            for rule in (4, 8):
                if len(regions[rule]) < 2:
                    raise _Infeasible("Task 81 needs at least two compartments")
                _targets(regions[rule])
        except _Infeasible:
            continue
        break
    else:
        raise ValueError("Task 81 could not draw a readable floor plan")
    background, wall, large_color, small_color = plan["palette"]
    outputs = {}
    meta = {"grid_h": plan["h"], "grid_w": plan["w"],
            "background_color": background, "wall_color": wall,
            "color_large": large_color, "color_small": small_color,
            "layout_family": plan["family"], "planned_family": plan["family"],
            "frame_thickness": _frame_depth(grid, wall),
            "wall_thicknesses": info["thicknesses"],
            "varying_wall_thickness": len(info["thicknesses"]) > 1,
            "n_rooms": len(rooms), "room_kinds": kinds,
            "n_nested": info["nested"], "n_doorways": len(info["doorways"]),
            "doorway_cells": info["doorways"],
            "corner_contacts": _corner_contacts(rooms),
            "planned_room_count": len(regions[4]), "attempts": attempt,
            "compartment_cells": sum(map(len, rooms)), "n_colors": 2}
    for rule in (4, 8):
        large, small = _targets(regions[rule])
        output = grid.copy()
        for cells, color in ((large, large_color), (small, small_color)):
            for r, c in cells:
                output[r, c] = color
        outputs[rule] = output
        areas = sorted(map(len, regions[rule]))
        meta.update({f"room_count_{rule}": len(regions[rule]), f"areas_{rule}": areas,
                     f"ties_large_{rule}": areas.count(areas[-1]) > 1,
                     f"ties_small_{rule}": areas.count(areas[0]) > 1})
    # No input edits or rejection based on this choice are permitted below.
    connectivity = int(rng.choice((4, 8)))
    chosen = regions[connectivity]
    large, small = _targets(chosen)
    first = min(chosen, key=min)
    meta.update({"connectivity": connectivity,
                 "connectivity_matters": not np.array_equal(outputs[4], outputs[8]),
                 "room_count": meta[f"room_count_{connectivity}"],
                 "largest_area": meta[f"areas_{connectivity}"][-1],
                 "smallest_area": meta[f"areas_{connectivity}"][0],
                 "ties_large": meta[f"ties_large_{connectivity}"],
                 "ties_small": meta[f"ties_small_{connectivity}"],
                 "largest_is_first": large is first,
                 "smallest_is_first": small is first})
    example = {"input": grid.tolist(), "output": outputs[connectivity].tolist(),
               "metadata": meta}
    validate_compartments(example)
    return example
