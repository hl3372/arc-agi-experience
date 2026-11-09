"""Readable enclosures and repairs determined by visible wall ends (tasks 36/37).

Geometry is deliberately limited to familiar polygons, rounded outlines and rooms.
No random walks, cellular silhouettes, bumps, hairs or noise-carved holes.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from common import topology as T
from generators.scene_shapes_56_60 import halo, placement_map

D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
FAMILIES = ("rectangle", "chamfer", "ellipse", "diamond", "L", "T", "plus", "step", "rooms")
MIN_SIDE = 8
FILLS = ("accents", "bands", "diagonal_bands", "repeated_motifs",
         "large_patches", "sparse_pattern", "local_patches")
FILL_WEIGHTS = (.30, .18, .10, .12, .10, .10, .10)
FAMILY_WEIGHTS = dict(zip(FAMILIES, (.23, .16, .12, .09, .10, .06, .05, .07, .12)))
PROMPT37 = (
    "The wall consists of every cell of color {color}; every other cell is empty, whatever its color. "
    "Repair every straight break in the wall. A broken end has exactly one wall neighbor among its "
    "eight surrounding cells. Continue away from that neighbor in the same horizontal, vertical or "
    "45-degree diagonal direction: the first wall cell reached is its matching broken end, whose "
    "only wall neighbor continues in the same line away from the gap. Every broken end has exactly "
    "one such partner, and the empty runs between partners do not cross or overlap. "
    "Fill all cells between each pair of matching ends with color {color}. "
    "The repaired wall encloses one or more empty regions; empty cells connect only up, down, left "
    "and right. Leave every other cell unchanged."
)


def regions(mask):
    """Independent visible-grid flood fill; returns only enclosed empty regions."""
    wall = np.asarray(mask, dtype=bool)
    h, w = wall.shape
    unseen = {(r, c) for r in range(h) for c in range(w) if not wall[r, c]}
    holes = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        q, cells, edge = deque([start]), {start}, False
        while q:
            r, c = q.popleft()
            edge |= r in (0, h - 1) or c in (0, w - 1)
            for dr, dc in D4:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    q.append(p)
        if not edge:
            holes.append(cells)
    return holes


def solve_closed(grid, color):
    return "closed" if regions(np.asarray(grid) == color) else "open"


def solve_repair(grid, color):
    """No construction metadata: pair every visible end along its unique outgoing ray."""
    a = np.asarray(grid)
    h, w = a.shape
    cells = {tuple(p) for p in np.argwhere(a == color)}
    ends = {}
    for r, c in cells:
        neighbors = [(r + dr, c + dc) for dr, dc in D8 if (r + dr, c + dc) in cells]
        if len(neighbors) == 1:
            nr, nc = neighbors[0]
            ends[r, c] = r - nr, c - nc
    if not ends:
        raise ValueError("No broken wall ends")
    runs, paired, used = [], set(), set()
    for end, (dr, dc) in sorted(ends.items()):
        if end in paired:
            continue
        r, c = end[0] + dr, end[1] + dc
        gap = []
        while 0 <= r < h and 0 <= c < w and (r, c) not in cells:
            gap.append((r, c))
            r, c = r + dr, c + dc
        partner = r, c
        if not gap or ends.get(partner) != (-dr, -dc) or partner in paired:
            raise ValueError("A wall end has no uniquely aligned partner")
        if used.intersection(gap):
            raise ValueError("Repair runs overlap")
        # Crossing diagonal segments can intersect between cells.
        segment = (end, partner)
        for previous, _, _ in runs:
            if _cross(segment, previous):
                raise ValueError("Repair runs cross")
        runs.append((segment, gap, (dr, dc)))
        paired.update((end, partner))
        used.update(gap)
    if paired != set(ends):
        raise ValueError("Some broken ends remain unmatched")
    out = a.copy()
    for r, c in used:
        out[r, c] = color
    if not regions(out == color):
        raise ValueError("The repair encloses no empty region")
    return out


def _cross(first, second):
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    a, b = first
    c, d = second
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _erode(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return mask & p[:h, 1:w + 1] & p[2:h + 2, 1:w + 1] & p[1:h + 1, :w] & p[1:h + 1, 2:w + 2]


def _planned_holes(wall):
    labels, count = T.holes(wall)
    return [set(map(tuple, np.argwhere(labels == i))) for i in range(1, count + 1)]


def _body(h, w, family, rng):
    r, c = np.mgrid[:h, :w]
    a = np.ones((h, w), dtype=bool)
    if family == "chamfer":
        k = int(rng.integers(1, max(2, min(h, w) // 3 + 1)))
        a = (r + c >= k) & (r + w - 1 - c >= k) & (h - 1 - r + c >= k) & (h - 1 - r + w - 1 - c >= k)
    elif family in ("ellipse", "diamond"):
        yy, xx = abs((r - (h - 1) / 2) / (h / 2)), abs((c - (w - 1) / 2) / (w / 2))
        a = yy * yy + xx * xx <= 1 if family == "ellipse" else yy + xx <= 1
    elif family in ("L", "T", "plus", "step"):
        th, tw = max(3, int(round(h * .50))), max(3, int(round(w * .50)))
        if family == "L":
            a = (r >= h - th) | (c < tw)
        elif family == "T":
            a = (r < th) | ((c >= (w - tw) // 2) & (c < (w + tw) // 2))
        elif family == "plus":
            a = ((r >= (h - th) // 2) & (r < (h + th) // 2)) | ((c >= (w - tw) // 2) & (c < (w + tw) // 2))
        else:
            a = (r >= h // 3) | (c < 2 * w // 3)
            a &= (r < 2 * h // 3) | (c >= w // 3)
    return a


def _wall(rng, h, w, family, thickness=1, scale=(.65, 1.0), subdivide=False):
    bh = max(3, min(h, int(round(h * rng.uniform(*scale)))))
    bw = max(3, min(w, int(round(w * rng.uniform(*scale)))))
    if family == "diamond":
        bh = bw = min(bh, bw)
    body = (_body(bw, bh, family, rng).T if family in ("L", "T", "plus", "step") and rng.integers(2)
            else _body(bh, bw, family, rng))
    inner = body.copy()
    for _ in range(thickness):
        inner = _erode(inner)
    wall = body & ~inner
    if (family == "rooms" or subdivide) and max(bh, bw) >= 8:
        # A small, readable floor plan. Irregular spacing, at least two-cell rooms.
        axis = int(bw > bh) if rng.random() < .65 else int(rng.integers(2))
        extent = bh if axis == 0 else bw
        candidates = list(range(3, extent - 3))
        rng.shuffle(candidates)
        selected = []
        for p in candidates:
            if all(abs(p - q) >= 3 + thickness for q in selected):
                selected.append(p)
                if len(selected) >= int(rng.integers(1, min(4, max(2, extent // 5)))):
                    break
        for p in selected:
            if axis == 0:
                wall[p:p + thickness, :] |= body[p:p + thickness, :]
            else:
                wall[:, p:p + thickness] |= body[:, p:p + thickness]
    if rng.integers(2):
        wall = wall[::-1]
    if rng.integers(2):
        wall = wall[:, ::-1]
    rr, cc = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    result = np.zeros((h, w), dtype=bool)
    result[rr:rr + bh, cc:cc + bw] = wall
    return result


def _cut_options(wall):
    """Enumerate straight boundary stretches with enough visible wall at both ends."""
    cells = {tuple(p) for p in np.argwhere(wall)}
    straight = {}
    for r, c in cells:
        neighbors = {(dr, dc) for dr, dc in D8 if (r + dr, c + dc) in cells}
        if len(neighbors) == 2:
            dr, dc = min(neighbors)
            if (-dr, -dc) in neighbors:
                straight[r, c] = (dr, dc)
    options = []
    for start, (dr, dc) in straight.items():
        if straight.get((start[0] - dr, start[1] - dc)) == (dr, dc):
            continue
        chain, p = [], start
        while straight.get(p) == (dr, dc):
            chain.append(p)
            p = p[0] + dr, p[1] + dc
        for left in range(1, len(chain) - 1):
            for right in range(left + 1, len(chain)):
                options.append((chain[left:right], ("main_diagonal" if dr == dc else "anti_diagonal") if dr and dc else "vertical" if dr else "horizontal"))
    return options


def _open_region(rng, wall, hole, wide):
    """Remove a rectangular doorway through a straight wall, never a jagged channel."""
    outside = T.outside_mask(wall)
    h, w = wall.shape
    columns = {}
    for r, c in hole:
        for dr, dc in D4:
            y, x = r + dr, c + dc
            run = []
            while 0 <= y < h and 0 <= x < w and wall[y, x]:
                run.append((y, x))
                y, x = y + dr, x + dc
            if run and (not (0 <= y < h and 0 <= x < w) or outside[y, x]):
                normal = r if dr else c
                tangent = c if dr else r
                columns.setdefault((dr, dc, normal, len(run)), {})[tangent] = run
    choices = []
    for key, spans in columns.items():
        ordered = sorted(spans)
        groups = []
        for v in ordered:
            if not groups or v != groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(v)
        for group in groups:
            choices.append((key, spans, group))
    if not choices:
        return None
    key, spans, group = choices[int(rng.integers(len(choices)))]
    cap = len(group)
    length = int(rng.integers(max(1, cap // 2), cap + 1)) if wide else int(rng.integers(1, min(3, cap) + 1))
    offset = int(rng.integers(cap - length + 1))
    removed = [p for v in group[offset:offset + length] for p in spans[v]]
    result = wall.copy()
    for p in removed:
        result[p] = False
    return result, removed, length


def _paired_breaches(rng, wall):
    """Prepare both labels with identical local wall statistics on the same body.

    Open every room once, or open all but one and give another room a second
    doorway. Both examples remove the same number and length of wall runs.
    Degree histograms are checked, rather than assumed, before accepting a pair.
    The remaining difference is which regions communicate with the outside.
    """
    labels, count = T.holes(wall)
    if count < 2:
        return None
    h, w = wall.shape
    options = {i: [] for i in range(1, count + 1)}
    for run, direction in _cut_options(wall):
        touching = set()
        for r, c in run:
            for dr, dc in D4:
                y, x = r + dr, c + dc
                if not (0 <= y < h and 0 <= x < w):
                    touching.add(0)
                elif not wall[y, x]:
                    touching.add(int(labels[y, x]))
        if len(touching) == 2 and 0 in touching:
            options[max(touching)].append((run, direction))
    if any(not choices for choices in options.values()):
        return None
    cap = min(max(len(run) for run, _ in choices) for choices in options.values())
    length = int(rng.integers(1, cap + 1))
    options = {i: [(run, d) for run, d in choices if len(run) == length] for i, choices in options.items()}
    pools = {True: {}, False: {}}
    for trial in range(32):
        is_closed = bool(trial % 2)
        order = list(options)
        if is_closed:
            order.remove(int(rng.choice(order)))
            order.append(int(rng.choice(order)))
        rng.shuffle(order)
        removed = set()
        for i in order:
            candidates = [run for run, _ in options[i]
                          if all(max(abs(r - y), abs(c - x)) >= 3 for r, c in run for y, x in removed)]
            if not candidates:
                break
            removed.update(candidates[int(rng.integers(len(candidates)))])
        else:
            a = wall.copy()
            for p in removed:
                a[p] = False
            padded = np.pad(a, 1)
            degrees = []
            for neighbors in (D4, D8):
                degree = sum(padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w].astype(int) for dy, dx in neighbors)
                degrees.extend(int((a & (degree == k)).sum()) for k in range(len(neighbors) + 1))
            signature = tuple(degrees)
            if bool(T.count_holes(a)) != is_closed:
                continue
            pools[is_closed][signature] = (a, removed)
            if signature in pools[not is_closed]:
                return pools[True][signature], pools[False][signature], [length] * count
    return None


def _calm_field(rng, h, w, colors, family):
    """Draw from canvas coordinates alone, without holes, ends or gap positions.

    Two or three colors carry the main pattern. Additional palette colors get
    small patches later, rather than turning each band or motif into a rainbow.
    """
    a = np.full((h, w), colors[0], dtype=np.uint8)
    details = {"background_scale": [], "background_patch_count": 0}
    if len(colors) == 1 or family == "accents":
        return a, details
    yy, xx = np.indices((h, w))
    n = min(len(colors), 2+int(rng.random() < .35))
    primary = colors[:n]
    if family in ("bands", "diagonal_bands"):
        axis = int(w > h) if min(h, w) < 7 else int(rng.integers(2))
        levels = xx if axis else yy
        if family == "diagonal_bands":
            other = yy if axis else xx
            levels = levels + int(rng.choice((-2, -1, 1, 2)))*other
        fractions = [rng.uniform(.30, .70)] if n == 2 else [rng.uniform(.22, .40), rng.uniform(.60, .78)]
        thresholds = [float(levels.min()+fraction*(levels.max()-levels.min()+1)) for fraction in fractions]
        a = np.asarray(primary, dtype=np.uint8)[np.searchsorted(thresholds, levels)]
        details["background_scale"] = [axis, *map(lambda v: round(float(v), 2), thresholds)]
    elif family in ("repeated_motifs", "sparse_pattern"):
        py = int(rng.integers(4, max(5, min(10, h//2)+1)))
        px = int(rng.integers(4, max(5, min(10, w//2)+1)))
        dy, dx = int(rng.integers(py)), int(rng.integers(px))
        y, x = (yy+dy) % py, (xx+dx) % px
        kind = str(rng.choice(("blocks", "bars", "crosses"))) if family == "repeated_motifs" else str(rng.choice(("lattice", "dotted_rows", "offset_dots")))
        if kind == "blocks":
            sh, sw = max(1, py//3), max(1, px//3)
            marked = (y < sh) & (x < sw)
        elif kind == "bars":
            marked = (y == 0) & (x < max(2, px//2))
        elif kind == "crosses":
            marked = ((y == 1) & (x < 3)) | ((x == 1) & (y < 3))
        elif kind == "lattice":
            marked = (y == 0) & (x == 0)
        elif kind == "dotted_rows":
            marked = (y == 0) & ((xx+dx) % max(2, px//2) == 0)
        else:
            marked = (y == 0) & ((xx+dx+(yy+dy)//py*(px//2)) % px == 0)
        index = ((yy+dy)//py+(xx+dx)//px) % (n-1)
        a[marked] = np.asarray(primary[1:], dtype=np.uint8)[index[marked]]
        details.update(background_scale=[py, px], background_motif=kind)
    elif family in ("large_patches", "local_patches"):
        # A few broad shapes or one local cluster, with a quiet field around it.
        if family == "local_patches":
            bh, bw = max(2, int(.65*h)), max(2, int(.65*w))
            top, left = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
            count, fraction = int(rng.integers(2, 5)), rng.uniform(.20, .32)
        else:
            top, left, bh, bw = 0, 0, h, w
            count, fraction = n-1, rng.uniform(.22, .38)
        patch_dimensions = []
        for i in range(count):
            target = fraction*h*w/count
            sh = max(1, min(bh, int(round(np.sqrt(target*h/w)*rng.uniform(.70, 1.30)))))
            sw = max(1, min(bw, int(round(target/sh))))
            r, c = top+int(rng.integers(bh-sh+1)), left+int(rng.integers(bw-sw+1))
            a[r:r+sh, c:c+sw] = primary[1+i % (n-1)]
            patch_dimensions.append([sh, sw])
        details.update(background_scale=patch_dimensions, background_patch_count=count)
    else:
        raise ValueError(f"Unknown calm background style {family}")
    return a, details


def _palette_patches(rng, a, wall, colors, protected, family):
    """Place visible missing colors as rectangles, preserving existing colors."""
    h, w = wall.shape
    occupied = protected.copy()
    missing = []
    for color in colors[1:]:
        spots = np.argwhere((a == color) & ~wall)
        if len(spots):
            occupied[tuple(spots[int(rng.integers(len(spots)))])] = True
        else:
            missing.append(color)
    dimensions_used = []
    for color in missing:
        target = max(1, min(24 if family == "accents" else 16,
                            int((.14 if family == "accents" else .08)*h*w/max(1, len(missing)))))
        dimensions = [(sh, sw) for sh in range(1, min(h, 6)+1) for sw in range(1, min(w, 6)+1)
                      if max(1, target//2) <= sh*sw <= target and max(sh/sw, sw/sh) <= 3]
        rng.shuffle(dimensions)
        allowed = ~halo(occupied) if min(h, w) >= 8 else ~occupied
        for sh, sw in dimensions:
            spots = np.argwhere(placement_map(np.ones((sh, sw), bool), allowed))
            if not len(spots):
                continue
            r, c = map(int, spots[int(rng.integers(len(spots)))])
            a[r:r+sh, c:c+sw] = color
            occupied[r:r+sh, c:c+sw] = True
            dimensions_used.append([sh, sw])
            break
        else:
            return None
    return dimensions_used


def _paint(rng, wall, colors, family, protected=None, details=None):
    if (~wall).sum() < len(colors)-1:
        return None
    # The unbroken outline is used only to keep added accent patches away from
    # every repair location alike. Broad patterns continue naturally through
    # the missing cells, and never derive their phase or colors from the gaps.
    protected = wall if protected is None else protected
    a, info = _calm_field(rng, *wall.shape, colors[1:], family)
    accents = _palette_patches(rng, a, wall, colors, protected, family)
    if accents is None:
        return None
    a[wall] = colors[0]
    if len(np.unique(a)) != len(colors):
        return None
    info["background_patch_count"] += len(accents)
    info["background_palette_patch_sizes"] = accents
    info["background_base_share"] = float(np.count_nonzero((a == colors[1]) & ~wall)/(~wall).sum())
    if details is not None:
        details.update(info)
    return a


def _palette(rng, h, w):
    # Keep ten-color scenes, but do not require tiny canvases to carry nine
    # background colors. Ordinary low-color scenes are more common.
    cap = min(10, 1 + (h - 2) * (w - 2), 2+h*w//24)
    weights = np.array([.20, .16, .13, .11, .10, .09, .08, .07, .06])[:cap-1]
    n = int(rng.choice(np.arange(2, cap+1), p=weights/weights.sum()))
    return [int(c) for c in rng.permutation(10)[:n]]


def _family(rng, choices):
    weights = np.array([FAMILY_WEIGHTS[f] for f in choices])
    return str(rng.choice(choices, p=weights/weights.sum()))


def validate_example(ex, task):
    if min(np.asarray(ex["input"]).shape) < MIN_SIDE:
        raise ValueError("Wall canvases require at least eight cells on each side")
    color = ex["metadata"]["obj_color"]
    answer = solve_closed(ex["input"], color) if task == 36 else solve_repair(ex["input"], color).tolist()
    if answer != ex["output"]:
        raise ValueError("Visible topology answer differs from construction")
    if len(np.unique(ex["input"])) != ex["metadata"]["n_colors"]:
        raise ValueError("Palette budget was lost")


def generate_open_closed(rng):
    h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
    closed = bool(rng.integers(2))
    families = FAMILIES if min(h, w) >= 7 else ("rectangle", "rooms", "ellipse")
    family = _family(rng, families)
    colors = _palette(rng, h, w)
    fill = str(rng.choice(FILLS, p=FILL_WEIGHTS))
    maximum_thickness = max(1, min(h, w) // (5 if family in ("L", "T", "plus", "step") else 4))
    thickness = min(maximum_thickness, int(rng.choice((1, 2, 3), p=(.65, .28, .07)))) if min(h, w) >= 8 else 1
    wide = bool(rng.integers(2))
    subdivide = rng.random() < .30
    partial = closed and rng.random() < .60
    for attempt in range(1, 201):
        wall = _wall(rng, h, w, family, thickness, subdivide=subdivide)
        holes = _planned_holes(wall)
        if not holes:
            continue
        initial = len(holes)
        pair = _paired_breaches(rng, wall) if thickness == 1 and initial > 1 else None
        target = int(rng.integers(1, initial)) if closed and partial and initial > 1 else initial if closed else 0
        if closed and not partial:
            target = initial
        widths, removed = [], []
        if pair is not None:
            (wall, edits), widths = pair[0 if closed else 1], pair[2]
            removed = list(edits)
            holes = _planned_holes(wall)
            target = len(holes)
        while len(holes) > target:
            hole = holes[int(rng.integers(len(holes)))]
            opened = _open_region(rng, wall, hole, wide)
            if opened is None:
                break
            wall, edit, width = opened
            removed.extend(edit)
            widths.append(width)
            holes = _planned_holes(wall)
        if bool(holes) != closed or not wall.any():
            continue
        background_info = {}
        a = _paint(rng, wall, colors, fill, details=background_info)
        if a is None:
            continue
        ex = {"input": a.tolist(), "output": "closed" if closed else "open", "metadata": {
            "obj_color": colors[0], "n_colors": len(colors), "grid_h": h, "grid_w": w,
            "case": family + ("_breached" if removed else "_intact"), "silhouette": family,
            "thickness": thickness, "background": fill if len(colors) > 2 else "solid",
            "n_holes": len(holes), "hole_areas": [len(x) for x in holes],
            "initial_holes": initial, "gap_widths": widths, "foreground_cells": int(wall.sum()),
            "paired_local_statistics": pair is not None,
            "attempts": attempt, **background_info}}
        validate_example(ex, 36)
        return ex
    raise RuntimeError("Task 36 could not realize its fixed size, shape and palette")


def generate_close_open_object(rng):
    h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
    families = ["rectangle", "rooms"]
    if min(h, w) >= 11:
        families += ["L", "step", "chamfer", "diamond"]
    if min(h, w) >= 15:
        families += ["T", "plus"]
    if min(h, w) >= 16:
        families += ["ellipse"]
    family = _family(rng, families)
    colors = _palette(rng, h, w)
    fill = str(rng.choice(FILLS, p=FILL_WEIGHTS))
    n_gaps = 1 + int(rng.random() < .28)
    length_mode = str(rng.choice(("short", "medium", "long"), p=(.40, .40, .20)))
    for attempt in range(1, 301):
        closed = _wall(rng, h, w, family, scale=(.80, 1.0))
        original_holes = _planned_holes(closed)
        if not original_holes:
            continue
        options = _cut_options(closed)
        if not options:
            continue
        wall, runs, directions, taken = closed.copy(), [], [], set()
        for _ in range(n_gaps):
            candidates = [(run, direction) for run, direction in options
                          if all(max(abs(r - y), abs(c - x)) >= 2 for r, c in run for y, x in taken)]
            if not candidates:
                break
            maximum = max(len(run) for run, _ in candidates)
            low, high = ((1, min(3, maximum)) if length_mode == "short" else
                         (max(1, maximum // 3), max(1, 2 * maximum // 3)) if length_mode == "medium" else
                         (max(1, maximum // 2), maximum))
            candidates = [(run, direction) for run, direction in candidates if low <= len(run) <= high]
            run, direction = candidates[int(rng.integers(len(candidates)))]
            for p in run:
                wall[p] = False
            taken.update(run)
            runs.append(run)
            directions.append(direction)
        if len(runs) != n_gaps:
            continue
        background_info = {}
        a = _paint(rng, wall, colors, fill, protected=closed, details=background_info)
        if a is None:
            continue
        out = a.copy()
        out[closed] = colors[0]
        try:
            actual = solve_repair(a, colors[0])
        except ValueError:
            continue
        if not np.array_equal(actual, out):
            continue
        inner = T.hole_mask(closed)
        if not (inner & T.outside_mask(wall)).any():
            continue
        ex = {"input": a.tolist(), "output": out.tolist(), "metadata": {
            "obj_color": colors[0], "n_colors": len(colors), "grid_h": h, "grid_w": w,
            "case": family, "silhouette": family, "background": fill if len(colors) > 2 else "solid",
            "n_holes": len(original_holes), "hole_areas": [len(x) for x in original_holes],
            "n_gaps": n_gaps, "gap_lengths": [len(run) for run in runs], "gap_directions": directions,
            "gap_vectors": [{"horizontal": [0, -1], "vertical": [-1, 0],
                             "main_diagonal": [-1, -1], "anti_diagonal": [-1, 1]}[d] for d in directions],
            "length_mode": length_mode, "removed_cells": [list(map(int, p)) for p in sorted(taken)],
            "foreground_cells": int(wall.sum()), "attempts": attempt, **background_info}}
        validate_example(ex, 37)
        return ex
    raise RuntimeError("Task 37 could not realize its fixed visible-repair plan")
