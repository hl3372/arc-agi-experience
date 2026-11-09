"""Count colors in readable closed enclosures, with an inferred ground color."""
from __future__ import annotations

from collections import Counter, deque
import numpy as np

from generators.clean_topology_36_37 import _body, _erode
from generators.rigid_scene import infer_ground
from generators.scene_shapes_56_60 import decorate

FAMILIES = ("rectangle", "chamfer", "ellipse", "diamond", "L", "T", "plus", "step")
LAYOUTS = ("scattered", "bands", "diagonal_bands", "blocks", "panels", "clusters")

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5


def enclosed_region(grid, wall_color):
    """Independent border flood, treating every non-wall color as passable."""
    a = np.asarray(grid)
    h, w = a.shape
    seen = np.zeros(a.shape, bool)
    queue = deque()
    for y in range(h):
        for x in range(w):
            if (y in (0, h-1) or x in (0, w-1)) and a[y, x] != wall_color:
                seen[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and a[yy, xx] != wall_color:
                seen[yy, xx] = True
                queue.append((yy, xx))
    inside = ~seen & (a != wall_color)
    cells = set(map(tuple, np.argwhere(inside)))
    if not cells:
        raise ValueError("The wall has no enclosed region")
    found = {next(iter(cells))}
    queue = deque(found)
    while queue:
        y, x = queue.popleft()
        for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if p in cells and p not in found:
                found.add(p)
                queue.append(p)
    if found != cells:
        raise ValueError("The wall encloses more than one region")
    return inside


def solve(grid, wall_color):
    a = np.asarray(grid)
    bg = infer_ground(a)
    if bg == wall_color:
        raise ValueError("Wall and ground must be distinct")
    inside = enclosed_region(a, wall_color)
    counts = Counter(a[inside & (a != bg)].tolist())
    ranked = counts.most_common()
    if not ranked or (len(ranked) > 1 and ranked[0][1] <= ranked[1][1]):
        raise ValueError("No unique most frequent interior color")
    return str(ranked[0][0]), counts, inside


def validate_example(ex):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(3 <= d <= 30 for d in a.shape) or a.size < 30:
        raise ValueError("Invalid enclosure canvas")
    answer, counts, inside = solve(a, m["enclosure_color"])
    values = sorted(counts.values(), reverse=True)
    if answer != ex["output"] or values[0]-(values[1] if len(values)>1 else 0) < 2:
        raise ValueError("Enclosure answer or uniqueness margin differs")
    if int(inside.sum()) != m["interior_area"] or len(counts) != m["n_interior_colors"]:
        raise ValueError("Visible enclosure differs from planned composition")
    if len(np.unique(a)) != m["n_colors"] or infer_ground(a) != m["background_color"]:
        raise ValueError("Enclosure palette or background differs from its plan")


def _counts(rng, total, n, close):
    """Allocate every planned color without reducing the requested cardinality."""
    if n == 1:
        return [total]
    # Distribute first, then transfer just enough to make a unique winner.
    parts = rng.multinomial(total-n, np.ones(n)/n)+1
    if not close:
        bonus = int(total*float(rng.uniform(.08, .30)))
        for j in rng.permutation(np.arange(1, n)):
            take = min(bonus, int(parts[j])-1)
            parts[j] -= take
            parts[0] += take
            bonus -= take
    for _ in range(total):
        j = 1+int(np.argmax(parts[1:]))
        if parts[0] >= parts[j]+2:
            return parts.tolist()
        if parts[j] <= 1:
            return None
        parts[j] -= 1
        parts[0] += 1
    return None


def _order(rng, cells, layout):
    y, x = cells[:, 0], cells[:, 1]
    if layout == "scattered":
        return rng.permutation(len(cells))
    if layout == "bands":
        key = y*(x.max()+2)+x if rng.integers(2) else x*(y.max()+2)+y
    elif layout == "diagonal_bands":
        key = y+float(rng.uniform(.4, 2))*x
    elif layout == "blocks":
        size = int(rng.integers(2, 5))
        key = (y//size)*(x.max()+2)+(x//size)+(y%size)*.02+(x%size)*.001
    elif layout == "panels":
        key = np.minimum(y-y.min(), y.max()-y)+np.minimum(x-x.min(), x.max()-x)*.07
    else:
        cy, cx = float(rng.uniform(y.min(), y.max()+1)), float(rng.uniform(x.min(), x.max()+1))
        key = (y-cy)**2+(x-cx)**2
    if rng.integers(2):
        key = -key
    return np.argsort(key, kind="stable")


def generate(rng):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
        if h*w >= 30:
            break
    ncolors = int(rng.integers(3, min(10, 3+max(0, (h*w-25)//2))+1))
    palette = [int(c) for c in rng.permutation(10)[:ncolors]]
    bg, wall = palette[:2]
    large_enough = min(h, w) >= 12
    contrast = large_enough and ncolors >= 4 and rng.random() < .32
    family = str(rng.choice(["diamond", "L", "T", "plus", "step"])) if contrast else str(rng.choice(FAMILIES, p=[.43, .10, .10, .08, .08, .07, .07, .07]))
    # Nonrectangular walls need a 5x5 body plus enough exterior field.
    # In particular, 6x6 and 6x7 canvases cannot support that body, a
    # full palette and the required background margin at the same time.
    if min(h, w) < 6 or h*w < 60:
        family = "rectangle"
    layout = str(rng.choice(LAYOUTS))
    style = str(rng.choice(["solid", "lattice", "stripes", "checker", "diagonal"], p=[.56, .14, .12, .07, .11]))
    leave_bg = bool(rng.random() < .4)
    for attempt in range(1, 401):
        # Retain enough surrounding field to infer it without a color giveaway.
        # Thin canvases must retain compact 3x4 walls: a larger minimum wall
        # can exhaust the exterior field when all ten colors are requested.
        low_h = 3 if min(h, w) < 5 else max(3, int(h*.38))
        low_w = 3 if min(h, w) < 5 else max(3, int(w*.38))
        eh = int(rng.integers(low_h, max(4, min(h, int(h*.88))+1)))
        ew = int(rng.integers(low_w, max(4, min(w, int(w*.88))+1)))
        if min(h, w) >= 8:
            eh, ew = max(5, eh), max(5, ew)
        if contrast:
            eh, ew = max(7, eh), max(7, ew)
        if eh*ew > .60*h*w:
            continue
        if family != "rectangle" and min(eh, ew) < 5:
            continue
        body = _body(eh, ew, family, rng)
        if rng.integers(2):
            body = body[::-1]
        if rng.integers(2):
            body = body[:, ::-1]
        thickness = int(rng.choice(np.arange(1, min(4, max(1, (min(eh, ew)-2)//3))+1)))
        if contrast:
            thickness = 1
        inner = body.copy()
        for _ in range(thickness):
            inner = _erode(inner)
        if inner.sum() < 2:
            continue
        boundary = body & ~inner
        r, c = int(rng.integers(h-eh+1)), int(rng.integers(w-ew+1))
        wm, im = np.zeros((h, w), bool), np.zeros((h, w), bool)
        wm[r:r+eh, c:c+ew] = boundary
        im[r:r+eh, c:c+ew] = inner
        area = int(im.sum())
        nbg = min(area-2, int(area*float(rng.uniform(.12, .6)))) if leave_bg else 0
        counted = area-nbg
        # One cell per losing color and three for the winner suffice.
        # Requiring two cells per color unnecessarily erased larger contests.
        upper = min(ncolors-2, counted-2, 8) if counted > 2 else 1
        if upper < 1 or (contrast and upper < 2):
            continue
        options = np.arange(1, upper+1)
        weights = np.array([.02, .15, .16, .15, .29, .25, .22, .18])[:upper]
        if contrast:
            weights[0] = 0
        n = int(rng.choice(options, p=weights/weights.sum()))
        counts = _counts(rng, counted, n, contrast or rng.random() < .55)
        if counts is None:
            continue
        colors = [int(v) for v in rng.choice(palette[2:], size=n, replace=False)]
        grid = np.full((h, w), bg, int)
        grid[wm] = wall
        cells = np.argwhere(im)
        order = _order(rng, cells, layout)
        runs = [(col, cnt) for col, cnt in zip(colors, counts)] + ([(bg, nbg)] if nbg else [])
        rng.shuffle(runs)
        pos = 0
        for col, cnt in runs:
            pp = cells[order[pos:pos+cnt]]
            grid[pp[:, 0], pp[:, 1]] = col
            pos += cnt
        # Counterexamples put a competing color in exterior corners/concavities.
        # A modest share of scenes requires tracing the actual wall boundary.
        intruders = np.zeros(grid.shape, bool)
        if contrast:
            yy, xx = np.nonzero(wm)
            box = np.zeros(grid.shape, bool)
            box[yy.min()+1:yy.max(), xx.min()+1:xx.max()] = True
            spots = np.argwhere(box & ~wm & ~im)
            competitor = 1+int(np.argmax(counts[1:]))
            needed = counts[0]-counts[competitor]+1
            if len(spots) < needed:
                continue
            selected = spots[_order(rng, spots, "clusters")[:needed]]
            intruders[selected[:, 0], selected[:, 1]] = True
            grid[intruders] = colors[competitor]
        try:
            info = decorate(rng, grid, wm | im | intruders, bg, palette[2:], style,
                            density=.10, max_pattern_fraction=.06 if h*w < 80 else .16)
            if infer_ground(grid) != bg:
                continue
            answer, actual, inside = solve(grid, wall)
        except ValueError:
            continue
        if answer != str(colors[0]) or not np.array_equal(inside, im):
            continue
        yy, xx = np.nonzero(wm)
        box_values = grid[yy.min()+1:yy.max(), xx.min()+1:xx.max()]
        box_counts = Counter(v for v in box_values.flat if v not in (bg, wall))
        best = max(box_counts.values(), default=0)
        box_winners = [col for col, ct in box_counts.items() if ct == best]
        box_wrong = box_winners != [colors[0]]
        if contrast and not box_wrong:
            continue
        ranked = sorted(actual.values(), reverse=True)
        field_counts = Counter(grid.flat)
        ex = {"input": grid.tolist(), "output": answer, "metadata": {
            "enclosure_color": wall, "background_color": bg, "ignore_background": True,
            "enclosure_family": family, "wall_thickness": thickness, "enclosure_h": eh, "enclosure_w": ew,
            "interior_area": area, "interior_bg_fraction": nbg/area, "interior_layout": layout,
            "n_interior_colors": n, "n_colors": ncolors, "winner_count": ranked[0],
            "runner_up_count": ranked[1] if len(ranked)>1 else 0,
            "interior_counts": [[int(col), int(ct)] for col, ct in actual.most_common()],
            "background_margin": field_counts[bg]/max(ct for col, ct in field_counts.items() if col != bg),
            "boundary_contrast": bool(contrast), "bbox_answer_differs": bool(box_wrong),
            "outside_style": info["background_style"], "decoration_families": info["decoration_families"],
            "attempts": attempt, "frequency_rule": "unique most frequent non-background color strictly inside the wall; runner-up at least 2 cells behind"}}
        validate_example(ex)
        return ex
    raise RuntimeError(f"Task 57 cannot realize its fixed {h}x{w} {family} enclosure")
