"""Shared, varied component-search scenes for recoloring and counting.

Exact/family questions and allowed orientations are drawn after the complete
input. Every scene supports either question; distractor budgets do not disclose
which instruction will be asked. Only task 58 reserves an output-only color.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from generators.scene_shapes_56_60 import crop, shape, halo, place, placement_map, decorate

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5

FAMILY_DESCRIPTIONS = {
    "rectangle": "a filled non-square rectangle with both sides at least 2 cells long",
    "square": "a filled square",
    "bar": "a straight line at least 2 cells long and exactly 1 cell thick",
    "L": "a one-cell-thick L with both arms at least 2 cells long",
    "T": "a one-cell-thick T whose stem meets the midpoint of its crossbar",
    "U": "a one-cell-thick U with two equal parallel arms",
    "H": "a one-cell-thick H whose joining bar meets the midpoints of its two parallel bars",
    "plus": "a one-cell-thick plus with equal opposite arms",
    "frame": "a complete rectangular frame with walls exactly 1 cell thick",
}


def key(mask):
    a = crop(mask)
    return (a.shape, a.tobytes())


def transforms(mask, rule="dihedral"):
    candidates = []
    for flip in range(2 if rule == "dihedral" else 1):
        base = mask[:, ::-1] if flip else mask
        for k in range(1 if rule == "exact" else 4):
            candidate = crop(np.rot90(base, k))
            if not any(np.array_equal(candidate, other) for other in candidates):
                candidates.append(candidate)
    return candidates


def components(grid, color):
    a = np.asarray(grid)
    h, w = a.shape
    seen = set()
    result = []
    for yy, xx in np.argwhere(a == color):
        start = int(yy), int(xx)
        if start in seen:
            continue
        queue = deque([start])
        cells = {start}
        seen.add(start)
        while queue:
            y, x = queue.popleft()
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in seen and a[p] == color:
                    seen.add(p)
                    cells.add(p)
                    queue.append(p)
        r, c = min(y for y, _ in cells), min(x for _, x in cells)
        mask = np.zeros((max(y for y, _ in cells)-r+1, max(x for _, x in cells)-c+1), bool)
        for y, x in cells:
            mask[y-r, x-c] = True
        result.append((cells, mask))
    return result


def family_matches(mask, family):
    """Literal geometry predicates, independent of the shape constructor.

    Both bounding-box orientations are considered before rotating. This avoids
    the old unequal-arm L bug, which silently rejected quarter-turn matches.
    """
    h, w = mask.shape
    if family == "square":
        return h == w and bool(mask.all())
    if family == "rectangle":
        return min(h, w) >= 2 and h != w and bool(mask.all())
    if family == "bar":
        return min(h, w) == 1 and max(h, w) >= 2 and bool(mask.all())
    for hh, ww in {(h, w), (w, h)}:
        yy, xx = np.indices((hh, ww))
        if family == "L" and min(hh, ww) >= 2:
            candidate = (xx == 0) | (yy == hh-1)
        elif family == "frame" and min(hh, ww) >= 3:
            candidate = (yy == 0) | (yy == hh-1) | (xx == 0) | (xx == ww-1)
        elif family == "T" and hh >= 2 and ww >= 3 and ww % 2:
            candidate = (yy == 0) | (xx == ww//2)
        elif family == "U" and hh >= 2 and ww >= 3:
            candidate = (xx == 0) | (xx == ww-1) | (yy == hh-1)
        elif family == "H" and min(hh, ww) >= 3 and hh % 2:
            candidate = (xx == 0) | (xx == ww-1) | (yy == hh//2)
        elif family == "plus" and min(hh, ww) >= 3 and hh % 2 and ww % 2:
            candidate = (yy == hh//2) | (xx == ww//2)
        else:
            continue
        if any(np.array_equal(mask, v) for v in transforms(candidate)):
            return True
    return False


def matching_groups(grid, color, template, rule, family=None):
    if family is not None:
        return [cells for cells, mask in components(grid, color) if family_matches(mask, family)]
    allowed = transforms(template, rule)
    return [cells for cells, mask in components(grid, color)
            if any(np.array_equal(mask, v) for v in allowed)]


def solve(grid, color, template, rule, *, family=None, highlight=None):
    groups = matching_groups(grid, color, template, rule, family)
    if highlight is None:
        return str(len(groups))
    result = np.asarray(grid).copy()
    for cells in groups:
        for p in cells:
            result[p] = highlight
    return result.tolist()


def validate_example(ex, *, counting):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or a.size < 12:
        raise ValueError("Invalid shape-search canvas")
    template = np.array([[c == "#" for c in row] for row in m["template_rows"]])
    family = m["query_shape_family"] if counting and m["query_mode"] == "generic" else None
    expected = solve(a, m["template_color"], template, m["orientation_rule"], family=family,
                     highlight=None if counting else m["highlight_color"])
    if expected != ex["output"]:
        raise ValueError("Search output differs from the full visible-component solution")
    if len(np.unique(a)) != m["n_colors"]:
        raise ValueError("Search palette differs from its plan")
    if not counting and m["highlight_color"] in a:
        raise ValueError("Highlight color is already present")
    groups = matching_groups(a, m["template_color"], template, m["orientation_rule"], family)
    if len(groups) != m["n_occurrences"] or (len(groups) == 0) != m["zero_case"]:
        raise ValueError("Search count or zero plan was not retained")


def _layouts(h, w):
    return [(nr, nc, (h-nr+1)//nr, (w-nc+1)//nc)
            for nr in range(1, min(12, h)+1) for nc in range(1, min(12, w)+1)
            if 2 <= nr*nc <= 24 and (h-nr+1)//nr >= 1 and (w-nc+1)//nc >= 1
            and ((h-nr+1)//nr)*((w-nc+1)//nc) >= 2]


def _boxes(rng, h, w, plan):
    nr, nc, sh, sw = plan
    hs = np.full(nr, sh)+rng.multinomial(h-nr*sh-(nr-1), np.ones(nr)/nr)
    ws = np.full(nc, sw)+rng.multinomial(w-nc*sw-(nc-1), np.ones(nc)/nc)
    ys, xs = np.r_[0, np.cumsum(hs+1)[:-1]], np.r_[0, np.cumsum(ws+1)[:-1]]
    result = [(int(y), int(x), int(hh), int(ww)) for y, hh in zip(ys, hs) for x, ww in zip(xs, ws)]
    rng.shuffle(result)
    return result


def _families(h, w, counting):
    names = ["bar"]
    if min(h, w) >= 2:
        names += ["square", "L"]
        if max(h, w) >= 3:
            names.append("rectangle")
    if min(h, w) >= 3:
        names += ["T", "U", "H", "plus", "frame"]
    if not counting:
        if min(h, w) >= 2:
            names.append("staircase")
        if min(h, w) >= 5:
            names += ["chamfer", "diamond"]
    return names


def _template(rng, h, w, family, counting):
    floor = 1 if family == "bar" else 2 if family in ("rectangle", "square", "L", "staircase") else 3 if family in ("T", "U", "H", "plus", "frame") else 5
    hh = int(rng.integers(max(floor, int(h*.55)), h+1))
    ww = int(rng.integers(max(floor, int(w*.55)), w+1))
    if family == "bar" and max(hh, ww) < 2:
        if h >= 2:
            hh = 2
        else:
            ww = 2
    if family == "rectangle" and hh == ww:
        if ww > 2:
            ww -= 1
        elif w >= 3:
            ww = 3
        else:
            hh = 3
    if counting:
        if family in ("T", "plus") and ww % 2 == 0:
            ww -= 1
        if family in ("H", "plus") and hh % 2 == 0:
            hh -= 1
    return shape(rng, hh, ww, family, thin=counting)[0]


def orientation_text(template, rule):
    exact, rotated, full = [{key(x) for x in transforms(template, r)} for r in ("exact", "rotations", "dihedral")]
    if full == exact:
        return ""
    if rule == "exact":
        return "in this exact orientation only; other orientations do not count"
    if rule == "rotations":
        return "rotated copies also count" + (", reflected copies do not" if full != rotated else "")
    return "rotated and reflected copies also count"


def choose_question(rng, counting):
    rule = str(rng.choice(["exact", "rotations", "dihedral"], p=[.45, .25, .30]))
    mode = "generic" if counting and rng.random() < .5 else "exact_pattern"
    return rule, mode


def near_misses(rng, template, h, w, family, counting):
    """Familiar foils that defeat cell-count and bounding-box shortcuts.

    Draw these before choosing the question. They fail every exact orientation
    and, on task 59, the family question too. No random cell carving is needed.
    """
    area = int(template.sum())
    allowed = transforms(template)
    same_area, same_box = [], []

    def accept(mask):
        return (not any(np.array_equal(mask, v) for v in allowed)
                and (not counting or not family_matches(mask, family)))

    # Derive dimensions from each familiar silhouette's area formula.
    proposals = []
    for hh in range(1, h+1):
        for ww in range(1, w+1):
            if hh*ww == area:
                proposals.append((hh, ww, "rectangle"))
            if min(hh, ww) >= 2 and hh+ww-1 == area:
                proposals.append((hh, ww, "L"))
            if min(hh, ww) >= 3:
                if hh+ww-1 == area and ww % 2:
                    proposals.append((hh, ww, "T"))
                if 2*hh+ww-2 == area:
                    proposals.append((hh, ww, "U"))
                if 2*hh+ww-2 == area and hh % 2:
                    proposals.append((hh, ww, "H"))
                if hh+ww-1 == area and hh % 2 and ww % 2:
                    proposals.append((hh, ww, "plus"))
                if 2*hh+2*ww-4 == area:
                    proposals.append((hh, ww, "frame"))
    rng.shuffle(proposals)
    for hh, ww, fam in proposals:
        mask, _ = shape(rng, hh, ww, fam, thin=True)
        if int(mask.sum()) == area and accept(mask):
            same_area.append((mask, fam))
            if len(same_area) >= 12:
                break
    th, tw = template.shape
    for fam in _families(th, tw, False):
        mask, _ = shape(rng, th, tw, fam, thin=True)
        if mask.shape == template.shape and accept(mask):
            same_box.append((mask, fam))
    return same_area, same_box


def generate(rng, *, counting):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
        if h*w >= 12 and _layouts(h, w):
            break
    ncolors = int(rng.integers(2, min(10 if counting else 9, max(2, h*w//7+1))+1))
    permutation = [int(c) for c in rng.permutation(10)]
    palette = permutation[:ncolors]
    bg, color = palette[:2]
    highlight = None if counting else permutation[ncolors]
    zero = bool(rng.random() < .085)
    corner = not counting and min(h, w) >= 3 and rng.random() < .28
    style = str(rng.choice(["solid", "lattice", "stripes", "checker", "diagonal", "split"], p=[.48, .16, .10, .07, .12, .07]))
    options = _layouts(h, w)
    roomy = [x for x in options if min(x[2:]) >= 3]
    if roomy and rng.random() < .85:
        options = roomy
    if rng.random() < .48:
        subset = [x for x in options if x[0]*x[1] <= 6]
    else:
        subset = [x for x in options if x[0]*x[1] >= 6]
    options = subset or options
    weights = np.array([1/(1+abs(np.log(x[2]/x[3]))) for x in options])
    plan = options[int(rng.choice(len(options), p=weights/weights.sum()))]
    sh, sw = plan[2:]
    family = str(rng.choice(_families(sh, sw, counting)))
    template = _template(rng, sh, sw, family, counting)
    all_variants = transforms(template)
    total = plan[0]*plan[1]
    nbase = 0 if zero else int(rng.integers(1, total))
    nfamily = 0 if zero else int(rng.integers(0, total-nbase))
    area_foils, box_foils = near_misses(rng, template, sh, sw, family, counting)
    first_foil = int(rng.integers(2))
    for attempt in range(1, 201):
        boxes = _boxes(rng, h, w, plan)
        grid = np.full((h, w), bg, int)
        occupied = np.zeros(grid.shape, bool)
        families, kinds = [], []
        planned = []
        valid = True
        for i, (r, c, hh, ww) in enumerate(boxes):
            variants = [v for v in all_variants if v.shape[0] <= hh and v.shape[1] <= ww]
            if i < nbase:
                mask = template if i == 0 or rng.random() < .35 else variants[int(rng.integers(len(variants)))]
                kind, fam = "template", family
            elif i < nbase+nfamily:
                mask = _template(rng, hh, ww, family, counting)
                kind, fam = "size_variant", family
            else:
                mask = None
                pools = [(area_foils, "same_area"), (box_foils, "same_box")]
                if (i-nbase-nfamily+first_foil) % 2:
                    pools.reverse()
                for pool, label in pools:
                    if pool:
                        mask, fam = pool[int(rng.integers(len(pool)))]
                        kind = label
                        break
                for _ in range(100 if mask is None else 0):
                    candidate, fam = shape(rng, int(rng.integers(1, hh+1)), int(rng.integers(1, ww+1)))
                    if candidate.sum() < 1 or any(np.array_equal(candidate, v) for v in all_variants):
                        continue
                    if counting and family_matches(candidate, family):
                        continue
                    # On a 1-cell strip the non-line distractor is a single cell.
                    mask, kind = candidate, "other_shape"
                    break
                if mask is None:
                    valid = False
                    break
            if mask.shape[0] > hh or mask.shape[1] > ww:
                valid = False
                break
            rr, cc = r+int(rng.integers(hh-mask.shape[0]+1)), c+int(rng.integers(ww-mask.shape[1]+1))
            occupied[rr:rr+mask.shape[0], cc:cc+mask.shape[1]] |= mask
            grid[rr:rr+mask.shape[0], cc:cc+mask.shape[1]][mask] = color
            families.append(fam)
            kinds.append(kind)
            planned.append(({(rr+int(y), cc+int(x)) for y, x in np.argwhere(mask)}, mask.copy(), i < nbase+nfamily))
        if not valid or occupied.sum()+ncolors-1 > h*w:
            continue
        corner_added = False
        if corner:
            masks = [template] if not zero else [mask for _, mask in components(grid, color)]
            rng.shuffle(masks)
            for mask in masks:
                allowed = placement_map(mask, ~halo(occupied, False))
                if allowed is None:
                    continue
                touching = np.zeros(allowed.shape, bool)
                border = halo(occupied) & ~halo(occupied, False)
                for yy, xx in np.argwhere(mask):
                    touching |= border[yy:yy+allowed.shape[0], xx:xx+allowed.shape[1]]
                spots = np.argwhere(allowed & touching)
                if len(spots):
                    yy, xx = map(int, spots[int(rng.integers(len(spots)))])
                    patch = occupied[yy:yy+mask.shape[0], xx:xx+mask.shape[1]]
                    patch |= mask
                    grid[yy:yy+mask.shape[0], xx:xx+mask.shape[1]][mask] = color
                    planned.append(({(yy+int(y), xx+int(x)) for y, x in np.argwhere(mask)}, mask.copy(), not zero))
                    corner_added = True
                    break
        try:
            decoration = decorate(rng, grid, occupied, bg, palette[2:], style, density=.14)
        except ValueError:
            continue
        # Both modes exist on the same finished input, independent of layout.
        rule, mode = choose_question(rng, counting)
        query_family = family if mode == "generic" else None
        allowed = transforms(template, rule)
        exact_groups = [cells for cells, mask, _ in planned if any(np.array_equal(mask, v) for v in allowed)]
        family_groups = [cells for cells, _, matches_family in planned if matches_family]
        groups = family_groups if query_family is not None else exact_groups
        if (len(groups) == 0) != zero:
            raise ValueError("Search construction failed to preserve its zero plan")
        rows = ["".join("#" if v else "." for v in row) for row in template]
        output = str(len(groups)) if counting else grid.copy()
        if not counting:
            for group in groups:
                for p in group:
                    output[p] = highlight
            output = output.tolist()
        exact_count = len(exact_groups)
        family_count = len(family_groups) if counting else None
        ex = {"input": grid.tolist(), "output": output, "metadata": {
            "template_color": color, "highlight_color": highlight, "bg": bg,
            "template_rows": rows, "template_pattern": "//".join(rows), "template_cells": int(template.sum()),
            "template_family": family, "template_size": list(template.shape), "orientation_rule": rule,
            "orientation_text": orientation_text(template, rule), "n_allowed_variants": len(transforms(template, rule)),
            "query_mode": mode, "query_shape_family": family,
            "family_description": FAMILY_DESCRIPTIONS.get(family, ""),
            "n_occurrences": len(groups), "zero_case": zero, "n_colors": ncolors,
            "n_input_colors": ncolors, "grid_size": [h, w], "corner_contact": corner_added,
            "scene_object_count": len(components(grid, color)), "scene_families": families,
            "scene_kinds": kinds, "exact_count": exact_count, "family_count": family_count,
            "attempts": attempt, **decoration}}
        validate_example(ex, counting=counting)
        return ex
    raise RuntimeError(f"Task {59 if counting else 58} cannot realize its fixed {h}x{w} {family} scene")
