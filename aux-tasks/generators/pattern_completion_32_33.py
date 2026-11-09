"""Coherent, uniquely recoverable patterns with family-preserving mask sampling."""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from generators.cell_operations_29_30 import _cell_order

FAMILIES = ("periodic", "horizontal_mirror", "vertical_mirror", "both_axis_mirror",
            "rotational_180", "main_diagonal_mirror", "anti_diagonal_mirror", "full_square_symmetry")
WEIGHTS = (.3233333333333333,) + (.10777777777777778,) * 6 + (.03,)
SQUARE = frozenset(FAMILIES[-3:])
CONTENT = ("rectangular_patches", "bands", "diagonal_bands", "concentric_frames",
           "diamonds", "block_mosaic", "letters", "crosses", "staircases", "ellipses", "lattice", "mosaic")


@lru_cache(maxsize=20)
def feasible_sizes(task, family):
    result = []
    for h in range(1 if task == "32" else 2, 31):
        for w in range(1 if task == "32" else 2, 31):
            if family in SQUARE and h != w:
                continue
            if task == "32":
                if h * w < 3 or (family == "horizontal_mirror" and h < 2) or (family == "vertical_mirror" and w < 2):
                    continue
                if (family == "full_square_symmetry" and h < 3) or (family == "both_axis_mirror" and h == w == 2):
                    continue
            elif ((family in SQUARE and h < 4)
                  or (family == "horizontal_mirror" and h < 4)
                  or (family == "vertical_mirror" and w < 4)
                  or (family in ("both_axis_mirror", "rotational_180") and max(h, w) < 4)
                  or (family == "periodic" and h * w <= 4)):
                continue
            result.append((h, w))
    return tuple(result)


def _classes(h, w, family, th=0, tw=0):
    """Construction uses explicit group orbits, not the verifier's formulae."""
    from generators.pattern_symmetry import _symmetry_orbit
    lookup, groups = {}, []
    labels = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            key = (r % th, c % tw) if family == "periodic" else _symmetry_orbit(family, r, c, h, w)
            if key not in lookup:
                lookup[key] = len(groups)
                groups.append([])
            i = lookup[key]
            labels[r, c] = i
            groups[i].append(r * w + c)
    return labels, groups


@lru_cache(maxsize=96)
def _rectangles(h, w, family, th, tw, minimum):
    """Every recoverable solid rectangle, including full-span and large ones.

    A rectangle must not contain a whole symmetry orbit / tile residue class.
    The interval criteria below are exact for these eight families. The final
    visible-input solver checks that criterion independently on every example.
    """
    rows = np.asarray([(a, b) for a in range(h) for b in range(a + minimum, h + 1)])
    cols = np.asarray([(a, b) for a in range(w) for b in range(a + minimum, w + 1)])
    if not len(rows) or not len(cols):
        return np.empty((0, 4), dtype=int)
    rs = (rows[:, 1] <= h // 2) | (rows[:, 0] >= (h + 1) // 2)
    cs = (cols[:, 1] <= w // 2) | (cols[:, 0] >= (w + 1) // 2)
    if family == "periodic":
        rs = np.ones(len(rows), dtype=bool)
        cs = np.ones(len(cols), dtype=bool)
        for residue in range(th):
            last = residue + ((h - 1 - residue) // th) * th
            rs &= ~((rows[:, 0] <= residue) & (rows[:, 1] > last))
        for residue in range(tw):
            last = residue + ((w - 1 - residue) // tw) * tw
            cs &= ~((cols[:, 0] <= residue) & (cols[:, 1] > last))
        valid = rs[:, None] | cs[None, :]
    elif family == "horizontal_mirror":
        valid = np.broadcast_to(rs[:, None], (len(rows), len(cols)))
    elif family == "vertical_mirror":
        valid = np.broadcast_to(cs[None, :], (len(rows), len(cols)))
    elif family in ("both_axis_mirror", "rotational_180", "full_square_symmetry"):
        valid = rs[:, None] | cs[None, :]
    else:
        comparison = cols if family == "main_diagonal_mirror" else np.column_stack((w - cols[:, 1], w - cols[:, 0]))
        valid = ((rows[:, 1, None] <= comparison[None, :, 0]) |
                 (comparison[None, :, 1] <= rows[:, 0, None]))
    ri, ci = np.where(valid)
    return np.column_stack((rows[ri, 0], cols[ci, 0], rows[ri, 1], cols[ci, 1]))


def _rectangle_mask(rng, h, w, family, th, tw, minimum, difficulty):
    boxes = _rectangles(h, w, family, th, tw, minimum)
    if not len(boxes):
        raise ValueError("No recoverable rectangle for this fixed geometry")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    maximum = int(areas.max())
    low, high = {"small": (0., .15), "medium": (.15, .60), "large": (.60, 1.)}[difficulty]
    lo, hi = max(minimum ** 2, int(np.ceil(low * maximum))), max(minimum ** 2, int(high * maximum))
    choices = np.flatnonzero((areas >= lo) & (areas <= max(lo, hi)))
    if not len(choices):
        choices = np.flatnonzero(areas == maximum)
    r, c, b, d = map(int, boxes[int(rng.choice(choices))])
    mask = np.zeros((h, w), dtype=bool)
    mask[r:b, c:d] = True
    return mask, [r, c, b - r, d - c], maximum


def _content(rng, h, w, groups, labels, palette, family):
    representatives = np.asarray([g[0] for g in groups])
    if family == "mosaic":
        order = rng.permutation(len(groups))
    else:
        cell_order = _cell_order(rng, h, w, family)
        rank = np.empty(h * w, dtype=int)
        rank[cell_order] = np.arange(h * w)
        order = np.argsort(rank[representatives])
    n = len(palette)
    weights = rng.uniform(.6, 1.8, size=n)
    raw = (len(groups) - n) * weights / weights.sum()
    amounts = np.floor(raw).astype(int) + 1
    for j in np.argsort(-(raw - np.floor(raw)))[:len(groups) - int(amounts.sum())]:
        amounts[j] += 1
    values = np.empty(len(groups), dtype=np.uint8)
    offset = 0
    for color, count in zip(palette, amounts):
        values[order[offset:offset + count]] = color
        offset += count
    return values[labels]


def reconstruct(grid, family, mask_color, tile_h=None, tile_w=None, *, patch=False):
    """An independent equality-class proof from the final visible input.

    Coordinate formulae identify classes without using construction orbits,
    chosen mask bounds, rendering state, or planned output values.
    """
    a = np.asarray(grid)
    h, w = a.shape
    r, c = np.indices((h, w))
    if family in SQUARE and h != w:
        raise ValueError("A diagonal/square pattern needs a square grid")
    if family == "periodic":
        if not tile_h or not tile_w or tile_h > h or tile_w > w:
            raise ValueError("Invalid periodic dimensions")
        key = (r % tile_h) * tile_w + c % tile_w
    elif family == "horizontal_mirror":
        key = np.minimum(r, h - 1 - r) * w + c
    elif family == "vertical_mirror":
        key = r * w + np.minimum(c, w - 1 - c)
    elif family == "both_axis_mirror":
        key = np.minimum(r, h - 1 - r) * w + np.minimum(c, w - 1 - c)
    elif family == "rotational_180":
        key = np.minimum(r * w + c, (h - 1 - r) * w + w - 1 - c)
    elif family == "main_diagonal_mirror":
        key = np.minimum(r * w + c, c * w + r)
    elif family == "anti_diagonal_mirror":
        key = np.minimum(r * w + c, (h - 1 - c) * w + w - 1 - r)
    elif family == "full_square_symmetry":
        x, y = np.minimum(r, h - 1 - r), np.minimum(c, w - 1 - c)
        key = np.minimum(x, y) * w + np.maximum(x, y)
    else:
        raise ValueError("Unknown pattern family")
    hidden = a == mask_color
    if not hidden.any():
        raise ValueError("The mask is absent")
    values = {}
    for k, color in zip(key[~hidden], a[~hidden]):
        k, color = int(k), color.item()
        if k in values and values[k] != color:
            raise ValueError("Visible cells contradict the given pattern")
        values[k] = color
    if any(int(k) not in values for k in np.unique(key[hidden])):
        raise ValueError("Mask hides an entire class; the answer is not unique")
    result = np.asarray([values[int(k)] for k in key.ravel()], dtype=a.dtype).reshape(h, w)
    if patch:
        yy, xx = np.where(hidden)
        r0, r1, c0, c1 = int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1
        if min(r1 - r0, c1 - c0) < 2 or not hidden[r0:r1, c0:c1].all():
            raise ValueError("Occluder is not one solid rectangle with both sides at least two")
        return result[r0:r1, c0:c1]
    return result


def validate_example(ex, task):
    m = ex["metadata"]
    answer = reconstruct(ex["input"], m["pattern_family"], m["mask_color" if task == "32" else "occluder_color"],
                         m.get("tile_h"), m.get("tile_w"), patch=task == "33")
    if answer.tolist() != ex["output"]:
        raise ValueError("Pattern reconstruction differs from output")
    if len(np.unique(ex["input"])) != m["n_colors"]:
        raise ValueError("Pattern palette budget was lost")


def generate_pattern(rng, task):
    family = str(rng.choice(FAMILIES, p=WEIGHTS))
    sizes = feasible_sizes(task, family)
    h, w = sizes[int(rng.integers(len(sizes)))]
    th = tw = 0
    if family == "periodic":
        periods = [(a, b) for a in range(1, min(8, h) + 1) for b in range(1, min(8, w) + 1)
                   if a * b >= 2 and ((h >= a + (2 if task == "33" else 1)) or (w >= b + (2 if task == "33" else 1)))]
        th, tw = periods[int(rng.integers(len(periods)))]
    labels, groups = _classes(h, w, family, th, tw)
    n_colors = int(rng.integers(3, min(10, len(groups) + 1) + 1))
    palette = [int(c) for c in rng.permutation(10)[:n_colors]]
    mask_color = palette[-1]
    content = str(rng.choice(CONTENT))
    full = _content(rng, h, w, groups, labels, palette[:-1], content)
    difficulty = str(rng.choice(("small", "medium", "large"), p=(.20, .45, .35)))
    style = "rectangle" if task == "33" else str(rng.choice(
        ("rectangle", "patches", "bands", "diamonds", "scatter", "combined_symmetry"), p=(.24, .24, .14, .14, .12, .12)))
    box = None
    if style == "rectangle":
        mask, box, max_hidden = _rectangle_mask(rng, h, w, family, th, tw, 2 if task == "33" else 1, difficulty)
    else:
        order_family = {"patches": "rectangular_patches", "bands": "bands", "diamonds": "diamonds",
                        "scatter": "scattered_cells", "combined_symmetry": "scattered_cells"}[style]
        order = _cell_order(rng, h, w, order_family)
        max_hidden = h * w - len(groups)
        low, high = {"small": (.05, .2), "medium": (.2, .65), "large": (.65, 1.)}[difficulty]
        target = max(1, int(round(max_hidden * float(rng.uniform(low, high)))))
        remaining = np.asarray([len(g) for g in groups])
        mask = np.zeros(h * w, dtype=bool)
        # Every class retains visible evidence. For combined-symmetry masks,
        # retaining one random member per orbit makes subgroup shortcuts rare.
        if style == "combined_symmetry" and family in ("both_axis_mirror", "full_square_symmetry"):
            mask[:] = True
            for group in groups:
                mask[int(rng.choice(group))] = False
        else:
            hidden = 0
            for position in order:
                group = labels.ravel()[position]
                if remaining[group] > 1:
                    mask[position] = True
                    remaining[group] -= 1
                    hidden += 1
                    if hidden == target:
                        break
        mask = mask.reshape(h, w)
    a = full.copy()
    a[mask] = mask_color
    if task == "33":
        r, c, bh, bw = box
        out = full[r:r + bh, c:c + bw]
    else:
        out = full
    m = {"pattern_family": family, "symmetry_type": "none" if family == "periodic" else family,
         "n_colors": n_colors, "pattern_content": content, "mask_style": style,
         "mask_difficulty": difficulty, "masked_cells": int(mask.sum()),
         "masked_fraction": float(mask.mean()), "max_mask_cells": max_hidden, "attempts": 1,
         "mask_color" if task == "32" else "occluder_color": mask_color}
    if family == "periodic":
        m.update(tile_h=th, tile_w=tw)
    if task == "33":
        m.update(occluder_height=box[2], occluder_width=box[3])
    ex = {"input": a.tolist(), "output": out.tolist(), "metadata": m}
    validate_example(ex, task)
    return ex
