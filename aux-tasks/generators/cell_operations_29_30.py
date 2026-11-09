"""Literal colour operations on varied, exactly budgeted geometric canvases.

These tasks count/recolour *every* matching pixel.  Patterns, individual cells,
and filled shapes therefore all have the same semantics; no background needs
to be identified.  Geometry, palette and count budgets are chosen before the
canvas is drawn.  Every requested size is supported without rejection.
"""
from __future__ import annotations

import numpy as np

from common.backgrounds import apply_background, SOLID_KINDS


CONTENT_FAMILIES = (
    "rectangular_patches", "bands", "diagonal_bands", "concentric_frames",
    "diamonds", "block_mosaic", "letters", "crosses", "staircases",
    "ellipses", "lattice", "scattered_cells",
)
_FAMILY_WEIGHTS = np.array([14, 10, 8, 8, 8, 12, 10, 6, 6, 8, 7, 3], dtype=float)
_FAMILY_WEIGHTS /= _FAMILY_WEIGHTS.sum()


def _axis_view(rng, h, w):
    r, c = np.indices((h, w))
    if rng.random() < .5:
        r = h - 1 - r
    if rng.random() < .5:
        c = w - 1 - c
    return (c, r, w, h) if rng.random() < .5 else (r, c, h, w)


def _patch_order(rng, h, w):
    """Random guillotine subdivisions: large coherent blocks, not pixel noise."""
    regions = [(0, 0, h, w)]
    target = int(rng.integers(3, min(15, h * w) + 1)) if h * w >= 3 else h * w
    while len(regions) < target:
        eligible = [i for i, (_, _, a, b) in enumerate(regions) if max(a, b) >= 2]
        if not eligible:
            break
        weights = np.array([regions[i][2] * regions[i][3] for i in eligible], float)
        i = int(rng.choice(eligible, p=weights / weights.sum()))
        r, c, a, b = regions.pop(i)
        vertical = b > 1 and (a == 1 or rng.random() < .5)
        if vertical:
            cut = int(rng.integers(1, b))
            regions.extend([(r, c, a, cut), (r, c + cut, a, b - cut)])
        else:
            cut = int(rng.integers(1, a))
            regions.extend([(r, c, cut, b), (r + cut, c, a - cut, b)])
    rng.shuffle(regions)
    order = []
    for r, c, a, b in regions:
        rr, cc = _axis_view(rng, a, b)[:2]
        local = np.argsort((rr * (max(a, b) + 1) + cc).ravel(), kind="stable")
        order.extend((r + int(i) // b) * w + c + int(i) % b for i in local)
    return np.asarray(order, dtype=int)


def _cell_order(rng, h, w, family):
    """Order cells by drawn geometry; a prefix can realize any feasible count.

    A final partially filled region supplies the exact integer remainder.  This
    retains arbitrary counts without peppering a finished image with pixels.
    """
    if family == "rectangular_patches":
        return _patch_order(rng, h, w)
    if family == "scattered_cells":
        return rng.permutation(h * w)
    r, c, a, b = _axis_view(rng, h, w)
    tie = r * (b + 1) + c
    if family == "bands":
        thick = int(rng.integers(1, max(2, a // 5 + 1)))
        period = int(rng.integers(2, 5))
        rank = (r // thick + int(rng.integers(period))) % period
    elif family == "diagonal_bands":
        thick = int(rng.integers(1, max(2, min(a, b) // 5 + 1)))
        period = int(rng.integers(2, 5))
        rank = ((r + c) // thick + int(rng.integers(period))) % period
    elif family == "concentric_frames":
        depth = np.minimum(np.minimum(r, a - 1 - r), np.minimum(c, b - 1 - c))
        thick = int(rng.integers(1, max(2, min(a, b) // 7 + 1)))
        rank = (depth // thick + int(rng.integers(3))) % int(rng.integers(2, 4))
    elif family == "block_mosaic":
        bh = int(rng.integers(1, max(2, a // 3 + 1)))
        bw = int(rng.integers(1, max(2, b // 3 + 1)))
        nr, nc = (a + bh - 1) // bh, (b + bw - 1) // bw
        ranks = rng.permutation(nr * nc).reshape(nr, nc)
        rank = ranks[r // bh, c // bw]
    elif family == "lattice":
        sy, sx = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        rank = (r % sy) * sx + c % sx
    elif family in ("letters", "crosses", "staircases"):
        # Repeated broad glyphs; holes/notches are deliberate rectangles.
        bh, bw = max(3, a // int(rng.integers(1, 4))), max(3, b // int(rng.integers(1, 4)))
        y, x = r % bh, c % bw
        ty, tx = max(1, bh // 3), max(1, bw // 3)
        if family == "crosses":
            mask = ((y >= (bh - ty) // 2) & (y < (bh + ty) // 2)) | ((x >= (bw - tx) // 2) & (x < (bw + tx) // 2))
        elif family == "staircases":
            mask = x * bh <= (y + 1) * bw
        else:
            kind = int(rng.integers(4))
            if kind == 0:  # L
                mask = (x < tx) | (y >= bh - ty)
            elif kind == 1:  # U
                mask = (x < tx) | (x >= bw - tx) | (y >= bh - ty)
            elif kind == 2:  # T
                mask = (y < ty) | ((x >= (bw - tx) // 2) & (x < (bw + tx) // 2))
            else:  # H
                mask = (x < tx) | (x >= bw - tx) | ((y >= (bh - ty) // 2) & (y < (bh + ty) // 2))
        tile = (r // bh) * ((b + bw - 1) // bw) + c // bw
        rank = (~mask).astype(int) * (h * w + 1) + tile
    else:
        # Several independent geometric bodies prevent a single bounding-box
        # measurement from replacing counting on the main distribution.
        count = int(rng.integers(1, min(6, max(2, h * w // 45)) + 1))
        rank = np.full((h, w), np.inf)
        for _ in range(count):
            cy, cx = float(rng.uniform(0, max(1, a - 1))), float(rng.uniform(0, max(1, b - 1)))
            sy, sx = float(rng.uniform(.6, 1.6)), float(rng.uniform(.6, 1.6))
            dy, dx = np.abs(r - cy) / sy, np.abs(c - cx) / sx
            d = dy + dx if family == "diamonds" else dy ** 2 + dx ** 2
            rank = np.minimum(rank, d)
    return np.lexsort((tie.ravel(), np.asarray(rank).ravel()))


def _fill_other_colors(rng, grid, query_mask, colors, *, distinct_quotas=False):
    """Use sizable contiguous regions for the remaining palette budget."""
    h, w = grid.shape
    rest = int((~query_mask).sum())
    if not colors:
        return "uniform", {}
    family = str(rng.choice(("rectangular_patches", "bands", "block_mosaic", "diagonal_bands", "concentric_frames")))
    order = _cell_order(rng, h, w, family)
    order = order[~query_mask.ravel()[order]]
    n = len(colors)
    if rest < n:
        raise ValueError("palette cannot fit the non-query cells")
    weights = rng.lognormal(0., .85, size=n)
    if n > 1 and rng.random() < .65:
        weights[0] = weights[1:].sum() * float(rng.uniform(1.2, 2.5))
    if distinct_quotas:
        # Sorted increments added to 1..n give strictly different quotas.
        # Reserve their minimum footprint before sampling the primary count.
        weights.sort()
        extra = rest - n * (n + 1) // 2
        increments = np.floor(extra * weights / weights.sum()).astype(int)
        amounts = np.arange(1, n + 1) + increments
        remainder = rest - int(amounts.sum())
        if remainder:
            amounts[-remainder:] += 1
        rng.shuffle(amounts)
    else:
        raw = (rest - n) * weights / weights.sum()
        amounts = np.floor(raw).astype(int) + 1
        for j in np.argsort(-(raw - np.floor(raw)))[:rest - int(amounts.sum())]:
            amounts[j] += 1
    # A library pattern can guide the layout, but may not change the colour
    # quotas.  Using its layers as a stable ordering retains arbitrary exact
    # counts without repairing a finished scene with scattered extra pixels.
    if min(h, w) >= 8 and n >= 2 and rng.random() < .4:
        kind = str(rng.choice(SOLID_KINDS))
        layout = grid.copy()
        layout[~query_mask] = colors[0]
        painted = apply_background(rng, layout, colors[0], query_mask, kind=kind,
                                   colors=colors[1:min(n, 4)], exclude=[])
        if painted is not None:
            layers = np.zeros(grid.size, dtype=int)
            for i, color in enumerate(colors[1:min(n, 4)], start=1):
                layers[painted[0].ravel() == color] = i
            order = order[np.argsort(layers[order], kind="stable")]
            family += "+" + kind
    off = 0
    for col, amount in zip(colors, amounts):
        grid.ravel()[order[off:off + int(amount)]] = col
        off += int(amount)
    return family, {color: int(amount) for color, amount in zip(colors, amounts)}


def _scene(rng, *, zero_probability):
    h, w = (int(x) for x in rng.integers(1, 31, size=2))
    area = h * w
    query = int(rng.integers(10))
    absent = bool(rng.random() < zero_probability)
    uniform = area == 1 or rng.random() < .025
    max_palette = min(area, 9 if absent else 10)
    palette_size = 1 if uniform else int(rng.integers(2, max_palette + 1))
    available = np.array([c for c in range(10) if c != query])
    rng.shuffle(available)
    other_colors = [int(c) for c in available[:palette_size - (not absent)]]
    mode = "zero" if absent else "uniform" if palette_size == 1 else str(rng.choice(
        ("small", "sparse", "medium", "dense", "any"), p=(.20, .20, .30, .20, .10)))
    n_other = len(other_colors)
    distinct_minimum = n_other * (n_other + 1) // 2
    distinct_quotas = bool(distinct_minimum <= area - (not absent) and rng.random() < .85)
    maximum = area - (distinct_minimum if distinct_quotas else n_other)
    if absent:
        count = 0
    elif palette_size == 1:
        count = area
    elif mode == "small":
        count = int(rng.integers(1, min(16, maximum) + 1))
    elif mode == "any":
        count = int(rng.integers(1, maximum + 1))
    else:
        low, high = {"sparse": (.02, .18), "medium": (.18, .65), "dense": (.65, 1.)}[mode]
        count = min(maximum, max(1, int(round(float(rng.uniform(low, high)) * maximum))))
    family = str(rng.choice(CONTENT_FAMILIES, p=_FAMILY_WEIGHTS))
    order = _cell_order(rng, h, w, family)
    mask = np.zeros((h, w), dtype=bool)
    mask.ravel()[order[:count]] = True
    grid = np.full((h, w), query, dtype=int)
    fill, planned_counts = _fill_other_colors(rng, grid, mask, other_colors, distinct_quotas=distinct_quotas)
    planned_counts[query] = count
    if int(np.count_nonzero(grid == query)) != count or len(np.unique(grid)) != palette_size:
        raise ValueError("drawn scene violates its count/palette plan")
    if any(int(np.count_nonzero(grid == color)) != amount for color, amount in planned_counts.items()):
        raise ValueError("drawn palette regions violate their planned counts")
    primary = query
    # Query choice is independent of region size and visual role.  All counts
    # were planned before painting; selecting among them does not privilege
    # the conspicuous primary shape or the dominant fill.
    if not absent:
        query = int(rng.choice([primary, *other_colors]))
        count = planned_counts[query]
    counts = np.bincount(grid.ravel(), minlength=10)
    ground_candidates = np.flatnonzero(counts == counts.max())
    ground = int(rng.choice(ground_candidates))
    return grid, query, {
        "count": count, "count_mode": mode if query == primary else "palette_region",
        "scene_count_mode": mode, "query_is_primary": query == primary,
        "distinct_secondary_quotas": distinct_quotas,
        "query_selection": "absent" if absent else "uniform_present",
        "content_family": family,
        "fill_family": fill, "palette_size": palette_size, "background_color": ground,
        "attempts": 1,
    }


def validate_recolor(grid, output, source, target):
    """A second, literal scalar implementation, independent of NumPy stamping."""
    if len(grid) != len(output) or any(len(a) != len(b) for a, b in zip(grid, output)):
        raise ValueError("recolour output dimensions changed")
    for before, after in zip(grid, output):
        for a, b in zip(before, after):
            if b != (target if a == source else a):
                raise ValueError("recolour output differs from the visible-input rule")


def validate_count(grid, output, query):
    count = sum(1 for row in grid for value in row if value == query)
    if str(count) != output:
        raise ValueError("count differs from the visible input")


def generate_recolor(rng):
    grid, source, meta = _scene(rng, zero_probability=.08)
    present = [int(c) for c in np.unique(grid) if c != source]
    absent = [c for c in range(10) if c != source and c not in present]
    draw = float(rng.random())
    if draw < .25 and meta["background_color"] != source:
        target = meta["background_color"]
    elif draw < .65 and present:
        target = int(rng.choice(present))
    else:
        target = int(rng.choice(absent or present))
    output = grid.copy()
    output[grid == source] = target
    inp, out = grid.tolist(), output.tolist()
    validate_recolor(inp, out, source, target)
    meta.update(source_color=source, target_color=target, changed_cells=meta.pop("count"))
    return {"input": inp, "output": out, "metadata": meta}


def generate_count(rng):
    grid, query, meta = _scene(rng, zero_probability=.06)
    inp = grid.tolist()
    output = str(meta["count"])
    validate_count(inp, output, query)
    # Background is a diagnostic only, never a task given.
    meta.update(query_color=query)
    return {"input": inp, "output": output, "metadata": meta}
