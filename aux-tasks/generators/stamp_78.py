"""Opaque placement of a shown stamp at explicitly named marker cells.

Sizes, drawing, palette and composition are sampled once.  A freely placed,
disjoint collection of visible rectangles witnesses the available capacity;
the positive copy count is then drawn inside that capacity.  This needs no
scene retries, token-budget limits, regular placement slots or fallback image.
"""
from __future__ import annotations

import numpy as np

from generators.grid_content_66_70 import _partition, _regions
from generators.scene_shapes_56_60 import shape as familiar_shape


COMPOSITIONS = ("single", "few", "many")
CANVAS_STYLES = ("solid", "bands", "checks", "patchwork", "nested_frames")
MAX_WITNESSES = 14


def _odd_side(rng, side, composition):
    ratios = {"single": (.50, .90), "few": (.22, .52), "many": (.10, .30)}
    low, high = ratios[composition]
    lo = max(3, int(np.ceil(side * low)))
    hi = max(lo, int(np.floor(side * high)))
    hi = min(side, hi)
    choices = [n for n in range(3, side + 1, 2) if lo <= n <= hi]
    if not choices:
        # An interval shorter than two may lie between consecutive odd sides.
        # Its nearest feasible odd side is a geometric rounding, not a retry.
        choices = [min(range(3, side + 1, 2), key=lambda n: abs(n - (lo + hi) / 2))]
    return int(rng.choice(choices))


def _motif(rng, sh, sw):
    bh = int(rng.integers(max(2, int(np.ceil(sh * .78))), sh + 1))
    bw = int(rng.integers(max(2, int(np.ceil(sw * .78))), sw + 1))
    kinds = ["rectangle", "bar", "frame", "staircase"]
    if max(bh, bw) <= 2.5 * min(bh, bw):
        kinds += ["L", "T", "U", "H", "plus"]
    if min(bh, bw) >= 5:
        kinds.append("chamfer")
        if max(bh, bw) <= 2 * min(bh, bw):
            kinds.append("diamond")
    kind = str(rng.choice(kinds))
    if kind == "rectangle" and (bh, bw) == (sh, sw):
        # A stamp carries a drawn motif and visible surrounding/background cells.
        # Preserve one complete margin instead of repairing a uniform stamp.
        if rng.integers(2):
            bh -= 1
        else:
            bw -= 1
    if kind == "bar":
        # A broad stamp should carry a readable bar, not one ink row inside
        # twenty rows of opaque padding.  Thin pixel bars remain on small stamps.
        if bh > bw or (bh == bw and rng.integers(2)):
            bw = max(1, min(bh // 3, int(round(bw * rng.uniform(.25, .45)))))
        else:
            bh = max(1, min(bw // 3, int(round(bh * rng.uniform(.25, .45)))))
        mask, actual = np.ones((bh, bw), dtype=bool), kind
    else:
        mask, actual = familiar_shape(rng, bh, bw, kind)
    if actual != kind:
        raise RuntimeError("Task 78: motif family changed")
    bh, bw = mask.shape
    r = int(rng.integers(sh - bh + 1))
    c = int(rng.integers(sw - bw + 1))
    whole = np.zeros((sh, sw), dtype=bool)
    whole[r:r + bh, c:c + bw] = mask
    return whole, {"motif_family": kind, "motif_bounds": [r, c, bh, bw]}


def _canvas_capacity(h, w, style):
    if style == "solid":
        return 1
    if style == "bands":
        return min(6, max(h, w) // 2)
    if style == "checks":
        return min(6, h // 2 + w // 2 - 1)
    if style == "patchwork":
        return min(6, (h // 2) * (w // 2))
    return min(6, (min(h, w) + 1) // 2 - 1)


def _plan(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    composition = str(rng.choice(COMPOSITIONS, p=(.24, .56, .20)))
    sh, sw = (_odd_side(rng, s, composition) for s in (h, w))
    motif, info = _motif(rng, sh, sw)
    style = str(rng.choice(CANVAS_STYLES, p=(.55, .15, .10, .12, .08)))
    inks = int(rng.choice((1, 2, 3), p=(.55, .30, .15)))
    capacity = _canvas_capacity(h, w, style)
    same_base = bool(rng.random() < .55)
    # The marker is reserved and each motif has only 1–3 coherent ink regions.
    # Richness on the canvas comes from bands/patches, never palette speckles.
    capacity = min(capacity, 9 - (inks + 1) + int(same_base))
    if style == "solid":
        n_canvas = 1
    else:
        weights = np.asarray((.38, .23, .18, .13, .08))[:capacity - 1]
        n_canvas = int(rng.choice(np.arange(2, capacity + 1), p=weights / weights.sum()))
    colors = list(map(int, rng.permutation(10)))
    marker = colors.pop()
    stamp_colors = [colors.pop() for _ in range(inks + 1)]
    canvas_colors = [stamp_colors[0] if same_base else colors.pop()]
    canvas_colors.extend(colors.pop() for _ in range(n_canvas - 1))
    return {
        "h": h, "w": w, "stamp_h": sh, "stamp_w": sw,
        "composition": composition, "canvas_style": style,
        "layout": "complete" if rng.random() < .55 else "clipped",
        "marker_color": marker, "stamp_palette": stamp_colors,
        "canvas_palette": canvas_colors, "canvas_palette_capacity": capacity,
        "motif": motif, **info,
    }


def _draw_stamp(rng, plan):
    mask, colors = plan["motif"], plan["stamp_palette"]
    stamp = np.full(mask.shape, colors[0], dtype=int)
    r, c, h, w = plan["motif_bounds"]
    eligible = [axis for axis, side in enumerate((h, w)) if side >= len(colors) - 1]
    axis = int(rng.choice(eligible))
    widths = _partition(rng, (h, w)[axis], len(colors) - 1)
    strip = np.repeat(np.asarray(colors[1:]), widths)
    fill = np.broadcast_to(strip[:, None] if axis == 0 else strip[None, :], (h, w))
    local = stamp[r:r + h, c:c + w]
    local[mask[r:r + h, c:c + w]] = fill[mask[r:r + h, c:c + w]]
    return stamp, {"motif_fill_axis": axis, "motif_band_widths": widths.tolist()}


def _draw_canvas(rng, plan):
    h, w, palette = plan["h"], plan["w"], plan["canvas_palette"]
    style = plan["canvas_style"]
    if style == "solid":
        return np.full((h, w), palette[0], dtype=int), {}
    if style == "bands":
        choices = [axis for axis, side in enumerate((h, w)) if side >= 2 * len(palette)]
        axis = int(rng.choice(choices))
        length = (h, w)[axis]
        n_bands = int(rng.integers(len(palette), min(length // 2, len(palette) + 3) + 1))
        widths = _partition(rng, length, n_bands, minimum=2)
        strip = np.repeat(np.asarray(palette)[np.arange(n_bands) % len(palette)], widths)
        return (np.broadcast_to(strip[:, None] if axis == 0 else strip[None, :], (h, w)).copy(),
                {"canvas_band_axis": axis, "canvas_band_widths": widths.tolist()})
    if style == "checks":
        options = [(nr, nc) for nr in range(2, min(h // 2, 6) + 1)
                   for nc in range(2, min(w // 2, 6) + 1) if nr + nc - 1 >= len(palette)]
        nr, nc = options[int(rng.integers(len(options)))]
        rows, cols = _partition(rng, h, nr, minimum=2), _partition(rng, w, nc, minimum=2)
        layer = np.repeat(np.arange(nr), rows)[:, None] + np.repeat(np.arange(nc), cols)[None, :]
        return (np.asarray(palette)[layer % len(palette)],
                {"canvas_row_widths": rows.tolist(), "canvas_col_widths": cols.tolist()})
    if style == "patchwork":
        count = int(rng.integers(len(palette), min((h // 2) * (w // 2), len(palette) + 3) + 1))
        regions = _regions(rng, h, w, count, minimum=2)
        canvas = np.empty((h, w), dtype=int)
        for i, (r, c, rh, cw) in enumerate(regions):
            canvas[r:r + rh, c:c + cw] = palette[i % len(palette)]
        return canvas, {"canvas_patches": [list(region) for region in regions]}
    yy, xx = np.indices((h, w))
    depth = np.minimum.reduce((yy, xx, h - 1 - yy, w - 1 - xx))
    widths = _partition(rng, int(depth.max()), len(palette))
    widths[-1] += 1  # The innermost color owns at least a 3x3 (or 4x4) block.
    bands = np.repeat(np.asarray(palette), widths)
    return bands[depth], {"canvas_ring_widths": widths.tolist()}


def _placement_witness(rng, plan):
    h, w, sh, sw = (plan[key] for key in ("h", "w", "stamp_h", "stamp_w"))
    rh, cw = sh // 2, sw // 2
    motif = plan["motif"]
    minimum_visible = max(1, int(np.ceil(int(motif.sum()) * .12)))
    candidates = []
    for r in range(h):
        for c in range(w):
            r0, r1 = max(0, r - rh), min(h, r + rh + 1)
            c0, c1 = max(0, c - cw), min(w, c + cw + 1)
            pr, pc = r0 - r + rh, c0 - c + cw
            visible = int(motif[pr:pr + r1 - r0, pc:pc + c1 - c0].sum())
            if visible < minimum_visible:
                continue
            clipped = (r1 - r0 != sh) or (c1 - c0 != sw)
            if plan["layout"] == "complete" and clipped:
                continue
            candidates.append((r, c, r0, c0, r1, c1, clipped))
    if not candidates:
        raise RuntimeError("Task 78: no placement witnesses for the fixed plan")
    order = list(map(int, rng.permutation(len(candidates))))
    if plan["layout"] == "clipped":
        border = [i for i, x in enumerate(candidates) if x[-1]]
        if not border:
            raise RuntimeError("Task 78: clipped plan has no visible boundary witness")
        first = int(rng.choice(border))
        order.remove(first)
        order.insert(0, first)
    occupied = np.zeros((h, w), dtype=bool)
    witness = []
    examined = 0
    for i in order:
        examined += 1
        r, c, r0, c0, r1, c1, _ = candidates[i]
        if np.any(occupied[r0:r1, c0:c1]):
            continue
        occupied[r0:r1, c0:c1] = True
        witness.append((r, c))
        if len(witness) == MAX_WITNESSES:
            break
    return witness, minimum_visible, len(candidates), examined


def _count(rng, capacity, composition):
    if composition == "single":
        return 1
    if composition == "few":
        cap = min(5, capacity)
        weights = np.asarray((.12, .34, .28, .18, .08))[:cap]
        return int(rng.choice(np.arange(1, cap + 1), p=weights / weights.sum()))
    return int(rng.integers(min(4, capacity), capacity + 1))


def solve_stamp(stamp, canvas, marker_color):
    """Literal reader of the two visible grids and the named marker color.

    This intentionally uses destination-cell loops, not the builder's slices,
    and discovers centers by scanning the final input canvas, not metadata.
    """
    stamp, canvas = np.asarray(stamp), np.asarray(canvas)
    if stamp.ndim != 2 or canvas.ndim != 2 or min(stamp.shape + canvas.shape) < 1:
        raise ValueError("Task 78: stamp and canvas must be rectangular grids")
    sh, sw = stamp.shape
    if sh % 2 != 1 or sw % 2 != 1:
        raise ValueError("Task 78: stamp has no unique center cell")
    centers = [(r, c) for r, row in enumerate(canvas) for c, value in enumerate(row) if value == marker_color]
    if not centers:
        raise ValueError("Task 78: no named marker is shown")
    result = canvas.copy()
    for r, row in enumerate(canvas):
        for c, _ in enumerate(row):
            covering = [(r - mr + sh // 2, c - mc + sw // 2) for mr, mc in centers
                        if abs(r - mr) <= sh // 2 and abs(c - mc) <= sw // 2]
            if len(covering) > 1:
                raise ValueError("Task 78: stamp footprints overlap")
            if covering:
                result[r, c] = stamp[covering[0]]
    return result


def _visible_stats(stamp, canvas, output, marker_color, stamp_bg, canvas_bg):
    sh, sw = stamp.shape
    footprint = np.zeros(canvas.shape, dtype=bool)
    centers, visible, clipped, clipped_axes = [], [], [], []
    for rr, cc in np.argwhere(canvas == marker_color):
        r, c = int(rr), int(cc)
        centers.append([r, c])
        r0, r1 = max(0, r - sh // 2), min(len(canvas), r + sh // 2 + 1)
        c0, c1 = max(0, c - sw // 2), min(len(canvas[0]), c + sw // 2 + 1)
        pr, pc = r0 - r + sh // 2, c0 - c + sw // 2
        visible.append(int(np.count_nonzero(stamp[pr:pr + r1 - r0, pc:pc + c1 - c0] != stamp_bg)))
        axes = int(r1 - r0 != sh) + int(c1 - c0 != sw)
        clipped_axes.append(axes)
        clipped.append(bool(axes))
        footprint[r0:r1, c0:c1] = True
    transparent = canvas.copy()
    for r, c in centers:
        for a, b in np.argwhere(stamp != stamp_bg):
            rr, cc = r + int(a) - sh // 2, c + int(b) - sw // 2
            if 0 <= rr < len(canvas) and 0 <= cc < len(canvas[0]):
                transparent[rr, cc] = stamp[a, b]
    outside = ~footprint
    details = (canvas != canvas_bg) & (canvas != marker_color)
    return {
        "stamp_centers": centers, "n_stamps": len(centers),
        "visible_motif_cells_per_copy": visible,
        "n_clipped": sum(clipped), "n_complete": len(centers) - sum(clipped),
        "clipped_axes_per_copy": clipped_axes,
        "footprint_cells": int(footprint.sum()), "visible_motif_cells": sum(visible),
        "outside_footprint_cells": int(outside.sum()),
        "outside_detail_cells": int(np.count_nonzero(outside & details)),
        "covered_detail_cells": int(np.count_nonzero(footprint & details)),
        "opaque_vs_transparent_changes": bool(np.any(transparent != output)),
        "output_changed_cells": int(np.count_nonzero(output != canvas)),
        "outside_preserved": bool(np.array_equal(output[outside], canvas[outside])),
    }


def validate_stamp(ex):
    """Re-solve the visible pair, then verify all recorded design guarantees."""
    stamp = np.asarray(ex["input"]["stamp"])
    canvas = np.asarray(ex["input"]["canvas"])
    output = np.asarray(ex["output"])
    m = ex["metadata"]
    for grid in (stamp, canvas, output):
        if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer) or np.any((grid < 0) | (grid > 9)):
            raise ValueError("Task 78: invalid grid colors or shape")
    if min(canvas.shape) < 5 or max(canvas.shape) > 30 or min(stamp.shape) < 3 or max(stamp.shape) > 29:
        raise ValueError("Task 78: grid dimensions outside the design envelope")
    marker = m["marker_color"]
    expected = solve_stamp(stamp, canvas, marker)
    if not np.array_equal(expected, output):
        raise ValueError("Task 78: output disagrees with visible-grid stamping")
    if marker in stamp or marker in output:
        raise ValueError("Task 78: reserved marker color in the stamp or output")
    stats = _visible_stats(stamp, canvas, output, marker, m["stamp_background_color"], m["canvas_background_color"])
    if any(m[key] != value for key, value in stats.items()):
        raise ValueError("Task 78: recorded placement statistics disagree with the visible grids")
    if m["grid_h"] != len(canvas) or m["grid_w"] != len(canvas[0]) or m["stamp_h"] != len(stamp) or m["stamp_w"] != len(stamp[0]):
        raise ValueError("Task 78: recorded dimensions disagree")
    if m["planned_n_stamps"] != stats["n_stamps"] or not 1 <= stats["n_stamps"] <= m["witnessed_capacity"] <= MAX_WITNESSES:
        raise ValueError("Task 78: copy count does not realize the placement plan")
    if min(stats["visible_motif_cells_per_copy"]) < m["minimum_visible_motif_cells"]:
        raise ValueError("Task 78: a clipped stamp loses its planned visible motif")
    if (m["layout_mode"] == "complete") != (stats["n_clipped"] == 0) or m["planned_layout"] != m["layout_mode"]:
        raise ValueError("Task 78: placement changed the planned clipping mode")
    if set(map(int, np.unique(stamp))) != set(m["stamp_palette"]) or set(map(int, np.unique(canvas))) != set(m["canvas_palette"]) | {marker}:
        raise ValueError("Task 78: drawing lost a planned palette color")
    combined = set(map(int, np.unique(stamp))) | set(map(int, np.unique(canvas)))
    if len(combined) != m["n_colors"] or set(m["palette"]) != combined:
        raise ValueError("Task 78: combined palette statistics disagree")
    if np.count_nonzero(stamp != m["stamp_background_color"]) != m["motif_cells"]:
        raise ValueError("Task 78: motif area statistics disagree")
    if m["motif_family"] != m["planned_motif_family"] or m["background_style"] != m["planned_background_style"]:
        raise ValueError("Task 78: drawing changed a planned family")
    if ex["input"]["stamp"] != m["stamp_pattern"]:
        raise ValueError("Task 78: recorded stamp differs from the shown stamp")


def generate_stamp(rng):
    plan = _plan(rng)
    stamp, fill_info = _draw_stamp(rng, plan)
    canvas, canvas_info = _draw_canvas(rng, plan)
    witness, minimum_visible, candidates, examined = _placement_witness(rng, plan)
    count = _count(rng, len(witness), plan["composition"])
    # The first witness carries the clipping guarantee.  Other placements are
    # selected freely from the same witnessed set; no regular slots are drawn.
    if plan["layout"] == "clipped":
        chosen = [witness[0]]
        if count > 1:
            chosen.extend(witness[i] for i in rng.choice(np.arange(1, len(witness)), size=count - 1, replace=False))
    else:
        chosen = [witness[i] for i in rng.choice(len(witness), size=count, replace=False)]
    marker = plan["marker_color"]
    for r, c in chosen:
        canvas[r, c] = marker
    output = canvas.copy()
    sh, sw = stamp.shape
    for r, c in chosen:
        r0, c0 = max(0, r - sh // 2), max(0, c - sw // 2)
        r1, c1 = min(len(canvas), r + sh // 2 + 1), min(len(canvas[0]), c + sw // 2 + 1)
        sr, sc = r0 - r + sh // 2, c0 - c + sw // 2
        output[r0:r1, c0:c1] = stamp[sr:sr + r1 - r0, sc:sc + c1 - c0]
    combined = sorted(set(map(int, np.unique(stamp))) | set(map(int, np.unique(canvas))))
    stats = _visible_stats(stamp, canvas, output, marker, plan["stamp_palette"][0], plan["canvas_palette"][0])
    metadata = {
        "grid_h": plan["h"], "grid_w": plan["w"], "stamp_h": sh, "stamp_w": sw,
        "marker_color": marker, "stamp_background_color": plan["stamp_palette"][0],
        "canvas_background_color": plan["canvas_palette"][0],
        "stamp_palette": plan["stamp_palette"], "canvas_palette": plan["canvas_palette"],
        "palette": combined, "n_colors": len(combined),
        "canvas_palette_capacity": plan["canvas_palette_capacity"],
        "stamp_pattern": stamp.tolist(), "motif_family": plan["motif_family"],
        "planned_motif_family": plan["motif_family"], "motif_bounds": plan["motif_bounds"],
        "motif_cells": int(plan["motif"].sum()), "composition": plan["composition"],
        "background_style": plan["canvas_style"], "planned_background_style": plan["canvas_style"],
        "layout_mode": plan["layout"], "planned_layout": plan["layout"],
        "planned_n_stamps": count, "witnessed_capacity": len(witness),
        "placement_candidates": candidates, "placement_candidates_examined": examined,
        "minimum_visible_motif_cells": minimum_visible, "attempts": 1,
        **fill_info, **canvas_info, **stats,
    }
    ex = {"input": {"stamp": stamp.tolist(), "canvas": canvas.tolist()},
          "output": output.tolist(), "metadata": metadata}
    validate_stamp(ex)
    return ex
