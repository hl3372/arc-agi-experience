"""Drawn scenes for the literal whole-grid operations in tasks 66 and 70.

Canvas, family, density and palette are fixed before drawing.  Neither answer
parameter participates in the drawing: the diagonal and swap pair are chosen
only after the complete input is available.  Failed square-reflection checks
retry the same scene plan, never a smaller palette or an easier family.
"""
from __future__ import annotations

import numpy as np

from generators.scene_shapes_56_60 import shape as familiar_shape


FAMILIES = ("shapes", "multicolor_shapes", "bands", "checks", "patchwork", "nested_frames")
FAMILY_WEIGHTS = (.24, .24, .12, .12, .14, .14)
PALETTE_WEIGHTS = np.asarray((.18, .14, .12, .11, .10, .10, .09, .08, .08))
MAX_ATTEMPTS = 64


def _partition(rng, length, count, *, minimum=1):
    """Contiguous bands with variable positive widths, not exact color quotas."""
    if count * minimum > length:
        raise ValueError("The requested bands do not fit")
    extra = length - count * minimum
    widths = np.full(count, minimum, dtype=int)
    if extra:
        weights = rng.dirichlet(np.full(count, 1.5))
        widths += rng.multinomial(extra, weights)
    return widths


def _regions(rng, h, w, count, *, minimum=2):
    """Irregular rectangular free areas, with no fixed rows or placement slots."""
    regions = [(0, 0, h, w)]
    while len(regions) < count:
        capacity = sum((a // minimum) * (b // minimum) for _, _, a, b in regions)
        options = {}
        for i, (_, _, a, b) in enumerate(regions):
            own = (a // minimum) * (b // minimum)
            cuts = []
            for vertical, side, other in ((True, b, a), (False, a, b)):
                cuts.extend((vertical, cut) for cut in range(minimum, side - minimum + 1)
                            if capacity - own + (cut // minimum + (side - cut) // minimum) * (other // minimum) >= count)
            if cuts:
                options[i] = cuts
        choices = list(options)
        if not choices:
            raise ValueError("The planned region count does not fit")
        areas = np.asarray([regions[i][2] * regions[i][3] for i in choices], float)
        # Split the larger free spaces most often, while keeping uneven layouts.
        i = int(rng.choice(choices, p=areas ** 2 / np.sum(areas ** 2)))
        r, c, a, b = regions.pop(i)
        cuts = options[i]
        vertical, cut = cuts[int(rng.integers(len(cuts)))]
        if vertical:
            regions.extend(((r, c, a, cut), (r, c + cut, a, b - cut)))
        else:
            regions.extend(((r, c, cut, b), (r + cut, c, a - cut, b)))
    rng.shuffle(regions)
    return regions


def _plan(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    family = str(rng.choice(FAMILIES, p=FAMILY_WEIGHTS))
    # Tiny drawings have a real geometric capacity.  Rich palettes remain
    # available on small patterns without fragmenting individual shapes.
    if family in ("shapes", "multicolor_shapes"):
        capacity = min(10, 1 + h * w // 14)
    elif family == "nested_frames":
        capacity = min(10, (min(h, w) + 1) // 2)
    elif family == "bands":
        capacity = min(10, max(h, w))
    elif family == "checks":
        capacity = min(10, h + w - 1, max(2, h * w // 4 + 1))
    else:
        capacity = min(10, (h // 2) * (w // 2))
    weights = PALETTE_WEIGHTS[:capacity - 1]
    n_colors = int(rng.choice(np.arange(2, capacity + 1), p=weights / weights.sum()))
    palette = list(map(int, rng.choice(10, size=n_colors, replace=False)))
    density = str(rng.choice(("open", "full"))) if family in ("shapes", "multicolor_shapes") else "structured"
    plan = {"h": h, "w": w, "family": family, "palette": palette,
            "density": density, "palette_capacity": capacity}
    if family in ("shapes", "multicolor_shapes"):
        minimum = n_colors - 1 if family == "shapes" else (n_colors + 1) // 3
        maximum = max(minimum, min(8, h * w // 24))
        plan["n_shapes"] = int(rng.integers(minimum, maximum + 1))
    return plan


def _shape_scene(rng, plan):
    h, w, palette = plan["h"], plan["w"], plan["palette"]
    bg, colors = palette[0], palette[1:]
    grid = np.full((h, w), bg, dtype=int)
    regions = _regions(rng, h, w, plan["n_shapes"])
    shapes = []
    for r, c, a, b in regions:
        low, high = (.48, .77) if plan["density"] == "open" else (.80, 1.01)
        sh = max(2, min(a, int(round(a * float(rng.uniform(low, high))))))
        sw = max(2, min(b, int(round(b * float(rng.uniform(low, high))))))
        # The vocabulary contains ordinary bars, letters, frames and geometric
        # bodies.  Avoid a one-cell bar when three colors must fit in one body.
        kinds = ["rectangle", "L", "staircase"]
        if min(sh, sw) >= 3:
            kinds += ["T", "U", "H", "plus", "frame"]
        if min(sh, sw) >= 5:
            kinds += ["chamfer", "diamond"]
        mask, kind = familiar_shape(rng, sh, sw, str(rng.choice(kinds)))
        sh, sw = mask.shape
        rr = r + int(rng.integers(a - sh + 1))
        cc = c + int(rng.integers(b - sw + 1))
        shapes.append((rr, cc, mask, kind))
    capacities = [1 if plan["family"] == "shapes" else min(3, max(mask.shape))
                  for _, _, mask, _ in shapes]
    if sum(capacities) < len(colors):
        return None
    assigned = [[colors[i % len(colors)]] for i in range(len(shapes))]
    for color in colors[len(shapes):]:
        eligible = [i for i in range(len(shapes)) if len(assigned[i]) < capacities[i]]
        assigned[int(rng.choice(eligible))].append(color)
    if plan["family"] == "multicolor_shapes" and len(colors) >= 2:
        for ink, capacity in zip(assigned, capacities):
            want = min(capacity, len(colors), int(rng.choice((1, 2, 3), p=(.35, .45, .20))))
            available = [color for color in colors if color not in ink]
            if want > len(ink):
                ink.extend(map(int, rng.choice(available, size=want - len(ink), replace=False)))
        if all(len(ink) == 1 for ink in assigned):
            i = int(rng.choice([i for i, capacity in enumerate(capacities) if capacity >= 2]))
            assigned[i].append(int(rng.choice([color for color in colors if color not in assigned[i]])))
    records = []
    for (r, c, mask, kind), ink in zip(shapes, assigned):
        sh, sw = mask.shape
        # Every row and column of these cropped shapes is occupied.  Splitting
        # the longer axis into contiguous bands guarantees each color appears.
        axis = 0 if sh >= sw else 1
        widths = _partition(rng, mask.shape[axis], len(ink))
        strip = np.repeat(np.asarray(ink), widths)
        fill = np.broadcast_to(strip[:, None] if axis == 0 else strip[None, :], mask.shape)
        part = grid[r:r + sh, c:c + sw]
        part[mask] = fill[mask]
        records.append({"kind": kind, "bounds": [r, c, sh, sw],
                        "cells": int(mask.sum()), "colors": list(ink)})
    if not np.any(grid == bg):
        return None
    return grid, {"shapes": records, "background_style": "uniform"}


def _bands(rng, plan):
    h, w, palette = plan["h"], plan["w"], plan["palette"]
    eligible = [i for i, side in enumerate((h, w)) if side >= len(palette)]
    axis = int(rng.choice(eligible))
    side = (h, w)[axis]
    count = int(rng.integers(len(palette), min(side, 2 * len(palette) + 2) + 1))
    widths = _partition(rng, side, count)
    strip = np.repeat(np.asarray(palette)[np.arange(count) % len(palette)], widths)
    grid = np.broadcast_to(strip[:, None] if axis == 0 else strip[None, :], (h, w)).copy()
    return grid, {"band_axis": axis, "band_widths": widths.tolist(), "background_style": "bands"}


def _checks(rng, plan):
    h, w, palette = plan["h"], plan["w"], plan["palette"]
    # Variable block widths retain the recognizable checker relation while
    # avoiding a compulsory 1x1 texture or a square/symmetric drawing.
    choices = [(a, b) for a in range(2, min(h, 7) + 1)
               for b in range(2, min(w, 7) + 1) if a + b - 1 >= len(palette)]
    nr, nc = choices[int(rng.integers(len(choices)))]
    rows, cols = _partition(rng, h, nr), _partition(rng, w, nc)
    layer = np.repeat(np.arange(nr), rows)[:, None] + np.repeat(np.arange(nc), cols)[None, :]
    grid = np.asarray(palette)[layer % len(palette)]
    return grid, {"row_widths": rows.tolist(), "col_widths": cols.tolist(), "background_style": "checks"}


def _patchwork(rng, plan):
    h, w, palette = plan["h"], plan["w"], plan["palette"]
    maximum = min(14, (h // 2) * (w // 2))
    count = int(rng.integers(len(palette), max(len(palette), maximum) + 1))
    regions = _regions(rng, h, w, count)
    grid = np.empty((h, w), dtype=int)
    for i, (r, c, a, b) in enumerate(regions):
        grid[r:r + a, c:c + b] = palette[i % len(palette)]
    return grid, {"patches": [list(x) for x in regions], "background_style": "patchwork"}


def _nested_frames(rng, plan):
    h, w, palette = plan["h"], plan["w"], plan["palette"]
    depth = len(palette) - 1
    minimum = max(2, 2 * depth - 1)
    sh = int(rng.integers(minimum, h + 1))
    sw = int(rng.integers(minimum, w + 1))
    r, c = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
    yy, xx = np.indices((sh, sw))
    distance = np.minimum.reduce((yy, xx, sh - 1 - yy, sw - 1 - xx))
    # Positive ring widths, with the last color filling the center.
    widths = _partition(rng, (min(sh, sw) + 1) // 2, depth)
    ring = np.repeat(np.asarray(palette[1:]), widths)
    grid = np.full((h, w), palette[0], dtype=int)
    grid[r:r + sh, c:c + sw] = ring[distance]
    if np.all(grid != palette[0]):
        # A full-canvas set of rings is equally natural.  Its outermost ring
        # uses the base color, retaining the chosen palette with no repair.
        if (min(sh, sw) + 1) // 2 < len(palette):
            return None
        widths = _partition(rng, (min(sh, sw) + 1) // 2, len(palette))
        grid = np.repeat(np.asarray(palette), widths)[distance]
    return grid, {"frame_bounds": [r, c, sh, sw], "ring_widths": widths.tolist(),
                  "background_style": "nested_frames"}


def _draw(rng, plan):
    family = plan["family"]
    if family in ("shapes", "multicolor_shapes"):
        return _shape_scene(rng, plan)
    return {"bands": _bands, "checks": _checks, "patchwork": _patchwork,
            "nested_frames": _nested_frames}[family](rng, plan)


def _nonidentity_diagonals(grid):
    return (grid.shape[0] != grid.shape[1]
            or (not np.array_equal(grid, grid.T) and not np.array_equal(grid, grid[::-1, ::-1].T)))


def draw_content(rng, *, for_reflection=False):
    plan = _plan(rng)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _draw(rng, plan)
        if result is None:
            continue
        grid, info = result
        if set(map(int, np.unique(grid))) != set(plan["palette"]):
            raise ValueError("The drawing lost a planned color")
        if for_reflection and not _nonidentity_diagonals(grid):
            continue
        counts = np.bincount(grid.ravel(), minlength=10)
        bg = plan["palette"][0]
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"],
            "planned_family": plan["family"], "content_family": plan["family"],
            "planned_density": plan["density"], "palette": plan["palette"],
            "palette_capacity": plan["palette_capacity"], "n_colors": len(plan["palette"]),
            "background_color": bg, "color_counts": counts.tolist(),
            "foreground_cells": int(grid.size - counts[bg]),
            "foreground_occupancy": float(1 - counts[bg] / grid.size),
            "largest_color_cells": int(counts.max()), "attempts": attempt, **info,
        }
        if "n_shapes" in plan:
            meta["planned_n_shapes"] = plan["n_shapes"]
            meta["n_shapes"] = len(info["shapes"])
            meta["largest_shape_cells"] = max(shape["cells"] for shape in info["shapes"])
            meta["multicolor_shape_count"] = sum(len(shape["colors"]) > 1 for shape in info["shapes"])
        return grid, meta
    raise RuntimeError(f"Could not draw the fixed 66/70 scene plan after {MAX_ATTEMPTS} attempts: {plan}")


def _validate_content(example):
    grid = np.asarray(example["input"])
    if grid.ndim != 2 or not all(5 <= side <= 30 for side in grid.shape):
        raise ValueError("Grid sides must be between five and thirty")
    if not np.issubdtype(grid.dtype, np.integer) or grid.min() < 0 or grid.max() > 9:
        raise ValueError("Grid colors must be integers from zero to nine")
    meta = example["metadata"]
    counts = [sum(value == color for row in example["input"] for value in row) for color in range(10)]
    present = {i for i, count in enumerate(counts) if count}
    bg = meta["background_color"]
    measured = {"grid_h": grid.shape[0], "grid_w": grid.shape[1], "n_colors": len(present),
                "color_counts": counts, "foreground_cells": grid.size - counts[bg],
                "foreground_occupancy": 1 - counts[bg] / grid.size,
                "largest_color_cells": max(counts)}
    if len(present) < 2 or present != set(meta["palette"]):
        raise ValueError("The visible palette disagrees with the plan")
    if meta["content_family"] != meta["planned_family"] or meta["content_family"] not in FAMILIES:
        raise ValueError("The content family changed during drawing")
    if any(meta[key] != value for key, value in measured.items()):
        raise ValueError("Content metadata disagrees with the visible grid")
    if "shapes" in meta:
        if len(meta["shapes"]) != meta["n_shapes"] or meta["n_shapes"] != meta["planned_n_shapes"]:
            raise ValueError("The planned shapes were dropped")
        visible_sizes = []
        for record in meta["shapes"]:
            r, c, sh, sw = record["bounds"]
            patch = grid[r:r + sh, c:c + sw]
            size = int(np.count_nonzero(patch != bg))
            if patch.shape != (sh, sw) or size != record["cells"] or set(np.unique(patch)) - {bg} != set(record["colors"]):
                raise ValueError("A drawn shape disagrees with its visible footprint")
            visible_sizes.append(size)
        if sum(visible_sizes) != measured["foreground_cells"] or max(visible_sizes) != meta["largest_shape_cells"]:
            raise ValueError("Shape sizes disagree with visible foreground coverage")
        if sum(len(record["colors"]) > 1 for record in meta["shapes"]) != meta["multicolor_shape_count"]:
            raise ValueError("Multicolor shape count disagrees with the visible footprints")
    return grid, meta, counts


def validate_diagonal(example):
    """Independent scalar coordinate map, separate from NumPy transposition."""
    grid, meta, _ = _validate_content(example)
    h, w = grid.shape
    diagonal = meta["diagonal"]
    if diagonal not in ("main", "anti"):
        raise ValueError("Unknown diagonal")
    out = example["output"]
    if len(out) != w or any(len(row) != h for row in out):
        raise ValueError("Diagonal reflection must exchange the canvas dimensions")
    for r in range(h):
        for c in range(w):
            rr, cc = (c, r) if diagonal == "main" else (w - 1 - c, h - 1 - r)
            if out[rr][cc] != example["input"][r][c]:
                raise ValueError("Diagonal output disagrees with the visible input")
    if not _nonidentity_diagonals(grid):
        raise ValueError("Either diagonal would leave the grid unchanged")
    differs = any(grid[r, c] != grid[h - 1 - r, w - 1 - c]
                  for r in range(h) for c in range(w))
    if meta["diagonal_matters"] != differs:
        raise ValueError("Diagonal counterfactual metadata is incorrect")


def validate_swap(example):
    """Literal per-cell swap, independent of the construction's lookup table."""
    grid, meta, counts = _validate_content(example)
    a, b = meta["color_a"], meta["color_b"]
    if a == b or a not in range(10) or b not in range(10) or not counts[a] or not counts[b]:
        raise ValueError("Swap colors must be different colors present in the grid")
    out = example["output"]
    if len(out) != len(grid) or any(len(row) != grid.shape[1] for row in out):
        raise ValueError("Swap output dimensions changed")
    for before, after in zip(example["input"], out):
        for old, new in zip(before, after):
            expected = b if old == a else a if old == b else old
            if new != expected:
                raise ValueError("Swap output disagrees with the visible input")
    measured = {"count_a": counts[a], "count_b": counts[b],
                "changed_cells": counts[a] + counts[b],
                "changed_fraction": (counts[a] + counts[b]) / grid.size,
                "swap_includes_background": meta["background_color"] in (a, b),
                "swap_includes_largest_color": counts[a] == max(counts) or counts[b] == max(counts),
                "count_ratio": min(counts[a], counts[b]) / max(counts[a], counts[b])}
    if any(meta[key] != value for key, value in measured.items()):
        raise ValueError("Swap metadata disagrees with the visible color counts")


def choose_diagonal(rng):
    return str(rng.choice(("main", "anti")))


def choose_swap_pair(rng, present):
    return tuple(map(int, rng.choice(present, size=2, replace=False)))


def generate_diagonal(rng):
    grid, meta = draw_content(rng, for_reflection=True)
    diagonal = choose_diagonal(rng)
    out = grid.T.copy() if diagonal == "main" else grid[::-1, ::-1].T.copy()
    meta.update(diagonal=diagonal, diagonal_desc=("main diagonal (top-left to bottom-right)" if diagonal == "main"
                else "anti-diagonal (top-right to bottom-left)"),
                diagonal_matters=not np.array_equal(grid, grid[::-1, ::-1]))
    example = {"input": grid.tolist(), "output": out.tolist(), "metadata": meta}
    validate_diagonal(example)
    return example


def generate_swap(rng):
    grid, meta = draw_content(rng)
    a, b = choose_swap_pair(rng, sorted(meta["palette"]))
    lookup = np.arange(10)
    lookup[a], lookup[b] = b, a
    out = lookup[grid]
    count_a, count_b = meta["color_counts"][a], meta["color_counts"][b]
    meta.update(color_a=a, color_b=b, query_selection="uniform_present_pair",
                count_a=count_a, count_b=count_b, changed_cells=count_a + count_b,
                changed_fraction=(count_a + count_b) / grid.size,
                swap_includes_background=meta["background_color"] in (a, b),
                swap_includes_largest_color=max(count_a, count_b) == meta["largest_color_cells"],
                count_ratio=min(count_a, count_b) / max(count_a, count_b))
    example = {"input": grid.tolist(), "output": out.tolist(), "metadata": meta}
    validate_swap(example)
    return example
