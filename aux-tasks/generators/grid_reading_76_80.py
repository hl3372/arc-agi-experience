"""Coherent drawings for color histograms and literal half-grid extraction.

The drawing primitives are shared, without modifying the reviewed 66/70/75
tasks. Canvas, family, palette, density and shape count are fixed before any
placement retries. Histogram retries keep that entire plan; the crop direction
is chosen only after the drawing is complete.
"""
from __future__ import annotations

from collections import Counter
from math import isqrt

import numpy as np

from common.grid import COLOR_NAMES, COLOR_INDEX
from generators.grid_content_66_70 import _draw, PALETTE_WEIGHTS


FAMILIES = ("shapes", "multicolor_shapes", "bands", "checks", "patchwork", "nested_frames")
FAMILY_WEIGHTS = (.27, .27, .12, .12, .14, .08)
# Rich plans get extra weight where geometric capacity permits them. Without
# this correction, the small-canvas/family caps would erase most of that tail.
HISTOGRAM_PALETTE_WEIGHTS = np.asarray((.09, .085, .085, .09, .095, .105, .135, .145, .17))
MAX_ATTEMPTS = 128
HALF_CANVASES = tuple((h, w) for h in range(5, 31) for w in range(5, 31)
                      if h % 2 == 0 or w % 2 == 0)


def _plan(rng, *, histogram):
    if histogram:
        h, w = map(int, rng.integers(5, 31, size=2))
    else:
        # Uniform over the 507 feasible pairs. Only the cut axis must be even;
        # both-odd canvases have no unambiguous half under this task's rule.
        h, w = HALF_CANVASES[int(rng.integers(len(HALF_CANVASES)))]
    family = str(rng.choice(FAMILIES, p=FAMILY_WEIGHTS))
    density = str(rng.choice(("open", "full"))) if family in FAMILIES[:2] else "structured"
    if family in FAMILIES[:2]:
        # Distinct populations need more capacity than simply showing colors,
        # especially for open drawings with several monochrome objects.
        budget = (36 if density == "open" else 25) if histogram else 14
        capacity = max(2, min(10, 1 + h * w // budget))
    elif family == "nested_frames":
        capacity = min(10, (min(h, w) + 1) // 2)
    elif family == "bands":
        capacity = min(10, (isqrt(1 + 8 * max(h, w)) - 1) // 2) if histogram else min(10, max(h, w))
    elif family == "checks":
        capacity = min(10, 2 + min(h, w) // 2) if histogram else min(10, h + w - 1, max(2, h * w // 4 + 1))
    else:
        capacity = max(2, min(10, 1 + h * w // 28, (h // 2) * (w // 2))) if histogram else min(10, (h // 2) * (w // 2))
    weights = (HISTOGRAM_PALETTE_WEIGHTS if histogram else PALETTE_WEIGHTS)[:capacity - 1]
    n = int(rng.choice(np.arange(2, capacity + 1), p=weights / weights.sum()))
    palette = list(map(int, rng.choice(10, n, replace=False)))
    plan = {"h": h, "w": w, "family": family, "palette": palette,
            "density": density, "palette_capacity": capacity, "histogram": histogram}
    if family in FAMILIES[:2]:
        minimum = n - 1 if family == "shapes" else (n + 1) // 3
        maximum = max(minimum, min(8, h * w // 24))
        plan["n_shapes"] = int(rng.integers(minimum, maximum + 1))
    return plan


def _distinct_bands(rng, plan):
    """Construct unequal total widths before drawing; never repair pixels."""
    h, w, palette = plan["h"], plan["w"], plan["palette"]
    n = len(palette)
    minimum = n * (n + 1) // 2
    axes = [i for i, side in enumerate((h, w)) if side >= minimum]
    axis = int(rng.choice(axes))
    widths = np.arange(1, n + 1)
    remaining = (h, w)[axis] - minimum
    while remaining:
        suffix = int(rng.integers(1, min(n, remaining) + 1))
        widths[-suffix:] += 1
        remaining -= suffix
    rng.shuffle(widths)
    bands = []
    for color, width in zip(palette, widths):
        if width >= 4 and rng.random() < .4:
            first = int(rng.integers(2, width - 1))
            bands.extend(((color, first), (color, int(width) - first)))
        else:
            bands.append((color, int(width)))
    rng.shuffle(bands)
    strip = np.concatenate([np.full(width, color, dtype=int) for color, width in bands])
    grid = np.broadcast_to(strip[:, None] if axis == 0 else strip[None, :], (h, w)).copy()
    return grid, {"band_axis": axis, "band_widths": [width for _, width in bands],
                  "band_colors": [color for color, _ in bands], "background_style": "bands"}


def _draw_planned(rng, plan):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _distinct_bands(rng, plan) if plan["histogram"] and plan["family"] == "bands" else _draw(rng, plan)
        if result is None:
            continue
        grid, info = result
        counts = np.bincount(grid.ravel(), minlength=10)
        if set(np.flatnonzero(counts)) != set(plan["palette"]):
            raise ValueError("Drawing lost a planned color")
        if "n_shapes" in plan and len(info["shapes"]) != plan["n_shapes"]:
            raise ValueError("Drawing lost a planned shape")
        if plan["histogram"] and len(set(counts[counts > 0])) != len(plan["palette"]):
            continue
        return grid, info, attempt
    raise RuntimeError(f"Could not draw the fixed grid-reading plan: {plan}")


def _metadata(grid, plan, info, attempts):
    counts = np.bincount(grid.ravel(), minlength=10)
    shapes = info.get("shapes", [])
    bg = plan["palette"][0]
    return {
        "grid_h": plan["h"], "grid_w": plan["w"],
        "content_family": plan["family"], "planned_family": plan["family"],
        "density": plan["density"], "planned_density": plan["density"],
        "palette": plan["palette"], "palette_capacity": plan["palette_capacity"],
        "n_colors": len(plan["palette"]), "color_counts": counts.tolist(),
        "background_color": bg, "foreground_cells": int(grid.size - counts[bg]),
        "foreground_occupancy": float(1 - counts[bg] / grid.size),
        "planned_n_shapes": plan.get("n_shapes", 0), "n_shapes": len(shapes),
        "object_cells": sum(shape["cells"] for shape in shapes),
        "largest_shape_cells": max((shape["cells"] for shape in shapes), default=0),
        "multicolor_shape_count": sum(len(shape["colors"]) > 1 for shape in shapes),
        "attempts": attempts, **info,
    }


def _read_input(example):
    """Literal grid/color checks, independent of the drawing machinery."""
    grid = example["input"]
    h = len(grid)
    w = len(grid[0]) if h else 0
    if not 5 <= h <= 30 or not 5 <= w <= 30 or any(len(row) != w for row in grid):
        raise ValueError("Input sides must be between five and thirty")
    if any(type(value) is not int or not 0 <= value <= 9 for row in grid for value in row):
        raise ValueError("Input cells must be integer colors from zero to nine")
    counts = Counter(value for row in grid for value in row)
    if len(counts) < 2:
        raise ValueError("The drawing must contain at least two colors")
    meta = example.get("metadata", {})
    if meta:
        measured = {"grid_h": h, "grid_w": w, "n_colors": len(counts),
                    "color_counts": [counts[c] for c in range(10)]}
        if any(meta[key] != value for key, value in measured.items()):
            raise ValueError("Drawing statistics disagree with the visible grid")
        if set(counts) != set(meta["palette"]) or meta["content_family"] != meta["planned_family"]:
            raise ValueError("Drawing changed its palette or family plan")
        if meta["n_shapes"] != meta["planned_n_shapes"]:
            raise ValueError("Drawing changed its shape-count plan")
        if meta["density"] != meta["planned_density"]:
            raise ValueError("Drawing changed its density plan")
        bg = meta["background_color"]
        if meta["foreground_cells"] != h * w - counts[bg]:
            raise ValueError("Foreground count disagrees with the visible drawing")
        if meta["foreground_occupancy"] != 1 - counts[bg] / (h * w):
            raise ValueError("Foreground occupancy disagrees with the visible drawing")
        bodies = meta.get("shapes", [])
        sizes = []
        for body in bodies:
            r, c, sh, sw = body["bounds"]
            if not (0 <= r < r + sh <= h and 0 <= c < c + sw <= w):
                raise ValueError("A shape's bounds lie outside the drawing")
            ink = [grid[y][x] for y in range(r, r + sh) for x in range(c, c + sw)
                   if grid[y][x] != bg]
            if len(ink) != body["cells"] or set(ink) != set(body["colors"]):
                raise ValueError("A shape's statistics disagree with its visible footprint")
            sizes.append(len(ink))
        if len(bodies) != meta["n_shapes"] or sum(sizes) != meta["object_cells"]:
            raise ValueError("Object counts or areas disagree with the drawing")
        if max(sizes, default=0) != meta["largest_shape_cells"]:
            raise ValueError("Largest-shape statistic is incorrect")
        if bodies and sum(sizes) != h * w - counts[bg]:
            raise ValueError("The shape records do not explain the foreground")
    return grid, counts


def validate_histogram(example):
    _, counts = _read_input(example)
    if len(set(counts.values())) != len(counts):
        raise ValueError("The histogram prompt requires distinct color counts")
    try:
        parts = [part.strip().rsplit(":", 1) for part in example["output"].split(",")]
        actual = [(COLOR_INDEX[color], int(count)) for color, count in parts]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid histogram answer format") from exc
    if actual != sorted(counts.items(), key=lambda pair: pair[1]):
        raise ValueError("Histogram answer disagrees with the visible cell counts")


def generate_histogram(rng):
    plan = _plan(rng, histogram=True)
    grid, info, attempts = _draw_planned(rng, plan)
    colors, counts = np.unique(grid, return_counts=True)
    answer = ", ".join(f"{COLOR_NAMES[int(colors[i])]}:{int(counts[i])}" for i in np.argsort(counts))
    meta = _metadata(grid, plan, info, attempts)
    example = {"input": grid.tolist(), "output": answer, "metadata": meta}
    validate_histogram(example)
    return example


def feasible_halves(grid):
    h, w = len(grid), len(grid[0])
    return (["top", "bottom"] if h % 2 == 0 else []) + (["left", "right"] if w % 2 == 0 else [])


def choose_half(rng, grid):
    return str(rng.choice(feasible_halves(grid)))


def validate_half(example):
    grid, _ = _read_input(example)
    h, w = len(grid), len(grid[0])
    side = example["metadata"]["which_half"]
    if side not in feasible_halves(grid):
        raise ValueError("The requested half does not split evenly")
    oh = h // 2 if side in ("top", "bottom") else h
    ow = w // 2 if side in ("left", "right") else w
    out = example["output"]
    if len(out) != oh or any(len(row) != ow for row in out):
        raise ValueError("Incorrect output half dimensions")
    r0, c0 = (oh if side == "bottom" else 0), (ow if side == "right" else 0)
    for r in range(oh):
        for c in range(ow):
            if out[r][c] != grid[r + r0][c + c0]:
                raise ValueError("Output is not the requested visible half")
    if example["metadata"]["feasible_halves"] != feasible_halves(grid):
        raise ValueError("Incorrect feasible-half metadata")


def generate_half(rng):
    plan = _plan(rng, histogram=False)
    grid, info, attempts = _draw_planned(rng, plan)
    side = choose_half(rng, grid)
    h, w = grid.shape
    bounds = {"top": (0, h // 2, 0, w), "bottom": (h // 2, h, 0, w),
              "left": (0, h, 0, w // 2), "right": (0, h, w // 2, w)}
    r0, r1, c0, c1 = bounds[side]
    meta = _metadata(grid, plan, info, attempts)
    meta.update(which_half=side, feasible_halves=feasible_halves(grid),
                both_axes_even=h % 2 == 0 and w % 2 == 0)
    example = {"input": grid.tolist(), "output": grid[r0:r1, c0:c1].tolist(), "metadata": meta}
    validate_half(example)
    return example
