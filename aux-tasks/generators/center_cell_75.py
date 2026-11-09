"""Natural, query-independent drawings for the center-cell question.

The scene is finished before its center is read.  There is no center marker,
repaint, rejection by answer color, or preferential placement at the center.
The drawing primitives are shared with the reviewed whole-grid transforms;
this task supplies its own odd canvas and preserves its complete scene plan.
"""
from __future__ import annotations

import numpy as np

from generators.grid_content_66_70 import _draw, PALETTE_WEIGHTS


FAMILIES = ("shapes", "multicolor_shapes", "bands", "checks", "patchwork", "nested_frames")
FAMILY_WEIGHTS = (.27, .27, .12, .12, .14, .08)
MAX_ATTEMPTS = 64


def _plan(rng):
    h, w = map(int, 2 * rng.integers(2, 15, size=2) + 1)
    family = str(rng.choice(FAMILIES, p=FAMILY_WEIGHTS))
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


def _draw_scene(rng, plan):
    """Retry placement without replacing the chosen canvas, palette or family."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _draw(rng, plan)
        if result is None:
            continue
        grid, info = result
        if set(map(int, np.unique(grid))) != set(plan["palette"]):
            raise ValueError("Task 75 drawing lost a planned color")
        if "n_shapes" in plan and len(info["shapes"]) != plan["n_shapes"]:
            raise ValueError("Task 75 drawing lost a planned object")
        return grid, info, attempt
    raise RuntimeError(f"Task 75 could not draw its fixed scene plan: {plan}")


def validate_center_color(example):
    """Independently strip opposite borders to find the one remaining cell."""
    grid = np.asarray(example["input"])
    if grid.ndim != 2 or not all(5 <= side <= 29 and side % 2 for side in grid.shape):
        raise ValueError("Task 75 requires odd grid sides between five and twenty-nine")
    if not np.issubdtype(grid.dtype, np.integer) or grid.min() < 0 or grid.max() > 9:
        raise ValueError("Task 75 requires integer colors from zero to nine")
    remaining = grid
    while remaining.shape[0] > 1:
        remaining = remaining[1:-1, :]
    while remaining.shape[1] > 1:
        remaining = remaining[:, 1:-1]
    if example["output"] != str(int(remaining.item())):
        raise ValueError("Task 75 answer disagrees with the visible center cell")
    meta = example.get("metadata", {})
    if meta:
        if (meta["grid_h"], meta["grid_w"]) != grid.shape:
            raise ValueError("Task 75 canvas metadata is inaccurate")
        if meta["n_colors"] != len(np.unique(grid)):
            raise ValueError("Task 75 palette metadata is inaccurate")
        if meta["n_shapes"] != meta["planned_n_shapes"]:
            raise ValueError("Task 75 did not preserve its object plan")


def generate_center_color(rng):
    plan = _plan(rng)
    grid, info, attempts = _draw_scene(rng, plan)
    counts = np.bincount(grid.ravel(), minlength=10)
    shapes = info.get("shapes", [])
    bg = plan["palette"][0]
    answer = int(grid[plan["h"] // 2, plan["w"] // 2])
    metadata = {
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
        "center_role": ("field" if answer == bg else "object") if shapes else "pattern",
        "attempts": attempts, **info,
    }
    example = {"input": grid.tolist(), "output": str(answer), "metadata": metadata}
    validate_center_color(example)
    return example
