"""Shared helpers for second-batch core generators."""
from __future__ import annotations

import numpy as np

from common.grid import NUM_COLORS, make_grid, pick_colors
from common.objects import bounding_box, random_shape


MAX_RETRIES = 80


def unused_colors(rng: np.random.Generator, n: int, *exclude: int) -> list[int]:
    """Pick n distinct colors excluding the provided color indices."""
    return [int(c) for c in pick_colors(rng, n, exclude=[int(c) for c in exclude])]


def boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    """Return True when boxes overlap or are closer than gap cells."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def bbox_outline(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    """Return cells on the outline of a bounding box."""
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def place_shape(
    grid: np.ndarray,
    rng: np.random.Generator,
    shape: frozenset[tuple[int, int]],
    color: int,
    bg: int,
    placed_boxes: list[tuple[int, int, int, int]],
    gap: int = 1,
    max_attempts: int = 200,
) -> set[tuple[int, int]] | None:
    """Place a shape on background cells without touching existing boxes."""
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = bounding_box(shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    if sh > h or sw > w:
        return None
    for _ in range(max_attempts):
        r0 = int(rng.integers(0, h - sh + 1))
        c0 = int(rng.integers(0, w - sw + 1))
        box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        if any(boxes_overlap(box, other, gap=gap) for other in placed_boxes):
            continue
        cells = {(r0 + r - sr0, c0 + c - sc0) for r, c in shape}
        if all(grid[r, c] == bg for r, c in cells):
            for r, c in cells:
                grid[r, c] = color
            placed_boxes.append(box)
            return cells
    return None


def random_scene_objects(
    rng: np.random.Generator,
    *,
    h: int,
    w: int,
    bg: int,
    n_objects: int,
    same_color: int | None = None,
    min_cells: int = 4,
    max_cells: int = 10,
    gap: int = 1,
) -> tuple[np.ndarray, list[dict]] | None:
    """Create separated single-color 4-connected objects."""
    grid = make_grid(h, w, fill=bg)
    colors = (
        [int(same_color)] * n_objects
        if same_color is not None
        else unused_colors(rng, n_objects, bg)
    )
    objects: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    for color in colors:
        for _ in range(80):
            size = int(rng.integers(min_cells, max_cells + 1))
            shape = random_shape(rng, (size, size), connectivity=4)
            cells = place_shape(grid, rng, shape, int(color), bg, boxes, gap=gap)
            if cells is not None:
                objects.append({
                    "color": int(color),
                    "cells": cells,
                    "bbox": bounding_box(cells),
                })
                break
        else:
            return None
    return grid, objects


def sorted_components(comps: list[dict]) -> list[dict]:
    """Sort connected-component dictionaries by top-left bounding-box corner."""
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))


def as_uint_grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


def available_colors(*exclude: int) -> list[int]:
    excluded = {int(c) for c in exclude}
    return [c for c in range(NUM_COLORS) if c not in excluded]


def inside_any_box(r: int, c: int, boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes)


def add_patterned_background(
    grid: np.ndarray,
    rng: np.random.Generator,
    *,
    bg: int,
    exclude_colors: list[int] | tuple[int, ...] = (),
    protected_cells: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    protected_boxes: list[tuple[int, int, int, int]] | None = None,
    solid_probability: float = 0.30,
) -> dict:
    """Add structured background detail while preserving semantic cells."""
    if rng.random() < solid_probability:
        return {"style": "solid", "colors": [], "period": 0}
    palette = available_colors(bg, *exclude_colors)
    if not palette:
        return {"style": "solid", "colors": [], "period": 0}
    protected_cells = set(protected_cells or set())
    protected_boxes = protected_boxes or []
    n_colors = min(len(palette), int(rng.integers(1, 3)))
    colors = [int(c) for c in rng.choice(np.array(palette, dtype=np.uint8), size=n_colors, replace=False)]
    style = str(rng.choice(["checkerboard", "horizontal_stripes", "vertical_stripes", "diagonal_stripes", "dotted_grid"]))
    period = int(rng.integers(2, 5))
    offset = int(rng.integers(0, period))
    used: set[int] = set()
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg or (r, c) in protected_cells or inside_any_box(r, c, protected_boxes):
                continue
            pattern_slot: int | None = None
            if style == "checkerboard" and ((r // period) + (c // period) + offset) % 2 == 0:
                pattern_slot = ((r // period) + (c // period)) % len(colors)
            elif style == "horizontal_stripes" and (r + offset) % period == 0:
                pattern_slot = (r // period) % len(colors)
            elif style == "vertical_stripes" and (c + offset) % period == 0:
                pattern_slot = (c // period) % len(colors)
            elif style == "diagonal_stripes" and (r + c + offset) % period == 0:
                pattern_slot = ((r + c) // period) % len(colors)
            elif style == "dotted_grid" and (r + offset) % period == 0 and (c + offset) % period == 0:
                pattern_slot = ((r // period) + (c // period)) % len(colors)
            if pattern_slot is not None:
                color = colors[pattern_slot]
                grid[r, c] = color
                used.add(color)

    return {"style": style, "colors": sorted(used), "period": period}
