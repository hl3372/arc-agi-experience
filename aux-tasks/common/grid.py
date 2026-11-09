"""Grid creation, color palette, background generators, and scene composition."""
from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple

from common.objects import (
    random_shape, place_shape_on_grid, stamp_rectangle,
    POLYOMINOES, random_polyomino, normalize_shape,
)

NUM_COLORS = 10

COLOR_NAMES: dict[int, str] = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "magenta", 7: "orange", 8: "cyan", 9: "maroon",
}

COLOR_INDEX: dict[str, int] = {v: k for k, v in COLOR_NAMES.items()}


# ---------------------------------------------------------------------------
# Basic grid helpers
# ---------------------------------------------------------------------------

def make_grid(height: int, width: int, fill: int = 0) -> np.ndarray:
    return np.full((height, width), fill, dtype=np.uint8)


def random_grid_size(
    rng: np.random.Generator, min_dim: int = 3, max_dim: int = 30
) -> Tuple[int, int]:
    h = int(rng.integers(min_dim, max_dim + 1))
    w = int(rng.integers(min_dim, max_dim + 1))
    return h, w


def random_odd_grid_size(
    rng: np.random.Generator, min_dim: int = 3, max_dim: int = 29
) -> Tuple[int, int]:
    lo = min_dim if min_dim % 2 == 1 else min_dim + 1
    hi = max_dim if max_dim % 2 == 1 else max_dim - 1
    odds = list(range(lo, hi + 1, 2))
    h = int(rng.choice(odds))
    w = int(rng.choice(odds))
    return h, w


def grid_to_list(grid: np.ndarray) -> list:
    return grid.astype(int).tolist()


def pick_colors(rng: np.random.Generator, n: int, exclude: Optional[List[int]] = None) -> List[int]:
    """Pick n distinct colors, excluding the given set."""
    ex = set(exclude or [])
    available = [c for c in range(NUM_COLORS) if c not in ex]
    if n > len(available):
        n = len(available)
    return list(rng.choice(available, size=n, replace=False))


def pick_color(rng: np.random.Generator, exclude: Optional[List[int]] = None) -> int:
    """Pick one color not in exclude."""
    return pick_colors(rng, 1, exclude)[0]


# ---------------------------------------------------------------------------
# Background generators
# ---------------------------------------------------------------------------

def uniform_background(rng: np.random.Generator, h: int, w: int,
                       bg_color: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if bg_color is None:
        bg_color = int(rng.integers(0, NUM_COLORS))
    return make_grid(h, w, fill=bg_color), bg_color


def noisy_background(rng: np.random.Generator, h: int, w: int,
                     bg_color: Optional[int] = None,
                     noise_rate: float = 0.1) -> Tuple[np.ndarray, int]:
    """Uniform background with random noise cells."""
    if bg_color is None:
        bg_color = int(rng.integers(0, NUM_COLORS))
    grid = make_grid(h, w, fill=bg_color)
    noise_colors = [c for c in range(NUM_COLORS) if c != bg_color]
    n_noise = max(1, int(h * w * noise_rate))
    for _ in range(n_noise):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        grid[r, c] = int(rng.choice(noise_colors))
    return grid, bg_color


def checkered_background(rng: np.random.Generator, h: int, w: int,
                         bg_color: Optional[int] = None,
                         cell_size: int = 1) -> Tuple[np.ndarray, int]:
    """Two-color checkerboard."""
    if bg_color is None:
        bg_color = int(rng.integers(0, NUM_COLORS))
    alt_color = pick_color(rng, exclude=[bg_color])
    grid = make_grid(h, w, fill=bg_color)
    for r in range(h):
        for c in range(w):
            if ((r // cell_size) + (c // cell_size)) % 2 == 1:
                grid[r, c] = alt_color
    return grid, bg_color


def striped_background(rng: np.random.Generator, h: int, w: int,
                       bg_color: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Horizontal or vertical stripes with 2-3 colors."""
    if bg_color is None:
        bg_color = int(rng.integers(0, NUM_COLORS))
    n_stripe_colors = int(rng.integers(1, 3))
    stripe_colors = pick_colors(rng, n_stripe_colors, exclude=[bg_color])
    all_colors = [bg_color] + list(stripe_colors)
    horizontal = rng.random() < 0.5
    stripe_width = int(rng.integers(1, 4))
    grid = make_grid(h, w, fill=bg_color)
    dim = h if horizontal else w
    for i in range(dim):
        color = all_colors[(i // stripe_width) % len(all_colors)]
        if horizontal:
            grid[i, :] = color
        else:
            grid[:, i] = color
    return grid, bg_color


def random_background(rng: np.random.Generator, h: int, w: int,
                      style: Optional[str] = None,
                      bg_color: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Generate a background. style: uniform, noisy, checkered, striped, or random."""
    if style is None or style == "random":
        style = str(rng.choice(["uniform", "uniform", "noisy", "checkered", "striped"]))

    if style == "uniform":
        return uniform_background(rng, h, w, bg_color)
    elif style == "noisy":
        rate = float(rng.uniform(0.03, 0.12))
        return noisy_background(rng, h, w, bg_color, noise_rate=rate)
    elif style == "checkered":
        cs = int(rng.integers(1, 3))
        return checkered_background(rng, h, w, bg_color, cell_size=cs)
    elif style == "striped":
        return striped_background(rng, h, w, bg_color)
    else:
        return uniform_background(rng, h, w, bg_color)


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

def generate_scene(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_objects: int = 4,
    bg_style: str = "uniform",
    bg_color: Optional[int] = None,
    size_range: Tuple[int, int] = (3, 8),
    connectivity: int = 4,
    palette: Optional[List[int]] = None,
    spacing: int = 1,
    gap: int = 0,
) -> Tuple[np.ndarray, int, list]:
    """Generate a complete scene with background + diverse objects.

    Returns (grid, bg_color, placed_objects) where placed_objects is a list
    of dicts with 'color', 'cells' keys.

    If *gap* > 0, each object is placed with a preferred *gap*-cell clear ring so
    objects rarely touch; placement falls back to no gap only when the scene is
    too crowded to separate them (so some incidental touching remains). *gap*
    defaults to 0, preserving existing callers.
    """
    grid, bg = random_background(rng, h, w, style=bg_style, bg_color=bg_color)

    if palette is None:
        n_colors = int(rng.integers(2, min(6, NUM_COLORS - 1) + 1))
        palette = pick_colors(rng, n_colors, exclude=[bg])

    placed = []
    for _ in range(n_objects):
        color = int(rng.choice(palette))
        shape = random_shape(rng, size_range=size_range, connectivity=connectivity)
        # Record cells before placement to track what we placed
        h_grid, w_grid = grid.shape
        old_grid = grid.copy()
        if gap > 0:
            success = bool(
                place_shape_on_grid(grid, rng, shape, color, bg, margin=0, max_attempts=110, gap=gap)
                or place_shape_on_grid(grid, rng, shape, color, bg, margin=0, max_attempts=25)
            )
        else:
            success = place_shape_on_grid(grid, rng, shape, color, bg, margin=0)
        if success:
            # Find the new cells
            diff = np.argwhere((grid != old_grid) & (grid == color))
            cells = set(map(tuple, diff.tolist()))
            if cells:
                placed.append({"color": color, "cells": cells})

    return grid, bg, placed


def decorate_background_cells(
    rng: np.random.Generator,
    base_grid: np.ndarray,
    bg: int,
    forbidden_colors: set,
) -> np.ndarray:
    """Recolor some background cells with safe pattern colors.

    The returned grid preserves all object cells exactly. Only cells that
    are *bg* in *base_grid* may change, and pattern colors never overlap the
    object palette (``forbidden_colors``). This lets tasks keep visually richer
    backgrounds while analysing objects against the original single-color grid:
    compute the answer on the plain ``base_grid``, then display the decorated one.
    """
    decorated = base_grid.copy()
    bg_mask = base_grid == bg
    if not np.any(bg_mask):
        return decorated

    if rng.random() < 0.45:
        return decorated

    available = [
        c for c in range(NUM_COLORS)
        if c != bg and c not in forbidden_colors
    ]
    if not available:
        return decorated

    style = str(rng.choice(["striped", "checkered", "speckle"]))
    accent_count = min(len(available), 2)
    accents = [int(c) for c in rng.choice(available, size=accent_count, replace=False)]
    rows, cols = decorated.shape

    if style == "striped":
        period = int(rng.integers(2, 5))
        axis = str(rng.choice(["row", "col"]))
        accent = accents[0]
        for r in range(rows):
            for c in range(cols):
                if not bg_mask[r, c]:
                    continue
                if axis == "row" and ((r + c // max(period, 1)) % period == 0):
                    decorated[r, c] = accent
                elif axis == "col" and ((c + r // max(period, 1)) % period == 0):
                    decorated[r, c] = accent
    elif style == "checkered":
        accent_a = accents[0]
        accent_b = accents[-1]
        for r in range(rows):
            for c in range(cols):
                if not bg_mask[r, c]:
                    continue
                parity = (r + c) % 4
                if parity == 1:
                    decorated[r, c] = accent_a
                elif parity == 3 and accent_b != accent_a:
                    decorated[r, c] = accent_b
    else:
        density = float(rng.uniform(0.12, 0.28))
        bg_positions = np.argwhere(bg_mask)
        n_cells = int(round(len(bg_positions) * density))
        if n_cells > 0:
            chosen = rng.choice(len(bg_positions), size=n_cells, replace=False)
            for idx in chosen.tolist():
                r, c = bg_positions[idx]
                decorated[r, c] = accents[int(rng.integers(0, len(accents)))]

    return decorated


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compat, but prefer generate_scene)
# ---------------------------------------------------------------------------

def scatter_cells(
    grid: np.ndarray, rng: np.random.Generator,
    color: int, count: int, bg_color: int = 0,
) -> np.ndarray:
    """Place `count` cells of `color` at random bg positions."""
    bg_positions = list(zip(*np.where(grid == bg_color)))
    if not bg_positions:
        return grid
    count = min(count, len(bg_positions))
    chosen = rng.choice(len(bg_positions), size=count, replace=False)
    for idx in chosen:
        r, c = bg_positions[idx]
        grid[r, c] = color
    return grid


def place_rectangle(
    grid: np.ndarray, rng: np.random.Generator,
    color: int, min_side: int = 2, max_side: int = 6,
) -> np.ndarray:
    """Stamp a filled rectangle at a random position."""
    h_grid, w_grid = grid.shape
    # Clamp the lower side to the grid extent so a 1-wide grid doesn't crash on rng.integers(2, 2).
    # For any grid >= min_side in a dimension this is identical to the original draw (min(min_side, dim) == min_side).
    rh = int(rng.integers(min(min_side, h_grid), min(max_side, h_grid) + 1))
    rw = int(rng.integers(min(min_side, w_grid), min(max_side, w_grid) + 1))
    r0 = int(rng.integers(0, h_grid - rh + 1))
    c0 = int(rng.integers(0, w_grid - rw + 1))
    grid[r0 : r0 + rh, c0 : c0 + rw] = color
    return grid


def populate_scene(
    grid: np.ndarray, rng: np.random.Generator,
    bg_color: int, n_objects: int = 3,
    palette: Optional[List[int]] = None,
) -> np.ndarray:
    """Legacy: add random objects. Prefer generate_scene for new code."""
    if palette is None:
        available = [c for c in range(NUM_COLORS) if c != bg_color]
        n_colors = int(rng.integers(2, min(6, len(available)) + 1))
        palette = list(rng.choice(available, size=n_colors, replace=False))

    for _ in range(n_objects):
        color = int(rng.choice(palette))
        shape = random_shape(rng, size_range=(3, 8))
        if not place_shape_on_grid(grid, rng, shape, color, bg_color):
            # Fallback to rectangle
            max_side = max(2, min(6, min(grid.shape) // 2))
            place_rectangle(grid, rng, color, min_side=2, max_side=max_side)

    return grid
