"""Metadata family generators: tasks 1, 2, 25, 30, 75, 76."""
from __future__ import annotations

import numpy as np

from common.grid import (
    COLOR_NAMES,
    NUM_COLORS,
    generate_scene,
    grid_to_list,
    make_grid,
    random_grid_size,
    random_odd_grid_size,
)
from common.objects import place_shape_on_grid, random_polyomino, random_shape
from registry import register

MAX_RETRIES = 20


@register("1")
def generate_grid_size(rng: np.random.Generator) -> dict:
    """Task 1: Return the grid dimensions as 'HxW'."""
    h, w = random_grid_size(rng, min_dim=1, max_dim=30)
    bg_style = str(rng.choice(["uniform", "noisy", "checkered", "striped", "random"]))
    grid, bg, _ = generate_scene(
        rng, h, w,
        n_objects=int(rng.integers(2, 6)),
        bg_style=bg_style,
    )
    answer = f"{h}x{w}"
    assert answer == f"{grid.shape[0]}x{grid.shape[1]}"
    return {"input": grid_to_list(grid), "output": answer}


# ---------------------------------------------------------------------------
# Task 2 -- list the coordinates of one colour
# ---------------------------------------------------------------------------

TASK2_MAX_CELLS = 60          # answer length cap (was 15); 30x30 + 60 coordinates ~ 2000 tokens
_T2_COMPOSITIONS = ("dots", "shapes", "big", "mixed")
_T2_COMPOSITION_P = (0.20, 0.35, 0.25, 0.20)


def _rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


def _t2_sample_n(rng: np.random.Generator, area: int) -> int:
    """Target cell count: at most 40% of the canvas and never above the cap. Three bands
    (1-8 / 9-30 / 31-60) so short lists stay common while long ones are well represented."""
    cap = max(1, min(TASK2_MAX_CELLS, int(0.4 * area)))
    u = rng.random()
    if cap <= 8 or u < 0.40:
        return _rint(rng, 1, min(8, cap))
    if cap <= 30 or u < 0.75:
        return _rint(rng, 9, min(30, cap))
    return _rint(rng, 31, cap)


def _t2_piece_size(rng: np.random.Generator, composition: str, remaining: int) -> int:
    if composition == "dots":
        return 1
    if composition == "shapes":
        return min(remaining, _rint(rng, 2, 8))
    if composition == "big":
        return min(remaining, _rint(rng, 9, 40)) if remaining >= 9 else remaining
    u = rng.random()  # mixed
    if u < 0.30:
        return 1
    if u < 0.75:
        return min(remaining, _rint(rng, 2, 8))
    return min(remaining, _rint(rng, 9, 40)) if remaining >= 9 else remaining


def _t2_piece(rng: np.random.Generator, size: int, H: int, W: int) -> frozenset:
    """A 4-connected piece of exactly ``size`` cells that fits the canvas: a rectangle or bar
    when one fits (40%), else a random polyomino / blob-like walk."""
    if size == 1:
        return frozenset({(0, 0)})
    if rng.random() < 0.40:
        dims = [(h, size // h) for h in range(1, size + 1) if size % h == 0 and h <= H and size // h <= W]
        if dims:
            h, w = dims[int(rng.integers(0, len(dims)))]
            return frozenset((r, c) for r in range(h) for c in range(w))
    for _ in range(12):
        cells = random_polyomino(rng, size, 4)
        hh = max(r for r, _ in cells) + 1
        ww = max(c for _, c in cells) + 1
        if hh <= H and ww <= W:
            return frozenset(cells)
    return None


def _t2_fit(rng: np.random.Generator, blocked: np.ndarray, cells) -> tuple[int, int] | None:
    """Random anchor where every cell of the piece lands on an unblocked cell."""
    H, W = blocked.shape
    offs = sorted(cells)
    hs = max(r for r, _ in offs) + 1
    ws = max(c for _, c in offs) + 1
    if hs > H or ws > W:
        return None
    nh, nw = H - hs + 1, W - ws + 1
    valid = np.ones((nh, nw), dtype=bool)
    allowed = ~blocked
    for dr, dc in offs:
        valid &= allowed[dr:dr + nh, dc:dc + nw]
    anchors = np.argwhere(valid)
    if len(anchors) == 0:
        return None
    r0, c0 = anchors[int(rng.integers(0, len(anchors)))]
    return int(r0), int(c0)


def _t2_dilate(mask: np.ndarray) -> np.ndarray:
    """8-neighbourhood dilation (pieces of the target colour are kept apart)."""
    H, W = mask.shape
    pad = np.zeros((H + 2, W + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    out = np.zeros_like(mask)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out |= pad[dr:dr + H, dc:dc + W]
    return out


def _t2_build(rng: np.random.Generator, H: int, W: int):
    """One attempt at a task-2 scene; None when the target cells cannot all be placed."""
    area = H * W
    n = _t2_sample_n(rng, area)
    composition = _T2_COMPOSITIONS[int(rng.choice(len(_T2_COMPOSITIONS), p=_T2_COMPOSITION_P))]
    bg = int(rng.integers(0, NUM_COLORS))
    target = int(rng.choice([c for c in range(NUM_COLORS) if c != bg]))
    others = [c for c in range(NUM_COLORS) if c not in (bg, target)]

    # Background: uniform / noisy / striped, never using the target colour.
    style = str(rng.choice(["uniform", "noisy", "striped"], p=[0.50, 0.25, 0.25]))
    grid = make_grid(H, W, fill=bg)
    if style == "noisy":
        rate = float(rng.uniform(0.03, 0.12))
        k = max(1, int(area * rate))
        noise_pool = [others[i] for i in rng.choice(len(others), size=_rint(rng, 1, len(others)), replace=False)]
        for idx in rng.choice(area, size=min(k, area), replace=False).tolist():
            grid.flat[idx] = noise_pool[int(rng.integers(0, len(noise_pool)))]
    elif style == "striped" and max(H, W) >= 3:
        n_stripe = _rint(rng, 1, 2)
        stripe_colors = [others[i] for i in rng.choice(len(others), size=n_stripe, replace=False)]
        seq = [bg] + stripe_colors
        width = _rint(rng, 1, 3)
        horizontal = rng.random() < 0.5
        for i in range(H if horizontal else W):
            col = seq[(i // width) % len(seq)]
            if horizontal:
                grid[i, :] = col
            else:
                grid[:, i] = col
    else:
        style = "uniform"

    # Distractor objects in other colours (round-robin over a palette of 1..8 colours).
    object_mask = np.zeros((H, W), dtype=bool)
    n_objects = _rint(rng, 0, min(8, max(1, area // 20)))
    n_palette = _rint(rng, 1, min(8, len(others)))
    palette = [others[i] for i in rng.permutation(len(others))[:n_palette].tolist()]
    n_placed_objects = 0
    for i in range(n_objects):
        color = palette[i % len(palette)]
        shape = random_shape(rng, size_range=(3, 8))
        before = grid.copy()
        if place_shape_on_grid(grid, rng, shape, color, bg):
            object_mask |= (grid != before)
            n_placed_objects += 1

    # Target cells: exactly n, composed of pieces kept one cell apart.
    target_mask = np.zeros((H, W), dtype=bool)
    remaining = n
    stalls = 0
    while remaining > 0:
        size = _t2_piece_size(rng, composition, remaining)
        placed = False
        while size >= 1 and not placed:
            cells = _t2_piece(rng, size, H, W)
            if cells is not None:
                blocked = object_mask | _t2_dilate(target_mask)
                anchor = _t2_fit(rng, blocked, cells)
                if anchor is not None:
                    r0, c0 = anchor
                    for dr, dc in cells:
                        target_mask[r0 + dr, c0 + dc] = True
                    remaining -= size
                    placed = True
            if not placed:
                size = size // 2 if size > 1 else 0
        if not placed:
            stalls += 1
            if stalls > 3:
                return None
    grid[target_mask] = target

    # Independent recount from the displayed grid; the answer is built from it.
    coords = sorted((int(r), int(c)) for r, c in np.argwhere(grid == target))
    if len(coords) != n or n < 1 or n > TASK2_MAX_CELLS:
        return None
    answer = "[" + ",".join(f"({r},{c})" for r, c in coords) + "]"
    return {
        "input": grid_to_list(grid),
        "output": answer,
        "metadata": {
            "query_color_index": target,
            "n_cells": int(n),
            "composition": composition,
            "background": bg,
            "background_style": style,
            "n_other_objects": int(n_placed_objects),
            "n_visible_colors": int(len(np.unique(grid))),
            "grid_h": int(H),
            "grid_w": int(W),
        },
    }


@register("2")
def generate_coordinate_listing(rng: np.random.Generator) -> dict:
    """Task 2: list all coordinates of one colour in row-major order (0-indexed).

    Sizes 1..30 per side, independent.  The number of target cells is sampled
    first (1..60, at most 40% of the canvas) and the target colour is built to
    exactly that count from pieces (isolated dots, 2-8 cell shapes, 9-40 cell
    rectangles / blobs, or a mix) that never touch each other, on top of a
    uniform / noisy / striped background and 0-8 distractor objects in up to 8
    other colours -- so a grid can show all 10 colours.  Nothing is truncated:
    the answer is recounted from the displayed grid.
    """
    for _ in range(MAX_RETRIES * 3):
        h, w = random_grid_size(rng, min_dim=1, max_dim=30)
        example = _t2_build(rng, h, w)
        if example is not None:
            return example
    raise RuntimeError("task 2: could not place the target cells")


# ---------------------------------------------------------------------------
# Task 25: identify the visible background (not a frequency-counting task).
# ---------------------------------------------------------------------------
from generators.background_color_25 import generate_background_color

register("25")(generate_background_color)

from generators.cell_operations_29_30 import generate_count as generate_count_cells

register("30")(generate_count_cells)


from generators.center_cell_75 import generate_center_color

register("75")(generate_center_color)


@register("76")
def generate_color_histogram(rng: np.random.Generator) -> dict:
    """Count all visible color populations in a coherent drawing."""
    from generators.grid_reading_76_80 import generate_histogram
    return generate_histogram(rng)
