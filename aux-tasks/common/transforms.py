"""Transform utilities for ARC grids: rotation, reflection, magnification, translation."""
from __future__ import annotations

import numpy as np


def rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    """Rotate *grid* by k*90 degrees **clockwise** (k=1,2,3).

    np.rot90 rotates counter-clockwise, so clockwise-k == np.rot90(grid, 4-k).
    """
    k = k % 4
    if k == 0:
        return grid.copy()
    return np.rot90(grid, k=4 - k).copy()


def reflect_grid(grid: np.ndarray, axis: str) -> np.ndarray:
    """Reflect *grid* across the specified axis.

    axis:
        "horizontal"     -- flip rows (up-down)
        "vertical"       -- flip columns (left-right)
        "main_diagonal"  -- transpose (swap rows/cols)
        "anti_diagonal"  -- rotate 90 CW then flip up-down  (equiv. transpose + flip both)
    """
    if axis == "horizontal":
        return np.flipud(grid).copy()
    elif axis == "vertical":
        return np.fliplr(grid).copy()
    elif axis == "main_diagonal":
        return grid.T.copy()
    elif axis == "anti_diagonal":
        # Anti-diagonal reflection: reverse both axes then transpose
        return np.flipud(np.fliplr(grid)).T.copy()
    else:
        raise ValueError(f"Unknown axis: {axis!r}")


def magnify_grid(grid: np.ndarray, scale: int) -> np.ndarray:
    """Replace each cell with a scale x scale block of the same colour."""
    return np.kron(grid, np.ones((scale, scale), dtype=grid.dtype))


def demagnify_grid(grid: np.ndarray, scale: int) -> np.ndarray:
    """Compress each scale x scale block to one cell (inverse of magnify).

    Assumes every block is monochrome -- takes the top-left element of each block.
    """
    h, w = grid.shape
    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"Grid shape {grid.shape} not evenly divisible by scale {scale}"
        )
    return grid[::scale, ::scale].copy()


def translate_grid(
    grid: np.ndarray, dr: int, dc: int, bg: int = 0
) -> np.ndarray:
    """Shift all content by (dr, dc); exposed area filled with *bg*, out-of-bounds clipped."""
    h, w = grid.shape
    out = np.full_like(grid, bg)
    # Source region (what we read from grid)
    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)
    # Destination region (where we write in out)
    dst_r0 = max(0, dr)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c0 = max(0, dc)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    if dst_r1 > dst_r0 and dst_c1 > dst_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
    return out
