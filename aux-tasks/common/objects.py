"""Object analysis utilities: connected components, bounding boxes, shape stamps, polyominoes."""
from __future__ import annotations

from collections import deque
from typing import List, Set, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Polyomino library — canonical free polyominoes up to size 5
# Each is a frozenset of (row, col) offsets normalized to top-left origin.
# ---------------------------------------------------------------------------

_TRIOMINOES = [
    frozenset([(0,0),(0,1),(0,2)]),            # I
    frozenset([(0,0),(0,1),(1,0)]),            # L
]

_TETROMINOES = [
    frozenset([(0,0),(0,1),(0,2),(0,3)]),      # I
    frozenset([(0,0),(0,1),(1,0),(1,1)]),      # O
    frozenset([(0,0),(0,1),(0,2),(1,1)]),      # T
    frozenset([(0,0),(0,1),(1,1),(1,2)]),      # S
    frozenset([(0,1),(0,2),(1,0),(1,1)]),      # Z
    frozenset([(0,0),(0,1),(0,2),(1,0)]),      # L
    frozenset([(0,0),(0,1),(0,2),(1,2)]),      # J
]

_PENTOMINOES = [
    frozenset([(0,0),(0,1),(0,2),(0,3),(0,4)]),  # I
    frozenset([(0,0),(0,1),(0,2),(0,3),(1,0)]),  # L
    frozenset([(0,0),(0,1),(0,2),(0,3),(1,3)]),  # J
    frozenset([(0,0),(0,1),(0,2),(1,0),(1,1)]),  # P
    frozenset([(0,0),(0,1),(0,2),(1,1),(1,2)]),  # b (mirror P)
    frozenset([(0,0),(0,1),(0,2),(1,0),(2,0)]),  # V
    frozenset([(0,0),(0,1),(1,1),(1,2),(2,2)]),  # W
    frozenset([(0,0),(0,1),(0,2),(1,1),(2,1)]),  # Y-ish (T)
    frozenset([(0,1),(1,0),(1,1),(1,2),(2,1)]),  # + (plus)
    frozenset([(0,0),(1,0),(1,1),(2,1),(2,2)]),  # S big
    frozenset([(0,0),(0,1),(1,0),(2,0),(2,1)]),  # U-ish
    frozenset([(0,0),(0,1),(0,2),(1,2),(2,2)]),  # corner
]

POLYOMINOES = {
    3: _TRIOMINOES,
    4: _TETROMINOES,
    5: _PENTOMINOES,
}

DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DELTAS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def bounding_box(cells) -> tuple:
    """Return (min_row, min_col, max_row, max_col) for a set of (row, col) cells."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_shape(cells) -> frozenset:
    """Translate a set of cells so min_row=0, min_col=0."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def rotate_shape_cw(cells) -> frozenset:
    """Rotate a normalized shape 90 degrees clockwise."""
    rotated = frozenset((c, -r) for r, c in cells)
    return normalize_shape(rotated)


def reflect_shape(cells) -> frozenset:
    """Reflect a shape horizontally."""
    reflected = frozenset((r, -c) for r, c in cells)
    return normalize_shape(reflected)


def all_orientations(cells) -> list:
    """Return all 8 orientations (4 rotations x 2 reflections) of a shape."""
    orientations = set()
    shape = normalize_shape(cells)
    for _ in range(4):
        orientations.add(shape)
        orientations.add(reflect_shape(shape))
        shape = rotate_shape_cw(shape)
    return list(orientations)


def random_polyomino(
    rng: np.random.Generator,
    size: int,
    connectivity: int = 4,
) -> frozenset:
    """Grow a random connected polyomino of the given size via random walk."""
    deltas = DELTAS_4 if connectivity == 4 else DELTAS_8
    cells = {(0, 0)}
    frontier = set()
    for dr, dc in deltas:
        frontier.add((dr, dc))

    while len(cells) < size:
        if not frontier:
            break
        candidates = list(frontier)
        idx = int(rng.integers(0, len(candidates)))
        cell = candidates[idx]
        cells.add(cell)
        frontier.discard(cell)
        for dr, dc in deltas:
            nb = (cell[0] + dr, cell[1] + dc)
            if nb not in cells:
                frontier.add(nb)

    return normalize_shape(cells)


def random_shape(
    rng: np.random.Generator,
    size_range: tuple = (3, 8),
    connectivity: int = 4,
    allow_named: bool = True,
) -> frozenset:
    """Generate a random shape — either a named polyomino or a random walk."""
    lo, hi = size_range
    size = int(rng.integers(lo, hi + 1))

    # 50% chance of using a named polyomino if available and allowed
    if allow_named and size in POLYOMINOES and rng.random() < 0.5:
        shapes = POLYOMINOES[size]
        base = shapes[int(rng.integers(0, len(shapes)))]
        # Random orientation
        orients = all_orientations(base)
        return orients[int(rng.integers(0, len(orients)))]

    return random_polyomino(rng, size, connectivity)


def random_square_shape(
    rng: np.random.Generator,
    size_range: tuple = (3, 8),
    connectivity: int = 4,
    max_attempts: int = 50,
) -> frozenset:
    """Generate a random shape whose bounding box is square (height == width).

    Useful for rotation tasks where non-square bboxes cause ambiguity.
    """
    for _ in range(max_attempts):
        shape = random_shape(rng, size_range=size_range, connectivity=connectivity)
        r0, c0, r1, c1 = bounding_box(shape)
        if (r1 - r0) == (c1 - c0):
            return shape
    # Fallback: pad a shape to make its bbox square
    shape = random_shape(rng, size_range=size_range, connectivity=connectivity)
    r0, c0, r1, c1 = bounding_box(shape)
    sh = r1 - r0
    sw = c1 - c0
    if sh < sw:
        # Add a cell to extend height
        shape = set(shape)
        shape.add((r1 + (sw - sh), c0))
        shape = normalize_shape(shape)
    elif sw < sh:
        shape = set(shape)
        shape.add((r0, c1 + (sh - sw)))
        shape = normalize_shape(shape)
    return shape


def place_shape_on_grid(
    grid: np.ndarray,
    rng: np.random.Generator,
    shape: frozenset,
    color: int,
    bg_color: int,
    margin: int = 0,
    max_attempts: int = 30,
    gap: int = 0,
    require_touch: bool = False,
) -> bool:
    """Try to place a shape on the grid at a random position. Returns True if successful.

    If *gap* > 0, the placement also requires a *gap*-cell (Chebyshev) ring of
    background around the shape, so the placed object does not touch any existing
    object. *gap* defaults to 0 (touching allowed), preserving existing callers.

    If *require_touch* is set, the placement must instead be 8-adjacent to at
    least one existing object of a *different* color and to no object of the same
    color, i.e. it deliberately touches another object without merging into it.
    """
    h, w = grid.shape
    r0, c0, r1, c1 = bounding_box(shape)
    sh = r1 - r0 + 1
    sw = c1 - c0 + 1

    for _ in range(max_attempts):
        tr = int(rng.integers(margin, max(margin + 1, h - sh - margin + 1)))
        tc = int(rng.integers(margin, max(margin + 1, w - sw - margin + 1)))

        # Check if all cells are on bg
        ok = True
        placed_cells = []
        cellset = set()
        for sr, sc in shape:
            nr, nc = tr + sr - r0, tc + sc - c0
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                ok = False
                break
            if grid[nr, nc] != bg_color:
                ok = False
                break
            placed_cells.append((nr, nc))
            cellset.add((nr, nc))

        # Optionally require a clear background ring so objects do not touch.
        if ok and gap > 0:
            for nr, nc in placed_cells:
                for dr in range(-gap, gap + 1):
                    for dc in range(-gap, gap + 1):
                        br, bc = nr + dr, nc + dc
                        if (0 <= br < h and 0 <= bc < w
                                and (br, bc) not in cellset
                                and grid[br, bc] != bg_color):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

        # Optionally require the placement to deliberately touch a different
        # colored object (and never a same-colored one, which would merge).
        if ok and require_touch:
            touches_other = False
            same_color_adj = False
            for nr, nc in placed_cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        br, bc = nr + dr, nc + dc
                        if (0 <= br < h and 0 <= bc < w
                                and (br, bc) not in cellset):
                            v = int(grid[br, bc])
                            if v != bg_color:
                                if v == color:
                                    same_color_adj = True
                                else:
                                    touches_other = True
            if same_color_adj or not touches_other:
                ok = False

        if ok:
            for nr, nc in placed_cells:
                grid[nr, nc] = color
            return True

    return False


def connected_components(
    grid: np.ndarray, connectivity: int = 4, background: Optional[int] = 0
) -> list:
    """Extract connected components from the grid via BFS.

    Returns list of dicts sorted by (min_row, min_col):
        {"color": int, "cells": set[tuple[int,int]], "bbox": (r0,c0,r1,c1)}
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    deltas = DELTAS_4 if connectivity == 4 else DELTAS_8

    components = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            if background is not None and color == background:
                visited[r, c] = True
                continue
            cells = set()
            queue = deque()
            queue.append((r, c))
            visited[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            bbox = bounding_box(cells)
            components.append({"color": color, "cells": cells, "bbox": bbox})

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return components


def stamp_rectangle(
    grid: np.ndarray,
    r: int,
    c: int,
    h: int,
    w: int,
    color: int,
    filled: bool = True,
) -> np.ndarray:
    """Draw a rectangle on the grid at (r, c) with dimensions h x w."""
    if filled:
        grid[r : r + h, c : c + w] = color
    else:
        grid[r, c : c + w] = color
        grid[r + h - 1, c : c + w] = color
        grid[r : r + h, c] = color
        grid[r : r + h, c + w - 1] = color
    return grid
