"""A shared vocabulary of object shapes for scene generators.

Every builder returns a normalized ``frozenset`` of (row, col) offsets whose
bounding box starts at (0, 0).  Families cover the shapes real ARC grids are
made of: single cells, dominoes, the free polyominoes up to size 5, bars,
solid rectangles/squares, hollow frames, letters (L/T/U/plus/H/staircase),
diagonal chains, compact blobs and large irregular blobs.  Use
``sample_shape`` for a weighted draw, or the builders directly.

Shapes from the ``diagonal``/``diag_pair`` families are only 8-connected; all
other families are 4-connected.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from common.objects import (
    all_orientations,
    bounding_box,
    normalize_shape,
    random_polyomino,
)

Shape = frozenset

# The 12 free pentominoes + 5 free tetrominoes + 2 free triominoes + domino +
# monomino, each in one canonical orientation.
FREE_POLYOMINOES: dict[str, Shape] = {
    "mono": frozenset({(0, 0)}),
    "domino": frozenset({(0, 0), (0, 1)}),
    "I3": frozenset({(0, 0), (0, 1), (0, 2)}),
    "L3": frozenset({(0, 0), (0, 1), (1, 0)}),
    "I4": frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    "O4": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "T4": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    "S4": frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    "L4": frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    "F5": frozenset({(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)}),
    "I5": frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}),
    "L5": frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)}),
    "N5": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)}),
    "P5": frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}),
    "T5": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "U5": frozenset({(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}),
    "V5": frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    "W5": frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}),
    "X5": frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}),
    "Y5": frozenset({(0, 1), (1, 0), (1, 1), (2, 1), (3, 1)}),
    "Z5": frozenset({(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)}),
}

FAMILIES = (
    "polyomino", "bar", "rect", "square", "frame", "L", "T", "U", "plus",
    "H", "staircase", "diagonal", "blob", "big_blob", "scaled",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "polyomino": 0.22, "bar": 0.10, "rect": 0.10, "square": 0.05, "frame": 0.07,
    "L": 0.06, "T": 0.04, "U": 0.04, "plus": 0.04, "H": 0.02, "staircase": 0.04,
    "diagonal": 0.0, "blob": 0.12, "big_blob": 0.06, "scaled": 0.04,
}


def random_orientation(rng: np.random.Generator, shape: Shape) -> Shape:
    orients = all_orientations(shape)
    return orients[int(rng.integers(0, len(orients)))]


def shape_size(shape: Shape) -> tuple[int, int]:
    r0, c0, r1, c1 = bounding_box(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def polyomino(rng: np.random.Generator, size: Optional[int] = None) -> Shape:
    """A free polyomino (1..5 cells) in a random orientation."""
    names = list(FREE_POLYOMINOES)
    if size is not None:
        names = [n for n in names if len(FREE_POLYOMINOES[n]) == size] or names
    name = names[int(rng.integers(0, len(names)))]
    return random_orientation(rng, FREE_POLYOMINOES[name])


def bar(rng: np.random.Generator, length: int, thickness: int = 1) -> Shape:
    cells = {(r, c) for r in range(thickness) for c in range(length)}
    shape = frozenset(cells)
    return shape if rng.random() < 0.5 else normalize_shape({(c, r) for r, c in cells})


def rect(rng: np.random.Generator, h: int, w: int) -> Shape:
    return frozenset((r, c) for r in range(h) for c in range(w))


def frame(rng: np.random.Generator, h: int, w: int, thickness: int = 1) -> Shape:
    """Hollow rectangle; h, w >= 2*thickness + 1."""
    cells = set()
    for r in range(h):
        for c in range(w):
            if r < thickness or r >= h - thickness or c < thickness or c >= w - thickness:
                cells.add((r, c))
    return frozenset(cells)


def letter_L(rng: np.random.Generator, arm_a: int, arm_b: int, thickness: int = 1) -> Shape:
    cells = {(r, c) for r in range(arm_a) for c in range(thickness)}
    cells |= {(r, c) for r in range(arm_a - thickness, arm_a) for c in range(arm_b)}
    return random_orientation(rng, normalize_shape(cells))


def letter_T(rng: np.random.Generator, top: int, stem: int, thickness: int = 1) -> Shape:
    cells = {(r, c) for r in range(thickness) for c in range(top)}
    mid = (top - thickness) // 2
    cells |= {(r, c) for r in range(thickness, stem + thickness) for c in range(mid, mid + thickness)}
    return random_orientation(rng, normalize_shape(cells))


def letter_U(rng: np.random.Generator, height: int, width: int, thickness: int = 1) -> Shape:
    cells = {(r, c) for r in range(height) for c in range(thickness)}
    cells |= {(r, c) for r in range(height) for c in range(width - thickness, width)}
    cells |= {(r, c) for r in range(height - thickness, height) for c in range(width)}
    return random_orientation(rng, normalize_shape(cells))


def plus(rng: np.random.Generator, arm: int, thickness: int = 1) -> Shape:
    size = 2 * arm + thickness
    lo, hi = arm, arm + thickness
    cells = {(r, c) for r in range(size) for c in range(lo, hi)}
    cells |= {(r, c) for r in range(lo, hi) for c in range(size)}
    return frozenset(cells)


def letter_H(rng: np.random.Generator, height: int, width: int, thickness: int = 1) -> Shape:
    cells = {(r, c) for r in range(height) for c in range(thickness)}
    cells |= {(r, c) for r in range(height) for c in range(width - thickness, width)}
    mid = (height - thickness) // 2
    cells |= {(r, c) for r in range(mid, mid + thickness) for c in range(width)}
    return random_orientation(rng, normalize_shape(cells))


def staircase(rng: np.random.Generator, steps: int, step: int = 1) -> Shape:
    """A filled staircase: row k spans columns 0..(k+1)*step-1."""
    cells = set()
    for k in range(steps):
        for r in range(k * step, (k + 1) * step):
            for c in range((k + 1) * step):
                cells.add((r, c))
    return random_orientation(rng, frozenset(cells))


def diagonal(rng: np.random.Generator, length: int) -> Shape:
    """An 8-connected diagonal chain of ``length`` cells."""
    cells = {(i, i) for i in range(length)}
    return random_orientation(rng, frozenset(cells))


def blob(rng: np.random.Generator, size: int, compactness: float = 1.2) -> Shape:
    """A compact 4-connected blob grown with a neighbour-count bias."""
    canvas = 2 * size + 3
    grid = np.zeros((canvas, canvas), dtype=bool)
    start = (canvas // 2, canvas // 2)
    grid[start] = True
    cells = {start}
    nb = np.zeros((canvas, canvas), dtype=np.int16)
    frontier: set[tuple[int, int]] = set()

    def add(cell):
        cells.add(cell)
        grid[cell] = True
        frontier.discard(cell)
        r, c = cell
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < canvas and 0 <= nc < canvas:
                    nb[nr, nc] += 1
                    if (dr == 0 or dc == 0) and not grid[nr, nc]:
                        frontier.add((nr, nc))

    add(start)
    while len(cells) < size and frontier:
        cand = list(frontier)
        weights = np.exp(compactness * np.array([nb[r, c] for r, c in cand], dtype=float))
        weights /= weights.sum()
        add(cand[int(rng.choice(len(cand), p=weights))])
    return normalize_shape(cells)


def big_blob(rng: np.random.Generator, size: int) -> Shape:
    """A large irregular shape: overlapping rectangles with bites taken out."""
    side = max(3, int(np.sqrt(size)) + 2)
    canvas = np.zeros((side + 2, side + 2), dtype=bool)
    for i in range(int(rng.integers(2, 5))):
        rh = int(rng.integers(max(2, side // 3), side + 1))
        rw = int(rng.integers(max(2, side // 3), side + 1))
        r0 = int(rng.integers(0, side + 2 - rh + 1))
        c0 = int(rng.integers(0, side + 2 - rw + 1))
        trial = canvas.copy()
        trial[r0:r0 + rh, c0:c0 + rw] = True
        if i == 0 or _connected(trial):
            canvas = trial
    for _ in range(int(rng.integers(0, 4))):
        r0 = int(rng.integers(0, side + 2))
        c0 = int(rng.integers(0, side + 2))
        bh = int(rng.integers(1, 3))
        bw = int(rng.integers(1, 3))
        trial = canvas.copy()
        trial[r0:r0 + bh, c0:c0 + bw] = False
        if trial.any() and _connected(trial):
            canvas = trial
    cells = {(int(r), int(c)) for r, c in np.argwhere(canvas)}
    return normalize_shape(cells)


def _connected(mask: np.ndarray) -> bool:
    from common.topology import is_connected
    return is_connected(mask, 4)


def scaled(rng: np.random.Generator, shape: Shape, factor: int) -> Shape:
    cells = {(r * factor + dr, c * factor + dc) for r, c in shape for dr in range(factor) for dc in range(factor)}
    return normalize_shape(cells)


def sample_shape(
    rng: np.random.Generator,
    max_cells: int,
    max_h: int,
    max_w: int,
    weights: Optional[dict[str, float]] = None,
    connectivity: int = 4,
    families: Optional[Sequence[str]] = None,
) -> tuple[Shape, str]:
    """Draw a shape from the vocabulary that fits max_cells and max_h×max_w.

    Returns (shape, family).  Families whose smallest instance does not fit
    are skipped; diagonal families are only drawn when ``connectivity`` is 8.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if connectivity == 8 and weights is None:
        w["diagonal"] = 0.08
    if families is not None:
        w = {k: v for k, v in w.items() if k in families}
    if connectivity != 8:
        w.pop("diagonal", None)
    names = [k for k, v in w.items() if v > 0]
    probs = np.array([w[k] for k in names], dtype=float)
    probs /= probs.sum()
    for _ in range(40):
        fam = names[int(rng.choice(len(names), p=probs))]
        shape = _build(rng, fam, max_cells, max_h, max_w)
        if shape is None:
            continue
        sh, sw = shape_size(shape)
        if len(shape) <= max_cells and sh <= max_h and sw <= max_w:
            return shape, fam
    return frozenset({(0, 0)}), "polyomino"


_MIN_SIDE = {
    "polyomino": 1, "bar": 1, "rect": 1, "square": 2, "frame": 3, "L": 2, "T": 3, "U": 3,
    "plus": 3, "H": 3, "staircase": 2, "diagonal": 2, "blob": 2, "big_blob": 4, "scaled": 2,
}
_MIN_CELLS = {
    "polyomino": 1, "bar": 2, "rect": 2, "square": 4, "frame": 8, "L": 3, "T": 4, "U": 5,
    "plus": 5, "H": 7, "staircase": 3, "diagonal": 2, "blob": 4, "big_blob": 12, "scaled": 8,
}


def _build(rng: np.random.Generator, fam: str, max_cells: int, max_h: int, max_w: int) -> Optional[Shape]:
    side = min(max_h, max_w)
    if side < _MIN_SIDE.get(fam, 1) or max_cells < _MIN_CELLS.get(fam, 1):
        return None
    if fam == "bar" and max(max_h, max_w) < 2:
        return None
    if fam == "polyomino":
        size = int(rng.integers(1, min(5, max_cells) + 1))
        return polyomino(rng, size)
    if fam == "bar":
        length = int(rng.integers(2, min(max(max_h, max_w), max_cells, 15) + 1))
        thick = 1 if rng.random() < 0.85 or length < 4 else 2
        return bar(rng, length, thick)
    if fam == "rect":
        h = int(rng.integers(1, min(max_h, 8) + 1))
        w = int(rng.integers(1, min(max_w, 8) + 1))
        if h * w > max_cells or h == w == 1:
            return None
        return rect(rng, h, w)
    if fam == "square":
        s = int(rng.integers(2, min(side, 7) + 1)) if side >= 2 else 1
        if s * s > max_cells:
            return None
        return rect(rng, s, s)
    if fam == "frame":
        if side < 3:
            return None
        h = int(rng.integers(3, min(max_h, 9) + 1))
        w = int(rng.integers(3, min(max_w, 9) + 1))
        t = 1 if rng.random() < 0.8 or min(h, w) < 5 else 2
        return frame(rng, h, w, t)
    if fam == "L":
        a = int(rng.integers(2, min(side, 7) + 1))
        b = int(rng.integers(2, min(side, 7) + 1))
        t = 1 if rng.random() < 0.8 or min(a, b) < 4 else 2
        return letter_L(rng, a, b, t)
    if fam == "T":
        top = int(rng.integers(3, min(side, 7) + 1))
        stem = int(rng.integers(2, min(side, 6) + 1))
        t = 1 if rng.random() < 0.85 or top < 5 else 2
        return letter_T(rng, top, stem, t)
    if fam == "U":
        h = int(rng.integers(2, min(side, 6) + 1))
        w = int(rng.integers(3, min(side, 7) + 1))
        return letter_U(rng, h, w, 1)
    if fam == "plus":
        arm = int(rng.integers(1, max(1, (side - 1) // 2) + 1))
        t = 1 if rng.random() < 0.85 or side < 7 else 2
        return plus(rng, arm, t)
    if fam == "H":
        h = int(rng.integers(3, min(side, 7) + 1))
        w = int(rng.integers(3, min(side, 7) + 1))
        return letter_H(rng, h, w, 1)
    if fam == "staircase":
        steps = int(rng.integers(2, min(side, 6) + 1))
        return staircase(rng, steps, 1)
    if fam == "diagonal":
        return diagonal(rng, int(rng.integers(2, min(side, 8) + 1)))
    if fam == "blob":
        size = int(rng.integers(4, min(max_cells, 25) + 1)) if max_cells >= 4 else None
        return blob(rng, size, compactness=float(rng.uniform(0.6, 2.0))) if size else None
    if fam == "big_blob":
        if max_cells < 12:
            return None
        return big_blob(rng, int(rng.integers(12, min(max_cells, 60) + 1)))
    if fam == "scaled":
        base = polyomino(rng, int(rng.integers(2, 6)))
        k = int(rng.integers(2, 4))
        return scaled(rng, base, k)
    return None
