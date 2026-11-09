"""Organic silhouettes, hole carving, outlines and topology oracles (numpy only).

Shared by the topology family (tasks 22 count-holes, 36 open/closed, 37 close
the gap) and by any generator that needs an enclosure with a known interior.

Conventions
-----------
* Masks are boolean ``np.ndarray`` of shape (H, W); ``True`` = object cell.
* "Outside" is every non-object cell that can reach the array border through
  non-object cells moving up/down/left/right.  A *hole* is a 4-connected
  non-object region that cannot.  Diagonal contact never connects regions.
* Every silhouette builder works on a canvas that already includes a 1-cell
  empty margin, so erosion / outside detection at the array edge is exact.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional, Sequence

import numpy as np

Cell = tuple[int, int]

DELTAS_4: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DELTAS_8: tuple[Cell, ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)


# ---------------------------------------------------------------------------
# Morphology and flood fills
# ---------------------------------------------------------------------------

def dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def dilate8(mask: np.ndarray) -> np.ndarray:
    out = dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def erode4(mask: np.ndarray) -> np.ndarray:
    """4-neighbour erosion; cells outside the array count as empty."""
    padded = np.pad(mask, 1, constant_values=False)
    out = padded[1:-1, 1:-1].copy()
    out &= padded[:-2, 1:-1]
    out &= padded[2:, 1:-1]
    out &= padded[1:-1, :-2]
    out &= padded[1:-1, 2:]
    return out


def neighbour_count8(mask: np.ndarray) -> np.ndarray:
    """Number of filled 8-neighbours of every cell (array edge counts as empty)."""
    p = np.pad(mask.astype(np.int16), 1, constant_values=0)
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )


def flood_from_border(passable: np.ndarray) -> np.ndarray:
    """Cells reachable from the array border through ``passable`` cells (4-conn)."""
    h, w = passable.shape
    reached = np.zeros_like(passable)
    reached[0, :] = passable[0, :]
    reached[-1, :] = passable[-1, :]
    reached[:, 0] = passable[:, 0]
    reached[:, -1] = passable[:, -1]
    if not reached.any():
        return reached
    while True:
        grown = dilate4(reached) & passable
        if grown.sum() == reached.sum():
            return grown
        reached = grown


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label connected ``True`` regions; returns (labels starting at 1, count)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    deltas = DELTAS_4 if connectivity == 4 else DELTAS_8
    n = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if labels[r0, c0]:
            continue
        n += 1
        labels[r0, c0] = n
        dq = deque([(int(r0), int(c0))])
        while dq:
            r, c = dq.popleft()
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not labels[nr, nc]:
                    labels[nr, nc] = n
                    dq.append((nr, nc))
    return labels, n


def component_count(mask: np.ndarray, connectivity: int = 4) -> int:
    return label_components(mask, connectivity)[1]


def is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    return bool(mask.any()) and component_count(mask, connectivity) == 1


def largest_component(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    labels, n = label_components(mask, connectivity)
    if n <= 1:
        return mask.copy()
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (int(np.argmax(sizes)) + 1)


def outside_mask(object_mask: np.ndarray) -> np.ndarray:
    """Non-object cells connected to the array border."""
    return flood_from_border(~object_mask)


def hole_mask(object_mask: np.ndarray) -> np.ndarray:
    """All enclosed non-object cells."""
    passable = ~object_mask
    return passable & ~flood_from_border(passable)


def holes(object_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the enclosed regions; returns (labels, count)."""
    return label_components(hole_mask(object_mask), 4)


def count_holes(object_mask: np.ndarray) -> int:
    return holes(object_mask)[1]


def count_holes_in_grid(grid: np.ndarray, obj_color: int) -> int:
    """The task-22 oracle: holes of the set of cells equal to ``obj_color``."""
    return count_holes(np.asarray(grid) == obj_color)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return mask | hole_mask(mask)


def trim(mask: np.ndarray, margin: int = 0) -> np.ndarray:
    """Crop to the bounding box of ``True`` cells, optionally re-padding."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty mask")
    out = mask[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    if margin:
        out = np.pad(out, margin, constant_values=False)
    return out


def min_wall_thickness(object_mask: np.ndarray) -> Optional[int]:
    """Fewest object cells a 4-path must cross from any hole to the outside.

    Returns ``None`` when the object has no holes.  A value of 1 means some
    hole cell and some outside cell are separated by a single object cell.
    """
    hmask = hole_mask(object_mask)
    if not hmask.any():
        return None
    out = outside_mask(object_mask)
    h, w = object_mask.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    dq: deque[Cell] = deque()
    for r, c in zip(*np.nonzero(hmask)):
        dist[r, c] = 0
        dq.append((int(r), int(c)))
    best: Optional[int] = None
    while dq:
        r, c = dq.popleft()
        d = int(dist[r, c])
        if best is not None and d >= best:
            break
        for dr, dc in DELTAS_4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if out[nr, nc]:
                if best is None or d < best:
                    best = d
                continue
            if object_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = d + 1
                dq.append((nr, nc))
    return best


# ---------------------------------------------------------------------------
# Region growth (used for hole shapes and small blobs)
# ---------------------------------------------------------------------------

def grow_region(
    rng: np.random.Generator,
    allowed: np.ndarray,
    seed: Cell,
    target: int,
    compactness: float = 1.0,
) -> set[Cell]:
    """Grow a 4-connected region from ``seed`` inside ``allowed``.

    ``compactness`` > 0 prefers frontier cells with many already-grown
    8-neighbours (round blobs); 0 is a uniform random walk (jagged shapes);
    negative values prefer thin, tendril-like growth.
    """
    h, w = allowed.shape
    if not allowed[seed]:
        return set()
    cells: set[Cell] = {seed}
    nb = np.zeros((h, w), dtype=np.int16)
    frontier: set[Cell] = set()

    def add(cell: Cell) -> None:
        cells.add(cell)
        frontier.discard(cell)
        r, c = cell
        for dr, dc in DELTAS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nb[nr, nc] += 1
                if (dr == 0 or dc == 0) and allowed[nr, nc] and (nr, nc) not in cells:
                    frontier.add((nr, nc))

    add(seed)
    while len(cells) < target and frontier:
        cand = list(frontier)
        if compactness != 0.0 and len(cand) > 1:
            weights = np.exp(compactness * np.array([nb[r, c] for r, c in cand], dtype=float))
            weights /= weights.sum()
            pick = cand[int(rng.choice(len(cand), p=weights))]
        else:
            pick = cand[int(rng.integers(0, len(cand)))]
        add(pick)
    return cells


def cells_to_mask(cells: Iterable[Cell], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


# ---------------------------------------------------------------------------
# Silhouette builders.  All return a boolean canvas with an empty 1-cell margin
# whose object is a single 4-connected component without holes.
# ---------------------------------------------------------------------------

def _finish(mask: np.ndarray) -> Optional[np.ndarray]:
    if not mask.any():
        return None
    mask = fill_holes(largest_component(mask, 4))
    return trim(mask, margin=1)


def radial_blob(
    rng: np.random.Generator,
    h: int,
    w: int,
    lobes: Optional[int] = None,
    irregularity: float = 0.35,
) -> Optional[np.ndarray]:
    """A rounded organic blob: an ellipse whose radius wobbles with the angle."""
    if lobes is None:
        lobes = int(rng.integers(3, 9))
    knots = rng.uniform(1.0 - irregularity, 1.0, size=lobes)
    phase = float(rng.uniform(0, 2 * np.pi))
    rr, cc = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (rr - cy) / (h / 2.0)
    nx = (cc - cx) / (w / 2.0)
    ang = (np.arctan2(ny, nx) + phase) % (2 * np.pi)
    pos = ang / (2 * np.pi) * lobes
    i0 = np.floor(pos).astype(int) % lobes
    i1 = (i0 + 1) % lobes
    t = pos - np.floor(pos)
    # smoothstep between knots so lobes are round, not pointy
    t = t * t * (3 - 2 * t)
    radius = knots[i0] * (1 - t) + knots[i1] * t
    inside = np.sqrt(ny * ny + nx * nx) <= radius
    return _finish(inside)


def cellular_blob(
    rng: np.random.Generator,
    h: int,
    w: int,
    fill: float = 0.58,
    iterations: int = 4,
) -> Optional[np.ndarray]:
    """Cave-like blob from noise smoothed by a majority cellular automaton."""
    noise = rng.random((h, w)) < fill
    noise[0, :] = noise[-1, :] = noise[:, 0] = noise[:, -1] = False
    mask = noise
    for _ in range(iterations):
        cnt = neighbour_count8(mask)
        mask = (cnt >= 5) | (mask & (cnt >= 4))
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return _finish(mask)


def ellipse_mask(h: int, w: int) -> np.ndarray:
    rr, cc = np.mgrid[0:h, 0:w]
    ny = (rr - (h - 1) / 2.0) / (h / 2.0)
    nx = (cc - (w - 1) / 2.0) / (w / 2.0)
    return np.pad(ny * ny + nx * nx <= 1.0, 1, constant_values=False)


def diamond_mask(h: int, w: int) -> np.ndarray:
    rr, cc = np.mgrid[0:h, 0:w]
    ny = np.abs(rr - (h - 1) / 2.0) / (h / 2.0)
    nx = np.abs(cc - (w - 1) / 2.0) / (w / 2.0)
    return np.pad(ny + nx <= 1.0, 1, constant_values=False)


def rounded_rect_mask(h: int, w: int, radius: int) -> np.ndarray:
    """A rectangle with its corners cut by a Manhattan radius."""
    rr, cc = np.mgrid[0:h, 0:w]
    dy = np.maximum(0, np.maximum(radius - rr, rr - (h - 1 - radius)))
    dx = np.maximum(0, np.maximum(radius - cc, cc - (w - 1 - radius)))
    return np.pad(dy + dx <= radius, 1, constant_values=False)


def union_of_rects(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_rects: int = 3,
) -> Optional[np.ndarray]:
    """Union of overlapping/adjacent rectangles inside an h×w box."""
    mask = np.zeros((h, w), dtype=bool)
    for i in range(n_rects):
        rh = int(rng.integers(max(2, h // 4), h + 1))
        rw = int(rng.integers(max(2, w // 4), w + 1))
        for _ in range(30):
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            trial = mask.copy()
            trial[r0:r0 + rh, c0:c0 + rw] = True
            if i == 0 or is_connected(trial, 4):
                mask = trial
                break
    return _finish(mask)


def star_polygon(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_vertices: Optional[int] = None,
    spikiness: float = 0.45,
) -> Optional[np.ndarray]:
    """Angular organic shape: a star-shaped polygon with random vertex radii."""
    if n_vertices is None:
        n_vertices = int(rng.integers(5, 11))
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n_vertices))
    radii = rng.uniform(1.0 - spikiness, 1.0, size=n_vertices)
    rr, cc = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (rr - cy) / (h / 2.0)
    nx = (cc - cx) / (w / 2.0)
    ang = np.arctan2(ny, nx) % (2 * np.pi)
    dist = np.sqrt(ny * ny + nx * nx)
    idx = np.searchsorted(angles, ang) % n_vertices
    a0 = angles[idx - 1]
    a1 = angles[idx]
    span = (a1 - a0) % (2 * np.pi)
    span = np.where(span == 0, 2 * np.pi, span)
    t = ((ang - a0) % (2 * np.pi)) / span
    radius = radii[idx - 1] * (1 - t) + radii[idx] * t
    return _finish(dist <= radius)


def add_bumps(
    rng: np.random.Generator,
    mask: np.ndarray,
    n_bumps: int,
    max_size: int = 4,
    bounds: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Attach small rectangles to the outer boundary (keeps holes intact).

    ``bounds`` = (max_h, max_w) of the trimmed object; bumps that would grow
    the object past it are skipped.
    """
    mask = np.pad(mask, max_size, constant_values=False)
    holes_before = count_holes(mask)
    out = outside_mask(mask)
    for _ in range(n_bumps):
        ring = dilate4(mask) & out
        cands = np.argwhere(ring)
        if cands.size == 0:
            break
        r, c = cands[int(rng.integers(0, len(cands)))]
        bh = int(rng.integers(1, max_size + 1))
        bw = int(rng.integers(1, max_size + 1))
        r0 = int(r) - int(rng.integers(0, bh))
        c0 = int(c) - int(rng.integers(0, bw))
        trial = mask.copy()
        trial[max(0, r0): r0 + bh, max(0, c0): c0 + bw] = True
        if count_holes(trial) != holes_before:
            continue
        if bounds is not None:
            th, tw = trim(trial).shape
            if th > bounds[0] or tw > bounds[1]:
                continue
        mask = trial
        out = outside_mask(mask)
    return trim(mask, margin=1)


def add_notches(
    rng: np.random.Generator,
    mask: np.ndarray,
    n_notches: int,
    max_size: int = 4,
) -> np.ndarray:
    """Carve small bays into the outer boundary that stay open to the outside."""
    holes_before = count_holes(mask)
    for _ in range(n_notches):
        out = outside_mask(mask)
        ring = mask & dilate4(out)
        cands = np.argwhere(ring)
        if cands.size == 0:
            break
        r, c = cands[int(rng.integers(0, len(cands)))]
        allowed = mask.copy()
        size = int(rng.integers(1, max_size * 2 + 1))
        cells = grow_region(rng, allowed, (int(r), int(c)), size, compactness=float(rng.uniform(0.0, 1.5)))
        trial = mask.copy()
        for rr, cc in cells:
            trial[rr, cc] = False
        if trial.any() and is_connected(trial, 4) and count_holes(trial) == holes_before:
            mask = trial
    return trim(mask, margin=1)


def random_silhouette(
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
    family: Optional[str] = None,
    min_h: int = 3,
    min_w: int = 3,
) -> Optional[tuple[np.ndarray, str]]:
    """A single connected hole-free blob that fits in max_h×max_w (plus margin).

    Returns (mask with 1-cell margin, family name) or None.
    """
    if max_h < min_h or max_w < min_w:
        return None
    if family is None:
        family = str(rng.choice(
            ["radial", "cellular", "rects", "star", "ellipse", "diamond", "rounded", "rect"],
            p=[0.22, 0.16, 0.16, 0.14, 0.08, 0.07, 0.09, 0.08],
        ))
    for _ in range(4):
        h = int(rng.integers(min_h, max_h + 1))
        w = int(rng.integers(min_w, max_w + 1))
        mask: Optional[np.ndarray]
        if family == "radial":
            mask = radial_blob(rng, h, w, irregularity=float(rng.uniform(0.15, 0.5)))
        elif family == "cellular":
            if h < 6 or w < 6:
                mask = radial_blob(rng, h, w)
            else:
                mask = cellular_blob(rng, h + 2, w + 2, fill=float(rng.uniform(0.55, 0.66)),
                                     iterations=int(rng.integers(3, 6)))
        elif family == "rects":
            mask = union_of_rects(rng, h, w, n_rects=int(rng.integers(2, 6)))
        elif family == "star":
            mask = star_polygon(rng, h, w, spikiness=float(rng.uniform(0.25, 0.55)))
        elif family == "ellipse":
            mask = ellipse_mask(h, w)
        elif family == "diamond":
            mask = diamond_mask(h, w)
        elif family == "rounded":
            mask = rounded_rect_mask(h, w, radius=int(rng.integers(1, max(2, min(h, w) // 3) + 1)))
        else:
            mask = np.pad(np.ones((h, w), dtype=bool), 1, constant_values=False)
        if mask is None:
            continue
        mask = _finish(mask)
        if mask is None:
            continue
        if rng.random() < 0.5:
            mask = add_bumps(
                rng, mask, n_bumps=int(rng.integers(1, 4)),
                max_size=max(1, min(4, min(h, w) // 3)), bounds=(max_h, max_w),
            )
        if rng.random() < 0.5:
            mask = add_notches(rng, mask, n_notches=int(rng.integers(1, 4)), max_size=max(1, min(3, min(h, w) // 3)))
        if rng.random() < 0.5:
            rotated = np.rot90(mask, int(rng.integers(1, 4)))
            if rotated.shape[0] - 2 <= max_h and rotated.shape[1] - 2 <= max_w:
                mask = np.ascontiguousarray(rotated)
        mask = _finish(mask)
        if mask is None:
            continue
        inner_h, inner_w = mask.shape[0] - 2, mask.shape[1] - 2
        if inner_h > max_h or inner_w > max_w or inner_h < min_h or inner_w < min_w:
            continue
        return mask, family
    return None


# ---------------------------------------------------------------------------
# Rooms: an object that is a lattice of walls around several enclosed rooms.
# ---------------------------------------------------------------------------

def rooms_partition(
    rng: np.random.Generator,
    inner_h: int,
    inner_w: int,
    n_rooms: int,
    wall: int = 1,
    frame: int = 1,
    min_room: int = 1,
) -> Optional[np.ndarray]:
    """Frame + internal walls partitioning the interior into exactly n rooms.

    Returns the object mask (with a 1-cell empty margin) or None if the
    interior is too small for the requested rooms.
    """
    rects: list[tuple[int, int, int, int]] = [(0, 0, inner_h, inner_w)]  # r0, c0, h, w
    guard = 0
    while len(rects) < n_rooms and guard < 200:
        guard += 1
        # split the room with the most slack, with some randomness
        order = sorted(range(len(rects)), key=lambda i: -(rects[i][2] * rects[i][3] + rng.uniform(0, 4)))
        done = False
        for idx in order:
            r0, c0, rh, rw = rects[idx]
            options = []
            if rh >= 2 * min_room + wall:
                options.append("h")
            if rw >= 2 * min_room + wall:
                options.append("v")
            if not options:
                continue
            axis = str(rng.choice(options))
            if axis == "h":
                cut = int(rng.integers(min_room, rh - wall - min_room + 1))
                a = (r0, c0, cut, rw)
                b = (r0 + cut + wall, c0, rh - cut - wall, rw)
            else:
                cut = int(rng.integers(min_room, rw - wall - min_room + 1))
                a = (r0, c0, rh, cut)
                b = (r0, c0 + cut + wall, rh, rw - cut - wall)
            rects[idx] = a
            rects.append(b)
            done = True
            break
        if not done:
            return None
    if len(rects) != n_rooms:
        return None
    H = inner_h + 2 * frame
    W = inner_w + 2 * frame
    mask = np.ones((H, W), dtype=bool)
    for r0, c0, rh, rw in rects:
        mask[frame + r0: frame + r0 + rh, frame + c0: frame + c0 + rw] = False
    return np.pad(mask, 1, constant_values=False)


# ---------------------------------------------------------------------------
# Hole carving
# ---------------------------------------------------------------------------

HOLE_KINDS = ("cell", "bar", "rect", "poly", "blob", "big")


def _rect_placements(allowed: np.ndarray, rh: int, rw: int) -> np.ndarray:
    """Top-left corners where an rh×rw block lies entirely inside ``allowed``."""
    h, w = allowed.shape
    if rh > h or rw > w:
        return np.zeros((0, 2), dtype=int)
    integral = np.pad(np.cumsum(np.cumsum(allowed.astype(np.int32), axis=0), axis=1), ((1, 0), (1, 0)))
    total = (
        integral[rh:, rw:] - integral[:-rh, rw:] - integral[rh:, :-rw] + integral[:-rh, :-rw]
    )
    return np.argwhere(total == rh * rw)


def carveable_cells(
    object_mask: np.ndarray,
    existing_holes: np.ndarray,
    min_wall: int = 1,
    diagonal_ok: bool = False,
) -> np.ndarray:
    """Object cells that may become hole cells without leaking or merging."""
    out = outside_mask(object_mask)
    forbidden = out.copy()
    for _ in range(min_wall):
        forbidden = dilate4(forbidden)
    if existing_holes.any():
        forbidden |= dilate4(existing_holes) if diagonal_ok else dilate8(existing_holes)
    return object_mask & ~forbidden


def carve_holes(
    rng: np.random.Generator,
    object_mask: np.ndarray,
    plan: Sequence[tuple[str, int]],
    min_wall: int = 1,
    diagonal_ok: bool = False,
    attempts_per_hole: int = 16,
    seed_bias: float = 0.0,
    object_connectivity: int = 4,
    strict: bool = True,
) -> Optional[tuple[np.ndarray, list[set[Cell]]]]:
    """Carve one hole per plan entry ``(kind, size)`` from a hole-free object.

    ``kind`` is one of HOLE_KINDS; ``size`` is a cell count (for "big" it is
    the percentage 1-100 of the currently carveable area to fill).  Each hole
    is verified to be a separate 4-connected region that keeps the object one
    component under ``object_connectivity`` (4, or 8 to allow diagonal-only
    joins in the object).  ``seed_bias`` > 0 prefers seeds deep inside the
    object, < 0 prefers seeds near the outer wall.  Returns (new object mask,
    list of hole cell sets); with ``strict`` a hole that cannot be placed makes
    the call return None, otherwise the holes carved so far are returned.
    """
    mask = object_mask.copy()
    holes_so_far = np.zeros_like(mask)
    carved: list[set[Cell]] = []
    h, w = mask.shape
    for kind, size in plan:
        placed: Optional[set[Cell]] = None
        for _ in range(attempts_per_hole):
            allowed = carveable_cells(mask, holes_so_far, min_wall=min_wall, diagonal_ok=diagonal_ok)
            n_allowed = int(allowed.sum())
            if n_allowed == 0:
                break
            cells: set[Cell] = set()
            if kind in ("cell", "bar", "rect"):
                if kind == "cell":
                    rh, rw = 1, 1
                elif kind == "bar":
                    length = max(1, size)
                    rh, rw = (1, length) if rng.random() < 0.5 else (length, 1)
                else:
                    rh = max(1, int(rng.integers(1, max(2, int(np.sqrt(size * 2))) + 1)))
                    rw = max(1, int(round(size / rh)))
                    if rng.random() < 0.5:
                        rh, rw = rw, rh
                spots = _rect_placements(allowed, rh, rw)
                if len(spots) == 0:
                    # shrink and retry
                    size = max(1, size - max(1, size // 3))
                    continue
                r0, c0 = spots[int(rng.integers(0, len(spots)))]
                cells = {(int(r0) + a, int(c0) + b) for a in range(rh) for b in range(rw)}
            else:
                seeds = np.argwhere(allowed)
                if seed_bias != 0.0:
                    depth = np.zeros((h, w), dtype=np.int32)
                    layer = allowed.copy()
                    d = 0
                    while layer.any():
                        d += 1
                        depth[layer] = d
                        layer = erode4(layer)
                    dv = depth[seeds[:, 0], seeds[:, 1]].astype(float)
                    weights = np.exp(seed_bias * dv)
                    weights /= weights.sum()
                    seed = tuple(int(v) for v in seeds[int(rng.choice(len(seeds), p=weights))])
                else:
                    seed = tuple(int(v) for v in seeds[int(rng.integers(0, len(seeds)))])
                if kind == "big":
                    target = max(2, int(round(n_allowed * min(100, max(1, size)) / 100.0)))
                    compact = float(rng.uniform(1.2, 2.5))
                elif kind == "blob":
                    target = max(1, size)
                    compact = float(rng.uniform(0.8, 2.0))
                else:  # poly
                    target = max(1, size)
                    compact = float(rng.uniform(-0.3, 0.4))
                cells = grow_region(rng, allowed, seed, target, compactness=compact)  # type: ignore[arg-type]
                if kind != "big" and len(cells) < max(1, target // 2):
                    continue
            if not cells:
                continue
            trial = mask.copy()
            for r, c in cells:
                trial[r, c] = False
            if not is_connected(trial, object_connectivity):
                continue
            new_holes = holes_so_far.copy()
            for r, c in cells:
                new_holes[r, c] = True
            if count_holes(trial) != len(carved) + 1:
                continue
            mask = trial
            holes_so_far = new_holes
            placed = cells
            break
        if placed is None:
            if strict:
                return None
            break
        carved.append(placed)
    return mask, carved


def hole_plan(
    rng: np.random.Generator,
    n_holes: int,
    budget: int,
    profile: Optional[str] = None,
) -> tuple[list[tuple[str, int]], str]:
    """Choose hole kinds/sizes for ``n_holes`` given ``budget`` carveable cells.

    Profiles: pinholes, small, mixed, large_dominant, rooms, jagged, blobs.
    Large holes come first so they get space.
    """
    if n_holes <= 0:
        return [], "none"
    if profile is None:
        options = ["pinholes", "small", "mixed", "large_dominant", "rooms", "jagged", "blobs"]
        if n_holes == 1:
            weights = [0.10, 0.18, 0.0, 0.42, 0.10, 0.10, 0.10]
        elif n_holes <= 3:
            weights = [0.12, 0.18, 0.26, 0.14, 0.12, 0.10, 0.08]
        else:
            weights = [0.16, 0.22, 0.28, 0.08, 0.12, 0.08, 0.06]
        profile = str(rng.choice(options, p=np.array(weights) / sum(weights)))
    per_hole = max(1, budget // max(1, n_holes))
    plan: list[tuple[str, int]] = []
    if profile == "pinholes":
        plan = [(("cell", 1), ("bar", 2), ("poly", 3))[int(rng.choice(3, p=[0.45, 0.35, 0.2]))] for _ in range(n_holes)]
    elif profile == "small":
        for _ in range(n_holes):
            kind = str(rng.choice(["cell", "bar", "rect", "poly", "blob"], p=[0.15, 0.2, 0.25, 0.25, 0.15]))
            plan.append((kind, int(rng.integers(2, min(6, max(3, per_hole)) + 1))))
    elif profile == "mixed":
        big_pct = int(rng.integers(20, 55))
        plan.append(("big", big_pct))
        for _ in range(n_holes - 1):
            kind = str(rng.choice(["cell", "bar", "rect", "poly", "blob"], p=[0.2, 0.2, 0.2, 0.2, 0.2]))
            plan.append((kind, int(rng.integers(2, min(6, max(3, per_hole // 2)) + 1))))
    elif profile == "large_dominant":
        plan.append(("big", int(rng.integers(45, 90))))
        for _ in range(n_holes - 1):
            plan.append((("cell", 1), ("bar", 2), ("poly", 3))[int(rng.choice(3, p=[0.4, 0.35, 0.25]))])
    elif profile == "rooms":
        for _ in range(n_holes):
            plan.append(("rect", int(rng.integers(2, min(16, max(3, per_hole)) + 1))))
    elif profile == "jagged":
        for _ in range(n_holes):
            plan.append(("poly", int(rng.integers(3, min(14, max(4, per_hole)) + 1))))
    else:  # blobs
        for _ in range(n_holes):
            plan.append(("blob", int(rng.integers(4, min(20, max(5, per_hole)) + 1))))
    return plan, profile


# ---------------------------------------------------------------------------
# Outlines and gaps (tasks 36 / 37)
# ---------------------------------------------------------------------------

def outline(mask: np.ndarray, thickness: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(boundary cells, interior cells) of a filled mask.

    The boundary is the mask minus its ``thickness``-fold 4-erosion.  Any
    4-path from the interior to the outside must cross a boundary cell, so
    the boundary always encloses the interior (it may be only 8-connected,
    e.g. along diagonal edges).
    """
    interior = mask.copy()
    for _ in range(thickness):
        interior = erode4(interior)
    return mask & ~interior, interior


def bridge_cells(object_mask: np.ndarray) -> np.ndarray:
    """Object cells 4-adjacent to both the outside and an enclosed region."""
    out = outside_mask(object_mask)
    inner = hole_mask(object_mask)
    return object_mask & dilate4(out) & dilate4(inner)


def straight_gap_candidates(object_mask: np.ndarray, length: int) -> list[list[Cell]]:
    """Straight runs of ``length`` bridge cells along a row or column.

    Every cell of the run is a bridge cell (4-adjacent to the interior and to
    the outside), and the cells immediately before and after the run along the
    same line are object cells, so the missing segment reads as a break in a
    straight wall.  For length 1 the run is any bridge cell whose two
    neighbours along at least one axis are both object cells.
    """
    h, w = object_mask.shape
    bridges = bridge_cells(object_mask)
    result: list[list[Cell]] = []
    for r, c in zip(*np.nonzero(bridges)):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, 0)):
            run = [(r + i * dr, c + i * dc) for i in range(length)]
            before = (r - dr, c - dc)
            after = (r + length * dr, c + length * dc)
            if not (0 <= before[0] < h and 0 <= before[1] < w and 0 <= after[0] < h and 0 <= after[1] < w):
                continue
            if not (object_mask[before] and object_mask[after]):
                continue
            if all(0 <= rr < h and 0 <= cc < w and bridges[rr, cc] for rr, cc in run):
                result.append(run)
    return result


def gap_is_unique_repair(object_mask: np.ndarray, gap: Sequence[Cell]) -> bool:
    """Verify the task-37 uniqueness conditions for removing ``gap``.

    With the wall cells of ``gap`` removed, the answer "fill exactly the gap"
    is the unique fewest-cells repair that restores exactly the original
    enclosed region provided that (a) every gap cell is 4-adjacent to an
    enclosed cell of the closed object and (b) removing the gap actually
    connects an enclosed cell to the outside.  (Any repair set S must avoid
    the interior; a gap cell g left out of S is 4-adjacent to an interior cell
    and would therefore become enclosed itself, changing the enclosed region;
    hence S ⊇ gap, and |S| = |gap| forces S = gap.)
    """
    inner = hole_mask(object_mask)
    if not inner.any():
        return False
    h, w = object_mask.shape
    for r, c in gap:
        if not object_mask[r, c]:
            return False
        if not any(0 <= r + dr < h and 0 <= c + dc < w and inner[r + dr, c + dc] for dr, dc in DELTAS_4):
            return False
    opened = object_mask.copy()
    for r, c in gap:
        opened[r, c] = False
    reached = outside_mask(opened)
    return bool((reached & inner).any())


# ---------------------------------------------------------------------------
# Background rendering (never touches object cells)
# ---------------------------------------------------------------------------

BACKGROUND_MODES = ("plain", "outside_texture", "all_texture", "holes_texture", "noise", "blocks")


def _texture(
    rng: np.random.Generator,
    display: np.ndarray,
    region: np.ndarray,
    accents: Sequence[int],
    style: Optional[str] = None,
) -> None:
    if not region.any() or not accents:
        return
    h, w = display.shape
    rr, cc = np.mgrid[0:h, 0:w]
    if style is None:
        style = str(rng.choice(["stripes", "checker", "speckle", "diag", "bands"], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
    if style == "stripes":
        period = int(rng.integers(2, 5))
        axis = rr if rng.random() < 0.5 else cc
        offset = int(rng.integers(0, period))
        sel = ((axis + offset) % period == 0) & region
        display[sel] = accents[0]
        if len(accents) > 1 and period >= 3 and rng.random() < 0.5:
            sel2 = ((axis + offset + 1) % period == 0) & region
            display[sel2] = accents[1]
    elif style == "checker":
        size = int(rng.integers(1, 3))
        parity = ((rr // size) + (cc // size)) % 2 == 0
        display[parity & region] = accents[0]
        if len(accents) > 1 and rng.random() < 0.35:
            display[(~parity) & region] = accents[1]
    elif style == "diag":
        period = int(rng.integers(2, 5))
        sign = 1 if rng.random() < 0.5 else -1
        sel = ((rr + sign * cc) % period == 0) & region
        display[sel] = accents[0]
    elif style == "bands":
        band = int(rng.integers(2, 5))
        axis = rr if rng.random() < 0.5 else cc
        colors = list(accents)
        idx = (axis // band) % (len(colors) + 1)
        for i, col in enumerate(colors, start=1):
            display[(idx == i) & region] = col
    else:  # speckle
        density = float(rng.uniform(0.06, 0.3))
        pos = np.argwhere(region)
        n = int(round(len(pos) * density))
        if n > 0:
            chosen = rng.choice(len(pos), size=n, replace=False)
            for i in chosen.tolist():
                r, c = pos[i]
                display[r, c] = accents[int(rng.integers(0, len(accents)))]


def render_background(
    rng: np.random.Generator,
    object_mask: np.ndarray,
    obj_color: int,
    mode: Optional[str] = None,
    mode_weights: Optional[Sequence[float]] = None,
    hole_texture_min_size: int = 3,
    max_accents: int = 3,
    n_colors: int = 10,
) -> tuple[np.ndarray, str]:
    """Colour every non-object cell; object cells get ``obj_color`` exactly.

    Modes: plain (one colour), outside_texture (texture outside, holes plain),
    all_texture (same texture everywhere), holes_texture (outside plain, big
    holes textured), noise (sparse multicolour speckle), blocks (a few solid
    patches).  Textured hole cells are limited to holes with at least
    ``hole_texture_min_size`` cells so pinholes always read as empty.
    """
    h, w = object_mask.shape
    others = [c for c in range(n_colors) if c != obj_color]
    base = int(rng.choice(others))
    accent_pool = [c for c in others if c != base]
    display = np.full((h, w), base, dtype=np.uint8)
    display[object_mask] = obj_color
    out = outside_mask(object_mask)
    hmask = ~object_mask & ~out
    big_holes = np.zeros_like(hmask)
    if hmask.any():
        labels, n = label_components(hmask, 4)
        sizes = np.bincount(labels.ravel())
        for k in range(1, n + 1):
            if sizes[k] >= hole_texture_min_size:
                big_holes |= labels == k
    if mode is None:
        weights = np.array(mode_weights if mode_weights is not None else [0.36, 0.26, 0.12, 0.08, 0.12, 0.06], dtype=float)
        mode = str(rng.choice(BACKGROUND_MODES, p=weights / weights.sum()))
    if mode == "holes_texture" and not big_holes.any():
        mode = "outside_texture" if out.any() else "plain"
    if mode == "plain" or not accent_pool:
        display[object_mask] = obj_color
        return display, "plain"
    n_acc = int(rng.integers(1, min(max_accents, len(accent_pool)) + 1))
    accents = [int(c) for c in rng.choice(accent_pool, size=n_acc, replace=False)]
    if mode == "outside_texture":
        _texture(rng, display, out, accents)
    elif mode == "all_texture":
        _texture(rng, display, out | big_holes, accents)
    elif mode == "holes_texture":
        _texture(rng, display, big_holes, accents)
    elif mode == "noise":
        _texture(rng, display, out | big_holes if rng.random() < 0.4 else out, accents, style="speckle")
    else:  # blocks
        n_blocks = int(rng.integers(2, 7))
        for _ in range(n_blocks):
            bh = int(rng.integers(1, max(2, h // 3) + 1))
            bw = int(rng.integers(1, max(2, w // 3) + 1))
            r0 = int(rng.integers(0, max(1, h - bh + 1)))
            c0 = int(rng.integers(0, max(1, w - bw + 1)))
            col = accents[int(rng.integers(0, len(accents)))]
            patch = np.zeros((h, w), dtype=bool)
            patch[r0:r0 + bh, c0:c0 + bw] = True
            display[patch & out] = col
    display[object_mask] = obj_color
    return display, mode


def place_on_grid(
    rng: np.random.Generator,
    mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    margin: int = 1,
) -> Optional[np.ndarray]:
    """Place a trimmed mask at a random position inside a grid_h×grid_w canvas."""
    body = trim(mask)
    bh, bw = body.shape
    if bh + 2 * margin > grid_h or bw + 2 * margin > grid_w:
        return None
    r0 = int(rng.integers(margin, grid_h - bh - margin + 1))
    c0 = int(rng.integers(margin, grid_w - bw - margin + 1))
    canvas = np.zeros((grid_h, grid_w), dtype=bool)
    canvas[r0:r0 + bh, c0:c0 + bw] = body
    return canvas


def sample_dim(rng: np.random.Generator, lo: int, hi: int, bias: float = 1.0) -> int:
    """Sample one grid side in [lo, hi] with a small-size bias.

    ``bias`` 1.0 = min of two uniforms (ARC-like: median ≈ lo + (hi-lo)/3);
    0 = uniform; >1 pulls harder toward ``lo``.
    """
    if hi <= lo:
        return lo
    if bias <= 0:
        return int(rng.integers(lo, hi + 1))
    u = rng.random() ** (1.0 / (1.0 + bias))  # bias 1 -> sqrt -> min-of-two-uniforms law
    return lo + int(round((1.0 - u) * (hi - lo)))
