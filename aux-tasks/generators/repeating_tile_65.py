"""Task 65: infer a primitive rectangular tile from a repeated, cropped drawing.

The canvas is selected first.  A tile plan then fixes its family, dimensions,
geometry and usable palette before any retries.  Motifs keep a visible margin;
their phase is free, so a tile need not start at the motif's bounding box.
"""
from __future__ import annotations

import numpy as np


FAMILIES = ("bands", "checks", "patchwork", "motif", "paired_motifs", "nested_frames")
FAMILY_WEIGHTS = (.10, .10, .12, .36, .16, .16)
MOTIFS = ("rectangle", "L", "T", "U", "plus", "diamond", "frame", "staircase")
MAX_ATTEMPTS = 48

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5


def smallest_tile(grid) -> np.ndarray:
    """Read the least periods of the complete visible grid, including edge cuts.

    A period is a translation equality on every overlapping cell.  Horizontal
    and vertical equalities are independent, so their least positive periods
    give the unique smallest rectangular tile.  No divisibility is assumed.
    """
    a = np.asarray(grid)
    if a.ndim != 2 or not a.size:
        raise ValueError("Expected a nonempty rectangular grid")
    h, w = a.shape
    ph = next(p for p in range(1, h + 1) if np.array_equal(a[p:], a[:-p]))
    pw = next(p for p in range(1, w + 1) if np.array_equal(a[:, p:], a[:, :-p]))
    return a[:ph, :pw].copy()


def validate_example(example: dict) -> None:
    """Check the answer from visible cells, never from a planted tile or period."""
    a = np.asarray(example["input"])
    if a.ndim != 2 or not a.size or max(a.shape) > 30:
        raise ValueError("The input must be a rectangular grid with sides 1-30")
    if len(np.unique(a)) < 2:
        raise ValueError("A constant grid does not provide a nontrivial tile")
    answer = smallest_tile(a)
    if any(side != 1 and side < 2 * period for side, period in zip(a.shape, answer.shape)):
        raise ValueError("The grid must show at least two full copies along each nontrivial axis")
    if not np.array_equal(answer, np.asarray(example["output"])):
        raise ValueError("Output must be the top-left copy of the smallest cropped-period tile")
    m = example.get("metadata", {})
    measured = {
        "grid_h": a.shape[0], "grid_w": a.shape[1],
        "tile_height": answer.shape[0], "tile_width": answer.shape[1],
        "tile_area": answer.size,
        "n_colors": len(np.unique(a)),
        "repeats_h": a.shape[0] // answer.shape[0],
        "repeats_w": a.shape[1] // answer.shape[1],
        "partial_rows": a.shape[0] % answer.shape[0],
        "partial_cols": a.shape[1] % answer.shape[1],
        "has_partial_copies": bool(a.shape[0] % answer.shape[0] or a.shape[1] % answer.shape[1]),
    }
    if any(key in m and m[key] != value for key, value in measured.items()):
        raise ValueError("Tile metadata disagrees with the visible pattern")
    if "palette" in m and set(m["palette"]) != set(np.unique(a)):
        raise ValueError("The planned palette disagrees with the visible colors")


def _partition(rng, length: int, count: int) -> np.ndarray:
    """Positive contiguous runs, with their actual lengths summing to length."""
    cuts = sorted(rng.choice(np.arange(1, length), size=count - 1, replace=False))
    return np.diff([0, *cuts, length])


def _palette(rng, capacity: int) -> list[int]:
    n = int(rng.integers(2, min(10, capacity) + 1))
    return list(map(int, rng.choice(10, size=n, replace=False)))


def _tile_sides(rng, cap: tuple[int, int], floor: int) -> tuple[int, int]:
    # Motifs should occupy useful space on large canvases while retaining smaller
    # tiles too.  Every feasible side remains available; no shared square draw.
    return tuple(int(rng.integers(floor, side + 1)) for side in cap)


def _shape_mask(h: int, w: int, kind: str, rng) -> np.ndarray:
    yy, xx = np.indices((h, w))
    thick = max(1, min(h, w) // 3)
    if kind == "rectangle":
        mask = np.ones((h, w), dtype=bool)
    elif kind == "L":
        mask = (xx < thick) | (yy >= h - thick)
    elif kind == "T":
        center = (w - thick) // 2
        mask = (yy < thick) | ((xx >= center) & (xx < center + thick))
    elif kind == "U":
        mask = (xx < thick) | (xx >= w - thick) | (yy >= h - thick)
    elif kind == "plus":
        mid_r, mid_c = (h - thick) // 2, (w - thick) // 2
        mask = ((yy >= mid_r) & (yy < mid_r + thick)) | ((xx >= mid_c) & (xx < mid_c + thick))
    elif kind == "diamond":
        mask = np.abs(2 * yy - h + 1) / h + np.abs(2 * xx - w + 1) / w <= 1
    elif kind == "frame":
        mask = (xx < thick) | (xx >= w - thick) | (yy < thick) | (yy >= h - thick)
    elif kind == "staircase":
        steps = min(4, h, w)
        mask = (xx * steps // w) <= (yy * steps // h)
    else:
        raise ValueError(kind)
    # Reflection changes orientation without forcing a square bounding box.
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _motif_mask(rng, shape: tuple[int, int], paired: bool) -> tuple[np.ndarray, list[str]]:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if paired:
        axis = int(rng.choice([axis for axis, side in enumerate(shape) if side >= 7]))
        length = shape[axis] - 3  # exterior margins and the gap between motifs
        split = int(rng.integers(2, length - 1))
        boxes = ((1, 1, h - 2, split), (1, split + 2, h - 2, length - split)) if axis == 1 else (
            (1, 1, split, w - 2), (split + 2, 1, length - split, w - 2)
        )
    else:
        # Keep breathing room while preventing a tiny stamp on a large tile.
        mh = int(rng.integers(max(3 if h >= 5 else 2, (h - 1) // 2), h - 1))
        mw = int(rng.integers(max(3 if w >= 5 else 2, (w - 1) // 2), w - 1))
        r = int(rng.integers(1, h - mh))
        c = int(rng.integers(1, w - mw))
        boxes = ((r, c, mh, mw),)
    kinds = []
    for r, c, bh, bw in boxes:
        kinds_here = ("rectangle",) if min(bh, bw) < 3 else MOTIFS
        kind = str(rng.choice(kinds_here))
        body = _shape_mask(bh, bw, kind, rng)
        mask[r:r + bh, c:c + bw] |= body
        kinds.append(kind)
    return mask, kinds


def _plan(rng) -> dict:
    # Every pair in MIN_SIDE..30 is feasible, including both strip orientations;
    # the old max(h, w) >= 4 guard (1x1 through 3x3 show no tile twice) is subsumed.
    h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
    cap = (max(1, h // 2), max(1, w // 2))
    eligible = [
        True, min(cap) >= 2, min(cap) >= 2, min(cap) >= 4,
        min(cap) >= 4 and max(cap) >= 7, min(cap) >= 5,
    ]
    weights = np.asarray(FAMILY_WEIGHTS) * eligible
    family = str(rng.choice(FAMILIES, p=weights / weights.sum()))
    plan = {"grid_shape": (h, w), "family": family}
    if family == "bands":
        axis = int(rng.choice([i for i, side in enumerate(cap) if side >= 2]))
        palette = _palette(rng, cap[axis])
        period = int(rng.integers(len(palette), cap[axis] + 1))
        shape = (period, 1) if axis == 0 else (1, period)
        plan["band_axis"] = axis
    elif family == "checks":
        palette = _palette(rng, min(cap))
        n = len(palette)
        shape = tuple(n * int(rng.integers(1, side // n + 1)) for side in cap)
    elif family == "patchwork":
        shape = _tile_sides(rng, cap, 2)
        # Color is assigned to rectangular panels, not individual random cells.
        choices = [(r, c) for r in range(2, min(4, shape[0]) + 1)
                   for c in range(2, min(4, shape[1]) + 1)]
        panel_shape = choices[int(rng.integers(len(choices)))]
        palette = _palette(rng, panel_shape[0] * panel_shape[1])
        plan["panel_shape"] = panel_shape
    elif family in ("motif", "paired_motifs"):
        # A nonrectangular motif needs a 3x3 body plus its surrounding margin.
        # Use that capacity when it exists instead of accidentally selecting a
        # two-cell side and turning most familiar shape choices into rectangles.
        floor = 5 if family == "motif" and min(cap) >= 5 else 4
        shape = _tile_sides(rng, cap, floor)
        if family == "paired_motifs" and max(shape) < 7:
            axis = int(rng.choice([i for i, side in enumerate(cap) if side >= 7]))
            shape = tuple(int(rng.integers(7, cap[i] + 1)) if i == axis else side
                          for i, side in enumerate(shape))
        mask, kinds = _motif_mask(rng, shape, family == "paired_motifs")
        coordinates = [np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))]
        axis = int(np.argmax([len(values) for values in coordinates]))
        palette = _palette(rng, 1 + len(coordinates[axis]))
        plan.update(mask=mask, motif_kinds=kinds, fill_axis=axis)
    elif family == "nested_frames":
        shape = _tile_sides(rng, cap, 5)
        palette = _palette(rng, (min(shape) + 1) // 2)
    else:
        raise ValueError(family)
    plan.update(tile_shape=shape, palette=palette)
    # A phase is a translation of the complete periodic drawing, never a rotation
    # or reflection during reconstruction.  The required tile still starts at 0,0.
    plan["phase"] = tuple(int(rng.integers(side)) for side in shape)
    return plan


def _draw(rng, plan: dict) -> np.ndarray:
    family, shape, palette = plan["family"], plan["tile_shape"], plan["palette"]
    h, w = shape
    n = len(palette)
    if family == "bands":
        runs = _partition(rng, max(shape), n)
        values = np.repeat(palette, runs)
        tile = values[:, None] if h != 1 else values[None, :]
    elif family == "checks":
        yy, xx = np.indices(shape)
        tile = np.asarray(palette)[(yy // (h // n) + xx // (w // n)) % n]
    elif family == "patchwork":
        nr, nc = plan["panel_shape"]
        # One uniquely colored panel is an ordinary part of the patchwork.  It
        # also certifies that a planned block cannot silently reduce its period.
        values = [palette[i % (n - 1)] for i in range(nr * nc - 1)] + [palette[-1]]
        rng.shuffle(values)
        tile = np.repeat(np.repeat(np.asarray(values).reshape(nr, nc),
                                   _partition(rng, h, nr), axis=0),
                         _partition(rng, w, nc), axis=1)
    elif family in ("motif", "paired_motifs"):
        mask, axis = plan["mask"], plan["fill_axis"]
        coordinates = np.flatnonzero(mask.any(axis=1 - axis))
        runs = _partition(rng, len(coordinates), n - 1)
        color_at = dict(zip(coordinates, np.repeat(palette[1:], runs)))
        tile = np.full(shape, palette[0])
        coordinate_grid = np.indices(shape)[axis]
        for coordinate, color in color_at.items():
            tile[mask & (coordinate_grid == coordinate)] = color
    elif family == "nested_frames":
        yy, xx = np.indices(shape)
        depth = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx])
        runs = _partition(rng, int(depth.max()) + 1, n)
        tile = np.asarray(np.repeat(palette, runs))[depth]
    else:
        raise ValueError(family)
    return np.roll(tile, plan["phase"], axis=(0, 1))


def generate_example(rng: np.random.Generator) -> dict:
    plan = _plan(rng)
    h, w = plan["grid_shape"]
    th, tw = plan["tile_shape"]
    for attempts in range(1, MAX_ATTEMPTS + 1):
        tile = _draw(rng, plan)
        grid = tile[np.arange(h)[:, None] % th, np.arange(w)[None, :] % tw]
        if smallest_tile(grid).shape != (th, tw):
            continue
        if set(map(int, np.unique(grid))) != set(plan["palette"]):
            raise ValueError("A planned color disappeared from the tile")
        example = {
            "input": grid.tolist(), "output": tile.tolist(),
            "metadata": {
                "grid_h": h, "grid_w": w, "tile_height": th, "tile_width": tw,
                "tile_area": th * tw, "n_colors": len(plan["palette"]),
                "palette": plan["palette"], "family": plan["family"],
                "motif_kinds": plan.get("motif_kinds", []),
                "phase": list(plan["phase"]),
                "repeats_h": h // th, "repeats_w": w // tw,
                "partial_rows": h % th, "partial_cols": w % tw,
                "has_partial_copies": bool(h % th or w % tw),
                "first_row_column_sufficient": bool(
                    smallest_tile(grid[:1]).shape[1] == tw
                    and smallest_tile(grid[:, :1]).shape[0] == th
                ),
                "attempts": attempts, "minimal_tile_verified": True,
            },
        }
        validate_example(example)
        return example
    raise ValueError(f"Could not realize fixed tile plan after {MAX_ATTEMPTS} attempts: "
                     f"{plan['family']}, {th}x{tw}, {len(plan['palette'])} colors")
