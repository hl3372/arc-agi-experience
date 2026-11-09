"""Task 27: pack spatially connected, possibly multicolour, clean shapes.

The scene is drawn before the requested edge. Every scene fits on BOTH axes;
only the intentional identity branch depends on the requested destination.
This module has no registry side effects. ``misc`` owns task registration.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from common.backgrounds import MIN_SHORT_SIDE, sparse_mask


VARIANTS = (("bottom", "left"), ("bottom", "right"),
            ("top", "left"), ("top", "right"),
            ("left", "top"), ("left", "bottom"),
            ("right", "top"), ("right", "bottom"))
MAX_GEOMETRY_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 100


def prompt27(metadata: dict) -> str:
    edge = str(metadata["pack_edge"])
    start = str(metadata["start_position"])
    if (edge, start) not in VARIANTS:
        raise ValueError("invalid packing edge and starting side")
    gap = "column" if edge in ("top", "bottom") else "row"
    return (
        "Ignoring the background, each 4-connected group of at least two non-background cells is one object, "
        "even when it contains several colors. Cells connect horizontally or vertically; cells that touch "
        "only at a corner are not connected. "
        "Isolated single cells, whatever their color, are background texture; leave them unchanged. "
        f"Move every object by translation only, without rotating or reflecting it, against the {edge} edge, "
        f"starting at the {start} side, in descending order of its number of cells. All object cell counts are "
        "different. An object's bounding box is the smallest rectangle containing all its cells. "
        f"Align every bounding box flush with the {edge} edge and the first box flush with the {start} side, "
        f"with exactly one {gap} between consecutive boxes. Preserve each object's shape and colors. "
        "Fill vacated cells with the surrounding background color. Leave every other cell unchanged."
    )


# ---------------------------------------------------------------------------
# Independent final-visible-grid reconstruction. No construction records,
# shape builders, palette assignments or destination helpers are used here.
# ---------------------------------------------------------------------------

def inspect_visible(input_grid) -> dict[str, Any]:
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not grid.size:
        raise ValueError("expected a nonempty rectangular grid")
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(values) < 2 or counts[order[0]] * 2 <= grid.size:
        raise ValueError("the ground is not uniquely identifiable")
    ground = values[order[0]]
    if counts[order[0]] < 1.25 * counts[order[1]]:
        raise ValueError("insufficient ground margin")
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    detail = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == ground:
                continue
            cells = [(r, c)]
            visited[r, c] = True
            for rr, cc in cells:
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] != ground:
                        visited[nr, nc] = True
                        cells.append((nr, nc))
            if len(cells) == 1:
                detail.extend(cells)
                continue
            r0 = min(p[0] for p in cells)
            c0 = min(p[1] for p in cells)
            bh = max(p[0] for p in cells) - r0 + 1
            bw = max(p[1] for p in cells) - c0 + 1
            objects.append({"cells": cells, "area": len(cells), "r0": r0, "c0": c0, "h": bh, "w": bw})
    if not objects:
        raise ValueError("no objects")
    areas = [o["area"] for o in objects]
    if len(areas) != len(set(areas)):
        raise ValueError("object cell counts are not distinct")
    return {"ground": ground, "objects": objects, "detail": detail,
            "ground_margin": float(counts[order[0]] / counts[order[1]])}


def solve_pack_shapes(input_grid, edge: str, start: str) -> np.ndarray:
    """Solve solely from the visible grid and the two prompted parameters."""
    if (edge, start) not in VARIANTS:
        raise ValueError("invalid packing edge and starting side")
    grid = np.asarray(input_grid)
    scene = inspect_visible(grid)
    ordered = sorted(scene["objects"], key=lambda o: -o["area"])
    h, w = grid.shape
    horizontal = edge in ("top", "bottom")
    span = w if horizontal else h
    widths = [o["w"] if horizontal else o["h"] for o in ordered]
    if sum(widths) + len(ordered) - 1 > span:
        raise ValueError("objects do not fit along the requested edge")
    output = grid.copy()
    for obj in ordered:
        for r, c in obj["cells"]:
            output[r, c] = scene["ground"]
    cursor = 0 if start in ("left", "top") else span
    fixed_detail = set(scene["detail"])
    for obj, length in zip(ordered, widths):
        if start in ("left", "top"):
            offset = cursor
            cursor += length + 1
        else:
            offset = cursor - length
            cursor -= length + 1
        if horizontal:
            nr, nc = (0 if edge == "top" else h - obj["h"]), offset
        else:
            nr, nc = offset, (0 if edge == "left" else w - obj["w"])
        for r, c in obj["cells"]:
            rr, cc = nr + r - obj["r0"], nc + c - obj["c0"]
            if not (0 <= rr < h and 0 <= cc < w):
                raise ValueError("packed object extends beyond the grid")
            if (rr, cc) in fixed_detail:
                raise ValueError("packing would overwrite fixed background detail")
            output[rr, cc] = grid[r, c]
    return output


def solve_prompt27(input_grid, prompt: str) -> np.ndarray:
    match = re.search(r"against the (top|bottom|left|right) edge, starting at the (top|bottom|left|right) side", prompt)
    if match is None:
        raise ValueError("missing packing parameters")
    return solve_pack_shapes(input_grid, match[1], match[2])


def validate_example(example: dict) -> None:
    """Raise on a visible-scene invariant or independently recomputed mismatch."""
    grid = np.asarray(example["input"])
    scene = inspect_visible(grid)
    objects = scene["objects"]
    h, w = grid.shape
    if len(objects) < 2:
        raise ValueError("task 27 requires at least two objects")
    if sum(o["h"] for o in objects) + len(objects) - 1 > h or sum(o["w"] for o in objects) + len(objects) - 1 > w:
        raise ValueError("scene is not feasible on both packing axes")
    for r, c in scene["detail"]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr, c + dc] != scene["ground"]:
                    raise ValueError("background dots are not isolated from all non-ground cells")
    expected = solve_prompt27(grid, prompt27(example["metadata"]))
    if not np.array_equal(expected, np.asarray(example["output"])):
        raise ValueError("planned packing differs from final-visible-grid reconstruction")


# ---------------------------------------------------------------------------
# Construction: clean shape vocabulary, axis-independent budgets and placement.
# ---------------------------------------------------------------------------

def _mask(kind: str, h: int, w: int, thickness: int = 1) -> np.ndarray:
    rr, cc = np.indices((h, w))
    t = min(thickness, max(1, min(h, w) // 2))
    if kind == "rectangle":
        result = np.ones((h, w), dtype=bool)
    elif kind == "frame":
        result = (rr < t) | (rr >= h - t) | (cc < t) | (cc >= w - t)
    elif kind == "L":
        result = (cc < t) | (rr >= h - t)
    elif kind == "U":
        result = (cc < t) | (cc >= w - t) | (rr >= h - t)
    elif kind == "T":
        mid = (w - t) // 2
        result = (rr < t) | ((cc >= mid) & (cc < mid + t))
    elif kind == "H":
        mid = (h - t) // 2
        result = (cc < t) | (cc >= w - t) | ((rr >= mid) & (rr < mid + t))
    elif kind == "staircase":
        steps = min(4, h, w)
        levels = np.minimum(steps, rr * steps // h + 1)
        result = cc < (levels * w + steps - 1) // steps
    elif kind == "chamfer":
        radius = max(1, min(h, w) // 4)
        result = (np.minimum(rr, h - 1 - rr) + np.minimum(cc, w - 1 - cc)) >= radius
    else:
        raise ValueError(kind)
    return result


def _perimeter(mask: np.ndarray) -> int:
    return int(4 * mask.sum() - 2 * np.sum(mask[1:] & mask[:-1]) - 2 * np.sum(mask[:, 1:] & mask[:, :-1]))


def _orient(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()




def _random_kind(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, str]:
    allowed = ["rectangle"]
    if min(h, w) >= 2:
        allowed += ["L", "staircase"]
    if min(h, w) >= 3:
        allowed += ["frame", "U", "T", "H"]
    if min(h, w) >= 5:
        allowed += ["chamfer"]
    kind = allowed[int(rng.integers(len(allowed)))]
    t = int(rng.integers(1, max(1, min(h, w) // 3) + 1))
    return _orient(rng, _mask(kind, h, w, t)), ("bar" if min(h, w) == 1 else kind)


def _geometry(rng: np.random.Generator, h: int, w: int, count: int, palette_size: int) -> tuple[list[np.ndarray], list[str], dict]:
    # Each object receives its own share of BOTH packing-axis budgets. There
    # are no privileged "main" objects and no fixed 2/3/4/5-cell extras.
    bh, bw = h - count + 1, w - count + 1
    cap = int(.40 * h * w)
    can_reverse = min(h, w) >= max(12, 4 * count)
    want_reversal = bool(can_reverse and rng.random() < .28)

    def allocate(total, weights):
        values = np.ones(count, dtype=int)
        fractions = (total - count) * weights / weights.sum()
        values += np.floor(fractions).astype(int)
        order = np.argsort(-(fractions - np.floor(fractions)))
        values[order[:total - int(values.sum())]] += 1
        return values

    for attempt in range(1, MAX_GEOMETRY_ATTEMPTS + 1):
        total_h = int(rng.integers(max(count, int(.84 * bh)), bh + 1))
        total_w = int(rng.integers(max(count, int(.84 * bw)), bw + 1))
        weights = rng.uniform(.45, 1.65, count)
        heights = allocate(total_h, weights * rng.uniform(.65, 1.35, count))
        widths = allocate(total_w, weights * rng.uniform(.65, 1.35, count))
        masks, kinds = [], []
        for sh, sw in zip(heights, widths):
            mask, kind = _random_kind(rng, int(sh), int(sw))
            masks.append(mask)
            kinds.append(kind)
        areas = [int(m.sum()) for m in masks]
        if min(areas) < 2 or len(set(areas)) != count:
            continue
        if any(max(m.shape) > 3 * min(m.shape) for m, k in zip(masks, kinds) if k not in ("bar", "rectangle")):
            continue
        boxes = [m.size for m in masks]
        reversal = any((areas[i] - areas[j]) * (boxes[i] - boxes[j]) < 0
                       for i in range(count) for j in range(i))
        if want_reversal and not reversal:
            continue
        if sum(areas) <= cap and sum(min(3, a) for a in areas) >= palette_size - 1:
            return masks, kinds, {"geometry_attempts": attempt,
                                  "reversal_planned": want_reversal, "reversal_feasible": can_reverse}
    raise RuntimeError(f"task27 geometry failed without changing size/count/palette: {(h,w,count,palette_size,bh,bw)}")


def _packing_positions(masks: list[np.ndarray], h: int, w: int, edge: str, start: str) -> list[tuple[int, int]]:
    positions = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    horizontal = edge in ("top", "bottom")
    forward = start in ("left", "top")
    cursor = 0 if forward else (w if horizontal else h)
    for i in order:
        mh, mw = masks[i].shape
        length = mw if horizontal else mh
        along = cursor if forward else cursor - length
        cursor += (length + 1) * (1 if forward else -1)
        positions[i] = ((0 if edge == "top" else h - mh), along) if horizontal else (along, (0 if edge == "left" else w - mw))
    return positions


def _halo(mask: np.ndarray, diagonal: bool) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc) or (not diagonal and dr and dc):
                continue
            out[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)] |= mask[max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)]
    return out


def _source_positions(rng: np.random.Generator, masks: list[np.ndarray], h: int, w: int) -> tuple[list[tuple[int, int]], int, str]:
    # Constructive diagonal layouts are a deliberate minority. The same method
    # also proves placement exists, but a failed random search raises instead
    # of substituting another scene.
    if rng.random() < .18:
        order = rng.permutation(len(masks))
        positions = [None] * len(masks)
        total_h = sum(m.shape[0] for m in masks)
        total_w = sum(m.shape[1] for m in masks)
        r = int(rng.integers(h - total_h + 1))
        c = int(rng.integers(w - total_w + 1))
        reverse = bool(rng.integers(2))
        for i in order:
            mh, mw = masks[i].shape
            positions[i] = (r, w - c - mw if reverse else c)
            r += mh
            c += mw
        return positions, 1, "diagonal"
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    for attempt in range(1, MAX_PLACEMENT_ATTEMPTS + 1):
        occupied = np.zeros((h, w), dtype=bool)
        positions = [None] * len(masks)
        for i in order:
            m = masks[i]
            mh, mw = m.shape
            forbidden = _halo(occupied, False)
            possibilities = []
            for r in range(h - mh + 1):
                for c in range(w - mw + 1):
                    if not np.any(forbidden[r:r + mh, c:c + mw] & m):
                        possibilities.append((r, c))
            if not possibilities:
                break
            r, c = possibilities[int(rng.integers(len(possibilities)))]
            occupied[r:r + mh, c:c + mw] |= m
            positions[i] = (r, c)
        else:
            return positions, attempt, "scattered"
    raise RuntimeError("task27 could not place its fixed geometry")


def _paint_objects(rng: np.random.Generator, masks: list[np.ndarray], palette: list[int]) -> tuple[list[np.ndarray], list[str]]:
    painted = [None] * len(masks)
    modes = [None] * len(masks)
    order = list(map(int, rng.permutation(len(masks))))
    assignments = [[] for _ in masks]
    for j, i in enumerate(order):
        assignments[i] = [palette[j % len(palette)]]
    # Spread the palette across objects first. Only the remaining colors need
    # bands, capped at three colors per shape instead of rainbowing each one.
    for color in palette[len(masks):]:
        eligible = [i for i in order if len(assignments[i]) < min(3, int(masks[i].sum()))]
        i = min(eligible, key=lambda j: (len(assignments[j]), -int(masks[j].sum())))
        assignments[i].append(color)
    if len(palette) > 1 and rng.random() < .15:
        i = int(rng.choice(order))
        if len(assignments[i]) == 1 and masks[i].sum() >= 6:
            assignments[i].append(int(rng.choice([c for c in palette if c != assignments[i][0]])))
    for i in rng.permutation(len(masks)):
        m = masks[i]
        points = np.argwhere(m)
        by_column = bool(rng.integers(2))
        order = np.lexsort((points[:, 0], points[:, 1])) if by_column else np.lexsort((points[:, 1], points[:, 0]))
        if rng.random() < .5:
            order = order[::-1]
        points = points[order]
        n = len(points)
        colors = assignments[i]
        k = len(colors)
        a = np.full(m.shape, -1, dtype=int)
        for j, (r, c) in enumerate(points):
            a[r, c] = colors[j * k // n]
        painted[i] = a
        modes[i] = "solid" if k == 1 else ("column_bands" if by_column else "row_bands")
    return painted, modes


def _stamp(background: np.ndarray, painted: list[np.ndarray], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = background.copy()
    for piece, (r, c) in zip(painted, positions):
        region = grid[r:r + piece.shape[0], c:c + piece.shape[1]]
        region[piece >= 0] = piece[piece >= 0]
    return grid


def generate_pack_shapes(rng: np.random.Generator) -> dict:
    # Two >=2-cell objects with distinct areas and one-cell gaps require both
    # sides >=3. Six tiny size pairs cannot fit on both axes and are excluded.
    while True:
        h, w = (int(x) for x in rng.integers(3, 31, size=2))
        max_cells = max(2 + (h - 2) * (w - 3), 2 + (h - 3) * (w - 2))
        if max_cells >= 5 and not (h == 4 and w == 4):
            break
    max_count = min(6, max(2, min(h, w) // 4 + 1))
    counts = np.arange(2, max_count + 1)
    weights = np.array([.12, .27, .29, .21, .11])[:len(counts)]
    count = int(rng.choice(counts, p=weights / weights.sum()))
    color_cap = min(10, 3 * count, max_cells + 1, int(.40 * h * w) + 1)
    palette_size = int(rng.integers(2, color_cap + 1))
    masks, kinds, info = _geometry(rng, h, w, count, palette_size)
    colors = [int(c) for c in rng.choice(10, palette_size, replace=False)]
    ground, palette = colors[0], colors[1:]
    painted, fill_modes = _paint_objects(rng, masks, palette)
    source, placement_attempts, layout = _source_positions(rng, masks, h, w)
    packs = [_packing_positions(masks, h, w, edge, start) for edge, start in VARIANTS]
    occupied = np.zeros((h, w), dtype=bool)
    for positions in [source] + packs:
        for m, (r, c) in zip(masks, positions):
            occupied[r:r + m.shape[0], c:c + m.shape[1]] |= m
    background = np.full((h, w), ground, dtype=int)
    texture_kind = "plain"
    texture_cells = 0
    if min(h, w) >= MIN_SHORT_SIDE["sparse"] and rng.random() < .36:
        kind = str(rng.choice(("lattice", "dotted_stripes", "sparse_checker", "dotted_border", "dotted_diagonal")))
        dots = sparse_mask(rng, h, w, kind) & ~_halo(occupied, True)
        points = np.argwhere(dots)
        limit = max(1, int(.03 * h * w))
        if len(points) > limit:
            # Keep an evenly spaced subsequence of the structured pattern.
            points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
        if len(points):
            texture_kind, texture_cells = kind, len(points)
            color = int(rng.choice(palette))
            background[points[:, 0], points[:, 1]] = color
    # The requested variant is drawn after every ordinary input-scene choice.
    variant = int(rng.integers(len(VARIANTS)))
    edge, start = VARIANTS[variant]
    identity_planned = bool(rng.random() < .08)
    destination = packs[variant]
    if identity_planned:
        source = destination
    grid = _stamp(background, painted, source)
    output = _stamp(background, painted, destination)
    areas = [int(m.sum()) for m in masks]
    bbox_areas = [int(m.size) for m in masks]
    perimeters = [_perimeter(m) for m in masks]
    example = {"input": grid.tolist(), "output": output.tolist(), "metadata": {
        "pack_edge": edge, "start_position": start,
        "order_direction": ("from left to right" if start == "left" else "from right to left") if edge in ("top", "bottom") else ("from top to bottom" if start == "top" else "from bottom to top"),
        "object_count": count, "object_areas": areas, "object_bbox_areas": bbox_areas,
        "object_perimeters": perimeters, "shape_families": kinds, "fill_modes": fill_modes,
        "background_color": ground, "palette_size": palette_size,
        "background_kind": texture_kind, "texture_cells": texture_cells,
        "input_layout": "packed_identity" if identity_planned else layout,
        "sampled_layout": layout, "identity_planned": identity_planned,
        "placement_attempts": placement_attempts, **info,
        "bbox_reversal": any((areas[i] - areas[j]) * (bbox_areas[i] - bbox_areas[j]) < 0 for i in range(count) for j in range(i)),
        "perimeter_reversal": any((areas[i] - areas[j]) * (perimeters[i] - perimeters[j]) < 0 for i in range(count) for j in range(i)),
    }}
    if len(np.unique(grid)) != palette_size:
        raise RuntimeError("task27 palette coverage changed")
    validate_example(example)
    return example
