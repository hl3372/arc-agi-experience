"""Task 26: crop the interior of a uniquely visible, unnamed rectangular frame.

The marker colour is chosen internally, never given in the prompt.  Uniqueness
is checked geometrically for EVERY colour after the frame overwrites the scene.
In particular, a second ring does not become harmless because its colour also
appears elsewhere, or because a same-colour tail is attached outside it.

Input sides are independent uniform draws from 3..30.  Crop sides span 1..28;
full, large, unrestricted and strip crops share the same content families.  The
input size, crop size, palette and scene family are fixed before retries.
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

from common import shapes as S

MAX_ATTEMPTS = 100
SCENE_FAMILIES = ("objects", "patchwork", "bands", "diagonal_bands",
                  "stair_bands", "tiles", "mirror", "striped_panels", "objects_on_bands")
FRAME_DEFINITION = (
    "Extract every cell strictly inside the unique rectangular frame, excluding its border. "
    "The frame is a complete one-cell-thick outline of one color, with horizontal and vertical sides."
)


class Frame(NamedTuple):
    row: int
    col: int
    height: int
    width: int
    color: int


def find_frames(grid) -> list[Frame]:
    """Enumerate thin, monochrome rectangular outlines from visible cells only.

    The immediate interior edge is a different colour, distinguishing a hollow
    one-cell outline from a filled rectangle.  A whole extra parallel outer
    edge makes a wall thick rather than a one-cell frame.  Smaller external
    attachments remain candidates: they must not hide an alternative frame.
    No colour-frequency or designated-background assumption is made.
    """
    a = np.asarray(grid)
    if a.ndim != 2 or min(a.shape) < 3:
        return []
    h, w = a.shape
    result = []
    for color in np.unique(a):
        m = a == color
        # Every candidate has this top-left corner.  The first interior cell
        # must not be frame coloured, even for a one-cell-wide interior.
        corners = m[:-2, :-2] & m[:-2, 1:-1] & m[1:-1, :-2] & ~m[1:-1, 1:-1]
        for r, c in np.argwhere(corners):
            r, c = int(r), int(c)
            right = c + 1
            while right + 1 < w and m[r, right + 1]:
                right += 1
            bottom = r + 1
            while bottom + 1 < h and m[bottom + 1, c]:
                bottom += 1
            for b in range(r + 2, bottom + 1):
                if not m[b, c + 1]:
                    continue
                for d in range(c + 2, right + 1):
                    if not (m[b, d] and m[b, c:d + 1].all() and m[r:b + 1, d].all()):
                        continue
                    if (m[r + 1, c + 1:d].any() or m[b - 1, c + 1:d].any()
                            or m[r + 1:b, c + 1].any() or m[r + 1:b, d - 1].any()):
                        continue
                    thick = ((r > 0 and m[r - 1, c:d + 1].all())
                             or (b + 1 < h and m[b + 1, c:d + 1].all())
                             or (c > 0 and m[r:b + 1, c - 1].all())
                             or (d + 1 < w and m[r:b + 1, d + 1].all()))
                    if not thick:
                        result.append(Frame(r, c, b - r + 1, d - c + 1, int(color)))
    return sorted(result)


def _component_frames(grid) -> list[Frame]:
    """A separate component-based construction check; no geometry solver calls."""
    a = np.asarray(grid)
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(a[r, c])
            cells = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < h and 0 <= yy < w and not seen[xx, yy] and a[xx, yy] == color:
                        seen[xx, yy] = True
                        q.append((xx, yy))
            r0 = min(x for x, _ in cells)
            r1 = max(x for x, _ in cells)
            c0 = min(y for _, y in cells)
            c1 = max(y for _, y in cells)
            fh, fw = r1 - r0 + 1, c1 - c0 + 1
            if (fh >= 3 and fw >= 3 and len(cells) == 2 * (fh + fw) - 4
                    and all(x in (r0, r1) or y in (c0, c1) for x, y in cells)):
                result.append(Frame(r0, c0, fh, fw, color))
    return sorted(result)


def solve_extract_subgrid(grid) -> list[list[int]]:
    """Recover the unnamed frame and copy its interior, without metadata."""
    frames = find_frames(grid)
    if len(frames) != 1:
        raise ValueError(f"Task 26 requires exactly one visible frame; found {len(frames)}")
    r, c, h, w, _ = frames[0]
    return np.asarray(grid)[r + 1:r + h - 1, c + 1:c + w - 1].tolist()


def validate_example(example: dict) -> None:
    """Independent final-visible-grid proof; tampering and ambiguity raise."""
    a = np.asarray(example["input"])
    if a.ndim != 2 or not (3 <= a.shape[0] <= 30 and 3 <= a.shape[1] <= 30):
        raise ValueError("Task 26 input size is outside its feasible range")
    if not np.isin(a, np.arange(10)).all():
        raise ValueError("Task 26 input contains a non-ARC colour")
    expected = solve_extract_subgrid(a)
    if expected != example["output"]:
        raise ValueError("Task 26 output is not the visible frame's interior")
    frame = find_frames(a)[0]
    meta = example.get("metadata", {})
    if "frame_bbox" in meta and list(frame[:4]) != list(meta["frame_bbox"]):
        raise ValueError("Task 26 planned frame differs from the visible frame")
    if "n_colors" in meta and len(np.unique(a)) != meta["n_colors"]:
        raise ValueError("Task 26 palette was lost during rendering")


def _partition(rng, free: np.ndarray, n: int, *, bands=False):
    """Guillotine rectangles, each containing visible content after framing."""
    h, w = free.shape
    boxes = [(0, 0, h, w)]
    preferred_axis = int(rng.integers(2))
    while len(boxes) < n:
        choices = []
        for i, (r, c, bh, bw) in enumerate(boxes):
            count = int(free[r:r + bh, c:c + bw].sum())
            if count < 2:
                continue
            for axis in range(2):
                extent = (bh, bw)[axis]
                for cut in range(1, extent):
                    if axis == 0:
                        left = int(free[r:r + cut, c:c + bw].sum())
                    else:
                        left = int(free[r:r + bh, c:c + cut].sum())
                    if left and left < count:
                        balance = min(left, count - left)
                        weight = count * (0.2 + balance / count)
                        if bands and axis != preferred_axis:
                            weight *= 0.025
                        choices.append((i, axis, cut, weight))
        if not choices:
            raise RuntimeError("Insufficient visible cells for the fixed palette")
        weights = np.array([x[3] for x in choices])
        i, axis, cut, _ = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        r, c, bh, bw = boxes.pop(i)
        if axis == 0:
            boxes.extend(((r, c, cut, bw), (r + cut, c, bh - cut, bw)))
        else:
            boxes.extend(((r, c, bh, cut), (r, c + cut, bh, bw - cut)))
    return boxes


def _level_fill(rng, levels, free, colors):
    values = np.unique(levels[free])
    if len(values) < len(colors):
        return None
    # All colour intervals contain a visible cell, even if the frame removes
    # entire rows/columns of a pattern.
    cuts = sorted(rng.choice(np.arange(1, len(values)), len(colors) - 1, replace=False))
    thresholds = [(values[i - 1] + values[i]) / 2 for i in cuts]
    return np.asarray(colors, dtype=np.uint8)[np.searchsorted(thresholds, levels)]


def _geometric_shape(rng, h, w):
    """Readable geometric content; no random walks, bumps, or notch carving."""
    if min(h, w) < 3:
        return S.rect(rng, h, w), "bar" if min(h, w) == 1 else "rect"
    kind = str(rng.choice(["rect", "L", "T", "U", "H", "plus", "stair", "diamond", "ellipse"]))
    t = int(rng.integers(1, max(2, min(h, w) // 3 + 1)))
    if kind == "rect":
        cells = S.rect(rng, h, w)
    elif kind == "L":
        cells = S.letter_L(rng, h, w, t)
    elif kind == "T":
        cells = S.letter_T(rng, w, max(1, h - t), t)
    elif kind == "U":
        cells = S.letter_U(rng, h, w, t)
    elif kind == "H":
        cells = S.letter_H(rng, h, w, t)
    elif kind == "plus":
        cells = S.plus(rng, max(1, (min(h, w) - t) // 2), t)
    elif kind == "stair":
        step = max(1, min(h, w) // 4)
        cells = S.staircase(rng, max(2, min(h, w) // step), step)
    else:
        yy, xx = np.mgrid[:h, :w]
        yy = abs((yy - (h - 1) / 2) / (h / 2))
        xx = abs((xx - (w - 1) / 2) / (w / 2))
        mask = yy + xx <= 1 if kind == "diamond" else yy * yy + xx * xx <= 1
        cells = frozenset(zip(*np.where(mask)))
    # Library orientations may swap dimensions; transpose back to the slot.
    if max(x for x, _ in cells) >= h or max(y for _, y in cells) >= w:
        cells = frozenset((y, x) for x, y in cells)
    return cells, kind


def _render_scene(rng, free, colors, family):
    h, w = free.shape
    rr, cc = np.mgrid[:h, :w]
    info = {"object_kinds": [], "content_object_cells": 0}
    if len(colors) == 1:
        return np.full((h, w), colors[0], dtype=np.uint8), info
    if family in ("bands", "diagonal_bands", "stair_bands"):
        vertical = bool(rng.integers(2))
        x, y = (cc, rr) if vertical else (rr, cc)
        if family == "bands":
            levels = x
        elif family == "diagonal_bands":
            levels = x + int(rng.choice([-2, -1, 1, 2])) * y
        else:
            step = int(rng.integers(2, max(3, min(h, w) // 2 + 1)))
            levels = x + int(rng.choice([-1, 1])) * (y // step)
        grid = _level_fill(rng, levels, free, colors)
        if grid is not None:
            return grid, info
        # A short axis cannot display the fixed palette in parallel bands.
        # Partitioned bands still use clean horizontal/vertical strips.
    if family == "tiles":
        schemes = []
        for tile_h in range(1, len(colors) + 1):
            tile_w = int(np.ceil(len(colors) / tile_h))
            for sh in range(1, min(4, max(1, h // 3)) + 1):
                for sw in range(1, min(4, max(1, w // 3)) + 1):
                    labels = (((rr // sh) % tile_h) * tile_w + ((cc // sw) % tile_w)) % len(colors)
                    if len(np.unique(labels[free])) == len(colors):
                        schemes.append((tile_h, tile_w, sh, sw))
        if not schemes:
            return None
        tile_h, tile_w, sh, sw = schemes[int(rng.integers(len(schemes)))]
        labels = (((rr // sh) % tile_h) * tile_w + ((cc // sw) % tile_w)) % len(colors)
        info["tile_size"] = [sh, sw]
        info["motif_size"] = [tile_h, tile_w]
        return np.asarray(colors, dtype=np.uint8)[labels], info
    if family == "mirror":
        y = np.minimum(rr, h - 1 - rr) if rng.random() < .7 else rr
        x = np.minimum(cc, w - 1 - cc) if rng.random() < .7 else cc
        coords = np.unique(np.stack([y[free], x[free]], axis=1), axis=0)
        if len(coords) < len(colors):
            return None
        seeds = coords[rng.choice(len(coords), len(colors), replace=False)]
        distance = np.stack([abs(y - a) + abs(x - b) for a, b in seeds])
        grid = np.asarray(colors, dtype=np.uint8)[distance.argmin(axis=0)]
        return grid, info

    n = len(colors)
    if family == "patchwork":
        n += int(rng.integers(0, min(7, max(1, int(free.sum()) // 8))))
    boxes = _partition(rng, free, min(n, int(free.sum())), bands=family == "bands")
    rng.shuffle(boxes)
    grid = np.full((h, w), colors[0], dtype=np.uint8)
    occupied = np.zeros((h, w), dtype=bool)
    if family == "objects_on_bands":
        levels = rr if rng.integers(2) else cc
        grid[(levels // max(2, min(h, w) // 4)) % 2 == 0] = colors[1]
    for i, (r, c, bh, bw) in enumerate(boxes):
        color = colors[i % len(colors)]
        if family in ("objects", "objects_on_bands"):
            if i == 0:
                continue
            placed = False
            for _ in range(16):
                oh = int(rng.integers(max(1, int(bh * .55)), bh + 1))
                ow = int(rng.integers(max(1, int(bw * .55)), bw + 1))
                cells, kind = _geometric_shape(rng, oh, ow)
                sh = max(x for x, _ in cells) + 1
                sw = max(y for _, y in cells) + 1
                if sh > bh or sw > bw:
                    continue
                top = r + int(rng.integers(0, bh - sh + 1))
                left = c + int(rng.integers(0, bw - sw + 1))
                pts = [(top + x, left + y) for x, y in cells]
                if not any(free[p] for p in pts):
                    continue
                for p in pts:
                    grid[p] = color
                    occupied[p] = True
                info["object_kinds"].append(kind)
                placed = True
                break
            if not placed:
                return None
        else:
            grid[r:r + bh, c:c + bw] = color
            if family == "striped_panels" and i > 0:
                period = int(rng.integers(3, 6))
                patch = grid[r:r + bh, c:c + bw]
                axis = np.indices((bh, bw))[int(rng.integers(2))]
                visible = free[r:r + bh, c:c + bw]
                phases = [k for k in range(period) if (visible & (axis % period != k)).any()]
                phase = phases[int(rng.integers(len(phases)))]
                patch[axis % period == phase] = colors[0]
    info["content_object_cells"] = int((occupied & free).sum())
    info["_object_mask"] = occupied
    return grid, info


def _reuse_frame_color(rng, grid, border, frame):
    """An optional clean content rectangle in the marker colour, with a moat.

    This deliberately makes whole-colour exclusivity an unreliable frame
    recognizer.  The geometric guard still checks all potential frames.
    """
    h, w = grid.shape
    color = frame.color
    p = np.pad(border, 1)
    halo = np.zeros_like(border)
    for dr in range(3):
        for dc in range(3):
            halo |= p[dr:dr + h, dc:dc + w]
    allowed = ~halo
    painted = np.zeros_like(border)
    if not ((allowed[:, :-1] & allowed[:, 1:]).any() or (allowed[:-1] & allowed[1:]).any()):
        return painted  # No two-cell content shape fits apart from the frame.
    starts = np.argwhere(allowed)
    counts = np.bincount(grid.ravel(), minlength=10)
    for _ in range(60):
        r, c = starts[int(rng.integers(len(starts)))]
        rh = int(rng.integers(1, min(max(2, h // 4), h - r) + 1))
        rw = int(rng.integers(1, min(max(2, w // 4), w - c) + 1))
        if rh * rw < 2 or not allowed[r:r + rh, c:c + rw].all():
            continue
        removed = np.bincount(grid[r:r + rh, c:c + rw].ravel(), minlength=10)
        if any(removed[k] >= counts[k] for k in range(10) if counts[k] and k != color):
            continue
        saved = grid[r:r + rh, c:c + rw].copy()
        grid[r:r + rh, c:c + rw] = color
        if find_frames(grid) == [frame] and _component_frames(grid) == [frame]:
            painted[r:r + rh, c:c + rw] = True
            return painted
        grid[r:r + rh, c:c + rw] = saved
    # This decoration is optional.  A tiny two-colour crop may have no safe
    # extra: filling its centre can turn the other colour into a second frame.
    # Keep the original sampled scene, subject to the usual uniqueness proof.
    return painted


def generate_extract_subgrid(rng: np.random.Generator) -> dict:
    h, w = (int(rng.integers(3, 31)) for _ in range(2))
    crop_mode = str(rng.choice(["full", "large", "varied", "strip"], p=[.20, .25, .50, .05]))
    if crop_mode == "full":
        ih, iw = h - 2, w - 2
    elif crop_mode == "large":
        ih = int(rng.integers(max(1, int(np.ceil((h - 2) * .65))), h - 1))
        iw = int(rng.integers(max(1, int(np.ceil((w - 2) * .65))), w - 1))
    else:
        ih, iw = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        if crop_mode == "strip":
            if rng.integers(2):
                ih = 1
            else:
                iw = 1
    fh, fw = ih + 2, iw + 2
    free_count = h * w - (2 * (fh + fw) - 4)
    # Budget at least two visible cells per content colour where possible.  The
    # sole cell in a 3x3 framed input is the necessary exception.
    palette_cap = min(10, 1 + max(1, free_count // 2))
    n_colors = int(rng.integers(2, palette_cap + 1))
    palette = [int(x) for x in rng.choice(10, n_colors, replace=False)]
    frame_color, colors = palette[0], palette[1:]
    family = str(rng.choice(SCENE_FAMILIES))
    # Mirrored tiles can have fewer independent visible positions on tiny
    # inputs.  Feasibility is chosen once, never by switching modes on failure.
    if family == "mirror" and min(h, w) < 5:
        family = "patchwork"
    if family == "tiles" and free_count < 4 * len(colors):
        family = "patchwork"
    reuse_frame_color = bool(rng.random() < .35)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = int(rng.integers(0, h - fh + 1))
        c = int(rng.integers(0, w - fw + 1))
        frame = Frame(r, c, fh, fw, frame_color)
        border = np.zeros((h, w), dtype=bool)
        border[r, c:c + fw] = border[r + fh - 1, c:c + fw] = True
        border[r:r + fh, c] = border[r:r + fh, c + fw - 1] = True
        rendered = _render_scene(rng, ~border, colors, family)
        if rendered is None:
            continue
        grid, info = rendered
        grid[border] = frame_color
        extras = np.zeros_like(border)
        if reuse_frame_color:
            extras = _reuse_frame_color(rng, grid, border, frame)
        object_mask = info.pop("_object_mask", np.zeros_like(border))
        info["content_object_cells"] = int(((object_mask | extras) & ~border).sum())
        if extras.any():
            er, ec = np.where(extras)
            info["object_kinds"].append("bar" if np.ptp(er) == 0 or np.ptp(ec) == 0 else "rect")
        if len(np.unique(grid)) != n_colors:
            continue
        # The first check is a component proof, independent of the geometric
        # final-grid solver.  The second catches attached competing rings too.
        if _component_frames(grid) != [frame] or find_frames(grid) != [frame]:
            continue
        out = grid[r + 1:r + fh - 1, c + 1:c + fw - 1].tolist()
        result = {
            "input": grid.tolist(), "output": out,
            "metadata": {
                "grid_h": h, "grid_w": w, "n_colors": n_colors,
                "palette_cap": palette_cap, "palette": palette,
                "roi_color": frame_color, "frame_bbox": list(frame[:4]),
                "roi_top_left": [r, c], "roi_h": fh, "roi_w": fw,
                "crop_h": ih, "crop_w": iw, "crop_fraction": ih * iw / (h * w),
                "crop_mode": crop_mode, "family": family,
                "background_rule": "copy_every_interior_cell", "texture_named": False,
                "n_frame_candidates": 1, "frame_color_named": False,
                "frame_color_reused": bool(extras.any()),
                "frame_color_extra_cells": int(extras.sum()),
                "frame_touches_edge": bool(r == 0 or c == 0 or r + fh == h or c + fw == w),
                "attempts": attempt, **info,
            },
        }
        validate_example(result)
        return result
    raise RuntimeError(f"Task 26 could not realize fixed {h}x{w}, {ih}x{iw} crop, "
                       f"{n_colors}-colour {family} scene after {MAX_ATTEMPTS} attempts")
