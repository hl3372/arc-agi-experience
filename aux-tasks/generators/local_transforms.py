"""Local-transform family generators: tasks 9, 12, 39, 40, 41, 54."""
from __future__ import annotations

import numpy as np

from common.grid import (
    NUM_COLORS,
    generate_scene,
    grid_to_list,
    make_grid,
    pick_colors,
    random_background,
    scatter_cells,
)
from common.objects import (
    random_shape,
    random_square_shape,
    place_shape_on_grid,
    bounding_box,
    normalize_shape,
    rotate_shape_cw,
)
from common.backgrounds import apply_background, isolated_cell_mask
from common.shapes import FAMILIES as _SHAPE_FAMILIES, sample_shape, shape_size
from common.transforms import (
    magnify_grid,
    rotate_grid,
    translate_grid,
)
from registry import register

MAX_RETRIES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_content_grid(
    rng: np.random.Generator,
    min_dim: int = 3,
    max_dim: int = 10,
    *,
    square: bool = False,
    rectangular: bool = False,
) -> np.ndarray:
    """Create a small grid with random non-trivial content using generate_scene."""
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(min_dim, max_dim + 1))
        if square:
            w = h
        elif rectangular:
            w = int(rng.integers(min_dim, max_dim + 1))
            if w == h:
                w = h + 1 if h < max_dim else h - 1
        else:
            w = int(rng.integers(min_dim, max_dim + 1))

        n_objects = int(rng.integers(2, 5))
        grid, bg, placed = generate_scene(
            rng, h, w, n_objects=n_objects, bg_style="random",
            size_range=(2, max(3, min(h, w) // 2)),
        )

        if len(np.unique(grid)) >= 2:
            return grid

    return grid


def _stack_side_by_side(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    divider_color: int,
) -> np.ndarray:
    """Stack two grids side by side with a 1-cell divider column.

    If grids have different heights, pad the shorter one with 0 (black).
    Returns [grid_a | divider_col | grid_b].
    """
    ha, wa = grid_a.shape
    hb, wb = grid_b.shape
    h = max(ha, hb)

    # Pad to same height if needed
    if ha < h:
        pad = np.zeros((h - ha, wa), dtype=grid_a.dtype)
        grid_a = np.vstack([grid_a, pad])
    if hb < h:
        pad = np.zeros((h - hb, wb), dtype=grid_b.dtype)
        grid_b = np.vstack([grid_b, pad])

    divider = np.full((h, 1), divider_color, dtype=grid_a.dtype)
    combined = np.hstack([grid_a, divider, grid_b])
    return combined


def _stack_top_bottom(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    divider_color: int,
) -> np.ndarray:
    """Stack two grids vertically with a 1-cell divider row (pads narrower one with 0)."""
    ha, wa = grid_a.shape
    hb, wb = grid_b.shape
    w = max(wa, wb)
    if wa < w:
        grid_a = np.hstack([grid_a, np.zeros((ha, w - wa), dtype=grid_a.dtype)])
    if wb < w:
        grid_b = np.hstack([grid_b, np.zeros((hb, w - wb), dtype=grid_b.dtype)])
    divider = np.full((1, w), divider_color, dtype=grid_a.dtype)
    return np.vstack([grid_a, divider, grid_b])


def _pick_divider_color(grid_a: np.ndarray, grid_b: np.ndarray, rng: np.random.Generator) -> int:
    """Pick a color not used in either grid."""
    used = set(np.unique(grid_a).tolist()) | set(np.unique(grid_b).tolist())
    available = [c for c in range(NUM_COLORS) if c not in used]
    if available:
        return int(rng.choice(available))
    # Fallback: pick least used color
    counts_a = np.bincount(grid_a.flatten(), minlength=NUM_COLORS)
    counts_b = np.bincount(grid_b.flatten(), minlength=NUM_COLORS)
    total = counts_a + counts_b
    return int(np.argmin(total))


def _is_invariant_under_rotation(grid: np.ndarray, k: int) -> bool:
    """Check if grid is unchanged by k*90 CW rotation."""
    rotated = rotate_grid(grid, k)
    if grid.shape != rotated.shape:
        return False
    return np.array_equal(grid, rotated)


def _matching_rotation_angles(grid: np.ndarray, target: np.ndarray) -> list[int]:
    """Return all non-zero 90-degree rotation angles that map *grid* to *target*."""
    matches: list[int] = []
    for k in (1, 2, 3):
        rotated = rotate_grid(grid, k)
        if rotated.shape == target.shape and np.array_equal(rotated, target):
            matches.append(k * 90)
    return matches


def _matching_translations(
    grid: np.ndarray,
    target: np.ndarray,
    bg: int,
    max_shift: int = 4,
) -> list[tuple[int, int]]:
    """Return all translations in the search range that map *grid* to *target*."""
    matches: list[tuple[int, int]] = []
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(translate_grid(grid, dr, dc, bg=bg), target):
                matches.append((dr, dc))
    return matches


def _clear_background_color(grid: np.ndarray) -> int | None:
    """Return the unique dominant background color, or None if unclear."""
    counts = np.bincount(grid.flatten(), minlength=NUM_COLORS)
    top = int(counts.max())
    winners = [idx for idx, count in enumerate(counts.tolist()) if count == top]
    if len(winners) != 1:
        return None
    bg = winners[0]
    second = max((count for idx, count in enumerate(counts.tolist()) if idx != bg), default=0)
    if top < second + 2:
        return None
    return int(bg)


def _has_full_span_foreground(grid: np.ndarray, bg: int) -> bool:
    """Return True if any non-background color spans the full width or height."""
    if np.any(np.all(grid != bg, axis=0)) or np.any(np.all(grid != bg, axis=1)):
        return True
    h, w = grid.shape
    for color in np.unique(grid):
        color_i = int(color)
        if color_i == bg:
            continue
        cells = np.argwhere(grid == color_i)
        if cells.size == 0:
            continue
        rows = cells[:, 0]
        cols = cells[:, 1]
        if rows.max() - rows.min() + 1 == h:
            return True
        if cols.max() - cols.min() + 1 == w:
            return True
    return False


def _rotate_cells_around_center(
    cells: list[tuple[int, int]],
    center_r: float,
    center_c: float,
    k: int,
) -> list[tuple[int, int]]:
    """Rotate cell positions by k*90 CW around a center point.

    k=1: (r,c) -> (center_r + (c - center_c), center_c - (r - center_r))
    k=2: (r,c) -> (center_r - (r - center_r), center_c - (c - center_c))
    k=3: (r,c) -> (center_r - (c - center_c), center_c + (r - center_r))
    """
    result = []
    for r, c in cells:
        dr = r - center_r
        dc = c - center_c
        if k == 1:
            nr = center_r + dc
            nc = center_c - dr
        elif k == 2:
            nr = center_r - dr
            nc = center_c - dc
        elif k == 3:
            nr = center_r - dc
            nc = center_c + dr
        else:
            nr, nc = r, c
        result.append((round(nr), round(nc)))
    return result


# ---------------------------------------------------------------------------
# Task 9 — Infer rotation between two grids
# ---------------------------------------------------------------------------
#
# Two images A and B separated by a one-cell divider (side by side or stacked);
# B is a rotation of A and the answer is the angle in the direction the prompt
# names.  Sizes: the dimension that is not doubled by the layout runs to 30,
# the doubled one (2s + 1 <= 30) to 14.  For 180 degrees the halves are
# rectangular (h x w drawn independently, near-flat); for 90 / 270 degrees the
# rotated image has swapped dimensions, so both halves can only share the
# layout when they are square (s <= 14) -- that square share is structural.
# Content per half is drawn from several families (object scenes, optionally
# textured from common.backgrounds; a single object; sparse dots; mosaics;
# structured patterns with a marker), with a near-flat palette draw over
# 2..9 colours so the combined grid reaches all 10.  Every example is
# re-validated from the visible combined grid: the divider is the unique full
# line of a colour present nowhere else, exactly one rotation in {90, 180,
# 270} maps A to B, and the stated answer is that rotation in the stated
# direction.  No canned fallback: the generator raises when it cannot build
# a valid example.

_T9_FAMILIES = ("scene", "single", "dots", "mosaic", "structured")
_T9_FAMILY_P = (0.38, 0.12, 0.14, 0.14, 0.22)
_T9_COLOR_P = (0.13, 0.13, 0.125, 0.125, 0.125, 0.12, 0.12, 0.125)   # n = 2..9, near-flat target
_T9_ATTEMPTS = 400


def _t9_rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


def _t9_n_colors(rng: np.random.Generator, cap: int) -> int:
    """Target number of colours in a half (background included): near-flat over 2..9."""
    n = 2 + int(rng.choice(8, p=np.array(_T9_COLOR_P) / sum(_T9_COLOR_P)))
    return max(2, min(n, cap))


def _t9_half_dims(rng: np.random.Generator, k: int, layout: str) -> tuple[int, int]:
    """Half size.  90/270: square s in 3..14.  180: the doubled side 3..14, the free side 3..30."""
    if k in (1, 3):
        s = _t9_rint(rng, 3, 14)
        return s, s
    # the free side leans to 15..30 (60 %) because only 180-degree examples reach it,
    # which keeps the per-value density of the combined grid's free dimension near-flat
    free = _t9_rint(rng, 15, 30) if rng.random() < 0.60 else _t9_rint(rng, 3, 14)
    if layout == "horizontal":      # side by side: width is doubled, height is free
        return free, _t9_rint(rng, 3, 14)
    return _t9_rint(rng, 3, 14), free


def _t9_place(rng: np.random.Generator, grid: np.ndarray, occupied: np.ndarray, shape, color: int,
              moat: bool = True) -> bool:
    """Place ``shape`` at a random anchor where its cells (and, with ``moat``, their
    8-neighbourhood) are free.  Returns True when placed."""
    H, W = grid.shape
    offs = sorted(shape)
    sh = max(r for r, _ in offs) + 1
    sw = max(c for _, c in offs) + 1
    if sh > H or sw > W:
        return False
    blocked = occupied
    if moat:
        pad = np.zeros((H + 2, W + 2), dtype=bool)
        pad[1:-1, 1:-1] = occupied
        blocked = np.zeros_like(occupied)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                blocked |= pad[dr:dr + H, dc:dc + W]
    nh, nw = H - sh + 1, W - sw + 1
    valid = np.ones((nh, nw), dtype=bool)
    allowed = ~blocked
    for dr, dc in offs:
        valid &= allowed[dr:dr + nh, dc:dc + nw]
    anchors = np.argwhere(valid)
    if len(anchors) == 0:
        return False
    r0, c0 = (int(v) for v in anchors[int(rng.integers(0, len(anchors)))])
    for dr, dc in offs:
        grid[r0 + dr, c0 + dc] = color
        occupied[r0 + dr, c0 + dc] = True
    return True


def _t9_scene(rng: np.random.Generator, h: int, w: int, n_colors: int) -> tuple[np.ndarray, dict]:
    """1..6 objects from the shared shape vocabulary on a flat ground, sizes scaling
    with the half's area; optionally a library texture on top (short side >= 8)."""
    area = h * w
    bg = int(rng.integers(0, NUM_COLORS))
    grid = make_grid(h, w, fill=bg)
    occupied = np.zeros((h, w), dtype=bool)
    hi = max(1, min(8, area // 12))
    lo = max(1, min(hi, n_colors - 1))       # enough objects to show the target palette
    n_obj = _t9_rint(rng, lo, hi)
    palette = [int(c) for c in pick_colors(rng, max(1, min(n_colors - 1, n_obj)), exclude=[bg])]
    order = [palette[i] for i in rng.permutation(len(palette)).tolist()]
    max_cells = max(2, min(60, int(0.6 * area / n_obj)))
    max_h = max(1, (3 * h + 3) // 4)
    max_w = max(1, (3 * w + 3) // 4)
    placed = 0
    for i in range(n_obj):
        shape, _fam = sample_shape(rng, max_cells, max_h, max_w)
        if len(shape) < 2 and area >= 12:
            shape, _fam = sample_shape(rng, max_cells, max_h, max_w)
        if _t9_place(rng, grid, occupied, shape, order[i % len(order)], moat=(rng.random() < 0.85)):
            placed += 1
    info = {"content_family": "scene", "n_objects": placed,
            "texture_group": None, "texture_kind": None, "texture_colors": []}
    if min(h, w) >= 8 and rng.random() < 0.50:
        got = apply_background(rng, grid, bg, occupied)
        if got is not None:
            grid, tinfo = got
            info.update({"texture_group": tinfo["group"], "texture_kind": tinfo["kind"],
                         "texture_colors": [int(c) for c in tinfo["texture_colors"]]})
    return grid, info


def _t9_single(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, dict]:
    """One object (up to half the canvas) on a flat ground."""
    bg = int(rng.integers(0, NUM_COLORS))
    grid = make_grid(h, w, fill=bg)
    occupied = np.zeros((h, w), dtype=bool)
    color = int(pick_colors(rng, 1, exclude=[bg])[0])
    for _ in range(20):
        shape, _fam = sample_shape(rng, max(2, min(60, h * w // 2)), h, w)
        if len(shape) >= 2 and _t9_place(rng, grid, occupied, shape, color, moat=False):
            break
    return grid, {"content_family": "single", "n_objects": 1,
                  "texture_group": None, "texture_kind": None, "texture_colors": []}


def _t9_dots(rng: np.random.Generator, h: int, w: int, n_colors: int) -> tuple[np.ndarray, dict]:
    """Sparse dots in 1..8 colours on a flat ground (5-25 % of the cells)."""
    bg = int(rng.integers(0, NUM_COLORS))
    grid = make_grid(h, w, fill=bg)
    palette = [int(c) for c in pick_colors(rng, max(1, n_colors - 1), exclude=[bg])]
    n = max(1, int(round(h * w * float(rng.uniform(0.08, 0.30)))))
    n = min(n, h * w - 1)
    picks = rng.choice(h * w, size=n, replace=False)
    # every palette colour at least once (when there are enough dots), the rest at random
    cols = [palette[i % len(palette)] for i in range(min(n, len(palette)))]
    cols += [palette[int(i)] for i in rng.integers(0, len(palette), size=n - len(cols))]
    grid.flat[picks] = cols
    return grid, {"content_family": "dots", "n_objects": 0,
                  "texture_group": None, "texture_kind": None, "texture_colors": []}


def _t9_mosaic(rng: np.random.Generator, h: int, w: int, n_colors: int) -> tuple[np.ndarray, dict]:
    """I.i.d. colour noise over 2..9 colours."""
    colors = np.array([int(c) for c in pick_colors(rng, max(2, n_colors))], dtype=np.uint8)
    grid = colors[rng.integers(0, len(colors), size=(h, w))]
    return grid, {"content_family": "mosaic", "n_objects": 0,
                  "texture_group": None, "texture_kind": None, "texture_colors": []}


def _t9_structured(rng: np.random.Generator, h: int, w: int, n_colors: int) -> tuple[np.ndarray, dict]:
    """Stripes / checker / concentric / diagonal / mirrored pattern in 2..4 colours, plus a
    small marker object in a fresh colour so the pattern is not rotation-symmetric."""
    kind = ("stripes", "checker", "concentric", "diagonal", "mirror")[int(rng.integers(0, 5))]
    rr, cc = np.indices((h, w))
    # stripes / diagonals / rings carry up to 8 colours, checker 2-3, mirrors up to 4
    n_pat = max(2, min(8 if kind in ("stripes", "diagonal", "concentric") else 4, n_colors - 1))
    colors = np.array([int(c) for c in pick_colors(rng, n_pat)], dtype=np.uint8)
    if kind == "mirror":
        mode = ("lr", "ud", "both")[int(rng.integers(0, 3))]
        hh = (h + 1) // 2 if mode in ("ud", "both") else h
        ww = (w + 1) // 2 if mode in ("lr", "both") else w
        part = colors[rng.integers(0, len(colors), size=(hh, ww))]
        if mode in ("ud", "both"):
            part = np.concatenate([part, part[: h - hh][::-1]], axis=0)
        if mode in ("lr", "both"):
            part = np.concatenate([part, part[:, : w - ww][:, ::-1]], axis=1)
        grid = part.astype(np.uint8)
    else:
        t = _t9_rint(rng, 1, 3)
        if kind == "stripes":
            layer = (rr if rng.random() < 0.5 else cc) // t
        elif kind == "checker":
            layer = (rr // t) + (cc // t)
            colors = colors[:2] if rng.random() < 0.7 else colors[:3]
        elif kind == "concentric":
            t = 1 if min(h, w) < 6 else _t9_rint(rng, 1, 2)
            layer = np.minimum(np.minimum(rr, cc), np.minimum(h - 1 - rr, w - 1 - cc)) // t
        else:
            layer = ((rr + cc) if rng.random() < 0.5 else (rr - cc + w)) // t
        grid = colors[layer % len(colors)].astype(np.uint8)
    # marker: a 1-4 cell blob in a colour the pattern does not use
    free = [c for c in range(NUM_COLORS) if c not in set(int(v) for v in np.unique(grid))]
    if free:
        mcol = int(free[int(rng.integers(0, len(free)))])
        size = _t9_rint(rng, 1, min(4, max(1, h * w // 6)))
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        cells = {(r, c)}
        while len(cells) < size:
            pr, pc = list(cells)[int(rng.integers(0, len(cells)))]
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(0, 4))]
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
            elif len(cells) == 1 and size > 1 and h * w < 4:
                break
        for (r2, c2) in cells:
            grid[r2, c2] = mcol
    return grid, {"content_family": "structured", "n_objects": 0, "pattern": kind,
                  "texture_group": None, "texture_kind": None, "texture_colors": []}


def _t9_content(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, dict]:
    fam = _T9_FAMILIES[int(rng.choice(len(_T9_FAMILIES), p=_T9_FAMILY_P))]
    n_colors = _t9_n_colors(rng, cap=9)
    if fam == "scene":
        grid, info = _t9_scene(rng, h, w, n_colors)
    elif fam == "single":
        grid, info = _t9_single(rng, h, w)
    elif fam == "dots":
        grid, info = _t9_dots(rng, h, w, n_colors)
    elif fam == "mosaic":
        grid, info = _t9_mosaic(rng, h, w, n_colors)
    else:
        grid, info = _t9_structured(rng, h, w, n_colors)
    info["n_colors_target"] = int(n_colors)
    return np.asarray(grid, dtype=np.uint8), info


def _t9_stack(a: np.ndarray, b: np.ndarray, layout: str, divider: int) -> np.ndarray:
    if layout == "horizontal":
        assert a.shape[0] == b.shape[0]
        return np.hstack([a, np.full((a.shape[0], 1), divider, dtype=a.dtype), b])
    assert a.shape[1] == b.shape[1]
    return np.vstack([a, np.full((1, a.shape[1]), divider, dtype=a.dtype), b])


def _t9_find_divider(combined: np.ndarray, layout: str):
    """The unique full row (stacked) / column (side by side) whose colour occurs
    nowhere else in the grid; returns its index or None."""
    H, W = combined.shape
    found = []
    n = H if layout == "vertical" else W
    for i in range(n):
        line = combined[i, :] if layout == "vertical" else combined[:, i]
        if not np.all(line == line[0]):
            continue
        col = int(line[0])
        total = int((combined == col).sum())
        if total == len(line):
            found.append(i)
    return found[0] if len(found) == 1 else None


def _t9_validate(combined: np.ndarray, layout: str, answer: int, direction: str) -> dict:
    """Independent check from the visible combined grid; raises on any violation."""
    idx = _t9_find_divider(combined, layout)
    if idx is None:
        raise RuntimeError("task 9: divider is not uniquely identifiable")
    if layout == "vertical":
        a, b = combined[:idx, :], combined[idx + 1:, :]
    else:
        a, b = combined[:, :idx], combined[:, idx + 1:]
    if a.size == 0 or b.size == 0:
        raise RuntimeError("task 9: empty half")
    angles = [k * 90 for k in (1, 2, 3)
              if rotate_grid(a, k).shape == b.shape and np.array_equal(rotate_grid(a, k), b)]
    if len(angles) != 1:
        raise RuntimeError(f"task 9: {len(angles)} rotations map A to B")
    cw = angles[0]
    expected = cw if direction == "clockwise" else (360 - cw) % 360
    if expected == 0:
        expected = 180
    if int(answer) != expected:
        raise RuntimeError(f"task 9: stated {answer} != {expected} ({direction})")
    if np.array_equal(a, b):
        raise RuntimeError("task 9: halves identical")
    return {"cw_angle": cw, "half_h": int(a.shape[0]), "half_w": int(a.shape[1])}


@register("9")
def generate_infer_rotation(rng: np.random.Generator) -> dict:
    """Task 9: infer the rotation angle between two images shown side by side or stacked.

    Sizes: the free dimension runs 3..30, the doubled one 3..14 (combined 2s + 1
    <= 29); 180-degree examples have rectangular halves, 90/270 square ones
    (required by the layout).  Content families: object scenes (optionally
    textured), single object, sparse dots, mosaic, structured pattern with a
    marker; near-flat palette 2..9 per half so the combined grid reaches 10.
    """
    k = int(rng.integers(1, 4))
    cw_angle = k * 90
    direction = "clockwise" if rng.random() < 0.5 else "counterclockwise"
    answer = cw_angle if direction == "clockwise" else (360 - cw_angle) % 360
    if answer == 0:
        answer = 180
    layout = "horizontal" if rng.random() < 0.5 else "vertical"
    h, w = _t9_half_dims(rng, k, layout)

    for _ in range(_T9_ATTEMPTS):
        grid, info = _t9_content(rng, h, w)
        if len(np.unique(grid)) < 2:
            continue
        rotated = rotate_grid(grid, k)
        if rotated.shape == grid.shape and np.array_equal(grid, rotated):
            continue
        if _matching_rotation_angles(grid, rotated) != [cw_angle]:
            continue
        used = {int(v) for v in np.unique(grid)}
        free = [c for c in range(NUM_COLORS) if c not in used]
        if not free:
            continue
        divider = int(free[int(rng.integers(0, len(free)))])
        combined = _t9_stack(grid, rotated, layout, divider)
        check = _t9_validate(combined, layout, answer, direction)
        metadata = {
            "angle": int(answer),
            "rotation_direction": direction,
            "layout": layout,
            "cw_angle": int(check["cw_angle"]),
            "half_h": int(h),
            "half_w": int(w),
            "divider_color": divider,
            "n_colors": int(len(used)),
            **info,
        }
        return {"input": grid_to_list(combined), "output": str(answer), "metadata": metadata}
    raise RuntimeError("Task 9: failed to generate a uniquely inferable rotation")


# ---------------------------------------------------------------------------
# Task 12 — Rotate one object
# ---------------------------------------------------------------------------

# Only the irregular sparse kinds are used as texture: a regular dot lattice would suggest
# a continuation under the object, and the cells the object vacates become plain background.
_T12_TEXTURE_KINDS = ("random_dots", "multi_speckle")
_T12_TEXTURE_P = 0.22
_T12_BG_MARGIN = 1.25          # identifiability: background cells >= 1.25x every other colour
_T12_SIDE2_P = 0.08            # share of side-2 targets (the L-tromino) when a larger box fits; it used to be a quarter
# Vocabulary families for the target: square-boxed, rotation-asymmetric candidates; the
# rotation-symmetric families (square, frame, plus, X) can never be targets.
_T12_TARGET_WEIGHTS = {"polyomino": 0.24, "L": 0.12, "T": 0.10, "U": 0.10, "staircase": 0.10,
                       "blob": 0.16, "big_blob": 0.06, "scaled": 0.06, "rect": 0.03, "H": 0.03}


def _t12_rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


def _t12_object_budget(area: int) -> tuple[tuple[int, int], int]:
    """(range of non-target objects, max cells per object): grows with the canvas so large
    grids carry large scenes."""
    if area < 100:
        return (1, 3), 12
    if area < 250:
        return (2, 5), 20
    if area < 450:
        return (3, 8), 30
    if area < 700:
        return (4, 11), 42
    return (5, 14), 56


def _t12_rotate_in_box(cells, side: int, k: int) -> frozenset:
    """Rotate box-relative cells k quarter turns clockwise about the centre of a side x side box."""
    out = set()
    for r, c in cells:
        for _ in range(k % 4):
            r, c = c, side - 1 - r
        out.add((int(r), int(c)))
    return frozenset(out)


def _t12_box_walk(rng: np.random.Generator, side: int, conn: int, n_cells: int) -> frozenset:
    """A connected random walk inside a side x side box, then extended so its bounding box
    is exactly the box (a run from an extreme cell to each untouched edge)."""
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1)) if conn == 4 else \
        ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    start = (int(rng.integers(0, side)), int(rng.integers(0, side)))
    cells = {start}
    while len(cells) < n_cells:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in deltas
                           if 0 <= r + dr < side and 0 <= c + dc < side} - cells)
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if min(rs) > 0:
        r, c = min(cells, key=lambda x: (x[0], rng.random()))
        cells.update((rr, c) for rr in range(0, r))
    if max(rs) < side - 1:
        r, c = max(cells, key=lambda x: (x[0], rng.random()))
        cells.update((rr, c) for rr in range(r + 1, side))
    if min(cs) > 0:
        r, c = min(cells, key=lambda x: (x[1], rng.random()))
        cells.update((r, cc) for cc in range(0, c))
    if max(cs) < side - 1:
        r, c = max(cells, key=lambda x: (x[1], rng.random()))
        cells.update((r, cc) for cc in range(c + 1, side))
    return frozenset(cells)


def _t12_target_shape(rng: np.random.Generator, side: int, conn: int, k: int):
    """A shape with a square bounding box of at most side x side, with >= 3 cells, that
    changes under k quarter turns.  Half the draws come from the shape vocabulary (any
    square-boxed family that is not rotation-symmetric survives: L, T, U, staircases,
    blobs, polyominoes, scaled shapes ...), the rest are box walks at the drawn side.
    Returns (shape, family, side)."""
    for attempt in range(80):
        if attempt % 5 != 4:   # four vocabulary draws for every box walk (walks used to be 79%)
            cap = max(2, side)
            shape, fam = sample_shape(rng, cap * cap, cap, cap, weights=_T12_TARGET_WEIGHTS, connectivity=conn)
            shape = normalize_shape(shape)
        else:
            n = _t12_rint(rng, max(3, side + 1), max(3, min(side * side - 1, 2 * side + 3)))
            shape, fam = _t12_box_walk(rng, side, conn, n), "walk"
        sh, sw = shape_size(shape)
        if sh != sw or sh < 2 or len(shape) < 3 or len(shape) >= sh * sw:
            continue
        if side >= 3 and sh < max(3, (side + 1) // 2):
            continue           # the vocabulary must honour the drawn side: no L-tromino when a bigger box was drawn
        if _t12_rotate_in_box(shape, sh, k) == frozenset(shape):
            continue
        return frozenset(shape), fam, int(sh)
    return None, None, None


def _t12_components(grid: np.ndarray, color: int, conn: int) -> list[set]:
    """Connected components of one colour (own BFS, used by the validator)."""
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1)) if conn == 4 else \
        ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    comps = []
    for r0, c0 in map(tuple, np.argwhere(grid == color)):
        if seen[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        seen[r0, c0] = True
        comp = set()
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for dr, dc in deltas:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc] and grid[rr, cc] == color:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)
    return comps


def _t12_validate(inp: np.ndarray, out: np.ndarray, target_color: int, k: int, conn: int, bg: int) -> None:
    """Independent check from the visible input: exactly one object of the target colour, a
    square bounding box, the output equals the input with that object rotated k quarter
    turns clockwise about the box centre and its vacated cells set to the background."""
    comps = _t12_components(inp, target_color, conn)
    if len(comps) != 1:
        raise RuntimeError(f"task 12: {len(comps)} objects of the target colour")
    cells = comps[0]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    side = max(rs) - r0 + 1
    if max(cs) - c0 + 1 != side:
        raise RuntimeError("task 12: target bounding box is not square")
    rel = {(r - r0, c - c0) for r, c in cells}
    rotated = {(r0 + r, c0 + c) for r, c in _t12_rotate_in_box(rel, side, k)}
    if rotated == set(cells):
        raise RuntimeError("task 12: rotation leaves the target unchanged")
    expected = inp.copy()
    for r, c in cells:
        expected[r, c] = bg
    for r, c in rotated:
        if inp[r, c] != bg and (r, c) not in cells:
            raise RuntimeError("task 12: rotated target collides with another object")
        expected[r, c] = target_color
    if not np.array_equal(expected, out):
        raise RuntimeError("task 12: output is not the input with the target rotated in place")


def _t12_build(rng: np.random.Generator, h: int, w: int, k: int, conn: int):
    """One attempt at a task-12 scene on an h x w grid; None when it cannot be built."""
    area = h * w
    bg = int(rng.integers(0, NUM_COLORS))
    n_total = _t12_rint(rng, 3, 10)                       # near-flat visible colours (bg + target + others)
    (lo_obj, hi_obj), max_cells = _t12_object_budget(area)
    n_other = _t12_rint(rng, lo_obj, hi_obj)
    n_other_colors = max(1, min(n_total - 2, n_other, 8))
    colors = [int(c) for c in pick_colors(rng, 1 + n_other_colors, exclude=[bg])]
    target_color, others = colors[0], colors[1:]

    cap = max(2, min(12, (2 * min(h, w)) // 3))       # a side-3 box fits from a short side of 5
    if cap <= 2 or rng.random() < _T12_SIDE2_P:
        side = 2                                              # 2-cell boxes stay, but do not dominate
    else:
        side = _t12_rint(rng, 3, cap)
    shape, fam, side = _t12_target_shape(rng, side, conn, k)
    if shape is None:
        return None
    r0 = int(rng.integers(0, h - side + 1))
    c0 = int(rng.integers(0, w - side + 1))
    grid = make_grid(h, w, fill=bg)
    target_cells = [(r0 + r, c0 + c) for r, c in shape]
    for r, c in target_cells:
        grid[r, c] = target_color

    # The whole box plus a one-cell moat is reserved: the rotated object stays inside the
    # box, and nothing else may touch it, so the target is the unique object of its colour.
    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, r0 - 1):r0 + side + 1, max(0, c0 - 1):c0 + side + 1] = True

    order = [others[i] for i in rng.permutation(len(others)).tolist()]
    max_h = max(1, (3 * h + 3) // 4)
    max_w = max(1, (3 * w + 3) // 4)
    placed = 0
    for i in range(n_other):
        shape_o, _fam = sample_shape(rng, max_cells, max_h, max_w, connectivity=conn)
        if len(shape_o) < 2:
            shape_o, _fam = sample_shape(rng, max_cells, max_h, max_w, connectivity=conn)
        if len(shape_o) < 2:
            continue
        if _t9_place(rng, grid, occupied, shape_o, order[i % len(order)], moat=(rng.random() < 0.8)):
            placed += 1
    if placed < 1:
        return None

    tex = None
    if min(h, w) >= 8 and rng.random() < _T12_TEXTURE_P:
        kind = _T12_TEXTURE_KINDS[int(rng.integers(0, len(_T12_TEXTURE_KINDS)))]
        painted = apply_background(rng, grid, bg, occupied, kind=kind, exclude=[target_color] + others)
        if painted is not None:
            grid, tex = painted

    # Identifiability without naming the background: after isolated single cells are set
    # aside, the background must be the strict plurality by a margin and own the largest
    # 4-connected region (a solver reading "become background" has to find it).
    margin = _t12_background_margin(grid, bg)
    if margin is None:
        return None

    out = grid.copy()
    for r, c in target_cells:
        out[r, c] = bg
    for r, c in _t12_rotate_in_box(shape, side, k):
        out[r0 + r, c0 + c] = target_color
    _t12_validate(grid, out, target_color, k, conn, bg)

    metadata = {
        "target_color": int(target_color),
        "angle": int(90 * k),
        "connectivity": str(conn),
        "background": int(bg),
        "n_objects": int(1 + placed),
        "n_colors": int(len(np.unique(grid))),
        "target_side": int(side),
        "target_cells": int(len(shape)),
        "shape_family": fam,
        "grid_h": int(h),
        "grid_w": int(w),
        "texture_group": None if tex is None else tex["group"],
        "texture_kind": None if tex is None else tex["kind"],
        "texture_colors": [] if tex is None else [int(c) for c in tex["texture_colors"]],
        "texture_named": False,
        "texture_coverage": 0.0 if tex is None else float(tex["coverage"]),
        "background_margin": round(float(margin), 3),
    }
    return {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": metadata}


def _t12_background_margin(grid: np.ndarray, bg: int):
    """Margin of the background over the runner-up colour once isolated single cells are
    set aside, or None when the background is not the strict plurality by
    ``_T12_BG_MARGIN`` or does not own the largest 4-connected region."""
    keep = ~isolated_cell_mask(grid)
    counts = np.bincount(grid[keep].ravel(), minlength=NUM_COLORS).astype(float)
    if int(np.argmax(counts)) != bg:
        return None
    runner = float(np.sort(counts)[-2]) if len(counts) > 1 else 0.0
    margin = counts[bg] / max(1.0, runner)
    if margin < _T12_BG_MARGIN:
        return None
    largest_bg = max(len(c) for c in _t12_components(grid, bg, 4))
    for color in {int(v) for v in np.unique(grid)} - {bg}:
        if max(len(c) for c in _t12_components(grid, color, 4)) >= largest_bg:
            return None
    return margin


@register("12")
def generate_rotate_one_object(rng: np.random.Generator) -> dict:
    """Task 12: rotate only the object of a designated colour by 90 / 180 / 270 degrees
    clockwise about the centre of its square bounding box; everything else stays.

    Sizes 5..30 per side, independent and uniform (content is retried at the same size,
    so rejections no longer tilt the size distribution).  Random background colour;
    visible colours near-flat over 3..10.  The target is a shape with a square bounding
    box (2..12 cells a side, vocabulary shapes with square boxes or a box walk) that
    changes under the requested rotation; the box and a one-cell moat are reserved, so
    the rotated object never collides and the target is the unique object of its colour.
    Non-target objects come from the shared shape vocabulary with a budget that scales
    with the area; 22% of canvases with a short side >= 8 carry an irregular sparse
    texture (random dots or multi-colour speckle) outside the reserved box.  Every
    example is re-solved from the visible input by an independent validator; the
    generator raises instead of falling back.
    """
    k = int(rng.integers(1, 4))
    conn = int(rng.choice([4, 8]))
    for _size_attempt in range(10):
        h = int(rng.integers(5, 31))
        w = int(rng.integers(5, 31))
        for _content_attempt in range(20):
            example = _t12_build(rng, h, w, k, conn)
            if example is not None:
                return example
    raise RuntimeError("task 12: could not build a scene after the retry budget")


# ---------------------------------------------------------------------------
# Task 39 — Rotate object around anchor with clipping
# ---------------------------------------------------------------------------

@register("39")
def generate_rotate_around_anchor(rng: np.random.Generator) -> dict:
    from generators.anchor_translate_39_40 import generate_rotate_around_anchor as generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 40 — Translate all objects with clipping
# ---------------------------------------------------------------------------

@register("40")
def generate_translate_all(rng: np.random.Generator) -> dict:
    from generators.anchor_translate_39_40 import generate_translate_all as generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 41 — Infer translation from input/output pair
# ---------------------------------------------------------------------------

@register("41")
def generate_infer_translation(rng: np.random.Generator) -> dict:
    from generators.infer_translation_41 import generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 54 — Infer magnification scale from pair
# ---------------------------------------------------------------------------

@register("54")
def generate_infer_magnification(rng: np.random.Generator) -> dict:
    from generators.block_scaling_54_55 import generate_pair
    return generate_pair(rng)
