"""Global-transform family generators: tasks 6, 10, 11, 55, 66."""
from __future__ import annotations

import numpy as np

from common.grid import (
    NUM_COLORS,
    generate_scene,
    grid_to_list,
    make_grid,
    pick_color,
    pick_colors,
)
from common.shapes import sample_shape, shape_size
from common.transforms import (
    demagnify_grid,
    magnify_grid,
    reflect_grid,
    rotate_grid,
)
from registry import register
from common import backgrounds as BG

MAX_RETRIES = 20
# Attempt budget for the rejection loops of tasks 10 and 55 (the new
# generators never fall back to an unguarded example: they raise instead).
MAX_ATTEMPTS = 200
# Full ARC range: any 1..30 x 1..30 input (the former 700-cell cap came from the SFT
# bundle budget, which is not a task-design input).
MAX_ROTATION_AREA = 900


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
    """Create a small grid with random non-trivial content using generate_scene.

    square=True  -> force h == w
    rectangular=True -> force h != w
    """
    for _ in range(MAX_RETRIES):
        h = int(rng.integers(min_dim, max_dim + 1))
        if square:
            w = h
        elif rectangular:
            w = int(rng.integers(min_dim, max_dim + 1))
            if w == h:
                # nudge so it's genuinely rectangular
                w = h + 1 if h < max_dim else h - 1
        else:
            w = int(rng.integers(min_dim, max_dim + 1))

        n_objects = int(rng.integers(2, 5))
        grid, bg, placed = generate_scene(
            rng, h, w, n_objects=n_objects, bg_style="random",
            size_range=(2, max(3, min(h, w) // 2)),
        )

        # Ensure at least two distinct colours so the grid is non-trivial
        if len(np.unique(grid)) >= 2:
            return grid

    # Fallback: just return whatever we got
    return grid


def _is_symmetric(grid: np.ndarray) -> bool:
    """Quick check: is the grid unchanged by 90-degree rotation AND both flips?"""
    if grid.shape[0] != grid.shape[1]:
        return False
    return (
        np.array_equal(grid, np.rot90(grid))
        and np.array_equal(grid, np.flipud(grid))
        and np.array_equal(grid, np.fliplr(grid))
    )


def _random_line_grid(rng: np.random.Generator) -> np.ndarray:
    """Create a nonuniform 1xN or Nx1 grid for edge-case rotation coverage."""
    length = int(rng.integers(5, 31))
    horizontal = bool(rng.integers(0, 2))
    colors = rng.choice(NUM_COLORS, size=min(4, NUM_COLORS), replace=False)
    values = [int(colors[int(rng.integers(0, len(colors)))]) for _ in range(length)]
    if len(set(values)) < 2:
        values[-1] = int(colors[1])
    array = np.array(values, dtype=np.uint8)
    return array.reshape(1, length) if horizontal else array.reshape(length, 1)


def _manual_diagonal_reflection(grid: np.ndarray, diag: str) -> np.ndarray:
    """Reflect *grid* across the requested diagonal using coordinate mapping.

    Works for rectangular grids too: an H x W grid reflects across the diagonal
    of its bounding box, producing a W x H grid. Both diagonal reflections are
    proper involutions (applying twice restores the original).
      main diagonal: (r, c) -> (c, r)
      anti-diagonal: (r, c) -> (W-1-c, H-1-r)
    """
    h, w = grid.shape
    out = np.zeros((w, h), dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            if diag == "main":
                nr, nc = c, r
            else:
                nr, nc = w - 1 - c, h - 1 - r
            out[nr, nc] = grid[r, c]
    return out


# ---------------------------------------------------------------------------
# Private helpers for tasks 10 and 55
#
# ``_random_content_grid`` / ``_random_line_grid`` above are shared with tasks
# 6, 11 and 66 and are frozen; everything below is private to 10 and 55.
# ---------------------------------------------------------------------------

def _weighted_choice(rng: np.random.Generator, table: dict) -> object:
    """Draw a key of *table* with probability proportional to its value."""
    keys = list(table)
    probs = np.array([table[k] for k in keys], dtype=float)
    probs /= probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


# Flat palette draw over 2..10 (user rule, Sept 2026: every part of 2..10 reachable,
# no bin overwhelmingly dominant).  The realised distribution still leans toward
# fewer colours because small canvases and simple families cannot show many, so the
# draw itself is flat rather than leaning as well.
_PALETTE_WEIGHTS_2_10 = {k: 1.0 for k in range(2, 11)}


def _draw_n_colors(rng: np.random.Generator, hi: int = 10, lo: int = 2) -> int:
    """Number of colours for one grid: near-flat over [lo, hi] (see _PALETTE_WEIGHTS_2_10)."""
    hi = max(lo, min(10, hi))
    table = {k: w for k, w in _PALETTE_WEIGHTS_2_10.items() if lo <= k <= hi}
    if not table:
        return lo
    return int(_weighted_choice(rng, table))


def _thin_content(rng: np.random.Generator, length: int,
                  n_colors: int | None = None) -> tuple[np.ndarray, str]:
    """Content for a 1 x length (or length x 1) strip, as a 1-D array.

    2-4 colours by default (``n_colors`` overrides, capped by the length); 90%
    coloured runs of length 1-4 on a base colour, 10% i.i.d. noise.  The strip
    is guaranteed to hold >= 2 colours and to be non-palindromic, so a
    180-degree rotation changes it and clockwise / anticlockwise 90-degree
    rotations differ.
    """
    if length < 2:
        raise ValueError("thin strips need length >= 2")
    for _ in range(MAX_ATTEMPTS):
        want = int(rng.integers(2, 5)) if n_colors is None else max(2, min(int(n_colors), length, 10))
        colors = [int(c) for c in pick_colors(rng, want)]
        if rng.random() < 0.10:
            vals = np.array([colors[int(i)] for i in rng.integers(0, len(colors), size=length)],
                            dtype=np.uint8)
            style = "thin_noise"
        else:
            base, accents = colors[0], colors[1:]
            vals = np.full(length, base, dtype=np.uint8)
            n_seg = int(rng.integers(1, max(1, length // 3) + 1))
            if n_colors is not None:
                n_seg = max(n_seg, len(accents))  # enough runs to show every accent colour
            for i in range(n_seg):
                seg = int(rng.integers(1, min(4, length) + 1))
                start = int(rng.integers(0, length - seg + 1))
                accent = accents[i % len(accents)] if n_colors is not None else accents[int(rng.integers(0, len(accents)))]
                vals[start:start + seg] = accent
            style = "thin_runs"
        if len(np.unique(vals)) >= 2 and not np.array_equal(vals, vals[::-1]):
            return vals, style
    raise RuntimeError("could not build a non-palindromic thin strip")


_SCENE_BG_STYLES = {"uniform": 0.55, "noise": 0.20, "striped": 0.15, "checker": 0.10}


def _object_scene(
    rng: np.random.Generator,
    h: int,
    w: int,
    *,
    n_objects_range: tuple[int, int] = (1, 5),
    max_object_colors: int = 4,
    bg_styles: dict | None = None,
    max_cells: int | None = None,
    target_colors: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Background (uniform / striped / checker / speckled) + objects from ``common.shapes``.

    The palette is explicit: 1 background colour, at most 1 texture colour and
    at most ``max_object_colors`` object colours (<= 6 colours in total by
    default).  ``max_cells`` caps the object size (default 60); ``target_colors``
    asks for that many colours in total (background and stripe/checker colour
    included, capped by the object count), assigned round-robin so every
    palette colour is used.  Defaults reproduce the original behaviour exactly.
    Speckle noise only ever uses palette colours.  Returns (grid, info).
    """
    style = str(_weighted_choice(rng, bg_styles or _SCENE_BG_STYLES))
    bg = int(pick_color(rng))
    grid = make_grid(h, w, fill=bg)
    used = [bg]
    tex = None
    if style in ("striped", "checker") and h * w >= 4:
        tex = int(pick_color(rng, exclude=[bg]))
        used.append(tex)
        if style == "striped":
            width = int(rng.integers(1, 4))
            horizontal = bool(rng.random() < 0.5)
            idx = (np.arange(h) if horizontal else np.arange(w)) // width % 2 == 1
            if horizontal:
                grid[idx, :] = tex
            else:
                grid[:, idx] = tex
        else:
            cs = int(rng.integers(1, 3))
            rr, cc = np.indices((h, w))
            grid[((rr // cs) + (cc // cs)) % 2 == 1] = tex
    else:
        style = "noise" if style == "noise" else "uniform"

    lo, hi = n_objects_range
    n_objects = int(rng.integers(lo, hi + 1))
    if target_colors is None:
        # Number of object colours: favour 2-3 (ARC grids use ~3 colours in total).
        cap = min(max_object_colors, n_objects)
        n_colors = int(_weighted_choice(rng, {k: (1.0, 2.0, 2.0, 1.0)[k - 1] for k in range(1, cap + 1)}))
    else:
        # enough objects to show the requested colours, when the canvas has room
        # (about 20 cells per object including spacing)
        room = max(1, (h * w) // 20)
        n_objects = max(n_objects, min(int(target_colors) - len(used), room))
        n_colors = max(1, min(int(target_colors) - len(used), n_objects, NUM_COLORS - len(used)))
    obj_colors = [int(c) for c in pick_colors(rng, n_colors, exclude=used)]
    used.extend(obj_colors)

    occ = np.zeros((h, w), dtype=bool)
    max_h = min(h, max(1, (3 * h + 3) // 4))
    max_w = min(w, max(1, (3 * w + 3) // 4))
    max_cells = max(1, min(60 if max_cells is None else int(max_cells), (3 * h * w) // 5))
    families: list[str] = []
    for i in range(n_objects):
        shape, fam = sample_shape(rng, max_cells, max_h, max_w)
        sh, sw = shape_size(shape)
        if target_colors is None:
            color = obj_colors[int(rng.integers(0, len(obj_colors)))]
        else:
            color = obj_colors[i % len(obj_colors)]
        rows = np.array([r for r, _ in shape])
        cols = np.array([c for _, c in shape])
        for _ in range(30):
            r0 = int(rng.integers(0, h - sh + 1))
            c0 = int(rng.integers(0, w - sw + 1))
            if occ[rows + r0, cols + c0].any():
                continue
            grid[rows + r0, cols + c0] = color
            occ[rows + r0, cols + c0] = True
            families.append(fam)
            break

    if style == "noise":
        free = np.flatnonzero(~occ)
        n_noise = int(round(len(free) * float(rng.uniform(0.03, 0.10))))
        if n_noise > 0:
            picks = rng.choice(free, size=n_noise, replace=False)
            speckle = [obj_colors[int(i)] for i in rng.integers(0, len(obj_colors), size=n_noise)]
            grid.flat[picks] = speckle

    info = {
        "bg_style": style,
        "n_objects": len(families),
        "families": sorted(set(families)),
        "n_colors": int(len(np.unique(grid))),
    }
    return grid, info


_ROTATION_SIZE_BANDS = ((2, 10, 0.30), (11, 20, 0.34), (21, 30, 0.36))  # ≈ flat per-value density


def _rotation_side(rng: np.random.Generator) -> int:
    """One side of a task-10 grid: mixture 50% U[2,12] / 35% U[6,22] / 15% U[12,30]."""
    u = float(rng.random())
    acc = 0.0
    for lo, hi, p in _ROTATION_SIZE_BANDS:
        acc += p
        if u < acc:
            return int(rng.integers(lo, hi + 1))
    lo, hi, _ = _ROTATION_SIZE_BANDS[-1]
    return int(rng.integers(lo, hi + 1))


def _rotation_dims(rng: np.random.Generator) -> tuple[int, int, bool]:
    """(h, w, thin).  5% thin strips 1xY / Yx1 with Y in [2, 30] (never 1x1);
    otherwise H, W independent from the band mixture, and
    H*W <= MAX_ROTATION_AREA."""
    if rng.random() < 0.05:
        y = int(rng.integers(2, 31))
        return ((1, y, True) if rng.random() < 0.5 else (y, 1, True))
    for _ in range(MAX_ATTEMPTS):
        h = _rotation_side(rng)
        w = _rotation_side(rng)
        if h * w <= MAX_ROTATION_AREA:
            return h, w, False
    raise RuntimeError("could not sample a rotation grid size under the area cap")


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture_info(bg_info: dict | None) -> dict:
    """Metadata block for a background-library texture (or the no-texture block)."""
    if bg_info is None:
        return dict(_NO_TEXTURE)
    return {"texture_group": bg_info["group"], "texture_kind": bg_info["kind"],
            "texture_colors": [int(c) for c in bg_info["texture_colors"]],
            "texture_named": False,  # magnify / rotate prompts carry no object rule
            "texture_coverage": float(bg_info["coverage"])}


def _rotation_object_budget(area: int) -> tuple[tuple[int, int], int]:
    """(n_objects_range, max_cells) for a task-10 object scene: the old 1-5 objects
    of <= 60 cells under 150 cells, then a budget that grows with the canvas so
    large grids carry large scenes (up to 16 objects of up to area // 5 cells;
    the floor rises with the area so the realised fill does not halve)."""
    if area < 150:
        return (1, 5), 60
    hi = max(5, min(16, area // 45))
    lo = 2 if area < 400 else 5 if area < 700 else 7
    return (lo, hi), max(60, area // 5)


def _rotation_content_grid(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A task-10 input grid: a thin strip, a pattern-family grid, an object scene
    (object budget scaled to the area) or an object scene over a background-library
    texture; colours drawn near-flat over 2..10.  Returns (grid, info)."""
    h, w, thin = _rotation_dims(rng)
    area = h * w
    tex = None
    if thin:
        vals, style = _thin_content(rng, max(h, w), n_colors=_draw_n_colors(rng, hi=6))
        grid = vals.reshape(h, w)
        info = {"content_style": style, "bg_style": "none", "n_objects": 0, "families": []}
    else:
        u = rng.random()
        n_col = _draw_n_colors(rng, hi=min(10, area))
        if u < 0.30:
            # patterned / sparse / mosaic content from the shared family builder, so
            # rotations are exercised on more than object scenes
            grid, style = _small_pattern_grid(rng, h, w, flat_colors=True)
            info = {"content_style": style, "bg_style": "pattern", "n_objects": 0, "families": []}
        elif u < 0.62 and min(h, w) >= 8:
            # object scene on a uniform ground plus a background-library texture
            rng_objs, cells = _rotation_object_budget(area)
            grid, info = _object_scene(rng, h, w, n_objects_range=rng_objs, max_cells=cells,
                                       target_colors=max(2, n_col - 1), bg_styles={"uniform": 1.0})
            bg = int(np.bincount(grid.ravel(), minlength=NUM_COLORS).argmax())
            occupied = grid != bg
            painted = BG.apply_background(rng, grid, bg, occupied)
            if painted is not None:
                grid, tex = painted
                info["content_style"] = "objects_texture_" + tex["group"]
                info["bg_style"] = "texture"
            else:
                info["content_style"] = "objects_" + info["bg_style"]
        else:
            rng_objs, cells = _rotation_object_budget(area)
            grid, info = _object_scene(rng, h, w, n_objects_range=rng_objs, max_cells=cells,
                                       target_colors=n_col)
            info["content_style"] = "objects_" + info["bg_style"]
    info["n_colors"] = int(len(np.unique(grid)))
    info["thin"] = thin
    info.update(_texture_info(tex))
    return grid, info


def _validate_rotation(inp: np.ndarray, out: np.ndarray, k_cw: int) -> None:
    """Independent check: output is the input rotated k_cw * 90 degrees clockwise."""
    expected = np.rot90(inp, -k_cw)  # np.rot90 is anticlockwise; -k is clockwise
    if not np.array_equal(expected, out):
        raise RuntimeError("rotation output does not match np.rot90(inp, -k)")
    h, w = inp.shape
    want = (w, h) if k_cw in (1, 3) else (h, w)
    if out.shape != want:
        raise RuntimeError(f"rotated shape {out.shape} != {want}")
    if np.array_equal(inp, out):
        raise RuntimeError("rotation left the grid unchanged")
    if k_cw in (1, 3) and np.array_equal(inp, np.rot90(inp, 2)):
        raise RuntimeError("180-symmetric input makes the direction word irrelevant")
    if min(h, w) < 1 or (h == 1 and w == 1) or h * w > MAX_ROTATION_AREA:
        raise RuntimeError(f"bad rotation grid size {inp.shape}")


# --- shared pattern builders for tasks 6 and 10 ------------------------------

_DEMAG_MAX_COLORS = 6
_SMALL_PATTERN_STYLES = {"scene": 0.40, "structured": 0.25, "single_object": 0.15,
                         "dots": 0.10, "noise": 0.10}
_MAGNIFY_PATTERN_STYLES = {"scene": 0.36, "structured": 0.24, "single_object": 0.08,
                           "dots": 0.15, "noise": 0.17}
_STRUCTURED_STYLES = ("stripes", "checker", "concentric", "diagonal", "mirror")


def _structured_pattern(rng: np.random.Generator, h: int, w: int,
                        n_colors: int | None = None) -> tuple[np.ndarray, str]:
    """Stripes / checker / concentric / diagonal / mirror patterns; 2-4 colours by
    default, ``n_colors`` overrides (checker keeps its 2-3 colour habit only by default)."""
    kind = _STRUCTURED_STYLES[int(rng.integers(0, len(_STRUCTURED_STYLES)))]
    rr, cc = np.indices((h, w))
    if kind == "mirror":
        # Random content on a half (or quarter), reflected left-right, up-down or both.
        want = int(rng.integers(2, 5)) if n_colors is None else max(2, min(int(n_colors), 10, max(2, (h * w) // 2)))
        colors = np.array([int(c) for c in pick_colors(rng, want)], dtype=np.uint8)
        mode = ("lr", "ud", "both")[int(rng.integers(0, 3))]
        hh = (h + 1) // 2 if mode in ("ud", "both") else h
        ww = (w + 1) // 2 if mode in ("lr", "both") else w
        part = colors[rng.integers(0, len(colors), size=(hh, ww))]
        if mode in ("ud", "both"):
            part = np.concatenate([part, part[: h - hh][::-1]], axis=0)
        if mode in ("lr", "both"):
            part = np.concatenate([part, part[:, : w - ww][:, ::-1]], axis=1)
        return part, "mirror_" + mode
    explicit = n_colors is not None
    n_colors = int(rng.integers(2, 5)) if not explicit else max(2, min(int(n_colors), 10))
    if explicit:
        # layers each kind can show at period 1; keep only kinds that can show n colours
        layers_at_1 = {"stripes": max(h, w), "checker": h + w - 1, "concentric": (min(h, w) + 1) // 2,
                       "diagonal": h + w - 1}
        able = [k for k in ("stripes", "checker", "concentric", "diagonal") if layers_at_1[k] >= n_colors]
        if kind != "mirror" and kind not in able and able:
            kind = able[int(rng.integers(0, len(able)))]
    colors = np.array([int(c) for c in pick_colors(rng, n_colors)], dtype=np.uint8)
    t = int(rng.integers(1, 3))
    if explicit and kind != "mirror" and (h + w - 1) // t < n_colors:
        t = 1
    if kind == "stripes":
        t = int(rng.integers(1, 4))
        axis = rr if rng.random() < 0.5 else cc
        if explicit:
            if max(h, w) // t < n_colors:
                t = 1
            axis = rr if h >= w else cc  # the longer axis shows more stripes
        layer = axis // t
    elif kind == "checker":
        layer = (rr // t) + (cc // t)
        if not explicit:
            colors = colors[:2] if rng.random() < 0.7 else colors[:3]
    elif kind == "concentric":
        t = 1 if min(h, w) < 6 else t  # thick rings on small grids collapse to one colour
        layer = np.minimum(np.minimum(rr, cc), np.minimum(h - 1 - rr, w - 1 - cc)) // t
    else:  # diagonal bands
        layer = ((rr + cc) if rng.random() < 0.5 else (rr - cc + w)) // t
    return colors[layer % len(colors)], kind


def _family_color_ceiling(style: str, h: int, w: int) -> int:
    """Most colours a small-grid family can show on an h x w canvas (<= 10)."""
    area = h * w
    if style == "single_object":
        return 2
    if style == "scene":
        return min(10, 1 + max(1, min(12, area // 9)))
    if style == "dots":
        return min(10, 1 + max(1, area // 5))
    if style == "structured":
        return min(10, h + w - 1)
    return min(10, area)  # noise


def _small_pattern_grid(rng: np.random.Generator, h: int, w: int,
                        n_colors: int | None = None, flat_colors: bool = False,
                        styles: dict | None = None) -> tuple[np.ndarray, str]:
    """The small (un-magnified) grid of task 55 (and the pattern family of tasks 6
    and 10), drawn from several content families.  ``n_colors`` = None keeps the
    original 2-6 colour behaviour (task 55); a number asks every family for that
    many colours (capped by what the family and the canvas can show);
    ``flat_colors`` draws the family first and then a near-flat colour count over
    what that family can show on this canvas (tasks 6 and 10)."""
    if min(h, w) == 1:
        if flat_colors and n_colors is None:
            n_colors = _draw_n_colors(rng, hi=min(6, max(h, w)))
        vals, style = _thin_content(rng, max(h, w), n_colors=n_colors)
        return vals.reshape(h, w), style
    style = str(_weighted_choice(rng, _SMALL_PATTERN_STYLES if styles is None else styles))
    if flat_colors and n_colors is None:
        n_colors = _draw_n_colors(rng, hi=_family_color_ceiling(style, h, w))
    if style == "scene":
        if n_colors is None:
            grid, _ = _object_scene(rng, h, w, n_objects_range=(2, 5))
        else:
            hi_obj = max(5, min(12, (h * w) // 9))
            grid, _ = _object_scene(rng, h, w, n_objects_range=(2, hi_obj), target_colors=n_colors)
        return grid, style
    if style == "structured":
        return _structured_pattern(rng, h, w, n_colors=n_colors)
    if style == "single_object":
        grid, _ = _object_scene(rng, h, w, n_objects_range=(1, 1), max_object_colors=1,
                                bg_styles={"uniform": 1.0})
        return grid, style
    if style == "dots":
        bg = int(pick_color(rng))
        if n_colors is None:
            n_dot_colors = int(_weighted_choice(rng, {1: 1.0, 2: 2.0, 3: 1.0}))
        else:
            n_dot_colors = max(1, min(int(n_colors) - 1, 9))
        dot_colors = [int(c) for c in pick_colors(rng, n_dot_colors, exclude=[bg])]
        grid = make_grid(h, w, fill=bg)
        n_dots = max(1, int(round(h * w * float(rng.uniform(0.05, 0.20)))))
        if n_colors is not None:
            n_dots = max(n_dots, len(dot_colors))
        picks = rng.choice(h * w, size=min(n_dots, h * w - 1), replace=False)
        if n_colors is None:
            grid.flat[picks] = [dot_colors[int(i)] for i in rng.integers(0, len(dot_colors), size=len(picks))]
        else:
            grid.flat[picks] = [dot_colors[i % len(dot_colors)] for i in range(len(picks))]
        return grid, style
    want = int(rng.integers(2, 5)) if n_colors is None else max(2, min(int(n_colors), 10))
    colors = np.array([int(c) for c in pick_colors(rng, want)], dtype=np.uint8)
    return colors[rng.integers(0, len(colors), size=(h, w))], "noise"


# ---------------------------------------------------------------------------
# Task 6 — Magnify grid
# ---------------------------------------------------------------------------

_MAGNIFY_SCALE_WEIGHTS = {2: 0.42, 3: 0.22, 4: 0.12, 5: 0.08, 6: 0.05, 7: 0.03}
_MAGNIFY_BIG_SCALE_P = 0.08  # rare huge factors (8..14) on tiny inputs


_MAGNIFY_STRIP_P = 0.05  # 1xN / Nx1 inputs: present but bounded (was ~16% by accident)


def _magnify_dims(rng: np.random.Generator, scale: int) -> tuple[int, int]:
    """Input dims for task 6: the output cap makes the feasible box [1, 30//scale]²,
    so the aggregate input heatmap is structurally bottom-left heavy; the small
    scales (whose boxes reach furthest) lean each side upward to spread mass
    toward the ceiling.  Sides are drawn from 2 upward; 1xN strips come from a
    separate 5% draw (never 1x1)."""
    hi = 30 // scale
    if hi < 2:
        return 2, 2
    if rng.random() < _MAGNIFY_STRIP_P:
        n = int(rng.integers(2, hi + 1))
        return (1, n) if rng.random() < 0.5 else (n, 1)

    def side() -> int:
        a = int(rng.integers(2, hi + 1))
        if scale <= 3 and rng.random() < 0.75:
            return max(a, int(rng.integers(2, hi + 1)))
        return a

    return side(), side()


@register("6")
def generate_magnify(rng: np.random.Generator) -> dict:
    """Task 6: magnify by n — every cell becomes an n x n block.

    n in 2..7 (weighted) plus rare 8..14 on tiny inputs; input content spans
    the same families as task 55's small grids (object scenes, structured
    patterns, single objects, sparse dots, mosaics, thin strips) with colours
    drawn near-flat over 2..10 (capped by the input area), plus a
    background-library texture on ~20% of inputs whose short side allows it."""
    for _ in range(MAX_ATTEMPTS):
        if rng.random() < _MAGNIFY_BIG_SCALE_P:
            scale = int(rng.integers(8, 15))
        else:
            scale = int(_weighted_choice(rng, _MAGNIFY_SCALE_WEIGHTS))
        h, w = _magnify_dims(rng, scale)
        grid, style = _small_pattern_grid(rng, h, w, flat_colors=True, styles=_MAGNIFY_PATTERN_STYLES)
        tex = None
        if min(h, w) >= 8 and style in ("scene", "single_object", "dots") and rng.random() < 0.50:
            bg = int(np.bincount(grid.ravel(), minlength=NUM_COLORS).argmax())
            painted = BG.apply_background(rng, grid, bg, grid != bg)
            if painted is not None:
                grid, tex = painted
        if len(np.unique(grid)) < 2:
            continue
        magnified = magnify_grid(grid, scale)
        # independent checks: shape, block-representative equality, monochrome blocks
        assert magnified.shape == (h * scale, w * scale)
        assert np.array_equal(magnified[::scale, ::scale], grid)
        blocks = magnified.reshape(h, scale, w, scale)
        assert (blocks == blocks[:, :1, :, :1]).all()
        meta = {"scale": int(scale), "in_h": int(h), "in_w": int(w),
                "content_style": style, "n_colors": int(len(np.unique(grid)))}
        meta.update(_texture_info(tex))
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(magnified),
            "metadata": meta,
        }
    raise RuntimeError("Task 6: failed to generate a magnify example")


# ---------------------------------------------------------------------------
# Task 10 — Rotate whole grid
# ---------------------------------------------------------------------------

@register("10")
def generate_rotate(rng: np.random.Generator) -> dict:
    """Task 10: Rotate a random grid by 90, 180, or 270 degrees CW or anti-CW.

    Sizes: H and W drawn independently from near-flat bands (30% U[2,10],
    34% U[11,20], 36% U[21,30]; H*W <= 900), plus 5% thin strips 1xY / Yx1
    with Y in [2, 30].  Content: object scenes from ``common.shapes`` on
    uniform / striped / checker / speckled backgrounds with an explicit
    palette (<= 6 colours), patterned/sparse/mosaic grids from the shared
    family builder, object scenes over a background-library texture, or
    coloured runs on the thin strips; colours near-flat over 2..10; the object
    budget scales with the canvas area.
    Every example is rejected unless the rotation changes the grid and, for
    90/270, the grid is not 180-symmetric (so the direction word matters).
    """
    direction = str(rng.choice(["clockwise", "anticlockwise"]))
    user_angle_choice = int(rng.integers(1, 4))  # 1, 2, or 3  -> 90, 180, 270
    user_angle = user_angle_choice * 90

    # Compute the internal CW rotation steps
    if direction == "clockwise":
        k = user_angle_choice          # 1,2,3 CW steps
    else:
        # anticlockwise N degrees == clockwise (360-N) degrees
        k = (4 - user_angle_choice) % 4  # 90 ACW->3 CW, 180 ACW->2 CW, 270 ACW->1 CW

    for _ in range(MAX_ATTEMPTS):
        grid, info = _rotation_content_grid(rng)
        rotated = rotate_grid(grid, k)
        if np.array_equal(grid, rotated):
            continue  # rotation-invariant content (only possible on squares / 180)
        if k in (1, 3) and np.array_equal(grid, rotate_grid(grid, 2)):
            continue  # CW 90 == ACW 90 on a 180-symmetric grid: direction not load-bearing
        break
    else:
        raise RuntimeError("task 10: no rotation-visible grid after MAX_ATTEMPTS")

    _validate_rotation(grid, rotated, k)

    # For a 180-degree rotation the direction is meaningless (CW 180 == ACW 180),
    # so omit it from metadata. The prompt module phrases 180 without a direction;
    # only 90/270 carry a direction word.
    h, w = grid.shape
    metadata = {
        "angle": user_angle,
        "k_cw": k,
        "height": h,
        "width": w,
        "thin": bool(info["thin"]),
        "content_style": info["content_style"],
        "bg_style": info["bg_style"],
        "n_objects": info["n_objects"],
        "families": info["families"],
        "n_colors": info["n_colors"],
        **{k: info[k] for k in _NO_TEXTURE},
    }
    if user_angle != 180:
        metadata["direction"] = direction

    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(rotated),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Task 11 — Reflect whole grid (horizontal / vertical)
# ---------------------------------------------------------------------------

_REFLECT_STRIP_P = 0.05
# Pattern families on canvases of 150+ cells: the single-object style left large grids
# nearly empty (under 3% content), so it is only drawn on small canvases.
_REFLECT_PATTERN_STYLES_BIG = {"scene": 0.40, "structured": 0.30, "dots": 0.15, "noise": 0.15}
_REFLECT_BIG_AREA = 150
_REFLECT_BIG_CONTENT_FLOOR = 0.03     # non-majority cells on a big canvas; sparser content is rebuilt at the same size


def _reflect_content_grid(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, dict]:
    """A task-11 input at a given size: a coloured-run strip when a side is 1, else a
    pattern-family grid (30%), an object scene over a background-library texture
    (32% of the rest, short side >= 8) or a plain object scene, with the object budget
    scaled to the area and colours drawn near-flat over 2..10.  Reflection is
    content-agnostic, so textures are content here: nothing is named.  Returns
    (grid, info)."""
    area = h * w
    tex = None
    if min(h, w) == 1:
        vals, style = _thin_content(rng, max(h, w), n_colors=_draw_n_colors(rng, hi=min(6, max(h, w))))
        grid = vals.reshape(h, w)
        info = {"content_style": style, "bg_style": "none", "n_objects": 0, "families": []}
    else:
        u = rng.random()
        n_col = _draw_n_colors(rng, hi=min(10, area))
        if u < 0.30:
            grid, style = _small_pattern_grid(rng, h, w, flat_colors=True,
                                              styles=_REFLECT_PATTERN_STYLES_BIG if area >= _REFLECT_BIG_AREA else None)
            info = {"content_style": style, "bg_style": "pattern", "n_objects": 0, "families": []}
        elif u < 0.62 and min(h, w) >= 8:
            rng_objs, cells = _rotation_object_budget(area)
            grid, info = _object_scene(rng, h, w, n_objects_range=rng_objs, max_cells=cells,
                                       target_colors=max(2, n_col - 1), bg_styles={"uniform": 1.0})
            bg = int(np.bincount(grid.ravel(), minlength=NUM_COLORS).argmax())
            painted = BG.apply_background(rng, grid, bg, grid != bg)
            if painted is not None:
                grid, tex = painted
                info["content_style"] = "objects_texture_" + tex["group"]
                info["bg_style"] = "texture"
            else:
                info["content_style"] = "objects_" + info["bg_style"]
        else:
            rng_objs, cells = _rotation_object_budget(area)
            grid, info = _object_scene(rng, h, w, n_objects_range=rng_objs, max_cells=cells,
                                       target_colors=n_col)
            info["content_style"] = "objects_" + info["bg_style"]
    info["n_colors"] = int(len(np.unique(grid)))
    info.update(_texture_info(tex))
    return grid, info


def _validate_reflection(inp: np.ndarray, out: np.ndarray, axis: str) -> None:
    """Independent check: the output is the input flipped across the named axis and differs from it."""
    expected = np.flipud(inp) if axis == "horizontal" else np.fliplr(inp)
    if out.shape != inp.shape or not np.array_equal(out, expected):
        raise RuntimeError("task 11: output is not the flipped input")
    if np.array_equal(inp, out):
        raise RuntimeError("task 11: reflection left the grid unchanged")
    h, w = inp.shape
    if not (1 <= h <= 30 and 1 <= w <= 30) or (h == 1 and w == 1):
        raise RuntimeError(f"task 11: bad size {inp.shape}")


@register("11")
def generate_reflect(rng: np.random.Generator) -> dict:
    """Task 11: reflect the whole grid across the horizontal (flip up-down) or vertical
    (flip left-right) axis.

    Sizes: H and W independent and uniform over 2..30, plus 5% strips (1xN or Nx1 along
    the axis that changes them; never 1x1).  Content comes from the shared family
    builder (patterns, object scenes with an area-scaled object budget, scenes over
    background-library textures, coloured-run strips) with colours near-flat over
    2..10.  Content that is invariant under the requested flip, or that leaves a canvas
    of 150+ cells under 3% non-majority cells, is rejected and rebuilt at the same
    size; the generator raises instead of falling back.
    """
    axis = str(rng.choice(["horizontal", "vertical"]))
    for _size_attempt in range(20):
        if rng.random() < _REFLECT_STRIP_P:
            n = int(rng.integers(2, 31))
            # a 1-row strip only changes under a left-right flip, a 1-column strip under up-down
            h, w = (1, n) if axis == "vertical" else (n, 1)
        else:
            h = int(rng.integers(2, 31))
            w = int(rng.integers(2, 31))
        for _content_attempt in range(12):
            grid, info = _reflect_content_grid(rng, h, w)
            reflected = reflect_grid(grid, axis)
            if np.array_equal(grid, reflected):
                continue
            if h * w >= _REFLECT_BIG_AREA and \
                    1.0 - np.bincount(grid.ravel()).max() / grid.size < _REFLECT_BIG_CONTENT_FLOOR:
                continue                       # a big canvas must carry content (>= 3% non-majority cells)
            _validate_reflection(grid, reflected, axis)
            description = "flip up-down" if axis == "horizontal" else "flip left-right"
            metadata = {"axis": axis, "description": description, "height": int(h), "width": int(w),
                        "thin": bool(min(h, w) == 1), **info}
            return {"input": grid_to_list(grid), "output": grid_to_list(reflected), "metadata": metadata}
    raise RuntimeError("task 11: no reflection-visible grid after the retry budget")


# ---------------------------------------------------------------------------
# Task 55 — Demagnify block-constant grid
# ---------------------------------------------------------------------------

@register("55")
def generate_demagnify(rng: np.random.Generator) -> dict:
    from generators.block_scaling_54_55 import generate_compression
    return generate_compression(rng)


# ---------------------------------------------------------------------------
# Task 66 — Reflect across a diagonal
# ---------------------------------------------------------------------------

@register("66")
def generate_reflect_diagonal(rng: np.random.Generator) -> dict:
    """Reflect a complete 5–30 scene across its prompted main or anti-diagonal."""
    from generators.grid_content_66_70 import generate_diagonal
    return generate_diagonal(rng)
