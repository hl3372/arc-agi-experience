"""Task 96 -- extract the interiors of every rectangular frame of color F.

The rule: a rectangular frame of color F is a complete one-cell-thick
rectangular ring of color F, at least 3 by 3, whose interior holds no cell of
color F.  Sort the frames by top row then left column, cut the block strictly
inside each ring, and lay the blocks out in a row (or a column) separated by a
gap of K columns (rows) of the given spacer colour.

The September 2026 rebuild replaces the inline version in
``generators/core2_simple.py``.  What it changes:

* **The canvas is drawn first**, independently over 5..30 per side, and the
  frames are fitted into it.  The old builder summed the frame sizes and added
  noise, so 3.1% of the grids had a side over 30 (illegal) and H and W
  correlated at 0.74.  The only structural floor is that two non-touching 3x3
  frames need a side of at least 7, so the long side starts there.
* **The output never exceeds 30 on a side either**: the interiors are planned
  against the arrangement, so the row (or column) of interiors plus its gaps
  fits an ARC canvas.
* **"Rectangular frame" is defined in the prompt**, and every clause of the
  definition is load-bearing: broken rings of color F (a ring with one to three
  cells missing), solid color F bodies, and complete color F rings with color F
  content inside them (which the "interior contains no cell of color F" clause
  excludes) are all drawn at planned rates.  A reader who takes every color F
  ring, or every color F region, gets a different answer.
* **The spacer colour is a task given, drawn independently of the background**
  and never equal to it.  The old prompt said "with 1 cells of background color
  4 between adjacent interiors", which both named the background (against
  policy) and was ungrammatical.
* **Interiors vary** in both dimensions (1..8 per side, scaled to the canvas)
  and in content (empty, solid fill, a clean body from the shared vocabulary,
  scattered detail, bands).  Ragged rows are the common case, so the padding
  rule ("as tall as the tallest interior, top-aligned, the rest is the spacer
  colour") decides real cells instead of being decoration.
* **The answer is read off the visible grid**, not sliced from the recorded
  coordinates: the generator detects the frames with a literal pure-Python
  implementation of the prompt's definition, checks that the set it finds is
  the set it drew, and the validator re-derives the whole output and raises on
  any disagreement.
* **Backgrounds** come from ``common/backgrounds.py`` (sparse and solid kinds,
  never in color F, never inside a frame) instead of a private painter, and
  nothing about the background is named.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import apply_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list
# module import: the helpers are looked up lazily to avoid a registry-time
# import cycle (a from-import fails when this module is imported first)
import generators.border_objects_90 as _b90
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

MIN_SIDE, MAX_SIDE = 5, 30
LONG_SIDE_FLOOR = 7           # two non-touching 3x3 frames need this much room
MAX_ATTEMPTS = 40
MAX_FRAMES = 8
MAX_INTERIOR = 8
MAX_OUTPUT_SIDE = 30
TEXTURE_MIN_SIDE = 8          # the library's floor for sparse / solid kinds
TEXTURE_P = 0.34              # attempt rate on eligible grids
SOLID_SHARE = 0.35            # of the attempted textures, the solid-group share
FRAME_AREA_SHARE = 0.55       # of the canvas, before moats

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))

_GAP_SIZES = np.array([1, 2, 3])
_GAP_P = np.array([0.45, 0.35, 0.20])
_PALETTE_SIZES = np.arange(2, 11)
_PALETTE_P = np.array([.09, .10, .11, .11, .11, .11, .12, .12, .13])
_CONTENT_KINDS = ("empty", "fill", "body", "detail", "bands", "mixed")
_TWO_COLOR_KINDS = ("detail", "bands", "mixed")
_CONTENT_P = np.array([0.10, 0.17, 0.29, 0.21, 0.12, 0.11])


def _body(*args, **kwargs):
    return _b90._body(*args, **kwargs)


# ---------------------------------------------------------------------------
# Literal reader (pure Python, no plan, no numpy): the prompt's rule verbatim
# ---------------------------------------------------------------------------

def color_groups(grid, color: int) -> list[list[tuple[int, int]]]:
    """Every maximal 4-connected group of ``color`` cells."""
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] == color}
    out: list[list[tuple[int, int]]] = []
    while pending:
        start = min(pending)
        pending.remove(start)
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _ORTH:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        out.append(comp)
    return out


def _bbox(cells) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def _ring_cells(box) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def find_frames(grid, color: int) -> list[tuple[int, int, int, int]]:
    """Every rectangular frame of ``color``, sorted by top row then left column.

    The prompt's definition read literally: a complete one-cell-thick
    rectangular ring (the group is exactly the boundary of its own bounding
    box), at least 3 by 3, whose interior contains no cell of ``color``.
    """
    rows = [list(row) for row in grid]
    out = []
    for comp in color_groups(rows, color):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        if set(comp) != _ring_cells((r0, c0, r1, c1)):
            continue
        if any(rows[r][c] == color for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
            continue
        out.append((r0, c0, r1, c1))
    return sorted(out, key=lambda b: (b[0], b[1]))


def interior_of(grid, box) -> list[list[int]]:
    r0, c0, r1, c1 = box
    return [[int(v) for v in row[c0 + 1:c1]] for row in grid[r0 + 1:r1]]


def arrange(parts, arrangement: str, gap: int, spacer: int) -> list[list[int]]:
    """Lay the interiors out, padding the short ones with the spacer colour."""
    gap = int(gap)
    spacer = int(spacer)
    if arrangement == "horizontal":
        out_h = max(len(p) for p in parts)
        out_w = sum(len(p[0]) for p in parts) + gap * (len(parts) - 1)
        out = [[spacer] * out_w for _ in range(out_h)]
        col = 0
        for part in parts:
            for i, row in enumerate(part):
                for j, value in enumerate(row):
                    out[i][col + j] = int(value)
            col += len(part[0]) + gap
        return out
    out_w = max(len(p[0]) for p in parts)
    out_h = sum(len(p) for p in parts) + gap * (len(parts) - 1)
    out = [[spacer] * out_w for _ in range(out_h)]
    row_at = 0
    for part in parts:
        for i, row in enumerate(part):
            for j, value in enumerate(row):
                out[row_at + i][j] = int(value)
        row_at += len(part) + gap
    return out


def solve(grid, color: int, arrangement: str, gap: int, spacer: int) -> list[list[int]]:
    """Task 96's answer, read literally off the visible grid."""
    boxes = find_frames(grid, color)
    if len(boxes) < 2:
        raise ValueError(f"task 96: {len(boxes)} rectangular frames of the frame colour")
    return arrange([interior_of(grid, box) for box in boxes], arrangement, gap, spacer)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_SIDE_SMALL = np.array([0.10, 0.20, 0.20, 0.16, 0.12, 0.09, 0.07, 0.06])
_SIDE_BIG = np.array([0.04, 0.10, 0.14, 0.16, 0.16, 0.15, 0.13, 0.12])


def _side(rng: np.random.Generator, lim: int, *, big: bool = False) -> int:
    """An interior side in 1..lim; roomy canvases lean to the bigger frames."""
    lim = int(np.clip(lim, 1, MAX_INTERIOR))
    weights = (_SIDE_BIG if big else _SIDE_SMALL)[:lim]
    weights = weights / weights.sum()
    return int(rng.choice(np.arange(1, lim + 1), p=weights))


def _colors(rng: np.random.Generator, n: int, exclude) -> list[int]:
    pool = [c for c in range(NUM_COLORS) if c not in {int(x) for x in exclude}]
    n = max(0, min(int(n), len(pool)))
    if n == 0:
        return []
    return [int(c) for c in rng.choice(pool, size=n, replace=False)]


def _paint_ring(grid: np.ndarray, box, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _interior_content(rng: np.random.Generator, grid: np.ndarray, box, bg: int,
                      primary: Optional[int], secondary: Optional[int],
                      palette: list[int], kind: str) -> str:
    """Draw something inside a frame.  Never the frame colour (the caller's
    palette excludes it), so the interior clause of the definition holds.

    ``primary`` is the colour this drawing owes to the palette plan: it is the
    one used for the body, the fill or the first band, so a planned colour
    always reaches the grid instead of being lost to a random redraw.
    """
    r0, c0, r1, c1 = box
    ih, iw = r1 - r0 - 1, c1 - c0 - 1
    if primary is None or kind == "empty" or ih <= 0 or iw <= 0:
        return "empty"
    colors = [int(primary)] + [int(c) for c in palette]
    view = grid[r0 + 1:r1, c0 + 1:c1]
    pick = lambda: int(colors[int(rng.integers(0, len(colors)))])
    if kind == "fill":
        view[:, :] = int(primary)
        return "fill"
    if kind == "bands":
        first = int(primary)
        second = int(secondary) if secondary is not None else pick()
        if rng.random() < 0.5:
            for i in range(ih):
                view[i, :] = first if i % 2 == 0 else second
        else:
            for j in range(iw):
                view[:, j] = first if j % 2 == 0 else second
        return "bands"
    if kind == "detail":
        drawn = 0
        for r in range(ih):
            for c in range(iw):
                if rng.random() < 0.38:
                    if drawn == 0:
                        view[r, c] = int(primary)
                    elif drawn == 1 and secondary is not None:
                        view[r, c] = int(secondary)
                    else:
                        view[r, c] = pick()
                    drawn += 1
        if not drawn:
            view[int(rng.integers(0, ih)), int(rng.integers(0, iw))] = int(primary)
        return "detail"
    built = _body(rng, ih, iw, min_cells=max(1, min(2, ih * iw)))
    if built is None:
        view[:, :] = int(primary)
        return "fill"
    mask = built[0]
    r = int(rng.integers(0, ih - mask.shape[0] + 1))
    c = int(rng.integers(0, iw - mask.shape[1] + 1))
    block = view[r:r + mask.shape[0], c:c + mask.shape[1]]
    block[mask] = int(primary)
    if kind == "mixed":
        for index in range(int(rng.integers(1, 4))):
            rr = int(rng.integers(0, ih))
            cc = int(rng.integers(0, iw))
            if view[rr, cc] == bg:
                view[rr, cc] = int(secondary) if (index == 0 and secondary is not None) else pick()
        return "mixed"
    return "body"


def _place_mask(rng: np.random.Generator, mask: np.ndarray, free: np.ndarray):
    """A free seat for ``mask``; returns the top-left corner or None."""
    ok = placement_map(mask, free)
    if ok is None or not ok.any():
        return None
    spots = np.argwhere(ok)
    r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
    return r, c


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(rng: np.random.Generator, grid: np.ndarray, bg: int, occupied: np.ndarray,
             *, exclude, probability: float) -> tuple[np.ndarray, dict]:
    """A sparse or solid library texture on the free cells outside every frame.

    The frame colour is excluded, so no texture cell can join, break or imitate
    a ring; the frames and their interiors are occupied, so nothing is painted
    inside one.  Nothing about it is named in the prompt.
    """
    h, w = grid.shape
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= probability:
        return grid, dict(_NO_TEXTURE)
    group = "solid" if rng.random() < SOLID_SHARE else "sparse"
    painted = apply_background(rng, grid, bg, occupied, group=group,
                               exclude=[int(c) for c in exclude])
    if painted is None:
        return grid, dict(_NO_TEXTURE)
    display, info = painted
    return display, {
        "texture_group": info["group"],
        "texture_kind": info["kind"],
        "texture_colors": [int(c) for c in info["texture_colors"]],
        "texture_named": False,
        "texture_coverage": round(float(info["coverage"]), 4),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _frame_cap(h: int, w: int) -> int:
    """How many frames the canvas carries: a 3x3 frame plus its moat is 16 cells."""
    return int(np.clip((h * w) // 30, 2, MAX_FRAMES))


def _plan(rng: np.random.Generator) -> dict:
    h = w = 0
    while max(h, w) < LONG_SIDE_FLOOR:
        h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
        w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    arrangement = "horizontal" if rng.random() < 0.5 else "vertical"
    gap = int(rng.choice(_GAP_SIZES, p=_GAP_P))
    n_frames = int(rng.integers(2, _frame_cap(h, w) + 1))
    ragged = bool(rng.random() < 0.65)
    lim_h = int(np.clip(h - 2, 1, MAX_INTERIOR))
    lim_w = int(np.clip(w - 2, 1, MAX_INTERIOR))
    big = area >= 400
    sizes = [[_side(rng, lim_h, big=big), _side(rng, lim_w, big=big)]
             for _ in range(n_frames)]
    if not ragged:
        if arrangement == "horizontal":
            for size in sizes:
                size[0] = sizes[0][0]
        else:
            for size in sizes:
                size[1] = sizes[0][1]
    axis = 1 if arrangement == "horizontal" else 0
    # the output is an ARC canvas too: the row (column) of interiors plus its
    # gaps never passes 30
    guard = 0
    while sum(s[axis] for s in sizes) + gap * (len(sizes) - 1) > MAX_OUTPUT_SIDE and guard < 200:
        guard += 1
        biggest = max(range(len(sizes)), key=lambda i: sizes[i][axis])
        if sizes[biggest][axis] > 1:
            sizes[biggest][axis] -= 1
        elif len(sizes) > 2:
            sizes.pop()
        else:
            break
    # ... and the frames have to fit the canvas with room for their moats
    guard = 0
    while (sum((s[0] + 2) * (s[1] + 2) for s in sizes) > FRAME_AREA_SHARE * area
           and guard < 200):
        guard += 1
        biggest = max(range(len(sizes)), key=lambda i: (sizes[i][0] + 2) * (sizes[i][1] + 2))
        if max(sizes[biggest]) > 1:
            axis_big = 0 if sizes[biggest][0] >= sizes[biggest][1] else 1
            sizes[biggest][axis_big] -= 1
        elif len(sizes) > 2:
            sizes.pop()
        else:
            break
    n_frames = len(sizes)

    # distractors: every clause of the frame definition gets something that
    # fails it, plus ordinary bodies in other colours
    broken = int(rng.integers(1, 3)) if rng.random() < 0.48 else 0
    spoiled = 1 if rng.random() < 0.40 else 0
    solid_f = int(rng.integers(1, 3)) if rng.random() < 0.52 else 0
    other_frames = int(rng.integers(1, 3)) if rng.random() < 0.45 else 0
    n_bodies = int(rng.integers(0, int(np.clip(area // 110, 1, 6)) + 1))

    bg = int(rng.integers(0, NUM_COLORS))
    frame_color = int(rng.choice([c for c in range(NUM_COLORS) if c != bg]))
    spacer_color = int(rng.choice([c for c in range(NUM_COLORS)
                                   if c not in (bg, frame_color)]))
    palette = int(rng.choice(_PALETTE_SIZES, p=_PALETTE_P))
    texture = bool(min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P)
    kinds = [str(rng.choice(_CONTENT_KINDS, p=_CONTENT_P)) for _ in sizes]
    # the palette is filled with real drawings, never speckle: one colour per
    # slot that actually paints something (a filled interior, a loose body, a
    # frame in another colour), so a planned colour is never stranded
    slots = (sum(2 if k in _TWO_COLOR_KINDS else 1 for k in kinds if k != "empty")
             + n_bodies + other_frames)
    n_other = int(np.clip(palette - 2 - (1 if texture else 0), 0, max(0, slots)))
    if palette > 2 and n_other == 0 and slots:
        n_other = 1
    content_colors = _colors(rng, n_other, [bg, frame_color])
    if content_colors and all(k == "empty" for k in kinds):
        kinds[int(rng.integers(0, len(kinds)))] = "body"
    if not content_colors:
        kinds = ["empty"] * len(kinds)
    return {
        "h": h, "w": w, "area": area, "arrangement": arrangement, "gap": gap,
        "sizes": sizes, "n_frames": n_frames, "ragged": ragged,
        "bg": bg, "frame_color": frame_color, "spacer_color": spacer_color,
        "content_colors": content_colors, "palette_target": palette,
        "plan_texture": texture, "broken": broken, "spoiled": spoiled,
        "solid_f": solid_f, "other_frames": other_frames, "n_bodies": n_bodies,
        "kinds": kinds,
    }


def _build(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    h, w, bg = plan["h"], plan["w"], plan["bg"]
    frame_color = plan["frame_color"]
    grid = np.full((h, w), bg, dtype=np.int16)
    occ = np.zeros((h, w), dtype=bool)        # frames, interiors, distractors
    f_occ = np.zeros((h, w), dtype=bool)      # everything drawn in the frame colour
    boxes: list[tuple[int, int, int, int]] = []
    kinds: list[str] = []
    sizes: list[list[int]] = []
    # every planned colour is owed a drawing of its own before any colour repeats
    pending = list(plan["content_colors"])

    def next_color() -> Optional[int]:
        if pending:
            return int(pending.pop(0))
        if plan["content_colors"]:
            return int(plan["content_colors"][int(rng.integers(0, len(plan["content_colors"])))])
        return None

    for index, (ih, iw) in enumerate(plan["sizes"]):
        seat = None
        for dh, dw in ((ih, iw), (max(1, ih - 1), max(1, iw - 1)), (1, 1)):
            mask = np.ones((dh + 2, dw + 2), dtype=bool)
            seat = _place_mask(rng, mask, ~halo(occ))
            if seat is not None:
                ih, iw = dh, dw
                break
        if seat is None:
            return None
        r0, c0 = seat
        box = (r0, c0, r0 + ih + 1, c0 + iw + 1)
        _paint_ring(grid, box, frame_color)
        want = plan["kinds"][index]
        primary = next_color() if want != "empty" else None
        secondary = next_color() if (primary is not None and want in _TWO_COLOR_KINDS) else None
        kind = _interior_content(rng, grid, box, bg, primary, secondary,
                                 plan["content_colors"], want)
        occ[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
        f_occ[box[0]:box[2] + 1, box[1]:box[3] + 1] |= (grid[box[0]:box[2] + 1,
                                                             box[1]:box[3] + 1] == frame_color)
        boxes.append(box)
        kinds.append(kind)
        sizes.append([ih, iw])

    # --- distractors that fail one clause of the definition -----------------
    broken = 0
    for _ in range(plan["broken"]):
        dh = int(rng.integers(3, 7))
        dw = int(rng.integers(3, 7))
        mask = np.ones((dh, dw), dtype=bool)
        seat = _place_mask(rng, mask, ~halo(occ))
        if seat is None:
            continue
        r0, c0 = seat
        ring = sorted(_ring_cells((0, 0, dh - 1, dw - 1)))
        gone = rng.choice(len(ring), size=int(rng.integers(1, 4)), replace=False)
        drawn = 0
        for i, (r, c) in enumerate(ring):
            if i in set(int(g) for g in gone):
                continue
            grid[r0 + r, c0 + c] = frame_color
            drawn += 1
        occ[r0:r0 + dh, c0:c0 + dw] = True
        f_occ[r0:r0 + dh, c0:c0 + dw] |= (grid[r0:r0 + dh, c0:c0 + dw] == frame_color)
        broken += 1 if drawn else 0

    spoiled = 0
    for _ in range(plan["spoiled"]):
        seat = None
        for _try in range(4):
            dh = int(rng.integers(5, 9))
            dw = int(rng.integers(5, 9))
            mask = np.ones((dh, dw), dtype=bool)
            seat = _place_mask(rng, mask, ~halo(occ))
            if seat is not None:
                break
        if seat is None:
            continue
        r0, c0 = seat
        box = (r0, c0, r0 + dh - 1, c0 + dw - 1)
        _paint_ring(grid, box, frame_color)
        # a color F body strictly inside, one cell clear of the ring, so the
        # ring stays a ring and only the interior clause rejects it
        inner_h, inner_w = dh - 4, dw - 4
        built = _body(rng, max(1, inner_h), max(1, inner_w), min_cells=1)
        block = built[0] if built is not None else np.ones((1, 1), dtype=bool)
        rr = r0 + 2 + int(rng.integers(0, max(1, inner_h - block.shape[0] + 1)))
        cc = c0 + 2 + int(rng.integers(0, max(1, inner_w - block.shape[1] + 1)))
        target = grid[rr:rr + block.shape[0], cc:cc + block.shape[1]]
        target[block[:target.shape[0], :target.shape[1]]] = frame_color
        occ[r0:r0 + dh, c0:c0 + dw] = True
        f_occ[r0:r0 + dh, c0:c0 + dw] |= (grid[r0:r0 + dh, c0:c0 + dw] == frame_color)
        spoiled += 1

    solid_f = 0
    for _ in range(plan["solid_f"]):
        built = _body(rng, int(rng.integers(2, 6)), int(rng.integers(2, 6)), min_cells=3)
        if built is None:
            continue
        mask = built[0]
        seat = _place_mask(rng, mask, ~halo(occ))
        if seat is None:
            continue
        r0, c0 = seat
        cells = np.zeros((h, w), dtype=bool)
        cells[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] = mask
        grid[cells] = frame_color
        occ |= cells
        f_occ |= cells
        solid_f += 1

    # --- ordinary distractors in other colours ------------------------------
    palette = [c for c in plan["content_colors"] if c != frame_color]
    other_frames = 0
    for _ in range(plan["other_frames"]):
        if not palette:
            break
        seat = None
        for _try in range(3):
            dh = int(rng.integers(3, 8))
            dw = int(rng.integers(3, 8))
            mask = np.ones((dh, dw), dtype=bool)
            seat = _place_mask(rng, mask, ~halo(f_occ) & ~occ)
            if seat is not None:
                break
        if seat is None:
            continue
        r0, c0 = seat
        color = next_color() or int(palette[int(rng.integers(0, len(palette)))])
        box = (r0, c0, r0 + dh - 1, c0 + dw - 1)
        thickness = 2 if (min(dh, dw) >= 6 and rng.random() < 0.35) else 1
        for t in range(thickness):
            _paint_ring(grid, (box[0] + t, box[1] + t, box[2] - t, box[3] - t), color)
        if min(dh, dw) > 2 * thickness and rng.random() < 0.35:
            fill = int(palette[int(rng.integers(0, len(palette)))])
            grid[r0 + thickness:r0 + dh - thickness, c0 + thickness:c0 + dw - thickness] = fill
        occ[r0:r0 + dh, c0:c0 + dw] = True
        other_frames += 1

    bodies = 0
    for _ in range(plan["n_bodies"]):
        if not palette:
            break
        built = _body(rng, int(rng.integers(2, 7)), int(rng.integers(2, 7)), min_cells=3)
        if built is None:
            continue
        mask = built[0]
        # ordinary bodies may touch each other, never a color F drawing
        seat = _place_mask(rng, mask, ~halo(f_occ) & ~occ)
        if seat is None:
            continue
        r0, c0 = seat
        cells = np.zeros((h, w), dtype=bool)
        cells[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] = mask
        grid[cells] = next_color() or int(palette[int(rng.integers(0, len(palette)))])
        occ |= cells
        bodies += 1

    return {"grid": grid, "occ": occ, "boxes": boxes, "kinds": kinds, "sizes": sizes,
            "broken": broken, "spoiled": spoiled, "solid_f": solid_f,
            "other_frames": other_frames, "bodies": bodies}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_96(example: dict) -> None:
    """Re-read the emitted grid, re-detect the frames and re-derive the output."""
    grid, meta = example["input"], example["metadata"]
    color = int(meta["frame_color"])
    spacer = int(meta["spacer_color"])
    boxes = find_frames(grid, color)
    if len(boxes) != meta["n_frames"]:
        raise ValueError("task 96: the frame count disagrees with the metadata")
    if [list(b) for b in boxes] != [list(b) for b in meta["frame_boxes"]]:
        raise ValueError("task 96: the detected frames are not the recorded ones")
    if len({(b[0], b[1]) for b in boxes}) != len(boxes):
        raise ValueError("task 96: two frames share a top row and a left column")
    if spacer == meta["background_color"] or spacer == color:
        raise ValueError("task 96: the spacer colour is the background or the frame colour")
    expected = solve(grid, color, meta["arrangement"], int(meta["gap"]), spacer)
    if expected != example["output"]:
        raise ValueError("task 96: the arrangement disagrees with the literal reading")
    if len(expected) != meta["output_h"] or len(expected[0]) != meta["output_w"]:
        raise ValueError("task 96: the output size disagrees with the metadata")
    if meta["output_h"] > MAX_OUTPUT_SIDE or meta["output_w"] > MAX_OUTPUT_SIDE:
        raise ValueError("task 96: the output is larger than an ARC canvas")
    for row in grid:
        for value in row:
            if not 0 <= int(value) <= 9:
                raise ValueError("task 96: a cell is outside the palette")
    if meta["texture_colors"]:
        if color in meta["texture_colors"]:
            raise ValueError("task 96: a texture colour is the frame colour")
        arr = np.asarray(grid)
        if meta["texture_group"] == "sparse":
            tex = np.isin(arr, meta["texture_colors"])
            if not bool(isolated_cell_mask(arr)[tex].all()):
                raise ValueError("task 96: a sparse texture cell is not isolated")


# ---------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = _plan(rng)
        scene = None
        for _ in range(3):
            scene = _build(rng, plan)
            if scene is not None:
                break
        if scene is None:
            continue
        grid, occ = scene["grid"], scene["occ"]
        display, tex = _texture(rng, grid, plan["bg"], occ,
                                exclude=[plan["frame_color"], plan["spacer_color"], *plan["content_colors"]],
                                probability=1.0 if plan["plan_texture"] else 0.0)
        rows = grid_to_list(display.astype(np.int16))

        # the answer comes from the picture, not from the plan: the frames are
        # detected again with the prompt's definition and must be the ones drawn
        boxes = find_frames(rows, plan["frame_color"])
        if len(boxes) < 2:
            continue
        if [list(b) for b in boxes] != [list(b) for b in sorted(scene["boxes"])]:
            continue
        if len({(b[0], b[1]) for b in boxes}) != len(boxes):
            continue
        parts = [interior_of(rows, box) for box in boxes]
        out = arrange(parts, plan["arrangement"], plan["gap"], plan["spacer_color"])
        if len(out) > MAX_OUTPUT_SIDE or len(out[0]) > MAX_OUTPUT_SIDE:
            continue

        interior_sizes = [[len(p), len(p[0])] for p in parts]
        ragged = (len({s[0] for s in interior_sizes}) > 1 if plan["arrangement"] == "horizontal"
                  else len({s[1] for s in interior_sizes}) > 1)
        padded = sum(1 for row in out for v in row if v == plan["spacer_color"])
        gaps = (len(out) * plan["gap"] * (len(parts) - 1) if plan["arrangement"] == "horizontal"
                else len(out[0]) * plan["gap"] * (len(parts) - 1))
        present = sorted({int(v) for v in np.unique(display)})
        out_present = sorted({int(v) for row in out for v in row})
        uniform = sum(1 for p in parts if len({v for row in p for v in row}) == 1)
        blank = sum(1 for p in parts
                    if {v for row in p for v in row} == {int(plan["bg"])})
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"],
            "n_colors": len(present), "n_colors_output": len(out_present),
            "frame_color": int(plan["frame_color"]),
            "spacer_color": int(plan["spacer_color"]),
            "background_color": int(plan["bg"]),
            "arrangement": plan["arrangement"], "gap": int(plan["gap"]),
            "n_frames": len(boxes),
            "frame_boxes": [list(b) for b in boxes],
            "interior_sizes": interior_sizes,
            "interior_kinds": scene["kinds"],
            "uniform_interiors": uniform,
            "blank_interiors": blank,
            "ragged": bool(ragged),
            "padding_cells": padded - gaps,
            "output_h": len(out), "output_w": len(out[0]),
            "n_broken_rings": scene["broken"],
            "n_spoiled_rings": scene["spoiled"],
            "n_solid_frame_color_bodies": scene["solid_f"],
            "n_other_color_frames": scene["other_frames"],
            "n_bodies": scene["bodies"],
            "spacer_in_input": bool(plan["spacer_color"] in present),
            "occupancy": round(float(occ.sum()) / plan["area"], 4),
            "object_cells": int(occ.sum()),
            "palette_target": plan["palette_target"],
            "frame_rule": ("a complete one-cell-thick rectangular ring of the frame color, at least "
                           "3 by 3, whose interior contains no cell of the frame color"),
            "ordering": "by top row ascending, then by left column ascending",
            "extraction_rule": "every cell strictly inside the ring, copied exactly",
            "padding_rule": ("the output is as tall as the tallest interior (as wide as the widest) "
                             "and every cell no interior covers is the spacer color"),
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows, "output": out, "metadata": meta}
        validate_96(example)
        return example
    raise RuntimeError(f"task 96: no valid scene after {MAX_ATTEMPTS} attempts")


@register("96")
def generate_extract_frame_interiors(rng: np.random.Generator) -> dict:
    """Task 96: extract and arrange the interiors of every color F frame."""
    return _example(rng)
