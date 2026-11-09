"""Task 95 -- outline the tight bounding box of every color X object.

The rule: find the color X objects, and draw the one-cell ring of cells
immediately outside each object's tight bounding box in the reserved outline
colour, over whatever is already there.  Everything else stays as it is.

The September 2026 rebuild replaces the inline version in
``generators/core2_simple.py``.  What it changes:

* **Sizes**: independent 5..30 sides, flat (the old builder drew 10..19 on both
  axes, so half the envelope never occurred).  The structural floor is the
  canvas itself: a two-cell body plus its ring fits on a 5x5, and a ring that
  reaches past the border is simply clipped.
* **Bodies** come from the shared vocabulary through ``border_objects_90``'s
  clean-body builder (rectangles, bars, frames, thick letters, staircases,
  blobs assembled from rectangles) instead of the host file's random walks,
  which grew 2.3 one-cell hairs per figure.  Sizes scale with the canvas.
* **Backgrounds** come from ``common/backgrounds.py`` (sparse and solid kinds)
  instead of a private full-length-stripe painter that left the modal colour
  ambiguous on 11% of the grids.  Nothing about the background is named.
* **The outline colour is reserved**: it never occurs in the input (it used to
  be there on a quarter of the grids, where "draw it in color H" was already
  half done).  Input palettes therefore reach 9 and outputs reach 10.
* **The connectivity is drawn per example**, after the whole scene exists, and
  is load-bearing: a corner-joined target group is ONE object with one ring
  under 8-connectivity and two or three objects with two or three rings under
  4-connectivity.  No input feature can predict the coin.
* **The isolated-cell clause is load-bearing**: lone color X cells, kept clear
  of every other color X cell, are background texture and get no ring.  A
  reader who outlines them draws extra rings.
* **The three ring policies are made true and stated where they can bite**:
  - a ring that would leave the grid is CLIPPED (the prompt says so, and a
    planned share of the targets touches the border so the clause bites);
  - a ring may cross another object or a texture cell, and the outline colour
    is written over it (the prompt says so; it happens on a planned share);
  - two rings may share cells; both write the same colour, so the union is the
    only reading and nothing has to be said about it.
  Distinct target figures are seated outside each other's expanded boxes, so
  one figure's ring never reaches another figure.  The one place a ring does
  take an object's cells is the 4-reading of a corner-joined figure, where the
  pieces are separate objects that touch at a corner: each piece's ring covers
  the corner cell of its neighbour.  That is why the prompt says the objects
  are all read off the input BEFORE any outline is drawn, and that the outline
  is written over whatever is there, another object's cells included.
* **The validator** re-reads the emitted grid with a literal pure-Python
  implementation of the prompt (groups, boxes, clipped rings) and raises on any
  disagreement.  The old builder had no check at all.
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
MAX_ATTEMPTS = 40
TEXTURE_MIN_SIDE = 8          # the library's floor for sparse / solid kinds
TEXTURE_P = 0.34              # attempt rate on eligible grids
SOLID_SHARE = 0.40            # of the attempted textures, the solid-group share
MAX_TARGETS = 4

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# The outline colour is reserved, so the input palette stops at 9; near-flat
# over 2..9 with a lean to the top (capacity truncates the tail).
_PALETTE_SIZES = np.arange(2, 10)
_PALETTE_P = np.array([.10, .11, .12, .12, .13, .13, .14, .15])


def _body(*args, **kwargs):
    return _b90._body(*args, **kwargs)


def _corner_join(*args, **kwargs):
    return _b90._corner_join(*args, **kwargs)


# ---------------------------------------------------------------------------
# Literal reader (pure Python, no plan, no numpy): the prompt's rule verbatim
# ---------------------------------------------------------------------------

def color_groups(grid, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    """Every maximal same-colour group of ``color`` under the connectivity."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] == color}
    out: list[list[tuple[int, int]]] = []
    while pending:
        start = min(pending)
        pending.remove(start)
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        out.append(comp)
    return out


def _is_isolated(rows, r: int, c: int) -> bool:
    """The prompt's "isolated single cell": no 8-neighbour of the same colour."""
    h, w = len(rows), len(rows[0])
    color = rows[r][c]
    for dr, dc in _ORTH + _DIAG:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and rows[rr][cc] == color:
            return False
    return True


def target_objects(grid, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    """The color X objects: groups of at least two cells.

    A one-cell group is background texture -- but only when it really is an
    isolated single cell.  A lone cell that still has a same-colour diagonal
    neighbour would be neither an object (one cell) nor texture (not isolated)
    under 4-connectivity, so the generator never draws one and this raises if
    one ever appears.
    """
    rows = [list(row) for row in grid]
    out = []
    for comp in color_groups(rows, color, connectivity):
        if len(comp) >= 2:
            out.append(comp)
            continue
        r, c = comp[0]
        if not _is_isolated(rows, r, c):
            raise ValueError("task 95: a one-cell group is neither an object nor an isolated cell")
    return out


def bbox(cells) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def outline_cells(box, h: int, w: int) -> list[tuple[int, int]]:
    """The one-cell ring just outside ``box``, with cells off the grid dropped."""
    r0, c0, r1, c1 = box
    out = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1):
                if 0 <= r < h and 0 <= c < w:
                    out.append((r, c))
    return out


def full_outline(box) -> list[tuple[int, int]]:
    """The ring before clipping (used to count what the grid edge removed)."""
    r0, c0, r1, c1 = box
    return [(r, c)
            for r in range(r0 - 1, r1 + 2)
            for c in range(c0 - 1, c1 + 2)
            if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)]


def solve(grid, color: int, connectivity: int, outline_color: int) -> list[list[int]]:
    """Task 95's answer, read literally off the visible grid."""
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    for comp in target_objects(rows, color, connectivity):
        for r, c in outline_cells(bbox(comp), h, w):
            rows[r][c] = int(outline_color)
    return rows


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _sized_body(rng: np.random.Generator, cells: float, lim_h: int, lim_w: int,
                *, min_cells: int = 3) -> Optional[tuple[np.ndarray, str]]:
    """A clean body of about ``cells`` cells whose box fits ``lim_h x lim_w``."""
    lim_h, lim_w = max(1, int(lim_h)), max(1, int(lim_w))
    floor = max(1, int(min_cells))
    if lim_h * lim_w < floor:
        floor = lim_h * lim_w
    side = float(np.sqrt(max(1.0, float(cells))))
    for _ in range(14):
        bh = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_h))
        bw = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_w))
        for _ in range(10):
            if bh * bw >= floor:
                break
            if bh < lim_h and (bh <= bw or bw >= lim_w):
                bh += 1
            elif bw < lim_w:
                bw += 1
            else:
                break
        built = _body(rng, bh, bw, min_cells=floor)
        if built is None:
            continue
        mask, fam = built
        return np.ascontiguousarray(mask), fam
    return None


def _figure(rng: np.random.Generator, budget: float, pieces: int,
            lim_h: int, lim_w: int) -> Optional[tuple[np.ndarray, list[str]]]:
    """One target figure: a single clean body, or pieces joined at corners."""
    if pieces == 1:
        built = _sized_body(rng, budget, lim_h, lim_w, min_cells=3)
        if built is None:
            return None
        return built[0], [built[1]]
    masks, fams = [], []
    per = max(2.0, budget / pieces)
    for _ in range(pieces):
        built = _sized_body(rng, per, max(1, lim_h - 1), max(1, lim_w - 1), min_cells=2)
        if built is None or int(built[0].sum()) < 2:
            return None
        masks.append(built[0])
        fams.append(built[1])
    joined = _corner_join(rng, masks)
    if joined is None:
        return None
    return np.ascontiguousarray(joined[0]), fams


def _expanded_box(box, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1))


def _place_target(
    rng: np.random.Generator,
    grid: np.ndarray,
    occ: np.ndarray,
    target_occ: np.ndarray,
    expanded: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    edge: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Seat a target figure so that no ring can ever cover another target.

    Two constraints, both exact: the new cells stay out of every earlier
    target's expanded box, and the new figure's own box stays one cell clear of
    every earlier target's cells (a box avoids a cell set exactly when the
    dilated cell set avoids the box).  Expanded boxes may still overlap, which
    is what lets two rings share cells.
    """
    h, w = grid.shape
    fh, fw = mask.shape
    if fh > h or fw > w:
        return None
    ok = placement_map(mask, ~occ & ~expanded)
    if ok is None:
        return None
    box_ok = placement_map(np.ones((fh, fw), dtype=bool), ~halo(target_occ))
    if box_ok is None:
        return None
    ok = ok & box_ok
    if edge is not None:
        keep = np.zeros(ok.shape, dtype=bool)
        if edge == "top":
            keep[0, :] = True
        elif edge == "bottom":
            keep[ok.shape[0] - 1, :] = True
        elif edge == "left":
            keep[:, 0] = True
        else:
            keep[:, ok.shape[1] - 1] = True
        ok = ok & keep
    spots = np.argwhere(ok)
    if not len(spots):
        return None
    r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
    cells = np.zeros(grid.shape, dtype=bool)
    cells[r:r + fh, c:c + fw] = mask
    grid[cells] = color
    occ |= cells
    return cells


def _place_body(
    rng: np.random.Generator,
    grid: np.ndarray,
    occ: np.ndarray,
    mask: np.ndarray,
    color: int,
    free: np.ndarray,
    *,
    prefer: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Drop a distractor body on a free spot, preferring spots that cross ``prefer``."""
    if mask.shape[0] > grid.shape[0] or mask.shape[1] > grid.shape[1]:
        return None
    ok = placement_map(mask, free)
    if ok is None or not ok.any():
        return None
    if prefer is not None and prefer.any():
        misses = placement_map(mask, ~prefer)
        want = ok & ~misses
        if want.any():
            ok = want
    spots = np.argwhere(ok)
    if not len(spots):
        return None
    r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
    fh, fw = mask.shape
    cells = np.zeros(grid.shape, dtype=bool)
    cells[r:r + fh, c:c + fw] = mask
    grid[cells] = color
    occ |= cells
    return cells


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(rng: np.random.Generator, grid: np.ndarray, bg: int, occupied: np.ndarray,
             *, exclude, probability: float) -> tuple[np.ndarray, dict]:
    """A sparse or solid library texture on the free cells.

    The objects are defined by COLOUR, so a solid region of another colour can
    never become a color X object; the outline colour is reserved, so a ring is
    always legible on top of whatever the texture painted.  Nothing about it is
    named in the prompt.
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


def _colors(rng: np.random.Generator, n: int, exclude) -> list[int]:
    pool = [c for c in range(NUM_COLORS) if c not in {int(x) for x in exclude}]
    n = max(0, min(int(n), len(pool)))
    if n == 0:
        return []
    return [int(c) for c in rng.choice(pool, size=n, replace=False)]


def _assign(rng: np.random.Generator, colors: list[int], n: int) -> list[int]:
    """One colour per planned body, every planned colour used at least once."""
    if not colors or n <= 0:
        return []
    out = list(colors)[:n]
    while len(out) < n:
        out.append(int(colors[int(rng.integers(0, len(colors)))]))
    return [int(out[int(i)]) for i in rng.permutation(len(out))]


# ---------------------------------------------------------------------------
# The plan (everything except the connectivity)
# ---------------------------------------------------------------------------

def _target_cap(h: int, w: int) -> int:
    """How many target FIGURES the canvas carries.

    Footprint-derived: a figure plus the expanded box that keeps its ring off
    every other figure costs about thirty cells.  The object count under
    4-connectivity can exceed it, because a corner-joined figure is two or
    three objects there.
    """
    return int(np.clip((h * w) // 30, 1, MAX_TARGETS))


def _plan(rng: np.random.Generator) -> dict:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    # scenes scale with the canvas: the ink share is lifted on roomy grids so a
    # 30x30 is not two small bodies in a sea of background
    density = float(rng.uniform(0.13, 0.33)) * (1.18 if area >= 400 else 1.0)
    ink = density * area
    cap = _target_cap(h, w)
    n_targets = int(rng.integers(1, cap + 1))
    # a corner-joined figure is the drawing on which the two connectivities
    # give different rings; it is planned on about half the grids
    corner = bool(rng.random() < 0.50)
    pieces = []
    for i in range(n_targets):
        if corner and (i == 0 or rng.random() < 0.35):
            pieces.append(3 if rng.random() < 0.22 else 2)
        else:
            pieces.append(1)
    target_budget = float(np.clip(ink * float(rng.uniform(0.25, 0.55)), 3.0, 140.0))
    # a target whose box touches the border loses part of its ring: the clip
    # clause has to bite somewhere
    edge = bool(rng.random() < 0.35)
    # distractors crossing the ring band make "over whatever is there" real
    cross = 0 if rng.random() < 0.25 else int(rng.integers(1, 3))
    # lone color X cells: background texture under the isolated-cell clause
    strays = int(rng.integers(1, 4)) if rng.random() < 0.35 else 0
    budget = int(np.clip(area // 22, 1, 14))
    n_distractors = 0 if rng.random() < 0.08 else int(rng.integers(1, budget + 1))
    palette = int(rng.choice(_PALETTE_SIZES, p=_PALETTE_P))
    texture = bool(min(h, w) >= TEXTURE_MIN_SIDE and rng.random() < TEXTURE_P)
    n_colors_objects = int(np.clip(palette - 2 - (1 if texture else 0), 0, budget))
    if n_distractors < n_colors_objects:
        n_distractors = n_colors_objects
    bg = int(rng.integers(0, NUM_COLORS))
    target_color = int(rng.choice([c for c in range(NUM_COLORS) if c != bg]))
    outline_color = int(rng.choice([c for c in range(NUM_COLORS)
                                    if c not in (bg, target_color)]))
    distractor_colors = _colors(rng, n_colors_objects, [bg, target_color, outline_color])
    return {
        "h": h, "w": w, "area": area, "density": density,
        "cap": cap, "n_targets": n_targets, "pieces": pieces, "corner": corner,
        "target_budget": target_budget, "edge": edge,
        "edge_side": ["top", "bottom", "left", "right"][int(rng.integers(0, 4))],
        "cross": cross, "strays": strays,
        "n_distractors": n_distractors, "bg": bg,
        "target_color": target_color, "outline_color": outline_color,
        "distractor_colors": distractor_colors,
        "colors": _assign(rng, distractor_colors, n_distractors),
        "palette_target": palette, "plan_texture": texture,
        "per_distractor": float(np.clip((ink - target_budget) / max(1, n_distractors), 8.0, 110.0)),
    }


def _build(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    h, w, bg = plan["h"], plan["w"], plan["bg"]
    grid = np.full((h, w), bg, dtype=np.int16)
    occ = np.zeros((h, w), dtype=bool)
    target_occ = np.zeros((h, w), dtype=bool)
    expanded = np.zeros((h, w), dtype=bool)
    lim_h = max(1, min(h - 1, 18))
    lim_w = max(1, min(w - 1, 18))
    families: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    per_target = plan["target_budget"] / max(1, plan["n_targets"])

    for index, pieces in enumerate(plan["pieces"]):
        built = None
        for shrink in (1.0, 0.6, 0.35):
            built = _figure(rng, max(3.0, per_target * shrink), pieces, lim_h, lim_w)
            if built is not None:
                break
        if built is None:
            return None
        mask, fams = built
        edge = plan["edge_side"] if (plan["edge"] and index == 0) else None
        cells = _place_target(rng, grid, occ, target_occ, expanded, mask,
                              plan["target_color"], edge=edge)
        if cells is None and edge is not None:
            cells = _place_target(rng, grid, occ, target_occ, expanded, mask,
                                  plan["target_color"])
        if cells is None:
            return None
        target_occ |= cells
        families.extend(fams)
        rr, cc = np.nonzero(cells)
        box = (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max()))
        boxes.append(box)
        er0, ec0, er1, ec1 = _expanded_box(box, h, w)
        expanded[er0:er1 + 1, ec0:ec1 + 1] = True

    # the ring band of the 8-reading: the cells a ring can possibly write on
    band = np.zeros((h, w), dtype=bool)
    for box in boxes:
        for r, c in outline_cells(box, h, w):
            band[r, c] = True
    band &= ~target_occ

    placed = 0
    crossing = 0
    distractor_families: list[str] = []
    for index, color in enumerate(plan["colors"]):
        prefer = band if index < plan["cross"] else None
        for shrink in (1.0, 0.55, 0.3):
            built = _sized_body(rng, plan["per_distractor"] * shrink, lim_h, lim_w, min_cells=3)
            if built is None:
                continue
            # a third of the bodies may touch what is already drawn: different
            # colours never merge under the rule
            free = (~occ) if rng.random() < 0.34 else (~halo(occ))
            cells = _place_body(rng, grid, occ, built[0], int(color), free, prefer=prefer)
            if cells is None:
                cells = _place_body(rng, grid, occ, built[0], int(color), ~halo(occ),
                                    prefer=prefer)
            if cells is not None:
                placed += 1
                distractor_families.append(built[1])
                if bool((cells & band).any()):
                    crossing += 1
                break

    # lone color X cells: isolated from every other color X cell, so they are
    # background texture under the prompt's isolated-cell clause
    strays = 0
    for _ in range(plan["strays"]):
        free = (~occ) & (~halo(grid == plan["target_color"]))
        spots = np.argwhere(free)
        if not len(spots):
            break
        r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
        grid[r, c] = plan["target_color"]
        occ[r, c] = True
        strays += 1

    return {"grid": grid, "occ": occ, "target_occ": target_occ, "boxes": boxes,
            "families": families, "distractor_families": distractor_families,
            "n_distractors": placed, "crossing": crossing, "strays": strays}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_95(example: dict) -> None:
    """Re-read the emitted grid and re-derive every ring, literally."""
    grid, meta = example["input"], example["metadata"]
    color = int(meta["target_color"])
    outline = int(meta["outline_color"])
    conn = int(meta["connectivity"])
    h, w = len(grid), len(grid[0])
    if any(int(v) == outline for row in grid for v in row):
        raise ValueError("task 95: the outline colour occurs in the input")
    objects = target_objects(grid, color, conn)
    if not objects:
        raise ValueError("task 95: no object of the target colour")
    if len(objects) != meta["n_targets"]:
        raise ValueError("task 95: the object count disagrees with the metadata")
    expected = solve(grid, color, conn, outline)
    if expected != example["output"]:
        raise ValueError("task 95: the outlined grid disagrees with the literal reading")
    other = 4 if conn == 8 else 8
    try:
        alt = solve(grid, color, other, outline)
    except ValueError:
        raise ValueError("task 95: the other reading is not well defined")
    if (alt != example["output"]) != bool(meta["differs_under_other_connectivity"]):
        raise ValueError("task 95: the load-bearing flag is wrong")
    ring = set()
    for comp in objects:
        ring.update(outline_cells(bbox(comp), h, w))
    if len(ring) != meta["ring_cells"]:
        raise ValueError("task 95: the ring size disagrees with the metadata")
    # Distinct target figures are seated outside each other's expanded boxes, so
    # the only cells a ring ever takes from an object are the pieces of ONE
    # corner-joined figure read under 4-connectivity, where each piece's ring
    # reaches the corner cell of its neighbour.  The prompt says so ("another
    # object's cells included") and every object is read off the input first.
    target_cells = {cell for comp in objects for cell in comp}
    if len(ring & target_cells) != meta["target_cells_covered"]:
        raise ValueError("task 95: the covered-target count disagrees with the metadata")
    if (ring & target_cells) and (conn != 4 or meta["n_targets"] == meta["n_targets_other_connectivity"]):
        raise ValueError("task 95: a ring covers a target outside the corner-joined 4-reading")
    for row in grid:
        for value in row:
            if not 0 <= int(value) <= 9:
                raise ValueError("task 95: a cell is outside the palette")
    if meta["texture_colors"]:
        if color in meta["texture_colors"] or outline in meta["texture_colors"]:
            raise ValueError("task 95: a texture colour is a task colour")
        arr = np.asarray(grid)
        if meta["texture_group"] == "sparse":
            tex = np.isin(arr, meta["texture_colors"])
            if not bool(isolated_cell_mask(arr)[tex].all()):
                raise ValueError("task 95: a sparse texture cell is not isolated")


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
                                exclude=[plan["target_color"], plan["outline_color"],
                                         *plan["distractor_colors"]],
                                probability=1.0 if plan["plan_texture"] else 0.0)
        rows = grid_to_list(display.astype(np.int16))
        h, w = plan["h"], plan["w"]

        # Nothing above this line looked at the connectivity: both readings are
        # well defined on every scene (every piece holds at least two cells and
        # every lone target cell is isolated), so the coin is fair.
        try:
            objs4 = target_objects(rows, plan["target_color"], 4)
            objs8 = target_objects(rows, plan["target_color"], 8)
        except ValueError:
            continue
        if not objs4 or not objs8:
            continue
        if len(objs8) != plan["n_targets"]:
            continue
        conn = 4 if rng.random() < 0.5 else 8
        objects = objs4 if conn == 4 else objs8
        out = solve(rows, plan["target_color"], conn, plan["outline_color"])
        alt = solve(rows, plan["target_color"], 4 if conn == 8 else 8, plan["outline_color"])

        boxes = [bbox(comp) for comp in objects]
        ring_count: dict[tuple[int, int], int] = {}
        clipped = 0
        for box in boxes:
            clipped += len(full_outline(box)) - len(outline_cells(box, h, w))
            for cell in outline_cells(box, h, w):
                ring_count[cell] = ring_count.get(cell, 0) + 1
        overlap = sum(1 for n in ring_count.values() if n > 1)
        target_cells = {cell for comp in objects for cell in comp}
        covered = sum(1 for cell in ring_count if cell in target_cells)
        over_object = sum(1 for cell in ring_count
                          if cell not in target_cells and bool(occ[cell[0], cell[1]]))
        over_texture = sum(1 for (r, c) in ring_count
                           if not bool(occ[r, c]) and rows[r][c] != plan["bg"])
        touches_border = sum(1 for (r0, c0, r1, c1) in boxes
                             if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1)
        present = sorted({int(v) for v in np.unique(display)})
        out_present = sorted({int(v) for row in out for v in row})
        meta = {
            "grid_h": h, "grid_w": w,
            "n_colors": len(present), "n_colors_output": len(out_present),
            "connectivity": conn,
            "target_color": int(plan["target_color"]),
            "outline_color": int(plan["outline_color"]),
            "background_color": int(plan["bg"]),
            "n_targets": len(objects),
            "n_targets_other_connectivity": len(objs4 if conn == 8 else objs8),
            "differs_under_other_connectivity": bool(alt != out),
            "target_cells": sum(len(comp) for comp in objects),
            "target_object_sizes": sorted(len(comp) for comp in objects),
            "target_boxes": [list(b) for b in sorted(boxes)],
            "figure_cap": plan["cap"],
            "n_figures": len(objs8),
            "ring_cells": len(ring_count),
            "ring_cells_clipped": clipped,
            "rings_clipped": bool(clipped > 0),
            "ring_overlap_cells": overlap,
            "rings_touch": bool(overlap > 0),
            "ring_cells_over_objects": over_object,
            "target_cells_covered": covered,
            "ring_cells_over_texture": over_texture,
            "rings_overwrite_object": bool(over_object > 0),
            "targets_touching_border": touches_border,
            "stray_target_cells": scene["strays"],
            "n_distractors": scene["n_distractors"],
            "distractors_crossing_ring": scene["crossing"],
            "object_kinds": scene["families"],
            "distractor_kinds": scene["distractor_families"],
            "occupancy": round(float(occ.sum()) / plan["area"], 4),
            "object_cells": int(occ.sum()),
            "planned_density": round(plan["density"], 4),
            "palette_target": plan["palette_target"],
            "selection_rule": "every object of the target color",
            "outline_rule": "the one-cell ring just outside the object's tight bounding box",
            "overwrite_rule": "the outline color is written over whatever occupies the ring cell",
            "reading_rule": "every object is read off the input before any outline is drawn",
            "clip_rule": "outline cells outside the grid are dropped",
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows, "output": out, "metadata": meta}
        validate_95(example)
        return example
    raise RuntimeError(f"task 95: no valid scene after {MAX_ATTEMPTS} attempts")


@register("95")
def generate_draw_bounding_box_outlines(rng: np.random.Generator) -> dict:
    """Task 95: outline the tight bounding box of every color X object."""
    return _example(rng)
