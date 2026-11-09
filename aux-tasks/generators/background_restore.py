"""Task 5a -- remove every object and give back the background (the converse of 4/5).

Plan
----
Input: a background is built FIRST over the whole canvas -- a plain ground, or a ground plus one
regular pattern from ``common.backgrounds`` (sparse: lattice, dotted_stripes, sparse_checker,
dotted_diagonal; solid: bands, blocks, border, split, diagonal_bands, concentric, grid_lines,
periodic; structured: mosaic) -- then objects from the shared shape vocabulary are stamped on
top, overwriting background cells.  Output: the background alone, i.e. the pattern continued
through the cells the objects covered.

Only kinds with a closed structure are used, because the answer must be inferable from the
visible cells: random_dots, multi_speckle and mirror are random fills (a hidden cell has no
witness) and dotted_border is a perimeter walk whose greedy corner filter breaks its period, so
those four are never drawn.  The layer painters are called directly (not through
``apply_background``) so clipped corner fragments are kept and every pattern is exactly what its
formula says; with no objects present at background time the coverage limits are moot (the
whole canvas is background).

Determinability proof (every example, independent of the generator's bookkeeping)
----------------------------------------------------------------------------------
``infer_fill(grid, hidden)`` takes the input grid and the mask of object cells and enumerates
every background hypothesis consistent with the VISIBLE cells from four families:

  A. 2-D periodic: v(r, c) = v(r mod py, c mod px), 1 <= py, px <= 18.  A candidate must be
     row-consistent (py) and column-consistent (px) on the visible cells, jointly consistent on
     its residue classes, and every hidden cell's class must hold a visible witness.  Candidates
     that are multiples of an already valid finer candidate are skipped (their classes are
     unions of witnessed single-valued classes, so they cannot produce a different fill).
     Covers plain, lattice, dotted_stripes, sparse_checker, dotted_diagonal, bands, blocks,
     diagonal_bands, grid_lines, periodic and mosaic-as-checker.
  B. Border distance: v = f(min(r, c, H-1-r, W-1-c)) -- border and concentric.
  C. Axis / corner split: v = a inside a rectangle anchored at one grid corner (or a top/bottom
     or left/right half-plane), b elsewhere; every boundary consistent with the visible cells is
     enumerated -- split.
  E. Tile-constant: v = T[(r+oy)//b, (c+ox)//b] for b in {2, 3} -- mosaic.

The example is accepted only if the set of fills the valid hypotheses assign to the hidden
cells has exactly one member AND that member equals the true background.  Objects that hide a
whole period, a split boundary, or a full mosaic tile make a second hypothesis valid (or leave
none), and the scene is rebuilt for the same spec.  The proof assumes the object cells are known;
a solver can tell them apart because object colours never occur in the background (ground or
texture) and objects are the only same-colour groups that break the pattern -- the same
inference the user accepted for the unnamed textures on tasks 4/5/7/8.

Size floor and visibility (user, 2026-09-04: "limit 5a to medium -> large grids where the
pattern can be established"; the independent solver's only misses were strips and grids under
100 cells, plus a 24x27 border ring buried under 24 objects at a third occupancy)
-------------------------------------------------------------------------------------------
* Canvas: short side >= 8 and area >= 120 (H and W independent, uniform over 8..30, the
  area floor rejecting only the smallest pairs); no strips.  Sparse kinds are always allowed
  (short side >= 8), mosaic needs a short side >= 12.
* Visibility guarantees, checked on the visible grid before the proof and rejected (rebuilt for
  the same spec) otherwise:
    (a) every pattern class has at least min(3, class size) visible witnesses -- the class map is
        the minimal 2-D period of the background for periodic kinds (incl. plain, bands, blocks,
        grid lines, diagonal bands, lattice, dotted stripes, sparse checker, dotted diagonal,
        periodic, mosaic), the border depth for border / concentric, and the colour regions for
        split;
    (b) at least 60% of the texture cells stay visible;
    (c) no object spans the full height or the full width of the canvas;
    (d) occupancy is capped at 28% (targets are drawn in 12-28% and the object budget follows).

Policies (checklist): colours near-flat over 2..10 where the scene allows; objects have >= 2
cells, never touch a same-colour object (1-cell moat), may touch other colours, and scale with
the canvas; ~5% of examples carry no object (output == input); the prompt names nothing about
the background or the textures; every answer is recomputed and proved; no canned fallback.
"""
from __future__ import annotations

import numpy as np

from common import backgrounds as bglib
from common import shapes as shape_lib
from common.grid import NUM_COLORS, grid_to_list, pick_colors
from registry import register

MAX_PERIOD = 18
_DEBUG_FAIL = None   # set to a collections.Counter to record why builds are rejected


def _fail(reason: str):
    if _DEBUG_FAIL is not None:
        _DEBUG_FAIL[reason] += 1
    return None
MAX_ATTEMPTS_PER_SPEC = 8
MAX_SPECS = 40
IDENTITY_P = 0.05
MIN_SHORT_SIDE = 8
MIN_AREA = 120
MAX_OCCUPANCY = 0.28
TEXTURE_VISIBLE_MIN = 0.60
CLASS_WITNESSES = 3

SPARSE_KINDS = ("lattice", "dotted_stripes", "sparse_checker", "dotted_diagonal")
SOLID_KINDS = ("bands", "blocks", "border", "split", "diagonal_bands", "concentric", "grid_lines", "periodic")
STRUCTURED_KINDS = ("mosaic",)
GROUP_OF = {**{k: "sparse" for k in SPARSE_KINDS}, **{k: "solid" for k in SOLID_KINDS},
            **{k: "structured" for k in STRUCTURED_KINDS}}
# group weights when every group fits the canvas; plain always available
_GROUP_P = {"plain": 0.18, "sparse": 0.28, "solid": 0.42, "structured": 0.12}
_COLOR_TARGET_P = np.array([1.15, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.85])  # 2..10, mild lean


def _rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def _choose_kind(rng: np.random.Generator, H: int, W: int) -> str:
    short = min(H, W)
    groups = ["plain", "solid", "sparse"]     # short side >= MIN_SHORT_SIDE always
    if short >= 12:
        groups.append("structured")
    p = np.array([_GROUP_P[g] for g in groups], dtype=float)
    group = groups[int(rng.choice(len(groups), p=p / p.sum()))]
    if group == "plain":
        return "plain"
    kinds = {"sparse": SPARSE_KINDS, "solid": SOLID_KINDS, "structured": STRUCTURED_KINDS}[group]
    return kinds[int(rng.integers(0, len(kinds)))]


def _build_background(rng: np.random.Generator, H: int, W: int, n_tex_colors_cap: int):
    """(background grid, info).  Ground colour plus, for a textured kind, 1-3 texture colours."""
    kind = _choose_kind(rng, H, W)
    bg = int(rng.integers(0, NUM_COLORS))
    info = {"kind": kind, "group": "plain" if kind == "plain" else GROUP_OF[kind], "texture_colors": []}
    grid = np.full((H, W), bg, dtype=np.int64)
    if kind == "plain":
        return grid, info
    group = GROUP_OF[kind]
    if group == "sparse":
        layers = bglib.sparse_mask(rng, H, W, kind).astype(np.int8)
        n_col = 1
    elif group == "solid":
        n_col = _rint(rng, 1, min(3, n_tex_colors_cap)) if kind == "periodic" else 1
        layers = bglib.solid_layers(rng, H, W, kind, 0, n_col)
        n_col = int(layers.max()) if layers.any() else 0
    else:
        layers = bglib.structured_layers(rng, H, W, kind, 0)
        n_col = int(layers.max()) if layers.any() else 0
    if n_col == 0 or not (layers > 0).any() or (layers > 0).all():
        # degenerate draw (empty or all-texture): fall back to a plain ground for this spec
        info["kind"], info["group"] = "plain", "plain"
        return grid, info
    colors = [int(c) for c in pick_colors(rng, n_col, exclude=[bg])]
    for i, col in enumerate(colors, start=1):
        grid[layers == i] = col
    info["texture_colors"] = colors
    return grid, info


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------

def _place_objects(rng: np.random.Generator, background: np.ndarray, palette: list[int], target_occ: float):
    """Stamp vocabulary shapes over the background.  Returns (grid, object mask, list of
    (colour, cells)).  Same-colour objects keep a one-cell moat; other colours may touch."""
    H, W = background.shape
    area = H * W
    grid = background.copy()
    occupied = np.zeros((H, W), dtype=bool)
    near: dict[int, np.ndarray] = {}
    objects: list[tuple[int, frozenset]] = []
    budget = int(round(min(target_occ, MAX_OCCUPANCY) * area))
    # a shape may never span the full height or width (it would read as a stripe or a split)
    max_side = max(2, min(12, min(H, W) - 1))
    order = [palette[i] for i in rng.permutation(len(palette)).tolist()]
    tries = 0
    while budget >= 2 and tries < 60:
        tries += 1
        # leave room for every palette colour to get an object before the budget runs out
        need = max(1, len(order) - len(objects))
        max_cells = int(max(2, min(60, budget // need, max(4, area // 5))))
        shape, _fam = shape_lib.sample_shape(rng, max_cells, min(H, max_side), min(W, max_side), connectivity=4)
        if len(shape) < 2:
            continue
        offs = sorted(shape)
        hs = max(r for r, _ in offs) + 1
        ws = max(c for _, c in offs) + 1
        if hs >= H or ws >= W:
            continue
        color = order[len(objects) % len(order)]
        allowed = ~occupied
        if color in near:
            allowed = allowed & ~near[color]
        nh, nw = H - hs + 1, W - ws + 1
        valid = np.ones((nh, nw), dtype=bool)
        for dr, dc in offs:
            valid &= allowed[dr:dr + nh, dc:dc + nw]
        anchors = np.argwhere(valid)
        if len(anchors) == 0:
            continue
        r0, c0 = (int(v) for v in anchors[int(rng.integers(0, len(anchors)))])
        cells = frozenset((r0 + dr, c0 + dc) for dr, dc in offs)
        rows = np.fromiter((r for r, _ in cells), dtype=int, count=len(cells))
        cols = np.fromiter((c for _, c in cells), dtype=int, count=len(cells))
        grid[rows, cols] = color
        occupied[rows, cols] = True
        # one-cell moat for the same colour
        m = np.zeros((H + 2, W + 2), dtype=bool)
        m[rows + 1, cols + 1] = True
        dil = np.zeros((H, W), dtype=bool)
        for dr in (0, 1, 2):
            for dc in (0, 1, 2):
                dil |= m[dr:dr + H, dc:dc + W]
        near.setdefault(color, np.zeros((H, W), dtype=bool))
        near[color] |= dil
        objects.append((color, cells))
        budget -= len(cells)
    return grid, occupied, objects


# ---------------------------------------------------------------------------
# Determinability proof
# ---------------------------------------------------------------------------

def _axis_consistent(v: np.ndarray, vis: np.ndarray, shift: int, axis: int) -> bool:
    n = v.shape[axis]
    if shift >= n:
        return True
    if axis == 0:
        a, b, va, vb = v[:-shift], v[shift:], vis[:-shift], vis[shift:]
    else:
        a, b, va, vb = v[:, :-shift], v[:, shift:], vis[:, :-shift], vis[:, shift:]
    return not np.any(va & vb & (a != b))


def _class_fill(cls: np.ndarray, v: np.ndarray, vis: np.ndarray, hidden: np.ndarray, n_cls: int):
    """Fill of the hidden cells under a class map, or None when a class holds two visible
    values or a hidden cell's class has no visible witness."""
    cv = cls[vis]
    vv = v[vis]
    order = np.argsort(cv, kind="stable")
    cs, vs = cv[order], vv[order]
    same = cs[1:] == cs[:-1]
    if np.any(same & (vs[1:] != vs[:-1])):
        return None
    first = np.full(n_cls, -1, dtype=np.int64)
    first[cs] = vs
    fill = first[cls[hidden]]
    if np.any(fill < 0):
        return None
    return fill


def infer_fill(grid: np.ndarray, hidden: np.ndarray):
    """Enumerate background hypotheses consistent with the visible cells (families A, B, C, E).

    Returns (fill or None, number of distinct fills, families that produced a valid
    hypothesis).  ``fill`` is the hidden cells' values in ``np.argwhere(hidden)`` order and is
    only returned when it is unique."""
    v = np.asarray(grid, dtype=np.int64)
    hidden = np.asarray(hidden, dtype=bool)
    H, W = v.shape
    vis = ~hidden
    n_hidden = int(hidden.sum())
    if n_hidden == 0:
        return np.zeros(0, dtype=np.int64), 1, ["none"]
    vals = np.unique(v[vis])
    if len(vals) == 1:
        return np.full(n_hidden, int(vals[0]), dtype=np.int64), 1, ["plain"]
    fills: dict[bytes, np.ndarray] = {}
    families: set[str] = set()
    rr, cc = np.indices((H, W))

    # --- A: 2-D periodic
    R = [py for py in range(1, min(H, MAX_PERIOD) + 1) if _axis_consistent(v, vis, py, 0)]
    C = [px for px in range(1, min(W, MAX_PERIOD) + 1) if _axis_consistent(v, vis, px, 1)]
    valid_min: list[tuple[int, int]] = []
    for py in R:
        for px in C:
            if any(py % a == 0 and px % b == 0 for a, b in valid_min):
                continue
            cls = (rr % py) * px + (cc % px)
            fill = _class_fill(cls, v, vis, hidden, py * px)
            if fill is not None:
                valid_min.append((py, px))
                fills.setdefault(fill.tobytes(), fill)
                families.add("periodic")

    # --- B: border distance
    depth = np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc))
    fill = _class_fill(depth, v, vis, hidden, int(depth.max()) + 1)
    if fill is not None:
        fills.setdefault(fill.tobytes(), fill)
        families.add("border")

    # --- C: half-plane and corner splits (two colours)
    hid_r, hid_c = np.nonzero(hidden)
    for flip_r in (False, True):
        for flip_c in (False, True):
            r_ = (H - 1 - rr) if flip_r else rr
            c_ = (W - 1 - cc) if flip_c else cc
            hr = (H - 1 - hid_r) if flip_r else hid_r
            hc = (W - 1 - hid_c) if flip_c else hid_c
            for mode in ("rows", "cols", "corner"):
                if mode == "rows" and flip_c:
                    continue
                if mode == "cols" and flip_r:
                    continue
                for R0, C0 in _split_boundaries(v, vis, r_, c_, mode, H, W):
                    if mode == "rows":
                        inside = r_ < R0
                        h_in = hr < R0
                    elif mode == "cols":
                        inside = c_ < C0
                        h_in = hc < C0
                    else:
                        inside = (r_ < R0) & (c_ < C0)
                        h_in = (hr < R0) & (hc < C0)
                    a_vals = np.unique(v[vis & inside])
                    b_vals = np.unique(v[vis & ~inside])
                    if len(a_vals) != 1 or len(b_vals) != 1 or a_vals[0] == b_vals[0]:
                        continue
                    fill = np.where(h_in, int(a_vals[0]), int(b_vals[0])).astype(np.int64)
                    fills.setdefault(fill.tobytes(), fill)
                    families.add("split")

    # --- E: tile-constant
    for b in (2, 3):
        for oy in range(b):
            for ox in range(b):
                ty = (rr + oy) // b
                tx = (cc + ox) // b
                n_tx = int(tx.max()) + 1
                cls = ty * n_tx + tx
                fill = _class_fill(cls, v, vis, hidden, (int(ty.max()) + 1) * n_tx)
                if fill is not None:
                    fills.setdefault(fill.tobytes(), fill)
                    families.add("tiles")

    if len(fills) != 1:
        return None, len(fills), sorted(families)
    return next(iter(fills.values())), 1, sorted(families)


def _split_boundaries(v, vis, r_, c_, mode, H, W):
    """Candidate boundaries for family C: only those the visible cells allow.  For the
    half-plane modes the boundary lies strictly between the last row (column) holding the inside
    colour and the first holding the outside colour; the corner mode enumerates both."""
    out = []
    if mode in ("rows", "cols"):
        axis = r_ if mode == "rows" else c_
        n = H if mode == "rows" else W
        # colour of the first visible line decides the inside colour
        first_vals = None
        lo, hi = 0, n
        for k in range(n):
            sel = vis & (axis == k)
            if not sel.any():
                continue
            u = np.unique(v[sel])
            if first_vals is None:
                first_vals = u
                if len(u) != 1:
                    return out
                lo = k + 1
                continue
            if len(u) == 1 and u[0] == first_vals[0]:
                lo = k + 1
            else:
                hi = k
                break
        if first_vals is None or lo > hi:
            return out
        for k in range(max(1, lo), min(n - 1, hi) + 1):
            out.append((k, k) if mode == "rows" else (k, k))
        return [(k, 0) for k, _ in out] if mode == "rows" else [(0, k) for _, k in out]
    # corner: inside = r_ < R0 & c_ < C0.  Every colour role is enumerated: when the corner
    # cell is visible its colour is the only possible inside colour, otherwise any visible
    # colour may be the inside one (taking the nearest visible cell was a raster tie-break
    # that could enumerate rectangles of the wrong colour -- verifier finding, 2026-09-04).
    sel0 = vis & (r_ == 0) & (c_ == 0)
    if sel0.any():
        candidates = [int(v[sel0][0])]
    else:
        candidates = [int(x) for x in np.unique(v[vis])]
    for a in candidates:
        a_cells = vis & (v == a)
        if not a_cells.any():
            continue
        ra = int(np.max(r_[a_cells]))
        ca = int(np.max(c_[a_cells]))
        b_cells = vis & (v != a)
        if not b_cells.any():
            continue
        # for each R0 > ra: C0 must exceed ca and be <= min c of b cells with r < R0
        rb = r_[b_cells]
        cb = c_[b_cells]
        n_before = len(out)
        for R0 in range(ra + 1, H):
            m = rb < R0
            c_max = int(np.min(cb[m])) if m.any() else W
            for C0 in range(ca + 1, min(W, c_max) + 1):
                # a-cells must all be inside (rows < R0 and cols < C0): guaranteed by ra, ca;
                # b-cells with r < R0 must have c >= C0: guaranteed by c_max
                out.append((R0, C0))
            if len(out) - n_before > 400:
                break
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _sample_size(rng: np.random.Generator) -> tuple[int, int]:
    """H, W independent and uniform over MIN_SHORT_SIDE..30; pairs under MIN_AREA are redrawn."""
    while True:
        h, w = _rint(rng, MIN_SHORT_SIDE, 30), _rint(rng, MIN_SHORT_SIDE, 30)
        if h * w >= MIN_AREA:
            return h, w


def _minimal_period(background: np.ndarray, axis: int) -> int:
    n = background.shape[axis]
    for p in range(1, min(n, MAX_PERIOD) + 1):
        if axis == 0:
            if np.array_equal(background[p:], background[:-p]):
                return p
        else:
            if np.array_equal(background[:, p:], background[:, :-p]):
                return p
    return n


def pattern_classes(background: np.ndarray, kind: str) -> np.ndarray:
    """Class map used by the visibility guarantee (a): border depth for border / concentric,
    the two colour regions for split, the minimal 2-D period otherwise (plain = one class)."""
    H, W = background.shape
    rr, cc = np.indices((H, W))
    if kind in ("border", "concentric"):
        return np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc))
    if kind == "split":
        vals = np.unique(background)
        return np.searchsorted(vals, background)
    py, px = _minimal_period(background, 0), _minimal_period(background, 1)
    return (rr % py) * px + (cc % px)


def visibility_report(background: np.ndarray, occupied: np.ndarray, kind: str, texture_colors) -> dict:
    """Guarantees (a) and (b) measured on one scene."""
    cls = pattern_classes(background, kind)
    n_cls = int(cls.max()) + 1
    total = np.bincount(cls.ravel(), minlength=n_cls)
    visible = np.bincount(cls[~occupied].ravel(), minlength=n_cls)
    need = np.minimum(total, CLASS_WITNESSES)
    if texture_colors:
        tex = np.isin(background, list(texture_colors))
        tex_visible = float((tex & ~occupied).sum() / max(1, tex.sum()))
    else:
        tex_visible = 1.0
    return {"n_classes": n_cls, "min_witnesses": int(visible.min()),
            "classes_short": int((visible < need).sum()), "texture_visible": tex_visible}


def _build_example(rng: np.random.Generator, H: int, W: int, n_colors_target: int, identity: bool, target_occ: float):
    n_tex_cap = max(1, n_colors_target - 2)
    background, binfo = _build_background(rng, H, W, n_tex_cap)
    bg_colors = sorted({int(c) for c in np.unique(background)})
    free = [c for c in range(NUM_COLORS) if c not in bg_colors]
    n_obj_colors = max(1, min(len(free), n_colors_target - len(bg_colors)))
    palette = [int(c) for c in pick_colors(rng, n_obj_colors, exclude=bg_colors)]
    if identity and binfo["kind"] != "plain":
        # identity outputs only over a textured ground (a plain identity grid is one colour)
        grid, occupied, objects = background.copy(), np.zeros((H, W), dtype=bool), []
    else:
        grid, occupied, objects = _place_objects(rng, background, palette, target_occ)
        if not objects:
            return _fail("no object placed")
    # --- independent checks on the visible grid
    if not np.array_equal(grid[~occupied], background[~occupied]):
        return _fail("background altered outside objects")
    for color, cells in objects:
        if len(cells) < 2 or color in bg_colors:
            return _fail("object too small or in a background colour")
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if max(rs) - min(rs) + 1 >= H or max(cs) - min(cs) + 1 >= W:
            return _fail("object spans the full canvas")
    if float(occupied.mean()) > MAX_OCCUPANCY + 1e-9:
        return _fail("occupancy above the cap")
    vis_report = visibility_report(background, occupied, binfo["kind"], binfo["texture_colors"])
    if vis_report["classes_short"]:
        return _fail("visibility: a pattern class lacks witnesses")
    if vis_report["texture_visible"] < TEXTURE_VISIBLE_MIN:
        return _fail("visibility: texture under 60% visible")
    fill, n_fills, fams = infer_fill(grid, occupied)
    if fill is None:
        return _fail("not determinable: no fill" if n_fills == 0 else f"not determinable: {n_fills} fills")
    if not np.array_equal(fill, background[occupied]):
        return _fail("unique fill differs from the background")
    if objects and not _components_match(grid, occupied, objects):
        return _fail("object components merged or split")
    used = sorted({c for c, _ in objects})
    metadata = {
        "background": int(background.flat[0]) if binfo["kind"] == "plain" else _ground_color(background, binfo),
        "background_colors": bg_colors,
        "texture_group": None if binfo["kind"] == "plain" else binfo["group"],
        "texture_kind": None if binfo["kind"] == "plain" else binfo["kind"],
        "texture_colors": list(binfo["texture_colors"]),
        "texture_named": False,
        "texture_coverage": float(np.mean(np.isin(background, binfo["texture_colors"]))) if binfo["texture_colors"] else 0.0,
        "connectivity": 8,
        "n_objects": len(objects),
        "object_colors": used,
        "object_cells": int(occupied.sum()),
        "occupancy": float(occupied.mean()),
        "n_colors": int(len(np.unique(grid))),
        "grid_h": int(H),
        "grid_w": int(W),
        "identity": bool(not objects),
        "determinability": {"distinct_fills": int(n_fills), "families": fams},
        "visibility": {"texture_visible": round(vis_report["texture_visible"], 4),
                       "n_classes": vis_report["n_classes"], "min_witnesses": vis_report["min_witnesses"]},
    }
    return {"input": grid_to_list(grid), "output": grid_to_list(background), "metadata": metadata}


def _ground_color(background: np.ndarray, binfo: dict) -> int:
    tex = set(binfo["texture_colors"])
    vals, counts = np.unique(background, return_counts=True)
    ground = [(int(c), int(n)) for c, n in zip(vals, counts) if int(c) not in tex]
    return max(ground, key=lambda t: t[1])[0]


def _components_match(grid: np.ndarray, occupied: np.ndarray, objects) -> bool:
    """Every placed object is its own 8-connected same-colour component of the object cells."""
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    n = 0
    for r in range(H):
        for c in range(W):
            if not occupied[r, c] or seen[r, c]:
                continue
            color = grid[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                a, b = stack.pop()
                size += 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        x, y = a + dr, b + dc
                        if 0 <= x < H and 0 <= y < W and occupied[x, y] and not seen[x, y] and grid[x, y] == color:
                            seen[x, y] = True
                            stack.append((x, y))
            n += 1
            if size < 2:
                return False
    return n == len(objects)


@register("5a")
def generate_background_restore(rng: np.random.Generator) -> dict:
    """Task 5a: output the background with every object removed (see the module docstring)."""
    for _spec in range(MAX_SPECS):
        H, W = _sample_size(rng)
        n_colors_target = 2 + int(rng.choice(9, p=_COLOR_TARGET_P / _COLOR_TARGET_P.sum()))
        identity = rng.random() < IDENTITY_P
        target_occ = float(rng.uniform(0.12, MAX_OCCUPANCY))
        for _attempt in range(MAX_ATTEMPTS_PER_SPEC):
            example = _build_example(rng, H, W, n_colors_target, identity, target_occ)
            if example is not None:
                return example
    raise RuntimeError("task 5a: could not build a provably restorable scene")
