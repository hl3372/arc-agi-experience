"""Color filtering and shape-matching task generators: tasks 7, 8, 20, 21, 29, 70, 84, 85."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from collections import Counter

from common import shapes as S
from common.grid import (
    NUM_COLORS,
    generate_scene,
    grid_to_list,
    make_grid,
    random_background,
    random_grid_size,
)
from common.objects import (
    bounding_box,
    connected_components,
    DELTAS_4,
    DELTAS_8,
    normalize_shape,
    place_shape_on_grid,
    random_shape,
    rotate_shape_cw,
)
from registry import register
from common import backgrounds as bglib

MAX_RETRIES = 20

def _bg_is_strict_plurality(grid, bg):
    """True iff bg is the unique most-frequent color, so a solver can infer the background."""
    vals, counts = np.unique(grid, return_counts=True)
    cmax = int(counts.max())
    return int(vals[int(counts.argmax())]) == bg and int((counts == cmax).sum()) == 1


def _rotation_signature(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Canonicalize a shape under 90-degree rotations only."""
    shape = normalize_shape(cells)
    variants = []
    current = shape
    for _ in range(4):
        variants.append(tuple(sorted(current)))
        current = rotate_shape_cw(current)
    return min(variants)


def _rot_key(cells) -> tuple:
    """Canonical key of a shape under 90-degree rotations (tasks 7/8 fast path).

    Same equivalence classes as ``_rotation_signature`` (which the sibling tasks keep),
    computed on a boolean mask with ``np.rot90`` instead of tuple sorting: the key is
    the smallest (height, width, bytes) triple over the four rotations."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    mask = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=np.uint8)
    mask[[r - r0 for r in rows], [c - c0 for c in cols]] = 1
    best = None
    for k in range(4):
        m = np.rot90(mask, k) if k else mask
        key = (m.shape[0], m.shape[1], m.tobytes())
        if best is None or key < best:
            best = key
    return best


def _random_rotated_shape(
    rng: np.random.Generator,
    shape: frozenset[tuple[int, int]],
) -> frozenset[tuple[int, int]]:
    """Return a random 90-degree rotation of *shape*."""
    rotated = normalize_shape(shape)
    for _ in range(int(rng.integers(0, 4))):
        rotated = rotate_shape_cw(rotated)
    return rotated


def _unique_rotation_variants(
    shape: frozenset[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]]:
    """Return the distinct normalized rotations of *shape*."""
    variants: list[frozenset[tuple[int, int]]] = []
    seen: set = set()
    current = normalize_shape(shape)
    for _ in range(4):
        key = frozenset(current)
        if key not in seen:
            variants.append(current)
            seen.add(key)
        current = rotate_shape_cw(current)
    return variants


def _shape_component_count(
    shape: set[tuple[int, int]] | frozenset[tuple[int, int]],
    connectivity: int,
) -> int:
    """Count components in a shape under the requested connectivity."""
    normalized = normalize_shape(shape)
    _, _, max_r, max_c = bounding_box(normalized)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=np.uint8)
    for r, c in normalized:
        mask[r, c] = 1
    return len(connected_components(mask, connectivity=connectivity, background=0))


def _requires_8_connectivity(
    shape: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> bool:
    """True when a shape is one object only if diagonal adjacency is allowed."""
    return _shape_component_count(shape, connectivity=4) > 1


def _pick_highlight_color(
    grid: np.ndarray,
    bg: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Prefer a color not already used in the scene for output highlighting."""
    used = set(int(v) for v in np.unique(grid))
    unused = [color for color in range(NUM_COLORS) if color != bg and color not in used]
    if unused:
        return int(rng.choice(unused)) if rng is not None else unused[0]
    for color in range(NUM_COLORS):
        if color != bg:
            return color
    return 0


def _build_scene_with_rotation_matches(
    rng: np.random.Generator,
    *,
    same_color_for_pairs: bool = False,
) -> Optional[Tuple[np.ndarray, int]]:
    """Build a scene whose matching relation is equality up to rotation."""
    grid_h = int(rng.integers(14, 29))
    grid_w = int(rng.integers(14, 29))
    grid, bg = random_background(rng, grid_h, grid_w, style="uniform")
    available_colors = [c for c in range(NUM_COLORS) if c != bg]

    n_shapes = int(rng.integers(5, 9))
    n_dups = int(rng.integers(1, min(3, n_shapes - 2) + 1))

    base_shapes: list[frozenset[tuple[int, int]]] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for idx in range(n_shapes):
        placed = False
        for _attempt in range(80):
            if idx == 0 or (idx == 1 and rng.random() < 0.6):
                shape = random_shape(rng, size_range=(5, 8), connectivity=4)
            elif rng.random() < 0.35:
                shape = random_shape(rng, size_range=(4, 7), connectivity=4)
            else:
                shape = random_shape(rng, size_range=(2, 6), connectivity=4)
            sig = _rotation_signature(shape)
            if sig in signatures:
                continue
            signatures.add(sig)
            base_shapes.append(shape)
            placed = True
            break
        if not placed:
            return None

    shape_sizes = np.array([len(shape) for shape in base_shapes], dtype=int)
    sorted_indices = list(np.argsort(shape_sizes)[::-1])
    largest_idx = int(sorted_indices[0])
    dup_pool = list(range(n_shapes))
    dup_indices: list[int] = []
    if rng.random() < 0.65:
        dup_indices.append(largest_idx)
        dup_pool.remove(largest_idx)
    while len(dup_indices) < n_dups:
        pick = int(rng.choice(dup_pool))
        dup_pool.remove(pick)
        dup_indices.append(pick)

    placements: list[tuple[frozenset[tuple[int, int]], int]] = []
    for idx, shape in enumerate(base_shapes):
        placements.append((_random_rotated_shape(rng, shape), idx))
    for idx in dup_indices:
        placements.append((_random_rotated_shape(rng, base_shapes[idx]), idx))
    if dup_indices and len(placements) < 10 and rng.random() < 0.3:
        extra_idx = int(rng.choice(dup_indices))
        placements.append((_random_rotated_shape(rng, base_shapes[extra_idx]), extra_idx))
    rng.shuffle(placements)

    placed_signatures: list[tuple[tuple[int, int], ...]] = []
    color_by_shape_id: dict[int, int] = {}
    for oriented_shape, shape_id in placements:
        if same_color_for_pairs:
            if shape_id not in color_by_shape_id:
                color_by_shape_id[shape_id] = int(rng.choice(available_colors))
            color = color_by_shape_id[shape_id]
        else:
            color = int(rng.choice(available_colors))

        if not place_shape_on_grid(grid, rng, oriented_shape, color, bg, margin=0):
            return None
        placed_signatures.append(tuple(sorted(normalize_shape(oriented_shape))))

    comps = connected_components(grid, connectivity=4, background=bg)
    if len(comps) < 2:
        return None

    rotation_groups: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for comp in comps:
        sig = _rotation_signature(comp["cells"])
        rotation_groups.setdefault(sig, []).append(comp)

    # Require at least one pair that matches only after rotation so the task
    # actually teaches the spec's orientation rule.
    if not any(
        len(group) >= 2
        and len({tuple(sorted(normalize_shape(comp["cells"]))) for comp in group}) >= 2
        for group in rotation_groups.values()
    ):
        return None

    return grid, bg

def _reflect_shape(
    shape: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> frozenset[tuple[int, int]]:
    """Mirror a shape across the vertical axis (normalized)."""
    return normalize_shape({(r, -c) for (r, c) in shape})


def _make_near_miss(
    rng: np.random.Generator,
    shape: frozenset[tuple[int, int]],
    connectivity: int,
    max_edits: int = 2,
) -> Optional[frozenset[tuple[int, int]]]:
    """Return a connected 1-2 cell perturbation of *shape* that is NOT a rotation
    of it (so it looks almost identical but does not match), or None."""
    base_sig = _rot_key(shape)
    base = set(normalize_shape(shape))
    deltas = DELTAS_4 if connectivity == 4 else DELTAS_8
    for _ in range(60):
        cur = set(base)
        for _ in range(int(rng.integers(1, max_edits + 1))):
            if rng.random() < 0.5 and len(cur) > 3:
                cand = list(cur)
                victim = cand[int(rng.integers(0, len(cand)))]
                trial = cur - {victim}
                if trial and _shape_component_count(trial, connectivity) == 1:
                    cur = trial
            else:
                frontier = set()
                for (r, c) in cur:
                    for dr, dc in deltas:
                        nb = (r + dr, c + dc)
                        if nb not in cur:
                            frontier.add(nb)
                if frontier:
                    fl = list(frontier)
                    cur.add(fl[int(rng.integers(0, len(fl)))])
        cur = normalize_shape(cur)
        if (len(cur) >= 2
                and _shape_component_count(cur, connectivity) == 1
                and _rot_key(cur) != base_sig):
            return cur
    return None

# ---------------------------------------------------------------------------
# Shared private helpers for the overhauled tasks 7/8 and 20/21
# ---------------------------------------------------------------------------

_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _mixture_side(
    rng: np.random.Generator,
    bands: tuple[tuple[int, int], ...],
    probs: tuple[float, ...],
    cap: Optional[int] = None,
) -> int:
    """One grid side drawn from a band mixture (uniform inside the chosen band).

    ``cap`` truncates the bands (bands that fall entirely above it are dropped) so a
    single side can be re-drawn to satisfy an area limit without touching the other
    side -- H and W therefore never re-couple."""
    usable: list[tuple[tuple[int, int], float]] = []
    for (lo, hi), p in zip(bands, probs):
        if cap is not None:
            hi = min(hi, cap)
        if hi >= lo:
            usable.append(((lo, hi), p))
    if not usable:
        return max(1, cap if cap is not None else bands[0][0])
    ps = np.array([p for _, p in usable], dtype=float)
    ps /= ps.sum()
    lo, hi = usable[int(rng.choice(len(usable), p=ps))][0]
    return int(rng.integers(lo, hi + 1))


def _fit_orientation(shape, max_h: int, max_w: int):
    """Return *shape* (or its 90-degree rotation, never a mirror) if it fits max_h x max_w."""
    sh, sw = S.shape_size(shape)
    if sh <= max_h and sw <= max_w:
        return shape
    if sw <= max_h and sh <= max_w:
        return rotate_shape_cw(normalize_shape(shape))
    return None


def _attach_diagonally(rng: np.random.Generator, base, piece):
    """Translate *piece* so it touches *base* only at corners.

    The union is 8-connected but NOT 4-connected across the seam: at least one diagonal
    contact, no orthogonal contact, no overlap. Returns the normalized union or None."""
    base_set = set(base)
    piece = normalize_shape(piece)
    cands: set[tuple[int, int]] = set()
    for (r, c) in base_set:
        for dr, dc in _DIAGONALS:
            for (pr, pc) in piece:
                cands.add((r + dr - pr, c + dc - pc))
    cand_list = sorted(cands)
    rng.shuffle(cand_list)
    for (tr, tc) in cand_list:
        moved = [(pr + tr, pc + tc) for pr, pc in piece]
        ok = True
        for (mr, mc) in moved:
            if (mr, mc) in base_set:
                ok = False
                break
            for dr, dc in DELTAS_4:
                if (mr + dr, mc + dc) in base_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return normalize_shape(base_set | set(moved))
    return None


def _small_piece(rng: np.random.Generator, size: int):
    """A 4-connected piece of exactly *size* cells (size <= 5 uses the free polyominoes)."""
    if size <= 1:
        return frozenset({(0, 0)})
    if size <= 5:
        return S.polyomino(rng, size)
    return S.blob(rng, size, compactness=float(rng.uniform(1.0, 2.0)))


def _with_diagonal_links(rng: np.random.Generator, shape, n_links: int, max_h: int, max_w: int):
    """Attach *n_links* small pieces (1-3 cells) to *shape* by corner contact only, so the
    result is one 8-connected object that falls apart under 4-connectivity."""
    cur = shape
    for _ in range(n_links):
        size = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
        attached = None
        for _try in range(4):
            attached = _attach_diagonally(rng, cur, _small_piece(rng, size))
            if attached is not None and _fit_orientation(attached, max_h, max_w) is not None:
                break
            attached = None
        if attached is None:
            return None
        cur = attached
    fitted = _fit_orientation(cur, max_h, max_w)
    if fitted is None or _shape_component_count(fitted, 4) < 2 or _shape_component_count(fitted, 8) != 1:
        return None
    return fitted


def _placed_cells(grid: np.ndarray, before: np.ndarray, color: int) -> set:
    diff = np.argwhere((grid != before) & (grid == color))
    return {(int(r), int(c)) for r, c in diff.tolist()}


def _place_corner_touching(grid, rng, shape, color, bg, anchor_cells):
    """Place *shape* (same colour as the anchor object) so it touches the anchor at one or
    more corners only -- never orthogonally -- and touches no other object. Returns the
    placed cell set or None."""
    h, w = grid.shape
    shape = normalize_shape(shape)
    anchor = set(anchor_cells)
    cands: set[tuple[int, int]] = set()
    for (r, c) in anchor:
        for dr, dc in _DIAGONALS:
            for (pr, pc) in shape:
                cands.add((r + dr - pr, c + dc - pc))
    cand_list = sorted(cands)
    rng.shuffle(cand_list)
    for (tr, tc) in cand_list:
        cells = [(pr + tr, pc + tc) for pr, pc in shape]
        ok = True
        for (cr, cc) in cells:
            if not (0 <= cr < h and 0 <= cc < w) or int(grid[cr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue
        cellset = set(cells)
        for (cr, cc) in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in cellset:
                        continue
                    if (nr, nc) in anchor:
                        if dr == 0 or dc == 0:
                            ok = False
                    elif int(grid[nr, nc]) != bg:
                        ok = False
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            for (cr, cc) in cells:
                grid[cr, cc] = color
            return cellset
    return None


def _texture_background(rng: np.random.Generator, grid: np.ndarray, bg: int, forbidden: set):
    """Repaint some background cells with 1-2 accent colours (disjoint from *forbidden*).

    Styles: ``stripes`` (every 3rd-5th row/column), ``checker`` (a dot lattice on the two
    diagonal sublattices, isolated cells under both connectivities) and ``gridlines``
    (sparse horizontal + vertical rules). Speckle is deliberately not offered. Returns
    ``(decorated, style, accents)`` or None."""
    avail = [c for c in range(NUM_COLORS) if c != bg and c not in forbidden]
    if not avail:
        return None
    h, w = grid.shape
    style = str(rng.choice(["stripes", "checker", "gridlines"]))
    bg_mask = grid == bg
    rr, cc = np.indices((h, w))
    if style == "stripes":
        accents = [int(rng.choice(avail))]
        period = int(rng.integers(3, 6))
        phase = int(rng.integers(0, period))
        axis = rr if rng.random() < 0.5 else cc
        mask = (axis % period) == phase
        disp = grid.copy()
        disp[mask & bg_mask] = accents[0]
    elif style == "checker":
        k = min(2, len(avail))
        accents = [int(c) for c in rng.choice(avail, size=k, replace=False)]
        mask_a = ((rr % 2) == 0) & ((cc % 2) == 0)
        disp = grid.copy()
        disp[mask_a & bg_mask] = accents[0]
        if k == 2 and rng.random() < 0.6:
            mask_b = ((rr % 2) == 1) & ((cc % 2) == 1)
            disp[mask_b & bg_mask] = accents[1]
        else:
            accents = accents[:1]
    else:
        accents = [int(rng.choice(avail))]
        period = int(rng.integers(4, 7))
        pr = int(rng.integers(0, period))
        pc = int(rng.integers(0, period))
        mask = ((rr % period) == pr) | ((cc % period) == pc)
        disp = grid.copy()
        disp[mask & bg_mask] = accents[0]
    if np.array_equal(disp, grid):
        return None
    return disp, style, accents


def _plurality_bg(grid: np.ndarray) -> Optional[int]:
    """The unique most-frequent colour, or None when there is a tie."""
    vals, counts = np.unique(grid, return_counts=True)
    cmax = int(counts.max())
    if int((counts == cmax).sum()) != 1:
        return None
    return int(vals[int(counts.argmax())])


# ---------------------------------------------------------------------------
# Tasks 7 & 8 — shape-pair matching (grid -> grid recolor | grid -> text pairs)
# ---------------------------------------------------------------------------

_PAIR_BANDS = ((5, 12), (13, 22), (23, 30))
_PAIR_PROBS = (0.35, 0.37, 0.28)  # ≈ width-proportional: near-flat per-value density
PAIR_RECOLOR_MAX_AREA = 900   # full ARC range (1..30 per side); token budgets are not a task-design input
PAIR_ATTEMPTS = 240


def _pair_grid_size(rng: np.random.Generator, area_cap: Optional[int]) -> tuple[int, int]:
    """Independent H, W (short side >= 5, long side <= 30) with a ~30% square share; when
    the area cap bites, only ONE side is re-drawn (from the truncated mixture)."""
    h = _mixture_side(rng, _PAIR_BANDS, _PAIR_PROBS)
    w = _mixture_side(rng, _PAIR_BANDS, _PAIR_PROBS)
    if area_cap is not None and h * w > area_cap:
        if rng.random() < 0.5:
            w = _mixture_side(rng, _PAIR_BANDS, _PAIR_PROBS, cap=area_cap // h)
        else:
            h = _mixture_side(rng, _PAIR_BANDS, _PAIR_PROBS, cap=area_cap // w)
    return h, w


_LARGE_FAMILY_WEIGHTS = {
    "blob": 0.24, "big_blob": 0.18, "frame": 0.12, "rect": 0.10, "scaled": 0.08, "L": 0.05,
    "T": 0.04, "U": 0.04, "plus": 0.04, "H": 0.03, "staircase": 0.05, "bar": 0.03,
}


def _pair_object_caps(rng: np.random.Generator, h: int, w: int, n_obj: int, shrink: float = 1.0) -> dict:
    """Size caps for one object, scaled to the room it has. Three size classes (small /
    medium / large) keep small and large objects mixed inside one scene; the large class
    draws from the big-shape families so it really produces large objects.  On canvases
    of 400+ cells the large class is more frequent and the per-object ceiling grows with
    the area (up to area // 8 cells, side up to 16), so big grids carry big scenes."""
    area = h * w
    room = area / max(1, n_obj)
    big = area >= 400
    ceiling = max(50.0, area / 8.0) if big else 50.0
    base = max(3.0, min(ceiling, room * (1.0 if big else 0.7) * shrink))   # shrink < 1 after a failed placement
    u = rng.random()
    weights = None
    p_small, p_medium = (0.10, 0.40) if big else (0.15, 0.60)
    if u < p_small:
        max_cells, min_cells = int(max(2, min(6, base))), 2
    elif u < p_medium:
        max_cells = int(max(3, min(20, base)))
        min_cells = min(4, max_cells)
    else:
        max_cells = int(base)
        min_cells = min(8, max(5, max_cells // 3))
        if max_cells >= 8:
            weights = _LARGE_FAMILY_WEIGHTS
    side = max(2, min(16 if big else 12, int(round(math.sqrt(max_cells) * 1.9))))
    return {"max_cells": max_cells, "max_h": min(h, side), "max_w": min(w, side),
            "min_cells": min_cells, "weights": weights}


def _pair_shape(
    rng: np.random.Generator,
    caps: dict,
    used_sigs: set,
    *,
    min_cells: int = 1,
    need_rotations: bool = False,
    diag_links: int = 0,
):
    """A vocabulary shape (optionally with diagonal links) whose rotation signature is new.
    Returns (shape, family, signature) or None."""
    max_cells, max_h, max_w = caps["max_cells"], caps["max_h"], caps["max_w"]
    min_cells = max(min_cells, caps["min_cells"])
    for attempt in range(30):
        shape, fam = S.sample_shape(rng, max_cells, max_h, max_w, weights=caps["weights"], connectivity=4)
        if attempt >= 12:
            min_cells = min(min_cells, 2)   # relax the size floor rather than fail the scene
        if len(shape) < min_cells or _shape_component_count(shape, 4) != 1:
            continue
        if diag_links:
            linked = _with_diagonal_links(rng, shape, diag_links, max_h, max_w)
            if linked is None:
                continue
            shape, fam = linked, fam + "+diag"
        if need_rotations and len(_unique_rotation_variants(shape)) < 2:
            continue
        sig = _rot_key(shape)
        if sig in used_sigs:
            continue
        return shape, fam, sig
    return None


def _pair_hard_negative(rng: np.random.Generator, caps: dict, used_sigs: set, connectivity: int, link_p: float):
    """Two look-alike shapes that do NOT match: a chiral shape and its mirror image, or a
    1-2 cell near-miss perturbation. Returns (A, B, kind, sigA, sigB) or None."""
    max_h, max_w = caps["max_h"], caps["max_w"]
    if caps["max_cells"] < 5:
        return None
    for _ in range(12):
        links = 1 if (connectivity == 8 and rng.random() < link_p) else 0
        got = _pair_shape(rng, caps, used_sigs, min_cells=5, diag_links=links)
        if got is None:
            continue
        shape_a, _fam, sig_a = got
        shape_b = None
        kind = ""
        if rng.random() < 0.5:
            refl = _reflect_shape(shape_a)
            if _rot_key(refl) != sig_a and _fit_orientation(refl, max_h, max_w) is not None:
                shape_b, kind = refl, "mirror"
        if shape_b is None:
            nm = _make_near_miss(rng, shape_a, connectivity)
            if nm is not None and _fit_orientation(nm, max_h, max_w) is not None:
                shape_b, kind = nm, "near_miss"
        if shape_b is None:
            continue
        sig_b = _rot_key(shape_b)
        if sig_b == sig_a or sig_b in used_sigs:
            continue
        return shape_a, shape_b, kind, sig_a, sig_b
    return None


def _plan_pairs(rng: np.random.Generator, h: int, w: int) -> dict:
    """Draw the matched-group structure and the object count for one scene."""
    area = h * w
    short = min(h, w)
    # Object cap grows with the canvas (objects are labelled A..Z by the solver, so never
    # above 26): 8 on a short side <= 8, 12 under 400 cells, 16 under 700, 20 at 700+.
    max_obj = 8 if short <= 8 else 12 if area < 400 else 16 if area < 700 else 20
    max_obj = min(max_obj, max(2, area // 12), 26)
    # Base draw; small grids degrade the plan below (fewer objects fit), so the base rates
    # sit above the targets for >=3 pairs and below for 0 pairs.
    r = rng.random()
    if r < 0.105:
        groups: list[int] = []
    elif r < 0.355:
        groups = [2]
    elif r < 0.665:
        groups = [2, 2]
    else:
        r2 = rng.random()
        if r2 < 0.45:
            groups = [3]
        elif r2 < 0.75:
            groups = [2, 2, 2]
        elif r2 < 0.90:
            groups = [3, 2]
        else:
            groups = [2, 2, 2, 2]
    allow_full = max_obj <= 3 and rng.random() < 0.5   # tiny grids: both objects may match
    need_single = 0 if allow_full else 1
    while groups and sum(groups) + need_single > max_obj:
        if 3 in groups:
            groups[groups.index(3)] = 2
        else:
            groups.pop()
    m = sum(groups)
    strict = rng.random() < 0.85
    n_min = max(2, m + (need_single if groups else 0))
    if strict and m:
        n_min = max(n_min, int(math.ceil(m / 0.6)))
    if n_min > max_obj:
        n_min = max(2, m + (need_single if groups else 0))
    n_min = min(n_min, max_obj)
    if area < 400:
        # small canvases lean low (min of two draws) so they stay readable
        n_obj = min(int(rng.integers(n_min, max_obj + 1)), int(rng.integers(n_min, max_obj + 1)))
    else:
        # large canvases: one uniform draw, so the median object count follows the cap
        n_obj = int(rng.integers(n_min, max_obj + 1))
    return {"groups": groups, "n_obj": n_obj, "max_obj": max_obj}


def _solve_pairs(grid: np.ndarray, bg: int, by_color: bool, connectivity: int, bg_colors=None):
    """Recompute everything the answer needs from the VISIBLE grid: objects (every colour in
    ``bg_colors`` is background, isolated single cells are background), raster-order labels,
    rotation groups, the unordered pair list and the matched cell set."""
    view = _object_view(grid, bg, {bg} | (set() if bg_colors is None else set(bg_colors)))
    comps = connected_components(view, connectivity=connectivity, background=bg)
    if len(comps) < 2:
        return None
    order = sorted(range(len(comps)), key=lambda i: min(comps[i]["cells"]))
    label_of = {idx: chr(ord("A") + rank) for rank, idx in enumerate(order)}
    groups: dict = {}
    for idx, comp in enumerate(comps):
        sig = _rot_key(comp["cells"])
        key = (sig, comp["color"]) if by_color else sig
        groups.setdefault(key, []).append(idx)
    matched_groups = [g for g in groups.values() if len(g) >= 2]
    pairs: list[tuple[str, str]] = []
    matched_cells: set = set()
    for g in matched_groups:
        labels = sorted(label_of[i] for i in g)
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                pairs.append((labels[a], labels[b]))
        for i in g:
            matched_cells.update(comps[i]["cells"])
    pairs.sort()
    return {
        "comps": comps,
        "label_of": label_of,
        "matched_groups": matched_groups,
        "pairs": pairs,
        "matched_cells": matched_cells,
    }



def _validate_pairs_example(inp: np.ndarray, output, by_color: bool, connectivity: int,
                            answer_mode: str, highlight: Optional[int], bg: int,
                            texture_colors=(), named_colors=(), colour_solution=None) -> bool:
    """Independent check from the visible input alone (no construction bookkeeping).

    The scene is solved two ways that must agree: colour-based (every texture colour is
    background) and structural (only the named colours are background, then the
    isolated-cell rule), i.e. exactly what the prompt lets the solver do.  A disagreement
    is a generator bug and raises."""
    assert answer_mode == "recolor"
    if highlight is None or highlight in {int(v) for v in np.unique(inp)}:
        return False
    if inp.shape[0] * inp.shape[1] > PAIR_RECOLOR_MAX_AREA:
        return False
    by_colour = colour_solution
    if by_colour is None:
        by_colour = _solve_pairs(inp, bg, by_color, connectivity, set(texture_colors))
    structural = _solve_pairs(inp, bg, by_color, connectivity, set(named_colors))
    if by_colour is None or structural is None:
        return False
    if by_colour["matched_cells"] != structural["matched_cells"] or by_colour["pairs"] != structural["pairs"]:
        raise RuntimeError("tasks 7/8: colour-based and structural solutions disagree")
    if len(by_colour["label_of"]) > 26:
        return False
    # The prompt never names the background: it must be inferable as the most frequent
    # colour once the named texture colours and isolated cells are set aside, by a clear
    # margin and as the largest 4-connected region; and no object may be as large as it.
    largest_object = max(len(c["cells"]) for c in by_colour["comps"])
    inferred = bglib.infer_background(inp, set(named_colors), beat_size=largest_object)
    if not inferred["identifiable"] or inferred["color"] != bg:
        _REJECTS["background not identifiable"] += 1
        return False
    # With the inferred colour equal to the stated background, the structural solve above
    # IS the inferred-background solve (same background, same named colours): third reading.
    if isinstance(colour_solution, dict):
        colour_solution["background_margin"] = round(inferred["count"] / max(1, inferred["runner_up_count"]), 3)
    expected = inp.copy()
    for (r, c) in by_colour["matched_cells"]:
        expected[r, c] = highlight
    return isinstance(output, np.ndarray) and np.array_equal(expected, output)


# Palette size: near-flat over 2..8 with a mild lean toward fewer colours (the input
# then shows up to 9 colours with the background, the output 10 with the highlight).
_PALETTE_SIZE_WEIGHTS = {2: 1.20, 3: 1.20, 4: 1.15, 5: 1.10, 6: 1.05, 7: 1.00, 8: 0.95}
# Background textures (common/backgrounds.py): sparse / solid / structured shares of all
# scenes.  Sparse textures are inferred through the isolated-cell rule in the prompt;
# solid and structured ones are named.
_PAIR_TEXTURE_P = (0.16, 0.11, 0.04)
_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}
_REJECTS: Counter = Counter()   # diagnostics: why a built scene was rejected by the validator


def _palette_size(rng: np.random.Generator, lo: int) -> int:
    sizes = [k for k in _PALETTE_SIZE_WEIGHTS if k >= lo]
    wts = np.array([_PALETTE_SIZE_WEIGHTS[k] for k in sizes], dtype=float)
    return int(sizes[int(rng.choice(len(sizes), p=wts / wts.sum()))])


def _pair_texture(rng: np.random.Generator, grid: np.ndarray, bg: int, exclude, h: int, w: int):
    """Optional background texture on the free cells; returns (display, texture_info)."""
    u = float(rng.random())
    p_sparse, p_solid, p_struct = _PAIR_TEXTURE_P
    if min(h, w) < 8 or u >= p_sparse + p_solid + p_struct:
        return grid, dict(_NO_TEXTURE)
    if u < p_sparse:
        group = "sparse"
    elif u < p_sparse + p_solid or min(h, w) < 12:
        group = "solid"
    else:
        group = "structured"
    # Structured fills limited to the regular mosaic tiling (the mirrored random fill is not
    # inferable as background; nothing is named in the prompt).
    painted = bglib.apply_background(rng, grid, bg, grid != bg, group=group,
                                     kind=("mosaic" if group == "structured" else None), exclude=list(exclude))
    if painted is None:
        return grid, dict(_NO_TEXTURE)
    display, info = painted
    return display, {"texture_group": info["group"], "texture_kind": info["kind"],
                     "texture_colors": [int(c) for c in info["texture_colors"]],
                     "texture_named": bool(info["named"]), "texture_coverage": float(info["coverage"])}


def _object_view(grid: np.ndarray, bg: int, bg_colors) -> np.ndarray:
    """The visible grid reduced to objects: every colour in ``bg_colors`` becomes ``bg``,
    then every isolated single cell (no 8-neighbour of its own colour) becomes ``bg`` --
    the prompt's isolated-cell rule."""
    view = np.where(np.isin(grid, list(bg_colors)), bg, grid).astype(grid.dtype)
    iso = bglib.isolated_cell_mask(view) & (view != bg)
    return np.where(iso, bg, view).astype(grid.dtype)


def _build_pair_scene(rng: np.random.Generator, h: int, w: int, plan: dict, by_color: bool, connectivity: int,
                      shrink: float = 1.0):
    """Place the planned scene. Returns (display, bg, info) or None on any failure; ``info``
    carries the highlight colour (reserved before any texture is painted) and the texture keys."""
    groups: list[int] = list(plan["groups"])
    n_obj: int = plan["n_obj"]
    m = sum(groups)
    slots = n_obj - m

    # ---- non-matched roles -------------------------------------------------------
    roles: list[str] = []
    if by_color:
        if groups and slots >= 1 and rng.random() < 0.85:
            k = min(slots, 1 if rng.random() < 0.4 else 2)
            roles += ["mismatch_copy"] * k
            slots -= k
        elif not groups and slots >= 2 and rng.random() < 0.85:
            roles += ["mismatch_pair", "mismatch_pair"]
            slots -= 2
    if slots >= 2 and rng.random() < (0.7 if not groups else 0.45):
        roles += ["hard_a", "hard_b"]
        slots -= 2
    roles += ["single"] * slots

    grid, bg = random_background(rng, h, w, style="uniform")
    available = [c for c in range(NUM_COLORS) if c != bg]
    palette = [int(c) for c in rng.choice(available, size=_palette_size(rng, 3 if by_color else 2), replace=False)]
    if by_color:
        # Matched groups share a colour and mismatch copies reuse group colours, so a large
        # palette is only visible if enough unmatched singletons exist: add singleton roles
        # (one per otherwise unused palette colour) when the canvas and the object cap allow.
        room = min(plan["max_obj"], 26, max(2, (h * w) // 20))
        want = len(palette) + len(groups)
        extra = max(0, min(want, room) - n_obj)
        if extra:
            roles += ["single"] * extra
            n_obj += extra
    queue: list[int] = []

    def next_color() -> int:
        """Cycle through a shuffled copy of the palette so every palette colour is used
        once there are enough objects (uniform draws left most of a big palette unused)."""
        if not queue:
            queue.extend(int(palette[i]) for i in rng.permutation(len(palette)).tolist())
        return queue.pop()

    teach_idx = int(rng.integers(0, len(groups))) if groups else -1
    diag_idx = int(rng.integers(0, len(groups))) if (groups and connectivity == 8) else -1
    # per-object chance of carrying diagonal links (8-connected scenes only); task 8's
    # same-shape copies inherit links, so it uses a lower rate to stay near 1/3 of objects
    link_p = (0.07 if by_color else 0.18) if connectivity == 8 else 0.0

    used_sigs: set = set()
    specs: list[dict] = []   # {"shape", "color", "tag", "family"}
    group_colors: dict[int, int] = {}
    group_variants: dict[int, list] = {}

    # ---- matched groups ----------------------------------------------------------
    for gi, gsize in enumerate(groups):
        caps = _pair_object_caps(rng, h, w, n_obj, shrink)
        links = int(rng.choice([1, 2], p=[0.7, 0.3])) if gi == diag_idx else 0
        got = _pair_shape(rng, caps, used_sigs, min_cells=2, need_rotations=(gi == teach_idx), diag_links=links)
        if got is None:
            return None
        shape, fam, sig = got
        used_sigs.add(sig)
        variants = _unique_rotation_variants(shape)
        group_variants[gi] = variants
        if by_color:
            group_colors[gi] = next_color()
        orient: list = []
        if gi == teach_idx:
            first_two = rng.choice(len(variants), size=2, replace=False)
            orient = [variants[int(first_two[0])], variants[int(first_two[1])]]
        while len(orient) < gsize:
            orient.append(variants[int(rng.integers(0, len(variants)))])
        for k, oriented in enumerate(orient):
            color = group_colors[gi] if by_color else next_color()
            specs.append({"shape": oriented, "color": color, "tag": f"group{gi}", "family": fam,
                          "diag": links > 0})

    # ---- distractors ---------------------------------------------------------------
    hard_kind = "none"
    n_copies = 0
    used_copy: set = set()
    for role in roles:
        if role == "mismatch_copy":
            # copies of the diagonal-link group are allowed but not preferred (keeps the
            # 8-connected scenes from filling up with diagonal objects)
            gi = int(rng.integers(0, len(groups)))
            if gi == diag_idx and len(groups) > 1 and rng.random() < 0.7:
                gi = int(rng.choice([g for g in range(len(groups)) if g != diag_idx]))
            variants = group_variants[gi]
            odd_colors = [c for c in palette if c != group_colors[gi] and (gi, c) not in used_copy]
            if not odd_colors:
                odd_colors = [c for c in available if c != group_colors[gi] and (gi, c) not in used_copy]
            if not odd_colors:
                continue
            color = int(rng.choice(odd_colors))
            used_copy.add((gi, color))
            specs.append({"shape": variants[int(rng.integers(0, len(variants)))], "color": color,
                          "tag": "mismatch_copy", "family": "copy", "diag": gi == diag_idx})
            n_copies += 1
        elif role == "mismatch_pair":
            if any(s["tag"] == "mismatch_pair" for s in specs):
                continue
            caps = _pair_object_caps(rng, h, w, n_obj, shrink)
            got = _pair_shape(rng, caps, used_sigs, min_cells=2,
                              diag_links=1 if rng.random() < link_p else 0)
            if got is None:
                return None
            shape, fam, sig = got
            used_sigs.add(sig)
            variants = _unique_rotation_variants(shape)
            c1, c2 = (int(c) for c in rng.choice(palette, size=2, replace=False))
            for color in (c1, c2):
                specs.append({"shape": variants[int(rng.integers(0, len(variants)))], "color": color,
                              "tag": "mismatch_pair", "family": fam, "diag": "+diag" in fam})
            n_copies += 2
        elif role == "hard_a":
            caps = _pair_object_caps(rng, h, w, n_obj, shrink)
            caps.update(max_cells=max(caps["max_cells"], 6), max_h=min(max(caps["max_h"], 3), h),
                        max_w=min(max(caps["max_w"], 3), w))
            hp = _pair_hard_negative(rng, caps, used_sigs, connectivity, link_p)
            if hp is None:
                # no room for a look-alike pair: fall back to two ordinary singletons
                for _ in range(2):
                    got = _pair_shape(rng, _pair_object_caps(rng, h, w, n_obj, shrink), used_sigs,
                                      diag_links=1 if rng.random() < link_p else 0)
                    if got is None:
                        return None
                    shape, fam, sig = got
                    used_sigs.add(sig)
                    specs.append({"shape": shape, "color": next_color(), "tag": "single",
                                  "family": fam, "diag": "+diag" in fam})
                continue
            shape_a, shape_b, hard_kind, sig_a, sig_b = hp
            used_sigs.update({sig_a, sig_b})
            color = next_color()
            specs.append({"shape": shape_a, "color": color, "tag": "hard_negative", "family": hard_kind,
                          "diag": _shape_component_count(shape_a, 4) > 1})
            specs.append({"shape": shape_b, "color": color, "tag": "hard_negative", "family": hard_kind,
                          "diag": _shape_component_count(shape_b, 4) > 1})
        elif role == "hard_b":
            continue
        else:
            caps = _pair_object_caps(rng, h, w, n_obj, shrink)
            links = 1 if rng.random() < link_p else 0
            got = _pair_shape(rng, caps, used_sigs, diag_links=links)
            if got is None:
                return None
            shape, fam, sig = got
            used_sigs.add(sig)
            specs.append({"shape": shape, "color": next_color(), "tag": "single",
                          "family": fam, "diag": links > 0})
    if len(specs) < 2:
        return None

    # ---- placement: large first, 1-cell gap, a small budget of deliberate touches ----
    order = list(range(len(specs)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(specs[i]["shape"]), reverse=True)
    touch_target = int(rng.choice([0, 1, 2], p=[0.74, 0.20, 0.06]))
    touches = 0
    n_placed = 0
    for i in order:
        spec = specs[i]
        # group copies keep the orientation chosen above (the teach pair shows two rotations)
        oriented = spec["shape"] if spec["tag"].startswith("group") else _random_rotated_shape(rng, spec["shape"])
        fitted = _fit_orientation(oriented, h, w)
        if fitted is None:
            return None
        placed = False
        if touches < touch_target and n_placed >= 1:
            if place_shape_on_grid(grid, rng, fitted, spec["color"], bg, margin=0, max_attempts=80, require_touch=True):
                touches += 1
                placed = True
        if not placed:
            placed = place_shape_on_grid(grid, rng, fitted, spec["color"], bg, margin=0, max_attempts=110, gap=1)
        if not placed:
            return None
        n_placed += 1

    comps = connected_components(grid, connectivity=connectivity, background=bg)
    if len(comps) != len(specs):
        return None
    if any(len(c["cells"]) < 2 for c in comps):
        return None
    # Reserve the highlight colour before texturing so the texture can never take it.
    highlight = _pick_highlight_color(grid, bg, rng)
    if highlight in {int(v) for v in np.unique(grid)}:
        return None
    display, texture = _pair_texture(rng, grid, bg, list(palette) + [highlight], h, w)
    info = {
        "families": sorted({s["family"] for s in specs}),
        "object_sizes": sorted(len(s["shape"]) for s in specs),
        "diag_link_objects": int(sum(1 for s in specs if s["diag"])),
        "hard_negative": hard_kind,
        "mismatch_copies": int(n_copies),
        "touches": int(touches),
        "group_sizes": sorted(groups),
        "highlight": int(highlight),
        **texture,
    }
    return display, bg, info


def _pairs_example(rng: np.random.Generator, by_color: bool, task_label: str) -> dict:
    connectivity = int(rng.choice([4, 8]))
    answer_mode = "recolor"  # the grid->text pair-list variant was removed on review feedback
    area_cap = PAIR_RECOLOR_MAX_AREA
    h, w = _pair_grid_size(rng, area_cap)
    # The grid size and the plan are kept across failed attempts (a failure only shrinks the
    # object caps), so placement failures do not bias the size or answer distributions.
    plan = _plan_pairs(rng, h, w)
    shrink = 1.0
    for attempt in range(PAIR_ATTEMPTS):
        if attempt and attempt % 10 == 0:
            plan = _plan_pairs(rng, h, w)   # same grid, fresh plan (rare)
            shrink = 1.0
        built = _build_pair_scene(rng, h, w, plan, by_color, connectivity, shrink)
        if built is None:
            shrink = max(0.4, shrink * 0.85)
            continue
        grid, bg, info = built
        solved = _solve_pairs(grid, bg, by_color, connectivity, set(info["texture_colors"]))
        if solved is None:
            continue
        # the visible grid must realise exactly the planned matching structure
        if sorted(len(g) for g in solved["matched_groups"]) != sorted(plan["groups"]):
            continue
        comps = solved["comps"]
        if plan["groups"]:
            teach = any(
                len({tuple(sorted(normalize_shape(comps[i]["cells"]))) for i in g}) >= 2
                for g in solved["matched_groups"]
            )
            if not teach:
                continue
            if connectivity == 8:
                n_diag_groups = sum(
                    1 for g in solved["matched_groups"]
                    if any(_requires_8_connectivity(comps[i]["cells"]) for i in g)
                )
                if n_diag_groups != 1:
                    continue
        highlight = info.pop("highlight")
        meta = {
            "answer_mode": answer_mode,
            "connectivity": int(connectivity),
            "background_color": int(bg),
            "has_pair": bool(solved["pairs"]),
            "n_objects": int(len(comps)),
            "n_pairs": int(len(solved["pairs"])),
            "n_matched": int(sum(len(g) for g in solved["matched_groups"])),
            "grid_h": int(h),
            "grid_w": int(w),
            "attempts": int(attempt + 1),
            **info,
        }
        out = grid.copy()
        for (r, c) in solved["matched_cells"]:
            out[r, c] = highlight
        named = info["texture_colors"] if info["texture_named"] else []
        if not _validate_pairs_example(grid, out, by_color, connectivity, "recolor", highlight, bg,
                                       texture_colors=info["texture_colors"], named_colors=named,
                                       colour_solution=solved):
            continue
        meta["highlight_color"] = int(highlight)
        meta["background_margin"] = solved.get("background_margin")
        meta["n_colors_input"] = int(len(np.unique(grid)))
        meta["n_colors_output"] = int(len(np.unique(out)))
        return {"input": grid_to_list(grid), "output": grid_to_list(out), "metadata": meta}
    raise RuntimeError(f"Task {task_label}: failed to generate valid example")


@register("7")
def generate_find_shape_pairs(rng: np.random.Generator) -> dict:
    """Task 7: objects matching another object's shape up to rotation (colour ignored)."""
    return _pairs_example(rng, by_color=False, task_label="7")


@register("8")
def generate_find_shape_color_pairs(rng: np.random.Generator) -> dict:
    """Task 8: objects matching another object in shape (up to rotation) AND colour."""
    return _pairs_example(rng, by_color=True, task_label="8")


# ---------------------------------------------------------------------------
# Task 20 — recolor every object with exactly n cells (grid -> grid)
# Connectivity 4 or 8 drawn per example (tasks 20 and 21 were merged on 2026-09-07).
# ---------------------------------------------------------------------------

_EXACT_BANDS = ((4, 10), (11, 20), (21, 30))
_EXACT_PROBS = (0.26, 0.37, 0.37)  # ≈ width-proportional: near-flat per-value density; no area cap (1..30 per side)
EXACT_MAX_OBJECTS = 24
EXACT_MAX_SIZE = 30
EXACT_MAX_MATCHES = 12
EXACT_ATTEMPTS = 240
EXACT_BG_MARGIN = 1.25
_EXACT_TEXTURE_P = 0.30
# Object palette size 1..8 (the ground makes 9 in the input; the recolour colour is the tenth and
# must be absent from the input): near-flat with a mild lean toward fewer colours.
_EXACT_PALETTE_WEIGHTS = (0.14, 0.14, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11)
# Number of matching objects: decaying weights over 1..cap (the cap follows the footprint of an
# n-cell object, as on tasks 4/5).
_EXACT_MATCH_WEIGHTS = (0.35, 0.27, 0.16, 0.10, 0.06, 0.03, 0.015, 0.01, 0.005, 0.005, 0.003, 0.002)
# Clusters: same-colour 4-connected pieces touching only at corners (one object under 8-connectivity,
# several under 4).  Their number and piece counts are drawn WITHOUT looking at the connectivity, so
# the picture never hints which reading applies; the connectivity only decides how sizes are planned
# (under 8 a cluster is one planned object split into pieces, under 4 it is several planned objects
# placed corner to corner).
_EXACT_CLUSTER_P = (0.25, 0.40, 0.23, 0.12)   # 0, 1, 2, 3 clusters per scene (capped by the canvas)
_EXACT_CLUSTER_3_P = 0.40                     # three pieces instead of two
_EXACT_CLUSTER_PIECE_N_P = 0.40               # one piece has exactly n cells (a match under 4, a flip under 8)
_EXACT_CLUSTER_SUM_N_P = 0.40                 # the pieces sum to n (a match under 8, a flip under 4)
_EXACT_CLUSTER_PIECE_MAX = 8                  # per-piece cap and ...
_EXACT_CLUSTER_SUM_MAX = 20                   # ... cluster cap, so a cluster fits either way

_EXACT_FAMILY_WEIGHTS = {
    "polyomino": 3.0, "bar": 1.5, "rect": 1.5, "square": 0.6, "frame": 1.0, "L": 1.0,
    "T": 0.6, "U": 0.6, "plus": 0.5, "H": 0.3, "staircase": 0.5, "blob": 2.0, "scaled": 0.5,
}


def _exact_family_options(size: int, max_h: int, max_w: int) -> list[tuple[str, tuple]]:
    """Every (family, parameters) from the shape vocabulary with EXACTLY *size* cells that can
    fit in max_h x max_w (in some orientation)."""
    side = min(max_h, max_w)
    longest = max(max_h, max_w)
    opts: list[tuple[str, tuple]] = []
    if size <= 5:
        opts.append(("polyomino", ()))
    if 2 <= size <= longest:
        opts.append(("bar", (size, 1)))
    if size % 2 == 0 and size >= 4 and size // 2 <= longest and side >= 2:
        opts.append(("bar", (size // 2, 2)))
    for a in range(2, int(math.isqrt(size)) + 1):
        if size % a == 0:
            b = size // a
            if b >= 2 and a <= side and b <= longest:
                if a == b:
                    opts.append(("square", (a,)))
                else:
                    opts.append(("rect", (a, b)))
    for fh in range(3, 10):
        rest = size + 4 - 2 * fh
        if rest % 2 == 0 and rest // 2 >= 3 and fh <= longest and rest // 2 <= longest and min(fh, rest // 2) <= side:
            opts.append(("frame", (fh, rest // 2, 1)))
    for a in range(2, size):
        b = size + 1 - a
        if b >= 2 and a <= longest and b <= longest and min(a, b) <= side:
            opts.append(("L", (a, b)))
    for top in range(3, size):
        stem = size - top
        if stem >= 1 and top <= longest and stem + 1 <= longest and min(top, stem + 1) <= side:
            opts.append(("T", (top, stem)))
    for uh in range(2, size):
        uw = size + 2 - 2 * uh
        if uw >= 3 and uh <= longest and uw <= longest and min(uh, uw) <= side:
            opts.append(("U", (uh, uw)))
    if size >= 5 and (size - 1) % 4 == 0 and 2 * ((size - 1) // 4) + 1 <= side:
        opts.append(("plus", ((size - 1) // 4,)))
    for hh in range(3, size):
        hw = size + 2 - 2 * hh
        if hw >= 3 and hh <= longest and hw <= longest and min(hh, hw) <= side:
            opts.append(("H", (hh, hw)))
    for k in range(2, 7):
        if k * (k + 1) // 2 == size and k <= side:
            opts.append(("staircase", (k,)))
    if size >= 4:
        opts.append(("blob", ()))
    for k in (2, 3):
        if size % (k * k) == 0 and 1 <= size // (k * k) <= 5:
            opts.append(("scaled", (size // (k * k), k)))
    return opts


def _build_exact(rng: np.random.Generator, fam: str, params: tuple, size: int):
    if fam == "polyomino":
        return S.polyomino(rng, size)
    if fam == "bar":
        return S.bar(rng, params[0], params[1])
    if fam == "square":
        return S.rect(rng, params[0], params[0])
    if fam == "rect":
        return S.rect(rng, params[0], params[1])
    if fam == "frame":
        return S.frame(rng, params[0], params[1], params[2])
    if fam == "L":
        return S.letter_L(rng, params[0], params[1], 1)
    if fam == "T":
        return S.letter_T(rng, params[0], params[1], 1)
    if fam == "U":
        return S.letter_U(rng, params[0], params[1], 1)
    if fam == "plus":
        return S.plus(rng, params[0], 1)
    if fam == "H":
        return S.letter_H(rng, params[0], params[1], 1)
    if fam == "staircase":
        return S.staircase(rng, params[0], 1)
    if fam == "blob":
        return S.blob(rng, size, compactness=float(rng.uniform(0.8, 2.2)))
    if fam == "scaled":
        return S.scaled(rng, S.polyomino(rng, params[0]), params[1])
    return None


def _exact_shape(rng: np.random.Generator, size: int, max_h: int, max_w: int):
    """A 4-connected vocabulary shape with EXACTLY *size* cells fitting max_h x max_w.
    Returns (shape, family) or None."""
    if size <= 0:
        return None
    if size == 1:
        return frozenset({(0, 0)}), "polyomino"
    # let the shared sampler try first (keeps its natural family mix when it hits the size)
    for _ in range(3):
        shape, fam = S.sample_shape(rng, size, max_h, max_w, connectivity=4)
        if len(shape) == size and _shape_component_count(shape, 4) == 1:
            return shape, fam
    opts = _exact_family_options(size, max_h, max_w)
    if not opts:
        return None
    weights = np.array([_EXACT_FAMILY_WEIGHTS[f] for f, _ in opts], dtype=float)
    weights /= weights.sum()
    for _ in range(8):
        fam, params = opts[int(rng.choice(len(opts), p=weights))]
        shape = _build_exact(rng, fam, params, size)
        if shape is None or len(shape) != size or _shape_component_count(shape, 4) != 1:
            continue
        fitted = _fit_orientation(shape, max_h, max_w)
        if fitted is not None:
            return fitted, fam
    return None


def _cluster_kind(pieces: list[int]) -> str:
    if len(pieces) >= 3:
        return "chain"
    return "pendant" if min(pieces) == 1 else "corner_pair"


def _cluster_shape(rng: np.random.Generator, pieces: list[int], max_h: int, max_w: int):
    """One 8-connected shape made of ``len(pieces)`` 4-connected pieces of the given sizes that
    touch only at corners.  Returns (shape, kind, family) or None."""
    for _ in range(6):
        got = _exact_shape(rng, pieces[0], max_h, max_w)
        if got is None:
            return None
        cur, fams = got[0], [got[1]]
        ok = True
        for size in pieces[1:]:
            nxt = _exact_shape(rng, size, max_h, max_w)
            if nxt is None:
                return None
            joined = _attach_diagonally(rng, cur, nxt[0])
            if joined is None or len(joined) != len(cur) + size:
                ok = False
                break
            cur = joined
            fams.append(nxt[1])
        if not ok:
            continue
        fitted = _fit_orientation(cur, max_h, max_w)
        if fitted is None:
            continue
        if _shape_component_count(fitted, 4) != len(pieces) or _shape_component_count(fitted, 8) != 1:
            continue
        return fitted, _cluster_kind(pieces), "+".join(fams)
    return None


def _draw_object_size(rng: np.random.Generator, remaining: int, big_allowed: bool,
                      max_size: int = 24, large_canvas: bool = False) -> Optional[int]:
    """Skewed object size: 35% 1-3 / 35% 4-6 / 20% 7-12 / 10% 13-max, capped by the cell budget
    (on large canvases 22 / 30 / 28 / 20 so the occupancy holds up)."""
    if remaining < 1:
        return None
    buckets = [(1, 3), (4, 6), (7, 12), (13, max(13, max_size))]
    probs = [0.22, 0.30, 0.28, 0.20] if large_canvas else [0.35, 0.35, 0.20, 0.10]
    if not big_allowed:
        buckets, probs = buckets[:3], ([0.25, 0.35, 0.40] if large_canvas else [0.35, 0.35, 0.30])
    pick = int(rng.choice(len(buckets), p=np.array(probs) / sum(probs)))
    for bi in range(pick, -1, -1):
        lo, hi = buckets[bi]
        hi = min(hi, remaining)
        if hi >= lo:
            return int(rng.integers(lo, hi + 1))
    return None


def _exact_match_cap(n: int, area: int) -> int:
    """How many n-cell objects fit with one-cell moats (footprint of a compact n-cell shape),
    keeping ~45% of the canvas for matches and one slot free: clamp to 1..EXACT_MAX_MATCHES."""
    side = max(1, math.isqrt(max(1, n)))
    other = max(1, -(-n // side))
    footprint = (side + 1) * (other + 1)
    return max(1, min(EXACT_MAX_MATCHES, int(0.45 * area / footprint) - 1))


def _draw_match_count(rng: np.random.Generator, cap: int) -> int:
    w = np.array(_EXACT_MATCH_WEIGHTS[:cap], dtype=float)
    return 1 + int(rng.choice(len(w), p=w / w.sum()))


def _color_component_sizes(grid: np.ndarray, color: int, connectivity: int) -> list[int]:
    """Sizes of the connected components of one colour (any colour, background included)."""
    mask = grid == color
    if not mask.any():
        return []
    marked = np.where(mask, 1, 0).astype(np.uint8)
    comps = connected_components(marked, connectivity=connectivity, background=0)
    return [len(c["cells"]) for c in comps]


def _recolor_size_n(grid: np.ndarray, bg_colors, n: int, target: int, connectivity: int) -> np.ndarray:
    """Recolour every component of exactly n cells whose colour is not in *bg_colors*."""
    bg_colors = {int(c) for c in (bg_colors if isinstance(bg_colors, (set, list, tuple)) else [bg_colors])}
    out = grid.copy()
    work = grid.copy()
    # fold every background colour into one value so the component labeller ignores all of them
    sentinel = int(min(bg_colors))
    for c in bg_colors:
        work[grid == c] = sentinel
    for comp in connected_components(work, connectivity=connectivity, background=sentinel):
        if len(comp["cells"]) == n:
            for (r, c) in comp["cells"]:
                out[r, c] = target
    return out


def _validate_exact_size(inp: np.ndarray, out: np.ndarray, n: int, target: int, connectivity: int,
                         bg: int, texture_colors) -> Optional[str]:
    """Independent check from the displayed grid.  Returns None when the example is valid, else
    the reason.  The background must be identifiable without being named (strict plurality by
    EXACT_BG_MARGIN after isolated single cells are set aside, and the largest region); the
    literal reading (only the ground is background, every other same-colour component of any
    size is an object) and the generator's reading (texture colours are background too) must give
    the same output, which holds iff no texture component has exactly n cells; the ground never
    forms a component of exactly n cells; >= 1 match, >= 1 real non-match; the recolour colour is
    absent from the input; nothing else changes."""
    if inp.shape != out.shape:
        return "shape"
    if target in {int(v) for v in np.unique(inp)}:
        return "target present"
    info = bglib.infer_background(inp, named_colors=(), margin=EXACT_BG_MARGIN)
    if not info["identifiable"] or int(info["color"]) != int(bg):
        return "background not identifiable"
    tex = {int(c) for c in texture_colors}
    if target in tex or bg in tex:
        return "texture colour clash"
    if any(sz == n for sz in _color_component_sizes(inp, bg, connectivity)):
        return "ground component of n cells"
    for c in tex:
        if any(sz == n for sz in _color_component_sizes(inp, c, connectivity)):
            return "texture component of n cells"
    literal = _recolor_size_n(inp, {bg}, n, target, connectivity)
    generator = _recolor_size_n(inp, {bg} | tex, n, target, connectivity)
    if not np.array_equal(literal, generator):
        return "readings disagree"
    if not np.array_equal(generator, out):
        return "output mismatch"
    work = inp.copy()
    for c in tex:
        work[inp == c] = bg
    comps = connected_components(work, connectivity=connectivity, background=bg)
    n_match = sum(1 for c in comps if len(c["cells"]) == n)
    if n_match < 1 or n_match == len(comps):
        return "match count"
    return None


def _exact_texture(rng: np.random.Generator, grid: np.ndarray, bg: int, n: int, connectivity: int,
                   palette: list[int]):
    """A library texture on the free cells that keeps the task's rule: no texture component
    (under the task's connectivity) has exactly n cells, the ground stays identifiable, and no
    ground component has exactly n cells.  Sparse dots are 1-cell objects, so they are never used
    when n == 1.  Returns (display, info) or None."""
    h, w = grid.shape
    if min(h, w) < 8:
        return None
    groups = ["solid"] if n == 1 else ["sparse", "solid"]
    weights = {"solid": 1.0} if n == 1 else {"sparse": 0.6, "solid": 0.4}
    occupied = grid != bg
    for _ in range(4):
        painted = bglib.apply_background(rng, grid, bg, occupied, weights=weights, exclude=list(palette))
        if painted is None:
            return None
        cand, info = painted
        if info["group"] not in groups:
            continue
        tex = [int(c) for c in info["texture_colors"]]
        if any(any(sz == n for sz in _color_component_sizes(cand, c, connectivity)) for c in tex):
            continue
        if any(sz == n for sz in _color_component_sizes(cand, bg, connectivity)):
            continue
        bgi = bglib.infer_background(cand, named_colors=(), margin=EXACT_BG_MARGIN)
        if not bgi["identifiable"] or int(bgi["color"]) != int(bg):
            continue
        return cand, info
    return None


_EXACT_REJECTS: dict[str, int] = {}


def _reject(reason: str):
    """Count a scene rejection (diagnostics only) and return None."""
    _EXACT_REJECTS[reason] = _EXACT_REJECTS.get(reason, 0) + 1
    return None


def _exact_cluster_cap(area: int) -> int:
    """How many clusters the canvas can take (connectivity-independent)."""
    n_max = min(EXACT_MAX_OBJECTS, 2 + area // 40)
    return max(0, min(3, (n_max - 2) // 2))


def _try_exact_size_scene(rng: np.random.Generator, h: int, w: int, connectivity: int,
                          n_clusters: int = 0) -> Optional[dict]:
    area = h * w
    short = min(h, w)
    cap_side = 12 if area < 400 else 14
    max_h, max_w = min(h, cap_side), min(w, cap_side)
    max_size = 24 if area < 500 else EXACT_MAX_SIZE

    # ---- object sizes under a foreground budget that scales with the canvas -------------
    fg_hi = 0.40 if area < 50 else 0.32
    fg_lo = 0.08 if area < 150 else 0.12
    frac = float(rng.uniform(fg_lo, fg_hi))
    budget = max(3, int(round(frac * area)))
    n_max = min(EXACT_MAX_OBJECTS, 2 + area // 40)
    base_max = max(2, n_max - 3 * n_clusters)   # leave room for the cluster pieces (<= 3 each)
    n_lo = 2 if area < 200 else max(3, base_max // 2)
    n_obj = int(rng.integers(min(n_lo, base_max), base_max + 1))
    large = area >= 300
    objs: list[dict] = []
    remaining = budget
    big = 0
    for _ in range(n_obj):
        s = _draw_object_size(rng, remaining, big_allowed=big < max(2, area // 100), max_size=max_size,
                              large_canvas=large)
        if s is None:
            break
        if s >= 13:
            big += 1
        objs.append({"size": s})
        remaining -= s
    if len(objs) < 2:
        return _reject("too few objects")
    if len({o["size"] for o in objs}) < 2:
        o = objs[int(rng.integers(0, len(objs)))]
        o["size"] = o["size"] + 1 if o["size"] < max_size else o["size"] - 1

    # ---- choose n --------------------------------------------------------------------
    distinct = sorted({o["size"] for o in objs})
    r = rng.random()
    if r < 0.07:
        mode = "one"
        if 1 not in distinct:
            if len(objs) < base_max:
                objs.append({"size": 1})
            else:
                big_ones = [o for o in objs if o["size"] != 1]
                big_ones[int(rng.integers(0, len(big_ones)))]["size"] = 1
        n = 1
    elif r < 0.32 and (max(distinct) >= 6 or (area >= 36 and len(objs) < base_max)):
        mode = "large"
        if max(distinct) < 6:
            objs.append({"size": int(rng.integers(6, min(12, max(7, area // 4)) + 1))})
        n = max(o["size"] for o in objs)
    else:
        mode = "uniform"
        pool = [s for s in distinct if s != 1] or distinct   # n = 1 is its own (rarer) mode
        n = int(rng.choice(pool))
    if len({o["size"] for o in objs}) < 2:
        return _reject("one size only")

    def n_matches() -> int:
        return sum(1 for o in objs if o["size"] == n)

    def unflagged(pred) -> list[dict]:
        return [o for o in objs if pred(o) and not o.get("flag")]

    # ---- how many objects match: footprint-based cap, decaying draw -------------------------
    k_target = _draw_match_count(rng, _exact_match_cap(n, area))
    while n_matches() < k_target:
        if len(objs) < base_max:
            objs.append({"size": n})
        else:
            cands = unflagged(lambda o: o["size"] != n)
            if len(cands) < 2:
                break
            cands[int(rng.integers(0, len(cands)))]["size"] = n
    while n_matches() > k_target:
        cands = unflagged(lambda o: o["size"] == n)
        if not cands:
            break
        cands[int(rng.integers(0, len(cands)))]["size"] = n + 1 if n < max_size else n - 1

    # ---- clusters (the count is fixed per example; the pieces are drawn the same way under both
    #      connectivities and ADDED to the base objects, so the picture never hints the reading) --
    clusters: list[dict] = []   # {"members": [obj index...], "pieces": [sizes], "kind", "option"}
    state = {"slots": len(objs)}   # visible pieces (a cluster counts as k slots under both readings)

    def new_size_not_n(cap: int = max_size) -> int:
        s_ = _draw_object_size(rng, max(remaining, 6), big_allowed=False, max_size=cap, large_canvas=large)
        if s_ is None or s_ == n:
            s_ = n + 1 if n < cap else n - 1
        return int(s_)

    for _ in range(n_clusters):
        k = 3 if rng.random() < _EXACT_CLUSTER_3_P else 2
        r = rng.random()
        if r < _EXACT_CLUSTER_PIECE_N_P and n <= _EXACT_CLUSTER_PIECE_MAX:
            option = "piece_n"
        elif r < _EXACT_CLUSTER_PIECE_N_P + _EXACT_CLUSTER_SUM_N_P and k <= n <= _EXACT_CLUSTER_SUM_MAX:
            option = "sum_n"
        else:
            option = "free"
        pieces = None
        for _try in range(8):
            if option == "piece_n":
                cand = [n] + [new_size_not_n(_EXACT_CLUSTER_PIECE_MAX) for _ in range(k - 1)]
                rng.shuffle(cand)
            elif option == "sum_n":
                cand = _composition_sizes(rng, n, k)
            else:
                cand = [new_size_not_n(_EXACT_CLUSTER_PIECE_MAX) for _ in range(k)]
                if sum(cand) == n:
                    continue
            if cand is not None and sum(cand) <= _EXACT_CLUSTER_SUM_MAX and max(cand) <= max(_EXACT_CLUSTER_PIECE_MAX, n):
                pieces = [int(a) for a in cand]
                break
        if pieces is None:
            return _reject("cluster pieces")
        if state["slots"] + k > n_max:
            return _reject("cluster slots")
        state["slots"] += k
        if connectivity == 8:
            objs.append({"size": sum(pieces), "flag": "cluster", "kind": "cluster", "pieces": pieces})
            members = [len(objs) - 1]
        else:
            members = []
            for a in pieces:
                objs.append({"size": a, "flag": "cluster", "kind": "plain",
                             "cluster": len(clusters), "cluster_pos": len(members)})
                members.append(len(objs) - 1)
        clusters.append({"members": members, "pieces": pieces, "kind": _cluster_kind(pieces), "option": option})
    for o in objs:
        o.setdefault("kind", "plain")

    # ---- decoy (n +- 1) / <= 60% matched ----------------------------------------------
    if not any(o["size"] in (n - 1, n + 1) for o in objs) and rng.random() < 0.85:
        d = n + 1 if (n == 1 or rng.random() < 0.5) else n - 1
        if d > max_size:
            d = n - 1
        if state["slots"] < n_max:
            state["slots"] += 1
            objs.append({"size": d, "kind": "plain"})
        else:
            cands = unflagged(lambda o: o["size"] != n)
            if cands:
                cands[int(rng.integers(0, len(cands)))]["size"] = d
    guard = 0
    while n_matches() > 0.6 * len(objs) and guard < 12:
        guard += 1
        if state["slots"] < n_max:
            state["slots"] += 1
            objs.append({"size": new_size_not_n(), "kind": "plain"})
        else:
            cands = unflagged(lambda o: o["size"] == n)
            if not cands:
                return _reject("match share (no free match)")
            cands[int(rng.integers(0, len(cands)))]["size"] = n + 1 if n < max_size else n - 1
    if n_matches() < 1 or n_matches() == len(objs) or n_matches() > 0.6 * len(objs):
        return _reject("match share")

    # ---- shapes + colours -----------------------------------------------------------------
    for o in objs:
        if o["kind"] == "cluster":
            got = _cluster_shape(rng, o["pieces"], max_h, max_w)
            if got is None:
                return _reject("cluster shape")
            o["shape"], _kind, o["family"] = got
        else:
            got2 = _exact_shape(rng, o["size"], max_h, max_w)
            if got2 is None:
                return _reject("exact shape")
            o["shape"], o["family"] = got2
        if len(o["shape"]) != o["size"]:
            return _reject("shape size")
    grid, bg = random_background(rng, h, w, style="uniform")
    available = [c for c in range(NUM_COLORS) if c != bg]
    pw = np.array(_EXACT_PALETTE_WEIGHTS, dtype=float)
    n_colors = min(len(objs), 1 + int(rng.choice(len(pw), p=pw / pw.sum())))
    palette = [int(c) for c in rng.choice(available, size=n_colors, replace=False)]
    order_colors = [palette[i] for i in rng.permutation(len(palette)).tolist()]
    for i, o in enumerate(objs):
        o["color"] = order_colors[i % len(order_colors)]
    for cl in clusters:
        first = objs[cl["members"][0]]["color"]
        for i in cl["members"]:
            objs[i]["color"] = first

    # ---- placement ----------------------------------------------------------------------------
    order = list(range(len(objs)))
    rng.shuffle(order)
    order.sort(key=lambda i: objs[i]["size"], reverse=True)
    if connectivity == 4:
        for cl in clusters:
            order = [i for i in order if i not in cl["members"]]
            insert_at = int(rng.integers(0, len(order) + 1))
            order[insert_at:insert_at] = list(cl["members"])
    touch_scene = rng.random() < 0.12
    touched = False
    cells_of: dict[int, set] = {}
    for i in order:
        o = objs[i]
        shape = _fit_orientation(_random_rotated_shape(rng, o["shape"]), h, w)
        if shape is None:
            return _reject("shape fit")
        before = grid.copy()
        if connectivity == 4 and o.get("cluster_pos", 0) > 0:
            prev = clusters[o["cluster"]]["members"][o["cluster_pos"] - 1]
            placed = _place_corner_touching(grid, rng, shape, o["color"], bg, cells_of[prev])
            if placed is None:
                return _reject("corner placement")
            cells_of[i] = placed
            continue
        ok = False
        if touch_scene and not touched and cells_of and not o.get("flag"):
            ok = place_shape_on_grid(grid, rng, shape, o["color"], bg, margin=0, max_attempts=60, require_touch=True)
            if ok:
                touched = True
        if not ok:
            ok = place_shape_on_grid(grid, rng, shape, o["color"], bg, margin=0, max_attempts=120, gap=1)
        if not ok:
            return _reject("placement")
        cells_of[i] = _placed_cells(grid, before, o["color"])
    comps = connected_components(grid, connectivity=connectivity, background=bg)
    if sorted(len(c["cells"]) for c in comps) != sorted(o["size"] for o in objs):
        return _reject("components differ from the plan")
    if any(sz == n for sz in _color_component_sizes(grid, bg, connectivity)):
        return _reject("ground component of n")
    bgi = bglib.infer_background(grid, named_colors=(), margin=EXACT_BG_MARGIN)
    if not bgi["identifiable"] or int(bgi["color"]) != int(bg):
        return _reject("background not identifiable")

    # ---- background texture from the shared library (rule: no texture component of n cells) ----
    disp = grid
    tex_info = None
    if short >= 8 and len(palette) <= 7 and rng.random() < _EXACT_TEXTURE_P:
        got = _exact_texture(rng, grid, bg, n, connectivity, palette)
        if got is not None:
            disp, tex_info = got
    texture_colors = [int(c) for c in tex_info["texture_colors"]] if tex_info else []

    present = {int(v) for v in np.unique(disp)}
    target_pool = [c for c in range(NUM_COLORS) if c not in present]
    if not target_pool:
        return _reject("no free colour for the target")
    target = int(rng.choice(target_pool))
    out = _recolor_size_n(disp, {bg} | set(texture_colors), n, target, connectivity)
    problem = _validate_exact_size(disp, out, n, target, connectivity, bg, texture_colors)
    if problem in ("readings disagree", "output mismatch"):
        raise RuntimeError(f"task 20: validator disagreement ({problem})")
    if problem is not None:
        return _reject("validator: " + problem)
    bgi = bglib.infer_background(disp, named_colors=(), margin=EXACT_BG_MARGIN)

    other = 12 - connectivity
    bg_set = {bg} | set(texture_colors)
    differs = not np.array_equal(_recolor_size_n(disp, bg_set, n, target, other),
                                 _recolor_size_n(disp, bg_set, n, target, connectivity))
    sizes = sorted(o["size"] for o in objs)
    meta = {
        "n": int(n),
        "target_color": int(target),
        "background": int(bg),
        "background_margin": round(float(bgi["count"]) / max(1, float(bgi["runner_up_count"])), 3) if bgi.get("runner_up_count") else 99.0,
        "connectivity": int(connectivity),
        "other_connectivity": int(other),
        "differs_under_other_connectivity": bool(differs),
        "n_matches": int(n_matches()),
        "n_target_matches": int(k_target),
        "n_objects": int(len(objs)),
        "object_sizes": sizes,
        "families": sorted({o["family"] for o in objs}),
        "bg_style": tex_info["kind"] if tex_info else "uniform",
        "texture_group": tex_info["group"] if tex_info else None,
        "texture_kind": tex_info["kind"] if tex_info else None,
        "texture_colors": texture_colors,
        "texture_named": False,
        "texture_coverage": float(tex_info["coverage"]) if tex_info else 0.0,
        "accent_colors": texture_colors,
        "decoy_sizes": sorted({s for s in sizes if s in (n - 1, n + 1)}),
        "n_mode": mode,
        "grid_h": int(h),
        "grid_w": int(w),
        "fg_fraction": round(float(sum(sizes)) / area, 3),
        "palette_size": int(len({o["color"] for o in objs})),
        "n_colors": int(len(present)),
        "touch": bool(touched),
        # clusters: same-colour pieces touching only at corners (one object under 8, several under 4)
        "n_clusters": int(len(clusters)),
        "cluster_pieces": [list(map(int, c["pieces"])) for c in clusters],
        "kinds": sorted({c["kind"] for c in clusters}),
        "cluster_options": [c["option"] for c in clusters],
        "diag_pair": bool(clusters),
        "diag_link": int(len(clusters)),
    }
    return {"input": grid_to_list(disp), "output": grid_to_list(out), "metadata": meta}


def _composition_sizes(rng: np.random.Generator, total: int, k: int) -> Optional[list[int]]:
    """k integers >= 1 summing to total, in random order (None when total < k)."""
    if total < k:
        return None
    cuts = sorted(rng.choice(np.arange(1, total), size=k - 1, replace=False).tolist()) if k > 1 else []
    bounds = [0] + cuts + [total]
    parts = [int(bounds[i + 1] - bounds[i]) for i in range(k)]
    rng.shuffle(parts)
    return parts


def _exact_size_example(rng: np.random.Generator) -> dict:
    """Task 20 builder: the connectivity (4 or 8) is drawn per example and stated in the prompt.

    H and W are drawn independently from near-flat bands over 4..30 with no area cap, and the
    content is retried at the same canvas so failures do not tilt the size field."""
    h = _mixture_side(rng, _EXACT_BANDS, _EXACT_PROBS)
    w = _mixture_side(rng, _EXACT_BANDS, _EXACT_PROBS)
    connectivity = 4 if rng.random() < 0.5 else 8
    # the cluster count is drawn once, without looking at the connectivity, and realized exactly
    n_clusters = min(int(rng.choice(4, p=np.array(_EXACT_CLUSTER_P))), _exact_cluster_cap(h * w))
    for attempt in range(EXACT_ATTEMPTS):
        if attempt >= 160 and n_clusters > 0 and attempt % 40 == 0:
            n_clusters -= 1   # last resort on a cramped canvas (recorded in the metadata)
        ex = _try_exact_size_scene(rng, h, w, connectivity, n_clusters)
        if ex is not None:
            ex["metadata"]["attempts"] = int(attempt + 1)
            return ex
    raise RuntimeError("Task 20: failed to generate valid example")


@register("20")
def generate_recolor_exact_size(rng: np.random.Generator) -> dict:
    """Task 20: recolor every object with exactly n cells (connectivity 4 or 8, prompted).
    Task 21 (the 8-connected twin) was merged into this task on 2026-09-07."""
    return _exact_size_example(rng)


# ---------------------------------------------------------------------------
# Task 29 — Recolor all cells of one color (grid -> grid)
# ---------------------------------------------------------------------------

from generators.cell_operations_29_30 import generate_recolor as generate_recolor_all_of_color

register("29")(generate_recolor_all_of_color)


# ---------------------------------------------------------------------------
# Task 70 — Swap two colors globally (grid -> grid)
# ---------------------------------------------------------------------------

@register("70")
def generate_swap_colors(rng: np.random.Generator) -> dict:
    """Swap two uniformly selected present colors in a varied drawn scene."""
    from generators.grid_content_66_70 import generate_swap
    return generate_swap(rng)


# ---------------------------------------------------------------------------
# Task 84 — Remove all matching pairs of objects (grid -> grid)
# ---------------------------------------------------------------------------

@register("84")
def generate_remove_paired_objects(rng: np.random.Generator) -> dict:
    """Remove every member of each repeated rotation-shape class."""
    from generators.object_erasure_84_85 import generate_erasure
    return generate_erasure(rng, "84")


# ---------------------------------------------------------------------------
# Task 85 — Remove objects matching a description (grid -> grid)
# ---------------------------------------------------------------------------

@register("85")
def generate_remove_by_description(rng: np.random.Generator) -> dict:
    """Erase objects selected by a color or occupied-cell-count filter."""
    from generators.object_erasure_84_85 import generate_erasure
    return generate_erasure(rng, "85")
