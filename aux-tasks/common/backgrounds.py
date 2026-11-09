"""Shared background textures for scene-based tasks.

Policy (agreed with the reviewer, September 2026):

* Three groups.  SPARSE textures are isolated single cells: no two texture
  cells are 8-adjacent, in any combination of texture colours, so they can
  never form a shape.  SOLID textures are contiguous regions of one named
  colour (bands, blocks, a border, a split canvas, diagonal bands, concentric
  rings, grid lines, periodic multi-colour patterns).  STRUCTURED textures are
  busy fills (mosaics, mirrored half-patterns).
* Sparse textures are *inferred* by the solver: the task's object rule says
  that an object has at least two cells and that isolated single cells,
  whatever their colour, are background texture (``prompts.ISOLATED_CELL_RULE``).
  Solid and structured textures are *named* in the prompt
  (``prompts.background_sentence``), because a solid region cannot be told
  from an object by any short structural rule.
* Texture colours are never the background colour, never an object colour,
  never a highlight / recolour colour (callers pass those in ``exclude``), and
  by default never any colour already present in the grid.
* Textures never touch object cells: every painter receives the occupancy
  mask and paints only where it is False.
* Size rules: no texture on a short side under 8; structured fills only on a
  short side of 12 or more.
* Coverage limits, measured over the free (non-object) cells: sparse <= 20 %
  (subsampled at random, which keeps isolation), solid <= 55 % (thinned by
  widening the period / narrowing the lines so regions stay solid),
  structured <= 60 %.

Everything is numpy-only and deterministic in the supplied ``rng``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

NUM_COLORS = 10

GROUPS = ("sparse", "solid", "structured")
SPARSE_KINDS = ("random_dots", "lattice", "dotted_stripes", "sparse_checker",
                "dotted_border", "dotted_diagonal", "multi_speckle")
SOLID_KINDS = ("bands", "blocks", "border", "split", "diagonal_bands", "concentric",
               "grid_lines", "periodic")
STRUCTURED_KINDS = ("mosaic", "mirror")
KIND_GROUP = {**{k: "sparse" for k in SPARSE_KINDS}, **{k: "solid" for k in SOLID_KINDS},
              **{k: "structured" for k in STRUCTURED_KINDS}}

MIN_SHORT_SIDE = {"sparse": 8, "solid": 8, "structured": 12}
COVERAGE_LIMIT = {"sparse": 0.20, "solid": 0.55, "structured": 0.60}
DEFAULT_GROUP_WEIGHTS = {"sparse": 0.55, "solid": 0.35, "structured": 0.10}
# how many texture colours a kind uses: (min, max)
COLOR_COUNT = {**{k: (1, 1) for k in SPARSE_KINDS}, **{k: (1, 1) for k in SOLID_KINDS},
               "multi_speckle": (2, 3), "periodic": (2, 3), "mosaic": (2, 2), "mirror": (2, 2)}
MAX_THIN = 4


def _rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] (hi <= lo returns lo)."""
    return int(lo) if hi <= lo else int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------
# Structural helpers shared with the solvers
# ---------------------------------------------------------------------------

def _shifted_same(mask: np.ndarray, grid: Optional[np.ndarray] = None) -> np.ndarray:
    """For every cell, True when some 8-neighbour is also in ``mask`` (and, when
    ``grid`` is given, has the same colour)."""
    H, W = mask.shape
    out = np.zeros((H, W), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            a = mask[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
            b = mask[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)]
            hit = a & b
            if grid is not None:
                ga = grid[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
                gb = grid[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)]
                hit = hit & (ga == gb)
            out[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)] |= hit
    return out


def infer_background(grid: np.ndarray, named_colors=(), margin: float = 1.25,
                     beat_size: Optional[int] = None) -> dict:
    """Infer the background the way a solver told only "ignore the background" can.

    The named texture colours and every isolated single cell (any colour) are set aside;
    the most frequent remaining colour is the candidate.  It is *identifiable* when it is
    the unique maximum, beats the runner-up by ``margin``, and is the colour of the largest
    4-connected region among the remaining cells.  Returns a dict with ``color``,
    ``count``, ``runner_up``, ``runner_up_count``, ``largest_region_color`` and
    ``identifiable``.  The policy: prompts say "Ignoring the background" and never name
    or define it, so every generated grid must satisfy this.

    ``beat_size`` (generators): the largest object's cell count.  Every considered region
    that is not the candidate colour is an object, so the candidate is the largest region
    exactly when one of its 4-connected regions has more than ``beat_size`` cells; the
    search stops as soon as such a region is found (the common case costs a few dozen
    steps instead of a full labelling).  In this mode the named texture cells count as
    background for connectivity (they are background in the prompt's reading).
    """
    g = np.asarray(grid)
    named = set(int(c) for c in named_colors)
    consider = ~isolated_cell_mask(g)
    if named:
        consider &= ~np.isin(g, list(named))
    counts = np.bincount(g[consider].ravel(), minlength=10) if consider.any() else np.zeros(10, dtype=int)
    order = np.argsort(-counts, kind="stable")
    color, runner = int(order[0]), int(order[1])
    count, runner_count = int(counts[color]), int(counts[runner])
    if beat_size is not None:
        # Named texture cells are background as well (the prompt says so), so they bridge
        # background regions split by bands, grid lines or blocks: the region is grown over
        # the candidate colour and the named colours together.
        H, W = g.shape
        mask = (consider & (g == color)) | (np.isin(g, list(named)) if named else np.zeros_like(consider))
        seen = np.zeros((H, W), dtype=bool)
        best = 0
        found = False
        mrows = mask.tolist()
        for r in range(H):
            if found:
                break
            for c in range(W):
                if seen[r, c] or not mrows[r][c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                size = 0
                while stack:
                    a, b = stack.pop()
                    size += 1
                    if size > beat_size:
                        found = True
                        break
                    for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                        if 0 <= na < H and 0 <= nb < W and not seen[na, nb] and mrows[na][nb]:
                            seen[na, nb] = True
                            stack.append((na, nb))
                best = max(best, size)
                if found:
                    break
        identifiable = (
            count > 0
            and count > runner_count
            and count >= margin * runner_count
            and found
        )
        return {"color": color, "count": count, "runner_up": runner, "runner_up_count": runner_count,
                "largest_region_color": color if found else -1, "largest_region_size": int(best),
                "identifiable": bool(identifiable)}
    # largest 4-connected region among the considered cells (numpy label propagation:
    # every cell takes the minimum label among itself and its same-colour 4-neighbours
    # until nothing changes; region sizes are then label counts).
    H, W = g.shape
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)
    labels = np.where(consider, idx, -1)
    same_v = (g[1:, :] == g[:-1, :]) & consider[1:, :] & consider[:-1, :]
    same_h = (g[:, 1:] == g[:, :-1]) & consider[:, 1:] & consider[:, :-1]
    while True:
        new = labels.copy()
        up = np.where(same_v, np.minimum(labels[1:, :], labels[:-1, :]), labels[1:, :])
        new[1:, :] = np.where(same_v, np.minimum(new[1:, :], up), new[1:, :])
        down = np.where(same_v, np.minimum(labels[1:, :], labels[:-1, :]), labels[:-1, :])
        new[:-1, :] = np.where(same_v, np.minimum(new[:-1, :], down), new[:-1, :])
        right = np.where(same_h, np.minimum(new[:, 1:], new[:, :-1]), new[:, 1:])
        new[:, 1:] = np.where(same_h, np.minimum(new[:, 1:], right), new[:, 1:])
        left = np.where(same_h, np.minimum(new[:, 1:], new[:, :-1]), new[:, :-1])
        new[:, :-1] = np.where(same_h, np.minimum(new[:, :-1], left), new[:, :-1])
        if np.array_equal(new, labels):
            break
        labels = new
    best_size, best_color = 0, -1
    lab = labels[consider]
    if lab.size:
        uniq, sizes = np.unique(lab, return_counts=True)
        j = int(np.argmax(sizes))
        best_size = int(sizes[j])
        best_color = int(g.ravel()[int(uniq[j])])
    identifiable = (
        count > 0
        and count > runner_count
        and count >= margin * runner_count
        and best_color == color
    )
    return {"color": color, "count": count, "runner_up": runner, "runner_up_count": runner_count,
            "largest_region_color": best_color, "largest_region_size": int(best_size),
            "identifiable": bool(identifiable)}


def isolated_cell_mask(grid: np.ndarray) -> np.ndarray:
    """Cells with no 8-neighbour of the same colour (any colour).  This is the
    structural definition of sparse background texture the solvers use."""
    grid = np.asarray(grid)
    return ~_shifted_same(np.ones(grid.shape, dtype=bool), grid)


def texture_cells_isolated(display: np.ndarray, colors: Sequence[int]) -> bool:
    """True when no two cells of the given colours are 8-adjacent to each other
    (in any combination of those colours)."""
    display = np.asarray(display)
    mask = np.isin(display, list(colors))
    if not mask.any():
        return True
    return not _shifted_same(mask).any()


def pick_texture_colors(rng: np.random.Generator, n: int, exclude: Sequence[int]) -> list[int]:
    """``n`` distinct colours outside ``exclude`` (fewer when the pool is smaller)."""
    pool = [c for c in range(NUM_COLORS) if c not in set(int(x) for x in exclude)]
    n = min(n, len(pool))
    if n <= 0:
        return []
    return [int(c) for c in rng.choice(pool, size=n, replace=False)]


# ---------------------------------------------------------------------------
# Sparse kinds (boolean masks, isolation guaranteed by construction)
# ---------------------------------------------------------------------------

def _jittered_lattice(rng: np.random.Generator, H: int, W: int, block: int, jitter: int,
                      keep: float) -> np.ndarray:
    """One dot per ``block`` x ``block`` cell, jittered inside its top-left
    ``(jitter+1)``² corner; dots of neighbouring blocks are >= block - jitter apart,
    so with block - jitter >= 2 no two dots are 8-adjacent."""
    assert block - jitter >= 2
    mask = np.zeros((H, W), dtype=bool)
    nb_r = (H + block - 1) // block
    nb_c = (W + block - 1) // block
    r0 = _rint(rng, 0, block - 1)
    c0 = _rint(rng, 0, block - 1)
    for br in range(-1, nb_r + 1):
        for bc in range(-1, nb_c + 1):
            if rng.random() >= keep:
                continue
            r = r0 + br * block + _rint(rng, 0, jitter)
            c = c0 + bc * block + _rint(rng, 0, jitter)
            if 0 <= r < H and 0 <= c < W:
                mask[r, c] = True
    return mask


def sparse_mask(rng: np.random.Generator, H: int, W: int, kind: str) -> np.ndarray:
    """Isolated-cell pattern of the given sparse kind on an empty H x W canvas."""
    if kind not in SPARSE_KINDS:
        raise ValueError(kind)
    mask = np.zeros((H, W), dtype=bool)
    rr, cc = np.indices((H, W))
    if kind in ("random_dots", "multi_speckle"):
        block = _rint(rng, 3, 4)
        return _jittered_lattice(rng, H, W, block, block - 2, keep=float(rng.uniform(0.55, 0.95)))
    if kind == "lattice":
        p, q = _rint(rng, 2, 4), _rint(rng, 2, 4)
        if p == 2 and q == 2:
            if rng.random() < 0.5:
                q = 3
            else:
                p = 3
        mask[_rint(rng, 0, p - 1)::p, _rint(rng, 0, q - 1)::q] = True
        return mask
    if kind == "dotted_stripes":
        p = _rint(rng, 3, 4)
        a, b = _rint(rng, 0, p - 1), _rint(rng, 0, 1)
        if rng.random() < 0.5:
            mask[a::p, b::2] = True
        else:
            mask[b::2, a::p] = True
        return mask
    if kind == "sparse_checker":
        a, b = _rint(rng, 0, 1), _rint(rng, 0, 3)
        return ((rr + a) % 2 == 0) & ((rr + cc + b) % 4 == 0)
    if kind == "dotted_border":
        # walk the perimeter as a cycle and keep every s-th cell; drop the last
        # one when the wrap-around would make it adjacent to the first
        path = [(0, c) for c in range(W)] + [(r, W - 1) for r in range(1, H)] \
            + [(H - 1, c) for c in range(W - 2, -1, -1)] + [(r, 0) for r in range(H - 2, 0, -1)]
        s = _rint(rng, 2, 3)
        start = _rint(rng, 0, s - 1)
        kept: list[tuple[int, int]] = []
        for r, c in path[start::s]:
            # the path turns at the corners, so cells two steps apart can be
            # diagonal neighbours: keep a cell only when it is clear of every
            # kept cell (the wrap-around is covered by checking the first one too)
            if all(max(abs(r - kr), abs(c - kc)) > 1 for kr, kc in kept):
                kept.append((r, c))
        for r, c in kept:
            mask[r, c] = True
        return mask
    if kind == "dotted_diagonal":
        p = _rint(rng, 3, 5)
        k = _rint(rng, 0, p - 1)
        par = _rint(rng, 0, 1)
        diag = (cc - rr) if rng.random() < 0.5 else (cc + rr)
        return ((diag - k) % p == 0) & (rr % 2 == par)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Solid kinds (integer layers: 0 untouched, 1..k texture colour index)
# ---------------------------------------------------------------------------

def solid_layers(rng: np.random.Generator, H: int, W: int, kind: str, thin: int = 0,
                 n_colors: int = 1) -> np.ndarray:
    """Contiguous-region pattern of the given solid kind; ``thin`` (0..MAX_THIN)
    widens periods / narrows lines so the pattern covers less.  ``n_colors`` is
    the number of texture colours available (only ``periodic`` uses more than one)."""
    if kind not in SOLID_KINDS:
        raise ValueError(kind)
    rr, cc = np.indices((H, W))
    layers = np.zeros((H, W), dtype=np.int8)
    if kind == "bands":
        period = _rint(rng, 3, 8) + thin
        thick = max(1, _rint(rng, 1, period // 2) - thin)
        off = _rint(rng, 0, period - 1)
        axis = rr if rng.random() < 0.5 else cc
        layers[((axis + off) % period) < thick] = 1
    elif kind == "blocks":
        b = _rint(rng, 2, 5)
        phase = 2 + min(thin, 2)          # 2 = checkerboard, 3 / 4 = sparser block lattices
        layers[((rr // b) + (cc // b)) % phase == _rint(rng, 0, phase - 1)] = 1
    elif kind == "border":
        t = 1 if (min(H, W) < 12 or thin) else _rint(rng, 1, 2)
        layers[:t, :] = 1
        layers[-t:, :] = 1
        layers[:, :t] = 1
        layers[:, -t:] = 1
    elif kind == "split":
        frac = float(rng.uniform(0.25, 0.5)) * (0.8 ** thin)
        mode = _rint(rng, 0, 2)
        if mode == 0:
            layers[: max(1, int(round(H * frac))), :] = 1
        elif mode == 1:
            layers[:, : max(1, int(round(W * frac)))] = 1
        else:
            side = float(np.sqrt(frac))
            layers[: max(1, int(round(H * side))), : max(1, int(round(W * side)))] = 1
        if rng.random() < 0.5:
            layers = layers[::-1, :]
        if rng.random() < 0.5:
            layers = layers[:, ::-1]
        layers = np.ascontiguousarray(layers)
    elif kind == "diagonal_bands":
        period = _rint(rng, 3, 8) + thin
        thick = max(1, _rint(rng, 1, period // 2) - thin)
        off = _rint(rng, 0, period - 1)
        diag = (rr + cc) if rng.random() < 0.5 else (rr - cc + W)
        layers[((diag + off) % period) < thick] = 1
    elif kind == "concentric":
        t = 1 if min(H, W) < 12 else _rint(rng, 1, 2)
        phase = 2 + min(thin, 2)
        depth = np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc)) // t
        parity = _rint(rng, 0, min(phase - 1, int(depth.max())))
        layers[depth % phase == parity] = 1
    elif kind == "grid_lines":
        k = _rint(rng, 3, 7) + thin
        a, b = _rint(rng, 0, k - 1), _rint(rng, 0, k - 1)
        layers[(rr % k == a) | (cc % k == b)] = 1
    else:  # periodic: background stripes alternate with 2-3 texture colours
        n = max(1, min(3, int(n_colors)))
        t = _rint(rng, 1, 3)
        gap = 1 + thin                     # background stripes between texture stripes
        if rng.random() < 0.7:
            axis = rr if rng.random() < 0.5 else cc
            idx = (axis // t) % (n * (1 + gap))
            for i in range(n):
                layers[idx == i * (1 + gap)] = i + 1
        else:
            b = _rint(rng, 1, 3)
            cell = (rr // b) + (cc // b)
            idx = cell % (n * (1 + gap))
            for i in range(n):
                layers[idx == i * (1 + gap)] = i + 1
    return layers


# ---------------------------------------------------------------------------
# Structured kinds
# ---------------------------------------------------------------------------

def structured_layers(rng: np.random.Generator, H: int, W: int, kind: str, thin: int = 0) -> np.ndarray:
    """Busy two-colour fills; ``thin`` raises the share of untouched cells."""
    if kind not in STRUCTURED_KINDS:
        raise ValueError(kind)
    p0 = min(0.9, 0.45 + 0.1 * thin)
    probs = np.array([p0, (1 - p0) * 0.55, (1 - p0) * 0.45], dtype=float)
    if kind == "mosaic":
        t = _rint(rng, 2, 3)
        nb_r, nb_c = (H + t - 1) // t, (W + t - 1) // t
        tiles = rng.choice(3, size=(nb_r, nb_c), p=probs).astype(np.int8)
        return np.ascontiguousarray(np.repeat(np.repeat(tiles, t, axis=0), t, axis=1)[:H, :W])
    # mirror
    mode = ("lr", "ud", "both")[_rint(rng, 0, 2)]
    hh = (H + 1) // 2 if mode in ("ud", "both") else H
    ww = (W + 1) // 2 if mode in ("lr", "both") else W
    part = rng.choice(3, size=(hh, ww), p=probs).astype(np.int8)
    if mode in ("ud", "both"):
        part = np.concatenate([part, part[: H - hh][::-1]], axis=0)
    if mode in ("lr", "both"):
        part = np.concatenate([part, part[:, : W - ww][:, ::-1]], axis=1)
    return np.ascontiguousarray(part)


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------

def allowed_groups(H: int, W: int) -> list[str]:
    short = min(H, W)
    return [g for g in GROUPS if short >= MIN_SHORT_SIDE[g]]


def _layers_for(rng: np.random.Generator, H: int, W: int, kind: str, n_colors: int, thin: int) -> np.ndarray:
    group = KIND_GROUP[kind]
    if group == "sparse":
        mask = sparse_mask(rng, H, W, kind)
        layers = mask.astype(np.int8)
        if kind == "multi_speckle" and n_colors > 1:
            idx = np.argwhere(mask)
            layers[idx[:, 0], idx[:, 1]] = rng.integers(1, n_colors + 1, size=len(idx)).astype(np.int8)
        return layers
    if group == "solid":
        return solid_layers(rng, H, W, kind, thin, n_colors)
    return structured_layers(rng, H, W, kind, thin)


def apply_background(
    rng: np.random.Generator,
    grid: np.ndarray,
    bg: int,
    occupied: np.ndarray,
    *,
    group: Optional[str] = None,
    kind: Optional[str] = None,
    colors: Optional[Sequence[int]] = None,
    exclude: Sequence[int] = (),
    weights: Optional[dict] = None,
) -> Optional[tuple[np.ndarray, dict]]:
    """Paint one background texture on the free cells of ``grid``.

    Returns ``(display, info)`` or ``None`` when nothing could be painted (size
    rule, no free colour, empty pattern); the caller then keeps the plain
    background.  ``info`` = {"group", "kind", "texture_colors", "coverage",
    "named"}.  Colours come from the free pool (not ``bg``, not ``exclude``,
    not present in ``grid``) unless ``colors`` is given explicitly.
    """
    grid = np.asarray(grid)
    occupied = np.asarray(occupied, dtype=bool)
    H, W = grid.shape
    free = ~occupied
    if not free.any():
        return None
    groups = allowed_groups(H, W)
    if kind is not None:
        group = KIND_GROUP[kind]
    if group is not None and group not in groups:
        return None
    if not groups:
        return None
    if group is None:
        wts = dict(DEFAULT_GROUP_WEIGHTS if weights is None else weights)
        names = [g for g in groups if wts.get(g, 0.0) > 0]
        if not names:
            return None
        p = np.array([wts[g] for g in names], dtype=float)
        group = names[int(rng.choice(len(names), p=p / p.sum()))]
    if kind is None:
        kinds = {"sparse": SPARSE_KINDS, "solid": SOLID_KINDS, "structured": STRUCTURED_KINDS}[group]
        kind = kinds[int(rng.integers(0, len(kinds)))]
    lo, hi = COLOR_COUNT[kind]
    if colors is None:
        present = {int(v) for v in np.unique(grid)}
        pool_exclude = present | {int(bg)} | {int(c) for c in exclude}
        want = _rint(rng, lo, hi)
        chosen = pick_texture_colors(rng, want, sorted(pool_exclude))
    else:
        chosen = [int(c) for c in colors]
        if int(bg) in chosen or len(set(chosen)) != len(chosen):
            return None
    if len(chosen) < lo:
        return None
    n_colors = len(chosen)
    limit = COVERAGE_LIMIT[group]
    n_free = int(free.sum())
    layers = None
    for thin in range(MAX_THIN + 1):
        cand = _layers_for(rng, H, W, kind, n_colors, thin)
        cand = np.where(free, cand, 0).astype(np.int8)
        if group != "sparse":
            # a solid / structured texture never leaves isolated fragments (a band clipped
            # at a corner, a ring reduced to one cell, a cell cut off by an object): drop
            # every texture cell without a same-colour 8-neighbour, so the solid property
            # holds by construction and no solid kind ever looks like sparse dots
            keep = _shifted_same(cand > 0, cand)
            cand = np.where(keep, cand, 0).astype(np.int8)
        n_tex = int((cand > 0).sum())
        if n_tex == 0:
            continue
        if n_tex / n_free <= limit:
            layers = cand
            break
        if group == "sparse":
            # random subsampling keeps isolation
            idx = np.argwhere(cand > 0)
            keep_idx = rng.choice(len(idx), size=int(limit * n_free), replace=False)
            sub = np.zeros_like(cand)
            sub[idx[keep_idx, 0], idx[keep_idx, 1]] = cand[idx[keep_idx, 0], idx[keep_idx, 1]]
            layers = sub
            break
    if layers is None or not (layers > 0).any():
        return None
    display = grid.copy()
    for i, col in enumerate(chosen, start=1):
        display[layers == i] = col
    # validation
    if not np.array_equal(display[occupied], grid[occupied]):
        return None
    if group == "sparse" and not texture_cells_isolated(display, chosen):
        return None
    used = sorted({int(i) for i in np.unique(layers) if i > 0})
    info = {
        "group": group,
        "kind": kind,
        "texture_colors": [chosen[i - 1] for i in used],
        "coverage": float((layers > 0).sum() / n_free),
        "named": group != "sparse",
    }
    return display, info
