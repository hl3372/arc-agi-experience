"""Largest-object family: tasks 17 and 18 (recolor the largest object).

* 17 -- largest bounding-box area, connectivity 4 or 8 (stated in the prompt)
* 18 -- most cells, connectivity 4 or 8 (stated in the prompt; tasks 18 and 19 were
        merged on 2026-09-07, the connectivity is now drawn per example like 17's)

Design (see the investigation brief for the measured problems this fixes):

* Envelope: H and W drawn independently and uniformly over [5, 30] (any
  canvas up to 30x30; squares only by coincidence; no bundle-derived cap).
  Object count, per-object size and the winner target scale with the area.
* Objects come from ``common.shapes`` (rectangles, bars, frames, letters,
  polyominoes, blobs, one big irregular blob in some large scenes; diagonal
  chains whose pieces have at least two cells each).  Every object is placed with a clear
  1-cell ring, so nothing ever touches by accident.  The only contacts are
  deliberate features: same-colour pieces that touch diagonally (separate
  under 4-conn, one object under 8-conn) and, in a few scenes, a small
  object of a different colour pressed against another one.
* The winner is designated first and everything else is built under its
  score minus a margin of at least 2, so the answer is unique by
  construction -- and then re-derived by an independent validator from the
  emitted grid (components under the prompted connectivity, background from
  the metadata, strict-plurality background, output == input with exactly the
  winner recoloured to a colour absent from the input).
* "The rule matters": task 18's chain scenes hold a diagonally bridged chain
  and a solid object sized so that the chain's union beats the solid while
  every piece loses to it -- the 8-conn reading recolours the chain, the
  4-conn reading the solid.  The scene is drawn without looking at the
  connectivity (same recipes, same rates), so nothing in the picture hints
  which reading applies; only the prompt does.  Task 17 pairs a sparse winner
  (frame / letter / chain) with a denser rival that has more cells but a
  smaller box.
* Backgrounds: plain, or a SPARSE texture from ``common.backgrounds`` (isolated
  single cells only: random dots, lattice, dotted stripes, sparse checker, dotted
  border / diagonal, multi-colour speckle) on grids with short side >= 8.  Nothing
  about the background is named in the prompt: the object rule says every object
  has at least two cells and isolated single cells, whatever their colour, are
  background texture.  A texture is kept only if the winner still wins by >= 2
  under BOTH readings (every texture cell counted as an object; isolated cells
  set aside) and the background stays identifiable (strict plurality by 1.25x
  after isolated cells are set aside, and the largest 4-connected region), so
  the answer never depends on how the solver treats the texture.
* Colours: a near-flat palette of 2..8 object colours (the input shows up to 9
  colours with the background; the output reaches 10 with the recolour target,
  which must be absent from the input), assigned round-robin so a big scene
  really uses them.

Tasks 22, 36 and 37 (topology) live in ``generators/topology_tasks.py``.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from common.backgrounds import apply_background, infer_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list
from common.shapes import (
    bar,
    big_blob,
    blob,
    diagonal,
    frame,
    letter_L,
    letter_T,
    letter_U,
    plus,
    polyomino,
    rect,
    sample_shape,
    shape_size,
    staircase,
)
from common.topology import dilate4, dilate8, label_components
from registry import register

Shape = frozenset

MAX_ATTEMPTS = 120
MIN_SIDE, MAX_SIDE = 5, 30
MIN_AREA, MAX_AREA = MIN_SIDE * MIN_SIDE, MAX_SIDE * MAX_SIDE   # no bundle-derived cap: 30x30 is reachable
TEXTURE_MIN_SIDE = 8
BG_MARGIN = 1.25              # identifiability: background count >= 1.25x every other colour

# Per-object shape mixture (rects 20 / bars 10 / frames 10 / letters 15 /
# named polyominoes 15 / blobs 25).  "diagonal" is added under 8-conn only;
# big blobs are placed explicitly (one per scene, some large scenes).
_BASE_WEIGHTS: dict[str, float] = {
    "polyomino": 0.13, "scaled": 0.02,
    "bar": 0.10,
    "rect": 0.16, "square": 0.05,
    "frame": 0.13,
    "L": 0.06, "T": 0.04, "U": 0.04, "plus": 0.03, "H": 0.02, "staircase": 0.02,
    "blob": 0.20,
}
# Families whose builders can hit an exact cell count (see _exact_shape).
_EXACT_FAMILIES = ("polyomino", "bar", "rect", "L", "blob", "frame", "plus", "T", "U", "staircase")
# Sparse / elongated families favoured for the task-17 winner (box >> cells).
_SPARSE_WEIGHTS: dict[str, float] = {
    "frame": 0.20, "L": 0.16, "T": 0.09, "U": 0.09, "plus": 0.07, "H": 0.04,
    "staircase": 0.04, "bar": 0.08, "polyomino": 0.08, "blob": 0.08,
    "rect": 0.04, "square": 0.01, "scaled": 0.02,
}
_THEMES = (
    ("rect", "square"), ("bar",), ("frame",), ("L", "T", "U", "plus", "H", "staircase"),
    ("polyomino", "scaled"), ("blob",),
)

_ROLE_TOUCH_P = 0.10          # different-colour touching feature (<= 15% of scenes)
_TEXTURE_P = 0.45             # texture attempt probability on eligible grids (sparse kinds only)
_N_COLORS = (2, 3, 4, 5, 6, 7, 8)
_N_COLORS_P = (0.13, 0.14, 0.14, 0.14, 0.15, 0.15, 0.15)   # flat (the visible count still leans lower on small canvases)


# ---------------------------------------------------------------------------
# Scores and component analysis (used by the validator and the metrics)
# ---------------------------------------------------------------------------

def _cells_score(cells: frozenset) -> int:
    return len(cells)


def _bbox_hw(cells) -> tuple[int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _bbox_score(cells: frozenset) -> int:
    h, w = _bbox_hw(cells)
    return h * w


def _components(grid: np.ndarray, conn: int, bg: int) -> list[tuple[int, frozenset]]:
    """Every same-colour component of the grid under ``conn`` (bg excluded)."""
    out: list[tuple[int, frozenset]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color == bg:
            continue
        labels, n = label_components(grid == color, conn)
        for k in range(1, n + 1):
            cells = frozenset((int(r), int(c)) for r, c in np.argwhere(labels == k))
            out.append((color, cells))
    return out


def _ranked(comps, score: Callable[[frozenset], int]):
    """Components sorted by score (desc), deterministic tie-break by position."""
    return sorted(comps, key=lambda cc: (-score(cc[1]), min(cc[1])))


def _unique_top(comps, score, min_margin: int = 2) -> Optional[tuple[frozenset, int, int]]:
    """(winner cells, winner score, runner-up score) or None if no clear winner."""
    ranked = _ranked(comps, score)
    if not ranked:
        return None
    top = score(ranked[0][1])
    second = score(ranked[1][1]) if len(ranked) > 1 else 0
    if len(ranked) > 1 and top - second < min_margin:
        return None
    return ranked[0][1], top, second


def _top_cells(comps, score) -> Optional[frozenset]:
    """Cells of the unique best component, or None if the top is tied."""
    ranked = _ranked(comps, score)
    if not ranked:
        return None
    if len(ranked) > 1 and score(ranked[0][1]) == score(ranked[1][1]):
        return None
    return ranked[0][1]


def _bg_is_strict_plurality(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    cmax = int(counts.max())
    return int(vals[int(counts.argmax())]) == bg and int((counts == cmax).sum()) == 1


def _plain_view(disp: np.ndarray, bg: int) -> np.ndarray:
    """The structural reading of the prompt: isolated single cells (any colour) are
    background texture, so they are set aside before objects are formed."""
    view = disp.copy()
    view[isolated_cell_mask(disp) & (disp != bg)] = bg
    return view


def _validate(disp: np.ndarray, out: np.ndarray, bg: int, conn: int, target: int, score) -> bool:
    """Independent check on the EMITTED grids: the answer must follow from the
    visible input and the prompt alone (no construction bookkeeping).

    Two readings must agree: (a) every non-background cell is an object cell
    (a solver that ignores the isolated-cell clause), (b) isolated single cells
    are set aside first (the prompt's reading).  Under (b) every object has at
    least two cells, and the background is identifiable without being named."""
    if disp.shape != out.shape or disp.ndim != 2:
        return False
    if int(disp.min()) < 0 or int(disp.max()) >= NUM_COLORS or int(out.min()) < 0 or int(out.max()) >= NUM_COLORS:
        return False
    if target in set(int(v) for v in np.unique(disp)):
        return False
    if not _bg_is_strict_plurality(disp, bg):
        return False
    changed = frozenset((int(r), int(c)) for r, c in np.argwhere(out != disp))
    for view in (disp, _plain_view(disp, bg)):
        comps = _components(view, conn, bg)
        top = _unique_top(comps, score, min_margin=2)
        if top is None or top[0] != changed:
            return False
    # every object in the prompt's reading has at least two cells
    if any(len(cells) < 2 for _, cells in _components(_plain_view(disp, bg), conn, bg)):
        return False
    info = infer_background(disp, margin=BG_MARGIN, beat_size=len(changed))
    if not info["identifiable"] or int(info["color"]) != int(bg):
        return False
    return all(int(out[r, c]) == target for r, c in changed)


# ---------------------------------------------------------------------------
# Envelope, budget, mixtures
# ---------------------------------------------------------------------------

def _envelope(rng: np.random.Generator) -> tuple[int, int]:
    """(H, W): each side uniform over [5, 30], drawn independently (near-flat per-value
    density, squares only by coincidence, 30x30 reachable, no bundle-derived cap)."""
    return int(rng.integers(MIN_SIDE, MAX_SIDE + 1)), int(rng.integers(MIN_SIDE, MAX_SIDE + 1))


def _budget(rng: np.random.Generator, h: int, w: int, n_colors: int = 0) -> tuple[int, int, int, float]:
    """(n_objects, per-object max cells, winner cell target, density target).

    Object count and per-object size grow with the canvas: a 30x30 grid holds up
    to 16 objects of up to 60 cells, a 6x6 grid two to four small ones.  The count
    is at least the palette size where the canvas allows it, so a large palette
    is actually seen (colours are assigned round-robin)."""
    area = h * w
    n_max = min(16, 2 + area // 40)
    if rng.random() < 0.5:
        n = int(rng.integers(2, n_max + 1))
    else:
        n = int(rng.integers(max(2, (n_max + 2) // 2), n_max + 1))
    n = max(n, min(n_max, n_colors))
    density = float(rng.uniform(0.12, 0.30))
    max_cells = min(max(area // 10, 6), 60)
    winner = int(round(2.0 * density * area / (n + 1) * float(rng.uniform(0.8, 1.2))))
    winner = min(max(winner, 4), max_cells)
    return n, max_cells, winner, density


def _scene_weights(rng: np.random.Generator, conn: int, base: Optional[dict] = None) -> dict:
    w = dict(_BASE_WEIGHTS if base is None else base)
    if conn == 8:
        w["diagonal"] = 0.06
    if rng.random() < 0.30:
        theme = _THEMES[int(rng.integers(0, len(_THEMES)))]
        for fam in theme:
            if fam in w:
                w[fam] *= 3.0
    return w


def _runner_cap(rng: np.random.Generator, winner: int) -> tuple[int, str]:
    """Cap for every other object's score: tight (2-3 below) or loose (<= winner/1.25)."""
    if rng.random() < 0.30:
        cap = winner - int(rng.integers(2, 4))
        mode = "tight"
    else:
        cap = min(winner * 4 // 5, winner - 2)
        mode = "loose"
    return max(cap, 2), mode


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _shape_near(rng, target: int, max_h: int, max_w: int, weights: dict, conn: int,
                min_frac: float = 0.6, tries: int = 8, min_cells: int = 2) -> tuple[Shape, str]:
    """A vocabulary shape with <= target cells, preferably >= min_frac*target."""
    best: Optional[tuple[Shape, str]] = None
    for _ in range(tries):
        shape, fam = sample_shape(rng, target, max_h, max_w, weights=weights, connectivity=conn)
        if len(shape) >= max(min_cells, min_frac * target):
            return shape, fam
        if len(shape) >= min_cells and (best is None or len(shape) > len(best[0])):
            best = (shape, fam)
    if best is None:
        return sample_shape(rng, target, max_h, max_w, weights=weights, connectivity=conn)
    return best


def _exact_shape(rng, size: int, max_h: int, max_w: int, tries: int = 12,
                 families: Optional[tuple] = None) -> Optional[tuple[Shape, str]]:
    """A 4-connected vocabulary shape with exactly ``size`` cells fitting max_h x max_w."""
    if size <= 0:
        return None
    if size == 1:
        return (frozenset({(0, 0)}), "polyomino") if families is None or "polyomino" in families else None
    side = min(max_h, max_w)
    longest = max(max_h, max_w)
    for _ in range(tries):
        opts: list[str] = []
        if size <= 5:
            opts.append("polyomino")
        if size <= longest:
            opts.append("bar")
        if any(size % a == 0 and a <= max_h and size // a <= max_w for a in range(2, size)):
            opts.append("rect")
        if 3 <= size <= 2 * side - 1:
            opts.append("L")
        if size >= 4:
            opts.append("blob")
        if size >= 8 and size % 2 == 0 and 6 <= size // 2 + 2 <= 2 * side:
            opts.append("frame")
        if size >= 5 and (size - 1) % 4 == 0 and 2 * ((size - 1) // 4) + 1 <= side:
            opts.append("plus")
        if 5 <= size <= min(side, 7) + min(side, 6):
            opts.append("T")
        if size >= 5 and any(3 <= size + 2 - 2 * hh <= min(side, 7) for hh in range(2, min(side, 6) + 1)):
            opts.append("U")
        steps = int((math.isqrt(8 * size + 1) - 1) // 2)
        if size >= 3 and steps * (steps + 1) // 2 == size and steps <= side:
            opts.append("staircase")
        if families is not None:
            opts = [o for o in opts if o in families]
        if not opts:
            return None
        fam = opts[int(rng.integers(0, len(opts)))]
        shape: Optional[Shape] = None
        if fam == "polyomino":
            shape = polyomino(rng, size)
        elif fam == "bar":
            shape = bar(rng, size, 1)
        elif fam == "rect":
            pairs = [(a, size // a) for a in range(2, size) if size % a == 0 and a <= max_h and size // a <= max_w]
            a, b = pairs[int(rng.integers(0, len(pairs)))]
            shape = rect(rng, a, b)
        elif fam == "L":
            a = int(rng.integers(max(2, size + 1 - side), min(side, size - 1) + 1))
            shape = letter_L(rng, a, size + 1 - a, 1)
        elif fam == "blob":
            shape = blob(rng, size, compactness=float(rng.uniform(0.6, 2.0)))
        elif fam == "frame":
            total = size // 2 + 2
            hs = [hh for hh in range(3, side + 1) if 3 <= total - hh <= side]
            hh = hs[int(rng.integers(0, len(hs)))]
            shape = frame(rng, hh, total - hh, 1)
        elif fam == "plus":
            shape = plus(rng, (size - 1) // 4, 1)
        elif fam == "T":
            tops = [t for t in range(3, min(side, 7) + 1) if 2 <= size - t <= min(side, 6)]
            top = tops[int(rng.integers(0, len(tops)))]
            shape = letter_T(rng, top, size - top, 1)
        elif fam == "U":
            hs = [hh for hh in range(2, min(side, 6) + 1) if 3 <= size + 2 - 2 * hh <= min(side, 7)]
            hh = hs[int(rng.integers(0, len(hs)))]
            shape = letter_U(rng, hh, size + 2 - 2 * hh, 1)
        elif fam == "staircase":
            shape = staircase(rng, steps, 1)
        if shape is None or len(shape) != size:
            continue
        sh, sw = shape_size(shape)
        if sh <= max_h and sw <= max_w:
            return shape, ("rect" if fam == "rect" and sh == sw else fam)
    return None


def _family_shape(rng, target: int, max_h: int, max_w: int, weights: dict, conn: int,
                  min_cells: int = 4) -> tuple[Shape, str]:
    """Family-first draw: pick a family by weight, then build it at (or near)
    ``target`` cells.  Keeps the kind distribution flat instead of letting the
    families that happen to fit any size (blobs) win every large draw."""
    names = [k for k, v in weights.items() if v > 0 and (k != "diagonal" or conn == 8)]
    if not names:
        return _shape_near(rng, target, max_h, max_w, weights, conn, min_cells=min_cells)
    probs = np.array([weights[k] for k in names], dtype=float)
    probs /= probs.sum()
    for _ in range(6):
        fam = names[int(rng.choice(len(names), p=probs))]
        if fam in _EXACT_FAMILIES and rng.random() < 0.75:
            got = _exact_shape(rng, target, max_h, max_w, families=(fam,))
            if got is not None and len(got[0]) >= min_cells:
                return got
        shape, f2 = sample_shape(rng, target, max_h, max_w, weights={fam: 1.0}, connectivity=conn)
        if f2 == fam and len(shape) >= max(min_cells, 0.6 * target):
            return shape, f2
    return _shape_near(rng, target, max_h, max_w, weights, conn, min_cells=min_cells)


def _dense_rival(rng, min_cells: int, max_cells: int, max_bbox: int, max_h: int, max_w: int,
                 allow_big: bool) -> Optional[tuple[Shape, str]]:
    """A compact shape with cells in [min_cells, max_cells] and box area <= max_bbox."""
    hi = min(max_cells, max_bbox)
    if hi < min_cells:
        return None
    for _ in range(12):
        opts = ["rect", "rect", "blob", "blob"]
        if allow_big and hi >= 12:
            opts.append("big_blob")
        fam = opts[int(rng.integers(0, len(opts)))]
        if fam == "rect":
            pairs = [(a, b) for a in range(2, max_h + 1) for b in range(2, max_w + 1)
                     if min_cells <= a * b <= hi]
            if not pairs:  # only thin lines fit
                pairs = [(a, b) for a in range(1, max_h + 1) for b in range(1, max_w + 1)
                         if min_cells <= a * b <= hi and not (a == 1 and b == 1)]
            if not pairs:
                continue
            a, b = pairs[int(rng.integers(0, len(pairs)))]
            shape: Shape = rect(rng, a, b)
            fam = "square" if a == b else ("bar" if min(a, b) == 1 else "rect")
        elif fam == "blob":
            size = int(rng.integers(min_cells, hi + 1))
            shape = blob(rng, size, compactness=float(rng.uniform(1.0, 2.2)))
        else:
            size = int(rng.integers(max(12, min_cells), hi + 1))
            shape = big_blob(rng, size)
        n = len(shape)
        sh, sw = shape_size(shape)
        if min_cells <= n <= hi and sh * sw <= max_bbox and sh <= max_h and sw <= max_w:
            return shape, fam
    return None


def _big_blob_shape(rng, s_w: int, max_h: int, max_w: int) -> Optional[Shape]:
    """The one large irregular blob (15-40 cells) some big scenes get."""
    for _ in range(6):
        shape = big_blob(rng, int(rng.integers(15, min(40, max(15, s_w)) + 1)))
        sh, sw = shape_size(shape)
        if 15 <= len(shape) <= 40 and sh <= max_h and sw <= max_w:
            return shape
    return None


def _positions(shape: Shape, allowed: np.ndarray, touch: Optional[np.ndarray] = None) -> np.ndarray:
    """Top-left offsets where every shape cell lands on ``allowed`` (and, if
    given, at least one lands on ``touch``)."""
    H, W = allowed.shape
    sh, sw = shape_size(shape)
    nh, nw = H - sh + 1, W - sw + 1
    if nh <= 0 or nw <= 0:
        return np.zeros((0, 2), dtype=int)
    ok = np.ones((nh, nw), dtype=bool)
    for dr, dc in shape:
        ok &= allowed[dr:dr + nh, dc:dc + nw]
    if touch is not None:
        t = np.zeros((nh, nw), dtype=bool)
        for dr, dc in shape:
            t |= touch[dr:dr + nh, dc:dc + nw]
        ok &= t
    return np.argwhere(ok)


def _normalize_group(pieces: list[set]) -> tuple[Shape, list[Shape]]:
    allc = [cell for p in pieces for cell in p]
    r0 = min(r for r, _ in allc)
    c0 = min(c for _, c in allc)
    norm = [frozenset((r - r0, c - c0) for r, c in p) for p in pieces]
    return frozenset(cell for p in norm for cell in p), norm


def _diag_chain(rng, sizes: list[int], max_piece_dim: int) -> Optional[tuple[Shape, list[Shape]]]:
    """Pieces of the given cell counts, each touching the previous one only
    diagonally: one 8-connected object made of len(sizes) 4-connected parts."""
    C = 8 * max_piece_dim + 4 * len(sizes) + 6
    canvas = np.zeros((C, C), dtype=bool)
    pieces: list[set] = []
    for i, size in enumerate(sizes):
        got = _exact_shape(rng, size, max_piece_dim, max_piece_dim)
        if got is None:
            return None
        shape = got[0]
        if i == 0:
            r0 = c0 = C // 2
            pos = [(r0, c0)]
        else:
            last = np.zeros((C, C), dtype=bool)
            for r, c in pieces[-1]:
                last[r, c] = True
            earlier = canvas & ~last
            allowed = ~dilate8(earlier) & ~dilate4(last)
            touch = dilate8(last) & ~dilate4(last)
            pos = _positions(shape, allowed, touch)
            if len(pos) == 0:
                return None
            pos = [tuple(int(v) for v in pos[int(rng.integers(0, len(pos)))])]
        r0, c0 = pos[0]
        cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
        for r, c in cells:
            canvas[r, c] = True
        pieces.append(cells)
    union, norm = _normalize_group(pieces)
    # sanity: the union is one 8-component split into len(sizes) 4-components
    sh, sw = shape_size(union)
    m = np.zeros((sh, sw), dtype=bool)
    for r, c in union:
        m[r, c] = True
    if label_components(m, 8)[1] != 1 or label_components(m, 4)[1] != len(sizes):
        return None
    return union, norm


def _composition(rng, total: int, k: int, lo: int, hi: int) -> Optional[list[int]]:
    """k integers in [lo, hi] summing to total (random), or None.

    Pieces of at least 2 cells are preferred whenever the total allows it, so
    a chain reads as a few parts touching at corners rather than as scatter."""
    if lo < 2 <= hi and 2 * k <= total:
        lo = 2
    if k * lo > total or k * hi < total:
        return None
    sizes = [lo] * k
    rem = total - k * lo
    order = list(rng.permutation(k))
    for idx in order:
        add = int(rng.integers(0, min(hi - lo, rem) + 1)) if rem > 0 else 0
        sizes[idx] += add
        rem -= add
    # distribute whatever is left greedily
    for idx in order:
        if rem <= 0:
            break
        room = hi - sizes[idx]
        take = min(room, rem)
        sizes[idx] += take
        rem -= take
    return sizes if rem == 0 else None


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class _Scene:
    def __init__(self, rng: np.random.Generator, h: int, w: int, bg: int):
        self.rng = rng
        self.h, self.w, self.bg = h, w, bg
        self.grid = np.full((h, w), bg, dtype=np.uint8)
        self.occ = np.zeros((h, w), dtype=bool)
        self.objects: list[dict] = []

    def place(self, shape: Shape, color: int, kind: str, role: str,
              pieces: Optional[list[Shape]] = None, touch_other: bool = False) -> Optional[dict]:
        rng = self.rng
        if touch_other:
            same = self.grid == color
            allowed = ~self.occ & ~dilate8(same)
            touch = dilate8(self.occ & ~same) & ~self.occ
        else:
            allowed = ~dilate8(self.occ)
            touch = None
        pos = _positions(shape, allowed, touch)
        if len(pos) == 0:
            return None
        r0, c0 = (int(v) for v in pos[int(rng.integers(0, len(pos)))])
        cells = frozenset((r0 + dr, c0 + dc) for dr, dc in shape)
        for r, c in cells:
            self.grid[r, c] = color
            self.occ[r, c] = True
        obj = {
            "cells": cells, "color": color, "kind": kind, "role": role,
            "pieces": [frozenset((r0 + dr, c0 + dc) for dr, dc in p) for p in pieces] if pieces else None,
        }
        self.objects.append(obj)
        return obj

    def density(self) -> float:
        return float(self.occ.sum()) / float(self.h * self.w)


def _pick_colors(rng: np.random.Generator) -> tuple[int, list[int]]:
    """Background + a near-flat palette of 2..8 object colours (one colour is always
    left free for the recolour target; a texture takes another when painted)."""
    bg = int(rng.integers(0, NUM_COLORS))
    n_colors = int(rng.choice(_N_COLORS, p=_N_COLORS_P))
    others = [c for c in range(NUM_COLORS) if c != bg]
    palette = [int(c) for c in rng.choice(others, size=n_colors, replace=False)]
    return bg, palette


def _color_cycle(rng: np.random.Generator, palette: list[int]):
    """Round-robin over a shuffled palette so every colour is used once the object
    count allows (uniform draws left most of a large palette unused)."""
    order = [palette[i] for i in rng.permutation(len(palette)).tolist()]
    i = 0
    while True:
        yield order[i % len(order)]
        i += 1


def _fill(scene: _Scene, palette: list[int], n_left: int, cap_fn, weights: dict, conn: int,
          max_cells: int, max_h: int, max_w: int) -> None:
    """Place up to n_left filler objects whose score stays under the cap."""
    rng = scene.rng
    used = {o["color"] for o in scene.objects}
    unused = [c for c in palette if c not in used]
    cycle = _color_cycle(rng, unused + [c for c in palette if c in used])
    for _ in range(n_left):
        color = next(cycle)
        for _try in range(4):
            mc = int(rng.integers(2, max_cells + 1))
            shape, fam = _family_shape(rng, mc, max_h, max_w, weights, conn, min_cells=2)
            if len(shape) < 2 or len(shape) > max_cells or not cap_fn(shape):
                continue
            if scene.place(shape, color, fam, "filler") is not None:
                break


def _touch_feature(scene: _Scene, palette: list[int], cap_fn, weights: dict, conn: int,
                   max_cells: int) -> bool:
    """A small object of a different colour pressed against an existing one."""
    rng = scene.rng
    if len(palette) < 2 or not scene.objects:
        return False
    for _ in range(4):
        mc = int(rng.integers(2, max(2, min(max_cells, 6)) + 1))
        shape, fam = _shape_near(rng, mc, scene.h, scene.w, weights, conn, min_frac=0.5, tries=4)
        if len(shape) < 2 or not cap_fn(shape):
            continue
        anchor = scene.objects[int(rng.integers(0, len(scene.objects)))]
        colors = [c for c in palette if c != anchor["color"]]
        color = int(rng.choice(colors))
        if scene.place(shape, color, fam, "touch", touch_other=True) is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Composition per task.  Each returns (scene, winner_obj, info) or None.
# ---------------------------------------------------------------------------

_CHAIN_SCENE_P = 0.55   # task 18: a chain-vs-solid scene (who wins depends on the connectivity)
_CHAIN_LOSER_P = 0.30   # task 18 solid scenes: a chain whose union still loses under both readings


def _place_chain(rng, scene: _Scene, sizes: list[int], piece_dim: int, color: int, role: str,
                 h: int, w: int) -> Optional[dict]:
    """Build and place a diagonal chain of the given piece sizes (every piece >= 2 cells)."""
    chain = _diag_chain(rng, sizes, max(2, min(5, piece_dim)))
    if chain is None:
        return None
    union, pieces = chain
    sh, sw = shape_size(union)
    if sh > h or sw > w:
        return None
    return scene.place(union, color, "chain", role, pieces=pieces)


def _compose_cells(rng, h: int, w: int) -> Optional[tuple]:
    """Task 18: score = cell count; connectivity 4 or 8 drawn per example (prompted).

    The scene is drawn WITHOUT looking at the connectivity: the same recipes at the same
    rates, shapes sampled 4-connected, chain pieces of at least two cells.  The connectivity
    only decides which object wins (and so which object gets the "winner" role):

    * chain scene (``_CHAIN_SCENE_P``): a same-colour chain of pieces touching only at
      corners and a solid object, sized so the chain's union beats the solid by >= 2 while
      every piece loses to the solid by >= 2.  Under 8-conn the chain is the winner and the
      solid the runner-up; under 4-conn the solid wins and the chain pieces are distractors.
      Two recipes with the same geometry: (a) union = winner target, solid = cap; (b) solid =
      winner target, union = target + 2..4 with pieces <= cap.
    * solid scene: solid winner, runner-up under the cap, sometimes a chain whose union
      still loses (harmless under both readings).
    """
    conn = 4 if rng.random() < 0.5 else 8
    bg, palette = _pick_colors(rng)
    scene = _Scene(rng, h, w, bg)
    n, max_cells, s_w, density = _budget(rng, h, w, len(palette))
    weights = _scene_weights(rng, 4)          # connectivity-independent (no diagonal family)
    area = h * w
    info: dict = {"scene_mode": "plain", "margin_mode": "", "chains": 0, "touch": False}

    big_scene = area >= 150 and rng.random() < 0.15
    if big_scene:
        s_w = max(s_w, 15)
    cap, margin_mode = _runner_cap(rng, s_w)
    info["margin_mode"] = margin_mode
    color_a = int(rng.choice(palette))
    placed = 0
    fill_cap = cap

    chain_scene = rng.random() < _CHAIN_SCENE_P and s_w >= 6 and area >= 40 and cap >= 4
    if chain_scene:
        recipe = "a" if rng.random() < 0.5 else "b"
        chain_role = "winner" if conn == 8 else "distractor"
        solid_role = "runner_up" if conn == 8 else "winner"
        chain_color = color_a
        solid_color = int(rng.choice(palette))
        if recipe == "a":
            # union = s_w, solid = cap, pieces <= cap - 2 (so the solid beats every piece by >= 2)
            piece_cap = cap - 2
            k_min = max(2, -(-s_w // piece_cap))
            if k_min > 4:
                return None
            k = k_min if k_min == 4 or rng.random() < 0.6 else k_min + 1
            sizes = _composition(rng, s_w, k, 2, piece_cap)
            if sizes is None:
                return None
            got = _exact_shape(rng, cap, h, w)
            if got is None:
                return None
            solid_shape, solid_fam = got
            fill_cap = piece_cap
        else:
            # solid = s_w, union = s_w + 2..4 with pieces <= cap
            shape = _big_blob_shape(rng, s_w, h, w) if big_scene else None
            solid_fam = "big_blob"
            if shape is None:
                shape, solid_fam = _family_shape(rng, s_w, h, w, weights, 4)
            if len(shape) < 4:
                return None
            solid_shape = shape
            s_w = len(shape)
            cap, margin_mode = _runner_cap(rng, s_w)
            info["margin_mode"] = margin_mode
            if cap < 2:
                return None
            k_min = max(2, -(-(s_w + 2) // cap))
            if k_min > 4:
                return None
            k = k_min if k_min == 4 or rng.random() < 0.7 else k_min + 1
            total = min(k * cap, s_w + 2 + int(rng.integers(0, 3)))
            sizes = _composition(rng, total, k, 2, cap)
            if sizes is None:
                return None
            fill_cap = cap
        # placement order does not depend on the connectivity
        order = ("chain", "solid") if rng.random() < 0.5 else ("solid", "chain")
        for what in order:
            if what == "chain":
                if _place_chain(rng, scene, sizes, fill_cap, chain_color, chain_role, h, w) is None:
                    return None
            else:
                if scene.place(solid_shape, solid_color, solid_fam, solid_role) is None:
                    return None
        placed = 2
        info["scene_mode"] = "chain_scene_" + recipe
        info["chains"] += 1
    else:
        # Solid winner from the mixture (optionally the one big irregular blob).
        shape = _big_blob_shape(rng, s_w, h, w) if big_scene else None
        fam = "big_blob"
        if shape is None:
            shape, fam = _family_shape(rng, s_w, h, w, weights, 4)
        if len(shape) < 4:
            return None
        s_w = len(shape)
        cap, margin_mode = _runner_cap(rng, s_w)
        info["margin_mode"] = margin_mode
        if scene.place(shape, color_a, fam, "winner") is None:
            return None
        placed = 1
        fill_cap = cap
        # runner-up: exact size in tight mode, near the cap in loose mode
        if margin_mode == "tight":
            got = _exact_shape(rng, cap, h, w)
            if got is None:
                return None
            ru_shape, ru_fam = got
        else:
            ru_target = int(rng.integers(max(2, cap * 3 // 5), cap + 1))
            ru_shape, ru_fam = _family_shape(rng, ru_target, h, w, weights, 4, min_cells=2)
        if len(ru_shape) > cap:
            return None
        if scene.place(ru_shape, int(rng.choice(palette)), ru_fam, "runner_up") is None:
            return None
        placed = 2
        if cap >= 4 and rng.random() < _CHAIN_LOSER_P:
            # a chain whose union still loses: harmless under both readings
            total = int(rng.integers(4, cap + 1))
            k = 2 if total < 6 or rng.random() < 0.7 else 3
            sizes = _composition(rng, total, k, 2, max(2, total - 2 * (k - 1)))
            if sizes is not None and _place_chain(rng, scene, sizes, total, int(rng.choice(palette)),
                                                  "filler", h, w) is not None:
                placed += 1
                info["scene_mode"] = "chain_harmless"
                info["chains"] += 1

    def cap_fn(shape: Shape) -> bool:
        return len(shape) <= fill_cap

    _fill(scene, palette, max(0, n - placed), cap_fn, weights, 4, min(fill_cap, max_cells), h, w)
    if rng.random() < _ROLE_TOUCH_P:
        info["touch"] = _touch_feature(scene, palette, cap_fn, weights, 4, min(fill_cap, max_cells))
    info["density_target"] = density
    return scene, info, conn, _cells_score


def _compose_bbox(rng, h: int, w: int) -> Optional[tuple]:
    """Task 17: score = bounding-box area; connectivity 4 or 8 (prompted)."""
    conn = 4 if rng.random() < 0.5 else 8
    bg, palette = _pick_colors(rng)
    scene = _Scene(rng, h, w, bg)
    n, max_cells, s_w, density = _budget(rng, h, w, len(palette))
    weights = _scene_weights(rng, conn)
    area = h * w
    info: dict = {"scene_mode": "plain", "margin_mode": "", "chains": 0, "touch": False}
    winner_color = int(rng.choice(palette))
    placed = 0
    win = None

    if conn == 8 and rng.random() < 0.25 and s_w >= 5:
        # chain winner: one 8-connected object made of diagonally touching parts
        k = 2 if s_w < 8 or rng.random() < 0.6 else 3
        sizes = _composition(rng, s_w, k, 1, max(1, s_w - (k - 1)))
        chain = _diag_chain(rng, sizes, max(2, min(5, s_w))) if sizes else None
        if chain is not None:
            union, pieces = chain
            sh, sw = shape_size(union)
            if sh <= h and sw <= w:
                win = scene.place(union, winner_color, "chain", "winner", pieces=pieces)
                if win is not None:
                    info["scene_mode"] = "chain_winner"
                    info["chains"] += 1
                    placed = 1
    if win is None:
        shape = _big_blob_shape(rng, s_w, h, w) if area >= 150 and rng.random() < 0.15 else None
        fam = "big_blob"
        if shape is None:
            sparse = rng.random() < 0.68
            wts = _scene_weights(rng, conn, base=_SPARSE_WEIGHTS) if sparse else weights
            shape, fam = _family_shape(rng, s_w, h, w, wts, conn)
        if len(shape) < 4:
            return None
        win = scene.place(shape, winner_color, fam, "winner")
        if win is None:
            return None
        placed = 1

    w_cells = len(win["cells"])
    b_w = _bbox_score(win["cells"])
    cap = b_w - 2
    if cap < 2:
        return None
    info["margin_mode"] = "bbox"

    # dense rival: more cells than the winner inside a smaller box
    if w_cells <= b_w - 2 and rng.random() < 0.80:
        allow_big = area >= 150 and rng.random() < 0.5
        got = _dense_rival(rng, w_cells + 1, max(max_cells, w_cells + 1), cap, h, w, allow_big)
        if got is not None and scene.place(got[0], int(rng.choice(palette)), got[1], "rival") is not None:
            placed += 1
            info["scene_mode"] = "dense_rival"

    if conn == 4 and rng.random() < 0.35:
        # same-colour pieces touching only diagonally: their union's box would
        # beat the winner, but under 4-conn each piece is a separate object.
        for _ in range(5):
            k = 2 if rng.random() < 0.7 else 3
            sizes = [int(rng.integers(2, min(6, max_cells) + 1)) for _ in range(k)]
            chain = _diag_chain(rng, sizes, max(2, min(5, max_cells)))
            if chain is None:
                continue
            union, pieces = chain
            if _bbox_score(union) < b_w + 2 or any(_bbox_score(p) > cap for p in pieces):
                continue
            sh, sw = shape_size(union)
            if sh <= h and sw <= w and scene.place(union, int(rng.choice(palette)), "chain", "distractor", pieces=pieces) is not None:
                placed += 1
                info["scene_mode"] = info["scene_mode"] + "+chain_distractor" if info["scene_mode"] != "plain" else "chain_distractor"
                info["chains"] += 1
                break

    def cap_fn(shape: Shape) -> bool:
        return _bbox_score(shape) <= cap

    _fill(scene, palette, max(0, n - placed), cap_fn, weights, conn, max_cells, h, w)
    if rng.random() < _ROLE_TOUCH_P:
        info["touch"] = _touch_feature(scene, palette, cap_fn, weights, conn, max_cells)
    info["density_target"] = density
    return scene, info, conn, _bbox_score


# ---------------------------------------------------------------------------
# Display-only sparse background textures (library kinds, guarded under both readings)
# ---------------------------------------------------------------------------

_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _textured(rng, grid: np.ndarray, bg: int, conn: int, score, winner: frozenset,
              used: set[int], occupied: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (display grid, texture info); falls back to the plain grid.

    Sparse kinds only (isolated single cells): the prompt never names the texture,
    the object rule's isolated-cell clause covers it.  The texture is kept only if
    the winner still wins by >= 2 with every texture cell counted as an object.
    Texture colours come from the free pool; at least one colour is left absent
    for the recolour target (the library excludes present colours itself)."""
    h, w = grid.shape
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= _TEXTURE_P:
        return grid, dict(_NO_TEXTURE)
    free = [c for c in range(NUM_COLORS) if c not in used]
    if len(free) < 2:  # a texture would leave no colour for the target
        return grid, dict(_NO_TEXTURE)
    painted = apply_background(rng, grid, bg, occupied, group="sparse", exclude=[])
    if painted is None:
        return grid, dict(_NO_TEXTURE)
    disp, info = painted
    if len(info["texture_colors"]) >= len(free):   # keep one colour for the target
        return grid, dict(_NO_TEXTURE)
    if not _bg_is_strict_plurality(disp, bg):
        return grid, dict(_NO_TEXTURE)
    top = _unique_top(_components(disp, conn, bg), score, min_margin=2)
    if top is None or top[0] != winner:
        return grid, dict(_NO_TEXTURE)
    return disp, {"texture_group": info["group"], "texture_kind": info["kind"],
                  "texture_colors": [int(c) for c in info["texture_colors"]],
                  "texture_named": False, "texture_coverage": float(info["coverage"])}


# ---------------------------------------------------------------------------
# Generator core
# ---------------------------------------------------------------------------

def _generate(rng: np.random.Generator, task: str) -> dict:
    for attempt in range(MAX_ATTEMPTS):
        h, w = _envelope(rng)
        composed = _compose_bbox(rng, h, w) if task == "17" else _compose_cells(rng, h, w)
        if composed is None:
            continue
        scene, info, conn, score = composed
        if len(scene.objects) < 2:
            continue
        dens = scene.density()
        if not (0.08 <= dens <= 0.38):
            continue
        grid, bg = scene.grid, scene.bg
        winner_obj = next(o for o in scene.objects if o["role"] == "winner")

        # The designated winner must be the unique winner with margin >= 2 on the
        # object-only grid (construction sanity; the emitted grid is re-checked).
        comps = _components(grid, conn, bg)
        top = _unique_top(comps, score, min_margin=2)
        if top is None or top[0] != winner_obj["cells"]:
            continue
        winner, w_score, ru_score = top

        # Metrics that make the prompt's rule observable.
        alt = 8 if conn == 4 else 4
        alt_top = _top_cells(_components(grid, alt, bg), score)
        connectivity_matters = alt_top != winner
        cells_top = _top_cells(comps, _cells_score)
        bbox_vs_cells = cells_top != winner
        rule_matters = bbox_vs_cells if task == "17" else connectivity_matters

        used = {int(v) for v in np.unique(grid)}
        disp, tex = _textured(rng, grid, bg, conn, score, winner, used, scene.occ)
        bg_style = tex["texture_kind"] or "plain"
        accents = list(tex["texture_colors"])
        present = {int(v) for v in np.unique(disp)}
        absent = [c for c in range(NUM_COLORS) if c not in present]
        if not absent:
            continue
        target = int(rng.choice(absent))
        out = disp.copy()
        for r, c in winner:
            out[r, c] = target

        if not _validate(disp, out, bg, conn, target, score):
            continue
        bg_info = infer_background(disp, margin=BG_MARGIN, beat_size=len(winner))

        bh, bw = _bbox_hw(winner)
        obj_colors = sorted({o["color"] for o in scene.objects})
        meta = {
            "target_color": target,
            "background_color": int(bg),
            "connectivity": str(conn),
            "winner_cells": len(winner),
            "winner_bbox": [bh, bw],
            "winner_bbox_area": bh * bw,
            "winner_score": int(w_score),
            "runner_up_score": int(ru_score),
            "margin": int(w_score - ru_score),
            # realized on the emitted grid: "tight" = the runner-up is 2 or 3 behind
            "margin_mode": ("tight" if w_score - ru_score <= 3 else "loose") if task != "17" else info["margin_mode"],
            "rule_matters": bool(rule_matters),
            "connectivity_matters": bool(connectivity_matters),
            "bbox_vs_cells_matters": bool(bbox_vs_cells),
            "scene_mode": info["scene_mode"],
            "winner_kind": winner_obj["kind"],
            "shape_kinds": [o["kind"] for o in scene.objects],
            "object_roles": [o["role"] for o in scene.objects],
            "n_objects": len(comps),
            "n_placed": len(scene.objects),
            # objects that are one 8-component but several 4-components
            "chain_features": sum(1 for o in scene.objects if o["kind"] in ("chain", "diagonal")),
            "touch_feature": bool(info["touch"]),
            "bg_style": bg_style,
            "accent_colors": accents,
            "background_margin": round(float(bg_info["count"]) / max(1, bg_info["runner_up_count"]), 3),
            "object_colors": obj_colors,
            "n_colors": int(len(present)),
            **tex,
            "grid_h": h,
            "grid_w": w,
            "density": round(dens, 3),
            "attempts": attempt + 1,
        }
        return {"input": grid_to_list(disp), "output": grid_to_list(out), "metadata": meta}
    raise RuntimeError(f"task {task}: no valid scene after {MAX_ATTEMPTS} attempts")


@register("17")
def generate_recolor_largest_span(rng: np.random.Generator) -> dict:
    """Task 17: recolor the object with the largest bounding-box area."""
    return _generate(rng, "17")


@register("18")
def generate_recolor_largest_cells(rng: np.random.Generator) -> dict:
    """Task 18: recolor the object with the most cells (connectivity 4 or 8, prompted).
    Task 19 (the 8-connected twin) was merged into this task on 2026-09-07."""
    return _generate(rng, "18")
