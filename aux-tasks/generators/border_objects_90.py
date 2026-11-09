"""Task 90 -- how many objects touch the outer border of the grid.

The September 2026 rebuild replaces the inline legacy version in
``generators/misc.py``.  What it fixes, and how:

* **The background is now part of the rule.**  The prompt carries the standard
  object rule ("Ignoring the background, treat each N-connected same-color
  group of at least two cells ... as one object; isolated single cells,
  whatever their color, are background texture and are ignored; cells of
  different colors are always separate, even where they touch"), so the
  ground's own border-touching regions and the sparse texture dots on the
  border are disposed of by the rule instead of by an unstated convention.
  The ground is never named: the generator only guarantees that it is
  identifiable (strict plurality by 1.25x and the largest 4-connected region
  once isolated cells are set aside).
* **The connectivity is drawn per example** (metadata ``connectivity``) and is
  drawn LAST, after the whole picture and its texture exist, so no visible
  property of a scene can hint at it.  Scenes are built so the other reading
  really changes the answer on a solid share of examples: a border group is a
  run of same-colour pieces joined only at corners, and when several of its
  pieces reach the border the 4-reading counts them one by one while the
  8-reading counts the group once.
* **The answer is decoupled from the object count.**  The number of border
  groups, the number of pieces per group and the number of interior figures
  are drawn independently, so the visible object count no longer brackets the
  answer the way ``n_objects in [max(3, answer), 7]`` used to.
* **Coverage**: sides 5..30 drawn independently (no size conditioning on the
  answer), clean bodies from ``common.shapes`` scaled with the canvas, object
  colours that may repeat, different-colour objects that touch, bodies that
  span edge to edge or look clipped by the border, and sparse textures from
  ``common.backgrounds``.

The September follow-up closed two further findings:

* **No confetti.**  ``per_piece`` used to be ``density * area / n_pieces`` with
  no lower bound while the piece count was free to reach ``area // 12``, so a
  full count arithmetically forced every body down to two to four cells: 24.6%
  of the visible 4-connected pieces had <= 3 cells and a quarter of the grids
  had a median piece of four cells or less.  The density is now drawn first and
  the piece budget follows from it (``density * area / MIN_BODY``), every body
  is drawn at >= ``MIN_BODY`` cells (>= ``MIN_CLIPPED`` after the border cuts
  it), and occupancy stays flat across the area bands because the bodies grow
  with the canvas instead of the counts.  Metadata carries ``piece_sizes``, one
  entry per 4-connected piece of the emitted grid, and the validator checks it.
* **The colour clause carries weight.**  A planned contact is aimed at the
  cells of pieces that reach the ring, so both parties to the touch are objects
  the question counts: a reader who ignored "cells of different colors are
  always separate" now loses an object on 18% of grids instead of 2.9%
  (metadata ``answer_color_blind`` / ``differs_under_color_blind``).

The scene machinery (``sample_plan`` / ``build_scene``) knows nothing about the
border question: task 92 ("objects touching another object"), which is still
legacy in ``generators/misc.py`` and lives in the same scene family, can reuse
it unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common import shapes as S
from common.backgrounds import apply_background, infer_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 40
TEXTURE_MIN_SIDE = 8
TEXTURE_P = 0.40          # attempt rate on eligible grids (short side >= 8)
SPECKLE_P = 0.40          # of those, the share drawn as a two/three-colour speckle
BG_MARGIN = 1.25
MAX_ANSWER = 12
MIN_BODY = 5              # the smallest body a piece is ever drawn with
MIN_CLIPPED = 4           # ... and after the border cuts it
CONTACT_P = 0.34          # plans that ask for two different-colour objects to touch

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# Clean bodies only: rectangles, squares, bars, frames, thick letters,
# staircases and blobs made of a few rectangles.  No random walks, no carving.
_FAMILY_WEIGHTS = {
    "rect": 0.19, "square": 0.06, "bar": 0.11, "frame": 0.11, "L": 0.11,
    "T": 0.06, "U": 0.06, "plus": 0.05, "H": 0.04, "staircase": 0.07,
    "rect_blob": 0.14,
}


# ---------------------------------------------------------------------------
# Bodies (shared vocabulary -> boolean masks)
# ---------------------------------------------------------------------------

def _to_mask(shape) -> np.ndarray:
    rows = max(r for r, _ in shape) + 1
    cols = max(c for _, c in shape) + 1
    mask = np.zeros((rows, cols), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _one_component(mask: np.ndarray) -> bool:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not cells:
        return False
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _ORTH:
            q = (r + dr, c + dc)
            if q in cells and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(cells)


def _rect_blob(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """A blob drawn as the union of two or three overlapping rectangles.

    Every part is at least two cells thick wherever the box allows it, so the
    silhouette reads as a drawn body and never grows one-cell hairs.
    """
    mask = np.zeros((h, w), dtype=bool)
    n = 2 if (min(h, w) < 4 or rng.random() < 0.62) else 3
    lo_h, lo_w = min(2, h), min(2, w)
    for i in range(n):
        rh = int(rng.integers(max(lo_h, h // 2), h + 1))
        rw = int(rng.integers(max(lo_w, w // 2), w + 1))
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        trial = mask.copy()
        trial[r0:r0 + rh, c0:c0 + rw] = True
        if i == 0 or _one_component(trial):
            mask = trial
    return mask


def _capacity(fam: str, bh: int, bw: int) -> int:
    """Cells the thinnest (thickness 1) member of ``fam`` puts in a bh x bw box.

    Conservative on purpose: it is used to drop families that cannot reach the
    size floor in this box, so the draw spends its attempts on bodies that can
    be drawn at full size instead of rejecting dominoes ten times over.
    """
    side = min(bh, bw)
    if fam == "rect":
        return bh * bw
    if fam == "square":
        return side * side
    if fam == "bar":
        return max(bh, bw) * (2 if side >= 2 else 1)
    if fam == "frame":
        return 2 * (bh + bw) - 4
    if fam in ("L", "T"):
        return bh + bw - 1
    if fam in ("U", "H"):
        return 2 * bh + bw - 2
    if fam == "plus":
        return 4 * max(1, (side - 1) // 2) + 1
    if fam == "staircase":
        return side * (side + 1) // 2
    if fam == "rect_blob":
        return max(2, bh // 2) * max(2, bw // 2)
    raise ValueError(f"unknown family {fam}")


def _families(bh: int, bw: int) -> list[str]:
    side = min(bh, bw)
    fams = ["rect", "bar"]
    if side >= 2:
        fams += ["square", "L", "staircase", "rect_blob"]
    if side >= 3:
        fams += ["frame", "T", "U", "plus", "H"]
    return fams


def _build_family(rng: np.random.Generator, fam: str, bh: int, bw: int, floor: int = 2):
    side = min(bh, bw)
    if fam == "rect":
        rh = rw = 1
        for _ in range(4):
            rh = int(rng.integers(max(1, bh - 1), bh + 1))
            rw = int(rng.integers(max(1, bw - 1), bw + 1))
            if rh * rw >= floor:
                break
        return S.rect(rng, rh, rw)
    if fam == "square":
        return S.rect(rng, side, side)
    if fam == "bar":
        length = max(2, max(bh, bw))
        # a short bar is drawn two cells thick rather than emitted as a domino
        thick = 2 if (side >= 2 and (length < floor
                                     or (length >= 5 and rng.random() < 0.25))) else 1
        cells = {(r, c) for r in range(thick) for c in range(length)}
        if bh >= bw:
            cells = {(c, r) for r, c in cells}
        return frozenset(cells)
    if fam == "frame":
        t = 2 if (side >= 6 and rng.random() < 0.25) else 1
        return S.frame(rng, max(bh, 2 * t + 1), max(bw, 2 * t + 1), t)
    if fam == "L":
        t = 2 if (side >= 5 and rng.random() < 0.3) else 1
        return S.letter_L(rng, max(bh, t + 1), max(bw, t + 1), t)
    if fam == "T":
        t = 2 if (side >= 6 and rng.random() < 0.25) else 1
        return S.letter_T(rng, max(bw, 2 * t + 1), max(1, bh - t), t)
    if fam == "U":
        t = 2 if (side >= 6 and rng.random() < 0.25) else 1
        return S.letter_U(rng, max(bh, t + 1), max(bw, 2 * t + 1), t)
    if fam == "plus":
        arm = max(1, (side - 1) // 2)
        t = 2 if (side >= 7 and rng.random() < 0.3) else 1
        return S.plus(rng, arm, t)
    if fam == "H":
        t = 2 if (side >= 7 and rng.random() < 0.25) else 1
        return S.letter_H(rng, max(bh, 2 * t + 1), max(bw, 2 * t + 1), t)
    if fam == "staircase":
        step = 2 if (side >= 6 and rng.random() < 0.35) else 1
        steps = max(2, side // step)
        return S.staircase(rng, steps, step)
    raise ValueError(f"unknown family {fam}")


def _body(rng: np.random.Generator, bh: int, bw: int,
          min_cells: int = MIN_BODY) -> Optional[tuple[np.ndarray, str]]:
    """A clean body whose box fits ``bh x bw`` and holds >= ``min_cells`` cells.

    The floor is a rejection: a family that cannot fill it in this box is
    discarded and the caller retries with a fresh (and, if needed, larger)
    box.  Returning the best short body instead is what produced the fields of
    dominoes -- once the piece count ate the ink budget, every body was drawn
    at two to four cells because nothing ever said no.
    """
    bh, bw = max(1, bh), max(1, bw)
    floor = min_cells if bh * bw >= min_cells else max(2, bh * bw)
    fams = _families(bh, bw)
    fit = [f for f in fams if _capacity(f, bh, bw) >= floor]
    fams = fit or fams
    weights = np.array([_FAMILY_WEIGHTS[f] for f in fams], dtype=float)
    weights = np.cumsum(weights / weights.sum())
    for _ in range(10):
        fam = fams[int(np.searchsorted(weights, rng.random()))]
        if fam == "rect_blob":
            mask = _rect_blob(rng, bh, bw)
        else:
            mask = _to_mask(_build_family(rng, fam, bh, bw, floor))
        if mask.shape[0] > bh or mask.shape[1] > bw or int(mask.sum()) < floor:
            continue
        # the library families are one 4-connected body by construction; the
        # blob is the only one assembled here, and it checks as it grows
        if fam == "rect_blob" and not _one_component(mask):
            continue
        if rng.random() < 0.5:
            mask = mask[::-1]
        if rng.random() < 0.5:
            mask = mask[:, ::-1]
        return np.ascontiguousarray(mask), fam
    return None


def _clip(rng: np.random.Generator, mask: np.ndarray, side: str,
          floor: int = MIN_CLIPPED) -> Optional[np.ndarray]:
    """Cut a body along one side so it reads as a figure running off the grid.

    What survives the cut is still a body: a clipped figure that came back as a
    two-cell stub was one of the ways the old plan leaked dominoes onto the ring.
    """
    h, w = mask.shape
    if side in ("top", "bottom") and h < 3:
        return None
    if side in ("left", "right") and w < 3:
        return None
    if side == "top":
        cut = int(rng.integers(1, max(2, h // 2)))
        out = mask[cut:]
    elif side == "bottom":
        cut = int(rng.integers(1, max(2, h // 2)))
        out = mask[:h - cut]
    elif side == "left":
        cut = int(rng.integers(1, max(2, w // 2)))
        out = mask[:, cut:]
    else:
        cut = int(rng.integers(1, max(2, w // 2)))
        out = mask[:, :w - cut]
    out = np.ascontiguousarray(out)
    if out.size == 0 or int(out.sum()) < floor or not _one_component(out):
        return None
    rows = np.flatnonzero(out.any(axis=1))
    cols = np.flatnonzero(out.any(axis=0))
    out = np.ascontiguousarray(out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])
    return out if int(out.sum()) >= floor else None


# ---------------------------------------------------------------------------
# Figures: one or more same-colour pieces joined only at corners
# ---------------------------------------------------------------------------

def _rim(cells: set) -> list:
    """The cells of a body that have a missing 8-neighbour."""
    return [(r, c) for (r, c) in cells
            if any((r + dr, c + dc) not in cells for dr, dc in _ORTH + _DIAG)]


def _corner_join(rng: np.random.Generator, masks: list[np.ndarray]):
    """Lay the masks out so each new piece touches the union only at a corner.

    Returns (figure mask, label array) where label ``i + 1`` marks piece ``i``,
    or None.  The union is one 8-connected object whose 4-connected components
    are exactly the pieces -- that is the whole load-bearing feature of the
    task: under 8-connectivity the group is one object, under 4-connectivity it
    is several.
    """
    cells = {(int(r), int(c)) for r, c in np.argwhere(masks[0])}
    pieces = [set(cells)]
    for mask in masks[1:]:
        offsets = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        # a cell whose eight neighbours are all inside its own body can never
        # touch anything, so only the rims generate candidate translations
        rim_from = _rim(set(offsets))
        rim_to = _rim(cells)
        cands = {(r + dr - pr, c + dc - pc)
                 for (r, c) in rim_to for dr, dc in _DIAG for (pr, pc) in rim_from}
        cand_list = sorted(cands)
        placed = None
        for idx in rng.permutation(len(cand_list)):
            tr, tc = cand_list[int(idx)]
            moved = {(pr + tr, pc + tc) for pr, pc in offsets}
            if moved & cells:
                continue
            if any((mr + dr, mc + dc) in cells for mr, mc in moved for dr, dc in _ORTH):
                continue
            if not any((mr + dr, mc + dc) in cells for mr, mc in moved for dr, dc in _DIAG):
                continue
            placed = moved
            break
        if placed is None:
            return None
        pieces.append(placed)
        cells |= placed
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    rows = max(r for r, _ in cells) - r0 + 1
    cols = max(c for _, c in cells) - c0 + 1
    lab = np.zeros((rows, cols), dtype=np.int8)
    for i, piece in enumerate(pieces, start=1):
        for r, c in piece:
            lab[r - r0, c - c0] = i
    return lab > 0, lab


# ---------------------------------------------------------------------------
# The plan (everything except the connectivity)
# ---------------------------------------------------------------------------

def _piece_budget(h: int, w: int, density: float) -> int:
    """How many pieces the canvas can carry without turning into confetti.

    The ink comes first: a plan that spends ``density * area`` cells can afford
    ``density * area / MIN_BODY`` bodies and no more.  The older budget was
    ``area // 12`` alone, and since the per-piece size was derived from it by
    division (``density * area / n_pieces``), a full budget forced every body
    down to two to four cells.  Counts are now what gives: on a bigger canvas
    the same occupancy buys bigger objects, not more of them.
    """
    area = h * w
    return int(max(2, min(26, area // 12, int(density * area / MIN_BODY))))


def _answer_cap(h: int, w: int, density: float) -> int:
    """How many border-touching pieces the outer ring can hold.

    Derived from the footprint: a piece plus its moat eats about six border
    cells, and two thirds of the canvas-wide piece budget may go to the
    border, so the cap never depends on the answer the way the legacy
    ``min_dim = 12 when the answer >= 4`` rule did.  The interior takes what
    the border leaves (``budget - used``), so the split is not wasted when the
    scene turns out to be a border-only one.
    """
    perimeter = 2 * (h + w) - 4
    return int(min(MAX_ANSWER, max(1, perimeter // 6),
                   max(1, (2 * _piece_budget(h, w, density) + 2) // 3)))


def sample_plan(rng: np.random.Generator) -> dict:
    """Draw the whole scene plan.  Nothing here looks at the connectivity."""
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    # The density is drawn BEFORE the counts: how much ink the scene spends
    # decides how many bodies it can afford, and every body gets at least
    # MIN_BODY cells.  Occupancy stays flat across the area bands because the
    # bodies grow with the canvas, not because more of them are squeezed in.
    density = float(rng.uniform(0.16, 0.33))
    budget = _piece_budget(h, w, density)
    cap = _answer_cap(h, w, density)

    scene_mode = str(rng.choice(("plain", "corner", "corner", "mixed"),
                                p=(0.26, 0.32, 0.24, 0.18)))
    corner_scene = scene_mode in ("corner", "mixed")

    # --- border groups ------------------------------------------------------
    # Under 4-connectivity the answer is the number of border-touching PIECES,
    # under 8-connectivity the number of corner-joined GROUPS they form.  A
    # plan-level coin (never the connectivity, which does not exist yet) decides
    # which of the two is drawn near-flat over 0..cap, so neither reading ends
    # up with a lopsided answer histogram.  Zeros are ~10% either way.
    anchor = "pieces" if rng.random() < 0.5 else "groups"
    groups: list[dict] = []
    if rng.random() >= 0.10:
        if anchor == "pieces":
            left = int(rng.integers(1, cap + 1))
            while left > 0:
                if corner_scene and left >= 2 and rng.random() < 0.66:
                    on_border = 3 if (left >= 3 and rng.random() < 0.22) else 2
                else:
                    on_border = 1
                groups.append({"pieces": on_border, "on_border": on_border})
                left -= on_border
        else:
            g_cap = max(1, (cap * 2) // 3) if corner_scene else cap
            left = cap
            for _ in range(int(rng.integers(1, g_cap + 1))):
                if left <= 0:
                    break
                if corner_scene and left >= 2 and rng.random() < 0.66:
                    on_border = 3 if (left >= 3 and rng.random() < 0.22) else 2
                else:
                    on_border = 1
                groups.append({"pieces": on_border, "on_border": on_border})
                left -= on_border
    for group in groups:
        # a decoy piece that stays off the border: the group still counts once
        # under 8-connectivity and once under 4-connectivity
        if group["pieces"] < 3 and rng.random() < 0.20:
            group["pieces"] += 1

    # --- interior figures ---------------------------------------------------
    inner = max(0, (h - 2)) * max(0, (w - 2))
    used = sum(g["pieces"] for g in groups)
    int_cap = int(min(12, inner // 13, max(0, budget - used)))
    if not groups:
        int_cap = max(1, int_cap)
    n_interior = int(rng.integers(0, int_cap + 1)) if int_cap > 0 else 0
    if int_cap > 0 and rng.random() < 0.5:
        # lean toward spending the piece budget: interior figures cost the
        # answer nothing, and a populated scene is what carries a wide palette
        n_interior = max(n_interior, int(rng.integers(0, int_cap + 1)))
    if not groups:
        n_interior = max(1, n_interior)
    interior = []
    for _ in range(n_interior):
        pieces = 2 if rng.random() < 0.28 else 1
        interior.append({"pieces": pieces, "on_border": 0})

    figures = groups + interior
    n_pieces = sum(f["pieces"] for f in figures)

    # --- sizes --------------------------------------------------------------
    # A body scales with the canvas but never swallows it: with only one or two
    # pieces the per-piece budget would otherwise be a single enormous slab.
    # The floor is the half that used to be missing: the budget above keeps
    # the piece count low enough that MIN_BODY never blows the occupancy.
    per_piece = min(max(float(MIN_BODY), density * area / max(1, n_pieces)),
                    max(float(MIN_BODY + 1), 0.16 * area), 90.0)
    span = rng.random() < 0.14 and bool(groups)

    # --- palette ------------------------------------------------------------
    # Near-flat over 2..10 with a little weight moved to the top: the palette
    # is truncated by the figure count (each figure carries one colour), and
    # without the lean the 8-10 tail would be spent on scenes that cannot hold
    # it.  Realised: 8-10 on about a fifth of grids, counting texture colours.
    target = int(rng.choice(np.arange(2, 11),
                            p=np.array([.09, .10, .11, .11, .11, .11, .12, .12, .13])))
    n_obj_colors = int(min(max(1, target - 1), len(figures), 9))
    if n_pieces >= 7:
        # a busy scene in a single colour reads as one confetti field
        n_obj_colors = max(n_obj_colors, 2)
    bg = int(rng.integers(0, NUM_COLORS))
    pool = [c for c in range(NUM_COLORS) if c != bg]
    obj_colors = [int(c) for c in rng.choice(pool, size=n_obj_colors, replace=False)]
    colors = list(obj_colors)
    while len(colors) < len(figures):
        colors.append(int(rng.choice(obj_colors)))
    colors = [colors[int(i)] for i in rng.permutation(len(colors))]

    contact = len(figures) >= 2 and n_obj_colors >= 2 and rng.random() < CONTACT_P
    clipped = bool(groups) and rng.random() < 0.28

    return {
        "h": h, "w": w, "area": area, "answer_cap": cap, "scene_mode": scene_mode,
        "piece_budget": budget,
        "border_groups": groups, "interior": interior, "figures": figures,
        "n_pieces": n_pieces, "density": density, "per_piece": per_piece,
        "span": span, "clipped": clipped, "contact": contact,
        "bg": bg, "obj_colors": obj_colors, "colors": colors,
        "planned_colors": n_obj_colors, "palette_target": target,
        "n_border_pieces": sum(g["on_border"] for g in groups),
        "answer_anchor": anchor,
        "n_border_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _piece_extents(lab: np.ndarray, k: int) -> list[tuple[int, int, int, int]]:
    out = []
    for i in range(1, k + 1):
        rr, cc = np.nonzero(lab == i)
        out.append((int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())))
    return out


def _border_touch_counts(lab: np.ndarray, k: int, h: int, w: int) -> np.ndarray:
    """How many pieces reach the outer ring for every top-left placement."""
    fh, fw = lab.shape
    nh, nw = h - fh + 1, w - fw + 1
    rr = np.arange(nh)[:, None]
    cc = np.arange(nw)[None, :]
    counts = np.zeros((nh, nw), dtype=np.int16)
    for pr0, pr1, pc0, pc1 in _piece_extents(lab, k):
        touch = np.zeros((nh, nw), dtype=bool)
        if pr0 == 0:
            touch |= rr == 0
        if pc0 == 0:
            touch |= cc == 0
        touch |= rr == h - 1 - pr1
        touch |= cc == w - 1 - pc1
        counts += touch
    return counts


def _shifted_any(field: np.ndarray, mask: np.ndarray, nh: int, nw: int) -> np.ndarray:
    out = np.zeros((nh, nw), dtype=bool)
    for dr, dc in np.argwhere(mask):
        out |= field[dr:dr + nh, dc:dc + nw]
    return out


class _Canvas:
    def __init__(self, h: int, w: int, bg: int):
        self.h, self.w, self.bg = h, w, bg
        self.grid = np.full((h, w), bg, dtype=np.int16)
        self.occ = np.zeros((h, w), dtype=bool)
        # cells of the PIECES that reach the outer ring: the contact clause
        # only carries weight when both parties to a touch are counted, so a
        # planned contact is aimed at these cells, not at any object at all
        self.ring_occ = np.zeros((h, w), dtype=bool)
        self.ring = np.zeros((h, w), dtype=bool)
        self.ring[0, :] = self.ring[h - 1, :] = True
        self.ring[:, 0] = self.ring[:, w - 1] = True
        self.figures: list[dict] = []

    def place(self, rng, fig: np.ndarray, lab: np.ndarray, k: int, color: int,
              on_border: int, *, contact: bool = False,
              contact_mask: Optional[np.ndarray] = None,
              flush: Optional[str] = None) -> Optional[dict]:
        fh, fw = fig.shape
        if fh > self.h or fw > self.w:
            return None
        if contact:
            # a different-colour object may be touched; a same-colour one never is
            free = ~self.occ & ~halo(self.grid == color)
        else:
            free = ~halo(self.occ)
        ok = placement_map(fig, free)
        if ok is None:
            return None
        nh, nw = ok.shape
        ok = ok & (_border_touch_counts(lab, k, self.h, self.w) == on_border)
        if contact:
            target = self.occ if contact_mask is None else contact_mask
            if not target.any():
                return None
            ok = ok & _shifted_any(halo(target, diagonal=False) & ~self.occ, fig, nh, nw)
        if flush is not None:
            keep = np.zeros((nh, nw), dtype=bool)
            if flush == "top":
                keep[0, :] = True
            elif flush == "bottom":
                keep[nh - 1, :] = True
            elif flush == "left":
                keep[:, 0] = True
            else:
                keep[:, nw - 1] = True
            ok = ok & keep
        spots = np.argwhere(ok)
        if not len(spots):
            return None
        r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
        cells = np.zeros((self.h, self.w), dtype=bool)
        cells[r:r + fh, c:c + fw] = fig
        self.grid[cells] = color
        self.occ |= cells
        for idx in range(1, k + 1):
            piece = np.zeros((self.h, self.w), dtype=bool)
            piece[r:r + fh, c:c + fw] = lab == idx
            if bool((piece & self.ring).any()):
                self.ring_occ |= piece
        record = {"origin": (r, c), "mask": fig, "lab": lab, "pieces": k,
                  "on_border": on_border, "color": color, "contact": contact,
                  "contact_border": bool(contact and contact_mask is not None)}
        self.figures.append(record)
        return record


def _figure_masks(rng: np.random.Generator, plan: dict, spec: dict, clip_side: Optional[str],
                  span_axis: Optional[str]):
    """Build one figure (1-3 corner-joined pieces) sized for the canvas."""
    h, w = plan["h"], plan["w"]
    k = spec["pieces"]
    budget = max(float(MIN_BODY), plan["per_piece"])
    lim_h = max(1, min(h if spec["on_border"] else h - 2, 15))
    lim_w = max(1, min(w if spec["on_border"] else w - 2, 15))
    room = lim_h * lim_w
    floor = MIN_BODY if room > MIN_BODY else max(2, room)
    masks = []
    for _ in range(k):
        side = float(np.sqrt(budget))
        bh = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_h))
        bw = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_w))
        if span_axis == "row" and not masks:
            bh, bw = 1 if rng.random() < 0.5 else min(lim_h, 2), w
        elif span_axis == "col" and not masks:
            bh, bw = h, 1 if rng.random() < 0.5 else min(lim_w, 2)
        # the box has to be able to HOLD a body of `floor` cells; a box drawn
        # smaller than that is what a three-cell piece comes out of
        for _ in range(8):
            if bh * bw > floor:
                break
            if bh < lim_h and (bh <= bw or bw >= lim_w):
                bh += 1
            elif bw < lim_w:
                bw += 1
            else:
                break
        built = _body(rng, bh, bw, min_cells=floor)
        if built is None:
            return None
        mask, fam = built
        if clip_side is not None and not masks:
            cut = _clip(rng, mask, clip_side, floor=min(MIN_CLIPPED, floor))
            if cut is not None and cut.shape[0] <= lim_h and cut.shape[1] <= lim_w:
                mask = cut
                fam = fam + "_clipped"
        masks.append((mask, fam))
    if k == 1:
        mask = masks[0][0]
        lab = mask.astype(np.int8)
        return mask, lab, [masks[0][1]]
    joined = _corner_join(rng, [m for m, _ in masks])
    if joined is None:
        return None
    fig, lab = joined
    if fig.shape[0] > h or fig.shape[1] > w:
        return None
    return fig, lab, [f for _, f in masks]


def build_scene(rng: np.random.Generator, plan: dict) -> Optional[_Canvas]:
    """Place every planned figure.  Returns None when the layout does not fit."""
    canvas = _Canvas(plan["h"], plan["w"], plan["bg"])
    groups = plan["border_groups"]
    interior = plan["interior"]
    colors = plan["colors"]
    span_index = 0 if (plan["span"] and groups) else -1
    clip_index = (int(rng.integers(0, len(groups))) if (plan["clipped"] and groups) else -1)
    contact_index, ring_contact = -1, False
    if plan["contact"]:
        # Prefer a single-piece BORDER figure that has a differently coloured
        # border figure to touch: then both parties to the touch are objects
        # the question counts, and a reader who merged colours would lose one.
        def _rival(i, pool):
            return any(colors[j] != colors[i] for j in pool if j != i)

        border = [i for i, f in enumerate(plan["figures"]) if f["on_border"]]
        single = [i for i in border if plan["figures"][i]["pieces"] == 1
                  and i != span_index and i != clip_index and _rival(i, border)]
        if single:
            contact_index, ring_contact = int(single[-1]), True
        else:
            rest = [i for i in range(len(plan["figures"]))
                    if _rival(i, range(len(plan["figures"])))]
            contact_index = int(rest[-1]) if rest else -1

    order = list(range(len(plan["figures"])))
    if contact_index >= 0:
        order = [i for i in order if i != contact_index] + [contact_index]

    for i in order:
        spec = plan["figures"][i]
        clip_side = None
        if i == clip_index:
            clip_side = str(rng.choice(("top", "bottom", "left", "right")))
        span_axis = None
        if i == span_index:
            span_axis = "row" if rng.random() < 0.5 else "col"
        placed = None
        for tries in range(10):
            built = _figure_masks(rng, plan, spec, clip_side, span_axis)
            if built is None:
                continue
            fig, lab, fams = built
            # a ring-to-ring contact is asked for first and relaxed to any
            # contact, then to none, rather than throwing the whole layout away
            want = i == contact_index and tries < 9
            mask = canvas.ring_occ if (want and ring_contact and tries < 6) else None
            placed = canvas.place(rng, fig, lab, spec["pieces"], colors[i],
                                  spec["on_border"], contact=want,
                                  contact_mask=mask, flush=clip_side)
            if placed is not None:
                placed["families"] = fams
                placed["clipped"] = clip_side is not None
                placed["span"] = span_axis is not None
                break
            # a full-width body or a cut edge may simply not fit any more
            span_axis = None
            clip_side = None
        if placed is None:
            return None
    return canvas


# ---------------------------------------------------------------------------
# Independent literal reader (no numpy, no access to the plan)
# ---------------------------------------------------------------------------

def _paired(rows: list[list[int]]) -> list[list[bool]]:
    """True where a cell has an 8-neighbour of its own colour (not isolated)."""
    h, w = len(rows), len(rows[0])
    flag = [[False] * w for _ in range(h)]
    for r in range(h):
        row = rows[r]
        below = rows[r + 1] if r + 1 < h else None
        for c in range(w):
            color = row[c]
            if c + 1 < w and row[c + 1] == color:
                flag[r][c] = flag[r][c + 1] = True
            if below is None:
                continue
            for cc in (c - 1, c, c + 1):
                if 0 <= cc < w and below[cc] == color:
                    flag[r][c] = flag[r + 1][cc] = True
    return flag


def read_ground(grid) -> int:
    """The colour a reader told only "ignoring the background" settles on:
    the most frequent colour once every isolated single cell is set aside."""
    rows = [list(row) for row in grid]
    flag = _paired(rows)
    tally: dict[int, int] = {}
    for r, row in enumerate(rows):
        marks = flag[r]
        for c, color in enumerate(row):
            if marks[c]:
                tally[color] = tally.get(color, 0) + 1
    return max(sorted(tally), key=lambda color: tally[color])


def _objects(grid, connectivity: int, *, color_blind: bool = False) -> list[list[tuple]]:
    """Objects as the prompt defines them: same-colour components of >= 2 cells
    under the stated connectivity, once the ground is set aside.

    ``color_blind`` drops the "cells of different colors are always separate"
    clause and is the misreading the contact scenes are built to punish.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    ground = read_ground(rows)
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] != ground}
    out = []
    while pending:
        start = min(pending)
        pending.remove(start)
        color = rows[start[0]][start[1]]
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending and (color_blind or rows[q[0]][q[1]] == color):
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) >= 2:
            out.append(comp)
    return out


def _on_border(comp, h: int, w: int) -> bool:
    return any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)


def solve(grid, connectivity: int) -> int:
    """Count the objects with at least one cell on the outer border.

    A literal reading of the prompt: set the background aside, group the
    remaining cells into same-colour components under the stated connectivity,
    drop the one-cell components (background texture), and count the survivors
    that reach row 0, row h-1, column 0 or column w-1.
    """
    h, w = len(grid), len(grid[0])
    return sum(1 for comp in _objects(grid, connectivity) if _on_border(comp, h, w))


def solve_color_blind(grid, connectivity: int) -> int:
    """The answer of a reader who let different colours join into one object."""
    h, w = len(grid), len(grid[0])
    return sum(1 for comp in _objects(grid, connectivity, color_blind=True)
               if _on_border(comp, h, w))


def count_objects(grid, connectivity: int) -> int:
    """Objects (>= 2 cells, background set aside) under the given connectivity."""
    return len(_objects(grid, connectivity))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> None:
    grid, meta = example["input"], example["metadata"]
    h, w = len(grid), len(grid[0])
    conn = int(meta["connectivity"])
    other = 8 if conn == 4 else 4
    if read_ground(grid) != meta["bg_color"]:
        raise ValueError("task 90: the visible ground disagrees with the plan")
    # one traversal per reading, all three recomputed from the visible grid
    comps = _objects(grid, conn)
    answer = sum(1 for comp in comps if _on_border(comp, h, w))
    if str(answer) != example["output"]:
        raise ValueError("task 90: the border count disagrees with the literal reading")
    if answer != meta["n_border_objects"]:
        raise ValueError("task 90: the border count disagrees with the metadata")
    if solve(grid, other) != meta["answer_other_connectivity"]:
        raise ValueError("task 90: the other-connectivity answer is not the recorded one")
    differs = meta["answer_other_connectivity"] != answer
    if differs != meta["differs_under_other_connectivity"]:
        raise ValueError("task 90: the load-bearing flag is wrong")
    if len(comps) != meta["n_objects"]:
        raise ValueError("task 90: the object count disagrees with the metadata")
    pieces4 = comps if conn == 4 else _objects(grid, 4)
    if sorted(len(comp) for comp in pieces4) != list(meta["piece_sizes"]):
        raise ValueError("task 90: the piece sizes disagree with the visible grid")
    merged = solve_color_blind(grid, conn)
    if merged != meta["answer_color_blind"]:
        raise ValueError("task 90: the colour-blind answer is not the recorded one")
    if (merged != answer) != meta["differs_under_color_blind"]:
        raise ValueError("task 90: the colour-separation flag is wrong")
    arr = np.asarray(grid)
    info = infer_background(arr, margin=BG_MARGIN)
    if not info["identifiable"] or int(info["color"]) != meta["bg_color"]:
        raise ValueError("task 90: the background is not identifiable")
    if meta["texture_colors"]:
        tex = np.isin(arr, meta["texture_colors"])
        if not bool(isolated_cell_mask(arr)[tex].all()):
            raise ValueError("task 90: a texture cell is not isolated")
        if set(meta["texture_colors"]) & set(meta["object_colors"] + [meta["bg_color"]]):
            raise ValueError("task 90: a texture colour is an object or ground colour")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(rng: np.random.Generator, canvas: _Canvas):
    """A sparse library texture on the free cells, clear of every object by one cell.

    Only sparse kinds: their cells are isolated single cells, which the prompt's
    object rule already disposes of -- a solid band touching the border would
    read as an object instead.  The texture may sit on the border; that is
    exactly what the isolated-cell rule is for.
    """
    h, w = canvas.h, canvas.w
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= TEXTURE_P:
        return canvas.grid, dict(_NO_TEXTURE)
    # multi_speckle is the one sparse kind that carries two or three colours;
    # asking for it explicitly keeps the 8-10 colour tail alive on scenes whose
    # figure count cannot fill the palette on its own
    kind = "multi_speckle" if rng.random() < SPECKLE_P else None
    painted = apply_background(rng, canvas.grid, canvas.bg, halo(canvas.occ),
                               group="sparse", kind=kind)
    if painted is None:
        return canvas.grid, dict(_NO_TEXTURE)
    display, info = painted
    return display, {"texture_group": info["group"], "texture_kind": info["kind"],
                     "texture_colors": [int(c) for c in info["texture_colors"]],
                     "texture_named": False, "texture_coverage": float(info["coverage"])}


def _sides_touched(cells: np.ndarray, h: int, w: int) -> int:
    rr, cc = np.nonzero(cells)
    # sum(), not "+": numpy booleans add as logical OR and would cap this at 1
    return sum(int(flag) for flag in
               (rr.min() == 0, rr.max() == h - 1, cc.min() == 0, cc.max() == w - 1))


def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = sample_plan(rng)
        # The plan (counts, palette, sizes, scene mode) is drawn once and gets
        # several layout attempts, so retries never drift toward the plans that
        # happen to be easiest to place.
        canvas = None
        for _ in range(3):
            canvas = build_scene(rng, plan)
            if canvas is not None:
                break
        if canvas is None:
            continue
        if canvas.occ.sum() > 0.42 * plan["area"]:
            continue          # a scene that crowds out its own ground
        display, tex = _texture(rng, canvas)
        grid = grid_to_list(display.astype(np.int16))

        # Nothing above this line has looked at the connectivity.
        conn = 4 if rng.random() < 0.5 else 8
        other = 8 if conn == 4 else 4
        comps = _objects(grid, conn)
        answer = sum(1 for comp in comps if _on_border(comp, plan["h"], plan["w"]))
        if answer != (plan["n_border_pieces"] if conn == 4 else plan["n_border_groups"]):
            continue
        # the 4-connected pieces of the VISIBLE grid: a corner-joined group is
        # two or three pieces here, not one figure, which is what a reader sees
        # and what the size floor has to hold for
        pieces4 = comps if conn == 4 else _objects(grid, 4)
        piece_sizes = sorted(len(comp) for comp in pieces4)
        comps_other = _objects(grid, other)
        alt = sum(1 for comp in comps_other if _on_border(comp, plan["h"], plan["w"]))
        merged = solve_color_blind(grid, conn)
        if read_ground(grid) != plan["bg"]:
            continue
        info = infer_background(display, margin=BG_MARGIN)
        if not info["identifiable"] or int(info["color"]) != plan["bg"]:
            continue

        occ_cells = int(canvas.occ.sum())
        sizes, sides, families = [], [], []
        for rec in canvas.figures:
            r, c = rec["origin"]
            box = np.zeros((plan["h"], plan["w"]), dtype=bool)
            box[r:r + rec["mask"].shape[0], c:c + rec["mask"].shape[1]] = rec["mask"]
            sizes.append(int(rec["mask"].sum()))
            sides.append(_sides_touched(box, plan["h"], plan["w"]) if rec["on_border"] else 0)
            families.extend(rec["families"])
        present = sorted({int(v) for v in np.unique(display)})
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"], "n_colors": len(present),
            "connectivity": conn, "answer_cap": plan["answer_cap"],
            "n_objects": len(comps),
            "n_objects_other_connectivity": len(comps_other),
            "n_border_objects": answer,
            "answer_other_connectivity": alt,
            "differs_under_other_connectivity": bool(alt != answer),
            "n_border_pieces": plan["n_border_pieces"],
            "n_border_groups": plan["n_border_groups"],
            "answer_anchor": plan["answer_anchor"],
            "n_interior_figures": len(plan["interior"]),
            "n_figures": len(canvas.figures),
            "n_pieces": plan["n_pieces"],
            "scene_mode": plan["scene_mode"],
            "corner_groups": sum(1 for g in plan["border_groups"] if g["on_border"] > 1),
            "multi_piece_figures": sum(1 for f in plan["figures"] if f["pieces"] > 1),
            "has_span": any(rec.get("span") for rec in canvas.figures),
            "has_clipped": any(rec.get("clipped") for rec in canvas.figures),
            "has_contact": any(rec.get("contact") for rec in canvas.figures),
            "has_ring_contact": any(rec.get("contact_border") for rec in canvas.figures),
            "answer_color_blind": merged,
            "differs_under_color_blind": bool(merged != answer),
            "object_colors": sorted({int(rec["color"]) for rec in canvas.figures}),
            # one entry per placed FIGURE: a corner-joined group counts once
            # (it is one object under 8-connectivity, several under 4)
            "object_sizes": sizes,
            # one entry per 4-connected PIECE of the emitted grid
            "piece_sizes": piece_sizes,
            "min_piece_cells": piece_sizes[0] if piece_sizes else 0,
            "object_kinds": families,
            "border_sides": sides,
            "max_border_sides": max(sides) if sides else 0,
            "bg_color": plan["bg"],
            "background_margin": round(float(info["count"]) / max(1, int(info["runner_up_count"])), 3),
            "occupancy": round(occ_cells / plan["area"], 4),
            "object_cells": occ_cells,
            "planned_density": round(plan["density"], 4),
            "piece_budget": plan["piece_budget"],
            "attempts": attempt,
            **tex,
        }
        example = {"input": grid, "output": str(answer), "metadata": meta}
        validate_example(example)
        return example
    raise RuntimeError(f"task 90: no valid scene after {MAX_ATTEMPTS} attempts")


@register("90")
def generate_count_border_objects(rng: np.random.Generator) -> dict:
    """Task 90: how many objects have at least one cell on the outer border."""
    return _example(rng)
