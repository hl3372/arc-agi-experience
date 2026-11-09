"""Task 22 -- count the empty regions completely enclosed by the object (redesigned 2026-09-08).

The rule (unchanged): the object is every cell of the named colour; every other cell is empty,
whatever its colour; an empty region counts when it cannot reach the grid edge by moving up,
down, left or right through empty cells (diagonal contact does not connect).

Why a redesign: the 2026-09-07 rebuild bought hole variety with noise.  Random-walk and
cellular silhouettes, holes carved as irregular polyominoes, six to nine object-colour
fragments per grid and textures over everything read as static, not as an object with holes.
The user: "22's shapes are overly unnatural complex ... go back to the drawing board".

Design principle: everything on the grid is something a person would draw deliberately.

1. Object bodies are clean geometric shapes with smooth boundaries: rectangles, chamfered
   rectangles, frames, thick-armed letters (L, U, T, H, plus, Z), flat-tipped diamonds, thick
   staircases and symmetric unions of centred rectangles.  One body mostly; two or three
   separate bodies in about a quarter of the grids (never more than three object-colour
   components, islands included).
2. Holes are readable shapes: single cells, squares and rectangles, plus / L / T / U holes,
   straight corridors, rooms made by straight full-span walls at irregular spacing, rings (a
   rectangle island inside a rectangular hole), nested frames (a frame island whose own
   interior counts too), blobs made of two or three overlapping rectangles, and big rooms.
   Variety comes from mixing kinds, sizes and placement (one-cell walls between holes, holes
   touching only at a corner, thick walls), never from noise.
3. Near-miss non-holes: a cavity opened to the outside (or merged with a neighbour) through a
   clean one-wide gap in a wall that is at least two cells thick; and the natural bays of U /
   H bodies.  Bounded share.
4. Naturalness is measured and enforced: no "hair" (an object cell with at most one object
   4-neighbour) anywhere, no notch on the outer boundary except a deliberate gap, at most three
   object components, and a compactness figure recorded per example.
5. Sizes 1..30 flat and independent (a side under 3 or fewer than 12 cells always answers 0);
   the answer comes from the footprint cap clamp((0.45*area - 12) / 12, 1, 16) with zeros ~8%;
   colours 2..10 near-flat; backgrounds from ``common.backgrounds`` on ~30% of eligible grids
   (outside only, never named); hole fills are solid colours or a small clean shape; distractors
   are clean shapes of other colours, some with cavities of their own (which do not count).
6. The answer is recomputed from the painted grid (``count_holes_in_grid``) and must equal the
   target; the generator raises after all attempts (no fallback).

Follow-up (2026-09-08, after the verifier): holes are mostly readable blocks.  One- and
two-cell holes and holes without a 2x2 block (slits, one-wide corridors, arms and rings) are
budgeted per grid (``TINY_MAX_SHARE``, ``THIN_MAX_SHARE``); corridors and letter arms are two
cells wide wherever the body has room; the body keeps at most ``MAX_HOLE_DENSITY`` holes per
remaining object cell (``_cells_needed`` and the footprint cap are sized for that), so a small
body never becomes a sieve.  A carve never changes the component count except by a deliberate
island and never leaves two components touching only at a corner.  ``hole_kinds`` follows the
hole labels in scan order (parallel to ``hole_areas``); a breached hole's kind moves to
``removed_hole_kinds`` and two merged holes become one ``merged`` region.  Colours reach the
target more often: more distractors (frames with a filled centre count twice), fills cycle
through distinct colours, and the top-up may sit inside a big empty hole or diagonal to the body.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from common import shapes as S
from common import topology as T
from common.backgrounds import apply_background
from common.grid import NUM_COLORS, grid_to_list
from registry import register

MAX_ATTEMPTS = 60
ANSWER_CAP = 16
P_ZERO = 0.06
P_ZERO_TINY = 0.15
ANSWER_TILT = 0.5            # p(k) proportional to 1 + tilt*(k-1) within the cap (aggregate stays near-flat)
P_SPACED = 0.45              # a placement that keeps two cells of wall from every other hole
P_STRIP = 0.15               # keep a draw with a side < 3 (answer 0; strips ~1-2%, never 1x1)
P_MULTI = 0.30               # two or three separate bodies (fewer land: the bodies must host the answer)
P_FULL = 0.06                # the body is the whole canvas (a block with holes, no outside)
P_WALL2 = 0.35               # all walls at least two cells thick
P_NEAR_MISS = 0.12           # one cavity opened through a clean gap (honest near-miss)
P_TEXTURE = 0.35             # textured outside (eligible grids only; ~30% land, a kind may not fit)
MIN_TEXTURE_SIDE = 8
MIN_OUTSIDE_FRAC = 0.04
MAX_COMPONENTS = 3
# Thin-hole budgets (2026-09-08 follow-up): a grid's holes are mostly readable blocks.  "Tiny"
# holes (one or two cells) and "thin" holes (no 2x2 block anywhere: slits, one-wide corridors
# and letter arms, one-wide rings) are each capped per grid; the budgets are drawn uniformly
# from 0 to the share below, so the aggregate stays well under it.
TINY_MAX_SHARE = 0.30
THIN_MAX_SHARE = 0.45
MAX_HOLE_DENSITY = 0.07      # holes per remaining object cell: below this the body reads as a body, not a sieve

Cell = tuple[int, int]
D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
D8_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))

BODY_KINDS = ("rect", "chamfer", "frame", "L", "U", "T", "H", "plus", "Z", "diamond", "stair", "union")
_BODY_W = {"rect": 0.08, "chamfer": 0.13, "frame": 0.12, "L": 0.08, "U": 0.08, "T": 0.07, "H": 0.06,
           "plus": 0.06, "Z": 0.05, "diamond": 0.10, "stair": 0.08, "union": 0.09}
HOLE_KINDS = ("pinhole", "rect", "letter", "corridor", "room", "ring", "nested", "blob", "big",
              "frame",      # a frame body's own interior
              "merged")     # two carved holes joined by a near-miss gap
_REJECTS: Counter = Counter()


# ---------------------------------------------------------------------------
# sizes and answers
# ---------------------------------------------------------------------------

def _sample_sides(rng: np.random.Generator) -> tuple[int, int]:
    for _ in range(100):
        h = int(rng.integers(1, 31))
        w = int(rng.integers(1, 31))
        if h == 1 and w == 1:
            continue
        if min(h, w) < 3 and rng.random() > P_STRIP:
            continue
        return h, w
    return 10, 10


HOLE_FOOTPRINT = 18          # a readable block hole with its wall and spacing on a large canvas
HOLE_FOOTPRINT_SMALL = 6     # a pinhole or a domino with its wall on a small canvas


def answer_cap(h: int, w: int) -> int:
    """Footprint cap (checklist rule): how many readable holes the canvas can host without
    turning into a sieve.  The footprint grows from 6 cells on a small canvas to 18 (a 2x2 or
    larger hole with its wall and spacing) at 450 cells; 14 is reached from ~600 cells and 16
    from ~700.  Together with ``_cells_needed`` this keeps the hole density of the body under
    ``MAX_HOLE_DENSITY``."""
    if min(h, w) < 3 or h * w < 12:
        return 0
    area = h * w
    footprint = HOLE_FOOTPRINT_SMALL + (HOLE_FOOTPRINT - HOLE_FOOTPRINT_SMALL) * min(1.0, area / 450.0)
    return int(max(1, min(ANSWER_CAP, (0.45 * area - 12) // footprint)))


def _sample_answer(rng: np.random.Generator, cap: int) -> int:
    if cap == 0 or rng.random() < (P_ZERO_TINY if cap <= 2 else P_ZERO):
        return 0
    p = 1.0 + ANSWER_TILT * np.arange(cap)
    return 1 + int(rng.choice(cap, p=p / p.sum()))


# ---------------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------------

def _shift(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    rs, re = max(0, dr), min(h, h + dr)
    cs, ce = max(0, dc), min(w, w + dc)
    if rs < re and cs < ce:
        out[rs:re, cs:ce] = mask[rs - dr:re - dr, cs - dc:ce - dc]
    return out


def _erode8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out &= _shift(mask, dr, dc)
    return out


def _nb4(mask: np.ndarray) -> np.ndarray:
    n = np.zeros(mask.shape, dtype=np.int64)
    for dr, dc in D4:
        n += _shift(mask, dr, dc)
    return n


def _hairs(obj: np.ndarray) -> int:
    """Object cells with at most one object 4-neighbour (a protrusion, a stub, a lone cell)."""
    return int((obj & (_nb4(obj) <= 1)).sum())


def _outer_notches(obj: np.ndarray) -> int:
    """Outside cells with three object 4-neighbours: one-cell dents in the outer boundary."""
    outside = T.outside_mask(obj)
    return int((outside & (_nb4(obj) >= 3)).sum())


def _bbox(cells) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalise(cells) -> list[Cell]:
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _rect_cells(h: int, w: int) -> list[Cell]:
    return [(r, c) for r in range(h) for c in range(w)]


# ---------------------------------------------------------------------------
# bodies: clean geometric silhouettes, thickness >= 2 (thin frames excepted)
# ---------------------------------------------------------------------------

def _body_rect(rng, bh, bw):
    return np.ones((bh, bw), dtype=bool)


def _body_chamfer(rng, bh, bw):
    """A rectangle with 45-degree chamfered corners (an octagon when all four are cut)."""
    if min(bh, bw) < 4:
        return None
    r = int(rng.integers(1, max(2, min(bh, bw) // 3) + 1))
    m = np.ones((bh, bw), dtype=bool)
    corners = ["tl", "tr", "bl", "br"]
    mode = rng.random()
    if mode < 0.6:
        cut = corners
    elif mode < 0.8:
        cut = corners[:2] if rng.random() < 0.5 else corners[2:]
    else:
        cut = [corners[int(rng.integers(0, 4))]]
    for i in range(bh):
        for j in range(bw):
            if "tl" in cut and i + j < r:
                m[i, j] = False
            if "tr" in cut and i + (bw - 1 - j) < r:
                m[i, j] = False
            if "bl" in cut and (bh - 1 - i) + j < r:
                m[i, j] = False
            if "br" in cut and (bh - 1 - i) + (bw - 1 - j) < r:
                m[i, j] = False
    return m


def _body_frame(rng, bh, bw):
    if min(bh, bw) < 3:
        return None
    t = int(rng.integers(1, max(1, min(bh, bw) // 3) + 1))
    t = min(t, 3)
    if bh - 2 * t < 1 or bw - 2 * t < 1:
        t = 1
    m = np.ones((bh, bw), dtype=bool)
    m[t:bh - t, t:bw - t] = False
    return m


def _thickness(rng, bh, bw, arms: int) -> Optional[int]:
    """Bar thickness for a letter with ``arms`` parallel bars on the short side."""
    hi = min(bh, bw) // (arms + (1 if arms > 1 else 0))
    if hi < 2:
        return None
    return int(rng.integers(2, min(hi, 8) + 1))


def _body_letter(rng, kind, bh, bw):
    m = np.zeros((bh, bw), dtype=bool)
    if kind == "L":
        t = _thickness(rng, bh, bw, 1)
        if t is None or bh < t + 2 or bw < t + 2:
            return None
        m[:, :t] = True
        m[bh - t:, :] = True
    elif kind == "U":
        t = _thickness(rng, bh, bw, 2)
        if t is None or bw < 2 * t + 1 or bh < t + 1:
            return None
        m[:, :t] = True
        m[:, bw - t:] = True
        m[bh - t:, :] = True
    elif kind == "T":
        t = _thickness(rng, bh, bw, 1)
        if t is None or bh < t + 2 or bw < t + 2:
            return None
        m[:t, :] = True
        c0 = (bw - t) // 2
        m[:, c0:c0 + t] = True
    elif kind == "H":
        t = _thickness(rng, bh, bw, 2)
        if t is None or bw < 2 * t + 1 or bh < 3:
            return None
        m[:, :t] = True
        m[:, bw - t:] = True
        r0 = (bh - t) // 2
        m[r0:r0 + t, :] = True
    elif kind == "plus":
        t = _thickness(rng, bh, bw, 1)
        if t is None or bh < t + 2 or bw < t + 2:
            return None
        r0 = (bh - t) // 2
        c0 = (bw - t) // 2
        m[r0:r0 + t, :] = True
        m[:, c0:c0 + t] = True
    elif kind == "Z":
        t = _thickness(rng, bh, bw, 2)
        if t is None or bh < 2 * t + 1 or bw < 2 * t + 1:
            return None
        m[:t, :] = True
        m[bh - t:, :] = True
        c0 = (bw - t) // 2
        m[:, c0:c0 + t] = True
        if rng.random() < 0.5:                       # a true Z: the bars start on opposite sides
            half = max(t, bw // 2)
            m[:] = False
            m[:t, :half + t // 2 + 1] = True
            m[bh - t:, bw - half - t // 2 - 1:] = True
            m[:, c0:c0 + t] = True
    else:
        return None
    return m


def _body_diamond(rng, bh, bw):
    """A rhombus rasterised with flat tips (at least two cells wide) and monotone steps."""
    if bh < 4 or bw < 4:
        return None
    m = np.zeros((bh, bw), dtype=bool)
    mid = (bh - 1) / 2.0
    for i in range(bh):
        d = abs(i - mid) / mid if mid > 0 else 0.0
        width = int(round(bw * (1.0 - d)))
        width = max(2, min(bw, width))
        if (bw - width) % 2:
            width = width + 1 if width < bw else width - 1
        c0 = (bw - width) // 2
        m[i, c0:c0 + width] = True
    return m


def _body_stair(rng, bh, bw):
    """A thick staircase: each step is a block at least two cells high and wide."""
    if bh < 4 or bw < 4:
        return None
    t = int(rng.integers(2, max(2, min(bh, bw) // 2) + 1))
    steps = max(2, min(bh // t, bw // t))
    t_h = bh // steps
    t_w = bw // steps
    if t_h < 2 or t_w < 2:
        return None
    m = np.zeros((bh, bw), dtype=bool)
    for s in range(steps):
        r0 = s * t_h
        r1 = bh if s == steps - 1 else (s + 1) * t_h
        m[r0:r1, :min(bw, (s + 1) * t_w) if s < steps - 1 else bw] = True
    return m


def _body_union(rng, bh, bw):
    """Two or three rectangles sharing a centre (a cross, a fat plus, a stepped block)."""
    if bh < 4 or bw < 4:
        return None
    m = np.zeros((bh, bw), dtype=bool)
    n = int(rng.integers(2, 4))
    for i in range(n):
        # no rectangle spans the whole box, so the union never collapses to a plain rectangle
        rh = int(rng.integers(2, bh + 1)) if i else bh
        rw = int(rng.integers(2, bw + 1)) if i else int(rng.integers(2, bw))
        if i == 0 and rng.random() < 0.5:
            rh, rw = int(rng.integers(2, bh)), bw
        if rh == bh and rw == bw:
            rw = bw - 1
        r0 = (bh - rh) // 2
        c0 = (bw - rw) // 2
        m[r0:r0 + rh, c0:c0 + rw] = True
    return m


_BODY_BUILDERS = {
    "rect": _body_rect, "chamfer": _body_chamfer, "frame": _body_frame,
    "diamond": _body_diamond, "stair": _body_stair, "union": _body_union,
}


def _orient(rng, m: np.ndarray) -> np.ndarray:
    m = np.rot90(m, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        m = m[:, ::-1]
    return np.ascontiguousarray(m)


def _make_body(rng, kind: str, bh: int, bw: int) -> Optional[np.ndarray]:
    if kind in _BODY_BUILDERS:
        m = _BODY_BUILDERS[kind](rng, bh, bw)
    else:
        m = _body_letter(rng, kind, bh, bw)
    if m is None or not m.any():
        return None
    m = _orient(rng, m)
    # trim to content (rot90 may swap sides; the caller re-fits)
    rows = np.nonzero(m.any(axis=1))[0]
    cols = np.nonzero(m.any(axis=0))[0]
    m = m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    if _hairs(m) or not T.is_connected(m, 4):
        return None
    return m


_SOLID_KINDS = ("rect", "chamfer", "frame", "diamond", "stair", "union")


def _pick_body_kind(rng, bh: int, bw: int, allow_frame: bool, solid_bias: float = 0.0) -> str:
    """``solid_bias`` in [0, 1] shifts weight from thin-armed letters to solid bodies (large
    answers need object area to carve into)."""
    names = [k for k in BODY_KINDS if allow_frame or k != "frame"]
    if min(bh, bw) < 4:
        names = [k for k in names if k in ("rect", "frame")] if allow_frame else ["rect"]
    p = np.array([_BODY_W[k] * (1.0 if k in _SOLID_KINDS else (1.0 - solid_bias)) for k in names])
    if p.sum() <= 0:
        p = np.ones(len(names))
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def _cells_needed(k: int) -> int:
    """Body cells (before carving) a carve of ``k`` holes needs so that the remaining object
    still reads as a body: a readable hole takes ~6-8 cells and, from three holes on, the
    object must keep at least 1 / MAX_HOLE_DENSITY cells per hole, plus an outer wall ring.
    One or two holes on a small canvas need only the hole and its wall."""
    return 4 + 9 * k if k <= 2 else 8 + 24 * k


def _capacity(body: np.ndarray, wall: int) -> int:
    """How many holes the body can host: interior cells (``wall`` cells in from the outside)
    over the footprint of a small readable hole with its spacing."""
    inner = _erode8(T.fill_holes(body)) & body
    for _ in range(wall - 1):
        inner = _erode8(inner) & body
    return int(inner.sum()) // (6 if wall == 1 else 9)


def _diag_contact(obj: np.ndarray) -> bool:
    """True when two different object components touch only at a corner (a "tab" hanging
    from the body by its diagonals reads as a stray piece)."""
    lbl, n = T.label_components(obj, 4)
    if n < 2:
        return False
    h, w = obj.shape
    for dr, dc in D8_DIAG:
        a = lbl[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
        b = lbl[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)]
        if ((a > 0) & (b > 0) & (a != b)).any():
            return True
    return False


def _has_2x2(cells: list[Cell]) -> bool:
    s = set(cells)
    return any((r + 1, c) in s and (r, c + 1) in s and (r + 1, c + 1) in s for r, c in s)


def _place_body(rng, canvas: np.ndarray, body: np.ndarray, region: tuple[int, int, int, int]) -> Optional[tuple[int, int]]:
    r0, c0, r1, c1 = region
    bh, bw = body.shape
    if bh > r1 - r0 or bw > c1 - c0:
        return None
    top = r0 + int(rng.integers(0, r1 - r0 - bh + 1))
    left = c0 + int(rng.integers(0, c1 - c0 - bw + 1))
    canvas[top:top + bh, left:left + bw] |= body
    return top, left


def _plan(rng, h: int, w: int, k: int, cap: int) -> dict:
    """Choices drawn once per example so that retries do not drift toward whatever fails least:
    the layout mode, a preferred body kind, the wall thickness and the near-miss flag."""
    need = _cells_needed(k)
    if min(h, w) < 3:
        mode = "strip"
    elif k >= 1 and rng.random() < P_FULL:
        mode = "full"
    elif min(h, w) >= 7 and h * w >= 60 and need <= 0.5 * h * w and rng.random() < P_MULTI:
        mode = "multi"
    else:
        mode = "single"
    solid_bias = min(1.0, max(0.0, (k - 3) / 8.0))
    kind = _pick_body_kind(rng, h, w, k >= 1, solid_bias)
    wall2_ok = min(h, w) >= 7 and 12 * max(1, k) <= 0.45 * h * w
    wall = 2 if (wall2_ok and rng.random() < P_WALL2) else 1
    near_miss = min(h, w) >= 5 and k < cap and rng.random() < P_NEAR_MISS
    return {"mode": mode, "kind": kind, "wall": wall, "near_miss": near_miss, "solid_bias": solid_bias, "need": need}


def _layout(rng, h: int, w: int, k: int, plan: dict) -> Optional[tuple[np.ndarray, dict]]:
    """The object before holes: one body, the whole canvas, or two/three separate bodies."""
    obj = np.zeros((h, w), dtype=bool)
    kinds: list[str] = []
    if min(h, w) < 3:
        # a strip: a plain bar or a couple of bars (answer 0 by the structural floor)
        n = 1 if rng.random() < 0.6 else 2
        for _ in range(n):
            L = int(rng.integers(2, max(2, max(h, w) // (n + 1)) + 1))
            if h == 1 or (w > h and rng.random() < 0.7):
                c = int(rng.integers(0, max(1, w - L + 1)))
                r = int(rng.integers(0, h))
                obj[r, c:c + L] = True
            else:
                r = int(rng.integers(0, max(1, h - L + 1)))
                c = int(rng.integers(0, w))
                obj[r:r + L, c] = True
        kinds.append("bar")
        return obj, {"layout": "strip", "family": "bar", "n_bodies": 1}
    allow_frame = k >= 1
    if plan["mode"] == "full":
        obj[:] = True
        if _capacity(obj, plan["wall"]) < k:
            _REJECTS["full capacity"] += 1
            return None
        return obj, {"layout": "full", "family": "full", "n_bodies": 1}
    need = plan["need"]
    solid_bias = max(0.4, plan["solid_bias"])
    multi = plan["mode"] == "multi"
    if multi:
        n = 2 if (rng.random() < 0.7 or max(h, w) < 12) else 3
        # split the canvas into n regions along the long side with one-cell gaps
        along_rows = h >= w
        total = h if along_rows else w
        m_len = max(3, total // (2 * n))
        cuts = None
        if total >= n * m_len + (n - 1):
            for _ in range(20):
                cand = sorted(rng.choice(np.arange(m_len, total - m_len + 1), size=n - 1, replace=False).tolist())
                bounds_try = [0] + cand + [total]
                if all(bounds_try[i + 1] - bounds_try[i] - (1 if i else 0) >= m_len for i in range(n)):
                    cuts = cand
                    break
        if cuts is None:
            multi = False
        else:
            bounds = [0] + cuts + [total]
            regions = []
            for i in range(n):
                a, b = bounds[i] + (1 if i else 0), bounds[i + 1]
                regions.append((a, 0, b, w) if along_rows else (0, a, h, b))
            placed = 0
            for reg in regions:
                rh, rw = reg[2] - reg[0], reg[3] - reg[1]
                if rh < 2 or rw < 2:
                    continue
                bh = int(rng.integers(max(2, int(0.6 * rh)), rh + 1))
                bw = int(rng.integers(max(2, int(0.6 * rw)), rw + 1))
                kind = _pick_body_kind(rng, bh, bw, allow_frame, solid_bias)
                body = _make_body(rng, kind, bh, bw)
                if body is None:
                    body = _body_rect(rng, bh, bw)
                    kind = "rect"
                if _place_body(rng, obj, body, reg) is not None:
                    kinds.append("rect" if body.all() else kind)   # label the silhouette that resulted
                    placed += 1
            if placed >= 2 and T.component_count(obj, 4) == placed and int(obj.sum()) >= need \
                    and _capacity(obj, plan["wall"]) >= k:
                return obj, {"layout": "multi", "family": "+".join(kinds), "n_bodies": placed}
            _REJECTS["multi layout"] += 1
            return None
    # single body: 40-100% of each side (larger when the answer needs the area), independent
    lo = min(0.95, max(0.4, float(np.sqrt(need / (h * w))) + 0.15))
    solid_bias = plan["solid_bias"]
    for i in range(20):
        bh = int(rng.integers(max(2, int(np.ceil(lo * h))), h + 1))
        bw = int(rng.integers(max(2, int(np.ceil(lo * w))), w + 1))
        kind = plan["kind"] if i < 8 else _pick_body_kind(rng, bh, bw, allow_frame, min(1.0, solid_bias + 0.1 * i))
        if kind == "frame" and not allow_frame:
            kind = "rect"
        body = _make_body(rng, kind, bh, bw)
        if body is None or int(body.sum()) < need or _capacity(body, plan["wall"]) < k:
            continue
        obj[:] = False
        if _place_body(rng, obj, body, (0, 0, h, w)) is None:
            continue
        return obj, {"layout": "single", "family": "rect" if body.all() else kind, "n_bodies": 1}
    obj[:] = False
    body = _body_chamfer(rng, max(2, h - 1), max(2, w - 1)) if min(h, w) >= 5 else None
    kind = "chamfer"
    if body is None:
        body = _body_rect(rng, max(2, h - 1), max(2, w - 1))
        kind = "rect"
    _place_body(rng, obj, body, (0, 0, h, w))
    return obj, {"layout": "single", "family": "rect" if body.all() else kind, "n_bodies": 1}


# ---------------------------------------------------------------------------
# holes: readable shapes carved with a wall around them
# ---------------------------------------------------------------------------

def _positions(cells: list[Cell], ok: np.ndarray) -> np.ndarray:
    """Boolean map of top-left positions where every cell of the shape lands on ``ok``."""
    h, w = ok.shape
    _, _, mh, mw = _bbox(cells)
    if mh >= h or mw >= w:
        return np.zeros((h, w), dtype=bool)
    pos = np.ones((h - mh, w - mw), dtype=bool)
    for dr, dc in cells:
        pos &= ok[dr:dr + h - mh, dc:dc + w - mw]
    out = np.zeros((h, w), dtype=bool)
    out[:h - mh, :w - mw] = pos
    return out


def _letter_hole(rng, max_h: int, max_w: int, thick: int = 1) -> Optional[list[Cell]]:
    """An L / T / plus / U shaped hole with arms ``thick`` cells wide (two on a thick body)."""
    span = min(max_h, max_w)
    arm = int(rng.integers(2, max(2, min(span - thick + 1, 5 + thick)) + 1)) if thick == 1 \
        else int(rng.integers(3, max(3, min(span - 1, 7)) + 1))
    kind = ["L", "T", "plus", "U"][int(rng.integers(0, 4))]
    cells: set[Cell] = set()
    if kind == "L":
        cells = {(i, 0) for i in range(arm)} | {(arm - 1, j) for j in range(arm)}
    elif kind == "T":
        cells = {(0, j) for j in range(arm)} | {(i, arm // 2) for i in range(arm)}
    elif kind == "plus":
        cells = {(arm // 2, j) for j in range(arm)} | {(i, arm // 2) for i in range(arm)}
    else:
        cells = {(i, 0) for i in range(arm)} | {(i, arm - 1) for i in range(arm)} | {(arm - 1, j) for j in range(arm)}
    if thick == 2:                                     # thicken every arm to two cells
        cells = cells | {(r + 1, c) for r, c in cells} | {(r, c + 1) for r, c in cells} | {(r + 1, c + 1) for r, c in cells}
    cells = set(_normalise(cells))
    r1 = max(r for r, _ in cells)
    c1 = max(c for _, c in cells)
    if len(cells) < 3 or r1 >= max_h or c1 >= max_w:
        return None
    m = T.cells_to_mask(cells, (r1 + 1, c1 + 1))
    if m.all():                                        # a thick letter that filled its box is a rectangle
        return None
    m = _orient(rng, m)
    return _normalise([tuple(x) for x in np.argwhere(m)])


def _blob_hole(rng, max_h: int, max_w: int) -> Optional[list[Cell]]:
    """Two or three overlapping rectangles."""
    bh, bw = min(max_h, int(rng.integers(3, 7))), min(max_w, int(rng.integers(3, 7)))
    if bh < 3 or bw < 3:
        return None
    m = np.zeros((bh, bw), dtype=bool)
    for _ in range(int(rng.integers(2, 4))):
        rh, rw = int(rng.integers(2, bh + 1)), int(rng.integers(2, bw + 1))
        r0, c0 = int(rng.integers(0, bh - rh + 1)), int(rng.integers(0, bw - rw + 1))
        m[r0:r0 + rh, c0:c0 + rw] = True
    if not T.is_connected(m, 4) or m.all():
        return None
    return _normalise([tuple(x) for x in np.argwhere(m)])


def _room_walls(rng, rh: int, rw: int, n_rooms: int) -> Optional[tuple[np.ndarray, int]]:
    """Straight full-span walls at irregular offsets partitioning an rh x rw hole; returns
    the wall mask (relative) and the number of rooms."""
    options = []
    for nr in range(0, 4):
        for nc in range(0, 4):
            rooms = (nr + 1) * (nc + 1)
            if 2 <= rooms <= n_rooms and rh >= 2 * (nr + 1) + nr and rw >= 2 * (nc + 1) + nc:
                options.append((nr, nc, rooms))
    if not options:
        return None
    nr, nc, rooms = options[int(rng.integers(0, len(options)))]
    walls = np.zeros((rh, rw), dtype=bool)

    def _cuts(length: int, n: int) -> Optional[list[int]]:
        for _ in range(20):
            cand = sorted(rng.choice(np.arange(2, length - 2), size=n, replace=False).tolist()) if length > 4 and n else []
            if n and len(cand) < n:
                return None
            okc = all(cand[i + 1] - cand[i] >= 3 for i in range(len(cand) - 1))
            if okc:
                return cand
        return None
    rcuts = _cuts(rh, nr)
    ccuts = _cuts(rw, nc)
    if rcuts is None or ccuts is None:
        return None
    for r in rcuts:
        walls[r, :] = True
    for c in ccuts:
        walls[:, c] = True
    return walls, rooms


def _first_cell(region: np.ndarray) -> Cell:
    r, c = np.argwhere(region)[0]
    return int(r), int(c)


def _carve(rng, obj: np.ndarray, k: int, wall: int, tiny_budget: int, thin_budget: int
           ) -> Optional[tuple[np.ndarray, list[str], list[Cell], dict]]:
    """Carve holes into ``obj`` until it encloses exactly ``k`` regions.

    Returns the object, one kind label per hole (carve order), one representative cell per
    hole (so a near miss can tell which hole it opened) and counters.  ``tiny_budget`` caps
    the one- and two-cell holes of this grid and ``thin_budget`` the holes without a 2x2 block
    (slits, one-wide corridors and letter arms, one-wide rings); every other hole is a
    readable block.  A carve never changes the component count except by the island a ring or
    a nested frame adds, and never leaves two components touching only at a corner."""
    obj = obj.copy()
    h, w = obj.shape
    kinds: list[str] = []
    reps: list[Cell] = []
    info = {"near_placements": 0, "islands": 0, "pinholes": 0, "tiny": 0, "thin": 0}
    labels0, count = T.holes(obj)
    if count > k:
        return None
    for lab in range(1, count + 1):                  # a frame body's own interior
        kinds.append("frame")
        reps.append(_first_cell(labels0 == lab))
    islands = 0
    fails = 0
    while count < k and fails < 45:
        remaining = k - count
        holes = T.hole_mask(obj)
        inner = _erode8(T.fill_holes(obj))          # ``wall`` cells from the outside (corners too)
        for _ in range(wall - 1):
            inner = _erode8(inner)
        blocked = T.dilate4(holes)                   # never 4-adjacent to another hole (corner contact allowed)
        spaced = wall >= 2 or (fails < 8 and rng.random() < P_SPACED)
        if spaced:
            blocked = T.dilate8(blocked)             # two cells of wall from every other hole
        ok = inner & obj & ~blocked
        if not ok.any():
            _REJECTS["carve:no space"] += 1
            break
        rows = np.nonzero(ok.any(axis=1))[0]
        cols = np.nonzero(ok.any(axis=0))[0]
        ah, aw = rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1
        free = int(ok.sum())
        # the largest square that fits gives an honest size scale for the ops
        sq = _largest_square(ok)
        n_comp = T.component_count(obj, 4)
        tight = free < 12 or sq < 2
        thick_ok = sq >= 2                           # room for a 2x2 block
        # a body too thin for any block (a bar, a thin frame) hosts the hole a person would
        # draw there: a slit or a single cell, budget or not
        tiny_ok = info["tiny"] < tiny_budget or not thick_ok
        thin_ok = info["thin"] < thin_budget or not thick_ok
        # choose an op; thin and tiny holes only while their budgets last
        opts = {"pinhole": (0.05 + (0.12 if tight else 0.0)) if tiny_ok else 0.0,
                "rect": (0.30 if thick_ok else 0.0) + ((0.25 if tight else 0.05) if (thin_ok or tiny_ok) else 0.0),
                "letter": 0.16 if (sq >= 5 or (sq >= 3 and thin_ok)) else 0.0,
                "corridor": 0.12 if (max(ah, aw) >= 4 and (sq >= 3 or thin_ok)) else 0.0,
                "rooms": 0.13 if (remaining >= 2 and sq >= 5) else 0.0,
                "ring": 0.10 if (n_comp < MAX_COMPONENTS and (sq >= 7 or (sq >= 4 and thin_ok))) else 0.0,
                "nested": 0.08 if (remaining >= 2 and sq >= 6 and n_comp < MAX_COMPONENTS) else 0.0,
                "blob": 0.12 if sq >= 3 else 0.0,
                "big": 0.10 if (len(kinds) <= 1 and sq >= 6) else 0.0}
        if fails >= 15:                              # crowded: the smallest readable blocks, tiny only within budget
            opts = {"rect": 2.0 if (thick_ok or thin_ok or tiny_ok) else 0.0, "pinhole": 0.3 if tiny_ok else 0.0}
            tight = True
        names = [n for n, p in opts.items() if p > 0]
        if not names:
            fails += 1
            continue
        p = np.array([opts[n] for n in names])
        op = names[int(rng.choice(len(names), p=p / p.sum()))]
        cells: Optional[list[Cell]] = None
        island: list[Cell] = []
        walls_rel: Optional[np.ndarray] = None
        added = 1
        if op == "pinhole":
            cells = [(0, 0)]
        elif op == "rect":
            if thick_ok and (not tight or not (thin_ok or tiny_ok) or rng.random() < 0.7):
                hi = 3 if fails >= 15 else 6
                rh = int(rng.integers(2, min(sq, hi) + 1))
                rw = int(rng.integers(2, min(sq, hi) + 1))
                if fails < 15 and rng.random() < 0.3 and max(ah, aw) >= 4:  # elongated: two or three thick
                    t = 2 if (sq < 3 or rng.random() < 0.7) else 3
                    if aw >= t + 2 and (ah < t + 2 or rng.random() < 0.5):
                        rh, rw = t, int(rng.integers(t + 1, min(aw, 8) + 1))
                    elif ah >= t + 2:
                        rh, rw = int(rng.integers(t + 1, min(ah, 8) + 1)), t
            elif tiny_ok and (not thin_ok or max(ah, aw) < 3 or rng.random() < 0.3):
                if aw >= 2 and (ah < 2 or rng.random() < 0.5):
                    rh, rw = 1, 2                    # a domino
                elif ah >= 2:
                    rh, rw = 2, 1
                else:
                    fails += 1
                    continue
            elif thin_ok and max(ah, aw) >= 3:
                L = int(rng.integers(3, min(max(ah, aw), 6) + 1))
                if aw >= L and (ah < L or rng.random() < 0.5):
                    rh, rw = 1, L                    # a one-wide slit
                else:
                    rh, rw = L, 1
            else:
                fails += 1
                continue
            cells = _rect_cells(rh, rw)
        elif op == "letter":
            thick = 2 if (sq >= 5 and (not thin_ok or rng.random() < 0.85)) else 1
            if thick == 1 and not thin_ok:
                fails += 1
                continue
            cells = _letter_hole(rng, min(sq, 8), min(sq, 8), thick)
        elif op == "corridor":
            width = 2 if (sq >= 3 and (not thin_ok or rng.random() < 0.85)) else 1
            if width == 1 and not thin_ok:
                fails += 1
                continue
            horizontal = rng.random() < 0.5
            span = aw if horizontal else ah
            across = ah if horizontal else aw
            if span < 4 or across < width:
                horizontal = not horizontal
                span, across = across, span
            if span < 4 or across < width:
                fails += 1
                continue
            L = int(rng.integers(3, max(3, min(span - 1, 12)) + 1))
            cells = [(i, j) for i in range(width) for j in range(L)] if horizontal \
                else [(j, i) for i in range(width) for j in range(L)]
        elif op == "big":
            rh = int(rng.integers(max(3, int(0.5 * sq)), sq + 1))
            rw = int(rng.integers(max(3, int(0.5 * sq)), sq + 1))
            cells = _rect_cells(rh, rw)
        elif op == "rooms":
            rh = int(rng.integers(5, min(sq, 14) + 1))
            rw = int(rng.integers(5, min(sq, 14) + 1))
            res = _room_walls(rng, rh, rw, min(remaining, 6))
            if res is None:
                fails += 1
                continue
            walls_rel, added = res
            cells = _rect_cells(rh, rw)
        elif op in ("ring", "nested"):
            if op == "ring":
                gap = 2 if (sq >= 7 and (not thin_ok or rng.random() < 0.6)) else 1
                if gap == 1 and not thin_ok:
                    fails += 1
                    continue
                lo = 2 + 2 * gap
                rh = int(rng.integers(lo, min(sq, 12) + 1))
                rw = int(rng.integers(lo, min(sq, 12) + 1))
                ih = int(rng.integers(2, rh - 2 * gap + 1))
                iw = int(rng.integers(2, rw - 2 * gap + 1))
                r0 = int(rng.integers(gap, rh - ih - gap + 1))
                c0 = int(rng.integers(gap, rw - iw - gap + 1))
                island = [(r0 + i, c0 + j) for i in range(ih) for j in range(iw)]
            else:
                rh = int(rng.integers(6, min(sq, 12) + 1))
                rw = int(rng.integers(6, min(sq, 12) + 1))
                ih = int(rng.integers(4, rh - 1))
                iw = int(rng.integers(4, rw - 1))
                r0 = int(rng.integers(1, rh - ih))
                c0 = int(rng.integers(1, rw - iw))
                t = 2 if (ih >= 6 and iw >= 6 and rng.random() < 0.4) else 1
                island = [(r0 + i, c0 + j) for i in range(ih) for j in range(iw)
                          if i < t or i >= ih - t or j < t or j >= iw - t]
                added = 2
            cells = _rect_cells(rh, rw)
        elif op == "blob":
            cells = _blob_hole(rng, min(sq + 1, 8), min(sq + 1, 8))
        if cells is None or added > remaining:
            fails += 1
            _REJECTS["carve:op unusable"] += 1
            continue
        pos = _positions(cells, ok)
        if not pos.any():
            fails += 1
            _REJECTS["carve:no position"] += 1
            continue
        # placement bias: corner-touch or a one-cell wall against an existing hole
        cand = np.argwhere(pos)
        choice = None
        if holes.any() and not spaced and rng.random() < 0.75:
            near8 = T.dilate8(holes)
            near2 = T.dilate4(T.dilate4(holes))
            corner = _shift_any(cells, near8, pos)          # a cell diagonal to a hole: corner touch
            thin = _shift_any(cells, near2 & ~near8, pos)    # one wall cell between: thin wall
            pool = corner if (rng.random() < 0.55 and corner.any()) else (thin if thin.any() else None)
            if pool is not None and pool.any():
                cc = np.argwhere(pool)
                choice = cc[int(rng.integers(0, len(cc)))]
                info["near_placements"] += 1
        if choice is None:
            choice = cand[int(rng.integers(0, len(cand)))]
        r0, c0 = int(choice[0]), int(choice[1])
        trial = obj.copy()
        for dr, dc in cells:
            trial[r0 + dr, c0 + dc] = False
        if walls_rel is not None:
            trial[r0:r0 + walls_rel.shape[0], c0:c0 + walls_rel.shape[1]] |= walls_rel
        for dr, dc in island:
            trial[r0 + dr, c0 + dc] = True
        new_count = T.count_holes(trial)
        if new_count != count + added:
            fails += 1
            _REJECTS["carve:count"] += 1
            continue
        expect_comp = n_comp + (1 if island else 0)
        if _hairs(trial) or T.component_count(trial, 4) != expect_comp or expect_comp > MAX_COMPONENTS:
            fails += 1
            _REJECTS["carve:hair/components"] += 1
            continue
        if _diag_contact(trial):
            fails += 1
            _REJECTS["carve:diagonal tab"] += 1
            continue
        # account for the new holes one by one (kind label, representative cell, thinness)
        new_lab, _ = T.holes(trial)
        new_ids = sorted(set(int(v) for v in new_lab[T.hole_mask(trial) & ~holes]))
        if len(new_ids) != added:
            fails += 1
            _REJECTS["carve:accounting"] += 1
            continue
        obj = trial
        count = new_count
        label = "room" if op == "rooms" else op
        for lab in new_ids:
            region = new_lab == lab
            size = int(region.sum())
            kinds.append(label)
            reps.append(_first_cell(region))
            if size <= 2:
                info["tiny"] += 1
            if not (region[:-1, :-1] & region[1:, :-1] & region[:-1, 1:] & region[1:, 1:]).any():
                info["thin"] += 1
            if size == 1:
                info["pinholes"] += 1
        if island:
            islands += 1
    if count != k:
        _REJECTS["carve:short"] += 1
        return None
    info["islands"] = islands
    return obj, kinds, reps, info


def _largest_square(ok: np.ndarray) -> int:
    """Side of the largest all-True square in ``ok`` (dynamic programming)."""
    h, w = ok.shape
    best = 0
    dp = np.zeros((h + 1, w + 1), dtype=np.int64)
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if ok[r - 1, c - 1]:
                dp[r, c] = 1 + min(dp[r - 1, c], dp[r, c - 1], dp[r - 1, c - 1])
                if dp[r, c] > best:
                    best = int(dp[r, c])
    return best


def _shift_any(cells: list[Cell], target: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Positions (subset of ``pos``) where at least one shape cell lands on ``target``."""
    h, w = target.shape
    _, _, mh, mw = _bbox(cells)
    out = np.zeros((h, w), dtype=bool)
    if mh >= h or mw >= w:
        return out
    acc = np.zeros((h - mh, w - mw), dtype=bool)
    for dr, dc in cells:
        acc |= target[dr:dr + h - mh, dc:dc + w - mw]
    out[:h - mh, :w - mw] = acc
    return out & pos


# ---------------------------------------------------------------------------
# near misses: a clean one-wide gap through a wall at least two cells thick
# ---------------------------------------------------------------------------

def _open_gap(rng, obj: np.ndarray, merge: bool) -> Optional[np.ndarray]:
    h, w = obj.shape
    labels, n = T.holes(obj)
    if n == 0:
        return None
    outside = T.outside_mask(obj)
    cands = []
    for r in range(h):
        for c in range(w):
            if labels[r, c] == 0:
                continue
            for dr, dc in D4:
                for L in (2, 3):
                    path = [(r + dr * i, c + dc * i) for i in range(1, L + 1)]
                    end = (r + dr * (L + 1), c + dc * (L + 1))
                    if not (0 <= end[0] < h and 0 <= end[1] < w):
                        continue
                    if not all(0 <= pr < h and 0 <= pc < w and obj[pr, pc] for pr, pc in path):
                        continue
                    if merge:
                        if labels[end] == 0 or labels[end] == labels[r, c]:
                            continue
                    elif not outside[end]:
                        continue
                    # lateral cells of the channel must be object (a clean gap, no hair)
                    lat_ok = True
                    for pr, pc in path:
                        for lr, lc in ((pr + dc, pc + dr), (pr - dc, pc - dr)):
                            if not (0 <= lr < h and 0 <= lc < w and obj[lr, lc]):
                                lat_ok = False
                    if lat_ok:
                        cands.append(path)
    if not cands:
        return None
    path = cands[int(rng.integers(0, len(cands)))]
    trial = obj.copy()
    for pr, pc in path:
        trial[pr, pc] = False
    if T.count_holes(trial) != n - 1 or _hairs(trial):
        return None
    if T.component_count(trial, 4) != T.component_count(obj, 4) or _diag_contact(trial):
        return None                                  # the gap must not split the body or leave a tab
    return trial


# ---------------------------------------------------------------------------
# painting
# ---------------------------------------------------------------------------

_DISTRACTOR_FAMILIES = ("rect", "square", "frame", "L", "T", "U", "plus", "H", "bar", "staircase")


def _place_free(rng, allowed: np.ndarray, cells, tries: int = 30) -> Optional[list[Cell]]:
    pos = _positions(list(cells), allowed)
    if not pos.any():
        return None
    cand = np.argwhere(pos)
    r0, c0 = cand[int(rng.integers(0, len(cand)))]
    return [(int(r0) + dr, int(c0) + dc) for dr, dc in cells]


def _paint(rng, obj: np.ndarray, obj_color: int, n_target: int) -> tuple[np.ndarray, dict]:
    h, w = obj.shape
    others = [c for c in range(NUM_COLORS) if c != obj_color]
    base = int(rng.choice(others))
    grid = np.full((h, w), base, dtype=np.int64)
    grid[obj] = obj_color
    used = {obj_color, base}
    outside = T.outside_mask(obj)
    labels, n_holes = T.holes(obj)
    info: dict = {"background_kind": "plain", "background_group": "plain", "texture_colors": [],
                  "texture_scope": "none", "hole_fills": [], "n_distractors": 0,
                  "distractor_families": [], "distractor_in_hole": False}
    budget = max(0, n_target - 2)
    free_colors = [c for c in range(NUM_COLORS) if c not in used]
    rng.shuffle(free_colors)

    def _take(n: int) -> list[int]:
        out = []
        while free_colors and len(out) < n:
            out.append(free_colors.pop())
        used.update(out)
        return out

    # distractor shapes outside (clean shapes of other colours), some with cavities of their own;
    # a frame distractor may carry a filled centre in a second colour
    area = h * w
    n_d = 0
    if outside.sum() >= 12 and budget > 0:
        hi = min(8, 2 + area // 40, budget)
        n_d = hi if rng.random() < 0.6 else int(rng.integers(0, hi + 1))
    free_out = outside & ~T.dilate8(obj)
    fams: list[str] = []
    for i in range(n_d):
        # a colour is taken only once its shape is on the grid; a shape that does not fit is
        # retried smaller (down to a domino) so the colour target is not starved by bad luck
        placed = None
        fam = "bar"
        for max_cells in (max(2, min(30, int(0.06 * area))), 6, 2):
            shape, fam = S.sample_shape(rng, max_cells, max(1, h - 2), max(1, w - 2), families=_DISTRACTOR_FAMILIES)
            placed = _place_free(rng, free_out, shape)
            if placed is not None:
                break
        if placed is None:
            break
        col = _take(1)
        if not col:
            break
        for r, c in placed:
            grid[r, c] = col[0]
        pm = T.cells_to_mask(placed, (h, w))
        if fam == "frame" and rng.random() < 0.7:            # a filled centre in its own colour
            centre = T.hole_mask(pm)
            if centre.any():
                extra = _take(1)
                if extra:
                    grid[centre] = extra[0]
                    fam = "frame+centre"
        free_out &= ~T.dilate4(pm)
        fams.append(fam)
    info["n_distractors"] = len(fams)
    info["distractor_families"] = fams
    budget = max(0, n_target - len(used))

    # hole fills: none / some / all, each filled hole in its own colour while the budget lasts;
    # a small clean shape inside a big hole
    if n_holes and budget > 0:
        mode = rng.random()
        fill_p = 0.0 if mode < 0.40 else (0.5 if mode < 0.75 else 1.0)
        if budget >= 2 and fill_p == 0.0 and rng.random() < 0.7:
            fill_p = 0.5 if rng.random() < 0.6 else 1.0
        to_fill = [lab for lab in range(1, n_holes + 1) if fill_p > 0 and rng.random() < fill_p]
        fill_colors = _take(min(budget, len(to_fill))) if to_fill else []
        fills = []
        fi = 0
        for lab in range(1, n_holes + 1):
            region = labels == lab
            if lab in to_fill and fill_colors:
                col = fill_colors[fi % len(fill_colors)]           # cycle: as many distinct colours as possible
                fi += 1
                grid[region] = col
                fills.append("solid")
            elif region.sum() >= 12 and rng.random() < 0.25:
                inner = _erode8(region)
                if inner.sum() >= 2:
                    extra = _take(1)
                    if extra:
                        shape, fam = S.sample_shape(rng, max(2, int(region.sum() // 3)), h, w,
                                                    families=("rect", "square", "plus", "L", "bar"))
                        placed = _place_free(rng, inner, shape)
                        if placed is not None:
                            for r, c in placed:
                                grid[r, c] = extra[0]
                            fills.append("shape")
                            info["distractor_in_hole"] = True
        info["hole_fills"] = fills
    budget = max(0, n_target - len(used))

    # texture on the outside (never inside holes, never on the object); a kind that does not
    # fit is retried with another group once
    if min(h, w) >= MIN_TEXTURE_SIDE and rng.random() < P_TEXTURE:
        occupied = ~outside | (grid != base)
        n_tex = max(1, min(3, budget)) if budget > 0 else 1
        colors = _take(n_tex) if budget > 0 else None
        first = "sparse" if rng.random() < 0.6 else "solid"
        for group in (first, "solid" if first == "sparse" else "sparse"):
            res = apply_background(rng, grid, base, occupied, group=group, colors=colors, exclude=[obj_color])
            if res is not None:
                grid, tinfo = res
                info["background_kind"] = tinfo["kind"]
                info["background_group"] = tinfo["group"]
                info["texture_colors"] = list(tinfo["texture_colors"])
                info["texture_scope"] = "outside"
                used.update(tinfo["texture_colors"])
                break
        if colors and info["texture_scope"] == "none":
            for c in colors:                                    # give the colours back
                used.discard(c)
                free_colors.append(c)
    budget = max(0, n_target - len(used))
    # still short of the colour target: more small clean shapes, outside (a diagonal touch with
    # the object is allowed now) or inside a big unfilled hole, each in a new colour
    free_out = outside & ~T.dilate4(obj) & (grid == base)
    hole_room = np.zeros_like(obj)
    for lab in range(1, n_holes + 1):
        region = labels == lab
        if region.sum() >= 9 and (grid[region] == base).all():
            hole_room |= _erode8(region)
    tries = 0
    while budget > 0 and (free_out.sum() >= 2 or hole_room.sum() >= 2) and tries < 30:
        tries += 1
        extra = _take(1)
        if not extra:
            break
        target_free = hole_room if (hole_room.sum() >= 2 and (free_out.sum() < 4 or rng.random() < 0.3)) else free_out
        shape, fam = S.sample_shape(rng, max(2, min(12, int(0.03 * area) + 2)), max(1, h - 2), max(1, w - 2),
                                    families=("rect", "square", "bar", "L", "plus"))
        placed = _place_free(rng, target_free, shape)
        if placed is None:
            domino = [(0, 0), (0, 1)] if rng.random() < 0.5 else [(0, 0), (1, 0)]
            placed = _place_free(rng, target_free, domino)
            fam = "bar"
        if placed is None and target_free is not free_out:
            placed = _place_free(rng, free_out, shape)
            if placed is None:
                domino = [(0, 0), (0, 1)] if rng.random() < 0.5 else [(0, 0), (1, 0)]
                placed = _place_free(rng, free_out, domino)
                fam = "bar"
        if placed is None:
            free_colors.append(extra[0]); used.discard(extra[0])
            continue
        for r, c in placed:
            grid[r, c] = extra[0]
        pm = T.cells_to_mask(placed, (h, w))
        free_out &= ~T.dilate4(pm)
        hole_room &= ~T.dilate4(pm)
        if target_free is hole_room:
            info["distractor_in_hole"] = True
        fams.append(fam)
        budget -= 1
    info["n_distractors"] = len(fams)
    info["distractor_families"] = fams
    return grid, info


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------

def _axes(obj: np.ndarray) -> dict:
    labels, n = T.holes(obj)
    outside = T.outside_mask(obj)
    corner = False
    thin = False
    pin = False
    sizes = []
    for lab in range(1, n + 1):
        region = labels == lab
        sizes.append(int(region.sum()))
        if region.sum() == 1:
            pin = True
        d8 = T.dilate8(region) & ~T.dilate4(region)
        if ((labels > 0) & (labels != lab) & d8).any():
            corner = True
        d2 = T.dilate4(T.dilate4(region)) & ~T.dilate4(region)
        if (((labels > 0) & (labels != lab)) | outside)[d2].any():
            thin = True
    # islands: object components not touching the outside; nested: a hole inside an island
    comp_lbl, n_comp = T.label_components(obj, 4)
    outside_ring = T.dilate4(outside)
    islands = 0
    nested = False
    for cid in range(1, n_comp + 1):
        comp = comp_lbl == cid
        if not (comp & outside_ring).any():
            islands += 1
            if T.count_holes(comp) > 0:
                nested = True
    filled = T.fill_holes(obj)
    per = int((_nb4(filled) < 4)[filled].sum())  # cells on the silhouette boundary
    min_wall = T.min_wall_thickness(obj)
    if min_wall is None and n:
        # no outside region (the body fills the canvas): the wall is the object between a hole
        # and the grid edge
        hh, ww = obj.shape
        rr, cc = np.nonzero(labels > 0)
        min_wall = int(np.min(np.minimum(np.minimum(rr, hh - 1 - rr), np.minimum(cc, ww - 1 - cc))))
    return {"has_pinhole": pin, "has_corner_touch": corner, "has_thin_wall": thin, "n_components": int(n_comp),
            "n_islands": int(islands), "has_nested": nested, "hole_areas": sizes,
            "largest_hole_share": round(max(sizes) / max(1, int(obj.sum())), 3) if sizes else 0.0,
            "min_wall": int(min_wall) if min_wall is not None else 0,   # 0: no hole to wall off
            "n_hairs": _hairs(obj), "n_outer_notches": _outer_notches(obj),
            "diagonal_contact": _diag_contact(obj),
            "hole_density": round(n / max(1, int(obj.sum())), 4),
            "compactness": round(per * per / max(1, int(filled.sum())), 2)}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _build(rng, h: int, w: int, k: int, cap: int, plan: dict) -> Optional[dict]:
    lay = _layout(rng, h, w, k, plan)
    if lay is None:
        _REJECTS["layout"] += 1
        return None
    obj, lay_info = lay
    if not obj.any():
        _REJECTS["empty"] += 1
        return None
    wall = plan["wall"]
    post: list[str] = []
    kinds: list[str] = []
    reps: list[Cell] = []
    removed: list[str] = []
    carve_info = {"near_placements": 0, "islands": 0, "pinholes": 0, "tiny": 0, "thin": 0}
    near_miss = plan["near_miss"]
    target = k + 1 if near_miss else k
    tiny_budget = thin_budget = 0
    if target > 0:
        tiny_budget = int(rng.integers(0, int(TINY_MAX_SHARE * target) + 1))
        thin_budget = max(tiny_budget, int(rng.integers(0, int(THIN_MAX_SHARE * target) + 1)))
        res = _carve(rng, obj, target, wall, tiny_budget, thin_budget)
        if res is None:
            _REJECTS["carve"] += 1
            return None
        obj, kinds, reps, carve_info = res
        if near_miss:
            merge = k >= 1 and rng.random() < 0.4
            opened = _open_gap(rng, obj, merge=merge)
            if opened is None:
                _REJECTS["gap"] += 1
                return None
            obj = opened
            post.append("merge" if merge else "breach")
        # the hole labels in scan order, each carrying the kind of the hole it came from: a
        # breached hole is gone, two merged holes become one "merged" region
        labels2, n2 = T.holes(obj)
        by_label: dict[int, list[str]] = {}
        for kind, rep in zip(kinds, reps):
            lab = int(labels2[rep])
            if lab == 0:
                removed.append(kind)
            else:
                by_label.setdefault(lab, []).append(kind)
        final: list[str] = []
        for lab in range(1, n2 + 1):
            ks = by_label.get(lab, [])
            if len(ks) == 1:
                final.append(ks[0])
            elif len(ks) > 1:
                final.append("merged")
                removed.extend(ks)
            else:
                _REJECTS["hole accounting"] += 1
                return None
        kinds = final
    elif T.count_holes(obj) != 0:
        _REJECTS["frame with k=0"] += 1
        return None
    if T.count_holes(obj) != k:
        _REJECTS["count"] += 1
        return None
    if (min(h, w) >= 3 and _hairs(obj)) or T.component_count(obj, 4) > MAX_COMPONENTS:
        _REJECTS["hair or components"] += 1
        return None
    if _diag_contact(obj):
        _REJECTS["diagonal tab"] += 1
        return None
    if k >= 3 and k > MAX_HOLE_DENSITY * int(obj.sum()):
        _REJECTS["hole density"] += 1
        return None
    outside = T.outside_mask(obj)
    if lay_info["layout"] != "full" and min(h, w) >= 3 and int(outside.sum()) < max(6 if h * w >= 60 else 2, int(MIN_OUTSIDE_FRAC * h * w)):
        _REJECTS["outside too small"] += 1
        return None
    if _outer_notches(obj) > len(post):
        _REJECTS["notch"] += 1
        return None
    obj_color = int(rng.integers(0, NUM_COLORS))
    n_target = int(rng.choice(np.arange(2, 11), p=np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]) / 9.9))
    n_target = min(n_target, max(2, 2 + (h * w) // 6))
    grid, paint = _paint(rng, obj, obj_color, n_target)
    answer = T.count_holes_in_grid(grid, obj_color)
    if answer != k or not ((grid == obj_color) == obj).all():
        _REJECTS["paint changed the answer"] += 1
        return None
    axes = _axes(obj)
    if axes["n_hairs"] and min(h, w) >= 3:
        _REJECTS["hair after paint"] += 1
        return None
    object_area = int(obj.sum())
    meta = {
        "target_holes": int(answer), "n_holes": int(answer),
        "obj_color": obj_color, "answer_cap": int(cap),
        "grid_h": h, "grid_w": w,
        "hole_kinds": kinds, "removed_hole_kinds": removed, "post_ops": post,
        "tiny_budget": int(tiny_budget), "thin_budget": int(thin_budget),
        "n_tiny_holes": int(carve_info["tiny"]), "n_thin_holes": int(carve_info["thin"]),
        "near_placements": carve_info["near_placements"],
        "object_area": object_area,
        "occupancy": round(object_area / (h * w), 3),
        "n_colors": int(len(np.unique(grid))),
        "n_colors_target": n_target,
        "wall_target": int(wall),
        "connectivity": 4,
        "background_rule": "all_non_object_cells",
    }
    meta.update(lay_info)
    meta.update(axes)
    meta.update(paint)
    return {"input": grid_to_list(grid), "output": str(answer), "metadata": meta}


@register("22")
def generate_count_holes(rng: np.random.Generator) -> dict:
    """Task 22: count the 4-connected empty regions fully enclosed by the object."""
    h, w = _sample_sides(rng)
    cap = answer_cap(h, w)
    k = _sample_answer(rng, cap)
    resized = 0
    plan = _plan(rng, h, w, k, cap)
    for attempt in range(MAX_ATTEMPTS):
        ex = _build(rng, h, w, k, cap, plan)
        if ex is not None:
            ex["metadata"]["attempts"] = attempt + 1
            ex["metadata"]["size_resampled"] = resized
            return ex
        if attempt % 5 == 4:
            mode = plan["mode"]
            plan = _plan(rng, h, w, k, cap)
            if attempt < 15:
                plan["mode"] = mode
        if attempt % 10 == 9:
            # a canvas that cannot host its answer is redrawn among canvases that can
            for _ in range(50):
                h2, w2 = _sample_sides(rng)
                if answer_cap(h2, w2) >= k and (k == 0 or min(h2, w2) >= 3):
                    h, w = h2, w2
                    cap = answer_cap(h, w)
                    plan = _plan(rng, h, w, k, cap)
                    resized += 1
                    break
    raise RuntimeError(f"Task 22: could not build an example with {k} holes on {h}x{w}")
