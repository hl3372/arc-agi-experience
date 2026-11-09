"""Identify the visible background under a drawn scene (task 25).

The answer is the colour that reads as the ground. Usually it owns the border
ring in a few large pieces and reaches every side, while the figures sit on it
(they may touch one another and may be cut by the border). Majority is NOT
required: one dominant figure inside its margin, with another figure beside it
on the ground, leaves the ground a minority colour. Nor is a ring majority: in the `heavily_clipped` archetype
(~15% of grids) several figures are cut by the border and TOGETHER own more of
the ring than the ground, which is still the one connected field that separates
them and reaches every side. A reader that answers "the colour owning the
border ring" is therefore wrong on those grids, while the drawing stays as
readable as anywhere else.

The scene grammar draws one archetype per example (a dominant figure, a few
large figures, a crowd with a strong size spread, a touching cluster, figures
cut by the border, spanning bars, a jittered motif row, a heavily clipped
scene); figures come from a clean vocabulary (rectangles, chamfers, frames,
windows, thick letters, diamonds, staircases, unions), some with an inner
detail or a nested figure in a pocket, and about a third of the grids carry a
sparse ground texture.

A picture like this can also be read the other way round -- "the big block is
the background and the colour around it is a border it was drawn with" -- and
the shared background library draws exactly such a border. `panel_reading` is
that reader, written out; no example may offer it, so a dominant figure either
sits in a margin too fat to read as a border or has another figure beside it,
outside its bounding box, on the ground. Only the second kind can leave the
ground a minority colour.

`figure_ground_certificate` (generators.exterior_field, with `panel_rules`)
certifies every example from the visible grid alone by one of its two paths,
and `literal_background` reads the same rule a second time in plain Python;
construction never falls back to a canned scene.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from common.backgrounds import sparse_mask
from generators.exterior_field import (COMPANION_CELLS, TWO_COLOR_SPAN, figure_ground_certificate,
                                       ground_margin, label_same_color)


def _body_builders():
    """count_holes registers a task, so import its body builders lazily (no import cycle)."""
    from generators.count_holes import _BODY_BUILDERS, _body_letter
    return _BODY_BUILDERS, _body_letter

MIN_SIDE = 8
SIZES = tuple((h, w) for h in range(MIN_SIDE, 31) for w in range(MIN_SIDE, 31))
MAX_ATTEMPTS = 120
ARCHETYPES = ("dominant_figure", "few_large", "crowd", "cluster", "border_cut",
              "spanning_bars", "motif_row", "heavily_clipped")
# heavily_clipped takes 15%; the rest keep their old proportions, with the dominant figure
# nudged back up because only that archetype leaves the ground a minority colour, and only
# on the canvases roomy enough for a body, its companion and the margin they both need
ARCHETYPE_WEIGHTS = (.25, .13, .125, .12, .11, .075, .04, .15)
COMPOSITIONS = ARCHETYPES
# Planned palette sizes: realised counts fall about one below the plan where a scene has
# no room for more figures or details, so the draw leans high to land near-flat.
COLOR_COUNT_WEIGHTS = {2: .13, 3: .05, 4: .04, 5: .06, 6: .07, 7: .08, 8: .12, 9: .17, 10: .28}
TEXTURE_KINDS = ("random_dots", "lattice", "dotted_stripes", "sparse_checker",
                 "dotted_diagonal", "multi_speckle")
TEXTURE_SHARE = .48       # planned; thin margins, dense scenes and full palettes drop ~40%
TEXTURE_MAX_COVERAGE = .08
BODY_KINDS = ("rect", "chamfer", "frame", "window", "L", "U", "T", "H", "plus", "Z",
              "diamond", "stair", "union")
# thick letters with three bars (U, H, Z) need two-cell bars: six cells on the short side
BODY_MIN_SIDE = {"rect": 2, "chamfer": 4, "frame": 4, "window": 6, "L": 4, "U": 6, "T": 4,
                 "H": 6, "plus": 4, "Z": 6, "diamond": 4, "stair": 4, "union": 4}
BODY_FILL = {"rect": 1.0, "chamfer": .85, "frame": .55, "window": .75, "L": .55, "U": .6,
             "T": .5, "H": .6, "plus": .5, "Z": .55, "diamond": .6, "stair": .65, "union": .7}
BODY_WEIGHTS = {"rect": .09, "chamfer": .12, "frame": .09, "window": .05, "L": .09, "U": .07,
                "T": .07, "H": .05, "plus": .06, "Z": .04, "diamond": .10, "stair": .08,
                "union": .09}
# A dominant figure is mostly solid so that its colour can outnumber the ground.
DOMINANT_WEIGHTS = {"rect": .42, "chamfer": .32, "union": .06, "stair": .03, "window": .06,
                    "frame": .04, "diamond": .03, "L": .015, "U": .01, "T": .01, "H": .005,
                    "plus": .0025, "Z": .0025}
MOTIF_KINDS = ("rect", "chamfer", "frame", "L", "T", "U", "H", "plus", "diamond", "stair")
RING_TOTAL = .34          # figures may hold at most this share of the border ring
RING_PER_COLOR = .22      # and one colour at most this share (certificate: 25%)
# heavily clipped scenes turn that round: the cut figures own MOST of the ring, split between
# colours none of which reaches the certificate's 30% cap, and the ground keeps a third of it
CLIPPED_RING_TOTAL = .62
CLIPPED_RING_PER_COLOR = .275
CLIPPED_RING_MIN = .52    # ... and together strictly more than the ground's half
CLIPPED_GROUND_SHARE = .33     # the ground still holds a third of the canvas (certificate: .30)
CLIPPED_MASS_RATIO = 1.35      # and outweighs every rival colour (certificate: 1.25)
CLIPPED_MIN_FIGURES = 3        # "several figures cut by the border", never two big blocks
CLIPPED_MIN_VISIBLE = 4        # cells a cut figure must keep on the canvas
DETAIL_MIN_CELLS = 12     # a figure needs this many cells and a 4x4 box for a detail
DETAIL_CAP = .45          # cumulative: every detail of one figure, together
DOMINANT_DETAIL_CAP = .35 # ... and on a body that can outnumber the ground
COVERAGE_MIN_AREA = 200   # from this area on, the scene must fill the canvas
COVERAGE_FLOOR = .12      # figure cells (texture dots aside) per canvas cell
COVERAGE_PLAN = 1.15      # plan above the floor so per-figure jitter stays over it
TEXTURE_FIGURE_SHARE = .45  # texture dots stay a minority of the figure mass
DOMINANT_BODY_SHARE = .25   # a "dominant figure" scene means exactly that
COMPANION_BAND = 3          # ... and the band of ground beside it that carries another figure

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


# ---------------------------------------------------------------------------
# small grid helpers
# ---------------------------------------------------------------------------

def _neighbours(mask, deltas, edge):
    """Count neighbours of each cell; ``edge`` continues the mask past the border."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)      # np.pad is slow on the small bodies
    padded[1:-1, 1:-1] = mask
    if edge:                                           # exactly np.pad(mask, 1, mode="edge")
        padded[0, 1:-1], padded[-1, 1:-1] = mask[0], mask[-1]
        padded[1:-1, 0], padded[1:-1, -1] = mask[:, 0], mask[:, -1]
        padded[0, 0], padded[0, -1] = mask[0, 0], mask[0, -1]
        padded[-1, 0], padded[-1, -1] = mask[-1, 0], mask[-1, -1]
    total = np.zeros((h, w), dtype=np.int64)
    for dr, dc in deltas:
        total += padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return total


def hairs(mask, edge=False):
    """Figure cells with at most one figure 4-neighbour (stubs, lone cells, thin arms)."""
    return int((mask & (_neighbours(mask, D4, edge) <= 1)).sum())


def notches(mask, corners=False):
    """Non-figure cells with three or more figure 4-neighbours (one-cell dents, pinholes).

    A canvas corner has only two in-grid neighbours and can never reach three,
    so ``corners`` (canvas-level checks only, never a body's own bounding box,
    where a cut corner is an intended chamfer) counts a corner whose two
    neighbours are both figure: a stranded one-cell ground island.
    """
    around = _neighbours(mask, D4, False)
    dents = ~mask & (around >= 3)
    if corners:
        corner = np.zeros(mask.shape, dtype=bool)
        corner[[0, 0, -1, -1], [0, -1, 0, -1]] = True
        dents |= ~mask & corner & (around >= 2)
    return int(dents.sum())


def halo(mask):
    return mask | (_neighbours(mask, D8, False) > 0)


def _trim(mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    return np.ascontiguousarray(mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])


def _connected(mask):
    """One 4-connected piece; walks the marked cells only, never the whole canvas."""
    points = np.flatnonzero(mask.ravel())
    if len(points) <= 1:
        return True
    h, w = mask.shape
    flat = mask.ravel()
    seen = np.zeros(flat.shape, dtype=bool)
    start = int(points[0])
    seen[start] = True
    pending, found = [start], 1
    while pending:
        p = pending.pop()
        r, c = divmod(p, w)
        for q, ok in ((p - w, r > 0), (p + w, r < h - 1), (p - 1, c > 0), (p + 1, c < w - 1)):
            if ok and flat[q] and not seen[q]:
                seen[q] = True
                found += 1
                pending.append(q)
    return found == len(points)


def _placements(mask, free):
    h, w = free.shape
    sh, sw = mask.shape
    if sh > h or sw > w:
        return np.zeros((0, 2), dtype=np.int64)
    ok = np.ones((h - sh + 1, w - sw + 1), dtype=bool)
    for y, x in np.argwhere(mask):
        ok &= free[y:y + ok.shape[0], x:x + ok.shape[1]]
    return np.argwhere(ok)


def _logu(rng, lo, hi):
    lo, hi = max(1e-9, lo), max(lo, hi)
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _weighted(rng, items, weights):
    p = np.asarray(weights, dtype=float)
    return items[int(rng.choice(len(items), p=p / p.sum()))]


# ---------------------------------------------------------------------------
# bodies: the clean vocabulary
# ---------------------------------------------------------------------------

def _window(rng, bh, bw):
    """A rectangle with one off-centre rectangular pocket; walls at least two cells."""
    if min(bh, bw) < 6:
        return None
    ph = int(rng.integers(2, bh - 3))
    pw = int(rng.integers(2, bw - 3))
    r0 = int(rng.integers(2, bh - ph - 1))
    c0 = int(rng.integers(2, bw - pw - 1))
    m = np.ones((bh, bw), dtype=bool)
    m[r0:r0 + ph, c0:c0 + pw] = False
    return m


def _raw_body(rng, kind, bh, bw):
    if bh < BODY_MIN_SIDE[kind] or bw < BODY_MIN_SIDE[kind]:
        return None
    builders, letter = _body_builders()
    if kind == "window":
        m = _window(rng, bh, bw)
    elif kind in builders:
        m = builders[kind](rng, bh, bw)
    else:
        m = letter(rng, kind, bh, bw)
    if m is None or not m.any():
        return None
    m = _trim(np.asarray(m, dtype=bool))
    if m.sum() < 4 or hairs(m) or notches(m) or not _connected(m):
        return None
    if kind in ("frame", "window") and (m.sum() == m.size or not _pocket(m).any()):
        return None
    return m


def _orient(m, rot, flip):
    m = np.rot90(m, rot)
    if flip:
        m = m[:, ::-1]
    return _trim(np.ascontiguousarray(m))


def _pocket(mask):
    """Enclosed non-figure cells of a body (its pockets)."""
    labels, _ = label_same_color(mask.astype(np.int64))
    border = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    outside = np.isin(labels, np.unique(border))
    return ~mask & ~outside


def _chip(rng, max_h, max_w, cells=4):
    """A small figure (4-11 cells): a little block, or a 3x3 / 3x4 block with cut corners."""
    if cells < 6:
        boxes = [(2, 2), (2, 3), (3, 2)]
    elif cells < 9:
        boxes = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
    else:
        boxes = [(3, 3), (3, 4), (4, 3)]
    boxes = [b for b in boxes if b[0] <= max_h and b[1] <= max_w] or [(2, 2)]
    bh, bw = boxes[int(rng.integers(len(boxes)))]
    m = np.ones((bh, bw), dtype=bool)
    kind = "rect"
    if min(bh, bw) >= 3 and rng.random() < .65:
        corners = [(0, 0), (0, bw - 1), (bh - 1, 0), (bh - 1, bw - 1)]
        cut = rng.choice(4, int(rng.integers(1, 3)), replace=False)
        for i in cut:
            m[corners[int(i)]] = False
        kind = "chamfer"
    return m, kind


def _kinds_for(cells, max_h, max_w, kinds=BODY_KINDS, explicit=False):
    if explicit:
        allowed = list(kinds)
    elif cells < 12:
        allowed = ["chip"]
    elif cells < 18:
        allowed = ["rect", "L", "frame", "stair", "chamfer", "union"]
    else:
        allowed = list(kinds)
    return [k for k in allowed if k == "chip" or (k in kinds and BODY_MIN_SIDE[k] <= min(max_h, max_w))]


def _box_for(rng, kind, cells, max_h, max_w):
    area = cells / BODY_FILL[kind]
    aspect = _logu(rng, 1 / 2.5, 2.5)
    bh = int(round(math.sqrt(area * aspect)))
    bh = max(BODY_MIN_SIDE[kind], min(max_h, bh))
    bw = int(round(area / bh))
    bw = max(BODY_MIN_SIDE[kind], min(max_w, bw))
    return bh, bw


def _figure(rng, cells, max_h, max_w, kinds=BODY_KINDS, orientation=None, tries=6, explicit=False):
    """A body of roughly ``cells`` cells fitting in max_h x max_w, or None."""
    if min(max_h, max_w) < 2:
        return None
    allowed = _kinds_for(cells, max_h, max_w, kinds, explicit)
    if not allowed:
        return None
    weights = [BODY_WEIGHTS.get(k, 1.0) for k in allowed]
    for _ in range(tries):
        kind = _weighted(rng, allowed, weights)
        if kind == "chip":
            m, kind = _chip(rng, max_h, max_w, cells)
        else:
            bh, bw = _box_for(rng, kind, cells, max_h, max_w)
            m = _raw_body(rng, kind, bh, bw)
        if m is None:
            continue
        rot, flip = (int(rng.integers(4)), bool(rng.integers(2))) if orientation is None else orientation
        m = _orient(m, rot, flip)
        if m.shape[0] <= max_h and m.shape[1] <= max_w:
            return m, kind
    return None


# ---------------------------------------------------------------------------
# details inside a figure
# ---------------------------------------------------------------------------

def _detail(rng, mask, core_only=False, max_fraction=DETAIL_CAP, avoid=None, styles=None):
    """A sub-mask of ``mask`` for a second colour: an inner body, a band or a cap.

    ``avoid`` marks cells already taken by an earlier detail of the same figure;
    ``max_fraction`` bounds every detail of this figure TOGETHER, so a body can
    never end up mostly painted in other colours.
    """
    bh, bw = mask.shape
    if styles is None:
        styles = ["core"] if core_only else ["core", "core", "band", "band", "cap"]
    style = str(rng.choice(styles))
    cells = int(mask.sum())
    taken = np.zeros_like(mask) if avoid is None else avoid
    budget = max_fraction * cells - int(taken.sum())        # cumulative, not per detail
    if budget < 2:
        return None
    for _ in range(4):
        detail = None
        if style == "core":
            inner = mask & (_neighbours(mask, D8, False) == 8) & ~halo(taken)
            if inner.sum() < 4:
                style = "band" if not core_only else style
                continue
            ch = int(rng.integers(2, max(2, bh - 2) + 1))
            cw = int(rng.integers(2, max(2, bw - 2) + 1))
            sub = None
            for _ in range(3):
                got = _figure(rng, ch * cw * .8, ch, cw, kinds=("rect", "chamfer", "diamond", "plus"))
                if got is not None:
                    sub = got[0]
                    break
            if sub is None:
                continue
            spots = _placements(sub, inner)
            if not len(spots):
                continue
            r, c = map(int, spots[int(rng.integers(len(spots)))])
            detail = np.zeros_like(mask)
            detail[r:r + sub.shape[0], c:c + sub.shape[1]] = sub
        elif style == "band":
            vertical = bool(rng.integers(2))
            length = bw if vertical else bh
            if length < 4:
                continue
            t = int(rng.integers(1, max(1, length // 3) + 1))
            start = int(rng.integers(1, length - t))
            detail = np.zeros_like(mask)
            if vertical:
                detail[:, start:start + t] = True
            else:
                detail[start:start + t, :] = True
            detail &= mask
        else:
            side = int(rng.integers(4))
            length = bh if side < 2 else bw
            if length < 4:
                continue
            k = int(rng.integers(1, max(1, length // 3) + 1))
            detail = np.zeros_like(mask)
            if side == 0:
                detail[:k, :] = True
            elif side == 1:
                detail[bh - k:, :] = True
            elif side == 2:
                detail[:, :k] = True
            else:
                detail[:, bw - k:] = True
            detail &= mask
        if detail is not None and (detail & halo(taken)).any():
            detail = None
        if detail is not None and 2 <= detail.sum() <= budget and (mask & ~detail & ~taken).sum() >= .5 * cells:
            return detail, style
    return None


# ---------------------------------------------------------------------------
# the scene canvas
# ---------------------------------------------------------------------------

class _Scene:
    def __init__(self, rng, h, w, ground, ring_total=RING_TOTAL, ring_per_color=RING_PER_COLOR):
        self.rng = rng
        self.h, self.w, self.ground = h, w, ground
        self.ring_total, self.ring_per_color = ring_total, ring_per_color
        self.a = np.full((h, w), ground, dtype=np.int64)
        self.fig = np.zeros((h, w), dtype=np.int64)          # figure id per cell, 0 = none
        self.figures = []                                     # dicts
        self.ring = np.zeros((h, w), dtype=bool)
        self.ring[[0, -1], :] = True
        self.ring[:, [0, -1]] = True
        self.ring_n = int(self.ring.sum())
        self.ring_used = {}

    # -- space -------------------------------------------------------------
    @property
    def occupied(self):
        return self.fig > 0

    def free(self, margin=1, moat=True, inside=None):
        free = ~halo(self.occupied) if moat else ~self.occupied
        if margin:
            free[:margin, :] = False
            free[-margin:, :] = False
            free[:, :margin] = False
            free[:, -margin:] = False
        if inside is not None:
            free &= inside
        return free

    def ring_ok(self, cells, color):
        n = int((cells & self.ring).sum())
        if n == 0:
            return True
        total = sum(self.ring_used.values()) + n
        by_color = self.ring_used.get(color, 0) + n
        return total <= self.ring_total * self.ring_n and by_color <= self.ring_per_color * self.ring_n

    # -- painting ------------------------------------------------------------
    def paint(self, cells, color, kind, **info):
        """Add a figure whose cells are given as a full-size boolean mask."""
        if not cells.any() or not self.ring_ok(cells, color):
            return None
        fid = len(self.figures) + 1
        self.fig[cells] = fid
        self.a[cells] = color
        n_ring = int((cells & self.ring).sum())
        if n_ring:
            self.ring_used[color] = self.ring_used.get(color, 0) + n_ring
        figure = {"id": fid, "cells": cells, "color": color, "kind": kind, "detail": None,
                  "touches": set(), "border": n_ring > 0, "clipped_sides": 0, "nested": False}
        figure.update(info)
        self.figures.append(figure)
        return figure

    def place(self, mask, color, kind, *, margin=1, moat=True, inside=None, **info):
        spots = _placements(mask, self.free(margin, moat, inside))
        if not len(spots):
            return None
        r, c = map(int, spots[int(self.rng.integers(len(spots)))])
        cells = np.zeros((self.h, self.w), dtype=bool)
        cells[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
        return self.paint(cells, color, kind, **info)

    def place_clipped(self, mask, color, kind, sides, tries=24, min_cells=0):
        """Place ``mask`` so that it runs past the border on ``sides`` (1-3 sides).

        ``min_cells`` asks for that much of the body to stay on the canvas (never
        more than four fifths of it, so a small body can still be cut).
        """
        h, w = self.h, self.w
        min_cells = min(min_cells, .80 * mask.sum())
        bh, bw = mask.shape
        free = self.free(0, True)
        for _ in range(tries):
            r = int(self.rng.integers(1, max(2, h - bh))) if bh < h - 1 else 1
            c = int(self.rng.integers(1, max(2, w - bw))) if bw < w - 1 else 1
            if "top" in sides:
                r = -int(self.rng.integers(1, max(2, bh - 1)))
            if "bottom" in sides:
                r = h - bh + int(self.rng.integers(1, max(2, bh - 1)))
            if "left" in sides:
                c = -int(self.rng.integers(1, max(2, bw - 1)))
            if "right" in sides:
                c = w - bw + int(self.rng.integers(1, max(2, bw - 1)))
            if "top" in sides and "bottom" in sides:
                r = -int(self.rng.integers(1, max(2, bh - h)))  # taller than the canvas
            if "left" in sides and "right" in sides:
                c = -int(self.rng.integers(1, max(2, bw - w)))
            cells = np.zeros((h, w), dtype=bool)
            r0, c0 = max(0, r), max(0, c)
            r1, c1 = min(h, r + bh), min(w, c + bw)
            if r1 <= r0 or c1 <= c0:
                continue
            cells[r0:r1, c0:c1] = mask[r0 - r:r1 - r, c0 - c:c1 - c]
            visible = int(cells.sum())
            if visible < 4 or visible < max(.30 * mask.sum(), min_cells) or visible == mask.sum():
                continue
            if not (cells & ~free).sum() == 0:
                continue
            if not _connected(cells) or hairs(cells, edge=True) or notches(cells, corners=True):
                continue
            touched = {name for name, hit in (("top", cells[0].any()), ("bottom", cells[-1].any()),
                                              ("left", cells[:, 0].any()), ("right", cells[:, -1].any())) if hit}
            if len(touched) > 3 or (cells[0].all() or cells[-1].all() or cells[:, 0].all() or cells[:, -1].all()):
                continue
            figure = self.paint(cells, color, kind, clipped_sides=len(touched))
            if figure is not None:
                return figure
        return None

    def attach(self, mask, color_options, kind, near=False, tries=12):
        """Place ``mask`` touching the current cluster (or one cell away when ``near``)."""
        cluster = self.occupied
        no_overlap = self.free(1, moat=False)
        if near:
            fits = _placements(mask, no_overlap & ~halo(cluster))
            far = _placements(mask, no_overlap & ~halo(halo(cluster)))
        else:
            fits = _placements(mask, no_overlap)
            far = _placements(mask, no_overlap & ~(_neighbours(cluster, D4, False) > 0) & ~cluster)
        far_set = {tuple(p) for p in far.tolist()}
        spots = [tuple(p) for p in fits.tolist() if tuple(p) not in far_set]
        if not spots:
            return None
        self.rng.shuffle(spots)
        old_pockets = _pocket(cluster)
        for r, c in spots[:tries]:
            cells = np.zeros((self.h, self.w), dtype=bool)
            cells[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
            union = cluster | cells
            if notches(union, corners=True) or hairs(union, edge=True):
                continue
            # no new enclosed ground pocket between touching figures
            if (_pocket(union) & ~old_pockets).any():
                continue
            touched_ids = set(np.unique(self.fig[halo(cells) & cluster]).tolist()) - {0}
            forbidden = {self.figures[i - 1]["color"] for i in touched_ids}
            if not near:
                adjacent = _neighbours(cells, D4, False) > 0
                touched_ids = set(np.unique(self.fig[adjacent & cluster]).tolist()) - {0}
                forbidden = {self.figures[i - 1]["color"] for i in touched_ids}
            choices = [col for col in color_options if col not in forbidden]
            if not choices:
                continue
            figure = self.paint(cells, choices[0], kind, near=near)
            if figure is None:
                continue
            figure["touches"] = touched_ids
            for i in touched_ids:
                self.figures[i - 1]["touches"].add(figure["id"])
            return figure
        return None

    # -- second colours ------------------------------------------------------
    def add_detail(self, figure, color, core_only=False, max_fraction=DETAIL_CAP):
        """Paint part of ``figure`` in ``color``; ``max_fraction`` is the figure's TOTAL budget."""
        rows, cols = np.nonzero(figure["cells"])
        r0, c0 = rows.min(), cols.min()
        r1, c1 = rows.max() + 1, cols.max() + 1
        local = figure["cells"][r0:r1, c0:c1]
        taken = (self.a[r0:r1, c0:c1] != figure["color"]) & local   # earlier details
        # a thin frame or window takes a whole end (cap): a band through it would leave marks
        styles = ["cap"] if figure["kind"] in ("frame", "window") and not core_only else None
        got = _detail(self.rng, local, core_only=core_only, max_fraction=max_fraction,
                      avoid=taken if taken.any() else None, styles=styles)
        if got is None:
            return False
        detail, style = got
        cells = np.zeros((self.h, self.w), dtype=bool)
        cells[r0:r0 + local.shape[0], c0:c0 + local.shape[1]] = detail
        if not self.ring_ok(cells, color):
            return False
        moved = int((cells & self.ring).sum())
        if moved:
            self.ring_used[figure["color"]] -= moved
            self.ring_used[color] = self.ring_used.get(color, 0) + moved
        self.a[cells] = color
        entry = {"color": color, "style": style, "cells": int(cells.sum())}
        figure["detail"] = entry if figure["detail"] is None else figure["detail"]
        figure.setdefault("details", []).append(entry)
        return True

    def nest(self, figure, cells_target, color, kinds=BODY_KINDS):
        """A smaller figure inside the largest pocket of ``figure`` with a one-cell moat."""
        pocket = _pocket(figure["cells"])
        if not pocket.any():
            return None
        labels, _ = label_same_color(pocket.astype(np.int64))
        ids, counts = np.unique(labels[pocket], return_counts=True)
        pocket = labels == ids[int(np.argmax(counts))]
        room = pocket & (_neighbours(pocket, D8, False) == 8)
        if room.sum() < 4:
            return None
        rows, cols = np.nonzero(room)
        got = _figure(self.rng, min(cells_target, .6 * room.sum()),
                      rows.max() - rows.min() + 1, cols.max() - cols.min() + 1, kinds=kinds)
        if got is None:
            return None
        mask, kind = got
        nested = self.place(mask, color, kind, margin=0, moat=False, inside=room, nested=True)
        return nested


# ---------------------------------------------------------------------------
# heavily clipped scenes: the cut figures own the ring, the ground still separates them
# ---------------------------------------------------------------------------

CLIP_SIDES = ("top", "bottom", "left", "right")
CLIP_CORNERS = {"tl": ("top", "left"), "tr": ("top", "right"),
                "bl": ("bottom", "left"), "br": ("bottom", "right")}
CLIP_SIDE_SETS = tuple(tuple(sorted(s)) for k in (2, 3, 4)
                       for s in itertools.combinations(CLIP_SIDES, k))


def _clip_capacity(sides, h, w):
    """Ring cells figures cut on ``sides`` could take while the ground keeps every side."""
    room = {"top": w - 2, "bottom": w - 2, "left": h - 4, "right": h - 4}
    return sum(room[s] for s in sides)


def _clip_sides(rng, h, w, ring_n, floor=.64):
    """Which sides this canvas is cut on: only sets with room for more than half the ring."""
    fit = {}
    for sides in CLIP_SIDE_SETS:
        if _clip_capacity(sides, h, w) >= floor * ring_n:
            fit.setdefault(len(sides), []).append(sides)
    sizes = sorted(fit)
    weights = [{2: .22, 3: .33, 4: .45}[k] for k in sizes]
    k = int(_weighted(rng, sizes, weights))
    return list(fit[k][int(rng.integers(len(fit[k])))])


def _clip_slots(rng, chosen, count, corner_rate=.55):
    """Anchors for the cut figures: enough to clip every side of ``chosen``, ``count`` in all.

    A corner seats one figure and no more (its position is fixed), and a side
    carries at most two, so every anchor list this returns has room to be drawn.
    """
    chosen = list(chosen)
    corners = sorted(k for k, s in CLIP_CORNERS.items() if set(s) <= set(chosen))
    slots, left, used = [], set(chosen), {s: 0 for s in CLIP_SIDES}

    def take(anchor):
        slots.append(anchor)
        for side in CLIP_CORNERS.get(anchor, (anchor,)):
            used[side] += 1
            left.discard(side)
        if anchor in corners:
            corners.remove(anchor)

    while left:
        ready = [k for k in corners if set(CLIP_CORNERS[k]) <= left]
        if ready and rng.random() < corner_rate:
            take(ready[int(rng.integers(len(ready)))])
        else:
            take(sorted(left)[int(rng.integers(len(left)))])
    while len(slots) < count:
        pool = [a for a in chosen + corners
                if all(used[s] < 2 for s in CLIP_CORNERS.get(a, (a,)))]
        if not pool:
            break
        take(pool[int(rng.integers(len(pool)))])
    return slots


def _clip_body(rng, bh, bw, tries=3):
    """A body for a cut figure, box-preserving so the planned extent really shows."""
    kinds = [k for k in BODY_KINDS if BODY_MIN_SIDE[k] <= min(bh, bw)]
    for _ in range(tries if kinds else 0):
        kind = _weighted(rng, kinds, [BODY_WEIGHTS[k] for k in kinds])
        mask = _raw_body(rng, kind, bh, bw)
        if mask is not None:
            return _orient(mask, 2 * int(rng.integers(2)), bool(rng.integers(2))), kind
    return np.ones((bh, bw), dtype=bool), "rect"     # a plain block always works


def _clip_place(scene, anchor, ring_target, color, tries=11):
    """One body cut by the border at ``anchor``, aiming to own ``ring_target`` ring cells.

    Later tries ask for less: a figure that cannot have the whole span it wanted
    still belongs in the scene, and the caller tops the ring up afterwards.
    """
    rng, h, w = scene.rng, scene.h, scene.w
    sides = CLIP_CORNERS.get(anchor, (anchor,))
    free = scene.free(0, moat=True)
    head_room = min(scene.ring_per_color * scene.ring_n - scene.ring_used.get(color, 0),
                    scene.ring_total * scene.ring_n - sum(scene.ring_used.values()))
    ring_target = max(2.0, min(ring_target, head_room))
    cut_rows = "top" in sides or "bottom" in sides
    cut_cols = "left" in sides or "right" in sides
    best = None
    for attempt in range(tries):
        asked = max(2.0, ring_target * (1.0 - .6 * attempt / max(1, tries - 1)))
        if cut_rows and cut_cols:                             # a corner: the ring cells split
            want_r = max(2, int(round((asked + 1) * rng.uniform(.35, .65))))
            want_c = max(2, int(round(asked + 1 - want_r)))
        elif cut_rows:
            want_c = max(2, int(round(asked * rng.uniform(.95, 1.2))))
            want_r = max(2, int(round(_logu(rng, 2, max(2, .40 * h)))))
        else:
            want_r = max(2, int(round(asked * rng.uniform(.95, 1.2))))
            want_c = max(2, int(round(_logu(rng, 2, max(2, .40 * w)))))
        want_r, want_c = min(h - 3, want_r), min(w - 3, want_c)
        bh = min(h - 1, want_r + int(rng.integers(1, 5))) if cut_rows else want_r
        bw = min(w - 1, want_c + int(rng.integers(1, 5))) if cut_cols else want_c
        if min(bh, bw) < 2:
            continue
        mask, kind = _clip_body(rng, bh, bw)
        mh, mw = mask.shape
        # the extent that shows is taken from the body actually built, so a row always runs off
        vr = max(2, min(want_r, mh - 1)) if cut_rows else mh
        vc = max(2, min(want_c, mw - 1)) if cut_cols else mw
        if (cut_rows and vr >= mh) or (cut_cols and vc >= mw):
            continue
        vis = mask[mh - vr:, :] if "top" in sides else (mask[:vr, :] if "bottom" in sides else mask)
        vis = vis[:, mw - vc:] if "left" in sides else (vis[:, :vc] if "right" in sides else vis)
        vis = np.ascontiguousarray(vis)
        if int(vis.sum()) < CLIPPED_MIN_VISIBLE or int(vis.sum()) >= int(mask.sum()):
            continue                                          # it must really run off the canvas
        if not _connected(vis):        # cheap and seat-independent: test it before looking
            continue
        vh, vw = vis.shape
        spots = _placements(vis, free)                        # every seat clear of the moats
        if not len(spots):
            continue
        rows, cols = spots[:, 0], spots[:, 1]
        keep = (rows == 0) if "top" in sides else \
            ((rows == h - vh) if "bottom" in sides else ((rows >= 1) & (rows + vh <= h - 1)))
        keep &= (cols == 0) if "left" in sides else \
            ((cols == w - vw) if "right" in sides else ((cols >= 1) & (cols + vw <= w - 1)))
        spots = spots[keep]
        if not len(spots):
            continue
        r0, c0 = map(int, spots[int(rng.integers(len(spots)))])
        cells = np.zeros((h, w), dtype=bool)
        cells[r0:r0 + vh, c0:c0 + vw] = vis
        if cells[0].all() or cells[-1].all() or cells[:, 0].all() or cells[:, -1].all():
            continue                                          # never a whole side
        touched = {name for name, hit in (("top", cells[0].any()), ("bottom", cells[-1].any()),
                                          ("left", cells[:, 0].any()), ("right", cells[:, -1].any())) if hit}
        if touched != set(sides):
            continue
        # naturalness on a window around the figure: a one-cell ground margin where the crop
        # is interior, and the canvas border itself where it is not, so both readings are right
        crop = cells[max(0, r0 - 1):min(h, r0 + vh + 1), max(0, c0 - 1):min(w, c0 + vw + 1)]
        if hairs(crop, edge=True) or notches(crop, corners=True):
            continue
        if not scene.ring_ok(cells, color):
            continue
        on_ring = int((cells & scene.ring).sum())
        score = (abs(on_ring - ring_target), -int(cells.sum()))
        if best is None or score < best[0]:
            best = (score, cells, kind, len(touched))
        if not score[0]:
            break
    if best is None:
        return None
    _score, cells, kind, n_sides = best
    return scene.paint(cells, color, kind, clipped_sides=n_sides)


def _separated_ground(scene):
    """The ground is ONE field reaching every side that every figure sits on (certificate ii)."""
    ground = ~scene.occupied
    if not ground.any():
        return False
    labels, _ = label_same_color(ground.astype(np.int64))
    parts = np.unique(labels[scene.ring & ground])
    if len(parts) != 1:
        return False
    field = labels == parts[0]
    if not (field[0].any() and field[-1].any() and field[:, 0].any() and field[:, -1].any()):
        return False
    if int(field.sum()) <= int((ground & ~field).sum()):        # no pocket outweighs the field
        return False
    near = np.zeros(field.shape, dtype=bool)
    near[1:, :] |= field[:-1, :]
    near[:-1, :] |= field[1:, :]
    near[:, 1:] |= field[:, :-1]
    near[:, :-1] |= field[:, 1:]
    marks, _ = label_same_color(scene.occupied.astype(np.int64))
    return (set(np.unique(marks[scene.occupied]).tolist())
            == set(np.unique(marks[scene.occupied & near]).tolist()))


def _clipped_ok(scene):
    """Everything the heavily-clipped archetype claims, measured on the built scene."""
    on_ring = sum(scene.ring_used.values())
    if on_ring <= scene.ring_n / 2 or on_ring < CLIPPED_RING_MIN * scene.ring_n:
        return False                                  # the figures must out-own the ground there
    cut = [f for f in scene.figures if f["clipped_sides"] > 0]
    if len(cut) < 2 or len(scene.figures) < CLIPPED_MIN_FIGURES:
        return False
    if min(int(f["cells"].sum()) for f in scene.figures) < CLIPPED_MIN_VISIBLE:
        return False
    ground = int((~scene.occupied).sum())
    if ground < CLIPPED_GROUND_SHARE * scene.h * scene.w:
        return False
    rival = max(int((scene.a == col).sum())
                for col in {f["color"] for f in scene.figures})
    return ground >= CLIPPED_MASS_RATIO * rival and _separated_ground(scene)


def _clipped_scene(scene, plan, colors):
    """Several bodies cut by the border, each on the ground, together owning most of the ring."""
    rng = scene.rng
    want = plan["ring_target"] * scene.ring_n
    # a corner seat is fixed, an edge seat can slide: place the tied-down figures first
    slots = sorted(plan["slots"], key=lambda a: a not in CLIP_CORNERS)
    used = {a: 1 for a in slots if a in CLIP_CORNERS}
    cut = 0

    def next_color():
        # spread the ring between colours: the one holding least of it goes next
        return min(colors, key=lambda col: (scene.ring_used.get(col, 0), colors.index(col)))

    for i, anchor in enumerate(slots):
        left = want - sum(scene.ring_used.values())
        target = max(2.0, left / max(1, len(slots) - i))
        cut += _clip_place(scene, anchor, target, next_color()) is not None
    # the figures must out-own the ground on the ring, so top up wherever there is still room
    pool = list(plan["clip_sides"]) + [k for k, s in CLIP_CORNERS.items()
                                       if set(s) <= set(plan["clip_sides"]) and k not in used]
    for _ in range(4):
        need = CLIPPED_RING_MIN * scene.ring_n - sum(scene.ring_used.values())
        if need <= 0 or not pool:
            break
        anchor = pool[int(rng.integers(len(pool)))]
        if _clip_place(scene, anchor, need, next_color()) is not None:
            cut += 1
            if anchor in CLIP_CORNERS:
                pool.remove(anchor)
        else:
            pool.remove(anchor)
    return cut >= 2


# ---------------------------------------------------------------------------
# archetypes
# ---------------------------------------------------------------------------

def _plan(rng, h, w):
    area = h * w
    archetype = _weighted(rng, ARCHETYPES, ARCHETYPE_WEIGHTS)
    plan = {"archetype": archetype, "area": area,
            "n_colors": int(_weighted(rng, list(COLOR_COUNT_WEIGHTS), list(COLOR_COUNT_WEIGHTS.values()))),
            "textured": bool(rng.random() < TEXTURE_SHARE),
            "texture_kind": str(rng.choice(TEXTURE_KINDS)),
            "texture_colors": int(_weighted(rng, (1, 2), (.7, .3))),
            "detail_rate": float(rng.uniform(.15, .45)),
            "spread": bool(rng.random() < .8)}
    if archetype == "motif_row":
        # a row needs room for at least three copies of the motif's minimum box
        kind = str(rng.choice(MOTIF_KINDS))
        side = BODY_MIN_SIDE[kind]
        vertical = bool(rng.random() < .35)
        span, thick = (h, w) if vertical else (w, h)
        if (span - 2 + 1) // (side + 1) < 3 or thick - 2 < side:
            vertical = not vertical
            span, thick = (h, w) if vertical else (w, h)
        top = min(6, (span - 2 + 1) // (side + 1))
        if top < 3 or thick - 2 < side:
            archetype = plan["archetype"] = "few_large"
        else:
            plan["vertical"] = vertical
            plan["count"] = int(rng.integers(3, top + 1))
            plan["kind"] = kind
            plan["mono"] = bool(rng.random() < .4)
    if archetype == "crowd":
        # count from moat footprints (a 2x2 chip with its halo takes ~11 cells), with slack
        # for placement without backtracking
        top = min(16, int(.60 * (h - 2) * (w - 2) / 10.9))
        if top < 3:
            archetype = plan["archetype"] = "few_large"
    if archetype == "dominant_figure":
        # A big body inside a uniform margin can also be read as a panel with a drawn border,
        # so a dominant figure comes one of two ways: alone inside a margin too fat to read as
        # a border, or with a second figure beside it, outside its bounding box. Only the
        # second lets the ground drop under half the canvas: what proves the ground is the
        # ground is then that other figure sitting on it, plus the off-centre margin.
        fat = border_look(h, w) + 1 + int(rng.integers(0, 2))
        floor_m = ground_margin(h, w)
        wide = (h - 2 * fat) * (w - 2 * fat) >= .34 * area
        # the band holds a body two cells across once the two outer rings and the moat are out,
        # and is wide enough that the margins it leaves read as off-centre (clause (j))
        narrow = max(COMPANION_BAND, 5 - floor_m)
        band = narrow + int(rng.integers(0, 3))
        rooms, roomy = {}, []
        for side in ("top", "bottom", "left", "right"):
            bh = h - 2 * floor_m - (band if side in ("top", "bottom") else 0)
            bw = w - 2 * floor_m - (band if side in ("left", "right") else 0)
            if bh >= 4 and bw >= 4 and bh * bw >= .34 * area:
                rooms[side] = bh * bw
            for wide_band in range(narrow, narrow + 3):
                vertical = side in ("top", "bottom")
                bh = h - 2 * floor_m - (wide_band if vertical else 0)
                bw = w - 2 * floor_m - (0 if vertical else wide_band)
                # ... and a body that is to outweigh the ground needs about half the canvas,
                # with the band holding the companions that make up the rest
                strip = (floor_m + wide_band - 1) * ((w - 2) if vertical else (h - 2))
                if (bh >= 4 and bw >= 4 and bh * bw >= .47 * area
                        and area - 2 * .95 * bh * bw + 1 <= .70 * strip):
                    roomy.append((side, wide_band))
        if roomy and rng.random() < .95:
            side, wide_band = roomy[int(rng.integers(len(roomy)))]
            plan["mode"] = "accompanied"
            plan["side"] = side
            plan["band"] = wide_band
            plan["margin"] = floor_m
            plan["nonmodal"] = True
            plan["scale"] = float(rng.uniform(.97, 1.0))
            plan["n_colors"] = max(3, plan["n_colors"])
        elif wide and (not rooms or rng.random() < .45):
            plan["mode"] = "wide_margin"
            plan["margin"] = fat
            plan["scale"] = float(rng.uniform(.88, 1.0))
        elif rooms:
            plan["mode"] = "accompanied"
            plan["side"] = str(rng.choice(sorted(rooms)))
            plan["band"] = band
            plan["margin"] = floor_m
            plan["nonmodal"] = False
            plan["scale"] = float(rng.uniform(.75, 1.0))
            plan["n_colors"] = max(3, plan["n_colors"])
        else:
            archetype = plan["archetype"] = "few_large"
    if archetype == "dominant_figure":
        plan["count"] = 1
    elif archetype == "few_large":
        top = 2 if area < 150 else (3 if area < 250 else 4)
        plan["count"] = min(top, int(_weighted(rng, (2, 3, 4), (.4, .4, .2))))
        plan["extras"] = int(rng.integers(0, 4)) if rng.random() < .4 and area >= 250 else 0
    elif archetype == "crowd":
        plan["count"] = int(rng.integers(min(4, top), top + 1))
    elif archetype == "cluster":
        plan["count"] = int(rng.integers(2, 6))          # attached figures
        plan["extras"] = int(rng.integers(0, 3)) if rng.random() < .5 else 0
        plan["near_rate"] = float(rng.uniform(0, .5))
        plan["n_colors"] = max(3, plan["n_colors"])      # touching figures differ in colour
    elif archetype == "border_cut":
        plan["count"] = int(_weighted(rng, (1, 2, 3), (.5, .35, .15)))
        plan["extras"] = int(rng.integers(0, 4))
    elif archetype == "spanning_bars":
        plan["count"] = int(_weighted(rng, (1, 2), (.6, .4))) if min(h, w) >= 12 else 1
        plan["extras"] = int(rng.integers(1, 5 if area >= 150 else 3))
        plan["crossing"] = bool(rng.random() < .45)
    elif archetype == "heavily_clipped":
        # cut bodies need room for their moats, so a small canvas clips fewer, cornerwise
        ring_n = 2 * w + 2 * (h - 2)
        top = 3 if area < 200 else (4 if area < 450 else 5)
        sides = _clip_sides(rng, h, w, ring_n)
        slots = _clip_slots(rng, sides, min(top, int(_weighted(rng, (3, 4, 5), (.45, .35, .20)))),
                            corner_rate=.55 if area >= 200 else .85)
        plan["clip_sides"] = sides
        plan["slots"] = slots
        plan["n_sides"] = len(sides)
        plan["count"] = len(slots)
        plan["ring_target"] = min(float(rng.uniform(.54, .61)),
                                  .85 * _clip_capacity(sides, h, w) / ring_n)
        plan["extras"] = int(rng.integers(0, 4 if area >= 300 else 2))
        plan["n_colors"] = max(4, plan["n_colors"])
    # figure colours: leave room for a texture colour and (from four colours) one detail
    target = plan["n_colors"]
    n_texture = plan["texture_colors"] if plan["textured"] else 0
    floor = 3 if archetype == "heavily_clipped" else (
        2 if archetype == "cluster" or plan.get("mode") == "accompanied" else 1)
    plan["n_fig"] = max(floor, min(9, target - 1 - n_texture - (1 if target >= 4 else 0)))
    if archetype == "few_large" and plan["n_fig"] == 1:
        plan["count"] = min(plan["count"], 3)          # one colour: the dominant figure's margin
        plan["extras"] = 0
    return plan


def _footprint(cells):
    """Cells of a body plus its one-cell moat, for a compact body of ~75% box fill."""
    side = math.sqrt(cells / .75)
    return (side + 1) ** 2


def _fit(sizes, budget):
    """Scale cell targets down until their moat footprints fit the free interior."""
    sizes = [max(4.0, float(s)) for s in sizes]
    for _ in range(6):
        total = sum(_footprint(s) for s in sizes)
        if total <= budget:
            break
        sizes = [max(4.0, s * (budget / total)) for s in sizes]
    return sizes


def _coverage_need(h, w):
    """Figure cells this canvas must carry; small canvases carry what they can."""
    return COVERAGE_FLOOR * h * w if h * w >= COVERAGE_MIN_AREA else 0.0


def _ensure_coverage(sizes, need, budget):
    """Scale cell targets up until they add up to ``need``; placement still bounds them."""
    total = sum(sizes)
    if need <= 0 or not sizes or total >= need:
        return sizes
    return _fit([s * need / total for s in sizes], budget)


def _budget(h, w):
    """Free interior a scene can spend on bodies and their moats."""
    return .85 * (h - 2) * (w - 2)


def _sizes(rng, plan, h, w):
    """Log-uniform target cell counts per figure, drawn per attempt within the plan."""
    area = plan["area"]
    kind = plan["archetype"]
    budget = _budget(h, w)
    need = COVERAGE_PLAN * _coverage_need(h, w)
    if kind == "few_large" and plan["n_fig"] == 1:
        m = ground_margin(h, w)                        # placed inside a dominant figure's margin
        budget = .60 * max(1, h - 2 * m) * max(1, w - 2 * m)
    if kind == "few_large":
        sizes = [_logu(rng, .07 * area, .40 * area) for _ in range(plan["count"])]
        if plan["spread"] and max(sizes) < 2 * min(sizes):
            sizes[int(np.argmin(sizes))] = max(sizes) / rng.uniform(2, 4)
        total = sum(sizes)
        if total > .72 * area:
            sizes = [s * .72 * area / total for s in sizes]
        sizes += [_logu(rng, 4, max(4, .04 * area)) for _ in range(plan.get("extras", 0))]
        sizes = _ensure_coverage(_fit(sizes, budget), need, budget)
    elif kind == "crowd":
        sizes = [_logu(rng, 4, max(8, .15 * area)) for _ in range(plan["count"])]
        sizes[0] = max(sizes[0], .06 * area)
        if plan["count"] >= 5:
            sizes[-1] = int(rng.integers(4, 9))
            sizes[-2] = int(rng.integers(4, 9))
        total = sum(sizes)
        if total > .50 * area:
            sizes = [max(4, s * .50 * area / total) for s in sizes]
        sizes = _ensure_coverage(_fit(sizes, budget), need, budget)
    elif kind == "cluster":
        core = _logu(rng, .10 * area, .30 * area)
        sizes = [core] + [_logu(rng, 4, max(4, .6 * core)) for _ in range(plan["count"])]
        total = sum(sizes)
        if total > .70 * area:
            sizes = [max(4, s * .70 * area / total) for s in sizes]
        sizes += [_logu(rng, 4, max(4, .08 * area)) for _ in range(plan.get("extras", 0))]
        sizes = _ensure_coverage(_fit(sizes, budget), need, budget)
    elif kind == "border_cut":
        # a cut figure shows about half of its body, so the row of them is drawn larger
        lo = max(.06 * area, need / plan["count"] / .55)
        sizes = [_logu(rng, lo, max(lo, .35 * area)) for _ in range(plan["count"])]
        sizes += _fit([_logu(rng, 4, max(4, .10 * area)) for _ in range(plan["extras"])], .5 * budget)
    elif kind == "spanning_bars":
        sizes = _fit([_logu(rng, 4, max(4, .12 * area)) for _ in range(plan["extras"])], .6 * budget)
    elif kind == "heavily_clipped":
        # the cut figures are sized from their ring targets; these are the interior ones
        sizes = _fit([_logu(rng, 4, max(4, .07 * area)) for _ in range(plan["extras"])], .30 * budget)
    else:
        sizes = []                                     # motif rows size copies from the row's room
    return [max(4.0, float(s)) for s in sizes]


def _build(rng, plan, h, w, ground, colors):
    """One construction attempt; returns the scene or None when placement fails."""
    arch = plan["archetype"]
    scene = (_Scene(rng, h, w, ground, CLIPPED_RING_TOTAL, CLIPPED_RING_PER_COLOR)
             if arch == "heavily_clipped" else _Scene(rng, h, w, ground))
    fig_colors = list(colors)
    rng.shuffle(fig_colors)
    palette = fig_colors[:]                       # cycled over figures
    sizes = _sizes(rng, plan, h, w)
    k = 0

    def next_color():
        nonlocal k
        color = palette[k % len(palette)]
        k += 1
        return color

    if arch == "dominant_figure":
        m = plan["margin"]
        # the room the body may use: the whole margin box, or that box less the band the
        # companion figure keeps beside it (which is what puts the body off-centre)
        top, bottom, left, right = m, h - m, m, w - m
        region = None
        if plan.get("mode") == "accompanied":
            side, band = plan["side"], plan["band"]
            if side == "top":
                top += band
            elif side == "bottom":
                bottom -= band
            elif side == "left":
                left += band
            else:
                right -= band
            region = np.zeros((h, w), dtype=bool)
            region[top:bottom, left:right] = True
        room_h, room_w = bottom - top, right - left
        if plan["n_colors"] == 2:      # clause (h): on two colours no figure spans the canvas
            room_h, room_w = min(room_h, int(TWO_COLOR_SPAN * h)), min(room_w, int(TWO_COLOR_SPAN * w))
        bh = int(round(rng.uniform(plan["scale"], 1.0) * room_h))
        bw = int(round(rng.uniform(plan["scale"], 1.0) * room_w))
        # a body that is to outweigh the ground has to hold about half the canvas by itself
        need = (.47 if plan.get("nonmodal") else DOMINANT_BODY_SHARE) * h * w
        # BODY_FILL is the average a kind reaches, so kinds a little under the need are still
        # worth trying: the loop below keeps only a body that really is that big
        kinds = [x for x in BODY_KINDS if BODY_MIN_SIDE[x] <= min(bh, bw)
                 and BODY_FILL[x] * bh * bw >= .85 * need]
        if not kinds:
            return None
        body = kind = None
        for _ in range(10):                # the label means a dominant body, so insist on one
            kind = _weighted(rng, kinds, [DOMINANT_WEIGHTS[x] for x in kinds])
            body = _raw_body(rng, kind, bh, bw)
            if body is not None and body.sum() >= need:
                break
            body = None
        if body is None:
            return None
        body = _orient(body, int(rng.integers(4)), bool(rng.integers(2)))
        if body.shape[0] > room_h or body.shape[1] > room_w:
            return None
        if region is not None and plan.get("nonmodal"):
            # flush against the far edge, so the whole band shows on one side: the margins
            # then differ by the band, which is what makes the figure read as off-centre
            if side == "top":
                region[:bottom - body.shape[0], :] = False
            elif side == "bottom":
                region[top + body.shape[0]:, :] = False
            elif side == "left":
                region[:, :right - body.shape[1]] = False
            else:
                region[:, left + body.shape[1]:] = False
        if scene.place(body, next_color(), kind, margin=m, inside=region) is None:
            return None
        if region is not None and not _companion(scene, plan, next_color):
            return None
    elif arch in ("few_large", "crowd", "border_cut", "cluster", "spanning_bars", "heavily_clipped"):
        if arch == "heavily_clipped":
            if not _clipped_scene(scene, plan, palette):
                return None
        if arch == "spanning_bars":
            if not _bars(scene, plan, next_color):
                return None
            # the bars are drawn first; the loose figures make up the rest of the coverage
            sizes = _ensure_coverage(sizes, COVERAGE_PLAN * _coverage_need(h, w) - int(scene.occupied.sum()),
                                     .6 * _budget(h, w))
        if arch == "border_cut":
            done = 0
            for done, cells in enumerate(sizes[:plan["count"]]):
                sides = list(rng.permutation(["top", "bottom", "left", "right"])[:int(_weighted(rng, (1, 2), (.65, .35)))])
                # most of the coverage comes from the cut figures: keep enough of each on canvas
                show = .7 * (COVERAGE_PLAN * _coverage_need(h, w) - int(scene.occupied.sum())) / (plan["count"] - done)
                figure = None
                for shrink in (1.0, .6, .35):     # a small canvas has little ring to give away
                    got = _figure(rng, max(4, cells * shrink), h, w)
                    if got is None:
                        continue
                    mask, kind = got
                    figure = scene.place_clipped(mask, next_color(), kind, sides, min_cells=show * shrink)
                    if figure is not None:
                        break
                if figure is None:
                    if not done or not plan.get("relax"):
                        return None               # a border_cut scene needs a cut figure
                    break                         # the rest of the row becomes loose figures
            # top the extras up with whatever the cut figures left past the border
            sizes = _ensure_coverage(sizes[plan["count"]:],
                                     COVERAGE_PLAN * _coverage_need(h, w) - int(scene.occupied.sum()),
                                     .5 * _budget(h, w))
        if arch == "cluster":
            if not _cluster(scene, plan, sizes, fig_colors):
                return None
            sizes = sizes[1 + plan["count"]:]
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
        # figures of one colour can outnumber the ground: keep the margin a dominant figure
        # needs, so the widest of them never reads as a panel inside a drawn border
        margin = ground_margin(h, w) if (arch == "few_large" and len(palette) == 1) else 1
        for i in order:
            got = _figure(rng, sizes[i], h - 2 * margin, w - 2 * margin)
            if got is None:
                return None
            mask, kind = got
            color = next_color()
            # the cut figures own the ring of a heavily clipped scene: the loose ones stay inside
            on_ring = margin == 1 and arch != "heavily_clipped" and rng.random() < .2
            figure = scene.place(mask, color, kind, margin=0) if on_ring else None
            if figure is None:
                figure = scene.place(mask, color, kind, margin=margin)
            if figure is None and arch == "crowd" and mask.sum() > 4:
                # a crowd keeps its planned count: the figure that no longer fits becomes a chip
                figure = scene.place(np.ones((2, 2), dtype=bool), color, "rect", margin=margin)
            if figure is None:
                # a plan the canvas cannot hold seats what it can rather than never landing,
                # but only once the attempts have shown that the full count does not fit
                if plan.get("relax") and scene.figures:
                    continue
                return None
    else:  # motif_row
        if not _motif_row(scene, plan, sizes, next_color):
            return None
    if int(scene.occupied.sum()) < _coverage_need(h, w):
        return None                       # a roomy canvas is never three chips in a sea of ground
    if arch == "heavily_clipped" and not _clipped_ok(scene):
        return None
    return scene


def _companion(scene, plan, next_color):
    """The figure that keeps a dominant one company, in the band outside its bounding box.

    It is what shows the ground carrying figures rather than framing a panel, so it is a body
    of its own (never a texture dot) and the first of them takes the next colour of the palette,
    which the dominant figure does not use. It may run up to the border -- the ground still owns
    two thirds of the ring by ``ring_ok`` -- which also stops the frame reading as a border.
    """
    rng, h, w = scene.rng, scene.h, scene.w
    side, band, m = plan["side"], plan["band"], plan["margin"]
    strip = np.zeros((h, w), dtype=bool)
    across = m + band - 1            # the band less the one-cell moat around the body
    if side == "top":
        strip[:m + band, :] = True
        max_h, max_w = across, w - 2
    elif side == "bottom":
        strip[h - m - band:, :] = True
        max_h, max_w = across, w - 2
    elif side == "left":
        strip[:, :m + band] = True
        max_h, max_w = h - 2, across
    else:
        strip[:, w - m - band:] = True
        max_h, max_w = h - 2, across
    if min(max_h, max_w) < 2:
        return False
    room = max_h * max_w
    # the ground outnumbers the body until the companions have made up the difference, so
    # beside a body that is to outweigh it the companion is a body too, never a chip
    short = h * w - 2 * int(scene.figures[0]["cells"].sum()) + 1 if plan.get("nonmodal") else 0
    placed = 0
    for _ in range(6):
        lo = max(COMPANION_CELLS, min(short, .70 * room)) if short > 0 else COMPANION_CELLS
        got = _figure(rng, _logu(rng, lo, max(lo, .70 * room if short > 0 else .55 * room)),
                      max_h, max_w)
        if got is not None and scene.place(got[0], next_color(), got[1], margin=0, inside=strip):
            placed += 1
            short -= int(got[0].sum())
            if placed == 3 or (short <= 0 and rng.random() < .6):
                break                    # another body in the band while it still has room
    return placed > 0 and short <= 0


def _bars(scene, plan, next_color):
    rng, h, w = scene.rng, scene.h, scene.w
    used = []
    for i in range(plan["count"]):
        vertical = bool(rng.integers(2)) if i == 0 else (not used[0] if plan["crossing"] else used[0])
        across = w if vertical else h                 # a vertical bar is placed along the width
        t = int(_weighted(rng, (1, 2, 3), (.45, .35, .20)))
        for _ in range(10):
            start = int(rng.integers(2, across - t - 1))
            cells = np.zeros((h, w), dtype=bool)
            if vertical:
                cells[:, start:start + t] = True
            else:
                cells[start:start + t, :] = True
            # parallel bars keep a gap of at least three cells; crossing bars may overlap
            if not plan["crossing"] and i and (halo(halo(halo(scene.occupied))) & cells).any():
                continue
            if plan["crossing"] and i:
                cells &= ~scene.occupied
            figure = scene.paint(cells, next_color(), "bar", spanning=True)
            if figure is not None:
                used.append(vertical)
                break
        else:
            return False
    return True


def _cluster(scene, plan, sizes, fig_colors):
    rng, h, w = scene.rng, scene.h, scene.w
    got = _figure(rng, sizes[0], h - 2, w - 2)
    if got is None:
        return False
    mask, kind = got
    core = scene.place(mask, fig_colors[0], kind, margin=1)
    if core is None:
        return False
    attached = 0
    for i, cells in enumerate(sizes[1:1 + plan["count"]]):
        got = _figure(rng, cells, h - 2, w - 2)
        if got is None:
            if plan.get("relax"):
                continue                           # a crowded little canvas takes what it can
            return False
        mask, kind = got
        options = fig_colors[1:] + fig_colors[:1]
        rng.shuffle(options)
        used = {f["color"] for f in scene.figures}
        options.sort(key=lambda col: col in used)   # prefer fresh colours to reach the palette
        # the label means a touching cluster: the first neighbour always touches the core
        near = i > 0 and rng.random() < plan["near_rate"]
        seated = scene.attach(mask, options, kind, near=near) is not None
        if not seated and not plan.get("relax"):
            return False
        attached += seated
    return attached >= 1                           # a cluster needs at least one real neighbour


def _motif_row(scene, plan, sizes, next_color):
    """Copies of one motif family along a row (or column): mixed sizes, jittered spacing."""
    rng, h, w = scene.rng, scene.h, scene.w
    kind, count, vertical = plan["kind"], plan["count"], plan["vertical"]
    rot, flip = int(rng.integers(4)), bool(rng.integers(2))
    span, thick = (h, w) if vertical else (w, h)
    side = BODY_MIN_SIDE[kind]
    room = (span - 2 - (count - 1)) / count           # along-axis length per copy, one-cell gaps
    base_along = rng.uniform(side, max(side, min(room, 2.5 * side + 4)))
    base_across = rng.uniform(side, max(side, min(thick - 2, 2.5 * side + 4)))
    # a row on a roomy canvas carries bigger copies, not the same chips in more space
    box = COVERAGE_PLAN * _coverage_need(h, w) / count / BODY_FILL[kind]
    for _ in range(2):
        grow = math.sqrt(box / max(1e-9, base_along * base_across))
        if grow <= 1:
            break
        base_along = min(room, base_along * grow)
        base_across = min(thick - 2, base_across * grow)
    bodies = []
    for _ in range(count):
        along = int(round(base_along * rng.uniform(.7, 1.3)))
        across = int(round(base_across * rng.uniform(.7, 1.3)))
        along = max(side, min(int(room), along))
        across = max(side, min(thick - 2, across))
        # choose the pre-orientation box so that the oriented body lies along the row
        bh, bw = ((across, along) if not vertical else (along, across))
        if rot % 2:
            bh, bw = bw, bh
        body = None
        for _ in range(8):
            body = _raw_body(rng, kind, bh, bw)
            if body is not None:
                break
        if body is None:
            return False
        bodies.append(_orient(body, rot, flip))
    mono = next_color() if plan.get("mono") else None
    total = sum(b.shape[0] if vertical else b.shape[1] for b in bodies)
    if total + (len(bodies) - 1) > span - 2:
        return False
    # jittered spacing: the leading offset and every extra gap cell come out of the slack
    slack = span - 2 - total - (len(bodies) - 1)
    lead = int(rng.integers(0, min(slack, 3) + 1))
    slack -= lead
    cursor = 1 + lead
    centre = int(rng.integers(max(2, thick // 3), max(3, thick - thick // 3)))
    for body in bodies:
        along = body.shape[0] if vertical else body.shape[1]
        across = body.shape[1] if vertical else body.shape[0]
        for _ in range(6):
            off = centre - across // 2 + int(rng.integers(-2, 3))
            off = max(1, min(thick - 1 - across, off))
            cells = np.zeros((h, w), dtype=bool)
            if cursor + along > span - 1 or off < 1 or off + across > thick - 1:
                return False
            if vertical:
                cells[cursor:cursor + along, off:off + across] = body
            else:
                cells[off:off + across, cursor:cursor + along] = body
            if (cells & halo(scene.occupied)).any():
                continue
            if scene.paint(cells, mono if mono is not None else next_color(), kind) is not None:
                break
        else:
            return False
        extra = int(rng.integers(0, min(slack, 2) + 1))
        slack -= extra
        cursor += along + 1 + extra
    return True


# ---------------------------------------------------------------------------
# palette, details, texture
# ---------------------------------------------------------------------------

def _dress(rng, scene, plan, ground, n_colors_target, fresh):
    """Inner details, nested figures and texture; returns (texture_kind, texture_colors)."""
    used = {ground} | {f["color"] for f in scene.figures}
    budget = max(0, n_colors_target - len(used))          # colours still to introduce
    fresh = [c for c in fresh if c not in used][:budget]
    textured = plan["textured"] and min(scene.h, scene.w) >= 8
    n_texture = min(plan["texture_colors"], len(fresh)) if textured else 0
    texture_fresh = fresh[len(fresh) - n_texture:] if n_texture else []
    fresh = fresh[:len(fresh) - n_texture]
    if n_texture == 0:
        textured = False                                 # texture colours never equal figure colours
    need = len(fresh)
    bodies = [f for f in scene.figures
              if f["cells"].sum() >= DETAIL_MIN_CELLS and not f["nested"] and f["kind"] != "bar"]
    rng.shuffle(bodies)
    ground_cells = int((scene.a == ground).sum())
    rival_cells = max([int((scene.a == f["color"]).sum()) for f in scene.figures] or [0])
    # a figure colour already outnumbers the ground: the scene has to read as ONE dominant
    # figure, so its body keeps a single detail and stays mostly its own colour
    dominant = rival_cells >= ground_cells
    detail_cap = DOMINANT_DETAIL_CAP if dominant else DETAIL_CAP
    if plan["archetype"] == "dominant_figure":
        # keep a dominant body ahead of the ground when it already outnumbers it
        body_cells = int((scene.a == scene.figures[0]["color"]).sum())
        detail_cap = min(detail_cap, .20 if body_cells <= ground_cells else
                         max(.04, min(.20, (body_cells - ground_cells) / body_cells - .03)))
    # big figures can carry two details (very big ones three) within that one budget, so the
    # palette can grow without clutter; a dominant body carries exactly one
    slots = []
    for figure in bodies:
        cells = int(figure["cells"].sum())
        slots += [figure] * (1 if dominant else (3 if cells >= 60 else (2 if cells >= 30 else 1)))
    # a figure nested in a ground pocket is walled off from the field, which the separating
    # certificate of a heavily clipped scene forbids: those frames keep their pockets empty
    pockets = [] if plan["archetype"] == "heavily_clipped" else \
        [f for f in scene.figures if f["kind"] in ("frame", "window") and not f["nested"]]
    n_details = sum(rng.random() < plan["detail_rate"] for _ in slots)
    n_details = min(len(slots), max(n_details, need))
    # nested figures inside pockets (some fraction of frames / windows)
    for figure in pockets:
        if rng.random() < .5:
            colour = (fresh.pop(0) if fresh and need > 0 else
                      rng.choice([c for c in used if c != ground and c != figure["color"]] or [figure["color"]]))
            nested = scene.nest(figure, _logu(rng, 4, max(4, .5 * figure["cells"].sum())), int(colour))
            if nested is not None and int(colour) not in used:
                used.add(int(colour))
                need -= 1
    for figure in slots[:n_details]:
        if fresh:
            colour = fresh.pop(0)
        else:
            # a reused detail colour must not match a figure within two cells of this one
            near = halo(halo(figure["cells"]))
            nearby = {f["color"] for f in scene.figures if f is not figure and (near & f["cells"]).any()}
            nearby |= {d["color"] for f in scene.figures if f is not figure and (near & f["cells"]).any()
                       for d in f.get("details", [])}
            others = [c for c in used if c != ground and c != figure["color"] and c not in nearby]
            if not others:
                continue
            colour = int(rng.choice(others))
        # a dominant body takes an inner block only: a band or a cap would cut it in two
        if scene.add_detail(figure, int(colour), core_only=bool(figure["touches"]) or dominant,
                            max_fraction=detail_cap):
            used.add(int(colour))
        elif colour not in used:
            fresh.insert(0, int(colour))
    texture_kind, texture_colors = "plain", []
    if textured:
        texture_colors = list(texture_fresh)
        if texture_colors:
            placed = _texture(rng, scene, plan["texture_kind"], texture_colors)
            if placed:
                texture_kind = plan["texture_kind"]
                texture_colors = placed
            else:
                texture_colors = []
    return texture_kind, texture_colors


def _texture(rng, scene, kind, colors):
    h, w = scene.h, scene.w
    inner = np.zeros((h, w), dtype=bool)
    inner[2:-2, 2:-2] = True
    for _ in range(3):                                   # a different phase may fit better
        dots = sparse_mask(rng, h, w, kind) & inner & ~halo(scene.occupied)
        points = np.argwhere(dots)
        if len(points) >= 3:
            break
    else:
        return None
    # dots stay a minority of the drawing: a scene is never mostly texture
    limit = min(max(3, int(TEXTURE_MAX_COVERAGE * h * w)),
                int(TEXTURE_FIGURE_SHARE * int(scene.occupied.sum())))
    if limit < 3:
        return None
    rng.shuffle(points)
    kept = np.zeros((h, w), dtype=bool)
    chosen = []
    for r, c in points[:limit]:
        if kept[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
            continue
        kept[r, c] = True
        chosen.append((int(r), int(c)))
    if len(chosen) < 3:
        return None
    used = set()
    for i, (r, c) in enumerate(chosen):
        colour = colors[i % len(colors)] if kind != "multi_speckle" else colors[int(rng.integers(len(colors)))]
        scene.a[r, c] = colour
        used.add(int(colour))
    scene.texture = kept
    return sorted(used)


# ---------------------------------------------------------------------------
# certificate, validation, metadata
# ---------------------------------------------------------------------------

def read_background(grid):
    """(ground colour, certificate path) from the visible grid alone (no metadata)."""
    try:
        return figure_ground_certificate(grid, panel_rules=True)
    except ValueError as error:
        raise ValueError(f"task25: no unambiguous figure/ground reading ({error})") from error


PANEL_BORDER_PERCENT = 12   # a uniform frame this thin beside the canvas reads as a drawn border


def border_look(h, w):
    """A uniform frame at most this thick reads as a drawn border, not a ground margin.

    Two cells (the shared library draws its own ``border`` field one or two thick), and an
    eighth of the shorter side once the canvas is big enough for that to be more: three cells
    of ground around a 25x25 body is a border, three cells around a 10x10 body is a margin.
    """
    return max(2, -(-min(h, w) * PANEL_BORDER_PERCENT // 100))


def panel_reading(grid):
    """The competing reading -- an inset panel with a drawn border -- or None.

    A person can read one colour as a border drawn around a panel rather than as
    the ground; the shared background library's own ``border`` field (one or two
    cells thick) looks exactly like that. The reading is available when a colour
    R fills the ENTIRE outer ring, every other colour lies strictly inside a
    single rectangle whose border is uniformly R -- the tightest one is the
    bounding box of the non-R cells, grown by one -- that frame is thin enough to
    read as a border rather than as a ground margin (at most ``border_look``
    cells on every side), and the rectangle holds a panel: one colour other than
    R covers more than half of it, reaches all four of its sides, and owns more
    of its ring than R does. The answer such a reader gives is that panel colour.

    No accepted example may offer this reading: ``validate_example`` raises when
    it does, and the tests hold the whole task to zero fires.
    """
    a = np.asarray(grid)
    if a.ndim != 2 or min(a.shape) < 3:
        return None
    h, w = a.shape
    ring = np.zeros((h, w), dtype=bool)
    ring[[0, -1], :] = True
    ring[:, [0, -1]] = True
    if len(np.unique(a[ring])) != 1:
        return None                            # no uniform frame: nothing reads as a border
    r = a[0, 0]
    inside = np.argwhere(a != r)
    if not len(inside):
        return None
    r0, c0 = inside.min(axis=0).tolist()
    r1, c1 = inside.max(axis=0).tolist()
    if max(r0, h - 1 - r1, c0, w - 1 - c1) > border_look(h, w):
        return None                            # a ground margin, not a drawn border
    panel = a[r0:r1 + 1, c0:c1 + 1]
    values, counts = np.unique(panel, return_counts=True)
    fill = values[int(np.argmax(counts))]
    if fill == r or int(counts.max()) <= .5 * panel.size:
        return None                            # no single fill: the ground shows through
    mask = panel == fill
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None                            # the fill does not reach the frame
    edge = np.zeros(panel.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    if int((panel[edge] == fill).sum()) <= int((panel[edge] == r).sum()):
        return None                            # R still lines the inside of the frame
    return fill.item() if hasattr(fill, "item") else fill


def infer_background(grid):
    """The visible ground colour by the figure/ground certificate (no metadata)."""
    return read_background(grid)[0]


def _literal_parts(cell):
    """4-connected same-value components of a {(r, c): value} map, as (value, set) pairs."""
    seen, parts = set(), []
    for start in cell:
        if start in seen:
            continue
        value = cell[start]
        part, pending = {start}, [start]
        seen.add(start)
        while pending:
            r, c = pending.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in cell and q not in seen and cell[q] == value:
                    seen.add(q)
                    part.add(q)
                    pending.append(q)
        parts.append((value, part))
    return parts


def literal_background(grid):
    """A second reading of the same certificate, written out literally in plain Python.

    Independent of ``figure_ground_certificate`` (no numpy, no shared helper):
    rows and columns are walked by hand so that ``validate_example`` can compare
    two implementations of the rule rather than one implementation with itself.
    Both readings of the rule are written out: the ring-majority path and the
    separating-field path, and exactly one colour must take one of them, with
    the scene clauses (h)-(j) that keep the picture from also reading as a panel
    inside a drawn border.
    """
    rows = [list(row) for row in grid]
    h = len(rows)
    w = len(rows[0]) if h else 0
    if h < 2 or w < 2 or any(len(row) != w for row in rows):
        raise ValueError("task25 (literal): a figure/ground reading needs a two-dimensional canvas")
    cell = {(r, c): rows[r][c] for r in range(h) for c in range(w)}
    if len(set(cell.values())) < 2:
        raise ValueError("task25 (literal): a single-colour canvas has no figure")
    ring = [p for p in cell if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    on_ring = {}
    for p in ring:
        on_ring[cell[p]] = on_ring.get(cell[p], 0) + 1
    edge = set(ring)
    parts = _literal_parts(cell)
    total = {}
    for value in cell.values():
        total[value] = total.get(value, 0) + 1

    def ring_majority(ground, ground_parts, ground_mass):
        if on_ring.get(ground, 0) < .60 * len(ring):                            # (a)
            raise ValueError("task25 (literal): no single colour owns the border ring")
        if any(k > .25 * len(ring) for v, k in on_ring.items() if v != ground):  # (b)
            raise ValueError("task25 (literal): a rival colour holds too much of the border ring")
        if len(ground_parts) > 4 or min(len(p) for p in ground_parts) < 4:      # (d)
            raise ValueError("task25 (literal): the ground breaks into small pieces along the border")
        mass = {}
        for value, part in parts:
            if part & edge and value != ground:
                mass[value] = mass.get(value, 0) + len(part)
        if ground_mass < 1.25 * max(list(mass.values()) or [0]):                # (e)
            raise ValueError("task25 (literal): another colour has comparable mass on the border")
        if total[ground] <= max(k for v, k in total.items() if v != ground):    # (g)
            # the clean two-cell margin of the certificate's own (g) is replaced here by the
            # scene clauses below: the DOMINANT figure keeps a proportional margin and company
            over = [v for v, k in total.items() if v != ground and k >= total[ground]]
            if len(over) != 1:
                raise ValueError("task25 (literal): several colours outweigh the ground")
            body = max([len(part) for value, part in parts if value == over[0]])
            if body < .60 * sum(k for v, k in total.items() if v != ground):
                raise ValueError("task25 (literal): the colours over the ground are not one figure")

    def scene_rules(ground, ground_parts):
        """Clauses (h)-(j): the picture must not also read as a panel inside a drawn border."""
        figures = {p: 0 for p in cell if cell[p] != ground}                     # colour-blind
        minority = total[ground] <= max(k for v, k in total.items() if v != ground)
        if len(set(cell.values())) > 2 and not minority and len(figures) < .25 * h * w:
            return                       # too little drawn for any figure to read as a panel
        blind = [part for _value, part in _literal_parts(figures)]
        if len(set(cell.values())) == 2:                                        # (h)
            if total[ground] < .50 * h * w:
                raise ValueError("task25 (literal): a two-colour picture needs a ground "
                                 "over half the canvas")
            for part in blind:
                if part & edge:
                    continue             # a figure the border cuts is not an inset panel
                rows = [r for r, _c in part]
                cols = [c for _r, c in part]
                if (max(rows) - min(rows) + 1 > .70 * h
                        or max(cols) - min(cols) + 1 > .70 * w):
                    raise ValueError("task25 (literal): a two-colour figure stretches "
                                     "across the canvas")
        body = max(blind, key=len)
        if len(body) < .60 * len(figures) or len(body) < .25 * h * w:
            if minority:
                raise ValueError("task25 (literal): a minority ground carries no one dominant figure")
            return                       # no figure dominates: nothing reads as a panel
        if body & edge:
            if minority:
                raise ValueError("task25 (literal): a dominant figure over a minority ground "
                                 "must sit inside the canvas")
            return                       # a figure the border cuts is not an inset panel
        rows = [r for r, _c in body]
        cols = [c for _r, c in body]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        margins = (r0, h - 1 - r1, c0, w - 1 - c1)
        if min(margins) < max(2, -(-min(h, w) * 10 // 100)):                    # (i)
            raise ValueError("task25 (literal): the ground margin around a dominant figure "
                             "reads as a border")
        if total[ground] >= .55 * h * w:
            return
        body_colors = {cell[p] for p in body}                                   # (j)
        for part in blind:
            if (len(part) >= 4 and {cell[p] for p in part} - body_colors
                    and not any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in part)):
                break                # a body, not a dot, sits on the ground beside it
        else:
            raise ValueError("task25 (literal): a thin ground carries one figure only")
        if abs(margins[0] - margins[1]) >= 3 or abs(margins[2] - margins[3]) >= 3:
            return
        exterior = set()
        for part in ground_parts:
            exterior |= part
        inside = {p: 0 for p in exterior if r0 <= p[0] <= r1 and c0 <= p[1] <= c1}
        for _value, pocket in _literal_parts(inside):
            depth = max(min(r - r0 + 1, r1 - r + 1, c - c0 + 1, c1 - c + 1) for r, c in pocket)
            if depth >= 2 and len(pocket) >= .08 * (r1 - r0 + 1) * (c1 - c0 + 1):
                return                   # the ground flows into the figure: not a panel either
        raise ValueError("task25 (literal): a centred convex figure over a thin ground "
                         "reads as a panel")

    def separating_field(ground, ground_parts, ground_mass):
        if any(k > .30 * len(ring) for v, k in on_ring.items() if v != ground):  # (b')
            raise ValueError("task25 (literal): a rival colour holds too much of the border ring")
        if len(ground_parts) != 1:                                              # (d')
            raise ValueError("task25 (literal): the ground on the border is not one field")
        field = ground_parts[0]
        figures = {p: 0 for p in cell if cell[p] != ground}                     # colour-blind
        for _value, part in _literal_parts(figures):
            if not any((r + dr, c + dc) in field for r, c in part
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                raise ValueError("task25 (literal): a figure does not sit on the field")
        if total[ground] < .30 * h * w:                                         # (f')
            raise ValueError("task25 (literal): the field is too thin to hold the picture")
        if total[ground] < 1.25 * max(k for v, k in total.items() if v != ground):
            raise ValueError("task25 (literal): another colour has comparable mass on the canvas")

    winners, reasons = [], []
    for ground in sorted(on_ring, key=lambda v: str(v)):
        if on_ring[ground] < .35 * len(ring):                                   # (A)
            continue
        if not (any(rows[0][c] == ground for c in range(w))                     # (C)
                and any(rows[h - 1][c] == ground for c in range(w))
                and any(rows[r][0] == ground for r in range(h))
                and any(rows[r][w - 1] == ground for r in range(h))):
            reasons.append("task25 (literal): the ground does not reach every side")
            continue
        ground_parts = [part for value, part in parts if value == ground and part & edge]
        pockets = [len(part) for value, part in parts if value == ground and not part & edge]
        ground_mass = sum(len(part) for part in ground_parts)
        if pockets and max(pockets) >= ground_mass:                             # (F)
            reasons.append("task25 (literal): an enclosed pocket outweighs the visible ground")
            continue
        for name, rule in (("ring_majority", ring_majority), ("separating_field", separating_field)):
            try:
                rule(ground, ground_parts, ground_mass)
                scene_rules(ground, ground_parts)
            except ValueError as error:
                reasons.append(str(error))
            else:
                winners.append((ground, name))
                break
    if len(winners) != 1:
        if len(winners) > 1:
            raise ValueError("task25 (literal): two colours read as the ground")
        raise ValueError("; ".join(dict.fromkeys(reasons)) if reasons else
                         "task25 (literal): no colour owns enough of the border ring")
    return winners[0][0]


def validate_example(example):
    if min(np.asarray(example["input"]).shape) < MIN_SIDE:
        raise ValueError("task25: canvases require at least eight cells on each side")
    inferred = infer_background(example["input"])
    literal = literal_background(example["input"])
    if str(inferred) != str(literal):
        raise ValueError(f"task25: the two certificate readings disagree ({inferred} vs {literal})")
    if str(inferred) != example["output"]:
        raise ValueError("task25: visible background disagrees with output")
    panel = panel_reading(example["input"])
    if panel is not None and str(panel) != example["output"]:
        raise ValueError(f"task25: the picture also reads as a panel of {panel} inside a border")


def scene_defects(grid, ground):
    """(hairs, notches) of the figure silhouette, texture dots set aside."""
    a = np.asarray(grid)
    figure = a != ground
    isolated = figure & (_neighbours(figure, D8, False) == 0)
    figure &= ~isolated
    return hairs(figure, edge=True), notches(figure, corners=True)


def generate_background_color(rng):
    h, w = SIZES[int(rng.integers(len(SIZES)))]
    plan = _plan(rng, h, w)
    colors = list(map(int, rng.permutation(10)))
    ground = colors[0]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        target = plan["n_colors"]
        n_fig = plan["n_fig"]
        # after a third of the budget the planned figure count is taken as unreachable on this
        # canvas, and the builders seat what fits instead of failing the attempt outright
        plan["relax"] = attempt > MAX_ATTEMPTS // 3
        scene = _build(rng, plan, h, w, ground, colors[1:1 + n_fig])
        if scene is None:
            continue
        scene.texture = np.zeros((h, w), dtype=bool)
        texture_kind, texture_colors = _dress(rng, scene, plan, ground, target, colors[1 + n_fig:] + colors[1:1 + n_fig])
        a = scene.a
        try:
            inferred, certificate_path = read_background(a)
        except ValueError:
            continue
        if inferred != ground:
            raise RuntimeError("task25: the certificate names a different ground than built")
        if panel_reading(a) is not None:
            continue                      # it would also read as a panel inside a drawn border
        n_colors = len(np.unique(a))
        if n_colors < 2:
            continue
        n_hairs, n_notches = scene_defects(a, ground)
        if n_hairs or n_notches:
            continue
        counts = {int(c): int(np.count_nonzero(a == c)) for c in np.unique(a)}
        ring = scene.ring
        ring_values, ring_counts = np.unique(a[ring], return_counts=True)
        ring_share = dict(zip(ring_values.tolist(), ring_counts.tolist()))
        rival_ring = max([v for c, v in ring_share.items() if c != ground] or [0]) / scene.ring_n
        figure_mask = (a != ground) & ~scene.texture
        figure_cells = int(figure_mask.sum())
        # a roomy canvas carries a scene, and the dots never take it over
        if figure_cells < _coverage_need(h, w) or int(scene.texture.sum()) >= .5 * figure_cells:
            continue
        labels, n_bodies = label_same_color(figure_mask.astype(np.int64))
        body_ids = np.unique(labels[figure_mask])
        bodies = [int((labels == i).sum()) for i in body_ids]
        touching_pairs = sum(len(f["touches"]) for f in scene.figures) // 2
        # the archetype label is a claim about the realised grid, so hold it to it
        if plan["archetype"] == "dominant_figure" and max(bodies) < DOMINANT_BODY_SHARE * a.size:
            continue
        if plan["archetype"] == "cluster" and touching_pairs == 0:
            continue
        # only the colour of the dominant body may outnumber the ground
        body_color = max((c for c in counts if c != ground), key=lambda c: counts[c], default=ground)
        if any(k > counts[ground] for c, k in counts.items() if c not in (ground, body_color)):
            continue
        ground_labels, _ = label_same_color((a == ground).astype(np.int64))
        gl = np.unique(ground_labels[a == ground])
        exterior_ids = np.unique(ground_labels[ring & (a == ground)])
        exterior = max(int((ground_labels == i).sum()) for i in exterior_ids)
        figure_sizes = [int(f["cells"].sum()) for f in scene.figures]
        multi = len(figure_sizes) > 1
        # nested = every ground cell around the figure lies in an enclosed pocket (a measure)
        enclosed_ground = (a == ground) & ~np.isin(ground_labels, exterior_ids)
        for f in scene.figures:
            around = halo(f["cells"]) & ~f["cells"] & (a == ground)
            f["nested"] = bool(around.any() and enclosed_ground[around].all())
        example = {"input": a.tolist(), "output": str(ground), "metadata": {
            "grid_h": h, "grid_w": w, "n_colors": int(n_colors),
            "background_color": ground, "answer_color": ground,
            "background_kind": texture_kind, "texture_kind": texture_kind,
            "texture_colors": [int(c) for c in texture_colors],
            "texture_coverage": float(scene.texture.sum() / a.size),
            "shape_families": [f["kind"] for f in scene.figures],
            "background_count": counts[ground],
            "background_is_modal": counts[ground] > max(v for c, v in counts.items() if c != ground),
            "ground_is_modal": counts[ground] > max(v for c, v in counts.items() if c != ground),
            "layout": plan["archetype"], "archetype": plan["archetype"],
            "background_share": counts[ground] / a.size,
            "ground_share": counts[ground] / a.size,
            "ground_ring_share": ring_share.get(ground, 0) / scene.ring_n,
            "rival_ring_share": rival_ring,
            "ground_ring_components": len(exterior_ids),
            "certificate_path": certificate_path,
            "object_count": len(scene.figures), "n_figures": len(scene.figures),
            "figure_sizes": figure_sizes,
            "figures_touching_border": sum(f["border"] for f in scene.figures),
            "edge_objects": sum(f["border"] for f in scene.figures),
            "clipped_figures": sum(f["clipped_sides"] > 0 for f in scene.figures),
            "spanning_bars": sum(f.get("spanning", False) for f in scene.figures),
            "two_color_figures": sum(f["detail"] is not None for f in scene.figures),
            "nested_figures": sum(f["nested"] for f in scene.figures),
            "touching_pairs": touching_pairs,
            "size_spread_ratio": (max(figure_sizes) / min(figure_sizes)) if multi else 1.0,
            "foreground_components": len(bodies),
            "foreground_cells": int(np.count_nonzero(a != ground)),
            "largest_foreground_body": max(bodies) if bodies else 0,
            "exterior_background_cells": exterior,
            "enclosed_background_components": int(len(gl) - len(exterior_ids)),
            "attempts": attempt}}
        validate_example(example)
        return example
    raise RuntimeError(f"task25: cannot realize the planned scene {(h, w, plan['archetype'], plan['n_colors'])}")
