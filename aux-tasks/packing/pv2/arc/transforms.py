"""The ARC transform library: parameterised grid -> grid rules (DESIGN.md section 8).

A ``Family`` is one kind of transform with an explicit, finite parameter set (``instances``); an
instance is ``(family name, params)``.  Every function returns a new int8 array or None when the
rule is undefined on that grid (a size above 30, no background, a tie where a unique object is
needed, blocks that are not uniform, ...).  ``Rule`` chains one or two instances (a composition).

Flags used elsewhere:

* ``bg``           background-dependent: the rule reads the background (strict plurality colour);
                   generation uses it only on plain-ground pictures (``grids.is_solid``).
* ``same_shape``   the result always has the input's size.
* ``sig``          what a same-size result can differ in, for the rival search (``rivals``):
                   ``fixed_map`` a fixed cell-wise colour map (recolor, swap ...), ``paint`` changed
                   cells all take the colour ``params[paint_idx]``, ``erase`` changed cells all
                   become the background, ``fill_bg`` only background cells change, ``hist`` the
                   colour histogram is kept, ``other``.
* ``shape``        ``shape(h, w, *p)`` -> the exact result size, None when it depends on the
                   content, False when the rule is undefined at that size.
* ``noop``         a cheap sufficient test that the rule leaves this grid unchanged.

``describe(*p)`` is the one canonical English phrase for an instance (menus never show two phrases
for one function, and the library test checks that no two instances behave identically on a
battery of grids unless they are declared synonyms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from pv2.arc.grids import (DT, MAX_SIDE, Info, arr, border_reachable, fits, inside_count, near_mask,
                           neighbour_count_same, regions, to_list)

COLORS = tuple(range(10))
CONNS = (4, 8)


@dataclass
class Family:
    name: str
    fn: Callable
    instances: tuple
    describe: Callable
    same_shape: bool = True
    sig: str = "other"
    paint_idx: Optional[int] = None
    bg: bool = False
    shape: Optional[Callable] = None
    noop: Optional[Callable] = None
    mono_ok: object = False          # True: single-colour outputs by design; "some": allowed, not on every pair
    geometric: bool = False          # a global geometric rule (DESIGN 13.10: 2 pairs suffice)
    new_colors: bool = False         # may output a colour the input lacks (paint colours, fixed maps)
    from_bg: bool = False            # only background cells ever change
    by_paint: dict = field(default_factory=dict)

    def __post_init__(self):
        self.instances = tuple(tuple(p) for p in self.instances)
        if self.sig in ("paint", "fixed_map"):
            self.new_colors = True
        if self.sig == "paint":
            for p in self.instances:
                self.by_paint.setdefault(p[self.paint_idx], []).append(p)

    @property
    def background_dependent(self) -> bool:
        return self.bg

    def out_shape(self, h: int, w: int, p: tuple):
        if self.same_shape:
            return (h, w)
        if self.shape is None:
            return None
        return self.shape(h, w, *p)


FAMILIES: dict[str, Family] = {}


def _fam(**kw) -> Family:
    f = Family(**kw)
    if f.name in FAMILIES:
        raise ValueError(f"duplicate family {f.name}")
    FAMILIES[f.name] = f
    return f


def cw(c: int) -> str:
    return f"color {c}"


def _conn_phrase(conn: int) -> str:
    return "sharing an edge" if conn == 4 else "sharing an edge or a corner"


def _obj_phrase(conn: int) -> str:
    return ("an object is a group of same-colored non-background cells connected through shared edges"
            if conn == 4 else
            "an object is a group of same-colored non-background cells connected through shared edges or corners")


def _cells(n: int) -> str:
    return "1 cell" if n == 1 else f"{n} cells"


# ================================================================================================
# geometry
# ================================================================================================

def _rot(I, k):
    return np.ascontiguousarray(np.rot90(I.a, -k))


_fam(name="rot", fn=_rot, instances=[(1,), (2,), (3,)], same_shape=False, sig="hist", geometric=True,
     shape=lambda h, w, k: (w, h) if k % 2 else (h, w),
     describe=lambda k: {1: "rotate the grid 90 degrees clockwise", 2: "rotate the grid 180 degrees",
                         3: "rotate the grid 90 degrees counterclockwise"}[k])


def _flip(I, axis):
    return np.ascontiguousarray(I.a[:, ::-1] if axis == "lr" else I.a[::-1, :])


_fam(name="flip", fn=_flip, instances=[("lr",), ("ud",)], sig="hist", geometric=True,
     describe=lambda axis: "mirror the grid left to right (reverse the order of the columns)" if axis == "lr"
     else "mirror the grid top to bottom (reverse the order of the rows)")


def _transpose(I, d):
    return np.ascontiguousarray(I.a.T if d == "main" else I.a[::-1, ::-1].T)


_fam(name="transpose", fn=_transpose, instances=[("main",), ("anti",)], same_shape=False, sig="hist",
     geometric=True, shape=lambda h, w, d: (w, h),
     describe=lambda d: "reflect the grid across its main diagonal (row r, column c goes to row c, column r)"
     if d == "main" else
     "reflect the grid across its anti-diagonal (the top-right to bottom-left diagonal)")

SCALES = [(k, k) for k in range(2, 11)] + [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]


def _scale_up(I, kh, kw):
    if not fits(I.h * kh, I.w * kw):
        return None
    return np.ascontiguousarray(np.repeat(np.repeat(I.a, kh, axis=0), kw, axis=1))


def _scale_desc_up(kh, kw):
    if kh == kw:
        return f"scale the grid up by {kh}: every cell becomes a {kh}x{kh} block of its color"
    return f"stretch every cell into a block {kh} {'cell' if kh == 1 else 'cells'} tall and {kw} " \
           f"{'cell' if kw == 1 else 'cells'} wide of its color"


_fam(name="scale_up", fn=_scale_up, instances=SCALES, same_shape=False, geometric=True,
     shape=lambda h, w, kh, kw: (h * kh, w * kw) if fits(h * kh, w * kw) else False,
     describe=_scale_desc_up)


def _scale_down(I, kh, kw):
    if I.h % kh or I.w % kw:
        return None
    small = I.a[::kh, ::kw]
    if not np.array_equal(np.repeat(np.repeat(small, kh, axis=0), kw, axis=1), I.a):
        return None
    return np.ascontiguousarray(small)


def _scale_desc_down(kh, kw):
    if kh == kw:
        return f"shrink the grid by {kh}: every {kh}x{kh} block of one color becomes one cell"
    return f"shrink the grid: every block {kh} {'cell' if kh == 1 else 'cells'} tall and {kw} " \
           f"{'cell' if kw == 1 else 'cells'} wide of one color becomes one cell"


_fam(name="scale_down", fn=_scale_down, instances=SCALES, same_shape=False, geometric=True,
     shape=lambda h, w, kh, kw: (h // kh, w // kw) if (h % kh == 0 and w % kw == 0) else False,
     describe=_scale_desc_down)

TILES = [(m, n) for m in range(1, 16) for n in range(1, 16) if (m, n) != (1, 1)]


def _tile(I, m, n):
    if not fits(I.h * m, I.w * n):
        return None
    return np.ascontiguousarray(np.tile(I.a, (m, n)))


def _times(k):
    return {1: "once", 2: "twice"}.get(k, f"{k} times")


_fam(name="tile", fn=_tile, instances=TILES, same_shape=False, geometric=True,
     shape=lambda h, w, m, n: (h * m, w * n) if fits(h * m, w * n) else False,
     describe=lambda m, n: f"repeat the grid {_times(m)} top to bottom and {_times(n)} left to right "
                           f"(copies placed edge to edge)")

MIRRORS = ("right", "left", "down", "up", "four")


def _mirror_tile(I, mode):
    a = I.a
    if mode == "right":
        out = np.hstack([a, a[:, ::-1]])
    elif mode == "left":
        out = np.hstack([a[:, ::-1], a])
    elif mode == "down":
        out = np.vstack([a, a[::-1, :]])
    elif mode == "up":
        out = np.vstack([a[::-1, :], a])
    else:
        top = np.hstack([a, a[:, ::-1]])
        out = np.vstack([top, top[::-1, :]])
    if not fits(*out.shape):
        return None
    return np.ascontiguousarray(out)


def _mirror_shape(h, w, mode):
    s = {"right": (h, 2 * w), "left": (h, 2 * w), "down": (2 * h, w), "up": (2 * h, w),
         "four": (2 * h, 2 * w)}[mode]
    return s if fits(*s) else False


_fam(name="mirror_tile", fn=_mirror_tile, instances=[(m,) for m in MIRRORS], same_shape=False, geometric=True,
     shape=_mirror_shape,
     describe=lambda mode: {
         "right": "append a left-right mirror image of the grid on its right",
         "left": "put a left-right mirror image of the grid on its left",
         "down": "stack a top-bottom mirror image of the grid underneath it",
         "up": "stack a top-bottom mirror image of the grid on top of it",
         "four": "make a 2x2 arrangement: the grid, its left-right mirror to the right, and the "
                 "top-bottom mirror of that pair underneath"}[mode])


def _fractal(I):
    bg = I.bg
    if bg is None or not fits(I.h * I.h, I.w * I.w):
        return None
    a = I.a
    out = np.full((I.h * I.h, I.w * I.w), bg, dtype=DT)
    for r in range(I.h):
        for c in range(I.w):
            if a[r, c] != bg:
                out[r * I.h:(r + 1) * I.h, c * I.w:(c + 1) * I.w] = a
    return out


_fam(name="fractal", fn=_fractal, instances=[()], same_shape=False, bg=True,
     shape=lambda h, w: (h * h, w * w) if fits(h * h, w * w) else False,
     describe=lambda: "replace every non-background cell by a copy of the whole grid and every "
                      "background cell by a block of background of the same size")


def _symmetrize(I, mode):
    a = I.a.copy()
    h, w = I.h, I.w
    if mode == "lr":
        a[:, w - w // 2:] = I.a[:, :w // 2][:, ::-1]
    elif mode == "rl":
        a[:, :w // 2] = I.a[:, w - w // 2:][:, ::-1]
    elif mode == "ud":
        a[h - h // 2:, :] = I.a[:h // 2, :][::-1, :]
    else:
        a[:h // 2, :] = I.a[h - h // 2:, :][::-1, :]
    return a


_fam(name="symmetrize", fn=_symmetrize, instances=[("lr",), ("rl",), ("ud",), ("du",)],
     describe=lambda m: {
         "lr": "overwrite the right half with the mirror image of the left half",
         "rl": "overwrite the left half with the mirror image of the right half",
         "ud": "overwrite the bottom half with the mirror image of the top half",
         "du": "overwrite the top half with the mirror image of the bottom half"}[m])


# ================================================================================================
# crops and extraction
# ================================================================================================

def _crop_box(a, mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return np.ascontiguousarray(a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])


def _crop_content(I):
    bg = I.bg
    if bg is None:
        return None
    return _crop_box(I.a, I.a != bg)


_fam(name="crop_content", fn=_crop_content, instances=[()], same_shape=False, bg=True,
     describe=lambda: "crop the grid to the smallest rectangle that contains every non-background cell")


def _crop_color(I, x):
    if x == "least":
        x = I.least()
        if x is None:
            return None
    if not I.hist[x]:
        return None
    return _crop_box(I.a, I.a == x)


_fam(name="crop_color", fn=_crop_color, instances=[(c,) for c in COLORS] + [("least",)], same_shape=False,
     mono_ok="some",
     describe=lambda x: "crop the grid to the smallest rectangle that contains every cell of the least "
                        "frequent color" if x == "least" else
                        f"crop the grid to the smallest rectangle that contains every cell of {cw(x)}")


def _extreme(objs, which, measure="cells", min_size=1):
    objs = [o for o in objs if o.size >= min_size]
    if not objs:
        return None
    key = (lambda o: o.size) if measure == "cells" else (lambda o: o.bbox_area)
    vals = sorted(key(o) for o in objs)
    target = vals[-1] if which == "largest" else vals[0]
    hits = [o for o in objs if key(o) == target]
    return hits[0] if len(hits) == 1 else None


def _crop_object(I, which, conn):
    objs = I.objects(conn)
    if not objs:
        return None
    o = _extreme(objs, which)
    if o is None:
        return None
    return np.ascontiguousarray(I.a[o.r0:o.r1 + 1, o.c0:o.c1 + 1])


_fam(name="crop_object", fn=_crop_object, same_shape=False, bg=True, mono_ok="some",
     instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
     describe=lambda wh, conn: f"crop the grid to the bounding box of the object with the "
                               f"{'most' if wh == 'largest' else 'fewest'} cells ({_obj_phrase(conn)})")


def _extract_row(I, r):
    if r == -1:
        r = I.h - 1
    if r >= I.h:
        return None
    return np.ascontiguousarray(I.a[r:r + 1, :])


def _extract_col(I, c):
    if c == -1:
        c = I.w - 1
    if c >= I.w:
        return None
    return np.ascontiguousarray(I.a[:, c:c + 1])


_fam(name="extract_row", fn=_extract_row, instances=[(r,) for r in range(MAX_SIDE)] + [(-1,)], same_shape=False,
     mono_ok="some",
     shape=lambda h, w, r: (1, w) if (r == -1 or r < h) else False,
     describe=lambda r: "keep only the bottom row" if r == -1 else
     ("keep only the top row" if r == 0 else f"keep only row {r} (rows counted from 0 at the top)"))
_fam(name="extract_col", fn=_extract_col, instances=[(c,) for c in range(MAX_SIDE)] + [(-1,)], same_shape=False,
     mono_ok="some",
     shape=lambda h, w, c: (h, 1) if (c == -1 or c < w) else False,
     describe=lambda c: "keep only the rightmost column" if c == -1 else
     ("keep only the leftmost column" if c == 0 else f"keep only column {c} (columns counted from 0 at the left)"))


def _half(I, which):
    h, w = I.h, I.w
    if which in ("top", "bottom"):
        if h % 2:
            return None
        return np.ascontiguousarray(I.a[:h // 2] if which == "top" else I.a[h // 2:])
    if w % 2:
        return None
    return np.ascontiguousarray(I.a[:, :w // 2] if which == "left" else I.a[:, w // 2:])


def _half_shape(h, w, which):
    if which in ("top", "bottom"):
        return (h // 2, w) if h % 2 == 0 else False
    return (h, w // 2) if w % 2 == 0 else False


_fam(name="half", fn=_half, instances=[(x,) for x in ("top", "bottom", "left", "right")], same_shape=False,
     shape=_half_shape, geometric=True,
     describe=lambda which: f"keep the {which} half of the grid")


def _compress(I, mode):
    a = I.a
    if mode in ("rows", "both"):
        rows = [ln.tobytes() for ln in a]
        a = a[[0] + [r for r in range(1, len(rows)) if rows[r] != rows[r - 1]]]
    if mode in ("cols", "both"):
        cols = [ln.tobytes() for ln in np.ascontiguousarray(a.T)]
        a = a[:, [0] + [c for c in range(1, len(cols)) if cols[c] != cols[c - 1]]]
    return np.ascontiguousarray(a)


_fam(name="compress", fn=_compress, instances=[("rows",), ("cols",), ("both",)], same_shape=False,
     describe=lambda m: {"rows": "merge every run of identical neighbouring rows into one row",
                         "cols": "merge every run of identical neighbouring columns into one column",
                         "both": "merge every run of identical neighbouring rows into one row, then every "
                                 "run of identical neighbouring columns into one column"}[m])


def _drop_bg_lines(I, mode):
    bg = I.bg
    if bg is None:
        return None
    a = I.a
    if mode in ("rows", "both"):
        a = a[(a != bg).any(axis=1)]
    if a.shape[0] and mode in ("cols", "both"):
        a = a[:, (a != bg).any(axis=0)]
    if a.shape[0] == 0 or a.shape[1] == 0:
        return None
    return np.ascontiguousarray(a)


_fam(name="drop_bg_lines", fn=_drop_bg_lines, instances=[("rows",), ("cols",), ("both",)], same_shape=False,
     bg=True,
     describe=lambda m: {"rows": "delete every row that holds only background",
                         "cols": "delete every column that holds only background",
                         "both": "delete every row and every column that holds only background"}[m])


def _halves(I, split):
    h, w = I.h, I.w
    if split == "lr":
        if w % 2:
            return None
        return I.a[:, :w // 2], I.a[:, w // 2:]
    if h % 2:
        return None
    return I.a[:h // 2], I.a[h // 2:]


_OPS = {"and": np.logical_and, "or": np.logical_or, "xor": np.logical_xor}


def _logic_halves(I, op, split, x):
    bg = I.bg
    parts = _halves(I, split)
    if bg is None or parts is None or x == bg:
        return None
    A, B = parts
    m = _OPS[op](A != bg, B != bg)
    return np.where(m, x, bg).astype(DT)


def _halves_shape(h, w, split):
    if split == "lr":
        return (h, w // 2) if w % 2 == 0 else False
    return (h // 2, w) if h % 2 == 0 else False


_fam(name="logic_halves", fn=_logic_halves, same_shape=False, bg=True, new_colors=True,
     instances=[(op, s, x) for op in ("and", "or", "xor") for s in ("lr", "ud") for x in COLORS],
     shape=lambda h, w, op, s, x: _halves_shape(h, w, s),
     describe=lambda op, s, x: (
         f"lay the {'left and right' if s == 'lr' else 'top and bottom'} halves on top of each other and "
         f"paint {cw(x)} where " + {"and": "both halves", "or": "at least one half",
                                    "xor": "exactly one half"}[op] +
         " has a non-background cell; every other cell is background"))


def _overlay_halves(I, split, top):
    bg = I.bg
    parts = _halves(I, split)
    if bg is None or parts is None:
        return None
    A, B = parts
    lower, upper = (A, B) if top == "second" else (B, A)
    return np.where(upper != bg, upper, lower).astype(DT)


def _overlay_desc(s, t):
    first, second = ("left", "right") if s == "lr" else ("top", "bottom")
    upper, lower = (first, second) if t == "first" else (second, first)
    return (f"lay the {upper} half over the {lower} half: each cell shows the {upper} half's cell unless that "
            f"cell is background, in which case it shows the {lower} half's cell")


_fam(name="overlay_halves", fn=_overlay_halves, same_shape=False, bg=True,
     instances=[(s, t) for s in ("lr", "ud") for t in ("first", "second")],
     shape=lambda h, w, s, t: _halves_shape(h, w, s),
     describe=_overlay_desc)


def _count_bar(I, x, orient):
    n = int(I.hist[x])
    if not 1 <= n <= MAX_SIDE:
        return None
    return np.full((1, n) if orient == "h" else (n, 1), x, dtype=DT)


_fam(name="count_bar", fn=_count_bar, same_shape=False, mono_ok=True,
     instances=[(x, o) for x in COLORS for o in ("h", "v")],
     describe=lambda x, o: f"give a single {'row' if o == 'h' else 'column'} of {cw(x)} with one cell for "
                           f"every cell of {cw(x)} in the grid")


def _mode_cell(I, which):
    c = I.bg if which == "most" else I.least()
    if c is None:
        return None
    return np.full((1, 1), c, dtype=DT)


_fam(name="mode_cell", fn=_mode_cell, instances=[("most",), ("least",)], same_shape=False, mono_ok=True,
     shape=lambda h, w, which: (1, 1),
     describe=lambda which: f"give a 1x1 grid of the {'most' if which == 'most' else 'least'} frequent color")


def _period(lines) -> int:
    rows = [ln.tobytes() for ln in lines]
    n = len(rows)
    for p in range(1, n + 1):
        if all(rows[i] == rows[i - p] for i in range(p, n)):
            return p
    return n


def _smallest_tile(I):
    ph = _period(I.a)
    pw = _period(I.a.T)
    return np.ascontiguousarray(I.a[:ph, :pw])


_fam(name="smallest_tile", fn=_smallest_tile, instances=[()], same_shape=False,
     describe=lambda: "give the smallest top-left tile whose repeated copies (cut off at the bottom and "
                      "right edges) rebuild the whole grid")


# ================================================================================================
# colours
# ================================================================================================

def _recolor(I, a, b):
    return np.where(I.a == a, b, I.a).astype(DT)


_fam(name="recolor", fn=_recolor, sig="fixed_map",
     instances=[(a, b) for a in COLORS for b in COLORS if a != b],
     noop=lambda I, a, b: not I.hist[a],
     describe=lambda a, b: f"replace every cell of {cw(a)} with {cw(b)}")


def _swap(I, a, b):
    out = I.a.copy()
    out[I.a == a] = b
    out[I.a == b] = a
    return out


_fam(name="swap", fn=_swap, sig="fixed_map",
     instances=[(a, b) for a in COLORS for b in COLORS if a < b],
     noop=lambda I, a, b: not I.hist[a] and not I.hist[b],
     describe=lambda a, b: f"swap {cw(a)} and {cw(b)} (every cell of either color takes the other)")


def _keep_fill(I, x, z):
    return np.where(I.a == x, x, z).astype(DT)


_fam(name="keep_fill", fn=_keep_fill, sig="fixed_map",
     instances=[(x, z) for x in COLORS for z in COLORS if x != z],
     noop=lambda I, x, z: int(I.hist[x]) + int(I.hist[z]) == I.h * I.w,
     describe=lambda x, z: f"keep the cells of {cw(x)} and paint every other cell {cw(z)}")


def _fill_const(I, x):
    return np.full(I.a.shape, x, dtype=DT)


_fam(name="fill_const", fn=_fill_const, sig="fixed_map", instances=[(x,) for x in COLORS], mono_ok=True,
     noop=lambda I, x: int(I.hist[x]) == I.h * I.w,
     describe=lambda x: f"paint every cell {cw(x)}")


def _keep_only(I, x):
    bg = I.bg
    if bg is None:
        return None
    return np.where(I.a == x, x, bg).astype(DT)


_fam(name="keep_only", fn=_keep_only, sig="erase", bg=True, instances=[(x,) for x in COLORS],
     describe=lambda x: f"keep only the cells of {cw(x)}; every other cell becomes background")


def _delete(I, x):
    bg = I.bg
    if bg is None:
        return None
    return np.where(I.a == x, bg, I.a).astype(DT)


_fam(name="delete", fn=_delete, sig="erase", bg=True, instances=[(x,) for x in COLORS],
     noop=lambda I, x: I.bg is not None and (not I.hist[x] or x == I.bg),
     describe=lambda x: f"turn every cell of {cw(x)} into background")


def _invert2(I):
    cs = I.colors
    if len(cs) != 2:
        return None
    a, b = cs
    return _swap(I, a, b)


_fam(name="invert2", fn=_invert2, instances=[()],
     describe=lambda: "the grid has two colors: swap them")


def _mono(I, x):
    bg = I.bg
    if bg is None:
        return None
    return np.where(I.a != bg, x, bg).astype(DT)


_fam(name="mono", fn=_mono, sig="paint", paint_idx=0, bg=True, instances=[(x,) for x in COLORS],
     describe=lambda x: f"paint every non-background cell {cw(x)}")


def _bg_to(I, x):
    bg = I.bg
    if bg is None:
        return None
    return np.where(I.a == bg, x, I.a).astype(DT)


_fam(name="bg_to", fn=_bg_to, sig="paint", paint_idx=0, bg=True, from_bg=True, instances=[(x,) for x in COLORS],
     describe=lambda x: f"paint every background cell {cw(x)}")


def _recolor_least(I, x):
    c = I.least()
    if c is None:
        return None
    return np.where(I.a == c, x, I.a).astype(DT)


_fam(name="recolor_least", fn=_recolor_least, sig="paint", paint_idx=0, instances=[(x,) for x in COLORS],
     describe=lambda x: f"repaint with {cw(x)} every cell of the least frequent color")


# ================================================================================================
# motion
# ================================================================================================

SHIFTS = [(dr, dc) for dr in range(-3, 4) for dc in range(-3, 4) if (dr, dc) != (0, 0)]


def _steps(dr, dc):
    parts = []
    if dr:
        parts.append(f"{abs(dr)} {'row' if abs(dr) == 1 else 'rows'} {'down' if dr > 0 else 'up'}")
    if dc:
        parts.append(f"{abs(dc)} {'column' if abs(dc) == 1 else 'columns'} {'right' if dc > 0 else 'left'}")
    return " and ".join(parts)


def _roll(I, dr, dc):
    if abs(dr) >= I.h and dr or abs(dc) >= I.w and dc:
        return None
    return np.ascontiguousarray(np.roll(I.a, (dr, dc), axis=(0, 1)))


_fam(name="roll", fn=_roll, sig="hist", instances=SHIFTS,
     describe=lambda dr, dc: f"shift the whole grid {_steps(dr, dc)}, wrapping cells that leave one edge "
                             f"around to the opposite edge")


def _shift(I, dr, dc):
    bg = I.bg
    h, w = I.h, I.w
    if bg is None or abs(dr) >= h or abs(dc) >= w:
        return None
    out = np.full((h, w), bg, dtype=DT)
    out[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)] = \
        I.a[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)]
    return out


_fam(name="shift", fn=_shift, bg=True, instances=SHIFTS,
     describe=lambda dr, dc: f"move everything {_steps(dr, dc)}; what leaves the grid is lost and the cells "
                             f"left behind become background")

DIRS = ("down", "up", "left", "right")


def _gravity(I, d):
    bg = I.bg
    if bg is None:
        return None
    a = I.a
    if d in ("left", "right"):
        a = a.T
    out = np.full(a.shape, bg, dtype=DT)
    for c in range(a.shape[1]):
        col = a[:, c]
        vals = col[col != bg]
        if len(vals) == 0:
            continue
        if d in ("down", "right"):
            out[a.shape[0] - len(vals):, c] = vals
        else:
            out[:len(vals), c] = vals
    if d in ("left", "right"):
        out = out.T
    return np.ascontiguousarray(out)


_fam(name="gravity", fn=_gravity, sig="hist", bg=True, instances=[(d,) for d in DIRS],
     describe=lambda d: {"down": "let every non-background cell fall straight down within its column, keeping "
                                 "their order, until they rest on the bottom edge or on each other",
                         "up": "let every non-background cell rise straight up within its column, keeping their "
                               "order, until they rest against the top edge or each other",
                         "left": "slide every non-background cell straight left within its row, keeping their "
                                 "order, until they rest against the left edge or each other",
                         "right": "slide every non-background cell straight right within its row, keeping their "
                                  "order, until they rest against the right edge or each other"}[d])


def _sort_cells(I, axis, order):
    a = np.sort(I.a, axis=1 if axis == "rows" else 0)
    if order == "desc":
        a = a[:, ::-1] if axis == "rows" else a[::-1, :]
    return np.ascontiguousarray(a).astype(DT)


_fam(name="sort_cells", fn=_sort_cells, sig="hist",
     instances=[(ax, o) for ax in ("rows", "cols") for o in ("asc", "desc")],
     describe=lambda ax, o: (f"sort the cells of every row by color number, {'smallest' if o == 'asc' else 'largest'} "
                             f"on the left" if ax == "rows" else
                             f"sort the cells of every column by color number, {'smallest' if o == 'asc' else 'largest'} "
                             f"at the top"))


def _sort_lines(I, axis, order):
    bg = I.bg
    if bg is None:
        return None
    a = I.a if axis == "rows" else I.a.T
    counts = (a != bg).sum(axis=1)
    idx = sorted(range(a.shape[0]), key=lambda i: (counts[i] if order == "asc" else -counts[i], i))
    out = a[idx]
    return np.ascontiguousarray(out if axis == "rows" else out.T)


_fam(name="sort_lines", fn=_sort_lines, sig="hist", bg=True,
     instances=[(ax, o) for ax in ("rows", "cols") for o in ("asc", "desc")],
     describe=lambda ax, o: (f"reorder the {'rows' if ax == 'rows' else 'columns'} by how many non-background cells "
                             f"they hold, {'fewest' if o == 'asc' else 'most'} "
                             f"{'at the top' if ax == 'rows' else 'on the left'}, keeping the original order "
                             f"among ties"))


# ================================================================================================
# painting around objects (background-dependent)
# ================================================================================================

def _outline(I, x, conn, t):
    bg = I.bg
    if bg is None:
        return None
    band = I.memo(("band", conn, t), lambda: near_mask(I.a != bg, conn, t) & (I.a == bg))
    return np.where(band, x, I.a).astype(DT)


_fam(name="outline", fn=_outline, sig="paint", paint_idx=0, bg=True, from_bg=True,
     instances=[(x, c, t) for x in COLORS for c in CONNS for t in (1, 2, 3)],
     describe=lambda x, conn, t: (f"paint {cw(x)} on every background cell within {t} "
                                  f"{'step' if t == 1 else 'steps'} of a non-background cell, a step "
                                  f"moving to a cell {_conn_phrase(conn)}"))


def _holes(I):
    bg = I.bg
    if bg is None:
        return None
    open_ = I.a == bg
    return open_ & ~border_reachable(open_)


def _fill_holes(I, x):
    if I.bg is None:
        return None
    hole = I.memo("holes", lambda: _holes(I))
    return np.where(hole, x, I.a).astype(DT)


_fam(name="fill_holes", fn=_fill_holes, sig="paint", paint_idx=0, bg=True, from_bg=True, instances=[(x,) for x in COLORS],
     describe=lambda x: f"paint {cw(x)} on every background cell that cannot reach the edge of the grid "
                        f"through background cells moving up, down, left or right")


def _fill_holes_own(I):
    bg = I.bg
    if bg is None:
        return None
    hole = I.memo("holes", lambda: _holes(I))
    out = I.a.copy()
    h, w = I.h, I.w
    for cells in regions(hole):
        ring = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not hole[rr, cc]:
                    ring.add(int(I.a[rr, cc]))
        if len(ring) != 1:
            return None
        v = ring.pop()
        for r, c in cells:
            out[r, c] = v
    return out


_fam(name="fill_holes_own", fn=_fill_holes_own, sig="fill_bg", bg=True, instances=[()],
     describe=lambda: "fill every enclosed pocket of background (cells that cannot reach the edge through "
                      "background moving up, down, left or right) with the color of the cells around it")


def _interior(I, conn):
    bg = I.bg
    if bg is None:
        return None
    return I.memo(("interior", conn),
                  lambda: (I.a != bg) & (neighbour_count_same(I.a, conn) == (4 if conn == 4 else 8)))


def _hollow(I, conn):
    m = _interior(I, conn)
    if m is None:
        return None
    return np.where(m, I.bg, I.a).astype(DT)


_fam(name="hollow", fn=_hollow, sig="erase", bg=True, instances=[(c,) for c in CONNS],
     describe=lambda conn: f"turn into background every non-background cell whose {conn} neighbours "
                           f"({_conn_phrase(conn)}) are all inside the grid and of its own color")


def _fill_interior(I, x, conn):
    m = _interior(I, conn)
    if m is None:
        return None
    return np.where(m, x, I.a).astype(DT)


_fam(name="fill_interior", fn=_fill_interior, sig="paint", paint_idx=0, bg=True,
     instances=[(x, c) for x in COLORS for c in CONNS],
     describe=lambda x, conn: f"paint {cw(x)} on every non-background cell whose {conn} neighbours "
                              f"({_conn_phrase(conn)}) are all inside the grid and of its own color")


def _lines(I, x, mode):
    m = I.a == x
    rows = m.any(axis=1)
    cols = m.any(axis=0)
    out = I.a.copy()
    if mode in ("row", "cross"):
        out[rows, :] = x
    if mode in ("col", "cross"):
        out[:, cols] = x
    return out


_fam(name="lines", fn=_lines, sig="paint", paint_idx=0,
     instances=[(x, m) for x in COLORS for m in ("row", "col", "cross")],
     noop=lambda I, x, m: not I.hist[x],
     describe=lambda x, m: {"row": f"paint the whole row of every cell of {cw(x)} with {cw(x)}",
                            "col": f"paint the whole column of every cell of {cw(x)} with {cw(x)}",
                            "cross": f"paint the whole row and the whole column of every cell of {cw(x)} "
                                     f"with {cw(x)}"}[m])

_DIRV = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_EDGE = {"up": "top", "down": "bottom", "left": "left", "right": "right"}


def _rays(I, x, d):
    bg = I.bg
    if bg is None:
        return None
    a = I.a
    starts = np.argwhere(a == x) if x != bg else []
    out = a.copy()
    dr, dc = _DIRV[d]
    for r, c in starts:
        rr, cc = r + dr, c + dc
        while 0 <= rr < I.h and 0 <= cc < I.w and a[rr, cc] == bg:
            out[rr, cc] = x
            rr += dr
            cc += dc
    return out


_fam(name="rays", fn=_rays, sig="paint", paint_idx=0, bg=True, from_bg=True,
     instances=[(x, d) for x in COLORS for d in DIRS],
     noop=lambda I, x, d: I.bg is not None and (not I.hist[x] or x == I.bg),
     describe=lambda x, d: f"from every cell of {cw(x)}, paint {cw(x)} on the background cells in a straight "
                           f"line toward the {_EDGE[d]} edge, stopping at the first non-background cell or at "
                           f"the edge")


def _diagonals(I, x):
    m = I.a == x
    out = I.a.copy()
    if not m.any():
        return out
    rr, cc = np.indices(I.a.shape)
    d1, d2 = rr - cc, rr + cc
    out[np.isin(d1, d1[m]) | np.isin(d2, d2[m])] = x
    return out


_fam(name="diagonals", fn=_diagonals, sig="paint", paint_idx=0, instances=[(x,) for x in COLORS],
     noop=lambda I, x: not I.hist[x],
     describe=lambda x: f"paint both full diagonal lines through every cell of {cw(x)} with {cw(x)}")


def _connect(I, x, mode):
    bg = I.bg
    if bg is None:
        return None
    a = I.a
    out = a.copy()
    if x == bg:
        return out
    lines = []
    if mode in ("row", "both"):
        lines += [(a[r], out[r]) for r in range(I.h)]
    if mode in ("col", "both"):
        lines += [(a[:, c], out[:, c]) for c in range(I.w)]
    for src, dst in lines:
        idx = np.nonzero(src == x)[0]
        for i, j in zip(idx[:-1], idx[1:]):
            if j - i > 1 and bool(np.all(src[i + 1:j] == bg)):
                dst[i + 1:j] = x
    return out


_fam(name="connect", fn=_connect, sig="paint", paint_idx=0, bg=True, from_bg=True,
     instances=[(x, m) for x in COLORS for m in ("row", "col", "both")],
     noop=lambda I, x, m: I.bg is not None and (I.hist[x] < 2 or x == I.bg),
     describe=lambda x, m: {"row": f"join every two cells of {cw(x)} that face each other in a row across "
                                   f"background only, painting the cells between them {cw(x)}",
                            "col": f"join every two cells of {cw(x)} that face each other in a column across "
                                   f"background only, painting the cells between them {cw(x)}",
                            "both": f"join every two cells of {cw(x)} that face each other in a row or a column "
                                    f"across background only, painting the cells between them {cw(x)}"}[m])


def _border(I, x):
    out = I.a.copy()
    out[0, :] = x
    out[-1, :] = x
    out[:, 0] = x
    out[:, -1] = x
    return out


_fam(name="border", fn=_border, sig="paint", paint_idx=0, instances=[(x,) for x in COLORS],
     describe=lambda x: f"paint the outermost ring of cells (the grid's edge rows and columns) {cw(x)}")


def _mirror_overlay(I, axis):
    bg = I.bg
    if bg is None:
        return None
    f = I.a[:, ::-1] if axis == "lr" else I.a[::-1, :]
    return np.where((I.a == bg) & (f != bg), f, I.a).astype(DT)


_fam(name="mirror_overlay", fn=_mirror_overlay, sig="fill_bg", bg=True, instances=[("lr",), ("ud",)],
     describe=lambda ax: f"add the {'left-right' if ax == 'lr' else 'top-bottom'} mirror image of the grid onto "
                         f"its own background cells (existing non-background cells stay)")


# ================================================================================================
# objects (background-dependent)
# ================================================================================================

def _paint_cells(I, cells, v):
    out = I.a.copy()
    flat = out.reshape(-1)
    flat[list(cells)] = v
    return out


def _remove_isolated(I, conn):
    bg = I.bg
    if bg is None:
        return None
    same = neighbour_count_same(I.a, conn)
    m = (I.a != bg) & (same == 0)
    return np.where(m, bg, I.a).astype(DT)


_fam(name="remove_isolated", fn=_remove_isolated, sig="erase", bg=True, instances=[(c,) for c in CONNS],
     describe=lambda conn: f"turn into background every non-background cell that has no neighbour of its own "
                           f"color {_conn_phrase(conn)} with it")

_CMP = {"eq": lambda s, n: s == n, "lt": lambda s, n: s < n, "gt": lambda s, n: s > n}
_CMP_WORD = {"eq": "exactly", "lt": "fewer than", "gt": "more than"}
SIZE_RANGE = range(1, 21)


def _remove_size(I, cmp, n, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    hit = [c for o in objs if _CMP[cmp](o.size, n) for c in o.cells]
    return _paint_cells(I, hit, I.bg)


def _size_table(I, conn):
    """size -> set of object colours, cached per grid."""
    def build():
        objs = I.objects(conn)
        if objs is None:
            return None
        t = {}
        for o in objs:
            t.setdefault(o.size, set()).add(o.color)
        return t
    return I.memo(("sizes", conn), build)


def _size_noop(I, cmp, n, conn):
    t = _size_table(I, conn)
    return t is not None and not any(_CMP[cmp](sz, n) for sz in t)


_fam(name="remove_size", fn=_remove_size, sig="erase", bg=True,
     # "exactly 1 cell" and "fewer than 2 cells" are remove_isolated: left out, so no two instances
     # are the same function
     instances=[(cmp, n, c) for cmp in ("eq", "lt", "gt") for n in SIZE_RANGE for c in CONNS
                if not (cmp == "lt" and n <= 2) and not (cmp == "eq" and n == 1)],
     noop=_size_noop,
     describe=lambda cmp, n, conn: f"turn into background every object with {_CMP_WORD[cmp]} {_cells(n)} "
                                   f"({_obj_phrase(conn)})")


def _keep_extreme(I, which, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    if len(objs) < 2:
        return I.a.copy()
    o = _extreme(objs, which)
    if o is None:
        return None
    keep = set(o.cells)
    rest = [c for ob in objs if ob is not o for c in ob.cells if c not in keep]
    return _paint_cells(I, rest, I.bg)


_fam(name="keep_extreme", fn=_keep_extreme, sig="erase", bg=True,
     instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
     describe=lambda wh, conn: f"keep only the object with the {'most' if wh == 'largest' else 'fewest'} cells "
                               f"and turn every other object into background ({_obj_phrase(conn)})")


def _recolor_size(I, n, x, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    hit = [c for o in objs if o.size == n and o.color != x for c in o.cells]
    return _paint_cells(I, hit, x)


_fam(name="recolor_size", fn=_recolor_size, sig="paint", paint_idx=1, bg=True,
     instances=[(n, x, c) for n in SIZE_RANGE for x in COLORS for c in CONNS],
     noop=lambda I, n, x, conn: _size_table(I, conn) is not None and not (
         _size_table(I, conn).get(n, set()) - {x}),
     describe=lambda n, x, conn: f"repaint with {cw(x)} every object with exactly {_cells(n)} ({_obj_phrase(conn)})")


def _recolor_extreme(I, which, measure, x, conn):
    objs = I.objects(conn)
    if not objs:
        return None
    o = _extreme(objs, which, measure, min_size=2)
    if o is None:
        return None
    return _paint_cells(I, o.cells, x)


_fam(name="recolor_extreme", fn=_recolor_extreme, sig="paint", paint_idx=2, bg=True,
     instances=[(wh, me, x, c) for wh in ("largest", "smallest") for me in ("cells", "bbox") for x in COLORS
                for c in CONNS],
     describe=lambda wh, me, x, conn: (
         f"repaint with {cw(x)} the object of at least 2 cells with the "
         + ({"largest": "most cells", "smallest": "fewest cells"}[wh] if me == "cells" else
            {"largest": "largest bounding box", "smallest": "smallest bounding box"}[wh])
         + f" ({_obj_phrase(conn)})"))


def _solidify(I, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    bg = I.bg
    out = I.a.copy()
    owner = {}
    for o in objs:
        for r in range(o.r0, o.r1 + 1):
            for c in range(o.c0, o.c1 + 1):
                if I.a[r, c] == bg:
                    prev = owner.get((r, c))
                    if prev is not None and prev != o.color:
                        return None
                    owner[(r, c)] = o.color
                    out[r, c] = o.color
    return out


_fam(name="solidify", fn=_solidify, sig="fill_bg", bg=True, instances=[(c,) for c in CONNS],
     describe=lambda conn: f"fill the background cells inside every object's bounding box with that object's "
                           f"color ({_obj_phrase(conn)})")


def _hollow_rects(I, t):
    objs = I.objects(4)
    if objs is None:
        return None
    out = I.a.copy()
    for o in objs:
        hh, ww = o.bbox_dims
        if o.size != hh * ww or hh < 2 or ww < 2:
            continue
        if hh > 2 * t and ww > 2 * t:
            out[o.r0 + t:o.r1 + 1 - t, o.c0 + t:o.c1 + 1 - t] = I.bg
    return out


_fam(name="hollow_rects", fn=_hollow_rects, sig="erase", bg=True, instances=[(t,) for t in (1, 2, 3)],
     describe=lambda t: f"hollow out every solid one-color rectangle, keeping a border {t} "
                        f"{'cell' if t == 1 else 'cells'} thick and turning its inside into background")


# ================================================================================================
# alternative readings (rivals only; generation never draws them) -- found by an independent
# verifier (2026-09-29): rays that pass obstacles, size comparisons and size extremes for
# recolouring, colour-blind objects for crops, lines chosen by their content, objects chosen by
# whether they touch the grid's edge, tiling by the grid's own size or scaling / tiling by its number
# of colours, keeping one colour-blind group,
# interiors where the grid's edge counts as enclosing (general wrap shifts: see pv2.arc.rivals)
# ================================================================================================

def _behind(m: np.ndarray, d: str) -> np.ndarray:
    """Cells that have a True cell before them when walking in direction d (the cell itself not
    counted): a ray cast in direction d from every True cell reaches exactly these."""
    if d in ("down", "right"):
        acc = np.maximum.accumulate(m, axis=0 if d == "down" else 1)
    else:
        acc = np.flip(np.maximum.accumulate(np.flip(m, axis=0 if d == "up" else 1), axis=0 if d == "up" else 1),
                      axis=0 if d == "up" else 1)
    out = np.zeros_like(m)
    if d == "down":
        out[1:, :] = acc[:-1, :]
    elif d == "up":
        out[:-1, :] = acc[1:, :]
    elif d == "right":
        out[:, 1:] = acc[:, :-1]
    else:
        out[:, :-1] = acc[:, 1:]
    return out


def _rays_skip(I, x, d):
    bg = I.bg
    if bg is None:
        return None
    if x == bg:
        return I.a.copy()
    reach = _behind(I.a == x, d) & (I.a == bg)
    return np.where(reach, x, I.a).astype(DT)


_fam(name="rays_skip", fn=_rays_skip, sig="paint", paint_idx=0, bg=True, from_bg=True,
     instances=[(x, d) for x in COLORS for d in DIRS],
     noop=lambda I, x, d: I.bg is not None and (not I.hist[x] or x == I.bg),
     describe=lambda x, d: f"from every cell of {cw(x)}, paint {cw(x)} on every background cell in a straight "
                           f"line up to the {_EDGE[d]} edge, passing other cells without changing them")


def _rays_over(I, x, d):
    return np.where(_behind(I.a == x, d), x, I.a).astype(DT)


_fam(name="rays_over", fn=_rays_over, sig="paint", paint_idx=0,
     instances=[(x, d) for x in COLORS for d in DIRS],
     noop=lambda I, x, d: not I.hist[x],
     describe=lambda x, d: f"from every cell of {cw(x)}, paint {cw(x)} on every cell in a straight line up to "
                           f"the {_EDGE[d]} edge, whatever is there")


def _recolor_size_cmp(I, cmp, n, x, conn):
    t = _size_table(I, conn)
    if t is None:
        return None
    # many thresholds pick the same sizes: compute each distinct result once per grid
    sizes = tuple(sorted(sz for sz in t if _CMP[cmp](sz, n)))

    def build():
        hit = [c for o in I.objects(conn) if o.size in sizes and o.color != x for c in o.cells]
        return _paint_cells(I, hit, x)
    return I.memo(("recolor_sizes", conn, x, sizes), build).copy()


def _recolor_cmp_noop(I, cmp, n, x, conn):
    t = _size_table(I, conn)
    return t is not None and not any(_CMP[cmp](sz, n) and (cols - {x}) for sz, cols in t.items())


# "fewer than 2 cells" is recolor_size(1, ...): left out, so no two instances are the same function
_fam(name="recolor_size_cmp", fn=_recolor_size_cmp, sig="paint", paint_idx=2, bg=True,
     instances=[(cmp, n, x, c) for cmp in ("lt", "gt") for n in SIZE_RANGE for x in COLORS for c in CONNS
                if not (cmp == "lt" and n <= 2)],
     noop=_recolor_cmp_noop,
     describe=lambda cmp, n, x, conn: f"repaint with {cw(x)} every object with {_CMP_WORD[cmp]} {_cells(n)} "
                                      f"({_obj_phrase(conn)})")


def _all_extreme(I, which, conn):
    objs = I.objects(conn)
    if not objs:
        return None
    sizes = [o.size for o in objs]
    t = max(sizes) if which == "largest" else min(sizes)
    return [o for o in objs if o.size == t]


def _recolor_all_extreme(I, which, x, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    hit = _all_extreme(I, which, conn) or []
    return _paint_cells(I, [c for o in hit if o.color != x for c in o.cells], x)


_fam(name="recolor_all_extreme", fn=_recolor_all_extreme, sig="paint", paint_idx=1, bg=True,
     instances=[(wh, x, c) for wh in ("largest", "smallest") for x in COLORS for c in CONNS],
     describe=lambda wh, x, conn: f"repaint with {cw(x)} every object of the {wh} size, ties included "
                                  f"({_obj_phrase(conn)})")


def _remove_all_extreme(I, which, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    hit = _all_extreme(I, which, conn) or []
    return _paint_cells(I, [c for o in hit for c in o.cells], I.bg)


_fam(name="remove_all_extreme", fn=_remove_all_extreme, sig="erase", bg=True,
     instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
     describe=lambda wh, conn: f"turn into background every object of the {wh} size, ties included "
                               f"({_obj_phrase(conn)})")


def _crop_blob(I, which, conn):
    blobs = I.blobs(conn)
    if not blobs:
        return None
    o = _extreme(blobs, which)
    if o is None:
        return None
    return np.ascontiguousarray(I.a[o.r0:o.r1 + 1, o.c0:o.c1 + 1])


_fam(name="crop_blob", fn=_crop_blob, same_shape=False, bg=True, mono_ok="some",
     instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
     describe=lambda wh, conn: f"crop the grid to the bounding box of the group of connected non-background "
                               f"cells, of any colors, with the {'most' if wh == 'largest' else 'fewest'} cells "
                               f"(cells connected through shared edges{'' if conn == 4 else ' or corners'})")


def _line_counts(I, axis):
    a = I.a if axis == "row" else I.a.T
    return (a != I.bg).sum(axis=1)


def _extract_fg(I, axis, which):
    if I.bg is None:
        return None
    cnt = _line_counts(I, axis)
    nz = np.nonzero(cnt)[0]
    if len(nz) == 0:
        return None
    if which == "first":
        k = int(nz[0])
    elif which == "last":
        k = int(nz[-1])
    else:
        top = np.nonzero(cnt == cnt.max())[0]
        if len(top) != 1:
            return None
        k = int(top[0])
    return np.ascontiguousarray(I.a[k:k + 1, :] if axis == "row" else I.a[:, k:k + 1])


_fam(name="extract_fg", fn=_extract_fg, same_shape=False, bg=True, mono_ok="some",
     instances=[(ax, wh) for ax in ("row", "col") for wh in ("first", "last", "most")],
     shape=lambda h, w, ax, wh: (1, w) if ax == "row" else (h, 1),
     describe=lambda ax, wh: {
         "first": f"keep only the first {'row from the top' if ax == 'row' else 'column from the left'} that "
                  f"holds a non-background cell",
         "last": f"keep only the last {'row from the top' if ax == 'row' else 'column from the left'} that "
                 f"holds a non-background cell",
         "most": f"keep only the {ax if ax == 'row' else 'column'} with the most non-background cells"}[wh])


def _extract_middle(I, axis):
    n = I.h if axis == "row" else I.w
    if n % 2 == 0:
        return None
    k = n // 2
    return np.ascontiguousarray(I.a[k:k + 1, :] if axis == "row" else I.a[:, k:k + 1])


_fam(name="extract_middle", fn=_extract_middle, same_shape=False, mono_ok="some",
     instances=[("row",), ("col",)],
     shape=lambda h, w, ax: ((1, w) if h % 2 else False) if ax == "row" else ((h, 1) if w % 2 else False),
     describe=lambda ax: f"keep only the middle {'row' if ax == 'row' else 'column'}")

def _touches_edge(I, o) -> bool:
    return o.r0 == 0 or o.c0 == 0 or o.r1 == I.h - 1 or o.c1 == I.w - 1


def _recolor_edge_objects(I, x, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    return _paint_cells(I, [c for o in objs if o.color != x and _touches_edge(I, o) for c in o.cells], x)


_fam(name="recolor_edge_objects", fn=_recolor_edge_objects, sig="paint", paint_idx=0, bg=True,
     instances=[(x, c) for x in COLORS for c in CONNS],
     describe=lambda x, conn: f"repaint with {cw(x)} every object that touches the edge of the grid "
                              f"({_obj_phrase(conn)})")


def _remove_edge_objects(I, which, conn):
    objs = I.objects(conn)
    if objs is None:
        return None
    hit = [c for o in objs if _touches_edge(I, o) == (which == "touching") for c in o.cells]
    return _paint_cells(I, hit, I.bg)


_fam(name="remove_edge_objects", fn=_remove_edge_objects, sig="erase", bg=True,
     instances=[(wh, c) for wh in ("touching", "inner") for c in CONNS],
     describe=lambda wh, conn: ("turn into background every object that touches the edge of the grid"
                                if wh == "touching" else
                                "turn into background every object that does not touch the edge of the grid")
     + f" ({_obj_phrase(conn)})")

def _tile_self(I, which):
    m = I.h if which in ("h", "hw") else I.w
    n = I.w if which in ("w", "hw") else I.h
    if not fits(I.h * m, I.w * n):
        return None
    return np.ascontiguousarray(np.tile(I.a, (m, n)))


def _tile_self_shape(h, w, which):
    m = h if which in ("h", "hw") else w
    n = w if which in ("w", "hw") else h
    return (h * m, w * n) if fits(h * m, w * n) else False


_fam(name="tile_self", fn=_tile_self, instances=[("h",), ("w",), ("hw",)], same_shape=False, shape=_tile_self_shape,
     describe=lambda which: {
         "h": "repeat the grid as many times top to bottom and left to right as it has rows",
         "w": "repeat the grid as many times top to bottom and left to right as it has columns",
         "hw": "repeat the grid as many times top to bottom as it has rows and as many times left to right as "
               "it has columns"}[which])


def _keep_blob(I, which, conn):
    blobs = I.blobs(conn)
    if blobs is None:
        return None
    if len(blobs) < 2:
        return I.a.copy()
    o = _extreme(blobs, which)
    if o is None:
        return None
    keep = set(o.cells)
    return _paint_cells(I, [c for b in blobs if b is not o for c in b.cells if c not in keep], I.bg)


_fam(name="keep_blob", fn=_keep_blob, sig="erase", bg=True,
     instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
     describe=lambda wh, conn: f"keep only the group of connected non-background cells, of any colors, with the "
                               f"{'most' if wh == 'largest' else 'fewest'} cells, and turn every other cell into "
                               f"background (cells connected through shared edges{'' if conn == 4 else ' or corners'})")


def _interior_edge(I, conn):
    """Non-background cells whose neighbours inside the grid all share their colour (the area
    beyond the grid counts as enclosing)."""
    bg = I.bg
    if bg is None:
        return None
    return I.memo(("interior_edge", conn),
                  lambda: (I.a != bg) & (neighbour_count_same(I.a, conn) == inside_count(I.h, I.w, conn)))


def _hollow_edge(I, conn):
    m = _interior_edge(I, conn)
    if m is None:
        return None
    return np.where(m, I.bg, I.a).astype(DT)


_fam(name="hollow_edge", fn=_hollow_edge, sig="erase", bg=True, instances=[(c,) for c in CONNS],
     describe=lambda conn: f"turn into background every non-background cell whose neighbours inside the grid "
                           f"({_conn_phrase(conn)}) all have its own color")


def _fill_interior_edge(I, x, conn):
    m = _interior_edge(I, conn)
    if m is None:
        return None
    return np.where(m, x, I.a).astype(DT)


_fam(name="fill_interior_edge", fn=_fill_interior_edge, sig="paint", paint_idx=0, bg=True,
     instances=[(x, c) for x in COLORS for c in CONNS],
     describe=lambda x, conn: f"paint {cw(x)} on every non-background cell whose neighbours inside the grid "
                              f"({_conn_phrase(conn)}) all have its own color")

def _n_colors(I, which):
    if which == "all":
        return len(I.colors)
    return None if I.bg is None else len(I.colors) - 1


def _scale_ncolors(I, which):
    k = _n_colors(I, which)
    if not k:
        return None
    return _scale_up(I, k, k) if k > 1 else I.a.copy()


_fam(name="scale_ncolors", fn=_scale_ncolors, instances=[("all",), ("fg",)], same_shape=False,
     describe=lambda which: f"scale the grid up by the number of {'colors' if which == 'all' else 'non-background colors'} "
                            f"it holds: every cell becomes a square block of that side")


def _tile_ncolors(I, which):
    k = _n_colors(I, which)
    if not k:
        return None
    return _tile(I, k, k) if k > 1 else I.a.copy()


_fam(name="tile_ncolors", fn=_tile_ncolors, instances=[("all",), ("fg",)], same_shape=False,
     describe=lambda which: f"repeat the grid as many times top to bottom and left to right as it holds "
                            f"{'colors' if which == 'all' else 'non-background colors'}")

# the families above are rivals only: generation (pv2.arc.gen.GEN) never draws them
RIVAL_ONLY = ("logic_halves", "overlay_halves", "mode_cell", "fill_const", "rays_skip", "rays_over",
              "recolor_size_cmp", "recolor_all_extreme", "remove_all_extreme", "crop_blob", "extract_fg",
              "extract_middle", "recolor_edge_objects", "remove_edge_objects", "tile_self", "keep_blob",
              "hollow_edge", "fill_interior_edge", "scale_ncolors", "tile_ncolors")


# ================================================================================================
# rules (one instance or a composition of two)
# ================================================================================================

@dataclass(frozen=True)
class Rule:
    """``steps``: ((family, params), ...) applied left to right."""
    steps: tuple

    @classmethod
    def single(cls, fam: str, *params) -> "Rule":
        return cls(((fam, tuple(params)),))

    @classmethod
    def chain(cls, *steps) -> "Rule":
        return cls(tuple((f, tuple(p)) for f, p in steps))

    def apply_info(self, info: Info) -> Optional[np.ndarray]:
        cur = info
        out = None
        for fam, p in self.steps:
            out = FAMILIES[fam].fn(cur, *p)
            if out is None or not fits(*out.shape):
                return None
            cur = Info(out)
        return out

    def apply(self, grid) -> Optional[list[list[int]]]:
        out = self.apply_info(Info(arr(grid)))
        return None if out is None else to_list(out)

    @property
    def token(self) -> str:
        return "+".join(f"{f}({','.join(str(x) for x in p)})" for f, p in self.steps)

    def describe(self) -> str:
        texts = [FAMILIES[f].describe(*p) for f, p in self.steps]
        if len(texts) == 1:
            return texts[0]
        return texts[0] + ", then " + texts[1]

    @property
    def bg(self) -> bool:
        return any(FAMILIES[f].bg for f, _ in self.steps)

    background_dependent = bg

    @property
    def geometric(self) -> bool:
        return len(self.steps) == 1 and FAMILIES[self.steps[0][0]].geometric

    @property
    def mono_ok(self) -> bool:
        return FAMILIES[self.steps[-1][0]].mono_ok

    @property
    def is_composition(self) -> bool:
        return len(self.steps) > 1


def all_instances():
    """Every (family, params) of the library."""
    for name, fam in FAMILIES.items():
        for p in fam.instances:
            yield name, p


def instance_count() -> int:
    return sum(len(f.instances) for f in FAMILIES.values())
