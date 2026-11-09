"""Scalar rules: grid -> number / colour / size, for ARC bundles whose outputs are values.

A scalar bundle shows train inputs with their values ("grid -> value" lines in the part context)
and asks for the test input's value.  ``certify_scalar`` checks the answer against the WHOLE scalar
library (every family, every parameter) plus the constant output: every rule consistent with all
train pairs must give the test value.  Numbers and colours are compared as integers across kinds
(a count rule whose values happen to be 3, 5, 1 also competes with a colour rule), sizes as text.

``reveals_dims`` marks rules computable from the grid's size alone (height, width, area, the
dimensions): their bundles never use an encoding or header that prints sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pv2.arc.grids import Info, arr, border_reachable, neighbour_count_same, regions
from pv2.arc.transforms import COLORS, CONNS, _extreme, _obj_phrase, cw


@dataclass
class ScalarFamily:
    name: str
    fn: Callable
    instances: tuple
    kind: str                     # "number" | "color" | "size"
    describe: Callable
    bg: bool = False
    reveals_dims: bool = False

    def __post_init__(self):
        self.instances = tuple(tuple(p) for p in self.instances)


SCALARS: dict[str, ScalarFamily] = {}


def _s(**kw):
    f = ScalarFamily(**kw)
    SCALARS[f.name] = f
    return f


def _dims(h, w) -> str:
    return f"{h}x{w}"


_s(name="count_color", kind="number", fn=lambda I, x: int(I.hist[x]), instances=[(x,) for x in COLORS],
   describe=lambda x: f"the number of cells of {cw(x)}")
_s(name="count_two", kind="number", fn=lambda I, x, y: int(I.hist[x] + I.hist[y]),
   instances=[(x, y) for x in COLORS for y in COLORS if x < y],
   describe=lambda x, y: f"the number of cells that are {cw(x)} or {cw(y)}")
_s(name="count_non_bg", kind="number", bg=True,
   fn=lambda I: None if I.bg is None else int(I.h * I.w - I.hist[I.bg]), instances=[()],
   describe=lambda: "the number of non-background cells")
_s(name="count_mode", kind="number", fn=lambda I: None if I.bg is None else int(I.hist[I.bg]), instances=[()],
   describe=lambda: "the number of cells of the most frequent color")
_s(name="count_least", kind="number",
   fn=lambda I: None if I.least() is None else int(I.hist[I.least()]), instances=[()],
   describe=lambda: "the number of cells of the least frequent color")
_s(name="n_colors", kind="number", fn=lambda I: len(I.colors), instances=[()],
   describe=lambda: "the number of different colors in the grid")
_s(name="n_fg_colors", kind="number", bg=True,
   fn=lambda I: None if I.bg is None else len(I.colors) - 1, instances=[()],
   describe=lambda: "the number of different non-background colors")
_s(name="height", kind="number", reveals_dims=True, fn=lambda I: I.h, instances=[()],
   describe=lambda: "the number of rows")
_s(name="width", kind="number", reveals_dims=True, fn=lambda I: I.w, instances=[()],
   describe=lambda: "the number of columns")
_s(name="area", kind="number", reveals_dims=True, fn=lambda I: I.h * I.w, instances=[()],
   describe=lambda: "the number of cells in the grid")
_s(name="max_side", kind="number", reveals_dims=True, fn=lambda I: max(I.h, I.w), instances=[()],
   describe=lambda: "the length of the grid's longer side")
_s(name="min_side", kind="number", reveals_dims=True, fn=lambda I: min(I.h, I.w), instances=[()],
   describe=lambda: "the length of the grid's shorter side")


def _objs(I, conn, min_size=1):
    objs = I.objects(conn)
    if objs is None:
        return None
    return [o for o in objs if o.size >= min_size]


_s(name="n_objects", kind="number", bg=True,
   fn=lambda I, c: None if _objs(I, c) is None else len(_objs(I, c)), instances=[(c,) for c in CONNS],
   describe=lambda c: f"the number of objects ({_obj_phrase(c)})")
_s(name="n_objects_min2", kind="number", bg=True,
   fn=lambda I, c: None if _objs(I, c, 2) is None else len(_objs(I, c, 2)), instances=[(c,) for c in CONNS],
   describe=lambda c: f"the number of objects of at least 2 cells ({_obj_phrase(c)})")
_s(name="n_objects_color", kind="number", bg=True,
   fn=lambda I, x, c: None if _objs(I, c) is None else sum(1 for o in _objs(I, c) if o.color == x),
   instances=[(x, c) for x in COLORS for c in CONNS],
   describe=lambda x, c: f"the number of objects of {cw(x)} ({_obj_phrase(c)})")
_s(name="n_blobs", kind="number", bg=True,
   fn=lambda I, c: None if I.blobs(c) is None else len(I.blobs(c)), instances=[(c,) for c in CONNS],
   describe=lambda c: "the number of groups of connected non-background cells, of any colors, connected "
                      + ("through shared edges" if c == 4 else "through shared edges or corners"))


def _size_of(I, which, c):
    objs = _objs(I, c)
    if not objs:
        return None
    sizes = [o.size for o in objs]
    return max(sizes) if which == "largest" else min(sizes)


_s(name="extreme_size", kind="number", bg=True, fn=_size_of,
   instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
   describe=lambda wh, c: f"the number of cells of the {wh} object ({_obj_phrase(c)})")


def _n_isolated(I, c):
    if I.bg is None:
        return None
    same = neighbour_count_same(I.a, c)
    return int(((I.a != I.bg) & (same == 0)).sum())


_s(name="n_isolated", kind="number", bg=True, fn=_n_isolated, instances=[(c,) for c in CONNS],
   describe=lambda c: f"the number of non-background cells with no neighbour of their own color "
                      f"{'sharing an edge' if c == 4 else 'sharing an edge or a corner'}")
_s(name="rows_with", kind="number", fn=lambda I, x: int((I.a == x).any(axis=1).sum()),
   instances=[(x,) for x in COLORS], describe=lambda x: f"the number of rows that contain {cw(x)}")
_s(name="cols_with", kind="number", fn=lambda I, x: int((I.a == x).any(axis=0).sum()),
   instances=[(x,) for x in COLORS], describe=lambda x: f"the number of columns that contain {cw(x)}")
_s(name="uniform_rows", kind="number", fn=lambda I: int((I.a == I.a[:, :1]).all(axis=1).sum()),
   instances=[()], describe=lambda: "the number of rows filled with a single color")
_s(name="uniform_cols", kind="number", fn=lambda I: int((I.a == I.a[:1, :]).all(axis=0).sum()),
   instances=[()], describe=lambda: "the number of columns filled with a single color")


def _hole_mask(I):
    if I.bg is None:
        return None
    open_ = I.a == I.bg
    return open_ & ~border_reachable(open_)


def _n_holes(I):
    m = I.memo("holes", lambda: _hole_mask(I))
    return None if m is None else len(regions(m))


def _hole_cells(I):
    m = I.memo("holes", lambda: _hole_mask(I))
    return None if m is None else int(m.sum())


_s(name="n_holes", kind="number", bg=True, fn=_n_holes, instances=[()],
   describe=lambda: "the number of enclosed pockets of background (background that cannot reach the edge "
                    "moving up, down, left or right through background)")
_s(name="hole_cells", kind="number", bg=True, fn=_hole_cells, instances=[()],
   describe=lambda: "the number of background cells that cannot reach the edge moving up, down, left or "
                    "right through background")

# --- colours --------------------------------------------------------------------------------------

_s(name="mode_color", kind="color", fn=lambda I: I.bg, instances=[()],
   describe=lambda: "the most frequent color")
_s(name="least_color", kind="color", fn=lambda I: I.least(), instances=[()],
   describe=lambda: "the least frequent color")


def _fg_rank(I, which):
    bg = I.bg
    if bg is None:
        return None
    items = sorted((int(I.hist[c]), c) for c in I.colors if c != bg)
    if not items:
        return None
    if which == "most":
        if len(items) >= 2 and items[-1][0] == items[-2][0]:
            return None
        return items[-1][1]
    if len(items) >= 2 and items[0][0] == items[1][0]:
        return None
    return items[0][1]


_s(name="fg_rank_color", kind="color", bg=True, fn=_fg_rank, instances=[("most",), ("least",)],
   describe=lambda wh: f"the {'most' if wh == 'most' else 'least'} frequent non-background color")


def _obj_color(I, which, c):
    objs = I.objects(c)
    if not objs:
        return None
    o = _extreme(objs, which)
    return None if o is None else o.color


_s(name="obj_color", kind="color", bg=True, fn=_obj_color,
   instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
   describe=lambda wh, c: f"the color of the object with the {'most' if wh == 'largest' else 'fewest'} cells "
                          f"({_obj_phrase(c)})")
_CORNER = {"tl": (0, 0), "tr": (0, -1), "bl": (-1, 0), "br": (-1, -1)}
_CORNER_WORD = {"tl": "top-left", "tr": "top-right", "bl": "bottom-left", "br": "bottom-right"}
_s(name="corner_color", kind="color", fn=lambda I, k: int(I.a[_CORNER[k]]), instances=[(k,) for k in _CORNER],
   describe=lambda k: f"the color of the {_CORNER_WORD[k]} corner cell")
_s(name="center_color", kind="color",
   fn=lambda I: int(I.a[I.h // 2, I.w // 2]) if I.h % 2 and I.w % 2 else None, instances=[()],
   describe=lambda: "the color of the center cell")

# --- sizes ------------------------------------------------------------------------------------------

_s(name="dims", kind="size", reveals_dims=True, fn=lambda I: _dims(I.h, I.w), instances=[()],
   describe=lambda: "the grid's size (rows x columns)")


def _bbox_dims(I, mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return _dims(int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1))


_s(name="content_dims", kind="size", bg=True,
   fn=lambda I: None if I.bg is None else _bbox_dims(I, I.a != I.bg), instances=[()],
   describe=lambda: "the size (rows x columns) of the smallest rectangle holding every non-background cell")
_s(name="color_dims", kind="size", fn=lambda I, x: _bbox_dims(I, I.a == x), instances=[(x,) for x in COLORS],
   describe=lambda x: f"the size (rows x columns) of the smallest rectangle holding every cell of {cw(x)}")


def _obj_dims(I, which, c):
    objs = I.objects(c)
    if not objs:
        return None
    o = _extreme(objs, which)
    return None if o is None else _dims(*o.bbox_dims)


_s(name="obj_dims", kind="size", bg=True, fn=_obj_dims,
   instances=[(wh, c) for wh in ("largest", "smallest") for c in CONNS],
   describe=lambda wh, c: f"the bounding-box size (rows x columns) of the object with the "
                          f"{'most' if wh == 'largest' else 'fewest'} cells ({_obj_phrase(c)})")


# --- rules and certification --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarRule:
    name: str
    params: tuple

    @property
    def family(self) -> ScalarFamily:
        return SCALARS[self.name]

    @property
    def kind(self) -> str:
        return self.family.kind

    @property
    def token(self) -> str:
        return f"{self.name}({','.join(str(x) for x in self.params)})"

    def value_info(self, info: Info):
        return self.family.fn(info, *self.params)

    def value(self, grid):
        return self.value_info(Info(arr(grid)))

    def describe(self) -> str:
        return self.family.describe(*self.params)


def all_scalar_rules():
    for name, fam in SCALARS.items():
        for p in fam.instances:
            yield ScalarRule(name, p)


def _norm(v):
    """Comparable form: numbers and colours as ints (they compete), sizes as text."""
    if v is None:
        return None
    return v if isinstance(v, str) else int(v)


@dataclass
class ScalarCertificate:
    values: Optional[list]
    reason: str
    consistent: list


def certify_scalar(train, tests, rule: ScalarRule) -> ScalarCertificate:
    """``train``: [(grid, value)].  The test values when every consistent scalar rule agrees."""
    vals = [_norm(v) for _, v in train]
    if len(set(vals)) < len(vals):
        return ScalarCertificate(None, "repeated_value", [])
    infos = [Info(arr(g)) for g, _ in train]
    tinfos = [Info(arr(t)) for t in tests]
    for info, v in zip(infos, vals):
        if _norm(rule.value_info(info)) != v:
            return ScalarCertificate(None, "rule_inconsistent", [])
    want = [_norm(rule.value_info(t)) for t in tinfos]
    if any(w is None for w in want):
        return ScalarCertificate(None, "rule_undefined_on_test", [])
    consistent = []
    for r in all_scalar_rules():
        ok = True
        for info, v in zip(infos, vals):
            if _norm(r.value_info(info)) != v:
                ok = False
                break
        if not ok:
            continue
        consistent.append(r.token)
        for t, w in zip(tinfos, want):
            pv = _norm(r.value_info(t))
            if pv is not None and pv != w:
                return ScalarCertificate(None, "rival_disagrees", consistent)
    return ScalarCertificate(want, "ok", consistent)
