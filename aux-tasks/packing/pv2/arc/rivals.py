"""Hypothesis elimination for ARC bundles (DESIGN.md sections 8 and 13.10).

The elimination set is a strict superset of what generation uses:

* every library instance (``transforms.FAMILIES``, every parameter) as a single rule, every wrap
  shift by -29..29 rows and columns (``general_rolls``) and every object-size rule with a size beyond
  20 (``size_thresholds``);
* every two-step composition ``T2 . T1`` of library instances;
* a FITTED colour map after the identity or any instance: any cell-wise colour function consistent
  with the train pairs (it stands for every fixed recolour / swap / keep-and-fill / constant-fill as
  a second step, and makes a test colour the train pairs never map AMBIGUOUS);
* the identity, the constant output and "copy train output k" (consistent only when every train
  output is the same grid, which ``certify`` refuses up front).

``certify`` returns the test outputs when every consistent rival gives the same output on every
test input (a rival undefined on a test input gives none), the generator's rule (or the frozen
generator's output) being that output; ``unique_test_output`` is the one-test form.

Every step of a rival must be observable: a two-step rival counts only when dropping either step
changes the output of at least one train pair.  A step whose effect never reaches a train output --
"remove the objects of 14 cells" when no train input has one, or when the second step crops them
away, or a shift that the gravity step after it undoes -- cannot be learned from the examples; the
one-step rival without it is in the set and fits the same pairs, so the two-step one is dominated
by it (Occam).  The strict reading would reject nearly every bundle: the test almost always holds
some feature that an unobservable step would act on.  A fitted map's first step is likewise dropped
when a pure colour map already fits.  Colours are the exception on purpose: a test colour that the
train pairs never show is AMBIGUOUS under the fitted map (DESIGN 13.10's witness rule), so such
bundles are rejected rather than resolved by the simpler reading.

How the search stays fast and exact.  Candidates come from the first train pair (the pivot): a
forward pass applies every instance to the pivot input; singles are the instances that reach the
pivot output; a two-step rival either (i) starts with a step that leaves the pivot input unchanged,
(ii) ends with a step that leaves the pivot output unchanged, or (iii) passes through a distinct
intermediate z, whose second steps ``solve(z, y)`` finds with per-family candidate generators and
necessary-condition filters (size, colours, what the changed cells became or were), then an exact
comparison.  Compositions with a step that alone fits every pair are dominated and skipped.  Two-step
candidates are kept as blocks (first steps x second steps) and split pair by pair only where their
intermediate grids differ, so equivalent parameter values are never checked one by one.  ``tests/test_arc_transforms.py``
checks ``solve`` against brute force, and ``tests/test_arc_bundle.py`` re-runs the whole rival set
on 2000+ parts with the separate nested loops of ``tests/_arc_naive.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pv2.arc.grids import Info, arr, fits
from pv2.arc.transforms import FAMILIES, Rule

# ------------------------------------------------------------------------------------------------
# per-family candidate generators for solve(z, y): which params could map z to y
# ------------------------------------------------------------------------------------------------


class Ctx:
    """Lazily computed facts about a (z, y) pair (diff / newc / oldc need the same size)."""
    __slots__ = ("z", "y", "_diff", "_newc", "_oldc", "_hist_eq", "_occurs")

    def __init__(self, z: Info, y: Info):
        self.z, self.y = z, y
        self._occurs = None
        self._diff = None
        self._newc = None
        self._oldc = None
        self._hist_eq = None

    @property
    def oldc(self) -> tuple:
        """Colours the changed cells had in z."""
        if self._oldc is None:
            self._oldc = tuple(int(c) for c in np.unique(self.z.a[self.diff]))
        return self._oldc

    @property
    def diff(self):
        if self._diff is None:
            self._diff = self.z.a != self.y.a
        return self._diff

    @property
    def newc(self) -> tuple:
        """Colours the changed cells take in y."""
        if self._newc is None:
            self._newc = tuple(int(c) for c in np.unique(self.y.a[self.diff]))
        return self._newc

    @property
    def hist_eq(self) -> bool:
        if self._hist_eq is None:
            self._hist_eq = bool(np.array_equal(self.z.hist, self.y.hist))
        return self._hist_eq


def _shape_filtered(fam, z, y):
    if fam.shape is None:
        yield from fam.instances
        return
    for p in fam.instances:
        s = fam.shape(z.h, z.w, *p)
        if s is False or (s is not None and s != y.shape):
            continue
        yield p


def _sub_ok(z, y):
    """y can be a proper sub-block of z: smaller, and no colour more frequent than in z."""
    return (y.h <= z.h and y.w <= z.w and (y.h, y.w) != (z.h, z.w)
            and bool(np.all(y.hist <= z.hist)))


def _occurs(z, y) -> bool:
    """y appears in z as a contiguous block (a crop's result always does)."""
    h, w = y.shape
    first = y.a[0].tobytes()
    for r in range(z.h - h + 1):
        row = z.a[r].tobytes()
        c = row.find(first)
        while 0 <= c <= z.w - w:
            if np.array_equal(z.a[r:r + h, c:c + w], y.a):
                return True
            c = row.find(first, c + 1)
    return False


def _block_ok(z, y, ctx) -> bool:
    if not _sub_ok(z, y):
        return False
    if ctx._occurs is None:
        ctx._occurs = _occurs(z, y)
    return ctx._occurs


def _c_block(fam, z, y, ctx):
    return fam.instances if _block_ok(z, y, ctx) else ()


def _c_top_left(fam, z, y, ctx):
    return fam.instances if _sub_ok(z, y) and np.array_equal(z.a[:y.h, :y.w], y.a) else ()


def _c_multiple(fam, z, y, ctx):
    """Scaling or tiling by a content-dependent factor: y's sides are the same multiple of z's."""
    if y.h % z.h or y.w % z.w or y.h // z.h != y.w // z.w:
        return ()
    return fam.instances


def _c_crop_content(fam, z, y, ctx):
    if not _block_ok(z, y, ctx) or z.bg is None:
        return ()
    bg = z.bg
    if int((y.a != bg).sum()) != int((z.a != bg).sum()):
        return ()
    return fam.instances


def _c_crop_color(fam, z, y, ctx):
    if not _block_ok(z, y, ctx):
        return ()
    out = []
    for (x,) in fam.instances:
        c = z.least() if x == "least" else x
        if c is None or not z.hist[c] or y.hist[c] != z.hist[c]:
            continue
        out.append((x,))
    return out


def _c_sub(fam, z, y, ctx):
    return fam.instances if _sub_ok(z, y) else ()


def _c_count_bar(fam, z, y, ctx):
    if y.h != 1 and y.w != 1:
        return ()
    cs = y.colors
    if len(cs) != 1:
        return ()
    x = cs[0]
    n = y.h * y.w
    if z.hist[x] != n:
        return ()
    out = []
    if y.h == 1:
        out.append((x, "h"))
    if y.w == 1:
        out.append((x, "v"))
    return out


def _c_extract(axis):
    def cands(fam, z, y, ctx):
        if axis == "row":
            if y.shape != (1, z.w):
                return ()
            hits = np.nonzero((z.a == y.a[0]).all(axis=1))[0]
            last = z.h - 1
        else:
            if y.shape != (z.h, 1):
                return ()
            hits = np.nonzero((z.a == y.a[:, 0:1]).all(axis=0))[0]
            last = z.w - 1
        out = [(int(i),) for i in hits]
        if last in hits:
            out.append((-1,))
        return out
    return cands


def _c_tile(fam, z, y, ctx):
    if y.h % z.h or y.w % z.w:
        return ()
    p = (y.h // z.h, y.w // z.w)
    return (p,) if p in fam._index else ()


def _c_scale(up):
    def cands(fam, z, y, ctx):
        if up:
            if y.h % z.h or y.w % z.w:
                return ()
            p = (y.h // z.h, y.w // z.w)
        else:
            if z.h % y.h or z.w % y.w:
                return ()
            p = (z.h // y.h, z.w // y.w)
        return (p,) if p in fam._index else ()
    return cands


def _c_logic(fam, z, y, ctx):
    bg = z.bg
    if bg is None:
        return ()
    cs = [c for c in y.colors if c != bg]
    if len(cs) > 1:
        return ()
    xs = cs if cs else [c for c in range(10) if c != bg]
    return [p for p in _shape_filtered(fam, z, y) if p[2] in xs]


def _c_fixed_map(fam, z, y, ctx):
    """Exact candidates for the fixed colour maps (used when they are not left to the fitted map)."""
    if z.shape != y.shape:
        return ()
    if not ctx.diff.any():
        return ()
    old = tuple(int(c) for c in np.unique(z.a[ctx.diff]))
    new = ctx.newc
    name = fam.name
    if name == "recolor":
        return [(old[0], new[0])] if len(old) == 1 and len(new) == 1 and old[0] != new[0] else ()
    if name == "swap":
        both = sorted(set(old) | set(new))
        return [tuple(both)] if len(both) == 2 else ()
    if name == "keep_fill":
        return [(x, new[0]) for x in range(10) if x != new[0]] if len(new) == 1 else ()
    if name == "fill_const":
        return [(y.colors[0],)] if len(y.colors) == 1 else ()
    return fam.instances


def _c_shift(fam, z, y, ctx):
    bg = z.bg
    if bg is None:
        return ()

    def lead(lines):
        n = 0
        for line in lines[:3]:
            if not bool(np.all(line == bg)):
                break
            n += 1
        return n
    top, bottom = lead(y.a), lead(y.a[::-1])
    left, right = lead(y.a.T), lead(y.a.T[::-1])
    return [(dr, dc) for dr, dc in fam.instances
            if (dr <= top if dr > 0 else -dr <= bottom) and (dc <= left if dc > 0 else -dc <= right)]


def _c_roll(fam, z, y, ctx):
    if not ctx.hist_eq:
        return ()
    h, w = z.h, z.w
    first = y.a[0].tobytes()
    out = []
    for dr, dc in fam.instances:
        if (dr and abs(dr) >= h) or (dc and abs(dc) >= w):
            continue
        rb = z.a[(-dr) % h].tobytes()
        k = (-dc) % w
        if rb[k:] + rb[:k] == first:
            out.append((dr, dc))
    return out


def _c_invert2(fam, z, y, ctx):
    return fam.instances if len(z.colors) == 2 and y.colors == z.colors else ()


def _touched(z, conn, diff):
    """(sizes of the objects wholly inside the changed cells, sizes of the untouched objects,
    colours of the changed objects), or None when an object is only partly changed."""
    objs = z.objects(conn)
    if objs is None:
        return None
    flat = diff.ravel()
    inside, outside, cols = [], [], []
    for o in objs:
        hit = flat[list(o.cells)]
        if hit.all():
            inside.append(o.size)
            cols.append(o.color)
        elif hit.any():
            return None
        else:
            outside.append((o.size, o.color))
    return inside, outside, cols


def _c_remove_size(fam, z, y, ctx):
    out = []
    for conn in (4, 8):
        t = _touched(z, conn, ctx.diff)
        if t is None or not t[0]:
            continue
        rem, keep = set(t[0]), {sz for sz, _ in t[1]}
        if len(rem) == 1:
            n = next(iter(rem))
            if n not in keep:
                out.append(("eq", n, conn))
        lo = max(rem) + 1
        hi = min(keep) if keep else 21
        out += [("lt", n, conn) for n in range(lo, hi + 1)]
        lo2 = max(keep) if keep else 1
        out += [("gt", n, conn) for n in range(lo2, min(rem))]
    return [p for p in out if p in fam._index]


def _c_recolor_size(fam, z, y, ctx):
    if len(ctx.newc) != 1:
        return ()
    x = ctx.newc[0]
    out = []
    for conn in (4, 8):
        t = _touched(z, conn, ctx.diff)
        if t is None or not t[0] or len(set(t[0])) != 1:
            continue
        n = t[0][0]
        if not any(sz == n and c != x for sz, c in t[1]):
            out.append((n, x, conn))
    return [p for p in out if p in fam._index]


def _c_recolor_size_cmp(fam, z, y, ctx):
    if len(ctx.newc) != 1:
        return ()
    x = ctx.newc[0]
    out = []
    for conn in (4, 8):
        t = _touched(z, conn, ctx.diff)
        if t is None or not t[0]:
            continue
        inside = t[0]
        others = [sz for sz, c in t[1] if c != x]
        hi = min(others) if others else 21
        out += [("lt", n, x, conn) for n in range(max(inside) + 1, hi + 1)]
        lo = max(others) if others else 1
        out += [("gt", n, x, conn) for n in range(lo, min(inside))]
    return [p for p in out if p in fam._index]


SAME_CANDS = {"shift": _c_shift, "roll": _c_roll, "invert2": _c_invert2,
              "remove_size": _c_remove_size, "recolor_size": _c_recolor_size,
              "recolor_size_cmp": _c_recolor_size_cmp}

CANDS = {
    "crop_content": _c_crop_content,
    "crop_color": _c_crop_color,
    "scale_ncolors": _c_multiple,
    "tile_ncolors": _c_multiple,
    "crop_object": _c_block,
    "crop_blob": _c_block,
    "compress": _c_sub,
    "drop_bg_lines": _c_sub,
    "smallest_tile": _c_top_left,
    "count_bar": _c_count_bar,
    "extract_row": _c_extract("row"),
    "extract_col": _c_extract("col"),
    "tile": _c_tile,
    "scale_up": _c_scale(True),
    "scale_down": _c_scale(False),
    "logic_halves": _c_logic,
}

for _f in FAMILIES.values():
    _f._index = set(_f.instances)

SAME_BY_SIG: dict[str, list] = {}
SHAPED: list = []
for _f in FAMILIES.values():
    if _f.same_shape:
        SAME_BY_SIG.setdefault(_f.sig, []).append(_f)
    else:
        SHAPED.append(_f)


def _eq(r: Optional[np.ndarray], y: Info) -> bool:
    return r is not None and r.shape == y.a.shape and bool(np.array_equal(r, y.a))


def solve(z: Info, y: Info, *, fixed_maps: bool = True) -> list[tuple]:
    """Every (family, params) with fn(z) == y exactly (instances that leave z unchanged are
    included only when z == y is excluded by the caller; here z != y is assumed for same-size
    pairs, and a no-op can never reach y)."""
    out = []
    same = z.shape == y.shape
    ctx = Ctx(z, y)
    if same and not ctx.diff.any():
        return out
    known = set(y.colors) <= set(z.colors)

    def try_all(fam, params):
        for p in params:
            if fam.noop is not None and fam.noop(z, *p):
                continue
            if _eq(fam.fn(z, *p), y):
                out.append((fam.name, p))

    if same:
        for sig, fams in SAME_BY_SIG.items():
            if sig == "fixed_map":
                if fixed_maps:
                    for fam in fams:
                        try_all(fam, _c_fixed_map(fam, z, y, ctx))
                continue
            if sig == "paint":
                if len(ctx.newc) != 1:
                    continue
                x = ctx.newc[0]
                only_bg = z.bg is not None and ctx.oldc == (z.bg,)
                for fam in fams:
                    if fam.from_bg and not only_bg:
                        continue
                    cands = SAME_CANDS.get(fam.name)
                    try_all(fam, cands(fam, z, y, ctx) if cands is not None else fam.by_paint.get(x, ()))
                continue
            if sig == "erase":
                if z.bg is None or ctx.newc != (z.bg,):
                    continue
            elif sig == "fill_bg":
                if z.bg is None or ctx.oldc != (z.bg,):
                    continue
            elif sig == "hist":
                if not ctx.hist_eq:
                    continue
            for fam in fams:
                if not known and not fam.new_colors:
                    continue
                cands = SAME_CANDS.get(fam.name)
                try_all(fam, cands(fam, z, y, ctx) if cands is not None else fam.instances)
    for fam in SHAPED:
        if fam.sig == "hist" and not ctx.hist_eq:
            continue
        if not known and not fam.new_colors:
            continue
        cands = CANDS.get(fam.name)
        params = cands(fam, z, y, ctx) if cands is not None else _shape_filtered(fam, z, y)
        try_all(fam, params)
    return out


# ------------------------------------------------------------------------------------------------
# forward pass and step evaluation with caches
# ------------------------------------------------------------------------------------------------

class Space:
    """Caches for one search: interned grids and step results."""

    def __init__(self):
        self.infos: dict = {}
        self.steps: dict = {}

    def intern(self, a: np.ndarray) -> Info:
        k = (a.shape[0], a.shape[1], a.tobytes())
        info = self.infos.get(k)
        if info is None:
            info = Info(np.ascontiguousarray(a))
            info._key = k
            self.infos[k] = info
        return info

    def step(self, inst: Optional[tuple], z: Optional[Info]) -> Optional[Info]:
        """Apply one instance (None: identity) to an interned grid."""
        if z is None:
            return None
        if inst is None:
            return z
        k = (inst, z.key)
        if k in self.steps:
            return self.steps[k]
        fam = FAMILIES[inst[0]]
        p = inst[1]
        res = None
        if not fam.same_shape:
            s = fam.out_shape(z.h, z.w, p)
            if s is False:
                self.steps[k] = None
                return None
        if fam.noop is not None and fam.noop(z, *p):
            res = z
        else:
            r = fam.fn(z, *p)
            if r is not None and fits(*r.shape):
                res = self.intern(r)
        self.steps[k] = res
        return res

    def forward(self, x: Info) -> dict:
        """key -> (Info, [instances]) for every instance defined on x (no-ops land on x's key)."""
        groups: dict = {}
        for name, fam in FAMILIES.items():
            for p in fam.instances:
                z = self.step((name, p), x)
                if z is None:
                    continue
                g = groups.get(z.key)
                if g is None:
                    groups[z.key] = g = (z, [])
                g[1].append((name, p))
        return groups


def cellmap(z: Info, y: Info) -> Optional[dict]:
    """The colour function f with f(z) == y cell by cell, or None."""
    if z.shape != y.shape:
        return None
    pairs = np.bincount(z.a.ravel().astype(np.int16) * 10 + y.a.ravel(), minlength=100).reshape(10, 10)
    m = {}
    for c in range(10):
        nz = np.nonzero(pairs[c])[0]
        if len(nz) > 1:
            return None
        if len(nz) == 1:
            m[c] = int(nz[0])
    return m


def _merge(m: dict, extra: dict) -> Optional[dict]:
    out = dict(m)
    for k, v in extra.items():
        if out.get(k, v) != v:
            return None
        out[k] = v
    return out


# ------------------------------------------------------------------------------------------------
# rivals and the certificate
# ------------------------------------------------------------------------------------------------

AMBIGUOUS = "ambiguous"


class PoolTooLarge(Exception):
    """The refined two-step candidates split into more blocks than the search will check (the
    bundle is refused)."""


@dataclass
class Rival:
    kind: str                       # "single" | "comp" | "fitted" | "roll" (a general wrap shift t2=(dr, dc))
    t1: Optional[tuple]             # first step (None: identity)
    t2: Optional[tuple]             # second step (None for fitted)
    cmap: Optional[dict] = None
    fail_at: int = -1               # first train pair it contradicts (-1: none)
    vacuous: bool = False           # a step never changes any train grid (see the module docstring)

    @property
    def token(self) -> str:
        def one(inst):
            if inst is None:
                return "identity"
            return f"{inst[0]}({','.join(str(x) for x in inst[1])})"
        if self.kind == "single":
            return one(self.t2)
        if self.kind == "comp":
            return one(self.t1) + "+" + one(self.t2)
        if self.kind == "roll":
            return f"roll_any({self.t2[0]},{self.t2[1]})"
        return one(self.t1) + "+fitted_map"

    def rule(self) -> Optional[Rule]:
        if self.kind == "single":
            return Rule((self.t2,))
        if self.kind == "comp":
            return Rule((self.t1, self.t2))
        return None


@dataclass
class Certificate:
    outputs: Optional[list]                 # int8 arrays per test input, or None
    reason: str
    n_pivot: int = 0                        # rivals consistent with the pivot pair
    n_vacuous: int = 0                      # consistent rivals with a step that never acts
    eliminated: list = field(default_factory=list)   # per pair i >= 1: rivals first contradicted there
    consistent: list = field(default_factory=list)   # tokens of the rivals consistent with every pair (capped)
    single_alive: bool = False                       # a single instance or a pure colour map fits every pair
    predictions: list = field(default_factory=list)  # per test: set of predicted keys


ROLL_RANGE = range(-29, 30)


def _roll_residues(x: Info, y: Info) -> set:
    """Offsets (a, b), 0 <= a < h, 0 <= b < w, with np.roll(x, (a, b)) == y."""
    if x.shape != y.shape or not np.array_equal(x.hist, y.hist):
        return set()
    h, w = x.shape
    first = y.a[0].tobytes()
    out = set()
    for a in range(h):
        rb = x.a[(-a) % h].tobytes()
        for b in range(w):
            k = (-b) % w
            if rb[k:] + rb[:k] == first and np.array_equal(np.roll(x.a, (a, b), axis=(0, 1)), y.a):
                out.add((a, b))
    return out


def general_rolls(P) -> list:
    """Every wrap shift (dr, dc), -29 <= dr, dc <= 29, that maps each train input to its output.
    The library's ``roll`` family holds the small shifts used for generation and compositions; on a
    width-6 grid a shift of 5 columns right is also one of 1 column left, and a test of another width
    tells them apart, so every shift is a (single-step) rival."""
    res = []
    for x, y in P:
        R = _roll_residues(x, y)
        if not R:
            return []
        res.append((x.h, x.w, R))
    out = []
    for dr in ROLL_RANGE:
        if not all(any(dr % h == a for a, _ in R) for h, w, R in res):
            continue
        for dc in ROLL_RANGE:
            if all((dr % h, dc % w) in R for h, w, R in res):
                out.append((dr, dc))
    return out


def size_thresholds(P, T) -> list:
    """Object-size rules with a size beyond the library's 1-20 (big grids hold bigger objects):
    remove / recolour objects of exactly, fewer than or more than n cells.  A rule's outcome only
    changes at the object sizes present, so n at every size s and s+1 found in the train and test
    inputs (above 20) stands for every integer.  Returns the instances that fit every pair."""
    sizes = set()
    for g in [x for x, _ in P] + list(T):
        for conn in (4, 8):
            objs = g.objects(conn)
            if objs:
                sizes |= {o.size for o in objs}
    ns = sorted({s for s in sizes if s > 20} | {s + 1 for s in sizes if s + 1 > 20})
    if not ns:
        return []
    x0, y0 = P[0]
    painted = set(y0.colors)
    cands = []
    for n in ns:
        for conn in (4, 8):
            cands += [("remove_size", (cmp, n, conn)) for cmp in ("eq", "lt", "gt")]
            for x in painted:
                cands.append(("recolor_size", (n, x, conn)))
                cands += [("recolor_size_cmp", (cmp, n, x, conn)) for cmp in ("lt", "gt")]
    out = []
    for name, p in cands:
        fam = FAMILIES[name]
        if all(_eq(fam.fn(x, *p), y) for x, y in P):
            out.append((name, p))
    return out


def _pairs_info(space: Space, pairs) -> list:
    return [(space.intern(arr(x)), space.intern(arr(y))) for x, y in pairs]


@dataclass
class Block:
    """Two-step rivals as a product: every first step in ``t1s`` followed by every second step in
    ``t2s``.  Many parameter values give the same grid (all size thresholds between two object
    sizes, say), so candidates are kept as products and split only where they behave differently."""
    t1s: list
    t2s: list

    @property
    def size(self) -> int:
        return len(self.t1s) * len(self.t2s)


@dataclass
class Pool:
    rivals: list                    # singles and fitted maps (``Rival`` with ``fail_at``)
    blocks: list                    # two-step rivals consistent with every train pair
    space: Space
    P: list
    T: list
    n_pivot: int                    # rivals consistent with the pivot pair
    eliminated: list                # per pair i >= 1: rivals it contradicts first


def search(pairs, tests, *, cap: int = 100000) -> Pool:
    """Every rival consistent with the pivot pair (pairs[0]), then checked against the others:
    singles and fitted maps get ``fail_at``; two-step rivals are refined as blocks."""
    space = Space()
    P = _pairs_info(space, pairs)
    # the pivot is the pair with the smallest input (cheapest forward pass); every rival is checked
    # on every pair anyway, so the choice changes speed only
    k = min(range(len(P)), key=lambda i: P[i][0].h * P[i][0].w)
    P = [P[k]] + P[:k] + P[k + 1:]
    T = [space.intern(arr(t)) for t in tests]
    x0, y0 = P[0]
    F = space.forward(x0)
    rivals: list[Rival] = []

    singles = list(F.get(y0.key, (None, []))[1]) if y0.key != x0.key else []
    # a composition with a step that alone fits every pair is dominated (see the docstring): find
    # the singles that fit every pair first and never pair them with a second step
    full = set()
    partial = []
    for inst in singles:
        if all(space.step(inst, x) is y for x, y in P[1:]):
            full.add(inst)
        else:
            partial.append(inst)
    for inst in singles:
        rivals.append(Rival("single", None, inst))
    pure = _pure_map_fits(P)

    blocks: list[Block] = []
    noop_x = list(F.get(x0.key, (None, []))[1])
    # (iii) genuine two-step candidates through an intermediate z != x0, y0
    for k, (z, t1s) in F.items():
        if k == x0.key or k == y0.key:
            continue
        t2s = [t2 for t2 in solve(z, y0, fixed_maps=False) if t2 not in full]
        firsts = [t1 for t1 in t1s if t1 not in full]
        if t2s and firsts:
            blocks.append(Block(firsts, t2s))
    # (i) a first step that leaves x0 unchanged, then a single that fits x0 but not every pair
    part_nonfixed = [s for s in partial if FAMILIES[s[0]].sig != "fixed_map"]
    if noop_x and part_nonfixed:
        blocks.append(Block(noop_x, part_nonfixed))
    # (ii) a single that fits x0 but not every pair, then a step that leaves y0 unchanged
    if partial:
        noop_y = []
        for name, fam in FAMILIES.items():
            if fam.sig == "fixed_map":
                continue
            for p in fam.instances:
                if space.step((name, p), y0) is y0:
                    noop_y.append((name, p))
        if noop_y:
            blocks.append(Block(partial, noop_y))
    # fitted colour maps after the identity or (when no pure map fits) any single step
    firsts = [(x0, [None] if pure else [None] + noop_x)]
    if not pure:
        firsts += [(z, t1s) for k, (z, t1s) in F.items() if k != x0.key]
    for z, t1s in firsts:
        m = cellmap(z, y0)
        if m is None:
            continue
        for t1 in t1s:
            rivals.append(Rival("fitted", t1, None, cmap=m))
    n_pivot = len(rivals) + sum(b.size for b in blocks)

    # check the other pairs, first failure recorded (blocks split by their intermediate grids)
    elim = []
    for i in range(1, len(P)):
        xi, yi = P[i]
        gone = 0
        for rv in rivals:
            if rv.fail_at != -1:
                continue
            z = space.step(rv.t1, xi)
            if z is None:
                rv.fail_at = i
            elif rv.kind == "fitted":
                m = cellmap(z, yi)
                merged = None if m is None else _merge(rv.cmap, m)
                if merged is None:
                    rv.fail_at = i
                else:
                    rv.cmap = merged
            elif space.step(rv.t2, z) is not yi:
                rv.fail_at = i
            gone += rv.fail_at == i
        refined = []
        for b in blocks:
            by_z = {}
            for t1 in b.t1s:
                z = space.step(t1, xi)
                if z is None:
                    gone += len(b.t2s)
                    continue
                g = by_z.get(z.key)
                if g is None:
                    by_z[z.key] = g = (z, [])
                g[1].append(t1)
            for z, t1s in by_z.values():
                keep = [t2 for t2 in b.t2s if space.step(t2, z) is yi]
                gone += len(t1s) * (len(b.t2s) - len(keep))
                if keep:
                    refined.append(Block(t1s, keep))
        if len(refined) > cap:
            raise PoolTooLarge(len(refined))
        blocks = refined
        elim.append(gone)
    # every wrap shift that fits every pair (single-step rivals beyond the library's small shifts)
    rolls = general_rolls(P)
    rivals += [Rival("roll", None, s) for s in rolls]
    # object-size rules with sizes beyond the library's 1-20
    sizes = [inst for inst in size_thresholds(P, T) if inst[1] not in FAMILIES[inst[0]]._index]
    rivals += [Rival("single", None, inst) for inst in sizes]
    return Pool(rivals, blocks, space, P, T, n_pivot + len(rolls) + len(sizes), elim)


def is_vacuous(rv: Rival, space: Space, P, pure_map_fits: bool) -> bool:
    """True when the first step of a consistent fitted-map rival is unobservable (a pure colour
    map already fits every pair); two-step blocks are pruned the same way in ``_observable``."""
    return rv.kind == "fitted" and rv.t1 is not None and pure_map_fits


def _observable(pool: Pool) -> tuple[list, int]:
    """Blocks without the unobservable steps: a second step that alone fits every pair, or a first
    step that alone does (dropping the other step changes no train output)."""
    sp, P = pool.space, pool.P
    alone = {}

    def fits_alone(inst):
        if inst not in alone:
            alone[inst] = all(sp.step(inst, x) is y for x, y in P)
        return alone[inst]
    out, dropped = [], 0
    for b in pool.blocks:
        t1s = [t for t in b.t1s if not fits_alone(t)]
        t2s = [t for t in b.t2s if not fits_alone(t)]
        dropped += b.size - len(t1s) * len(t2s)
        if t1s and t2s:
            out.append(Block(t1s, t2s))
    return out, dropped


def _pure_map_fits(P) -> bool:
    m = {}
    for x, y in P:
        c = cellmap(x, y)
        if c is None:
            return False
        m = _merge(m, c)
        if m is None:
            return False
    return True


def predict(rv: Rival, space: Space, t: Info):
    """A consistent single / fitted rival's output on a test input: an Info, None (undefined) or
    AMBIGUOUS."""
    if rv.kind == "roll":
        return space.intern(np.ascontiguousarray(np.roll(t.a, rv.t2, axis=(0, 1))))
    z = space.step(rv.t1, t)
    if z is None:
        return None
    if rv.kind == "fitted":
        cols = z.colors
        if any(c not in rv.cmap for c in cols):
            return AMBIGUOUS
        lut = np.arange(10, dtype=np.int8)
        for k, v in rv.cmap.items():
            lut[k] = v
        return space.intern(lut[z.a])
    return space.step(rv.t2, z)


def _block_predictions(b: Block, space: Space, t: Info) -> set:
    out = set()
    by_z = {}
    for t1 in b.t1s:
        z = space.step(t1, t)
        if z is not None:
            by_z[z.key] = z
    for z in by_z.values():
        for t2 in b.t2s:
            o = space.step(t2, z)
            if o is not None:
                out.add(o.key)
    return out


def _tokens(alive: list, blocks: list, limit: int = 50, scan: int = 5000) -> list:
    toks = {r.token for r in alive}
    for b in blocks:
        for t1 in b.t1s:
            for t2 in b.t2s:
                if len(toks) >= scan:
                    break
                toks.add(Rival("comp", t1, t2).token)
    return sorted(toks)[:limit]


def certify(pairs, tests, rule: Optional[Rule] = None, *, expected=None) -> Certificate:
    """Certify the test outputs of a bundle.

    ``rule`` (the generator's rule) or ``expected`` (outputs from a frozen generator) gives the
    intended outputs; they must be what every consistent rival predicts."""
    if len(pairs) < 1 or not tests:
        return Certificate(None, "empty")
    ys = [arr(y) for _, y in pairs]
    if all(y.shape == ys[0].shape and np.array_equal(y, ys[0]) for y in ys):
        return Certificate(None, "constant_output")
    if all(arr(x).shape == y.shape and np.array_equal(arr(x), y) for (x, _), y in zip(pairs, ys)):
        return Certificate(None, "identity")
    if rule is not None:
        want = []
        for (x, y) in pairs:
            r = rule.apply_info(Info(arr(x)))
            if r is None or r.shape != arr(y).shape or not np.array_equal(r, arr(y)):
                return Certificate(None, "rule_inconsistent")
        for t in tests:
            r = rule.apply_info(Info(arr(t)))
            if r is None:
                return Certificate(None, "rule_undefined_on_test")
            want.append(r)
    else:
        want = [arr(e) for e in expected]
    try:
        pool = search(pairs, tests)
    except PoolTooLarge:
        return Certificate(None, "pool_too_large")
    space, P, T = pool.space, pool.P, pool.T
    alive = [r for r in pool.rivals if r.fail_at == -1]
    pure = _pure_map_fits(P)
    n_vac = sum(1 for rv in alive if is_vacuous(rv, space, P, pure))
    alive = [rv for rv in alive if not is_vacuous(rv, space, P, pure)]
    blocks, dropped = _observable(pool)
    cert = Certificate(None, "ok", n_pivot=pool.n_pivot, eliminated=pool.eliminated, n_vacuous=n_vac + dropped,
                       consistent=_tokens(alive, blocks),
                       single_alive=any(r.kind in ("single", "roll") or (r.kind == "fitted" and r.t1 is None)
                                        for r in alive))
    for j, t in enumerate(T):
        wk = space.intern(want[j]).key
        preds = set()
        for rv in alive:
            out = predict(rv, space, t)
            if out is None:
                continue
            preds.add(AMBIGUOUS if out is AMBIGUOUS else out.key)
        for b in blocks:
            preds |= _block_predictions(b, space, t)
        cert.predictions.append(preds)
        if AMBIGUOUS in preds:
            cert.reason = "ambiguous_colour"
            return cert
        if preds != {wk}:
            cert.reason = "rival_disagrees" if preds else "no_rival"
            return cert
    cert.outputs = want
    return cert


def unique_test_output(train_pairs, test_input, generator_rule: Rule):
    """The certified output for ``test_input`` (a list-of-lists grid), or None."""
    cert = certify(train_pairs, [test_input], generator_rule)
    if cert.outputs is None:
        return None
    return [[int(v) for v in row] for row in cert.outputs[0]]
