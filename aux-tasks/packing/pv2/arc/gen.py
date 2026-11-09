"""What generation draws: rule families with parameter proposals, and the pictures they act on.

A ``GenSpec`` says how to draw one library rule for a bundle: structural parameters decided before
the pictures (``draw``: a scale factor, a direction ...), size limits that follow from them
(``limits``), optional constructed inputs (``make``: block-scaled pictures for scale-down, tiled
tiles for the smallest tile, pictures with duplicated rows for compress, noise on a plain ground
for removing isolated cells ...), and the remaining parameters chosen from the pictures
(``params``: e.g. a colour present in every input).  The rival search, not the proposal, decides
whether the bundle is sound; proposals only make acceptance likely.

Pictures are the reviewed generators' inputs (``pv2.pictures``), whole when they fit the size cap
and otherwise a window of one.  Most bundles take all their pictures from one task (a coherent
look, as in ARC) and map every picture's colours by frequency rank onto one bundle palette, so the
test colours are witnessed by the train pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from pv2.arc.grids import DT, Info, arr, fits, is_solid
from pv2.arc.transforms import FAMILIES, Rule
from pv2.pictures import picture_tasks, try_generate

from registry import generate  # noqa: E402  (pv2.pictures put the generator package on the path)


class Reject(Exception):
    """A bundle attempt failed a check; ``reason`` is counted for the report."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------------------------------------------
# pictures
# ------------------------------------------------------------------------------------------------

# generators slower than ~5 ms per example are not used as picture sources for library bundles
SLOW = {"22", "25", "68a", "68b", "84", "85", "13", "28", "97", "98", "99", "102", "7", "8", "90"}
SOURCES = [t for t in picture_tasks() if t not in SLOW]

# tasks whose pictures suit some rules (holes, solid rectangles, sparse markers)
HOLE_SOURCES = ["36", "37", "49", "50", "83", "26", "96", "81", "57", "59"]
RECT_SOURCES = ["49", "67", "83", "50", "27", "91", "29", "30"]


@dataclass
class Picture:
    grid: np.ndarray
    tid: str
    seed: int
    window: Optional[tuple] = None      # (r0, c0, h, w) when cut from a larger picture


def _two_sides(r, draw):
    """``draw(first_max, second_max)`` picks the first side, then the second under the cell cap.  Which
    of H and W goes first is random: height-first always made small windows tall (composition
    bundles: 60% taller than wide, 36% wider; parity review 2026-09-30)."""
    if r.random() < 0.5:
        return draw(False)
    out = draw(True)
    return None if out is None else (out[1], out[0])


def _window(r, a: np.ndarray, *, max_cells: int, max_h: int, max_w: int, min_side: int, min_cells: int = 0):
    H, W = a.shape

    def draw(swap):
        A, B, ma, mb = (W, H, max_w, max_h) if swap else (H, W, max_h, max_w)
        hi_a = min(A, ma, max(min_side, max_cells // min_side))
        lo_a = max(min_side, -(-min_cells // max(1, min(B, mb)))) if min_cells else min_side
        if hi_a < lo_a:
            return None
        x = int(r.integers(lo_a, hi_a + 1))
        hi_b = min(B, mb, max_cells // x)
        lo_b = max(min_side, -(-min_cells // x)) if min_cells else min_side
        if hi_b < lo_b:
            return None
        return x, int(r.integers(lo_b, hi_b + 1))

    sides = _two_sides(r, draw)
    if sides is None:
        return None
    h, w = sides
    r0 = int(r.integers(0, H - h + 1))
    c0 = int(r.integers(0, W - w + 1))
    return a[r0:r0 + h, c0:c0 + w].copy(), (r0, c0, h, w)


def _good(a: np.ndarray, solid: bool) -> bool:
    if len(np.unique(a)) < 2:
        return False
    if solid:
        info = Info(a)
        if info.bg is None or not is_solid(a):
            return False
        share = info.hist[info.bg] / a.size
        if share > 0.93:
            return False
    return True


def sample_pictures(r, n: int, *, solid: bool, max_cells: int, max_h: int = 30, max_w: int = 30,
                    min_side: int = 3, sources: Optional[list] = None, same_task: Optional[bool] = None,
                    tries: int = 60, min_cells: int = 0) -> list[Picture]:
    """n pictures within the size limits; one source task for the bundle most of the time.
    ``min_cells`` (large bundles): smaller pictures are skipped and windows are cut at least that
    big; a fixed source task that cannot supply them gives way to random tasks after 20 draws."""
    max_cells = max(min_side * min_side, max_cells)
    pool = sources or SOURCES
    if same_task is None:
        same_task = r.random() < 0.7
    fixed = pool[int(r.integers(len(pool)))] if same_task else None
    out: list[Picture] = []
    for k in range(tries):
        if len(out) == n:
            break
        if min_cells and fixed and k == 20:
            fixed = None
        tid = fixed or pool[int(r.integers(len(pool)))]
        seed = int(r.integers(0, 2 ** 31))
        ex = try_generate(tid, seed)
        if ex is None:
            continue
        g = ex["input"]
        a = arr(g)
        if a.size < min_cells:
            continue
        if a.shape[0] <= max_h and a.shape[1] <= max_w and a.size <= max_cells and min(a.shape) >= min_side:
            if _good(a, solid):
                out.append(Picture(a, tid, seed))
                continue
        for _w in range(6):
            cut = _window(r, a, max_cells=max_cells, max_h=max_h, max_w=max_w, min_side=min_side,
                          min_cells=min_cells)
            if cut is None:
                break
            b, win = cut
            if _good(b, solid):
                out.append(Picture(b, tid, seed, win))
                break
    if len(out) < n:
        raise Reject("pictures")
    return out


from common import shapes as _shapes  # noqa: E402  (the reviewed shape vocabulary, read-only use)

SCENE_FAMILIES = ("polyomino", "bar", "rect", "square", "frame", "L", "T", "U", "plus", "H", "staircase",
                  "blob", "big_blob", "scaled")


def _place(r, grid, occ, shape, color, tries=30) -> bool:
    """Put a shape where neither it nor its one-cell halo meets another object."""
    h, w = grid.shape
    sh = max(p[0] for p in shape) + 1
    sw = max(p[1] for p in shape) + 1
    if sh > h or sw > w:
        return False
    cells = np.array(sorted(shape))
    for _ in range(tries):
        r0 = int(r.integers(0, h - sh + 1))
        c0 = int(r.integers(0, w - sw + 1))
        rr, cc = cells[:, 0] + r0, cells[:, 1] + c0
        if occ[rr, cc].any():
            continue
        grid[rr, cc] = color
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                y, x = rr + dr, cc + dc
                ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
                occ[y[ok], x[ok]] = True
        return True
    return False


def draw_scene(r, h: int, w: int, palette: list[int], *, must: tuple = (), n_obj: Optional[int] = None,
               marker: Optional[int] = None, size: Optional[int] = None, margin: int = 0,
               pairs: bool = False) -> Optional[np.ndarray]:
    """A plain ground (palette[0]) with whole objects from ``common.shapes`` that never touch.

    ``must``: features every scene needs -- "hole" (a closed frame), "rect" (a solid rectangle of at
    least 3x3), "sizes" (two objects of ``size`` cells), "diag" (two same-colour objects meeting only
    at a corner, so 4- and 8-connectivity disagree).  ``margin``: rows / columns kept clear at the
    edges.  ``marker``: a colour placed as 1-3 lone cells (``pairs``: two cells facing each other
    across the ground in one row or column).  Object colours cycle through the palette, so a scene
    with enough objects shows every colour."""
    bg = palette[0]
    colors = [c for c in palette[1:]] or [c for c in range(10) if c != bg]
    shift = int(r.integers(len(colors)))
    grid = np.full((h, w), bg, dtype=DT)
    occ = np.zeros((h, w), dtype=bool)
    if margin:
        occ[:margin, :] = occ[-margin:, :] = True
        occ[:, :margin] = occ[:, -margin:] = True
    area = h * w
    if n_obj is None:
        n_obj = int(r.integers(2, max(3, min(8, area // 20)) + 1))
    count = [0]

    def col():
        c = colors[(shift + count[0]) % len(colors)]
        count[0] += 1
        return c

    for feat in must:
        if feat == "hole":
            fh, fw = int(r.integers(3, max(3, min(h - 2, 7)) + 1)), int(r.integers(3, max(3, min(w - 2, 7)) + 1))
            if not _place(r, grid, occ, _shapes.frame(r, fh, fw, 1), col()):
                return None
        elif feat == "rect":
            fh, fw = int(r.integers(3, max(3, min(h - 2, 7)) + 1)), int(r.integers(3, max(3, min(w - 2, 7)) + 1))
            if not _place(r, grid, occ, _shapes.rect(r, fh, fw), col()):
                return None
        elif feat == "sizes":
            n = size or int(r.integers(2, 6))
            c = col()
            for _ in range(2):
                shp = _shapes.polyomino(r, n) if n <= 5 else _shapes.bar(r, n, 1)
                if len(shp) != n or not _place(r, grid, occ, shp, c):
                    return None
        elif feat == "diag":
            c = col()
            a = _shapes.polyomino(r, int(r.integers(2, 5)))
            b = _shapes.polyomino(r, int(r.integers(2, 5)))
            ah = max(p[0] for p in a) + 1
            aw = max(p[1] for p in a) + 1
            # b sits so that its top-left cell touches a's bottom-right corner cell diagonally
            corner = max(a, key=lambda p: (p[0] + p[1], p[0]))
            b0 = min(b, key=lambda p: (p[0] + p[1], p[0]))
            off = (corner[0] + 1 - b0[0], corner[1] + 1 - b0[1])
            moved = {(y + off[0], x + off[1]) for y, x in b}
            if moved & set(a):
                return None
            both = set(a) | moved
            if min(y for y, _ in both) < 0 or min(x for _, x in both) < 0:
                return None
            # the two parts must not share an edge
            if any((y + dy, x + dx) in a for y, x in {(y + off[0], x + off[1]) for y, x in b}
                   for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                return None
            if not _place(r, grid, occ, frozenset(both), c):
                return None
    placed = len(must)
    for _ in range(4 * n_obj):
        if placed >= n_obj:
            break
        shape, _fam = _shapes.sample_shape(r, max_cells=max(2, area // 6), max_h=max(1, h - 1 - 2 * margin),
                                          max_w=max(1, w - 1 - 2 * margin), families=SCENE_FAMILIES)
        if len(shape) < 2:
            continue
        if _place(r, grid, occ, shape, col()):
            placed += 1
    if marker is not None:
        free = ~occ
        if pairs:
            ok = False
            for _ in range(30):
                y, x = int(r.integers(h)), int(r.integers(w))
                if r.random() < 0.5:
                    x2 = int(r.integers(w))
                    if abs(x2 - x) >= 2 and free[y, min(x, x2):max(x, x2) + 1].all():
                        grid[y, x] = grid[y, x2] = marker
                        ok = True
                        break
                else:
                    y2 = int(r.integers(h))
                    if abs(y2 - y) >= 2 and free[min(y, y2):max(y, y2) + 1, x].all():
                        grid[y, x] = grid[y2, x] = marker
                        ok = True
                        break
            if not ok:
                return None
        else:
            k = int(r.integers(1, 4))
            cells = np.argwhere(free)
            if len(cells) < k:
                return None
            for idx in r.choice(len(cells), size=k, replace=False):
                y, x = cells[idx]
                grid[y, x] = marker
    if len(np.unique(grid)) < 2:
        return None
    return grid


def scene_pictures(r, n: int, *, max_cells: int, must: tuple = (), marker: bool = False, pairs: bool = False,
                   solid: bool = True, ncolors: Optional[int] = None, margin: bool = False,
                   max_h: int = 30, max_w: int = 30, min_cells: int = 0) -> tuple[list, list]:
    """n drawn scenes sharing one palette (ground first) and, for "sizes", one object size."""
    order = [int(c) for c in r.permutation(10)]
    k = ncolors or int(r.integers(3, 7))
    palette = order[:k]
    mk = order[k] if marker else None
    size = int(r.integers(2, 6)) if "sizes" in must else None
    mg = int(r.integers(1, 3)) if margin else 0
    out, prov = [], []
    for _ in range(12 * n):
        if len(out) == n:
            break
        lo = 5 + 2 * mg
        if max_h < lo or max_w < lo:
            raise Reject("pictures")
        def draw(swap):
            ma, mb = (max_w, max_h) if swap else (max_h, max_w)
            hi = min(ma, max(lo, max_cells // lo))
            lo_a = min(hi, max(lo, -(-min_cells // mb))) if min_cells else lo
            x = int(r.integers(lo_a, hi + 1))
            hi_b = min(mb, max(lo, max_cells // x))
            lo_b = min(hi_b, max(lo, -(-min_cells // x))) if min_cells else lo
            return x, int(r.integers(lo_b, hi_b + 1))

        h, w = _two_sides(r, draw)
        g = draw_scene(r, h, w, palette, must=must, marker=mk, size=size, margin=mg, pairs=pairs)
        if g is None:
            continue
        if solid and mk is None and not _good(g, True):
            continue
        out.append(g)
        prov.append(("scene", h, w))
    if len(out) < n:
        raise Reject("pictures")
    return out, prov


def relabel(r, pics: list[np.ndarray], *, shuffle_p: float = 0.5) -> list[np.ndarray]:
    """Map every picture's colours, most frequent first, onto one random bundle palette."""
    order = r.permutation(10)
    out = []
    for a in pics:
        counts = np.bincount(a.ravel(), minlength=10)
        cols = [int(c) for c in sorted(np.nonzero(counts)[0], key=lambda c: (-counts[c], c))]
        if len(cols) > 2 and r.random() < shuffle_p:
            rest = cols[1:]
            rest = [rest[i] for i in r.permutation(len(rest))]
            cols = [cols[0]] + rest
        lut = np.arange(10, dtype=DT)
        for i, c in enumerate(cols):
            lut[c] = order[i]
        out.append(lut[a])
    return out


# ------------------------------------------------------------------------------------------------
# helpers for proposals
# ------------------------------------------------------------------------------------------------

def common(infos) -> list[int]:
    s = set(infos[0].colors)
    for i in infos[1:]:
        s &= set(i.colors)
    return sorted(s)


def union(infos) -> list[int]:
    s = set()
    for i in infos:
        s |= set(i.colors)
    return sorted(s)


def non_bg_common(infos) -> list[int]:
    bgs = {i.bg for i in infos}
    return [c for c in common(infos) if c not in bgs]


def pick(r, seq):
    seq = list(seq)
    if not seq:
        raise Reject("no_param")
    return seq[int(r.integers(len(seq)))]


# ------------------------------------------------------------------------------------------------
# constructed inputs
# ------------------------------------------------------------------------------------------------

def make_scaled(r, pics, draw):
    kh, kw = draw["k"]
    return [np.repeat(np.repeat(p, kh, axis=0), kw, axis=1) for p in pics]


def make_even(r, pics, draw):
    out = []
    for p in pics:
        h, w = p.shape
        if draw["which"] in ("top", "bottom"):
            h -= h % 2
        else:
            w -= w % 2
        if h < 2 or w < 1:
            raise Reject("pictures")
        out.append(p[:h, :w].copy())
    return out


def make_dupes(r, pics, draw):
    out = []
    for p in pics:
        a = p
        if draw["mode"] in ("rows", "both"):
            reps = r.integers(1, 3, size=a.shape[0])
            a = np.repeat(a, reps, axis=0)
        if draw["mode"] in ("cols", "both"):
            reps = r.integers(1, 3, size=a.shape[1])
            a = np.repeat(a, reps, axis=1)
        if not fits(*a.shape):
            raise Reject("size")
        out.append(np.ascontiguousarray(a))
    return out


def _bg_gaps(a, bg, axis):
    lines = a if axis == 0 else a.T
    empty = [bool(np.all(line == bg)) for line in lines]
    n = len(lines)
    return [q for q in range(n + 1) if q in (0, n) or empty[q - 1] or empty[q]]


def make_bg_lines(r, pics, draw):
    """Insert whole background rows / columns next to existing ones or at the edges (objects are
    never split)."""
    out = []
    for p in pics:
        info = Info(p)
        a = p
        for axis, key in ((0, "rows"), (1, "cols")):
            if draw["mode"] not in (key, "both"):
                continue
            for _ in range(int(r.integers(1, 3))):
                q = pick(r, _bg_gaps(a, info.bg, axis))
                a = np.insert(a, q, info.bg, axis=axis)
        if not fits(*a.shape):
            raise Reject("size")
        out.append(np.ascontiguousarray(a.astype(DT)))
    return out


def make_tiled(r, pics, draw):
    out = []
    for p in pics:
        for _ in range(10):
            th = int(r.integers(2, min(5, p.shape[0]) + 1))
            tw = int(r.integers(2, min(5, p.shape[1]) + 1))
            y0 = int(r.integers(0, p.shape[0] - th + 1))
            x0 = int(r.integers(0, p.shape[1] - tw + 1))
            tile = p[y0:y0 + th, x0:x0 + tw]
            if len(np.unique(tile)) >= 2:
                break
        else:
            raise Reject("pictures")
        H = int(r.integers(th + 1, min(30, 3 * th + 2) + 1))
        W = int(r.integers(tw + 1, min(30, 3 * tw + 2) + 1))
        big = np.tile(tile, (H // th + 1, W // tw + 1))[:H, :W]
        out.append(np.ascontiguousarray(big))
    return out


def make_noise(r, pics, draw):
    """Scatter isolated cells of the picture's own colours on its background."""
    out = []
    for p in pics:
        info = Info(p)
        bg = info.bg
        a = p.copy()
        cols = [c for c in info.colors if c != bg] or [c for c in range(10) if c != bg]
        free = np.argwhere(a == bg)
        r.shuffle(free)
        placed = 0
        want = int(r.integers(1, max(2, p.size // 25) + 1))
        for rr, cc in free:
            if placed >= want:
                break
            v = cols[int(r.integers(len(cols)))]
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    y, x = rr + dr, cc + dc
                    if (dr or dc) and 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x] != bg:
                        ok = False
            if ok:
                a[rr, cc] = v
                placed += 1
        if not placed:
            raise Reject("pictures")
        out.append(a)
    return out


# ------------------------------------------------------------------------------------------------
# generation specs
# ------------------------------------------------------------------------------------------------

@dataclass
class GenSpec:
    fam: str
    weight: float
    cands: Callable                        # (r, infos, draw) -> candidate params (tuples)
    draw: Callable = lambda r: {}
    limits: Callable = lambda d: (30, 30, 1.0)     # (max_h, max_w, output/input cell ratio)
    make: Optional[Callable] = None
    sources: Optional[list] = None
    min_side: int = 3
    solid: Optional[bool] = None
    scene_p: Optional[float] = None        # share of drawn scenes (default: 0.5 for background rules)
    must: tuple = ()                       # features every drawn scene needs
    marker: bool = False                   # drawn scenes carry lone marker cells of one extra colour
    pairs: bool = False                    # markers come as facing pairs
    ncolors: Optional[int] = None          # palette size of drawn scenes
    margin: bool = False                   # drawn scenes keep their edges clear

    @property
    def scene_share(self) -> float:
        if self.scene_p is not None:
            return self.scene_p
        return 0.5 if self.needs_solid else 0.0

    @property
    def needs_solid(self) -> bool:
        return FAMILIES[self.fam].bg if self.solid is None else self.solid


def choose_params(r, spec: GenSpec, infos, n_train: int, draw: dict, tries: int = 40):
    """Parameters under which the rule is defined on every input, changes every input, gives no
    single-colour output (unless by design) and not the same output on every train pair; returns
    (params, outputs)."""
    fam = FAMILIES[spec.fam]
    cands = list(spec.cands(r, infos, draw))
    if not cands:
        raise Reject("no_param")
    order = r.permutation(len(cands))[:tries]
    for k in order:
        p = tuple(cands[int(k)])
        outs = []
        for info in infos:
            o = fam.fn(info, *p)
            if o is None or not fits(*o.shape) or (o.shape == info.a.shape and np.array_equal(o, info.a)):
                break
            if fam.mono_ok is False and len(np.unique(o)) < 2:
                break
            outs.append(o)
        else:
            ys = outs[:n_train]
            if all(y.shape == ys[0].shape and np.array_equal(y, ys[0]) for y in ys[1:]):
                continue
            return p, outs
    raise Reject("no_param")


def _fixed(key, wrap=True):
    return lambda r, i, d: [(d[key],)] if wrap else [d[key]]


def _all(fam, pred=None):
    return lambda r, i, d: [p for p in FAMILIES[fam].instances if pred is None or pred(p)]


def prefer_new(r, cands, idx, infos, p_new=0.7):
    """Keep only candidates whose colour (params[idx]) no picture uses, most of the time."""
    used = set(union(infos))
    fresh = [c for c in cands if c[idx] not in used]
    if fresh and r.random() < p_new:
        return fresh
    return cands


def _painted(fam, idx, extra=None):
    def cands(r, i, d):
        cs = [p for p in FAMILIES[fam].instances if extra is None or extra(p)]
        return prefer_new(r, cs, idx, i)
    return cands


def _markers(limit):
    def rare(i):
        return [c for c in non_bg_common(i) if max(int(x.hist[c]) for x in i) <= limit]
    return rare


def _size_sets(infos, conn):
    out = []
    for i in infos:
        objs = i.objects(conn)
        out.append({o.size for o in objs} if objs else set())
    return out


def _remove_size_cands(r, infos, d):
    out = []
    for conn in (4, 8):
        sets = _size_sets(infos, conn)
        if not all(sets):
            continue
        both = set.intersection(*sets)
        out += [("eq", n, conn) for n in both if n <= 20]
        lo = max(min(s) for s in sets)
        out += [("lt", n, conn) for n in range(lo + 1, 21) if n >= 2]
        hi = min(max(s) for s in sets)
        out += [("gt", n, conn) for n in range(1, min(hi, 20))]
    return out


def _recolor_size_cands(r, infos, d):
    out = []
    for conn in (4, 8):
        sets = _size_sets(infos, conn)
        if not all(sets):
            continue
        for n in set.intersection(*sets):
            if n <= 20:
                out += [(n, x, conn) for x in range(10)]
    return prefer_new(r, out, 1, infos)


SMALL_SHIFTS = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]


def _k_choice(r):
    opts = [(2, 2)] * 4 + [(3, 3)] * 2 + [(4, 4), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    return opts[int(r.integers(len(opts)))]


def _tile_choice(r):
    opts = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2), (3, 3)]
    return opts[int(r.integers(len(opts)))]

GEN: list[GenSpec] = [
    # geometry
    GenSpec("rot", 3.0, _fixed("k"), draw=lambda r: {"k": int(r.integers(1, 4))}),
    GenSpec("flip", 2.5, _fixed("axis"), draw=lambda r: {"axis": pick(r, ("lr", "ud"))}),
    GenSpec("transpose", 2.0, _fixed("d"), draw=lambda r: {"d": pick(r, ("main", "anti"))}),
    GenSpec("scale_up", 2.0, _fixed("k", False), draw=lambda r: {"k": _k_choice(r)},
            limits=lambda d: (30 // d["k"][0], 30 // d["k"][1], d["k"][0] * d["k"][1]), min_side=2),
    GenSpec("scale_down", 1.5, _fixed("k", False), draw=lambda r: {"k": _k_choice(r)},
            limits=lambda d: (30 // d["k"][0], 30 // d["k"][1], d["k"][0] * d["k"][1]), make=make_scaled, min_side=2),
    GenSpec("tile", 1.5, _fixed("t", False), draw=lambda r: {"t": _tile_choice(r)},
            limits=lambda d: (30 // d["t"][0], 30 // d["t"][1], d["t"][0] * d["t"][1]), min_side=2),
    GenSpec("mirror_tile", 2.0, _fixed("m"), draw=lambda r: {"m": pick(r, ("right", "left", "down", "up", "four"))},
            limits=lambda d: {"right": (30, 15, 2), "left": (30, 15, 2), "down": (15, 30, 2), "up": (15, 30, 2),
                              "four": (15, 15, 4)}[d["m"]], min_side=2),
    GenSpec("half", 1.0, _fixed("which"), draw=lambda r: {"which": pick(r, ("top", "bottom", "left", "right"))},
            make=make_even),
    GenSpec("symmetrize", 1.0, _fixed("m"), draw=lambda r: {"m": pick(r, ("lr", "rl", "ud", "du"))}),
    # crops and extraction
    GenSpec("crop_content", 1.5, _all("crop_content"), scene_p=0.7, margin=True),
    GenSpec("crop_color", 1.5, lambda r, i, d: [(x,) for x in non_bg_common(i)] + [("least",)],
            limits=lambda d: (30, 30, 0.6), scene_p=0.6, must=("hole",)),
    GenSpec("crop_object", 1.0, _all("crop_object"), limits=lambda d: (30, 30, 0.5), scene_p=0.7, must=("diag",)),
    GenSpec("extract_row", 1.0, lambda r, i, d: [(x,) for x in [-1] + list(range(min(y.h for y in i)))],
            limits=lambda d: (30, 30, 0.3)),
    GenSpec("extract_col", 1.0, lambda r, i, d: [(x,) for x in [-1] + list(range(min(y.w for y in i)))],
            limits=lambda d: (30, 30, 0.3)),
    GenSpec("compress", 0.7, _fixed("mode"), draw=lambda r: {"mode": pick(r, ("rows", "cols", "both"))},
            make=make_dupes, limits=lambda d: (15, 15, 1.0)),
    GenSpec("drop_bg_lines", 0.7, _fixed("mode"), draw=lambda r: {"mode": pick(r, ("rows", "cols", "both"))},
            make=make_bg_lines, limits=lambda d: (26, 26, 1.0)),
    GenSpec("smallest_tile", 1.0, _all("smallest_tile"), make=make_tiled, limits=lambda d: (30, 30, 0.3)),
    GenSpec("count_bar", 0.6, lambda r, i, d: [(x, o) for x in common(i) for o in ("h", "v")],
            limits=lambda d: (30, 30, 0.3)),
    GenSpec("fractal", 0.4, _all("fractal"), limits=lambda d: (4, 4, 16), min_side=2),
    # colours
    GenSpec("recolor", 3.0, lambda r, i, d: prefer_new(r, [(a, b) for a in common(i) for b in range(10) if b != a], 1, i)),
    GenSpec("swap", 2.0, lambda r, i, d: [(a, b) for a in union(i) for b in union(i) if a < b]),
    GenSpec("keep_only", 1.0, lambda r, i, d: [(x,) for x in non_bg_common(i)]),
    GenSpec("keep_fill", 0.5, lambda r, i, d: prefer_new(r, [(x, z) for x in common(i) for z in range(10) if z != x], 1, i)),
    GenSpec("delete", 1.5, lambda r, i, d: [(x,) for x in non_bg_common(i)]),
    GenSpec("invert2", 0.5, _all("invert2"), scene_p=1.0, ncolors=2),
    GenSpec("mono", 1.0, _painted("mono", 0)),
    GenSpec("bg_to", 1.0, _painted("bg_to", 0)),
    GenSpec("recolor_least", 0.5, _painted("recolor_least", 0)),
    # motion
    GenSpec("roll", 1.0, lambda r, i, d: SMALL_SHIFTS),
    GenSpec("shift", 1.5, lambda r, i, d: SMALL_SHIFTS),
    GenSpec("gravity", 2.0, _all("gravity")),
    GenSpec("sort_cells", 0.6, _all("sort_cells")),
    GenSpec("sort_lines", 0.6, _all("sort_lines")),
    # painting
    GenSpec("outline", 2.0, _painted("outline", 0, lambda p: p[2] == 1 or (p[2] == 2 and p[0] % 3 == 0))),
    GenSpec("fill_holes", 1.5, _painted("fill_holes", 0), sources=HOLE_SOURCES, scene_p=0.7, must=("hole",)),
    GenSpec("fill_holes_own", 0.7, _all("fill_holes_own"), sources=HOLE_SOURCES, scene_p=0.7, must=("hole",)),
    GenSpec("hollow", 1.0, _all("hollow"), sources=RECT_SOURCES, scene_p=0.7, must=("rect",)),
    GenSpec("fill_interior", 1.0, _painted("fill_interior", 0), sources=RECT_SOURCES, scene_p=0.7, must=("rect",)),
    GenSpec("lines", 1.5, lambda r, i, d: [(x, m) for x in _markers(4)(i) for m in ("row", "col", "cross")],
            scene_p=0.8, marker=True),
    GenSpec("rays", 1.0, lambda r, i, d: [(x, m) for x in _markers(8)(i) for m in ("up", "down", "left", "right")],
            scene_p=0.8, marker=True, solid=False),
    GenSpec("diagonals", 0.7, lambda r, i, d: [(x,) for x in _markers(3)(i)], scene_p=0.8, marker=True),
    GenSpec("connect", 0.7, lambda r, i, d: [(x, m) for x in _markers(12)(i) for m in ("row", "col", "both")],
            scene_p=0.9, marker=True, pairs=True, solid=False),
    GenSpec("border", 0.7, _painted("border", 0)),
    GenSpec("mirror_overlay", 0.7, _all("mirror_overlay")),
    # objects
    GenSpec("remove_isolated", 1.0, _all("remove_isolated"), make=make_noise),
    GenSpec("remove_size", 1.5, _remove_size_cands, scene_p=0.7, must=("sizes",)),
    GenSpec("keep_extreme", 1.0, _all("keep_extreme"), scene_p=0.7, must=("diag",)),
    GenSpec("recolor_size", 1.5, _recolor_size_cands, scene_p=0.7, must=("sizes",)),
    GenSpec("recolor_extreme", 1.5, _painted("recolor_extreme", 2), scene_p=0.7, must=("diag",)),
    GenSpec("solidify", 0.5, _all("solidify"), scene_p=0.7, must=("diag",)),
    GenSpec("hollow_rects", 0.7, _all("hollow_rects", lambda p: p[0] <= 2), sources=RECT_SOURCES, scene_p=0.7,
            must=("rect",)),
]
GEN_BY_FAM = {g.fam: g for g in GEN}

# two-step compositions: first steps and second steps that read well together
COMP_FIRST = ["rot", "flip", "transpose", "crop_content", "crop_color", "gravity", "shift", "recolor", "delete",
              "mirror_tile", "outline", "fill_holes", "remove_size", "lines", "scale_up"]
COMP_SECOND = ["rot", "flip", "transpose", "recolor", "swap", "outline", "scale_up", "mirror_tile", "border",
               "fill_holes", "mono", "gravity", "crop_content", "tile", "delete"]


def choose_spec(r, pool=None) -> GenSpec:
    pool = pool or GEN
    w = np.array([g.weight for g in pool], dtype=float)
    return pool[int(r.choice(len(pool), p=w / w.sum()))]


# ------------------------------------------------------------------------------------------------
# siblings: the other parameter readings of a rule
# ------------------------------------------------------------------------------------------------

D4 = [("rot", (1,)), ("rot", (2,)), ("rot", (3,)), ("flip", ("lr",)), ("flip", ("ud",)),
      ("transpose", ("main",)), ("transpose", ("anti",))]


def step_siblings(step) -> list:
    """Instances that differ from ``step`` in exactly one parameter (and, for the D4 group, every
    other rotation / reflection)."""
    fam, p = step
    out = []
    if (fam, p) in D4:
        out += [s for s in D4 if s != (fam, p)]
    for q in FAMILIES[fam].instances:
        if q == p or len(q) != len(p):
            continue
        if sum(1 for a, b in zip(p, q) if a != b) == 1:
            out.append((fam, q))
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# alternatives every train pair should contradict when a rule has few siblings of its own
GENERIC = [Rule((s,)) for s in D4] + [Rule.single("gravity", d) for d in ("down", "up", "left", "right")] + \
    [Rule.single("crop_content"), Rule.single("compress", "both")]


def siblings(rule: Rule) -> list[Rule]:
    out = []
    for i, step in enumerate(rule.steps):
        for s in step_siblings(step):
            steps = list(rule.steps)
            steps[i] = s
            out.append(Rule(tuple(steps)))
    return out
