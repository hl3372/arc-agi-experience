"""Periodic background fields for the counting tasks 68a / 68b.

A *field* is a whole-canvas colouring that is a pure function of the canvas
coordinates -- bands, diagonal bands, tiles, grid lines, concentric rings or a
panel split -- never a local blob or patch.  Objects are stamped opaquely on
top of it, so the object layer of a finished grid is exactly ``grid != field``.

The module has three jobs.

* Render a field from ``(family, params, h, w)`` and decide whether those
  parameters describe an admissible background at that size (``family_ok``):
  every field component is at least two cells, the pattern repeats at least
  twice, and no class owns less than ``SHARE_LO`` or more than ``SHARE_HI`` of
  the canvas.
* Recover the field from a finished grid the way a solver must.  ``fit``
  derives the parameters from the visible cells (run lengths, boundary gaps,
  ring depths, panel cuts); ``brute`` sweeps the whole parameter space instead
  and is the assumption-free cross-check.  Both return one reading per
  *distinct field*, so two families describing the same colouring are one
  reading and two different colourings are an ambiguity the caller refuses.
* Measure how much of the field survives the objects (``visibility``), which is
  what keeps a pattern readable as a background instead of blending into the
  figures.  ``visibility_ok`` writes the shipped thresholds out family by
  family, and ``V2_DOC`` says why they differ between families.

The reader's admissibility rules are the published contract of the task
family: a background covers at least ``MIN_FIELD_SHARE`` of the canvas and no
colour covers more than ``MAX_COLOR_SHARE`` of a patterned grid.  That single
threshold is what separates the two readings -- a patterned grid can never pass
the plain-ground plurality certificate (which needs a colour over half the
canvas), and a plain grid can never pass this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

FAMILIES = ("bands", "diagonal_bands", "tiles", "grid_lines", "concentric", "split")
MIN_SIDE = {"bands": 10, "diagonal_bands": 10, "split": 10,
            "tiles": 12, "grid_lines": 12, "concentric": 12}

PERIOD_LO, PERIOD_HI = 4, 8          # bands / diagonal bands
TILE_LO, TILE_HI = 2, 5              # tile block side
LINE_LO, LINE_HI = 4, 7              # grid-line period
PANEL_MIN = 4                        # narrowest split panel

V2_REGIONS_LO, V2_REGIONS_HI = 1, 2  # V2: untouched regions a field class keeps
V2_REGIONS_CUT = 4                   # ... two of them once the class owns this many
V2_TILES_LO, V2_TILES_HI = 4, 6      # untouched tiles per colour
V2_TILES_CUT = 8
V2_LINES_TOTAL = 3                   # clean lattice lines, the two axes together

V2_DOC = """Whole regions of the field that no object touches.

Two of them per colour is the standard, but a class that owns only two or
three regions cannot spare two of them: AB2 makes every body straddle a colour
change, so each body already touches at least one region of each colour, and a
three-stripe field with two untouched stripes per colour would admit bodies in
one stripe only.  The threshold therefore rises with the number of regions the
class owns, and the families whose regions are not interchangeable state their
own rule: a lattice is read from its complete lines (one clean line on each
axis, three in all) and the blocks between them, a split from the clean
full-span line inside every panel, and concentric rings are nested -- a body at
any depth crosses several rings at once, and a canvas of twelve holds three
ring bands per colour at t = 1 and one or two at t = 2, so two untouched bands
of every colour is structurally unreachable there and the field keeps one band
of each colour and two in all."""

SHARE_LO, SHARE_HI = 0.24, 0.58      # every field class, as a share of the canvas
MAX_COLOR_SHARE = 0.48               # V6: no colour on a patterned grid
MIN_FIELD_SHARE = 0.65               # the field colours together, on a finished grid
MIN_UNCOVERED = 0.60                 # V1: uncovered share of every field class
MAX_COVERED = 0.35                   # reader-side slack over the generator's V4
MAX_OBJECT_EXTENT = 0.60             # an object is a bounded figure, not a band


# ---------------------------------------------------------------------------
# Rendering and small array helpers
# ---------------------------------------------------------------------------

def render(family: str, params: dict, h: int, w: int) -> np.ndarray:
    """Class indices (0 .. k-1) of the field; class 0 is the ground."""
    rr, cc = np.indices((h, w))
    if family == "bands":
        axis = rr if params["axis"] == 0 else cc
        return (((axis + params["phase"]) % params["period"]) < params["thickness"]).astype(np.int8)
    if family == "diagonal_bands":
        diag = (rr + cc) if params["sign"] > 0 else (rr - cc + w)
        return (((diag + params["phase"]) % params["period"]) < params["thickness"]).astype(np.int8)
    if family == "tiles":
        bh, bw, n = params["bh"], params["bw"], params["colors"]
        # Floor division, so the lattice continues unbroken above and left of
        # the phase origin.
        a = (rr - params["phase_r"]) // bh
        b = (cc - params["phase_c"]) // bw
        return ((a + b + params["offset"]) % n).astype(np.int8)
    if family == "grid_lines":
        k = params["k"]
        lines = (((rr + params["phase_r"]) % k) < params["tr"]) | (((cc + params["phase_c"]) % k) < params["tc"])
        return lines.astype(np.int8)
    if family == "concentric":
        depth = np.minimum(np.minimum(rr, cc), np.minimum(h - 1 - rr, w - 1 - cc))
        return (((depth // params["t"]) % 2) == params["parity"]).astype(np.int8)
    if family == "split":
        coord = rr if params["axis"] == 0 else cc
        labels = np.array(params["labels"], np.int8)
        return labels[np.searchsorted(np.array(params["cuts"]), coord, side="right")]
    raise ValueError(f"Unknown pattern family {family}")


def label4(arr: np.ndarray) -> np.ndarray:
    """4-connected components of equal value, by label propagation."""
    a = np.asarray(arr)
    h, w = a.shape
    lab = np.arange(h * w).reshape(h, w)
    same_v = a[1:, :] == a[:-1, :]
    same_h = a[:, 1:] == a[:, :-1]
    while True:
        new = lab.copy()
        np.minimum(new[1:, :], np.where(same_v, lab[:-1, :], lab[1:, :]), out=new[1:, :])
        np.minimum(new[:-1, :], np.where(same_v, lab[1:, :], lab[:-1, :]), out=new[:-1, :])
        np.minimum(new[:, 1:], np.where(same_h, new[:, :-1], new[:, 1:]), out=new[:, 1:])
        np.minimum(new[:, :-1], np.where(same_h, new[:, 1:], new[:, :-1]), out=new[:, :-1])
        if np.array_equal(new, lab):
            break
        lab = new
    return np.unique(lab, return_inverse=True)[1].reshape(h, w)


def _region_class(layers: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Class of every region id."""
    first = np.unique(lab, return_index=True)[1]
    return layers.ravel()[first]


def _full_lines(layers: np.ndarray, cls: int) -> tuple[list, list]:
    rows = [int(r) for r in np.flatnonzero((layers == cls).all(axis=1))]
    cols = [int(c) for c in np.flatnonzero((layers == cls).all(axis=0))]
    return rows, cols


def _runs(values: list) -> list[tuple]:
    """Maximal runs of equal values as (value, start, length)."""
    out = []
    for i, v in enumerate(values):
        if out and out[-1][0] == v:
            out[-1][2] += 1
        else:
            out.append([v, i, 1])
    return [tuple(r) for r in out]


def _bands_of(positions) -> list[tuple]:
    """Maximal runs of consecutive integers as (first, last)."""
    out = []
    for p in sorted(positions):
        if out and out[-1][1] + 1 == p:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [tuple(r) for r in out]


def _modal(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda v: (counts[v], -v))


def period_axes(family: str, params: dict, h: int, w: int) -> list[tuple]:
    """(coordinate array, period) for every axis along which the field repeats."""
    rr, cc = np.indices((h, w))
    if family == "bands":
        return [(rr if params["axis"] == 0 else cc, params["period"])]
    if family == "diagonal_bands":
        return [((rr + cc) if params["sign"] > 0 else (rr - cc + w), params["period"])]
    if family == "tiles":
        return [(rr, params["bh"]), (cc, params["bw"])]
    if family == "grid_lines":
        return [(rr, params["k"]), (cc, params["k"])]
    if family == "concentric":
        depth = np.minimum(np.minimum(rr, cc), np.minimum(h - 1 - rr, w - 1 - cc))
        return [(depth, 2 * params["t"])]
    return []


# ---------------------------------------------------------------------------
# Admissibility of a field on its own (before any object is stamped)
# ---------------------------------------------------------------------------

def class_shares(layers: np.ndarray) -> np.ndarray:
    """Share of the canvas owned by every field class."""
    return np.bincount(layers.ravel(), minlength=int(layers.max()) + 1) / layers.size


def params_in_range(family: str, params: dict, h: int, w: int) -> bool:
    """The declared parameter ranges of the family, before any measurement."""
    if family not in FAMILIES or min(h, w) < MIN_SIDE[family]:
        return False
    if family in ("bands", "diagonal_bands"):
        p, t, ph = params.get("period"), params.get("thickness"), params.get("phase", -1)
        if not (isinstance(p, int) and PERIOD_LO <= p <= PERIOD_HI):
            return False
        if not (isinstance(t, int) and 2 <= t <= p - 2 and 0 <= ph < p):
            return False
        if family == "bands":
            if params.get("axis") not in (0, 1):
                return False
            return (h if params["axis"] == 0 else w) >= 2 * p
        return params.get("sign") in (1, -1) and min(h, w) >= 2 * p
    if family == "tiles":
        bh, bw, n = params.get("bh"), params.get("bw"), params.get("colors")
        if not (isinstance(bh, int) and isinstance(bw, int)):
            return False
        if not (TILE_LO <= bh <= TILE_HI and TILE_LO <= bw <= TILE_HI and n in (2, 3)):
            return False
        if not (0 <= params.get("phase_r", -1) < bh and 0 <= params.get("phase_c", -1) < bw):
            return False
        if not 0 <= params.get("offset", -1) < n:
            return False
        return h >= 3 * bh and w >= 3 * bw
    if family == "grid_lines":
        k = params.get("k")
        if not (isinstance(k, int) and LINE_LO <= k <= LINE_HI):
            return False
        if not (1 <= params.get("tr", 0) <= 2 and 1 <= params.get("tc", 0) <= 2):
            return False
        if not (0 <= params.get("phase_r", -1) < k and 0 <= params.get("phase_c", -1) < k):
            return False
        return h >= 2 * k and w >= 2 * k
    if family == "concentric":
        if params.get("t") not in (1, 2) or params.get("parity") not in (0, 1):
            return False
        return (min(h, w) + 1) // 2 >= 3 * params["t"]
    if params.get("axis") not in (0, 1):                      # split
        return False
    cuts, labels = list(params.get("cuts", [])), list(params.get("labels", []))
    side = h if params["axis"] == 0 else w
    if not (1 <= len(cuts) <= 3) or len(labels) != len(cuts) + 1:
        return False
    if cuts != sorted(set(cuts)) or cuts[0] < PANEL_MIN or cuts[-1] > side - PANEL_MIN:
        return False
    if int(np.diff([0] + cuts + [side]).min()) < PANEL_MIN:
        return False
    if any(labels[i] == labels[i + 1] for i in range(len(labels) - 1)):
        return False
    return sorted(set(labels)) == list(range(max(labels) + 1))


def layers_ok(layers: np.ndarray, family: str, params: dict, lab=None) -> bool:
    """Measured contract of a background field, recomputed from the rendering."""
    k = int(layers.max()) + 1
    if not 2 <= k <= 3:
        return False
    counts = np.bincount(layers.ravel(), minlength=k)
    if counts.min() == 0:
        return False
    shares = counts / layers.size
    if shares.min() < SHARE_LO or shares.max() > SHARE_HI:
        return False
    if lab is None:
        lab = label4(layers)
    sizes = np.bincount(lab.ravel())
    if sizes.min() < 2:
        return False                               # never a one-cell field component
    per = np.bincount(_region_class(layers, lab), minlength=k)
    if family in ("bands", "diagonal_bands"):
        return int(per.min()) >= 2
    if family == "tiles":
        return int(per.min()) >= 6
    if family == "grid_lines":
        if per[1] != 1 or per[0] < 4:
            return False
        rows, cols = _full_lines(layers, 1)
        return len(_bands_of(rows)) >= 2 and len(_bands_of(cols)) >= 2
    if family == "concentric":
        h, w = layers.shape
        rr, cc = np.indices((h, w))
        depth = np.minimum(np.minimum(rr, cc), np.minimum(h - 1 - rr, w - 1 - cc))
        return int(depth.max()) // params["t"] + 1 >= 3
    return int(per.sum()) == len(params["cuts"]) + 1           # split: one region per panel


def family_ok(family: str, params: dict, h: int, w: int) -> bool:
    if not params_in_range(family, params, h, w):
        return False
    return layers_ok(render(family, params, h, w), family, params)


# ---------------------------------------------------------------------------
# Sampling (generator side)
# ---------------------------------------------------------------------------

def admissible_families(h: int, w: int, *, exclude=()) -> list[str]:
    return [f for f in FAMILIES if f not in exclude and min(h, w) >= MIN_SIDE[f]]


def standalone_families(h: int, w: int, *, exclude=()) -> list[str]:
    """Families that can colour a canvas with nothing drawn on top of them.

    Two classes always leave one of them more than half the canvas, which is
    the plain reading's own certificate, so a field that has to hold a grid on
    its own -- no objects, no marker -- needs three classes: a tile lattice
    cycling three colours, or a split of three or more panels.
    """
    out = []
    for family in admissible_families(h, w, exclude=exclude):
        if family == "tiles":
            out.append(family)
        elif family == "split" and max(h, w) >= 3 * PANEL_MIN:
            out.append(family)
    return out


def sample(rng, family: str, h: int, w: int, *, tries: int = 24):
    """Admissible parameters of ``family`` on an h x w canvas, or None."""
    for _ in range(tries):
        params = _sample_raw(rng, family, h, w)
        if params is not None and family_ok(family, params, h, w):
            return params
    return None


def _sample_raw(rng, family: str, h: int, w: int):
    def ri(lo, hi):
        return int(rng.integers(lo, hi + 1)) if hi > lo else int(lo)

    if family == "bands":
        axis = ri(0, 1)
        hi = min(PERIOD_HI, (h if axis == 0 else w) // 2)
        if hi < PERIOD_LO:
            return None
        p = ri(PERIOD_LO, hi)
        return {"axis": axis, "period": p, "thickness": ri(2, p - 2), "phase": ri(0, p - 1)}
    if family == "diagonal_bands":
        hi = min(PERIOD_HI, min(h, w) // 2)
        if hi < PERIOD_LO:
            return None
        p = ri(PERIOD_LO, hi)
        return {"sign": 1 if rng.random() < .5 else -1, "period": p,
                "thickness": ri(2, p - 2), "phase": ri(0, p - 1)}
    if family == "tiles":
        bh, bw = ri(TILE_LO, min(TILE_HI, h // 3)), ri(TILE_LO, min(TILE_HI, w // 3))
        n = 2 if rng.random() < .65 else 3
        return {"bh": bh, "bw": bw, "colors": n, "phase_r": ri(0, bh - 1),
                "phase_c": ri(0, bw - 1), "offset": ri(0, n - 1)}
    if family == "grid_lines":
        hi = min(LINE_HI, min(h, w) // 2)
        if hi < LINE_LO:
            return None
        k = ri(LINE_LO, hi)
        return {"k": k, "tr": ri(1, 2), "tc": ri(1, 2),
                "phase_r": ri(0, k - 1), "phase_c": ri(0, k - 1)}
    if family == "concentric":
        # Thicker rings read as a pattern rather than a fine texture once the
        # canvas is big enough to hold three of them.
        return {"t": 1 if min(h, w) < 14 else (2 if rng.random() < .7 else 1),
                "parity": ri(0, 1)}
    axis = ri(0, 1)                                            # split
    side = h if axis == 0 else w
    panels = ri(2, min(4, side // PANEL_MIN))
    if panels < 2:
        return None
    pool = np.arange(PANEL_MIN, side - PANEL_MIN + 1)
    if len(pool) < panels - 1:
        return None
    cuts = sorted(int(x) for x in rng.choice(pool, size=panels - 1, replace=False))
    if int(np.diff([0] + cuts + [side]).min()) < PANEL_MIN:
        return None
    ncls = 2 if (panels == 2 or rng.random() < .6) else 3
    labels = [0]
    for _ in range(1, panels):
        labels.append(int(rng.choice([c for c in range(ncls) if c != labels[-1]])))
    if sorted(set(labels)) != list(range(ncls)):
        return None
    return {"axis": axis, "cuts": cuts, "labels": labels}


# ---------------------------------------------------------------------------
# Visibility (after the objects are stamped)
# ---------------------------------------------------------------------------

def outline_components(grid) -> np.ndarray:
    """Cells of one-cell-thick rectangular outlines drawn on the grid.

    A marker rectangle is an annotation, not a cover: it crosses the field in a
    hairline that leaves the repetition plainly readable on both sides, so the
    region and period measures below look past it.  It still counts in V1 and
    in the coverage cap.
    """
    g = np.asarray(grid)
    lab = label4(g)
    out = np.zeros(g.shape, bool)
    sizes = np.bincount(lab.ravel())
    for lid in range(int(lab.max()) + 1):
        if sizes[lid] < 16:
            continue
        ys, xs = np.nonzero(lab == lid)
        r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if min(bh, bw) < 5 or sizes[lid] != 2 * (bh + bw) - 4:
            continue
        local = np.zeros((bh, bw), bool)
        local[ys - r0, xs - c0] = True
        ring = np.ones((bh, bw), bool)
        ring[1:-1, 1:-1] = False
        if np.array_equal(local, ring):
            out[ys, xs] = True
    return out


def visibility(family: str, params: dict, layers: np.ndarray, covered: np.ndarray, lab=None,
               annotations=None) -> dict:
    """How much of the field survives ``covered`` (the object layer)."""
    k = int(layers.max()) + 1
    h, w = layers.shape
    uncovered = ~covered
    solid = covered if annotations is None else (covered & ~annotations)
    cls_cells = np.bincount(layers.ravel(), minlength=k)
    cls_free = np.bincount(layers.ravel(), weights=uncovered.ravel().astype(float), minlength=k)
    vis = [float(f / max(1, n)) for f, n in zip(cls_free, cls_cells)]
    if lab is None:
        lab = label4(layers)
    nreg = int(lab.max()) + 1
    reg_cls = _region_class(layers, lab)
    reg_cov = np.bincount(lab.ravel(), weights=solid.ravel().astype(float), minlength=nreg)
    clean = reg_cov == 0
    untouched = np.bincount(reg_cls[clean], minlength=k).tolist()
    rr, cc = np.indices((h, w))
    top = np.full(nreg, h)
    left = np.full(nreg, w)
    np.minimum.at(top, lab.ravel(), rr.ravel())
    np.minimum.at(left, lab.ravel(), cc.ravel())
    tile_rows = [len({int(top[i]) for i in range(nreg) if clean[i] and reg_cls[i] == c}) for c in range(k)]
    tile_cols = [len({int(left[i]) for i in range(nreg) if clean[i] and reg_cls[i] == c}) for c in range(k)]
    periods = []
    for coord, p in period_axes(family, params, h, w):
        lo, hi = int(coord.min()), int(coord.max())
        seen_by = {int(v) for v in np.unique(coord[~solid])}
        full, start = 0, lo
        while start + p - 1 <= hi:
            if all((start + d) in seen_by for d in range(p)):
                full += 1
                start += p
            else:
                start += 1
        periods.append(full)
    rows1, cols1 = _full_lines(layers, 1)
    clean_rows = [r for r in rows1 if not solid[r, :].any()]
    clean_cols = [c for c in cols1 if not solid[:, c].any()]
    panel_lines = []
    if family == "split":
        axis = params["axis"]
        bounds = [0] + list(params["cuts"]) + [h if axis == 0 else w]
        for i in range(len(bounds) - 1):
            span = range(bounds[i], bounds[i + 1])
            panel_lines.append(sum(1 for j in span
                                   if not (solid[j, :] if axis == 0 else solid[:, j]).any()))
    return {"visible_share": vis, "untouched_regions": untouched,
            "region_counts": np.bincount(reg_cls, minlength=k).tolist(),
            "periods_visible": periods,
            "clean_line_rows": len(_bands_of(clean_rows)), "clean_line_cols": len(_bands_of(clean_cols)),
            "untouched_tile_rows": tile_rows, "untouched_tile_cols": tile_cols,
            "panel_clean_lines": panel_lines, "covered_share": float(covered.mean())}


def region_need(total: int) -> int:
    """V2: whole untouched regions a field class of ``total`` regions must keep."""
    return V2_REGIONS_HI if total >= V2_REGIONS_CUT else V2_REGIONS_LO


def visibility_ok(family: str, metrics: dict) -> bool:
    """V1 - V3: the field still reads as the background of the picture.

    V2 as shipped, region class by region class (``V2_DOC`` explains why the
    numbers differ between families):

    * bands, diagonal bands: two untouched stripes of every colour once that
      colour owns four or more, one otherwise;
    * tiles: six untouched tiles of every colour once it owns eight or more,
      four otherwise, spread over at least two tile rows and two tile columns;
    * grid lines: two untouched blocks of ground, a clean full line on each
      axis and three clean lines in all;
    * concentric rings: an untouched ring band of every colour and two in all;
    * split: a clean full-span line inside every panel.
    """
    if min(metrics["visible_share"]) < MIN_UNCOVERED:
        return False
    if metrics["covered_share"] > MAX_COVERED:
        return False
    untouched, total = metrics["untouched_regions"], metrics["region_counts"]
    if family in ("bands", "diagonal_bands"):
        if any(u < region_need(t) for u, t in zip(untouched, total)):
            return False
    elif family == "tiles":
        if any(u < (V2_TILES_HI if t >= V2_TILES_CUT else V2_TILES_LO)
               for u, t in zip(untouched, total)):
            return False
        if min(metrics["untouched_tile_rows"]) < 2 or min(metrics["untouched_tile_cols"]) < 2:
            return False
    elif family == "grid_lines":
        rows, cols = metrics["clean_line_rows"], metrics["clean_line_cols"]
        if untouched[0] < V2_REGIONS_HI or min(rows, cols) < 1:
            return False
        if rows + cols < V2_LINES_TOTAL:
            return False
    elif family == "concentric":
        # Rings are nested, so one body crosses several of them at once: with
        # two or three ring bands per colour on a small canvas, two untouched
        # bands of EVERY colour is unreachable (see ``V2_DOC``).
        if min(untouched) < 1 or sum(untouched) < 2:
            return False
    elif min(metrics["panel_clean_lines"] or [0]) < 1:         # split
        return False
    if family != "split" and min(metrics["periods_visible"] or [2]) < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# Recovering the field from a finished grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    family: str
    frozen: tuple           # params, hashable
    colors: tuple           # colour per class
    field: bytes            # field.tobytes(), for de-duplication
    shape: tuple

    @property
    def params(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.frozen}

    def array(self) -> np.ndarray:
        return np.frombuffer(self.field, dtype=np.int16).reshape(self.shape).astype(int)


def _freeze(params: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in params.items()))


def candidate_subsets(grid) -> list[tuple]:
    """Colour sets that could carry a field: each colour common enough to be a
    class that survived the objects, together covering most of the canvas."""
    g = np.asarray(grid)
    vals, cnt = np.unique(g, return_counts=True)
    floor = SHARE_LO * MIN_UNCOVERED * g.size
    pool = [(int(v), int(c)) for v, c in zip(vals, cnt) if c >= floor]
    out = []
    for size in (2, 3):
        for combo in combinations(pool, size):
            if sum(c for _, c in combo) >= MIN_FIELD_SHARE * g.size:
                out.append(tuple(sorted(v for v, _ in combo)))
    return out


def _accept(g: np.ndarray, family: str, params: dict, colors: tuple, outlines=None):
    """One candidate reading against the whole contract; None when it fails."""
    h, w = g.shape
    if not params_in_range(family, params, h, w):
        return None
    layers = render(family, params, h, w)
    if int(layers.max()) + 1 != len(colors):
        return None
    lab = label4(layers)
    if not layers_ok(layers, family, params, lab=lab):
        return None
    in_field = np.isin(g, colors)
    assign = []
    for cls in range(len(colors)):
        vals = np.unique(g[(layers == cls) & in_field])
        if len(vals) != 1:
            return None
        assign.append(int(vals[0]))
    if sorted(assign) != sorted(colors):
        return None
    field = np.array(assign, int)[layers]
    covered = g != field
    annot = None if outlines is None else (outlines & covered)
    metrics = visibility(family, params, layers, covered, lab=lab, annotations=annot)
    if not visibility_ok(family, metrics):
        return None
    if int(np.bincount(g.ravel(), minlength=10).max()) > MAX_COLOR_SHARE * g.size:
        return None
    if not objects_are_bounded(g, covered if annot is None else (covered & ~annot)):
        return None
    return Reading(family, _freeze(params), tuple(assign),
                   field.astype(np.int16).tobytes(), (h, w))


def objects_are_bounded(g: np.ndarray, solid: np.ndarray) -> bool:
    """No object stretches across the canvas.

    A colour that runs the whole way across is a panel of the background, not a
    figure drawn on it; without this a three-panel split would also read as a
    two-panel split with a very wide object lying over the middle.
    """
    if not solid.any():
        return True
    h, w = g.shape
    lab = label4(np.where(solid, g, -1))
    rr, cc = np.indices((h, w))
    n = int(lab.max()) + 1
    top, left = np.full(n, h), np.full(n, w)
    bottom, right = np.full(n, -1), np.full(n, -1)
    np.minimum.at(top, lab.ravel(), rr.ravel())
    np.minimum.at(left, lab.ravel(), cc.ravel())
    np.maximum.at(bottom, lab.ravel(), rr.ravel())
    np.maximum.at(right, lab.ravel(), cc.ravel())
    for lid in range(n):
        if not solid[lab == lid].any():
            continue
        if bottom[lid] - top[lid] + 1 > MAX_OBJECT_EXTENT * h:
            return False
        if right[lid] - left[lid] + 1 > MAX_OBJECT_EXTENT * w:
            return False
    return True


def _dedupe(readings: list) -> list:
    seen, out = set(), []
    for r in readings:
        if r.field in seen:
            continue
        seen.add(r.field)
        out.append(r)
    return out


def fit(grid, *, families=None) -> list[Reading]:
    """Readings recovered by DERIVING the parameters from the visible cells."""
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("A background field lives on a two-dimensional canvas")
    h, w = g.shape
    out = []
    subsets = candidate_subsets(g)
    outlines = outline_components(g) if subsets else None
    for colors in subsets:
        mask = np.isin(g, colors)
        for family in (families or FAMILIES):
            if min(h, w) < MIN_SIDE[family]:
                continue
            for params in _derive(g, mask, family, colors):
                reading = _accept(g, family, params, colors, outlines)
                if reading is not None:
                    out.append(reading)
    return _dedupe(out)


def brute(grid, *, families=None) -> list[Reading]:
    """Readings found by SWEEPING the parameter space -- the assumption-free
    cross-check on ``fit``."""
    g = np.asarray(grid)
    h, w = g.shape
    out = []
    subsets = candidate_subsets(g)
    if not subsets:
        return out
    outlines = outline_components(g)
    for family in (families or FAMILIES):
        if min(h, w) < MIN_SIDE[family]:
            continue
        for params in all_params(family, h, w):
            for colors in subsets:
                reading = _accept(g, family, params, colors, outlines)
                if reading is not None:
                    out.append(reading)
    return _dedupe(out)


def all_params(family: str, h: int, w: int):
    """Every parameter set of ``family`` whose declared ranges fit the canvas."""
    if family == "bands":
        for axis in (0, 1):
            for p in range(PERIOD_LO, PERIOD_HI + 1):
                for t in range(2, p - 1):
                    for ph in range(p):
                        cand = {"axis": axis, "period": p, "thickness": t, "phase": ph}
                        if params_in_range(family, cand, h, w):
                            yield cand
    elif family == "diagonal_bands":
        for sign in (1, -1):
            for p in range(PERIOD_LO, PERIOD_HI + 1):
                for t in range(2, p - 1):
                    for ph in range(p):
                        cand = {"sign": sign, "period": p, "thickness": t, "phase": ph}
                        if params_in_range(family, cand, h, w):
                            yield cand
    elif family == "tiles":
        for bh in range(TILE_LO, TILE_HI + 1):
            for bw in range(TILE_LO, TILE_HI + 1):
                for n in (2, 3):
                    for pr in range(bh):
                        for pc in range(bw):
                            for off in range(n):
                                cand = {"bh": bh, "bw": bw, "colors": n, "phase_r": pr,
                                        "phase_c": pc, "offset": off}
                                if params_in_range(family, cand, h, w):
                                    yield cand
    elif family == "grid_lines":
        for k in range(LINE_LO, LINE_HI + 1):
            for tr in (1, 2):
                for tc in (1, 2):
                    for pr in range(k):
                        for pc in range(k):
                            cand = {"k": k, "tr": tr, "tc": tc, "phase_r": pr, "phase_c": pc}
                            if params_in_range(family, cand, h, w):
                                yield cand
    elif family == "concentric":
        for t in (1, 2):
            for parity in (0, 1):
                cand = {"t": t, "parity": parity}
                if params_in_range(family, cand, h, w):
                    yield cand
    else:
        for axis in (0, 1):
            side = h if axis == 0 else w
            positions = range(PANEL_MIN, side - PANEL_MIN + 1)
            for ncuts in (1, 2, 3):
                for cuts in combinations(positions, ncuts):
                    if int(np.diff([0] + list(cuts) + [side]).min()) < PANEL_MIN:
                        continue
                    for labels in _label_patterns(ncuts + 1):
                        cand = {"axis": axis, "cuts": list(cuts), "labels": list(labels)}
                        if params_in_range(family, cand, h, w):
                            yield cand


def _label_patterns(n: int):
    """Panel class patterns: adjacent panels differ, classes dense from zero."""
    out = []

    def walk(prefix):
        if len(prefix) == n:
            if sorted(set(prefix)) == list(range(max(prefix) + 1)):
                out.append(tuple(prefix))
            return
        for c in range(3):
            if prefix[-1] != c:
                walk(prefix + [c])

    walk([0])
    return out


# ---------------------------------------------------------------------------
# Derivations: candidate parameters read off the visible cells.  They only have
# to propose; ``_accept`` verifies every proposal against the finished grid.
# ---------------------------------------------------------------------------

def _line_profile(g: np.ndarray, mask: np.ndarray, axis: int):
    """Per row (axis 0) or column (axis 1): the one field colour visible there,
    None when nothing is visible, False when the line is not monochrome."""
    prof = []
    for i in range(g.shape[axis]):
        line_g = g[i, :] if axis == 0 else g[:, i]
        line_m = mask[i, :] if axis == 0 else mask[:, i]
        vals = np.unique(line_g[line_m])
        prof.append(None if len(vals) == 0 else (int(vals[0]) if len(vals) == 1 else False))
    return prof


def _key_profile(g: np.ndarray, mask: np.ndarray, key: np.ndarray) -> dict:
    """The same reading, grouped by an arbitrary integer key array."""
    prof = {}
    for value in np.unique(key):
        vals = np.unique(g[(key == value) & mask])
        prof[int(value)] = None if len(vals) == 0 else (int(vals[0]) if len(vals) == 1 else False)
    return prof


def _band_periods(prof: dict, colors: tuple):
    """Periods and phases explaining a 1-D colour profile as two cyclic runs."""
    out = []
    if len(colors) != 2:
        return out
    for p in range(PERIOD_LO, PERIOD_HI + 1):
        residue, ok = {}, True
        for key, colour in prof.items():
            if colour is None:
                continue
            if residue.setdefault(key % p, colour) != colour:
                ok = False
                break
        if not ok or len(residue) != p or set(residue.values()) != set(colors):
            continue
        pattern = [residue[r] for r in range(p)]
        for one in colors:
            positions = [i for i, v in enumerate(pattern) if v == one]
            t = len(positions)
            if not 2 <= t <= p - 2:
                continue
            for phase in range(p):
                shifted = [pattern[(i - phase) % p] for i in range(p)]
                if all(v == one for v in shifted[:t]) and all(v != one for v in shifted[t:]):
                    out.append((p, t, phase))
                    break
    return out


def _line_lattice(positions: list[int], side: int):
    """Candidate (period, thickness, phase) of evenly spaced line bands."""
    runs = _bands_of(positions)
    if len(runs) < 2:
        return []
    starts = [a for a, _ in runs]
    gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    k = _modal(gaps)
    if not LINE_LO <= k <= LINE_HI:
        return []
    thick = _modal([b - a + 1 for a, b in runs])
    return [(k, thick, (-a) % k) for a, _ in runs]


def _lattice_from_boundaries(g: np.ndarray, mask: np.ndarray, axis: int):
    """Block size and phase of a tile lattice, from the lines across which every
    visible pair of field cells changes colour."""
    n = g.shape[axis]
    boundaries = []
    for i in range(n - 1):
        if axis == 0:
            a_g, a_m, b_g, b_m = g[i, :], mask[i, :], g[i + 1, :], mask[i + 1, :]
        else:
            a_g, a_m, b_g, b_m = g[:, i], mask[:, i], g[:, i + 1], mask[:, i + 1]
        both = a_m & b_m
        if int(both.sum()) < 2:
            continue
        if not bool((a_g[both] == b_g[both]).any()):
            boundaries.append(i + 1)
    if len(boundaries) < 2:
        return None
    gaps = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    block = _modal(gaps)
    if not TILE_LO <= block <= TILE_HI:
        return None
    return block, boundaries[0] % block


def _derive(g: np.ndarray, mask: np.ndarray, family: str, colors: tuple):
    h, w = g.shape
    if family == "bands":
        for axis in (0, 1):
            prof = _line_profile(g, mask, axis)
            if any(v is False for v in prof):
                continue
            for p, t, phase in _band_periods(dict(enumerate(prof)), colors):
                yield {"axis": axis, "period": p, "thickness": t, "phase": phase}
    elif family == "diagonal_bands":
        rr, cc = np.indices((h, w))
        for sign, key in ((1, rr + cc), (-1, rr - cc + w)):
            prof = _key_profile(g, mask, key)
            if any(v is False for v in prof.values()):
                continue
            for p, t, phase in _band_periods(prof, colors):
                yield {"sign": sign, "period": p, "thickness": t, "phase": phase}
    elif family == "tiles":
        rows = _lattice_from_boundaries(g, mask, 0)
        cols = _lattice_from_boundaries(g, mask, 1)
        if rows is None or cols is None:
            return
        for off in range(len(colors)):
            yield {"bh": rows[0], "bw": cols[0], "colors": len(colors),
                   "phase_r": rows[1], "phase_c": cols[1], "offset": off}
    elif family == "grid_lines":
        for line_colour in colors:
            rows = [r for r in range(h) if _line_is(g, mask, 0, r, line_colour)]
            cols = [c for c in range(w) if _line_is(g, mask, 1, c, line_colour)]
            for k1, tr, pr in _line_lattice(rows, h):
                for k2, tc, pc in _line_lattice(cols, w):
                    if k1 == k2:
                        yield {"k": k1, "tr": tr, "tc": tc, "phase_r": pr, "phase_c": pc}
    elif family == "concentric":
        rr, cc = np.indices((h, w))
        depth = np.minimum(np.minimum(rr, cc), np.minimum(h - 1 - rr, w - 1 - cc))
        for t in (1, 2):
            prof = _key_profile(g, mask, depth // t)
            if any(v is False for v in prof.values()):
                continue
            for parity in (0, 1):
                yield {"t": t, "parity": parity}
    else:                                                      # split
        for axis in (0, 1):
            prof = _line_profile(g, mask, axis)
            if any(v is False for v in prof):
                continue
            side = h if axis == 0 else w
            filled, ok = [], True
            for i, v in enumerate(prof):
                if v is not None:
                    filled.append(v)
                    continue
                left = next((prof[j] for j in range(i - 1, -1, -1) if prof[j] is not None), None)
                right = next((prof[j] for j in range(i + 1, side) if prof[j] is not None), None)
                if left is not None and left == right:
                    filled.append(left)
                else:
                    ok = False
                    break
            if not ok:
                continue
            runs = _runs(filled)
            order, labels = [], []
            for value, _start, _length in runs:
                if value not in order:
                    order.append(value)
                labels.append(order.index(value))
            yield {"axis": axis, "cuts": [start for _v, start, _l in runs[1:]], "labels": labels}


def _line_is(g, mask, axis, index, colour) -> bool:
    line_g = g[index, :] if axis == 0 else g[:, index]
    line_m = mask[index, :] if axis == 0 else mask[:, index]
    if not line_m.any():
        return False
    return bool((line_g[line_m] == colour).all())
