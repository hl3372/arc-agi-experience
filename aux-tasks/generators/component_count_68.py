"""Visible-component counting on a plain or patterned background.

The planner builds the scene first and draws the two instruction variants --
whether a lone cell is an object (the singleton rule) and whether cells touch
at corners (the connectivity) -- from the finished grid, so neither can be read
off the picture.  All four answers are planned structurally from the drawn
bodies and then recomputed twice from the final image, once by a pure-Python
flood fill and once by numpy label propagation.

A quarter of the canvases carry a background *field* from
``generators.pattern_fields``: bands, diagonal bands, tiles, grid lines,
concentric rings or a panel split.  The field is a pure function of the canvas
coordinates and the objects are stamped opaquely on top of it, so the object
layer is exactly ``grid != field``.  The prompt never names the background --
the same sentence appears on plain and patterned grids -- so the reader has to
recover the field, and the generator proves that exactly one field reading is
admissible on a patterned grid and none on a plain one.  The two readings are
kept apart by a single measurement: a plain field needs a colour on more than
half the canvas, a patterned one allows no colour over ``MAX_COLOR_SHARE``.

Objects on a patterned canvas must read as figures on that pattern, not as
part of it: every body straddles at least two field colours, keeps clear of
the canvas border, never repeats the lattice it sits on, never runs flush
along the pattern's own edges, and leaves the field visible (``AB``/``V``
checks below, all recomputed from the final grid by ``validate``).  AB5, the
flush-edge bound, is measured under BOTH connectivities, because the one the
prompt states is drawn after the scene and under 8-connectivity a diagonal
pair of one colour is a single object.

A scene is planned once and drawn on the canvas it was planned for.  A retry
redraws the same plan on the same canvas -- there is no canned example at the
end of the budget, and ``zero_class`` reports what the finished grid shows.
The zero classes are pictures in their own right: an empty unframed canvas
carries a background field of three classes (two classes always leave one of
them over half the canvas, which is the plain reading's own certificate), an
empty framed one shows the marker with the scene outside it, and where a
canvas is too small for such a field most of that share is drawn as the other
zero class, sparse lone cells, which is zero under rule B alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import common.backgrounds as bglib
from generators import pattern_fields as pf
from generators.exterior_field import infer_exterior_field
from generators.scene_shapes_56_60 import halo, placement_map

MAX_COUNT = 24
FAMILIES = ("rectangle", "bar", "L", "T", "U", "H", "plus", "frame", "staircase", "singleton")
PATTERN_FAMILIES = ("rectangle", "L", "T", "U", "H", "plus", "staircase", "frame")
RULES = ("count", "ignore")
KEYS = (("count", 4), ("count", 8), ("ignore", 4), ("ignore", 8))
PLAN_KEY = {("count", 4): "planned_count_A_4", ("count", 8): "planned_count_A_8",
            ("ignore", 4): "planned_count_B_4", ("ignore", 8): "planned_count_B_8"}

MAX_ISOLATED = 12                  # sparse dots are capped by COUNT, never by density
PATTERN_P = 0.42                   # share of eligible canvases where a field is tried
EDGE_ALIGNMENT_MAX = 0.50          # AB5
SCALE_FRACTION = 0.25              # AB6
MAX_BBOX_EXTENT = 0.60             # AB6
COVER_CAP = {False: 0.25, True: 0.32}      # V4, by ``framed``
FAMILY_WEIGHT = {"concentric": 6.5}


class SceneRetry(Exception):
    """A draw that cannot be completed; the caller redraws on the same canvas."""


# ---------------------------------------------------------------------------
# Two literal readings of the object layer
# ---------------------------------------------------------------------------

def components(grid, connectivity=4, background=None, mask=None):
    """Literal final-image flood fill; it does not know the planned bodies."""
    a = np.asarray(grid)
    if mask is None:
        todo = {(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])
                if background is None or a[r, c] != background}
    else:
        todo = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    result = []
    while todo:
        start = todo.pop()
        color = a[start]
        cells, pending = {start}, [start]
        for r, c in pending:
            for dr, dc in offsets:
                q = r + dr, c + dc
                if q in todo and a[q] == color:
                    todo.remove(q)
                    cells.add(q)
                    pending.append(q)
        result.append((color.item(), cells))
    return result


def _inside(cells, box):
    r0, c0, r1, c1 = box
    return all(r0 <= r < r1 and c0 <= c < c1 for r, c in cells)


def count_objects(grid, mask, rule, connectivity, *, box=None):
    """Objects of the prompt: components of the object layer, dropping the
    single cells when the singleton rule says to ignore them."""
    if rule not in RULES or connectivity not in (4, 8):
        raise ValueError("Unknown counting rule")
    total = 0
    for _color, cells in components(grid, connectivity, mask=mask):
        if rule == "ignore" and len(cells) < 2:
            continue
        if box is not None and not _inside(cells, box):
            continue
        total += 1
    return total


def _propagate(a, mask, connectivity):
    """Component labels of the object layer by numpy label propagation.

    Every update is a minimum against the current labels, so the array only
    ever decreases and the sweep terminates.
    """
    h, w = a.shape
    sentinel = h * w
    lab = np.where(mask, np.arange(h * w).reshape(h, w), sentinel)
    dirs = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if connectivity == 8 else [])
    pairs = []
    for dr, dc in dirs:
        s1 = (slice(0, h - dr), slice(max(0, -dc), w - max(0, dc)))
        s2 = (slice(dr, h), slice(max(0, dc), w - max(0, -dc)))
        pairs.append((s1, s2, mask[s1] & mask[s2] & (a[s1] == a[s2])))
    while True:
        before = lab.copy()
        for s1, s2, same in pairs:
            joint = np.minimum(lab[s1], lab[s2])
            lab[s1] = np.minimum(lab[s1], np.where(same, joint, lab[s1]))
            lab[s2] = np.minimum(lab[s2], np.where(same, joint, lab[s2]))
        if np.array_equal(before, lab):
            return np.where(mask, lab, -1)


def count_objects_numpy(grid, mask, rule, connectivity, *, box=None):
    """The independent second implementation of the same count."""
    a = np.asarray(grid)
    lab = _propagate(a, np.asarray(mask), connectivity)
    ids, sizes = np.unique(lab[lab >= 0], return_counts=True)
    total = 0
    for lid, size in zip(ids, sizes):
        if rule == "ignore" and size < 2:
            continue
        if box is not None:
            rr, cc = np.nonzero(lab == lid)
            r0, c0, r1, c1 = box
            if not (rr.min() >= r0 and rr.max() < r1 and cc.min() >= c0 and cc.max() < c1):
                continue
        total += 1
    return total


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def frame_candidates(grid, mask):
    """Object-layer components that form a hollow rectangular outline.

    Thick rectangular foreground frames are included as competing readings,
    although the principal marker itself has a one-cell border.
    """
    a = np.asarray(grid)
    candidates = []
    for color, cells in components(a, 4, mask=mask):
        r0, c0, r1, c1 = _bounds(cells)
        h, w = r1 - r0, c1 - c0
        if min(h, w) < 3:
            continue
        local = np.zeros((h, w), bool)
        for r, c in cells:
            local[r - r0, c - c0] = True
        for t in range(1, (min(h, w) - 1) // 2 + 1):
            ring = np.ones((h, w), bool)
            ring[t:-t, t:-t] = False
            if np.array_equal(local, ring) and len(cells) == int(ring.sum()):
                candidates.append((color, (r0, c0, r1, c1), t, cells))
                break
    return candidates


# ---------------------------------------------------------------------------
# Reading a finished grid
# ---------------------------------------------------------------------------

def read_field(grid, *, strict=False):
    """The background of the picture: a pattern when exactly one field reading
    is admissible, the plain plurality field when none is.  Two readings, or
    none of either kind, is an ambiguity and raises."""
    a = np.asarray(grid)
    readings = pf.fit(a)
    if strict and {r.field for r in pf.brute(a)} != {r.field for r in readings}:
        raise ValueError("The derived and the swept background readings disagree")
    if len(readings) > 1:
        raise ValueError("More than one admissible background pattern")
    values, counts = np.unique(a, return_counts=True)
    if readings:
        reading = readings[0]
        if counts.max() > .50 * a.size:
            raise ValueError("A patterned scene must not also read as a plain field")
        return {"style": "pattern", "field": reading.array(), "background": int(reading.colors[0]),
                "pattern_family": reading.family, "pattern_params": reading.params,
                "pattern_colors": [int(c) for c in reading.colors]}
    if len(values) == 1:
        bg = int(values[0])
        return {"style": "plain", "field": np.full(a.shape, bg, int), "background": bg,
                "pattern_family": None, "pattern_params": None, "pattern_colors": [bg]}
    order = np.argsort(counts)
    bg = int(values[order[-1]])
    if counts[order[-1]] <= .50 * a.size or counts[order[-1]] < 1.4 * counts[order[-2]]:
        raise ValueError("The plain field has insufficient visible context")
    return {"style": "plain", "field": np.full(a.shape, bg, int), "background": bg,
            "pattern_family": None, "pattern_params": None, "pattern_colors": [bg]}


def read(grid, *, framed=False, strict=False):
    """The field, all plausible markers, and the four answers, from visible cells.

    The plurality and exterior tests are conservative certificates for the plain
    branch of this drawing family, not a general definition of ARC background.
    """
    a = np.asarray(grid)
    if a.ndim != 2 or not (5 <= a.shape[0] <= 30 and 5 <= a.shape[1] <= 30):
        raise ValueError("Counting canvas sides must be 5 through 30")
    info = read_field(a, strict=strict)
    field, style, bg = info["field"], info["style"], info["background"]
    if style == "plain" and len(np.unique(a)) == 1:
        if framed:
            raise ValueError("A framed scene needs a visible rectangle")
        return {**info, "counts": {key: 0 for key in KEYS}, "frame": None, "frame_candidates": 0}
    if style == "plain" and not framed:
        if infer_exterior_field(a, margin=1.4, exterior_mass=False) != bg:
            raise ValueError("The spatial and color field readings disagree")
    mask = a != field
    if not framed:
        counts = {key: count_objects(a, mask, *key) for key in KEYS}
        return {**info, "counts": counts, "frame": None, "frame_candidates": 0}
    readings = []
    for _color, (r0, c0, r1, c1), t, border in frame_candidates(a, mask):
        if style == "plain":
            uncovered = a.copy()
            for q in border:
                uncovered[q] = bg
            try:
                if len(np.unique(uncovered)) > 1 and \
                        infer_exterior_field(uncovered, margin=1.4, exterior_mass=False) != bg:
                    continue
            except ValueError:
                continue
        # Full-component containment, rather than clipping a crossing object.
        box = (r0 + t, c0 + t, r1 - t, c1 - t)
        values = tuple(count_objects(a, mask, rule, conn, box=box) for rule, conn in KEYS)
        readings.append((values, (r0, c0, r1, c1)))
    answers = {values for values, _ in readings}
    if len(answers) != 1:
        raise ValueError("No uniquely determined marked-rectangle count")
    values = next(iter(answers))
    return {**info, "counts": dict(zip(KEYS, values)), "frame": readings[0][1],
            "frame_candidates": len(readings)}


def solve(grid, rule, connectivity, *, framed=False):
    if rule not in RULES or connectivity not in (4, 8):
        raise ValueError("Unknown counting rule")
    return read(grid, framed=framed)["counts"][(rule, connectivity)]


def naive_plurality_counts(grid, *, box=None):
    """The shortcut reader: whatever colour is most common is the background."""
    a = np.asarray(grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    mask = a != bg
    return {key: count_objects(a, mask, *key, box=box) for key in KEYS}


# ---------------------------------------------------------------------------
# Validation of a finished example
# ---------------------------------------------------------------------------

def validate(example, *, framed=False, strict=False):
    """Re-solve the final grid; metadata never supplies the field or answer."""
    m = example["metadata"]
    a = np.array(example["input"])
    result = read(a, framed=framed, strict=strict)
    if result["style"] != m["background_style"]:
        raise ValueError("The recovered background disagrees with the drawn one")
    field = result["field"]
    mask = a != field
    box = None
    if framed:
        if list(result["frame"]) != list(m["frame_bounds"]):
            raise ValueError("The recovered marker disagrees with the drawn rectangle")
        r0, c0, r1, c1 = m["frame_bounds"]
        box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    for key in KEYS:
        literal = count_objects(a, mask, *key, box=box)
        if literal != count_objects_numpy(a, mask, *key, box=box):
            raise ValueError("The two component implementations disagree")
        if literal != result["counts"][key]:
            raise ValueError("The marked-rectangle reading disagrees with the plain count")
        if literal != m[PLAN_KEY[key]]:
            raise ValueError("The visible components disagree with the fixed scene plan")
    if int(example["output"]) != result["counts"][(m["singleton_rule"], m["connectivity"])]:
        raise ValueError("Wrong visible-component answer")
    _validate_layers(a, field, m, framed=framed, result=result)
    return result


def _validate_layers(a, field, m, *, framed, result):
    """AB1 - AB6 and V1 - V6, every one recomputed from the final grid."""
    from generators.object_counting import _object_view

    mask = a != field
    field_colors = set(int(c) for c in result["pattern_colors"])
    if set(int(c) for c in m["object_colors"]) & field_colors:
        raise ValueError("An object colour occurs in the background field")     # AB1
    if m["background_style"] == "plain":
        if pf.fit(a):
            raise ValueError("A plain scene must have no admissible pattern reading")
        return
    if set(int(c) for c in m["isolated_colors"]) & field_colors:
        raise ValueError("An isolated cell's colour occurs in the background field")
    if framed and int(m["roi_color"]) in field_colors:
        raise ValueError("The marker colour occurs in the background field")
    # Three views of the same object layer: by colour (the field colours are
    # known), by geometry (the rendering of the recovered parameters) and
    # structurally, the way the prompt reads a grid -- field colours are
    # background and isolated single cells are texture.
    family, params = result["pattern_family"], result["pattern_params"]
    layers = pf.render(family, params, *a.shape)
    if not np.array_equal(field, np.array(result["pattern_colors"], int)[layers]):
        raise ValueError("The field is not the rendering of its own parameters")  # AB4
    if not np.array_equal(mask, ~np.isin(a, list(field_colors))):
        raise ValueError("The colour and geometric object views disagree")
    dots = {(int(r), int(c)) for r, c in m["isolated_cells"]}
    ground = int(m["ground_color"])
    structural = _object_view(a, ground, field_colors) != ground
    if not np.array_equal(structural, mask & ~_cell_mask(dots, a.shape)):
        raise ValueError("The structural and colour object views disagree")
    lab = pf.label4(layers)
    sizes = np.bincount(lab.ravel())
    metrics = pf.visibility(family, params, layers, mask, lab=lab,
                            annotations=pf.outline_components(a) & mask)
    if not pf.visibility_ok(family, metrics):
        raise ValueError("The background pattern is not visible enough")          # V1-V3
    if metrics["covered_share"] > COVER_CAP[framed]:
        raise ValueError("The objects cover too much of the patterned canvas")    # V4
    shares = pf.class_shares(layers)
    if shares.min() < pf.SHARE_LO or shares.max() > pf.SHARE_HI:
        raise ValueError("A background class owns an inadmissible share")         # V5
    if int(np.bincount(a.ravel(), minlength=10).max()) > pf.MAX_COLOR_SHARE * a.size:
        raise ValueError("A colour dominates the patterned canvas")               # V6
    marker = _outline_cells(m["frame_bounds"]) if framed else set()
    straddle = []
    for _color, cells in components(a, 4, mask=mask):
        if cells & marker:
            continue
        if len(cells) == 1 and next(iter(cells)) in dots:
            continue                                   # a dot answers to AB1 only
        classes = {int(layers[q]) for q in cells}
        straddle.append(len(classes))
        if len(classes) < 2:
            raise ValueError("An object hides inside a single background region")  # AB2
        region_min = int(sizes[[int(lab[q]) for q in cells]].min())
        if len(cells) > SCALE_FRACTION * region_min and \
                min(sum(1 for q in cells if layers[q] == k) for k in classes) < _straddle_need(len(cells)):
            raise ValueError("An object is too large for the region it sits in")  # AB6
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        if max(rr) - min(rr) + 1 > MAX_BBOX_EXTENT * a.shape[0] or \
                max(cc) - min(cc) + 1 > MAX_BBOX_EXTENT * a.shape[1]:
            raise ValueError("An object spans too much of the canvas")             # AB6
        if min(rr) == 0 or min(cc) == 0 or max(rr) == a.shape[0] - 1 or max(cc) == a.shape[1] - 1:
            raise ValueError("An object touches the border of a patterned canvas")
    alignment = _alignment_max(a, mask, layers, skip=marker)                      # AB5
    if alignment > EDGE_ALIGNMENT_MAX:
        raise ValueError("An object runs flush along the pattern's own edges")
    if abs(alignment - float(m["edge_alignment_max"])) > 5e-4:
        raise ValueError("The flush-edge measure disagrees with the final grid")
    if (min(straddle) if straddle else 0) != m["straddle_min_colors"]:
        raise ValueError("The straddle measure disagrees with the final grid")
    if framed:
        if result["frame_candidates"] != 1:
            raise ValueError("The marked rectangle is not unique in the object layer")
        r0, c0, r1, c1 = m["frame_bounds"]
        if _frame_sides_on_boundary(layers, (r0, c0, r1, c1)) > 1:
            raise ValueError("The marker hides more than one pattern boundary")
        if min(_periods_inside(family, params, (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                               *a.shape) or [2]) < 2:
            raise ValueError("The marker hides the repetition of the field")
    box = None if not framed else (m["frame_bounds"][0] + 1, m["frame_bounds"][1] + 1,
                                   m["frame_bounds"][2] - 1, m["frame_bounds"][3] - 1)
    naive = naive_plurality_counts(a, box=box)
    if any(naive[key] == result["counts"][key] for key in KEYS):
        raise ValueError("The plurality-is-background shortcut reaches the answer")


def _straddle_need(n: int) -> int:
    """AB6: how many cells an object must show on EVERY field colour it covers
    once it is large for the region it sits in.  Three on a body of nine cells
    or more, and never more than a third of the body, because a fine lattice
    has four-cell regions that no drawable body could straddle three-and-three."""
    return min(3, max(1, n // 3))


def _cell_mask(cells, shape):
    out = np.zeros(shape, bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _outline_cells(box):
    r0, c0, r1, c1 = box
    cells = set()
    for r in range(r0, r1):
        cells.add((r, c0))
        cells.add((r, c1 - 1))
    for c in range(c0, c1):
        cells.add((r0, c))
        cells.add((r1 - 1, c))
    return cells


def _alignment_max(grid, mask, layers, *, skip=()):
    """AB5 over the objects a reader sees under EITHER connectivity.

    The singleton rule and the connectivity are drawn after the scene, so the
    flush-edge bound has to hold under both readings of the picture: under the
    8-connectivity the prompt may state, a diagonal pair of one colour is ONE
    object, and its boundary runs further along the field's own edges than
    either half does.
    """
    worst = 0.0
    for connectivity in (4, 8):
        for _color, cells in components(grid, connectivity, mask=mask):
            if len(cells) < 6 or (skip and cells & skip):
                continue
            worst = max(worst, _edge_alignment(cells, layers, grid.shape))
    return worst


def _edge_alignment(cells, layers, shape):
    """Share of an object's boundary edges that coincide with a field edge."""
    h, w = shape
    boundary = aligned = 0
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w) or q in cells:
                continue
            boundary += 1
            if layers[q] != layers[r, c]:
                aligned += 1
    return aligned / boundary if boundary else 0.0


# ---------------------------------------------------------------------------
# Capacity witnesses and body shapes (unchanged in spirit; the patterned branch
# forbids bars, dominoes and lone cells so no body can mimic the field)
# ---------------------------------------------------------------------------

def capacity(h, w, minimum=2):
    """Packing capacity for minimum bounding boxes with a one-cell moat."""
    if h <= 0 or w <= 0:
        return 0
    return ((h+1)//(minimum+1))*((w+1)//(minimum+1))


def _partition(rng, box, n, minimum):
    """A random guillotine witness, with an empty cell between every leaf."""
    r0, c0, r1, c1 = box
    h, w = r1-r0, c1-c0
    if n == 1:
        return [box]
    cuts = []
    for axis, extent in ((0, h), (1, w)):
        for cut in range(minimum, extent-minimum):
            if axis == 0:
                ca, cb = capacity(cut, w, minimum), capacity(h-cut-1, w, minimum)
            else:
                ca, cb = capacity(h, cut, minimum), capacity(h, w-cut-1, minimum)
            if ca and cb and ca+cb >= n:
                lo, hi = max(1, n-cb), min(ca, n-1)
                if lo <= hi:
                    cuts.append((axis, cut, lo, hi))
    if not cuts:
        raise ValueError("The advertised footprint capacity has no partition witness")
    axis, cut, lo, hi = cuts[int(rng.integers(len(cuts)))]
    na = int(rng.integers(lo, hi+1))
    if axis == 0:
        left, right = (r0,c0,r0+cut,c1), (r0+cut+1,c0,r1,c1)
    else:
        left, right = (r0,c0,r1,c0+cut), (r0,c0+cut+1,r1,c1)
    return _partition(rng,left,na,minimum)+_partition(rng,right,n-na,minimum)


def _mask(family, h, w, t=1):
    y, x = np.indices((h,w))
    if family in ("rectangle", "bar", "singleton"):
        return np.ones((h,w),bool)
    if family == "L":
        return (x<t)|(y>=h-t)
    if family == "T":
        return (y<t)|((x>=(w-t)//2)&(x<(w-t)//2+t))
    if family == "U":
        return (x<t)|(x>=w-t)|(y>=h-t)
    if family == "H":
        return (x<t)|(x>=w-t)|((y>=(h-t)//2)&(y<(h-t)//2+t))
    if family == "plus":
        return ((x>=(w-t)//2)&(x<(w-t)//2+t))|((y>=(h-t)//2)&(y<(h-t)//2+t))
    if family == "frame":
        return (x<t)|(x>=w-t)|(y<t)|(y>=h-t)
    if family == "staircase":
        steps = min(4,h,w)
        return x < np.ceil((np.minimum(steps-1,y*steps//h)+1)*w/steps)
    raise ValueError(f"Unknown shape family {family}")


def _body(rng, h, w, budget, *, frames=True, patterned=False, singletons=True):
    choices, weights = [], []
    if not patterned:
        if singletons:
            choices += ["singleton"]; weights += [.06]
        if budget >= 2 and max(h,w) >= 2:
            choices += ["bar"]; weights += [.12]
    if min(h,w) >= 2 and budget >= 3:
        choices += ["L", "staircase"]; weights += [.15,.12]
    if min(h,w) >= 2 and budget >= 4:
        choices += ["rectangle"]; weights += [.22]
    if min(h,w) >= 3 and budget >= 5:
        choices += ["T", "plus"]; weights += [.10,.10]
    if min(h,w) >= 3 and budget >= 7:
        choices += ["U", "H"]; weights += [.09,.09]
    if frames and min(h,w) >= 3 and budget >= 8:
        choices += ["frame"]; weights += [.08]
    if not choices:
        raise SceneRetry("No body family fits this footprint")
    family = str(rng.choice(choices,p=np.array(weights)/sum(weights)))
    if family == "singleton":
        return np.ones((1,1),bool),family
    if family == "bar":
        vertical = h >= 2 and (w < 2 or bool(rng.integers(2)))
        cap = min(h if vertical else w,budget)
        length = int(rng.integers(max(2,int(np.ceil(.55*cap))),cap+1))
        return np.ones((length,1) if vertical else (1,length),bool),family
    minimum = 2 if family in ("L","rectangle","staircase") else 3
    min_area = int(_mask(family,minimum,minimum).sum())
    target = int(rng.integers(max(min_area,int(.60*budget)),budget+1))
    sh = int(rng.integers(max(minimum,int(np.ceil(.55*h))),h+1))
    sw = int(rng.integers(max(minimum,int(np.ceil(.55*w))),w+1))
    t = 1
    if family not in ("staircase","rectangle") and min(sh,sw) >= 6:
        t = int(rng.integers(1,min(3,min(sh,sw)//3)+1))
    mask = _mask(family,sh,sw,t)
    while int(mask.sum()) > target:
        if t > 1:
            t -= 1
        elif sh > minimum and (sw == minimum or rng.integers(2)):
            sh -= 1
        elif sw > minimum:
            sw -= 1
        else:
            raise ValueError("Shape minimum exceeds its certified area budget")
        mask = _mask(family,sh,sw,t)
    if rng.integers(2): mask = mask[::-1]
    if rng.integers(2): mask = mask[:,::-1]
    return mask.copy(),family


def _cluster(rng, h, w, budget, kind, *, frames, patterned=False, singletons=True):
    if kind == "single":
        mask,family = _body(rng,h,w,budget,frames=frames,patterned=patterned,
                            singletons=singletons)
        return mask.astype(int),[family]
    # A cut across rows/columns leaves enough room for both arbitrary bodies.
    def split(extent):
        # On a roomy footprint both sides support drawn bodies. Unrestricted
        # cuts created needless singleton pairs even on large canvases.
        lo=max(1,int(np.ceil(.35*extent)))
        hi=min(extent-1,int(np.floor(.65*extent)))
        if lo>hi:lo,hi=1,extent-1
        return int(rng.integers(lo,hi+1))
    ah,aw=split(h),split(w)
    bh,bw = h-ah,w-aw
    ba = max(1,min(ah*aw,budget//2))
    bb = max(1,min(bh*bw,budget-ba))
    a,fa = _body(rng,ah,aw,ba,frames=frames,patterned=patterned)
    b,fb = _body(rng,bh,bw,bb,frames=frames,patterned=patterned)
    if kind == "diagonal":
        # Last row of A ends one column before the first row of B begins:
        # there is diagonal contact and no edge contact, for any body shapes.
        dr = a.shape[0]
        dc = int(np.flatnonzero(a[-1])[-1])+1-int(np.flatnonzero(b[0])[0])
    else:
        dc = a.shape[1]
        dr = int(np.flatnonzero(a[:,-1])[-1])-int(np.flatnonzero(b[:,0])[0])
    ymin,xmin = min(0,dr),min(0,dc)
    ymax,xmax = max(a.shape[0],dr+b.shape[0]),max(a.shape[1],dc+b.shape[1])
    out = np.zeros((ymax-ymin,xmax-xmin),int)
    ay,ax = -ymin,-xmin; by,bx = dr-ymin,dc-xmin
    out[ay:ay+a.shape[0],ax:ax+a.shape[1]][a] = 1
    out[by:by+b.shape[0],bx:bx+b.shape[1]][b] = 2
    if out.shape[0] > h or out.shape[1] > w:
        raise ValueError("Contact group exceeds its footprint witness")
    if rng.integers(2): out = out[::-1]
    if rng.integers(2): out = out[:,::-1]
    return out.copy(),[fa,fb]


@dataclass
class Domain:
    box: tuple
    count: int
    kinds: list
    leaves: list
    count_cap: int
    minimum: int


def _domain_plan(rng, box, *, forced_zero=False, outside=False, count_limit=MAX_COUNT,
                 minimum=None):
    h,w = box[2]-box[0],box[3]-box[1]
    if minimum is None:
        minimum = 1 if min(h,w) < 2 else 2
    cap = min(count_limit,8 if outside else MAX_COUNT,capacity(h,w,minimum))
    if cap == 0 or forced_zero:
        return Domain(box,0,[],[],cap,minimum)
    diagonal = min(h,w) >= 2 and cap >= 2 and rng.random() < .68
    lo = 2 if diagonal else 1
    high = min(cap,8) if rng.random() < .55 else cap
    count = int(rng.integers(lo,max(lo,high)+1))
    nd = int(rng.integers(1,min(4,count//2)+1)) if diagonal else 0
    remaining = count-2*nd
    ne = int(rng.integers(1,min(3,remaining//2)+1)) if minimum>=2 and remaining>=2 and rng.random()<.36 else 0
    kinds = ["diagonal"]*nd+["different_color_edge"]*ne+["single"]*(count-2*nd-2*ne)
    rng.shuffle(kinds)
    leaves = _partition(rng,box,len(kinds),minimum)
    rng.shuffle(leaves)
    return Domain(box,count,kinds,leaves,cap,minimum)


def _outside_boxes(h,w,frame):
    r0,c0,r1,c1 = frame
    boxes = [(1,1,r0,w-1),(r1,1,h-1,w-1),
             (max(1,r0),1,min(h-1,r1),c0),
             (max(1,r0),c1,min(h-1,r1),w-1)]
    return [b for b in boxes if b[2]>b[0] and b[3]>b[1]]


def _paint_plan(rng, domains, *, framed, canvas_area, frame_area, unit=1, area_share=.43,
                fill_share=.47, patterned=False, max_side=None, object_area=None,
                need_body=None):
    parts = []
    # A visible majority of plain field remains after the frame and objects.
    min_area = unit*sum(d.count for d in domains)
    area_cap = max(min_area, object_area if object_area is not None
                   else int(area_share*canvas_area)-frame_area)
    leaves = [(d,k,box) for d in domains for k,box in zip(d.kinds,d.leaves)]
    available_area = sum((b[2]-b[0])*(b[3]-b[1]) for _,_,b in leaves)
    desired = min(area_cap,max(min_area,int(fill_share*available_area)))
    extra = max(0,desired-min_area)
    # Exact allocation before any geometry or placement retry.
    weights = np.array([(b[2]-b[0])*(b[3]-b[1]) for _,_,b in leaves],float)
    additions = np.zeros(len(leaves),int)
    if len(leaves) and extra:
        raw = extra*weights/weights.sum(); additions = np.floor(raw).astype(int)
        for i in np.argsort(-(raw-additions))[:extra-int(additions.sum())]: additions[i] += 1
    # One body of the counted region is drawn solid rather than left to chance:
    # a scene that is not a planned zero has something to count under EITHER
    # rule, and on a narrow canvas redrawing until it does is a lottery.
    solid = None
    if need_body is not None:
        room = [i for i,(d,k,b) in enumerate(leaves)
                if d is need_body and k == "single"
                and max(b[2]-b[0],b[3]-b[1]) >= 2 and (b[2]-b[0])*(b[3]-b[1]) >= 2]
        if room:
            solid = max(room, key=lambda i: (leaves[i][2][2]-leaves[i][2][0]) *
                        (leaves[i][2][3]-leaves[i][2][1]))
    for index,(domain,kind,box) in enumerate(leaves):
        h,w = box[2]-box[0],box[3]-box[1]
        if max_side is not None:
            # AB6 caps an object at 60% of either canvas side, so a roomy leaf
            # must not hand the body a footprint it may not use.
            h,w = min(h,max_side[0]),min(w,max_side[1])
        n = 1 if kind=="single" else 2
        budget = min(h*w,n*unit+int(additions[index]))
        if index == solid:
            budget = max(budget, 2)
        labels,families = _cluster(rng,h,w,budget,kind,frames=not framed,patterned=patterned,
                                   singletons=index != solid)
        y = int(rng.integers(box[0],box[2]-labels.shape[0]+1))
        x = int(rng.integers(box[1],box[3]-labels.shape[1]+1))
        parts.append({"labels":labels,"kind":kind,"families":families,
                      "domain":domain,"witness":(y,x),"budget":budget})
    return parts


def _assign_colors(rng,parts,colors):
    # Edge pairs receive different colors; diagonal pairs share one class.
    pending = list(colors);rng.shuffle(pending)
    for part in sorted(parts,key=lambda p:p["kind"]!="different_color_edge"):
        n = 2 if part["kind"]=="different_color_edge" else 1
        chosen = []
        for _ in range(n):
            if pending:
                color = pending.pop()
            else:
                options = [c for c in colors if c not in chosen]
                color = int(rng.choice(options))
            chosen.append(color)
        part["colors"] = chosen if n==2 else chosen*int(part["labels"].max())
    if pending:
        raise ValueError("The requested palette exceeds semantic component capacity")


def _stamp(grid,part,origin):
    y,x = origin; labels=part["labels"]
    view = grid[y:y+labels.shape[0],x:x+labels.shape[1]]
    for index,color in enumerate(part["colors"],1): view[labels==index]=color
    occupied = np.zeros(grid.shape,bool)
    occupied[y:y+labels.shape[0],x:x+labels.shape[1]] = labels>0
    return occupied


# ---------------------------------------------------------------------------
# Placement, pattern-aware
# ---------------------------------------------------------------------------

def _straddle_map(labels, layers, shape):
    """Origins at which EVERY body of the group covers two field colours."""
    ok = np.ones(shape, bool)
    k = int(layers.max())+1
    for value in range(1, int(labels.max())+1):
        cells = np.argwhere(labels == value)
        if len(cells) < 2:
            continue                       # a lone cell answers to AB1 only
        seen = np.zeros(shape, int)
        for cls in range(k):
            plane = (layers == cls).astype(np.int16)
            acc = np.zeros(shape, np.int16)
            for y, x in cells:
                acc += plane[y:y+shape[0], x:x+shape[1]]
            seen += (acc > 0)
        ok &= seen >= 2
    return ok


def _frame_wall(sub):
    for i in range(min(sub.shape)):
        if not sub[i, i]:
            return i
    return min(sub.shape)


def _object_ok(pattern, part, origin, h, w):
    """AB3, AB5 and AB6 for every body of the group at this origin."""
    layers, family, params = pattern["layers"], pattern["family"], pattern["params"]
    lab, sizes, depth = pattern["lab"], pattern["sizes"], pattern.get("depth")
    y, x = origin
    labels = part["labels"]
    for value in range(1, int(labels.max())+1):
        sub = labels == value
        mask = np.zeros((h, w), bool)
        mask[y:y+labels.shape[0], x:x+labels.shape[1]] = sub
        cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
        n = len(cells)
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        bh, bw = max(rr)-min(rr)+1, max(cc)-min(cc)+1
        if bh > MAX_BBOX_EXTENT*h or bw > MAX_BBOX_EXTENT*w:
            return False
        classes = {int(layers[q]) for q in cells}
        if n >= 2 and len(classes) < 2:
            return False
        region_min = int(sizes[[int(lab[q]) for q in cells]].min())
        if n > SCALE_FRACTION*region_min and \
                min(sum(1 for q in cells if layers[q] == k) for k in classes) < _straddle_need(n):
            return False
        if n >= 6 and _edge_alignment(cells, layers, (h, w)) > EDGE_ALIGNMENT_MAX:
            return False
        if family == "tiles" and n == bh*bw:
            if bh == params["bh"] and bw == params["bw"]:
                return False
            if bh % params["bh"] == 0 and bw % params["bw"] == 0 and \
                    (min(rr)-params["phase_r"]) % params["bh"] == 0 and \
                    (min(cc)-params["phase_c"]) % params["bw"] == 0:
                return False
        if family == "concentric":
            if len({int(depth[q]) // params["t"] for q in cells}) < 2:
                return False
            if part["families"][value-1] == "frame" and _frame_wall(sub) == params["t"]:
                return False
    if int(labels.max()) > 1 and len(set(part.get("colors") or [0])) == 1:
        # Under the 8-connectivity the prompt may state, a diagonal pair of one
        # colour is a single object; AB5 answers to that reading as well.
        mask = np.zeros((h, w), bool)
        mask[y:y+labels.shape[0], x:x+labels.shape[1]] = labels > 0
        cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
        if len(cells) >= 6 and _edge_alignment(cells, layers, (h, w)) > EDGE_ALIGNMENT_MAX:
            return False
    return True


def _place_scene(rng,h,w,field,frame,frame_color,parts,*,free,pattern=None):
    grid=field.copy()
    if frame:
        r0,c0,r1,c1=frame
        grid[r0:r1,[c0,c1-1]]=frame_color
        grid[[r0,r1-1],c0:c1]=frame_color
    blocked=np.zeros((h,w),bool)
    if frame:
        marker=grid==frame_color
        blocked|=halo(marker) if pattern is not None else marker
    inner=np.zeros((h,w),bool);inner[1:h-1,1:w-1]=True
    order=sorted(parts,key=lambda p:-int(np.count_nonzero(p["labels"])))
    for part in order:
        if free:
            box=part["domain"].box
            allowed=np.zeros((h,w),bool)
            # Whole-grid domains may use clear edge placements; the field
            # certificate rejects any placement that destroys surrounding space.
            if frame is None:
                allowed[:]=True
            else:
                allowed[box[0]:box[2],box[1]:box[3]]=True
            if pattern is not None:
                allowed&=inner
            mask=part["labels"]>0
            choices=placement_map(mask,allowed&~blocked)
            if choices is None:
                return None
            if pattern is not None:
                choices=choices&_straddle_map(part["labels"],pattern["layers"],choices.shape)
            positions=np.argwhere(choices)
            if not len(positions):return None
            rng.shuffle(positions)
            origin=None
            for candidate in positions[:64]:
                candidate=tuple(map(int,candidate))
                if pattern is None or _object_ok(pattern,part,candidate,h,w):
                    origin=candidate;break
            if origin is None:return None
        else:
            origin=part["witness"]
            if pattern is not None and not _object_ok(pattern,part,origin,h,w):
                return None
        occupied=_stamp(grid,part,origin)
        blocked|=halo(occupied)
    return grid,blocked


def _place_dots(rng, grid, blocked, box, n, colors):
    """Isolated single cells, each with a clear one-cell halo all round."""
    if n <= 0 or not colors:
        return [], None
    h, w = grid.shape
    allowed = np.zeros((h, w), bool)
    allowed[box[0]:box[2], box[1]:box[3]] = True
    allowed &= ~blocked
    kind = None
    order = []
    if n >= 4 and min(h, w) >= 8:
        for _ in range(4):
            candidate = str(rng.choice(bglib.SPARSE_KINDS))
            spots = np.argwhere(bglib.sparse_mask(rng, h, w, candidate) & allowed)
            if len(spots) >= n:
                kind, order = candidate, spots
                break
    if not len(order):
        order = np.argwhere(allowed)
    if len(order):
        rng.shuffle(order)
    wheel = list(colors)
    rng.shuffle(wheel)
    placed = []
    for y, x in order:
        if len(placed) == n:
            break
        y, x = int(y), int(x)
        if blocked[y, x]:
            continue
        color = int(wheel[len(placed) % len(wheel)])
        grid[y, x] = color
        spot = np.zeros((h, w), bool)
        spot[y, x] = True
        blocked |= halo(spot)
        placed.append(((y, x), color))
    if len(placed) < n and kind is not None:
        # The library arrangement did not fit; finish scattered rather than
        # thinning the count, which must not depend on the pattern.
        for y, x in np.argwhere(allowed & ~blocked):
            if len(placed) == n:
                break
            y, x = int(y), int(x)
            if blocked[y, x]:
                continue
            color = int(wheel[len(placed) % len(wheel)])
            grid[y, x] = color
            spot = np.zeros((h, w), bool)
            spot[y, x] = True
            blocked |= halo(spot)
            placed.append(((y, x), color))
        kind = "scattered" if len(placed) else kind
    return placed, (kind or "scattered")


# ---------------------------------------------------------------------------
# The scene
# ---------------------------------------------------------------------------

def _draw_isolated_budget(rng):
    """One distribution for every scene; the singleton rule is drawn later."""
    u = rng.random()
    if u < .40:
        return 0
    if u < .75:
        return int(rng.integers(1, 4))
    return int(rng.integers(4, MAX_ISOLATED + 1))


def _zero_class(planned):
    """What the finished scene shows, not what the draw asked for.

    ``empty`` counts nothing under either rule, ``dots`` counts only lone cells
    (so it is zero under rule B and positive under rule A), ``normal`` has
    bodies to count under both.
    """
    if not max(planned.values()):
        return "empty"
    if not planned[("ignore", 4)] and not planned[("ignore", 8)]:
        return "dots"
    return "normal"


def _plan_counts(parts, target, dots):
    """The four answers, from the drawn bodies rather than from the image."""
    out = {key: 0 for key in KEYS}
    for part in parts:
        if part["domain"] is not target:
            continue
        sizes = [int(np.count_nonzero(part["labels"] == i))
                 for i in range(1, int(part["labels"].max()) + 1)]
        if part["kind"] == "diagonal":
            out[("count", 4)] += 2
            out[("count", 8)] += 1
            out[("ignore", 4)] += sum(1 for s in sizes if s >= 2)
            out[("ignore", 8)] += 1 if sum(sizes) >= 2 else 0
        else:
            out[("count", 4)] += len(sizes)
            out[("count", 8)] += len(sizes)
            out[("ignore", 4)] += sum(1 for s in sizes if s >= 2)
            out[("ignore", 8)] += sum(1 for s in sizes if s >= 2)
    out[("count", 4)] += dots
    out[("count", 8)] += dots
    return out


def _frame_sides_on_boundary(layers, frame):
    """Frame sides that hide a colour change of the field underneath."""
    r0, c0, r1, c1 = frame
    h, w = layers.shape
    sides = 0
    for outside, inside, span in (((r0 - 1, None), (r0 + 1, None), range(c0, c1)),
                                  ((r1, None), (r1 - 2, None), range(c0, c1)),
                                  ((None, c0 - 1), (None, c0 + 1), range(r0, r1)),
                                  ((None, c1), (None, c1 - 2), range(r0, r1))):
        hits = total = 0
        for k in span:
            a = (outside[0], k) if outside[0] is not None else (k, outside[1])
            b = (inside[0], k) if inside[0] is not None else (k, inside[1])
            if not (0 <= a[0] < h and 0 <= a[1] < w and 0 <= b[0] < h and 0 <= b[1] < w):
                continue
            total += 1
            hits += int(layers[a] != layers[b])
        if total and hits / total >= .60:
            sides += 1
    return sides


def _counted_room(fh, fw):
    """The region inside a marker of this size can hold a body of two cells."""
    ih, iw = fh - 2, fw - 2
    return ih >= 1 and iw >= 1 and ih * iw >= 2


def _plain_object_cap(h, w, frame_area):
    """Drawn cells a plain field can carry and still certify itself.

    ``read_field`` calls a plain background only when one colour owns over half
    the canvas and 1.4 times the runner-up -- on a small canvas the marker IS
    the runner-up, so the counted region's budget is what the certificate has
    left, not a flat share of the canvas.
    """
    keep = max(int(.50 * h * w) + 1, int(np.ceil(1.4 * frame_area)))
    return max(0, h * w - frame_area - keep)


def _fit_frame(fh, fw, h, w):
    """Widen a marker until the region it encloses can hold a body of two cells.

    A three-cell side leaves a one-cell interior: on a narrow canvas that is a
    counted region in which nothing can be drawn, and the scene plan would have
    no realisation.  The margin around the marker shrinks instead, as far as
    the plain-field certificate allows.
    """
    for _ in range(4):
        if _counted_room(fh, fw):
            break
        if fw < w - 2 and (fw <= fh or fh >= h - 2):
            grown = (fh, fw + 1)
        elif fh < h - 2:
            grown = (fh + 1, fw)
        else:
            break
        if _plain_object_cap(h, w, 2 * (grown[0] + grown[1]) - 4) < 2:
            break
        fh, fw = grown
    return fh, fw


def _periods_inside(family, params, box, h, w):
    """Full periods of the field that fall inside a box."""
    r0, c0, r1, c1 = box
    window = np.zeros((h, w), bool)
    window[r0:r1, c0:c1] = True
    out = []
    for coord, p in pf.period_axes(family, params, h, w):
        present = {int(v) for v in np.unique(coord[window])}
        if not present:
            out.append(0)
            continue
        full, start, top = 0, min(present), max(present)
        while start + p - 1 <= top:
            if all((start + d) in present for d in range(p)):
                full += 1
                start += p
            else:
                start += 1
        out.append(full)
    return out


def scene(rng, *, framed=False):
    """Plan and realize a scene before the rule or the connectivity is sampled.

    Every retry redraws on the SAME canvas and with the SAME plan -- the
    isolated-cell budget and the zero class are drawn once -- so rejection
    never bends the size distribution and never rewrites what the scene was
    meant to show.  Only the background axis may fall back to plain; a canvas
    that cannot carry its plan raises rather than returning a canned example.
    """
    h, w = map(int, rng.integers(5, 31, size=2))
    exclude = ("concentric",) if framed else ()
    families = pf.admissible_families(h, w, exclude=exclude)
    n_iso = _draw_isolated_budget(rng)
    u = rng.random()
    zero_mode = "empty" if u < .060 else ("dots" if u < .085 else "normal")
    if zero_mode == "empty" and not framed:
        # An unframed empty scene on a plain ground is a single flat colour:
        # under rule A every drawn cell is an object, so nothing may be drawn
        # and only the background can make a picture of it.  Nothing covers
        # that field either, so it has to stand on its own (three classes).
        standalone = pf.standalone_families(h, w)
        if standalone:
            families = standalone
        else:
            # Too small for such a field: mostly draw the other zero class,
            # which is sparse dots, and leave a few flat canvases.
            zero_mode = "empty" if rng.random() < .38 else "dots"
    if zero_mode == "empty":
        n_iso = 0
    elif zero_mode == "dots" and n_iso == 0:
        n_iso = int(rng.integers(1, 4))
    # A framed empty scene already shows the marker and what lies outside it.
    empty_needs_field = zero_mode == "empty" and not framed
    patterned = bool(families) and (empty_needs_field or rng.random() < PATTERN_P)
    if patterned:
        for _ in range(8):
            try:
                return _build(rng, h, w, framed=framed, families=families,
                              n_iso=n_iso, zero_mode=zero_mode)
            except SceneRetry:
                continue
    for _ in range(12):
        try:
            return _build(rng, h, w, framed=framed, families=None,
                          n_iso=n_iso, zero_mode=zero_mode)
        except SceneRetry:
            continue
    raise ValueError("No scene of this plan can be drawn on this canvas")


def _build(rng, h, w, *, framed, families, n_iso, zero_mode):
    patterned = families is not None
    pattern = None
    if patterned:
        # Concentric rings need a big square-ish canvas and survive the object
        # placement least often, so they are drawn more and kept where they fit.
        weights = np.array([FAMILY_WEIGHT.get(f, 1.0) for f in families], float)
        family = str(rng.choice(families, p=weights/weights.sum()))
        # A class over this share cannot be pulled under half the canvas by the
        # objects alone, and V6 is what makes the plain reading fail.  With
        # nothing drawn on top, the field has to pass V6 by itself.
        ceiling = pf.MAX_COLOR_SHARE if (zero_mode == "empty" and not framed) else .565
        params, layers = None, None
        for _ in range(6):
            candidate = pf.sample(rng, family, h, w)
            if candidate is None:
                break
            drawn = pf.render(family, candidate, h, w)
            params, layers = candidate, drawn
            if float(pf.class_shares(drawn).max()) <= ceiling:
                break
        else:
            params = None if params is None else params
        if params is None:
            raise SceneRetry("No admissible parameters for this canvas")
        if float(pf.class_shares(layers).max()) > ceiling:
            raise SceneRetry("No balanced field of this family on this canvas")
        lab = pf.label4(layers)
        rr, cc = np.indices((h, w))
        pattern = {"family": family, "params": params, "layers": layers, "lab": lab,
                   "sizes": np.bincount(lab.ravel()),
                   "depth": np.minimum(np.minimum(rr, cc), np.minimum(h-1-rr, w-1-cc))}
        nfield = int(layers.max()) + 1
        # V6 needs the objects to pull the largest class under half the canvas.
        # A straddling body eats the two classes about evenly, so the largest
        # class loses roughly half of whatever the objects cover.
        area_share = float(min(.24, max(.12, 2.1 * (float(pf.class_shares(layers).max()) - .455))))
    else:
        nfield, area_share = 1, .43

    frame = None
    frame_area = 0
    if framed:
        def frame_length(side):
            lo = 3 if (side <= 7 or rng.random() < .20) else max(3, int(np.ceil(.50*side)))
            if patterned:
                lo = max(lo, min(side-2, max(7, int(np.ceil(.55*side)))))
            return int(rng.integers(lo, side-1)) if side-1 > lo else lo
        fh, fw = _fit_frame(frame_length(h), frame_length(w), h, w)
        seat = None
        for _ in range(1 if not patterned else 16):
            if patterned:
                fh, fw = _fit_frame(frame_length(h), frame_length(w), h, w)
            r0 = int(rng.integers(0, h-fh+1)); c0 = int(rng.integers(0, w-fw+1))
            box = (r0+1, c0+1, r0+fh-1, c0+fw-1)
            if not patterned:
                seat = (r0, c0, box)
                break
            if min(_periods_inside(pattern["family"], pattern["params"], box, h, w) or [2]) < 2:
                continue
            if len(np.unique(pattern["layers"][box[0]:box[2], box[1]:box[3]])) < 2:
                continue
            if _frame_sides_on_boundary(pattern["layers"], (r0, c0, r0+fh, c0+fw)) > 1:
                continue
            seat = (r0, c0, box)
            break
        if seat is None:
            raise SceneRetry("The marker finds no seat that leaves the field readable")
        r0, c0, target_box = seat
        frame = (r0, c0, r0+fh, c0+fw); frame_area = 2*(fh+fw)-4
    else:
        target_box = (1, 1, h-1, w-1)

    unit = 4 if patterned else 1
    minimum = 4 if patterned else None
    # Reserve the drawn marker and a clear field before selecting any object
    # count. Long frames on five-cell-wide canvases have a real limit.
    if patterned:
        # The marker covers the field too, so it pays into the same budget --
        # but the counted region must still hold drawable bodies.
        object_area = int(area_share*h*w) - frame_area
        object_area = max(object_area, int(.06*h*w))
        if object_area > int(COVER_CAP[framed]*h*w) - frame_area:
            object_area = int(COVER_CAP[framed]*h*w) - frame_area
        if object_area < 2*unit:
            raise SceneRetry("The marker leaves no room for drawable bodies")
    else:
        object_area = max(0, int(area_share*h*w)-frame_area)
        if framed:
            # A narrow canvas spends most of itself on the marker; the counted
            # region still needs the two cells of a drawable body, and never
            # more than the plain-field certificate can carry.
            object_area = min(max(object_area, 2), _plain_object_cap(h, w, frame_area))
    primitive_budget = object_area // unit
    target = _domain_plan(rng, target_box, forced_zero=zero_mode != "normal",
                          count_limit=primitive_budget, minimum=minimum)
    if target.count + n_iso > MAX_COUNT:
        n_iso = max(0, MAX_COUNT - target.count)
    if n_iso >= 4 and target.count < 2:
        n_iso = min(n_iso, 3)
    if n_iso >= 4:
        n_iso = min(n_iso, max(4, 2*target.count))
    domains = [target]
    if framed:
        boxes = _outside_boxes(h, w, frame); rng.shuffle(boxes)
        # Outside objects matter without overwhelming the counted region.
        limit = int(rng.integers(1, min(3, len(boxes))+1)) if boxes else 0
        remaining = primitive_budget-target.count
        for box in boxes[:limit]:
            outside = _domain_plan(rng, box, outside=True, count_limit=remaining, minimum=minimum)
            domains.append(outside)
            remaining -= outside.count
    try:
        parts = _paint_plan(rng, domains, framed=framed, canvas_area=h*w, frame_area=frame_area,
                            unit=unit, area_share=area_share, object_area=object_area,
                            fill_share=.85 if patterned else .47, patterned=patterned,
                            need_body=target if zero_mode == "normal" else None,
                        max_side=(max(2, int(MAX_BBOX_EXTENT*h)), max(2, int(MAX_BBOX_EXTENT*w)))
                        if patterned else None)
    except ValueError as exc:
        raise SceneRetry(str(exc)) from exc

    counted_room = _counted_room(target_box[2]-target_box[0]+2, target_box[3]-target_box[1]+2)
    if zero_mode == "normal" and target.count and counted_room and \
            _plan_counts(parts, target, 0)[("ignore", 4)] == 0:
        # A scene that is not a planned zero has something to count under EITHER
        # rule; the zero classes are drawn before the rule, never implied by it.
        # ``need_body`` above plans that body rather than waiting for a redraw,
        # so this is the safety net.  A counted region that cannot hold two
        # cells at all -- a marker on a five-by-five canvas, where the plain
        # field's plurality margin allows no bigger one -- keeps the lone cell
        # it can hold, and ``zero_class`` reports what was drawn.
        raise SceneRetry("Every body in the counted region is a lone cell")

    classes = sum(2 if p["kind"] == "different_color_edge" else 1 for p in parts)
    edge = any(p["kind"] == "different_color_edge" for p in parts)
    # An empty counted region is still a picture: inside the marker there is
    # nothing to count, and what is drawn outside it says so.
    outside_iso = int(rng.integers(1, 4)) if (framed and zero_mode == "empty" and
                                              not sum(d.count for d in domains[1:])) else 0
    dots_wanted = n_iso or outside_iso
    dot_room = 2 if dots_wanted else 0
    reserved = nfield + int(framed) + dot_room
    max_colors = min(10-reserved, classes)
    if classes and max_colors < (2 if edge else 1):
        raise SceneRetry("The palette cannot separate the field from the objects")
    ncolors = int(rng.integers(2 if edge else 1, max_colors+1)) if classes else 0
    n_dot_colors = min(dot_room, dots_wanted, 10-nfield-int(framed)-ncolors) if dots_wanted else 0
    palette = list(map(int, rng.choice(10, nfield+int(framed)+ncolors+n_dot_colors, replace=False)))
    field_colors = palette[:nfield]
    frame_color = palette[nfield] if framed else None
    object_colors = palette[nfield+int(framed):nfield+int(framed)+ncolors]
    dot_colors = palette[nfield+int(framed)+ncolors:]
    if dots_wanted and not dot_colors:
        dot_colors = object_colors or field_colors[:1]
        if not object_colors:
            raise SceneRetry("Isolated cells need a colour of their own")
    try:
        _assign_colors(rng, parts, object_colors)
    except ValueError as exc:
        raise SceneRetry(str(exc)) from exc

    field = np.array(field_colors, int)[pattern["layers"]] if patterned \
        else np.full((h, w), field_colors[0], int)
    grid = None; method = None; attempts = 0
    # Every trial retains exact footprints, colors, contacts, and all counts.
    for attempts in range(1, 9):
        placed = _place_scene(rng, h, w, field, frame, frame_color, parts,
                              free=True, pattern=pattern)
        if placed is None:
            continue
        grid, blocked = placed
        method = "free_placement"
        break
    if grid is None:
        if patterned:
            raise SceneRetry("No pattern-aware placement of the certified bodies")
        placed = _place_scene(rng, h, w, field, frame, frame_color, parts,
                              free=False, pattern=pattern)
        if placed is None:
            raise ValueError("The certified scene has no witness placement")
        grid, blocked = placed
        method = "capacity_witness"

    dot_blocked = blocked.copy()
    if frame:
        # A lone cell may sit against the marker -- the prompt says an object
        # may be adjacent to the border, and the marker is not an object.  What
        # it may not touch is another drawn body or another lone cell, which is
        # what ``blocked`` already carries.  On a narrow canvas every cell of
        # the counted region is against the marker.
        for r, c in _outline_cells(frame):
            dot_blocked[r, c] = True
    dots, isolated_kind = _place_dots(rng, grid, dot_blocked, target_box, n_iso, dot_colors)
    outside_dots = []
    if framed and dot_colors and (outside_iso or rng.random() < .5):
        for box in _outside_boxes(h, w, frame)[:1]:
            extra, _ = _place_dots(rng, grid, dot_blocked, box,
                                   outside_iso or int(rng.integers(1, 4)), dot_colors)
            outside_dots += extra
    n_iso = len(dots)
    if zero_mode == "dots" and not dots:
        raise SceneRetry("The dots-only scene lost its dots")

    planned = _plan_counts(parts, target, n_iso)
    if max(planned.values()) > MAX_COUNT:
        raise SceneRetry("The scene plan exceeds the answer cap")
    used = {int(v) for v in np.unique(grid)}
    structural = set(field_colors) | ({frame_color} if framed else set()) | set(object_colors)
    if not structural <= used or not used <= set(palette):
        raise SceneRetry("A selected palette color was lost")

    mask = grid != field
    metrics = {}
    if patterned:
        annot = pf.outline_components(grid) & mask
        metrics = pf.visibility(pattern["family"], pattern["params"], pattern["layers"],
                                mask, lab=pattern["lab"], annotations=annot)
        if not pf.visibility_ok(pattern["family"], metrics):
            raise SceneRetry("The objects hide too much of the field")
        if metrics["covered_share"] > COVER_CAP[framed]:
            raise SceneRetry("The objects cover too much of the patterned canvas")
        if int(np.bincount(grid.ravel(), minlength=10).max()) > pf.MAX_COLOR_SHARE*grid.size:
            raise SceneRetry("A colour dominates the patterned canvas")
    try:
        result = read(grid, framed=framed)
    except ValueError as exc:
        raise SceneRetry(str(exc)) from exc
    if result["style"] != ("pattern" if patterned else "plain"):
        raise SceneRetry("The finished grid does not read as the drawn background")
    if patterned and (result["pattern_family"] != pattern["family"] or
                      not np.array_equal(result["field"], field)):
        raise SceneRetry("The recovered field is not the drawn one")
    if any(result["counts"][key] != planned[key] for key in KEYS):
        raise ValueError("The certified scene disagrees with its component plan")
    if framed and list(result["frame"]) != list(frame):
        raise SceneRetry("The recovered marker is not the drawn rectangle")

    straddle_min = None
    alignment_max = 0.0
    box = (frame[0]+1, frame[1]+1, frame[2]-1, frame[3]-1) if framed else None
    if patterned:
        marker_cells = _outline_cells(frame) if framed else set()
        dot_cells = {q for q, _ in dots} | {q for q, _ in outside_dots}
        for _color, cells in components(grid, 4, mask=mask):
            if cells & marker_cells:
                continue
            if len(cells) == 1 and next(iter(cells)) in dot_cells:
                continue
            classes = {int(pattern["layers"][q]) for q in cells}
            straddle_min = len(classes) if straddle_min is None else min(straddle_min, len(classes))
        alignment_max = _alignment_max(grid, mask, pattern["layers"], skip=marker_cells)
    naive = naive_plurality_counts(grid, box=box)
    if patterned and any(naive[key] == result["counts"][key] for key in KEYS):
        raise SceneRetry("The plurality-is-background shortcut reaches the answer")
    if patterned and framed and _frame_sides_on_boundary(pattern["layers"], frame) > 1:
        raise SceneRetry("The marker hides more than one pattern boundary")

    meta = {
        "background": int(field_colors[0]), "ground_color": int(field_colors[0]),
        "background_style": "pattern" if patterned else "plain",
        "pattern_family": pattern["family"] if patterned else None,
        "pattern_params": pattern["params"] if patterned else None,
        "pattern_colors": [int(c) for c in field_colors] if patterned else [],
        "pattern_coverage": round(float(metrics.get("covered_share", 0.0)), 4) if patterned else 0.0,
        "pattern_min_visibility": round(float(min(metrics["visible_share"])), 4) if patterned else 1.0,
        "pattern_regions_uncovered": list(metrics.get("untouched_regions", [])) if patterned else [],
        "pattern_periods_visible": list(metrics.get("periods_visible", [])) if patterned else [],
        "straddle_min_colors": int(straddle_min) if straddle_min is not None else 0,
        "edge_alignment_max": round(float(alignment_max), 4),
        "object_colors": [int(c) for c in object_colors],
        "isolated_colors": sorted({int(c) for _q, c in dots + outside_dots}),
        "isolated_cells": [[int(r), int(c)] for (r, c), _col in dots + outside_dots],
        "n_isolated_cells": len(dots) + len(outside_dots),
        "n_isolated_counted": int(n_iso),
        "isolated_kind": isolated_kind if n_iso else None,
        "outside_isolated_cells": len(outside_dots),
        "zero_class": _zero_class(planned),
        "planned_count_A_4": planned[("count", 4)], "planned_count_A_8": planned[("count", 8)],
        "planned_count_B_4": planned[("ignore", 4)], "planned_count_B_8": planned[("ignore", 8)],
        "count_cap": target.count_cap, "capacity_minimum_side": target.minimum,
        "capacity_moat": 1, "planned_palette_count": len(used),
        "frame_bounds": list(frame) if frame else None, "roi_color": frame_color,
        "frame_sides_on_pattern_boundary":
            _frame_sides_on_boundary(pattern["layers"], frame) if (patterned and framed) else 0,
        "placement_method": method, "placement_attempts": attempts,
        "target_families": [f for p in parts if p["domain"] is target for f in p["families"]],
        "all_families": [f for p in parts for f in p["families"]],
        "target_diagonal_pairs": target.kinds.count("diagonal"),
        "target_different_color_pairs": target.kinds.count("different_color_edge"),
        "outside_primitives": sum(d.count for d in domains[1:]),
        "target_primitives": target.count,
        "planned_object_cells": sum(int(np.count_nonzero(p["labels"])) for p in parts),
        "object_area_share": round(float(mask.mean()), 4),
        "max_color_share": round(float(np.bincount(grid.ravel(), minlength=10).max()/grid.size), 4),
        "naive_plurality_count": naive[("count", 4)],
        "field_certificate": "one admissible background reading" if patterned
                             else "plain surrounding field after marker removal",
        "frame_candidates": result["frame_candidates"],
    }
    return grid, meta


def generate(rng, *, framed=False):
    grid, meta = scene(rng, framed=framed)
    # No construction branch, retry, size, or palette sees these two draws.
    rule = RULES[int(rng.integers(2))]
    connectivity = int(rng.choice([4, 8]))
    meta["singleton_rule"] = rule
    meta["connectivity"] = connectivity
    counts = {key: meta[PLAN_KEY[key]] for key in KEYS}
    meta["differs_under_other_rule"] = \
        counts[(rule, connectivity)] != counts[(RULES[1-RULES.index(rule)], connectivity)]
    meta["differs_under_other_connectivity"] = \
        counts[(rule, connectivity)] != counts[(rule, 12-connectivity)]
    example = {"input": grid.tolist(), "output": str(counts[(rule, connectivity)]),
               "metadata": meta}
    validate(example, framed=framed)
    return example
