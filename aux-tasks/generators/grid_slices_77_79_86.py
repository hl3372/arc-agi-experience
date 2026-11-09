"""Tasks 77, 79 and 86 -- read rows and columns by their 0-indexed numbers.

* 77: output one requested row, or one requested column (as a one-column grid).
* 79: output a requested range of rows, or of columns, both ends included.
* 86: output the rectangle where a requested row range meets a requested
  column range, both ends included.

They are the literal coordinate-reading companions of task 80 (return a half):
the drawing is an ordinary picture and the whole skill is finding the lines the
prompt numbers and copying them exactly.

Design (September 2026):

* **Sizes**: both sides independently 5..30, the floor the user set on 49-99.
  A strip has no second row to misread and makes 77's row request an identity.
* **Pictures** come from the shared 66/70 drawing primitives, never modified
  here: single-colour and multicolour shape scenes (open or full) on all three
  tasks, plus checks, patchwork and nested frames on 79/86.  One-axis bands are
  excluded: every line across the stripes is identical, so only one axis could
  ever be asked about and the scene would reveal the axis.  On shape scenes with
  a short side of at least 8 a sparse texture from ``common/backgrounds.py`` is
  laid on the ground, kept one cell clear of the shapes.  Nothing is ignored in
  these tasks, so the dots are simply part of the picture that gets copied.
* **Misreadings are always wrong.**  The rival readings of "row r" are the
  1-indexed one (r - 1) and the two counted from the far edge (L - 1 - r and
  L - r); a range or a crop has the same rivals per axis, and a crop also has
  the transposed reading (row numbers taken as columns).  A request is only
  made when its answer differs from every rival answer that fits in the grid
  (a longer or shorter answer differs by shape).  The drawings are chosen so
  that most lines already qualify -- 77 uses only shape scenes, where the
  textures lift the qualifying share to ~85-90% -- so the guard costs little
  and the requested line is not marked out by being "the one that changes".
* **Positions**: deep lines are scarce under uniform draws (a short side cannot
  hold them), so every candidate position is weighted by its natural share to
  the power -DEPTH_GAMMA: 77's index, 79's range start and 86's per-axis start
  lean toward the far edge on long axes and stay uniform on short ones.  Ranges
  are otherwise pairs of distinct lines (drawing the length first piled starts
  up at row 0), and a WIDE_P share spans at least half the axis, so long 79
  bands and big 86 crops are not thinned by the depth lean.
* **The axis is independent of the scene.**  A scene is kept only when BOTH
  axes admit a request; the row/column coin is then fair and drawn after the
  picture is complete.  Retries redraw the same plan; nothing falls back.
* **The validator** re-reads the emitted grid with a literal pure-Python
  slicer and a separate rival check, and raises on any disagreement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import apply_background
from common.topology import dilate8
from generators.grid_content_66_70 import _draw
from registry import register

MIN_SIDE, MAX_SIDE = 5, 30
MAX_DRAWS = 64            # scene redraws of one fixed plan
MAX_REQUESTS = 80         # request draws on one finished scene (86)
TEXTURE_MIN_SIDE = 8      # the library's floor for sparse textures

FAMILIES = ("shapes", "multicolor_shapes", "checks", "patchwork", "nested_frames")
SHAPE_FAMILIES = FAMILIES[:2]
FAMILY_WEIGHTS = {
    # single lines: only shape scenes, where lines differ from their neighbours
    "77": (.50, .50, .00, .00, .00),
    # windows rarely coincide with their shifted copies, so the blocky families join
    "79": (.26, .26, .14, .17, .17),
    "86": (.26, .26, .14, .17, .17),
}
TEXTURE_P = {"77": .85, "79": .45, "86": .45}   # of shape scenes with a short side >= 8
# near-flat over 2..10 palette colours with a mild lean to fewer; the texture adds one
PALETTE_WEIGHTS = np.asarray((1.15, 1.10, 1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00))
FULL_RANGE_P = 0.02       # 79: a range covering the whole axis (the identity) stays rare
SINGLE_LINE_P = 0.08      # 86: one of the two ranges is a single row or column
# each task draws its picture from its own stream, so one seed never shows the same
# picture under two of these tasks
STREAM_SALT = {"77": 77, "79": 79, "86": 86}
# Depth.  With flat 5..30 sides and a uniform index, deep lines are scarce (index >= 20 on
# ~8% of 77, a range starting at >= 20 on ~2% of 79) because only long axes hold them.  Each
# candidate position is weighted by its natural share to the power -DEPTH_GAMMA, so the far
# lines of a long axis are asked for more often while a short axis stays uniform: 0 is the
# uniform draw, 1 a strong lean to the far edge.
DEPTH_GAMMA = 0.6
# a range spanning at least half its axis: long 79 bands and big 86 crops, which depth
# weighting alone would thin (a deep start forces a short span)
WIDE_P = {"79": 0.10, "86": 0.30}


# ---------------------------------------------------------------------------
# Rival readings
# ---------------------------------------------------------------------------

def range_rivals(length: int, start: int, span: int) -> list[int]:
    """Start positions a misreading of lines ``start .. start+span-1`` lands on.

    1-indexed from the near edge (start - 1), and counted from the far edge,
    0-indexed (length - 1 - end) and 1-indexed (length - end).  Only rivals that
    fit on the axis and differ from the request are returned.  With span 1 this
    is the rival set of a single line.
    """
    end = start + span - 1
    starts = (start - 1, length - 1 - end, length - end)
    return sorted({s for s in starts if 0 <= s <= length - span and s != start})


def crop_rivals(h: int, w: int, r0: int, kh: int, c0: int, kw: int) -> list[tuple[int, int, int, int]]:
    """Windows ``(row, col, height, width)`` a misreading of the crop lands on.

    Every combination of a row reading and a column reading (each the request
    or one of its ``range_rivals``) except the request itself, plus the
    transposed reading, where the row numbers are taken as columns and the
    column numbers as rows.
    """
    rows = [r0] + range_rivals(h, r0, kh)
    cols = [c0] + range_rivals(w, c0, kw)
    out = {(r, c, kh, kw) for r in rows for c in cols if (r, c) != (r0, c0)}
    if c0 + kw <= h and r0 + kh <= w and (c0, r0, kw, kh) != (r0, c0, kh, kw):
        out.add((c0, r0, kw, kh))
    return sorted(out)


def _lines(grid: np.ndarray, axis: str) -> np.ndarray:
    return grid if axis == "row" else grid.T


def _window_informative(lines: np.ndarray, start: int, span: int) -> bool:
    window = lines[start:start + span]
    return not any(np.array_equal(window, lines[s:s + span])
                   for s in range_rivals(len(lines), start, span))


def informative_starts(grid: np.ndarray, axis: str, span: int) -> list[int]:
    lines = _lines(grid, axis)
    return [s for s in range(len(lines) - span + 1) if _window_informative(lines, s, span)]


def _crop_informative(grid: np.ndarray, r0: int, kh: int, c0: int, kw: int) -> bool:
    crop = grid[r0:r0 + kh, c0:c0 + kw]
    h, w = grid.shape
    for r, c, a, b in crop_rivals(h, w, r0, kh, c0, kw):
        if (a, b) == (kh, kw) and np.array_equal(crop, grid[r:r + a, c:c + b]):
            return False
    return True


def _natural_shares() -> tuple[np.ndarray, np.ndarray]:
    """How often each line index / range start occurs under uniform draws on flat sides."""
    lengths = range(MIN_SIDE, MAX_SIDE + 1)
    line = np.zeros(MAX_SIDE)
    start = np.zeros(MAX_SIDE)
    for length in lengths:
        line[:length] += 1.0 / length
        n_pairs = length * (length - 1) / 2
        for a in range(length - 1):
            start[a] += (length - 1 - a) / n_pairs
    return line, start


_LINE_SHARE, _START_SHARE = _natural_shares()
LINE_WEIGHT = _LINE_SHARE ** -DEPTH_GAMMA
START_WEIGHT = np.where(_START_SHARE > 0, _START_SHARE, 1.0) ** -DEPTH_GAMMA


def _weighted(rng: np.random.Generator, items: list, weights) -> object:
    weights = np.asarray(weights, dtype=float)
    return items[int(rng.choice(len(items), p=weights / weights.sum()))]


def _depth_pair(rng: np.random.Generator, length: int) -> tuple[int, int]:
    """Two distinct lines, their start weighted by depth."""
    pairs = [(a, b) for a in range(length) for b in range(a + 1, length)]
    return _weighted(rng, pairs, [START_WEIGHT[a] for a, _ in pairs])


def _wide_range(rng: np.random.Generator, length: int, allow_full: bool) -> tuple[int, int]:
    """A range covering at least half the axis, uniformly placed."""
    span = int(rng.integers(-(-length // 2), (length if allow_full else length - 1) + 1))
    start = int(rng.integers(0, length - span + 1))
    return start, start + span - 1


# ---------------------------------------------------------------------------
# The picture
# ---------------------------------------------------------------------------

def _plan(rng: np.random.Generator, task: str) -> dict:
    h, w = (int(v) for v in rng.integers(MIN_SIDE, MAX_SIDE + 1, size=2))
    family = str(rng.choice(FAMILIES, p=np.asarray(FAMILY_WEIGHTS[task]) / sum(FAMILY_WEIGHTS[task])))
    # palette capacities as in the shared 66/70 plan: what the drawing can actually show
    if family in SHAPE_FAMILIES:
        capacity = min(10, 1 + h * w // 14)
    elif family == "nested_frames":
        capacity = min(10, (min(h, w) + 1) // 2)
    elif family == "checks":
        capacity = min(10, h + w - 1, max(2, h * w // 4 + 1))
    else:
        capacity = min(10, (h // 2) * (w // 2))
    texture = bool(family in SHAPE_FAMILIES and min(h, w) >= TEXTURE_MIN_SIDE
                   and rng.random() < TEXTURE_P[task])
    # the TOTAL colour count is what is drawn near-flat; a texture colour comes out of it,
    # so a texture on a nine-colour drawing does not pile grids up at ten
    top = min(10, capacity + int(texture))
    weights = PALETTE_WEIGHTS[:top - 1]
    total = int(rng.choice(np.arange(2, top + 1), p=weights / weights.sum()))
    texture = texture and total >= 3
    n_colors = total - int(texture)
    palette = [int(c) for c in rng.choice(10, size=n_colors, replace=False)]
    density = str(rng.choice(("open", "full"))) if family in SHAPE_FAMILIES else "structured"
    plan = {"h": h, "w": w, "family": family, "palette": palette, "density": density,
            "palette_capacity": capacity, "texture": texture}
    if family in SHAPE_FAMILIES:
        # more, smaller shapes than 66/70 on roomy canvases: busier lines to read
        minimum = n_colors - 1 if family == "shapes" else (n_colors + 1) // 3
        maximum = max(minimum, min(12, h * w // 20))
        plan["n_shapes"] = int(rng.integers(minimum, maximum + 1))
    return plan


_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_coverage": 0.0, "texture_named": False}


def _picture(rng: np.random.Generator, plan: dict) -> Optional[tuple[np.ndarray, dict]]:
    drawn = _draw(rng, plan)
    if drawn is None:
        return None
    grid, info = drawn
    grid = np.asarray(grid, dtype=int)
    if set(map(int, np.unique(grid))) != set(plan["palette"]):
        raise ValueError("tasks 77/79/86: the drawing lost a planned color")
    tex = dict(_NO_TEXTURE)
    if plan["texture"]:
        bg = plan["palette"][0]
        # one clear cell around every shape, so a dot never reads as a hair on a body
        painted = apply_background(rng, grid, bg, dilate8(grid != bg), group="sparse")
        if painted is not None:
            grid, tinfo = painted
            grid = np.asarray(grid, dtype=int)
            tex = {"texture_group": tinfo["group"], "texture_kind": tinfo["kind"],
                   "texture_colors": [int(c) for c in tinfo["texture_colors"]],
                   "texture_coverage": round(float(tinfo["coverage"]), 4), "texture_named": False}
    return grid, {**info, **tex}


def _metadata(grid: np.ndarray, plan: dict, info: dict, draws: int) -> dict:
    counts = np.bincount(grid.ravel(), minlength=10)
    shapes = info.get("shapes", [])
    meta = {
        "grid_h": plan["h"], "grid_w": plan["w"],
        "content_family": plan["family"], "density": plan["density"],
        "palette": plan["palette"], "palette_capacity": plan["palette_capacity"],
        "n_colors": int(np.count_nonzero(counts)), "color_counts": counts.tolist(),
        "base_color": plan["palette"][0],
        "n_shapes": len(shapes),
        "multicolor_shape_count": sum(len(s["colors"]) > 1 for s in shapes),
        "planned_texture": plan["texture"],
        "draws": draws,
    }
    for key in ("texture_group", "texture_kind", "texture_colors", "texture_coverage", "texture_named"):
        meta[key] = info[key]
    return meta


def _scene(rng: np.random.Generator, task: str, admissible) -> tuple[np.ndarray, dict, dict, int]:
    """Draw the fixed plan until both axes admit a request."""
    plan = _plan(rng, task)
    for draw in range(1, MAX_DRAWS + 1):
        pictured = _picture(rng, plan)
        if pictured is None:
            continue
        grid, info = pictured
        if admissible(grid, "row") and admissible(grid, "column"):
            return grid, plan, info, draw
    raise RuntimeError(f"tasks 77/79/86: no admissible picture for the plan after {MAX_DRAWS} draws: "
                       f"{ {k: plan[k] for k in ('h', 'w', 'family', 'density', 'texture')} }")


def _stream(rng: np.random.Generator, task: str) -> np.random.Generator:
    return np.random.default_rng([int(rng.integers(2 ** 63)), STREAM_SALT[task]])


def _axis(rng: np.random.Generator) -> str:
    # drawn after the picture exists, and the picture was kept only when both axes work
    return "row" if rng.random() < 0.5 else "column"


def _as_list(grid: np.ndarray) -> list[list[int]]:
    return [[int(v) for v in row] for row in grid]


# ---------------------------------------------------------------------------
# Literal validation (pure Python, independent of the numpy helpers above)
# ---------------------------------------------------------------------------

def _read(example: dict) -> list[list[int]]:
    grid = example["input"]
    h, w = len(grid), len(grid[0])
    if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
        raise ValueError("tasks 77/79/86: grid sides must be 5..30")
    if any(len(row) != w for row in grid):
        raise ValueError("tasks 77/79/86: ragged input")
    if any(not (isinstance(v, int) and 0 <= v <= 9) for row in grid for v in row):
        raise ValueError("tasks 77/79/86: a cell is outside the palette")
    meta = example["metadata"]
    if (meta["grid_h"], meta["grid_w"]) != (h, w):
        raise ValueError("tasks 77/79/86: size metadata disagrees with the grid")
    counts = [0] * 10
    for row in grid:
        for v in row:
            counts[v] += 1
    if counts != meta["color_counts"] or sum(1 for n in counts if n) != meta["n_colors"]:
        raise ValueError("tasks 77/79/86: colour statistics disagree with the grid")
    return grid


def _cut(grid, r0: int, r1: int, c0: int, c1: int) -> list[list[int]]:
    """Rows r0..r1 and columns c0..c1, both inclusive, cell by cell."""
    out = []
    for r in range(r0, r1 + 1):
        out.append([grid[r][c] for c in range(c0, c1 + 1)])
    return out


def _literal_rival_windows(h: int, w: int, r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int, int, int]]:
    """Every misreading, written out per axis: (row_start, row_end, col_start, col_end)."""
    def per_axis(length, lo, hi):
        options = [(lo, hi), (lo - 1, hi - 1), (length - 1 - hi, length - 1 - lo), (length - hi, length - lo)]
        return [(a, b) for a, b in options if 0 <= a and b <= length - 1]
    out = []
    for ra, rb in per_axis(h, r0, r1):
        for ca, cb in per_axis(w, c0, c1):
            if (ra, rb, ca, cb) != (r0, r1, c0, c1):
                out.append((ra, rb, ca, cb))
    # transposed: the row numbers read as columns and the column numbers as rows
    if c1 <= h - 1 and r1 <= w - 1:
        out.append((c0, c1, r0, r1))
    return out


def _check_rivals(grid, answer, r0, r1, c0, c1) -> None:
    h, w = len(grid), len(grid[0])
    for window in _literal_rival_windows(h, w, r0, r1, c0, c1):
        if window != (r0, r1, c0, c1) and _cut(grid, *window) == answer:
            raise ValueError(f"tasks 77/79/86: the misreading {window} gives the same answer")


def validate_77(example: dict) -> None:
    grid = _read(example)
    meta = example["metadata"]
    h, w = len(grid), len(grid[0])
    i = meta["index"]
    if meta["axis"] == "row":
        if not 0 <= i < h:
            raise ValueError("task 77: row index outside the grid")
        window = (i, i, 0, w - 1)
    elif meta["axis"] == "column":
        if not 0 <= i < w:
            raise ValueError("task 77: column index outside the grid")
        window = (0, h - 1, i, i)
    else:
        raise ValueError(f"task 77: unknown axis {meta['axis']!r}")
    answer = _cut(grid, *window)
    if example["output"] != answer:
        raise ValueError("task 77: the output is not the requested line")
    # the far-edge and 1-indexed misreadings of the one index, on the requested axis only
    length = h if meta["axis"] == "row" else w
    for k in (i - 1, length - 1 - i, length - i):
        if 0 <= k < length and k != i:
            other = _cut(grid, k, k, 0, w - 1) if meta["axis"] == "row" else _cut(grid, 0, h - 1, k, k)
            if other == answer:
                raise ValueError(f"task 77: reading line {k} gives the same answer")


def validate_79(example: dict) -> None:
    grid = _read(example)
    meta = example["metadata"]
    h, w = len(grid), len(grid[0])
    a, b = meta["start"], meta["end"]
    length = h if meta["axis"] == "row" else w
    if meta["axis"] not in ("row", "column"):
        raise ValueError(f"task 79: unknown axis {meta['axis']!r}")
    if not 0 <= a < b <= length - 1:
        raise ValueError("task 79: the range is not at least two lines inside the grid")
    window = (a, b, 0, w - 1) if meta["axis"] == "row" else (0, h - 1, a, b)
    answer = _cut(grid, *window)
    if example["output"] != answer:
        raise ValueError("task 79: the output is not the requested range")
    for s in (a - 1, length - 1 - b, length - b):
        e = s + (b - a)
        if 0 <= s and e <= length - 1 and s != a:
            other = _cut(grid, s, e, 0, w - 1) if meta["axis"] == "row" else _cut(grid, 0, h - 1, s, e)
            if other == answer:
                raise ValueError(f"task 79: reading lines {s}..{e} gives the same answer")


def validate_86(example: dict) -> None:
    grid = _read(example)
    meta = example["metadata"]
    h, w = len(grid), len(grid[0])
    r0, r1, c0, c1 = meta["row_start"], meta["row_end"], meta["col_start"], meta["col_end"]
    if not (0 <= r0 <= r1 <= h - 1 and 0 <= c0 <= c1 <= w - 1):
        raise ValueError("task 86: the rectangle is not inside the grid")
    if r0 == r1 and c0 == c1:
        raise ValueError("task 86: a single cell is not a rectangle of ranges")
    if (r0, r1, c0, c1) == (0, h - 1, 0, w - 1):
        raise ValueError("task 86: the whole grid is not a crop")
    answer = _cut(grid, r0, r1, c0, c1)
    if example["output"] != answer:
        raise ValueError("task 86: the output is not the requested rectangle")
    _check_rivals(grid, answer, r0, r1, c0, c1)


# ---------------------------------------------------------------------------
# The examples
# ---------------------------------------------------------------------------

def generate_line(rng: np.random.Generator) -> dict:
    """Task 77: one row or one column."""
    rng = _stream(rng, "77")
    grid, plan, info, draws = _scene(rng, "77", lambda g, ax: bool(informative_starts(g, ax, 1)))
    axis = _axis(rng)
    good = informative_starts(grid, axis, 1)
    index = int(_weighted(rng, good, LINE_WEIGHT[good]))
    length = grid.shape[0] if axis == "row" else grid.shape[1]
    line = grid[index:index + 1, :] if axis == "row" else grid[:, index:index + 1]
    values = line.ravel()
    meta = _metadata(grid, plan, info, draws)
    meta.update(
        axis=axis, index=index, line_length=int(len(values)),
        output_h=int(line.shape[0]), output_w=int(line.shape[1]),
        rival_indices=range_rivals(length, index, 1), depth_gamma=DEPTH_GAMMA,
        informative_lines=len(good), informative_share=round(len(good) / length, 4),
        line_n_colors=int(len(set(values.tolist()))),
        line_monochrome=bool(len(set(values.tolist())) == 1),
    )
    example = {"input": _as_list(grid), "output": _as_list(line), "metadata": meta}
    validate_77(example)
    return example


def _informative_pairs(grid: np.ndarray, axis: str) -> list[tuple[int, int]]:
    lines = _lines(grid, axis)
    length = len(lines)
    return [(a, b) for a in range(length) for b in range(a + 1, length)
            if (a, b) != (0, length - 1) and _window_informative(lines, a, b - a + 1)]


def _admits_range(grid: np.ndarray, axis: str) -> bool:
    """Some range of at least two lines (not the whole axis) on this axis beats every misreading.

    A picture that is constant along the axis (a two-patch split, one thin ring
    edge) admits none, and neither does its crop along that axis."""
    lines = _lines(grid, axis)
    length = len(lines)
    return any(_window_informative(lines, a, b - a + 1)
               for a in range(length) for b in range(a + 1, length) if (a, b) != (0, length - 1))


def generate_range(rng: np.random.Generator) -> dict:
    """Task 79: rows a..b or columns a..b, at least two lines, both ends included."""
    rng = _stream(rng, "79")
    grid, plan, info, draws = _scene(rng, "79", _admits_range)
    axis = _axis(rng)
    length = grid.shape[0] if axis == "row" else grid.shape[1]
    good = _informative_pairs(grid, axis)
    wide = [(a, b) for a, b in good if b - a + 1 >= -(-length // 2)]
    roll = rng.random()
    if roll < FULL_RANGE_P:
        mode = "full"
        start, end = 0, length - 1            # the identity: no misreading fits on the axis
    elif roll < FULL_RANGE_P + WIDE_P["79"] and wide:
        mode = "wide"
        start, end = wide[int(rng.integers(len(wide)))]
    else:
        mode = "depth"
        start, end = _weighted(rng, good, [START_WEIGHT[a] for a, _ in good])
    span = end - start + 1
    window = grid[start:end + 1, :] if axis == "row" else grid[:, start:end + 1]
    meta = _metadata(grid, plan, info, draws)
    n_pairs = length * (length - 1) // 2 - 1
    meta.update(
        axis=axis, start=start, end=end, span=span, axis_length=length,
        full_range=bool(span == length), range_mode=mode, depth_gamma=DEPTH_GAMMA,
        output_h=int(window.shape[0]), output_w=int(window.shape[1]),
        rival_starts=range_rivals(length, start, span),
        informative_pairs=len(good), informative_share=round(len(good) / n_pairs, 4),
    )
    example = {"input": _as_list(grid), "output": _as_list(window), "metadata": meta}
    validate_79(example)
    return example


def _crop_axis(rng: np.random.Generator, length: int, single: bool) -> tuple[int, int, str]:
    if single:
        i = int(_weighted(rng, list(range(length)), LINE_WEIGHT[:length]))
        return i, i, "single"
    if rng.random() < WIDE_P["86"]:
        return (*_wide_range(rng, length, allow_full=True), "wide")
    return (*_depth_pair(rng, length), "depth")


def _crop_ranges(rng: np.random.Generator, h: int, w: int) -> tuple[int, int, int, int, str, str]:
    """(row_start, row_end, col_start, col_end, row_mode, col_mode): per axis a
    depth-weighted pair or a wide range, one axis occasionally a single line,
    never the whole grid."""
    while True:
        single = rng.random() < SINGLE_LINE_P
        which = int(rng.integers(2))
        r0, r1, row_mode = _crop_axis(rng, h, single and which == 0)
        c0, c1, col_mode = _crop_axis(rng, w, single and which == 1)
        if (r0, r1, c0, c1) != (0, h - 1, 0, w - 1):
            return r0, r1, c0, c1, row_mode, col_mode


def generate_crop(rng: np.random.Generator) -> dict:
    """Task 86: the rectangle where rows r0..r1 meet columns c0..c1."""
    rng = _stream(rng, "86")
    grid, plan, info, draws = _scene(rng, "86", _admits_range)
    h, w = grid.shape
    found = None
    # the proposal restricted to informative requests, by rejection: sample, test, keep
    for tries in range(1, MAX_REQUESTS * 12 + 1):
        r0, r1, c0, c1, row_mode, col_mode = _crop_ranges(rng, h, w)
        if _crop_informative(grid, r0, r1 - r0 + 1, c0, c1 - c0 + 1):
            found = (r0, r1 - r0 + 1, c0, c1 - c0 + 1)
            break
    if found is None:
        raise RuntimeError("task 86: no informative rectangle on the picture")
    r0, kh, c0, kw = found
    crop = grid[r0:r0 + kh, c0:c0 + kw]
    meta = _metadata(grid, plan, info, draws)
    meta.update(
        row_start=r0, row_end=r0 + kh - 1, col_start=c0, col_end=c0 + kw - 1,
        output_h=kh, output_w=kw,
        full_height=bool(kh == h), full_width=bool(kw == w),
        single_row=bool(kh == 1), single_column=bool(kw == 1),
        row_mode=row_mode, col_mode=col_mode, depth_gamma=DEPTH_GAMMA,
        n_rivals=len(crop_rivals(h, w, r0, kh, c0, kw)),
        crop_n_colors=int(len(np.unique(crop))),
        crop_monochrome=bool(len(np.unique(crop)) == 1),
        request_tries=tries,
    )
    example = {"input": _as_list(grid), "output": _as_list(crop), "metadata": meta}
    validate_86(example)
    return example


@register("77")
def generate_return_line(rng: np.random.Generator) -> dict:
    """Task 77: output one 0-indexed row or column of the grid."""
    return generate_line(rng)


@register("79")
def generate_return_line_range(rng: np.random.Generator) -> dict:
    """Task 79: output a 0-indexed range of rows or columns, both ends included."""
    return generate_range(rng)


@register("86")
def generate_return_intersection(rng: np.random.Generator) -> dict:
    """Task 86: output where a row range intersects a column range."""
    return generate_crop(rng)
