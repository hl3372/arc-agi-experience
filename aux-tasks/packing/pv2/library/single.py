"""Single-grid questions: size, cells, lines, ranges, colours, positions, runs, symmetry, edits."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from pv2.library.core import (Draw, Phrasing as P, Spec, cell_rivals, colour_counts, first_appearance,
                              grid_list, inflect, pick, range_rivals)


# --- shared draws -----------------------------------------------------------------------------------

def _colour(rng, ctx, absent_p=0.0):
    counts = colour_counts(ctx.g)
    absent = [v for v in range(10) if v not in counts]
    if absent and rng.random() < absent_p:
        return pick(rng, absent)
    return pick(rng, sorted(counts))


def _colour_fmt(ctx, x, name="X"):
    return {name: ctx.cw(x), name + "c": ctx.cwc(x)}


def _window(h, w, kind, i=None, j=None):
    """(top, bottom, left, right) of what a reading question reads, inclusive: two questions of a
    part that read the same window on the same grid ask the same thing (``parts._conflicts``)."""
    if kind == "cell":
        return [i, i, j, j]
    if kind == "row":
        return [i, i, 0, w - 1]
    if kind == "col":
        return [0, h - 1, i, i]
    if kind == "rows":
        return [i, j, 0, w - 1]
    if kind == "cols":
        return [0, h - 1, i, j]
    raise ValueError(kind)


# A colour covering more than this share of the grid makes location questions trivial (its bounding
# box is the whole grid, every row contains it); counts of it are fine.
DOMINANT_SHARE = 0.60


def _minor_colour(rng, ctx, weights_power=0.0, absent_p=0.0):
    """A present colour covering at most ``DOMINANT_SHARE`` of the grid (or, with ``absent_p``, an
    absent one); None when every present colour is dominant."""
    counts = colour_counts(ctx.g)
    absent = [v for v in range(10) if v not in counts]
    if absent and rng.random() < absent_p:
        return pick(rng, absent)
    minor = [v for v in sorted(counts) if counts[v] <= DOMINANT_SHARE * ctx.g.size]
    if not minor:
        return None
    return pick(rng, minor, [counts[v] ** -weights_power for v in minor] if weights_power else None)


def _yn(flag: bool) -> tuple[str, str]:
    return ("yes", "yes") if flag else ("no", "no")


def runs_text(row) -> str:
    """A row in the solver's run notation: color@cA-cB, color@cA for a single cell."""
    parts, start = [], 0
    row = [int(v) for v in row]
    for i in range(1, len(row) + 1):
        if i == len(row) or row[i] != row[start]:
            parts.append(f"{row[start]}@c{start}" if i - start == 1 else f"{row[start]}@c{start}-c{i - 1}")
            start = i
    return " ".join(parts)


def run_bounds(line, i: int) -> tuple[int, int]:
    """First and last index of the run of equal values containing position ``i``."""
    a = b = i
    while a > 0 and line[a - 1] == line[i]:
        a -= 1
    while b < len(line) - 1 and line[b + 1] == line[i]:
        b += 1
    return a, b


# --- size ------------------------------------------------------------------------------------------

def s_size(rng, ctx):
    h, w = ctx.g.shape
    return Draw([h, w], f"{h}x{w}")


def s_rows(rng, ctx):
    h = ctx.g.shape[0]
    return Draw(h, str(h), fmt=inflect(h))


def s_cols(rng, ctx):
    w = ctx.g.shape[1]
    return Draw(w, str(w), fmt=inflect(w))


def s_n_cells(rng, ctx):
    n = int(ctx.g.size)
    return Draw(n, str(n), fmt=inflect(n))


# --- single cells --------------------------------------------------------------------------------------

def s_cell(rng, ctx):
    g = ctx.g
    h, w = g.shape
    counts = colour_counts(g)
    cands, weights = [], []
    for r in range(h):
        for c in range(w):
            v = g[r, c]
            if all(g[a, b] != v for a, b in cell_rivals(h, w, r, c)):
                cands.append((r, c))
                weights.append(counts[int(v)] ** -0.7)
    if not cands:
        return None
    r, c = pick(rng, cands, weights)
    v = int(g[r, c])
    return Draw(v, ctx.label(v), fmt={"r": r, "c": c}, params={"r": r, "c": c, "window": _window(h, w, "cell", r, c)})


def s_first_cell(rng, ctx):
    v = int(ctx.g[0, 0])
    return Draw(v, ctx.label(v), params={"window": [0, 0, 0, 0]})


def s_last_cell(rng, ctx):
    v = int(ctx.g[-1, -1])
    h, w = ctx.g.shape
    return Draw(v, ctx.label(v), params={"window": _window(h, w, "cell", h - 1, w - 1)})


CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def corner_cell(h, w, name):
    return {"top-left": (0, 0), "top-right": (0, w - 1), "bottom-left": (h - 1, 0),
            "bottom-right": (h - 1, w - 1)}[name]


def s_corner(rng, ctx):
    g = ctx.g
    h, w = g.shape
    cells = {k: corner_cell(h, w, k) for k in CORNERS}
    # the named corner must differ in colour from some other corner cell (a wrong corner is caught)
    ok = [k for k in CORNERS if any(cells[o] != cells[k] and g[cells[o]] != g[cells[k]] for o in CORNERS)]
    if not ok:
        return None
    k = pick(rng, ok)
    v = int(g[cells[k]])
    return Draw(v, ctx.label(v), fmt={"corner": k}, params={"corner": k, "window": _window(h, w, "cell", *cells[k])})


# --- lines, ranges, crops ---------------------------------------------------------------------------------

def _line_ok(lines, i):
    return all(not np.array_equal(lines[i], lines[s]) for s in range_rivals(len(lines), i, 1))


def s_row(rng, ctx):
    g = ctx.g
    if g.shape[0] < 2:
        return None
    cands = [r for r in range(g.shape[0]) if _line_ok(g, r)]
    if not cands:
        return None
    r = pick(rng, cands)
    return Draw(grid_list(g[r:r + 1]), None, fmt={"r": r}, params={"r": r, "window": _window(*g.shape, "row", r)})


def s_col(rng, ctx):
    g = ctx.g
    if g.shape[1] < 2:
        return None
    cands = [c for c in range(g.shape[1]) if _line_ok(g.T, c)]
    if not cands:
        return None
    c = pick(rng, cands)
    return Draw(grid_list(g[:, c:c + 1]), None, fmt={"c": c}, params={"c": c, "window": _window(*g.shape, "col", c)})


def _ranges(lines):
    n = len(lines)
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            if a == 0 and b == n - 1:
                continue
            span = b - a + 1
            if all(not np.array_equal(lines[a:b + 1], lines[s:s + span]) for s in range_rivals(n, a, span)):
                out.append((a, b))
    return out


def s_rows_range(rng, ctx):
    g = ctx.g
    if g.shape[0] < 3:
        return None
    ok = _ranges(g)
    if not ok:
        return None
    a, b = pick(rng, ok)
    return Draw(grid_list(g[a:b + 1]), None, fmt={"a": a, "b": b},
                params={"a": a, "b": b, "window": _window(*g.shape, "rows", a, b)})


def s_cols_range(rng, ctx):
    g = ctx.g
    if g.shape[1] < 3:
        return None
    ok = _ranges(g.T)
    if not ok:
        return None
    a, b = pick(rng, ok)
    return Draw(grid_list(g[:, a:b + 1]), None, fmt={"a": a, "b": b},
                params={"a": a, "b": b, "window": _window(*g.shape, "cols", a, b)})


def crop_rivals(h, w, a, b, c0, c1):
    """(row, col, height, width) windows a misreading of the crop lands on (the task-86 guard)."""
    kh, kw = b - a + 1, c1 - c0 + 1
    rows = [a] + range_rivals(h, a, kh)
    cols = [c0] + range_rivals(w, c0, kw)
    out = {(r, c, kh, kw) for r in rows for c in cols if (r, c) != (a, c0)}
    if c0 + kw <= h and a + kh <= w and (c0, a, kw, kh) != (a, c0, kh, kw):
        out.add((c0, a, kw, kh))
    return sorted(out)


def s_crop(rng, ctx):
    g = ctx.g
    h, w = g.shape
    if h < 2 or w < 2:
        return None
    for _ in range(60):
        a, b = sorted(int(v) for v in rng.integers(0, h, size=2))
        c0, c1 = sorted(int(v) for v in rng.integers(0, w, size=2))
        if (a, b, c0, c1) == (0, h - 1, 0, w - 1) or (a == b and c0 == c1):
            continue
        crop = g[a:b + 1, c0:c1 + 1]
        if any(np.array_equal(crop, g[r:r + kh, c:c + kw]) for r, c, kh, kw in crop_rivals(h, w, a, b, c0, c1)):
            continue
        return Draw(grid_list(crop), None, fmt={"a": a, "b": b, "c0": c0, "c1": c1},
                    params={"a": a, "b": b, "c0": c0, "c1": c1, "window": [a, b, c0, c1]})
    return None


def s_half(rng, ctx):
    g = ctx.g
    h, w = g.shape
    options = []
    if h % 2 == 0 and not np.array_equal(g[:h // 2], g[h // 2:]):
        options += ["top", "bottom"]
    if w % 2 == 0 and not np.array_equal(g[:, :w // 2], g[:, w // 2:]):
        options += ["left", "right"]
    if not options:
        return None
    half = pick(rng, options)
    part = {"top": g[:h // 2], "bottom": g[h // 2:], "left": g[:, :w // 2], "right": g[:, w // 2:]}[half]
    cut = "across the middle" if half in ("top", "bottom") else "down the middle"
    window = {"top": [0, h // 2 - 1, 0, w - 1], "bottom": [h // 2, h - 1, 0, w - 1],
              "left": [0, h - 1, 0, w // 2 - 1], "right": [0, h - 1, w // 2, w - 1]}[half]
    return Draw(grid_list(part), None, fmt={"half": half, "cut": cut}, params={"half": half, "window": window})


# --- colours -----------------------------------------------------------------------------------------------

def s_count(rng, ctx):
    x = _colour(rng, ctx, absent_p=0.1)
    n = int((ctx.g == x).sum())
    return Draw(n, str(n), fmt={**_colour_fmt(ctx, x), **inflect(n)}, params={"X": x}, asks_presence=(x,))


def s_count_not(rng, ctx):
    x = _colour(rng, ctx)
    n = int((ctx.g != x).sum())
    return Draw(n, str(n), fmt={**_colour_fmt(ctx, x), **inflect(n)}, params={"X": x}, presupposes=(x,))


def s_most(rng, ctx):
    counts = colour_counts(ctx.g)
    top = max(counts.values())
    winners = [v for v, n in counts.items() if n == top]
    if len(winners) != 1:
        return None
    return Draw(winners[0], ctx.label(winners[0]))


def s_least(rng, ctx):
    counts = colour_counts(ctx.g)
    if len(counts) < 2:
        return None
    low = min(counts.values())
    losers = [v for v, n in counts.items() if n == low]
    if len(losers) != 1:
        return None
    return Draw(losers[0], ctx.label(losers[0]))


def s_colours(rng, ctx):
    tag = "first" if ctx.names or rng.random() < 0.4 else "sorted"
    cols = first_appearance(ctx.g) if tag == "first" else sorted(colour_counts(ctx.g))
    return Draw(cols, ", ".join(ctx.label(v) for v in cols), tag=tag,
                case="one" if len(cols) == 1 else "many", params={"order": tag})


def s_n_colours(rng, ctx):
    n = len(colour_counts(ctx.g))
    return Draw(n, str(n), fmt=inflect(n))


def s_has(rng, ctx):
    x = _colour(rng, ctx, absent_p=0.5)
    present = bool((ctx.g == x).any())
    text, case = _yn(present)
    return Draw(present, text, case=case, fmt=_colour_fmt(ctx, x), params={"X": x}, asks_presence=(x,))


def s_positions(rng, ctx):
    g = ctx.g
    x = _minor_colour(rng, ctx, weights_power=1.0)
    if x is None:
        return None
    cells = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == x]
    if len(cells) <= 10:
        tag, n = "all", len(cells)
        case = "one" if n == 1 else "many"
    else:
        tag, n, case = "cap", pick(rng, (3, 4, 5, 6, 8)), "many"
        cells = cells[:n]
    return Draw([[r, c] for r, c in cells], "[" + ",".join(f"({r},{c})" for r, c in cells) + "]",
                tag=tag, case=case, fmt={**_colour_fmt(ctx, x), "n": n},
                params={"X": x, "cap": n if tag == "cap" else None}, presupposes=(x,))


def s_histogram(rng, ctx):
    counts = colour_counts(ctx.g)
    tag = "first" if ctx.names or rng.random() < 0.4 else "sorted"
    order = first_appearance(ctx.g) if tag == "first" else sorted(counts)
    return Draw([[v, counts[v]] for v in order], ", ".join(f"{ctx.label(v)}:{counts[v]}" for v in order),
                tag=tag, params={"order": tag})


def s_bbox(rng, ctx):
    g = ctx.g
    x = _minor_colour(rng, ctx)
    if x is None:
        return None
    rows = [r for r in range(g.shape[0]) if (g[r] == x).any()]
    cols = [c for c in range(g.shape[1]) if (g[:, c] == x).any()]
    box = [rows[0], cols[0], rows[-1], cols[-1]]
    return Draw(box, "({},{},{},{})".format(*box), fmt=_colour_fmt(ctx, x), params={"X": x}, presupposes=(x,))


def s_rows_containing(rng, ctx):
    g = ctx.g
    x = _minor_colour(rng, ctx)
    if x is None:
        return None
    rows = [r for r in range(g.shape[0]) if (g[r] == x).any()]
    return Draw(rows, ", ".join(str(r) for r in rows), case="one" if len(rows) == 1 else "many",
                fmt=_colour_fmt(ctx, x), params={"X": x}, presupposes=(x,))


def s_n_rows_containing(rng, ctx):
    g = ctx.g
    x = _minor_colour(rng, ctx, absent_p=0.1)
    if x is None:
        return None
    n = sum(1 for r in range(g.shape[0]) if (g[r] == x).any())
    return Draw(n, str(n), fmt={**_colour_fmt(ctx, x), **inflect(n)}, params={"X": x}, asks_presence=(x,))


def s_first_row(rng, ctx):
    g = ctx.g
    # rarer colours first (the most frequent colours nearly all start in row 0); never a dominant one
    x = _minor_colour(rng, ctx, weights_power=0.7, absent_p=0.2)
    if x is None:
        return None
    rows = [r for r in range(g.shape[0]) if (g[r] == x).any()]
    if rows:
        return Draw(rows[0], str(rows[0]), case="found", fmt=_colour_fmt(ctx, x), params={"X": x},
                    asks_presence=(x,))
    return Draw(None, "none", case="none", fmt=_colour_fmt(ctx, x), params={"X": x}, asks_presence=(x,))


def s_border(rng, ctx):
    g = ctx.g
    h, w = g.shape
    ring = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    on_ring = sorted({int(g[p]) for p in ring})
    if rng.random() < 0.85:
        x = pick(rng, on_ring)
    else:
        x = _colour(rng, ctx, absent_p=0.5)
    n = sum(1 for p in ring if g[p] == x)
    return Draw(n, str(n), fmt={**_colour_fmt(ctx, x), **inflect(n)}, params={"X": x}, asks_presence=())


def s_line_count(axis):
    def sample(rng, ctx):
        g = ctx.g if axis == "row" else ctx.g.T
        if g.shape[0] < 2:
            return None
        x = _colour(rng, ctx)
        counts = [int((line == x).sum()) for line in g]
        ok = [i for i in range(len(g)) if all(counts[s] != counts[i] for s in range_rivals(len(g), i, 1))]
        if not ok:
            return None
        i = pick(rng, ok)
        key = "r" if axis == "row" else "c"
        return Draw(counts[i], str(counts[i]), fmt={**_colour_fmt(ctx, x), **inflect(counts[i]), key: i},
                    params={"X": x, key: i}, presupposes=(x,))
    return sample


# --- runs ---------------------------------------------------------------------------------------------------

def s_runs_row(rng, ctx):
    g = ctx.g
    if g.shape[0] < 2:
        return None
    cands = [r for r in range(g.shape[0]) if _line_ok(g, r)]
    if not cands:
        return None
    r = pick(rng, cands)
    return Draw(runs_text(g[r]), runs_text(g[r]), fmt={"r": r}, params={"r": r, "window": _window(*g.shape, "row", r)})


def _run_answers(g, orient, what):
    """answer[r][c] for every cell: the run length, or where the run ends."""
    h, w = g.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            line, i = (g[r], c) if orient == "h" else (g[:, c], r)
            a, b = run_bounds(line, i)
            out[r, c] = (b - a + 1) if what == "length" else b
    return out


def s_run(what):
    def sample(rng, ctx):
        g = ctx.g
        h, w = g.shape
        orient = pick(rng, ("h", "v"))
        if (w if orient == "h" else h) < 2:
            return None
        ans = _run_answers(g, orient, what)
        cands, weights = [], []
        for r in range(h):
            for c in range(w):
                if any(ans[p] == ans[r, c] for p in cell_rivals(h, w, r, c)):
                    continue
                pos = c if orient == "h" else r
                if what == "end" and ans[r, c] == pos:
                    weight = 0.3          # the cell is its run's last one: allowed, but rarely
                else:
                    weight = 1.0
                if what == "length":
                    weight = min(int(ans[r, c]), 10) ** 0.5
                cands.append((r, c))
                weights.append(weight)
        if not cands:
            return None
        r, c = pick(rng, cands, weights)
        n = int(ans[r, c])
        fmt = {"r": r, "c": c, "dir": "horizontal" if orient == "h" else "vertical",
               "line": "row" if orient == "h" else "column", **inflect(n)}
        return Draw(n, str(n), tag=orient, fmt=fmt, params={"r": r, "c": c, "orient": orient})
    return sample


# --- whole-grid properties -------------------------------------------------------------------------------------

# Yes/no balance.  Borrowed pictures are rarely symmetric (~4% per axis), and a third of them has
# no single-colour row, so "no" would dominate.  A "no" is always kept when it is a NEAR MISS (the
# grid differs from its mirror image in at most ``NEAR_MISS_CELLS`` cells; the row is one colour
# except one cell or one run; the two rows differ in one run) and otherwise only at the rates
# below (tuned so every yes/no question lands at 40-60% yes over the build's families).  The parts
# the library builds itself also decide yes or no first and draw symmetric / near-miss pictures.
NEAR_MISS_CELLS = 8
FAR_NO_KEEP = {"symmetric": 0.05, "row_uniform": 0.10, "row_equal": 0.35}


def _runs(line):
    out = []
    for v in line.tolist():
        if not out or out[-1] != v:
            out.append(v)
    return out


def _near_uniform(line) -> bool:
    """One colour except one cell or one run (two runs, or three with equal outer colours)."""
    r = _runs(line)
    return len(r) == 2 or (len(r) == 3 and r[0] == r[2])


def _near_equal(a, b) -> bool:
    """The rows differ in one contiguous stretch of cells."""
    idx = np.flatnonzero(a != b)
    return len(idx) > 0 and idx[-1] - idx[0] + 1 == len(idx)


def s_symmetric(axis):
    def sample(rng, ctx):
        g = ctx.g
        if (g.shape[1] if axis == "lr" else g.shape[0]) < 2:
            return None                 # a single column (row) is trivially symmetric
        off = int((g != (g[:, ::-1] if axis == "lr" else g[::-1])).sum())
        if off > NEAR_MISS_CELLS and rng.random() >= FAR_NO_KEEP["symmetric"]:
            return None
        flag = off == 0
        text, case = _yn(flag)
        return Draw(flag, text, case=case, params={"mismatched_cells": off})
    return sample


def s_row_uniform(rng, ctx):
    g = ctx.g
    if g.shape[1] < 2:
        return None
    yes = [r for r in range(g.shape[0]) if len(set(g[r].tolist())) == 1]
    no = [r for r in range(g.shape[0]) if r not in yes]
    if not yes and rng.random() >= FAR_NO_KEEP["row_uniform"]:
        return None
    if yes and (not no or rng.random() < 0.5):
        r = pick(rng, yes)
    else:
        r = pick(rng, no, [3.0 if _near_uniform(g[i]) else 1.0 for i in no])
    flag = r in yes
    text, case = _yn(flag)
    return Draw(flag, text, case=case, fmt={"r": r}, params={"r": r, "window": _window(*g.shape, "row", r)})


def s_n_uniform_rows(rng, ctx):
    g = ctx.g
    if g.shape[1] < 2:
        return None
    n = sum(1 for r in range(g.shape[0]) if len(set(g[r].tolist())) == 1)
    return Draw(n, str(n), fmt=inflect(n))


def s_row_equal(rng, ctx):
    g = ctx.g
    h = g.shape[0]
    if h < 2:
        return None
    pairs = [(a, b) for a in range(h) for b in range(a + 1, h)]
    same = [p for p in pairs if np.array_equal(g[p[0]], g[p[1]])]
    diff = [p for p in pairs if not np.array_equal(g[p[0]], g[p[1]])]
    if not same and rng.random() >= FAR_NO_KEEP["row_equal"]:
        return None
    if same and (not diff or rng.random() < 0.5):
        a, b = pick(rng, same)
    else:
        a, b = pick(rng, diff, [3.0 if _near_equal(g[p[0]], g[p[1]]) else 1.0 for p in diff])
    flag = bool(np.array_equal(g[a], g[b]))
    text, case = _yn(flag)
    return Draw(flag, text, case=case, fmt={"a": a, "b": b}, params={"a": a, "b": b})


def s_n_distinct_rows(rng, ctx):
    n = len({tuple(row) for row in ctx.g.tolist()})
    return Draw(n, str(n), fmt=inflect(n))


# --- edits ----------------------------------------------------------------------------------------------------

ROTATIONS = {
    "cw": ("90 degrees clockwise", "a quarter turn clockwise", -1),
    "ccw": ("90 degrees counterclockwise", "a quarter turn counterclockwise", 1),
    "180": ("180 degrees", "a half turn", 2),
}


def s_rotate(rng, ctx):
    g = ctx.g
    ok = [d for d in ROTATIONS if not np.array_equal(np.rot90(g, ROTATIONS[d][2]), g)]
    if not ok:
        return None
    d = pick(rng, ok)
    rot, turn, k = ROTATIONS[d]
    return Draw(grid_list(np.rot90(g, k)), None, fmt={"rot": rot, "turn": turn}, params={"dir": d})


REFLECTIONS = {
    "lr": ("left to right", "every row is reversed", "(every row reversed)"),
    "tb": ("top to bottom", "the order of the rows is reversed", "(the order of the rows reversed)"),
}


def s_reflect(rng, ctx):
    g = ctx.g
    flips = {"lr": g[:, ::-1], "tb": g[::-1]}
    ok = [a for a in REFLECTIONS if not np.array_equal(flips[a], g)]
    if not ok:
        return None
    a = pick(rng, ok)
    how, effect, detail = REFLECTIONS[a]
    return Draw(grid_list(flips[a]), None, fmt={"how": how, "effect": effect, "detail": detail},
                params={"axis": a})


def s_transpose(rng, ctx):
    g = ctx.g
    if np.array_equal(g.T, g):
        return None
    return Draw(grid_list(g.T), None)


def s_recolour(rng, ctx):
    g = ctx.g
    x = _colour(rng, ctx)
    y = pick(rng, [v for v in range(10) if v != x])
    out = np.where(g == x, y, g)
    return Draw(grid_list(out), None, fmt={**_colour_fmt(ctx, x), **_colour_fmt(ctx, y, "Y")},
                params={"X": x, "Y": y}, presupposes=(x,))


def s_upscale(rng, ctx):
    g = ctx.g
    h, w = g.shape
    ks = [k for k in (2, 3) if k * h <= 30 and k * w <= 30]
    if not ks:
        return None
    k = pick(rng, ks, [3, 1][:len(ks)])
    return Draw(grid_list(np.kron(g, np.ones((k, k), dtype=int))), None, fmt={"k": k}, params={"k": k})


# --- the catalogue ---------------------------------------------------------------------------------------------

# the stage-2 rendering and the runs encoding print every row's runs: positions, runs and run
# lengths / ends would be read off almost verbatim
RUNS_VIEW = frozenset({"stage2", "runs"})

YN = lambda yes, no: {"yes": yes + ", so the answer is ⟪answer⟫.", "no": no + ", so the answer is ⟪answer⟫."}  # noqa: E731

SPECS = [
    # ---- size
    Spec("size", 1, "text", s_size, (
        P("imperative", "Give the size of ⟪g0⟫ as HxW, where H is the number of rows and W the number of columns.",
          "The size of ⟪g0⟫ is ⟪answer⟫."),
        P("interrogative", "How many rows and columns does ⟪g0⟫ have? Write the size as rows x columns, like 4x6.",
          "⟪G0⟫ is ⟪answer⟫ in rows x columns."),
        P("request", "I need the dimensions of ⟪g0⟫, written as the number of rows, then x, then the number of "
                     "columns (for example 4x6).",
          "The dimensions of ⟪g0⟫ are ⟪answer⟫."),
        P("terse", "⟪G0⟫: dimensions as rows x columns?", "⟪G0s⟫ dimensions are ⟪answer⟫."),
        P("embedded", "Look at ⟪g0⟫ and measure it, then report its height in rows and its width in columns in "
                      "the form 4x6.",
          "Measured in rows x columns, ⟪g0⟫ is ⟪answer⟫."),
    ), dims=True),
    Spec("rows", 1, "number", s_rows, (
        P("interrogative", "How many rows does ⟪g0⟫ have?", "⟪G0⟫ has ⟪answer⟫ {rows}.",
          declared="The number of rows of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the rows of ⟪g0⟫.", "⟪G0⟫ has ⟪answer⟫ {rows}.",
          declared="The row count of ⟪g0⟫ is ⟪answer⟫."),
        P("request", "Please tell me the height of ⟪g0⟫, in rows.", "The height of ⟪g0⟫ is ⟪answer⟫ {rows}.",
          declared="The height of ⟪g0⟫, in rows, is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of rows?", "The number of rows of ⟪g0⟫ is ⟪answer⟫."),
        P("embedded", "Check ⟪g0⟫ from its first row to its last and say how many rows tall it is.",
          "⟪G0⟫ is ⟪answer⟫ {rows} tall.",
          declared="Counted from its first row to its last, the height of ⟪g0⟫ in rows is ⟪answer⟫."),
    ), dims=True),
    Spec("cols", 1, "number", s_cols, (
        P("interrogative", "How many columns does ⟪g0⟫ have?", "⟪G0⟫ has ⟪answer⟫ {columns}.",
          declared="The number of columns of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the columns of ⟪g0⟫.", "⟪G0⟫ has ⟪answer⟫ {columns}.",
          declared="The column count of ⟪g0⟫ is ⟪answer⟫."),
        P("request", "Please tell me the width of ⟪g0⟫, in columns.", "The width of ⟪g0⟫ is ⟪answer⟫ {columns}.",
          declared="The width of ⟪g0⟫, in columns, is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of columns?", "The number of columns of ⟪g0⟫ is ⟪answer⟫."),
        P("embedded", "Run along any row of ⟪g0⟫ and say how many columns wide it is.",
          "⟪G0⟫ is ⟪answer⟫ {columns} wide.",
          declared="Measured along a row, the width of ⟪g0⟫ in columns is ⟪answer⟫."),
    ), dims=True),
    Spec("n_cells", 1, "number", s_n_cells, (
        P("interrogative", "How many cells does ⟪g0⟫ have in total?", "⟪G0⟫ has ⟪answer⟫ {cells} in total.",
          declared="The total number of cells of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count all the cells of ⟪g0⟫, whatever their color.", "⟪G0⟫ has ⟪answer⟫ {cells} altogether.",
          declared="The count of all the cells of ⟪g0⟫, whatever their color, is ⟪answer⟫."),
        P("terse", "⟪G0⟫: total cell count?", "The total cell count of ⟪g0⟫ is ⟪answer⟫."),
        P("embedded", "Work out the size of ⟪g0⟫ and from it tell me how many cells it holds.",
          "⟪G0⟫ holds ⟪answer⟫ {cells}.",
          declared="Worked out from its size, the number of cells ⟪g0⟫ holds is ⟪answer⟫."),
    ), dims=True, weight=0.6),

    # ---- single cells
    Spec("cell", 1, "color", s_cell, (
        P("interrogative", "What color is the cell at row {r}, column {c} of ⟪g0⟫? {ZS}",
          "The cell at row {r}, column {c} of ⟪g0⟫ is {C}⟪answer⟫."),
        P("imperative", "Name the color of ⟪g0s⟫ cell in row {r}, column {c} {ZP}.",
          "⟪G0s⟫ cell in row {r}, column {c} is {C}⟪answer⟫."),
        P("request", "I would like to know the color at row {r}, column {c} of ⟪g0⟫. {ZS}",
          "The color at row {r}, column {c} of ⟪g0⟫ is {C}⟪answer⟫."),
        P("terse", "⟪G0⟫, cell ({r},{c}) as (row,column) {ZP}: color?", "Cell ({r},{c}) of ⟪g0⟫ is {C}⟪answer⟫."),
        P("embedded", "Go down to row {r} of ⟪g0⟫, then across to column {c}, and report the color you land on. {ZS}",
          "Row {r}, column {c} of ⟪g0⟫ holds {C}⟪answer⟫."),
    ), coords=True, weight=1.5, bounds_size=True),
    Spec("first_cell", 1, "color", s_first_cell, (
        P("interrogative", "What color is the first cell of ⟪g0⟫, the one in its top-left corner?",
          "The first cell of ⟪g0⟫ is {C}⟪answer⟫."),
        P("terse", "⟪G0⟫, first cell (top-left): color?", "⟪G0s⟫ first cell is {C}⟪answer⟫."),
        P("imperative", "Report the color of the top-left cell of ⟪g0⟫.", "The top-left cell of ⟪g0⟫ is {C}⟪answer⟫."),
        P("embedded", "Start reading ⟪g0⟫ row by row from the top-left; which color do you meet first?",
          "Reading ⟪g0⟫ row by row from the top-left, the first color met is {C}⟪answer⟫."),
    ), weight=0.7),
    Spec("last_cell", 1, "color", s_last_cell, (
        P("interrogative", "What color is the last cell of ⟪g0⟫, the one in its bottom-right corner?",
          "The last cell of ⟪g0⟫ is {C}⟪answer⟫."),
        P("terse", "⟪G0⟫, last cell (bottom-right): color?", "⟪G0s⟫ last cell is {C}⟪answer⟫."),
        P("request", "Tell me the color of the bottom-right cell of ⟪g0⟫.", "The bottom-right cell of ⟪g0⟫ is {C}⟪answer⟫."),
        P("embedded", "Read ⟪g0⟫ row by row from the top-left all the way to the end; which color is the very last "
                      "cell?",
          "Reading ⟪g0⟫ row by row from the top-left, the very last cell is {C}⟪answer⟫."),
    ), weight=0.7),
    Spec("corner", 1, "color", s_corner, (
        P("interrogative", "What color is the {corner} corner cell of ⟪g0⟫?", "The {corner} corner cell of ⟪g0⟫ is {C}⟪answer⟫."),
        P("imperative", "Give the color in the {corner} corner of ⟪g0⟫.", "The color in the {corner} corner of ⟪g0⟫ is {C}⟪answer⟫."),
        P("terse", "⟪G0⟫, {corner} corner: color?", "⟪G0s⟫ {corner} corner is {C}⟪answer⟫."),
        P("request", "Could you check which color sits in the {corner} corner of ⟪g0⟫?",
          "In the {corner} corner of ⟪g0⟫ sits {C}⟪answer⟫."),
    )),

    # ---- lines, ranges, crops (reads)
    Spec("row", 1, "grid", s_row, (
        P("imperative", "Write out row {r} of ⟪g0⟫ as a grid with one row. {ZS}", "Row {r} of ⟪g0⟫ is:"),
        P("interrogative", "What does row {r} of ⟪g0⟫ look like? {ZS}", "Row {r} of ⟪g0⟫ looks like this:"),
        P("request", "Please copy row {r} {ZP} of ⟪g0⟫.", "Here is row {r} of ⟪g0⟫:"),
        P("terse", "⟪G0⟫, row {r} {ZP}?", "⟪G0s⟫ row {r} is:"),
        P("embedded", "Take ⟪g0⟫, count its rows from 0 starting at the top, and give me row {r} exactly as it "
                      "appears.",
          "Row {r} of ⟪g0⟫, counting from 0 at the top, is:"),
    ), coords=True, grid_answer=True, weight=1.3, bounds_size=True),
    Spec("col", 1, "grid", s_col, (
        P("imperative", "Write out column {c} of ⟪g0⟫ as a grid with one column, top to bottom. {ZS}",
          "Column {c} of ⟪g0⟫, top to bottom, is:"),
        P("interrogative", "What does column {c} of ⟪g0⟫ look like, read from top to bottom? {ZS}",
          "Read from top to bottom, column {c} of ⟪g0⟫ looks like this:"),
        P("request", "Please copy column {c} {ZP} of ⟪g0⟫ as a one-column grid.",
          "Here is column {c} of ⟪g0⟫ as a one-column grid:"),
        P("terse", "⟪G0⟫, column {c} {ZP}, as one column?", "⟪G0s⟫ column {c} is:"),
        P("embedded", "Take ⟪g0⟫, count its columns from 0 starting at the left, and give me column {c} as a "
                      "one-column grid.",
          "Column {c} of ⟪g0⟫, counting from 0 at the left, is:"),
    ), coords=True, grid_answer=True, weight=1.3, bounds_size=True),
    Spec("rows_range", 1, "grid", s_rows_range, (
        P("imperative", "Give rows {a} through {b} of ⟪g0⟫, both included, in their original order. {ZS}",
          "Rows {a} through {b} of ⟪g0⟫ are:"),
        P("interrogative", "What do rows {a} to {b} of ⟪g0⟫ look like, including both ends? {ZS}",
          "Rows {a} to {b} of ⟪g0⟫ look like this:"),
        P("request", "Please cut rows {a} to {b} (inclusive) out of ⟪g0⟫ and show them. {ZS}",
          "Cut out of ⟪g0⟫, rows {a} to {b} are:"),
        P("terse", "⟪G0⟫, rows {a}-{b} inclusive {ZP}?", "⟪G0s⟫ rows {a}-{b} are:"),
    ), coords=True, grid_answer=True, bounds_size=True),
    Spec("cols_range", 1, "grid", s_cols_range, (
        P("imperative", "Give columns {a} through {b} of ⟪g0⟫, both included, keeping every row. {ZS}",
          "Columns {a} through {b} of ⟪g0⟫ are:"),
        P("interrogative", "What do columns {a} to {b} of ⟪g0⟫ look like, including both ends? {ZS}",
          "Columns {a} to {b} of ⟪g0⟫ look like this:"),
        P("request", "Please cut columns {a} to {b} (inclusive) out of ⟪g0⟫, with all their rows. {ZS}",
          "Cut out of ⟪g0⟫, columns {a} to {b} are:"),
        P("terse", "⟪G0⟫, columns {a}-{b} inclusive {ZP}?", "⟪G0s⟫ columns {a}-{b} are:"),
    ), coords=True, grid_answer=True, bounds_size=True),
    Spec("crop", 1, "grid", s_crop, (
        P("imperative", "Crop ⟪g0⟫ to rows {a} through {b} and columns {c0} through {c1}; both ranges include "
                        "their ends. {ZS}",
          "⟪G0⟫ cropped to rows {a} through {b} and columns {c0} through {c1} is:"),
        P("interrogative", "Which block of cells lies where rows {a}-{b} meet columns {c0}-{c1} in ⟪g0⟫ (ranges "
                           "inclusive)? {ZS}",
          "The block where rows {a}-{b} meet columns {c0}-{c1} in ⟪g0⟫ is:"),
        P("request", "I want the rectangle of ⟪g0⟫ covering rows {a} to {b} and columns {c0} to {c1}, inclusive. {ZS}",
          "The rectangle of ⟪g0⟫ covering rows {a} to {b} and columns {c0} to {c1} is:"),
        P("terse", "⟪G0⟫, rows {a}..{b} x columns {c0}..{c1}, inclusive {ZP}?",
          "⟪G0s⟫ rows {a}..{b} x columns {c0}..{c1} are:"),
        P("embedded", "Frame the part of ⟪g0⟫ that spans rows {a} to {b} and columns {c0} to {c1} (both ends "
                      "included) and copy out what lies inside the frame. {ZS}",
          "Inside the frame over rows {a} to {b} and columns {c0} to {c1}, ⟪g0⟫ shows:"),
    ), coords=True, grid_answer=True, weight=1.3, bounds_size=True),
    Spec("half", 1, "grid", s_half, (
        P("imperative", "Give the {half} half of ⟪g0⟫.", "The {half} half of ⟪g0⟫ is:"),
        P("interrogative", "What does the {half} half of ⟪g0⟫ look like?", "The {half} half of ⟪g0⟫ looks like this:"),
        P("request", "Please cut ⟪g0⟫ in two {cut} and show me the {half} half.",
          "Cut in two {cut}, ⟪g0⟫ has this {half} half:"),
        P("terse", "⟪G0⟫, {half} half?", "⟪G0s⟫ {half} half is:"),
    ), grid_answer=True, weight=0.8, bounds_size=True),

    # ---- colours
    Spec("count", 1, "number", s_count, (
        P("interrogative", "How many {X} cells are there in ⟪g0⟫?", "There {is_are} ⟪answer⟫ {X} {cells} in ⟪g0⟫.",
          declared="The number of {X} cells in ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the cells of ⟪g0⟫ that are {X}.", "⟪G0⟫ has ⟪answer⟫ {cells} that {is_are} {X}.",
          declared="The count of cells of ⟪g0⟫ that are {X} is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of {X} cells?", "The number of {X} cells in ⟪g0⟫ is ⟪answer⟫."),
        P("request", "Please find out how many cells in ⟪g0⟫ are {X}.", "The count of {X} cells in ⟪g0⟫ comes to ⟪answer⟫."),
        P("embedded", "Scan ⟪g0⟫ and keep a tally of the {X} cells; how many are there?",
          "A scan of ⟪g0⟫ finds ⟪answer⟫ {X} {cells}.",
          declared="A tally of the {X} cells in ⟪g0⟫ comes to ⟪answer⟫."),
    ), weight=1.5),
    Spec("count_not", 1, "number", s_count_not, (
        P("interrogative", "How many cells in ⟪g0⟫ are not {X}?", "⟪G0⟫ has ⟪answer⟫ {cells} that {is_are} not {X}.",
          declared="The number of cells in ⟪g0⟫ that are not {X} is ⟪answer⟫."),
        P("imperative", "Count the cells of ⟪g0⟫ whose color is anything other than {X}.",
          "The cells of ⟪g0⟫ whose color is anything other than {X} number ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of cells that are not {X}?", "In ⟪g0⟫, ⟪answer⟫ {cells} {is_are} not {X}.",
          declared="In ⟪g0⟫, the number of cells that are not {X} is ⟪answer⟫."),
    ), weight=0.6),
    Spec("most_frequent", 1, "color", s_most, (
        P("interrogative", "Which color covers the most cells of ⟪g0⟫? Exactly one color does.",
          "The color that covers the most cells of ⟪g0⟫ is {C}⟪answer⟫."),
        P("imperative", "Name the most frequent color in ⟪g0⟫ (one color has more cells than any other).",
          "The most frequent color in ⟪g0⟫ is {C}⟪answer⟫."),
        P("terse", "⟪G0⟫: most common color (it is unique)?", "The most common color in ⟪g0⟫ is {C}⟪answer⟫."),
        P("embedded", "Tally the colors of ⟪g0⟫ and tell me which one is used most; there is a single winner.",
          "The color used most in ⟪g0⟫ is {C}⟪answer⟫."),
    ), forbid=frozenset({"sparse"})),     # the sparse listing prints the most frequent colour as its fill
    Spec("least_frequent", 1, "color", s_least, (
        P("interrogative", "Of the colors that appear in ⟪g0⟫, which one covers the fewest cells? Exactly one does.",
          "Of the colors that appear in ⟪g0⟫, the one covering the fewest cells is {C}⟪answer⟫."),
        P("imperative", "Name the rarest color present in ⟪g0⟫ (exactly one present color has fewer cells than "
                        "every other).",
          "The rarest color present in ⟪g0⟫ is {C}⟪answer⟫."),
        P("terse", "⟪G0⟫: least common color among those present (it is unique)?",
          "The least common color present in ⟪g0⟫ is {C}⟪answer⟫."),
    )),
    Spec("colours_present", 1, "list", s_colours, (
        P("imperative", "List the colors that appear in ⟪g0⟫ in increasing order, separated by commas.",
          {"many": "The colors that appear in ⟪g0⟫, in increasing order, are {Cs}⟪answer⟫.",
           "one": "The only color that appears in ⟪g0⟫ is {C}⟪answer⟫."}, tag="sorted", only="digits",
          declared="The list of colors that appear in ⟪g0⟫, in increasing order, is ⟪answer⟫."),
        P("interrogative", "Which colors does ⟪g0⟫ use? Give them in increasing order, separated by commas.",
          {"many": "In increasing order, ⟪g0⟫ uses {Cs}⟪answer⟫.", "one": "⟪G0⟫ uses only {C}⟪answer⟫."},
          tag="sorted", only="digits", declared="In increasing order, the list of colors ⟪g0⟫ uses is ⟪answer⟫."),
        P("request", "Please list the colors of ⟪g0⟫ in the order they first appear when reading row by row from "
                     "the top-left, separated by commas.",
          {"many": "In order of first appearance, the colors of ⟪g0⟫ are {Cs}⟪answer⟫.",
           "one": "⟪G0⟫ has a single color, {C}⟪answer⟫."}, tag="first",
          declared="In order of first appearance, the list of colors of ⟪g0⟫ is ⟪answer⟫."),
        P("terse", "⟪G0⟫: colors present, in order of first appearance (row by row from the top-left), "
                   "comma-separated?",
          {"many": "⟪G0s⟫ colors, in order of first appearance, are {Cs}⟪answer⟫.",
           "one": "⟪G0s⟫ only color is {C}⟪answer⟫."}, tag="first",
          declared="⟪G0s⟫ color list, in order of first appearance, is ⟪answer⟫."),
    )),
    Spec("n_colours", 1, "number", s_n_colours, (
        P("interrogative", "How many different colors appear in ⟪g0⟫?", "The number of different colors in ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the distinct colors used in ⟪g0⟫.", "⟪G0⟫ uses ⟪answer⟫ distinct {colors}.",
          declared="The count of distinct colors used in ⟪g0⟫ is ⟪answer⟫."),
        P("terse", "⟪G0⟫: how many colors?", "⟪G0⟫ has ⟪answer⟫ {colors}.",
          declared="The number of colors in ⟪g0⟫ is ⟪answer⟫."),
        P("embedded", "Go through ⟪g0⟫ noting each color the first time you see it, and tell me how many colors "
                      "that makes.",
          "⟪G0⟫ contains ⟪answer⟫ different {colors}.",
          declared="Noting each color of ⟪g0⟫ once, the number of colors comes to ⟪answer⟫."),
    )),
    Spec("has_colour", 1, "bool", s_has, (
        P("interrogative", "Does {X} occur anywhere in ⟪g0⟫?", YN("{Xc} occurs in ⟪g0⟫", "{Xc} does not occur in ⟪g0⟫"),
          declared="Asked whether {X} occurs anywhere in ⟪g0⟫, the answer is ⟪answer⟫."),
        P("interrogative", "Is there at least one {X} cell in ⟪g0⟫?",
          YN("There is at least one {X} cell in ⟪g0⟫", "There is no {X} cell in ⟪g0⟫"),
          declared="Asked whether ⟪g0⟫ has at least one {X} cell, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫: any {X} cells?", YN("⟪G0⟫ has {X} cells", "⟪G0⟫ has no {X} cells"),
          declared="The answer to whether ⟪g0⟫ has any {X} cells is ⟪answer⟫."),
        P("request", "Tell me whether {X} is used in ⟪g0⟫.", YN("{Xc} is used in ⟪g0⟫", "{Xc} is not used in ⟪g0⟫"),
          declared="On whether {X} is used in ⟪g0⟫, the answer is ⟪answer⟫."),
        P("embedded", "Look over ⟪g0⟫ for any {X} cell, and tell me whether you find one.",
          YN("There is at least one {X} cell in ⟪g0⟫", "There is no {X} cell in ⟪g0⟫"),
          declared="The answer to whether a look over ⟪g0⟫ finds any {X} cell is ⟪answer⟫."),
    )),
    Spec("positions", 1, "list", s_positions, (
        P("imperative", "List the positions of every {X} cell in ⟪g0⟫ as [(r,c),...] in row-major order (top to "
                        "bottom, then left to right). {ZS}",
          {"many": "The {X} cells of ⟪g0⟫ are at ⟪answer⟫.", "one": "The only {X} cell of ⟪g0⟫ is at ⟪answer⟫."},
          tag="all", declared="The list of positions of the {X} cells of ⟪g0⟫ is ⟪answer⟫."),
        P("interrogative", "Where are the {X} cells of ⟪g0⟫? Give every position as (row,column), in row-major order "
                           "(top to bottom, then left to right), formatted [(r,c),...]. {ZS}",
          {"many": "The {X} cells of ⟪g0⟫ are at ⟪answer⟫.", "one": "The only {X} cell of ⟪g0⟫ is at ⟪answer⟫."},
          tag="all", declared="In row-major order, the positions of the {X} cells of ⟪g0⟫ form the list ⟪answer⟫."),
        P("terse", "⟪G0⟫: all {X} positions in row-major order, as [(r,c),...] {ZP}?",
          {"many": "⟪G0s⟫ {X} positions are ⟪answer⟫.", "one": "⟪G0s⟫ only {X} position is ⟪answer⟫."}, tag="all",
          declared="⟪G0s⟫ list of {X} positions is ⟪answer⟫."),
        P("imperative", "List the positions of the first {n} of the {X} cells in ⟪g0⟫, taking the cells in row-major "
                        "order (top to bottom, then left to right), as [(r,c),...]. {ZS}",
          "The first {n} of the {X} cells of ⟪g0⟫ are at ⟪answer⟫.", tag="cap"),
        P("request", "Scanning ⟪g0⟫ in row-major order (top to bottom, then left to right), give me the positions "
                     "of the first {n} of its {X} cells as [(r,c),...]. {ZS}",
          "In row-major order, the first {n} of the {X} cells of ⟪g0⟫ are at ⟪answer⟫.", tag="cap"),
        P("terse", "⟪G0⟫: first {n} of the {X} cells in row-major order, as [(r,c),...] {ZP}?",
          "The first {n} of ⟪g0s⟫ {X} cells are at ⟪answer⟫.", tag="cap"),
    ), coords=True, forbid=RUNS_VIEW),
    Spec("histogram", 1, "text", s_histogram, (
        P("imperative", "For every color in ⟪g0⟫, give its number of cells as color:count, listing the colors in "
                        "increasing order and separating the pairs with commas.",
          "The color counts of ⟪g0⟫, in increasing color order, are ⟪answer⟫.", tag="sorted", only="digits"),
        P("interrogative", "How many cells does each color of ⟪g0⟫ have? Write color:count pairs separated by "
                           "commas, colors in increasing order.",
          "In increasing color order, the cells per color of ⟪g0⟫ are ⟪answer⟫.", tag="sorted", only="digits"),
        P("request", "Please count the cells of each color in ⟪g0⟫ and write color:count pairs separated by commas, "
                     "taking the colors in the order they first appear reading row by row from the top-left.",
          "In order of first appearance, the color counts of ⟪g0⟫ are ⟪answer⟫.", tag="first"),
        P("terse", "⟪G0⟫: color:count for each color, comma-separated, in order of first appearance (row by row "
                   "from the top-left)?",
          "⟪G0s⟫ color counts, in order of first appearance, are ⟪answer⟫.", tag="first"),
    ), weight=0.8),
    Spec("bbox", 1, "text", s_bbox, (
        P("imperative", "Give the bounding box of the {X} cells of ⟪g0⟫, the smallest rectangle containing all of "
                        "them, as (top row,left column,bottom row,right column), all inclusive. {ZS}",
          "The bounding box of the {X} cells of ⟪g0⟫ is ⟪answer⟫."),
        P("interrogative", "Which rows and columns does the smallest rectangle around all the {X} cells of ⟪g0⟫ "
                           "span? Write it as (top row,left column,bottom row,right column), inclusive. {ZS}",
          "The smallest rectangle around all the {X} cells of ⟪g0⟫ spans ⟪answer⟫."),
        P("terse", "⟪G0⟫: bounding box of {X} as (top,left,bottom,right), inclusive {ZP}?",
          "The bounding box of {X} in ⟪g0⟫ is ⟪answer⟫."),
    ), coords=True),
    Spec("rows_containing", 1, "list", s_rows_containing, (
        P("interrogative", "Which rows of ⟪g0⟫ contain at least one {X} cell? List their numbers in increasing "
                           "order, separated by commas. {ZS}",
          {"many": "The rows of ⟪g0⟫ that contain at least one {X} cell are rows ⟪answer⟫.",
           "one": "The only row of ⟪g0⟫ that contains at least one {X} cell is row ⟪answer⟫."},
          declared="The rows of ⟪g0⟫ that contain at least one {X} cell are: ⟪answer⟫."),
        P("imperative", "List, in increasing order and separated by commas, the numbers of the rows of ⟪g0⟫ that "
                        "have any {X} cell. {ZS}",
          {"many": "The rows of ⟪g0⟫ that have any {X} cell are rows ⟪answer⟫.",
           "one": "Only row ⟪answer⟫ of ⟪g0⟫ has any {X} cell."},
          declared="In increasing order, the rows of ⟪g0⟫ that have any {X} cell are: ⟪answer⟫."),
        P("terse", "⟪G0⟫: rows containing {X} {ZP}, in increasing order, comma-separated?",
          {"many": "⟪G0s⟫ rows containing {X} are rows ⟪answer⟫.",
           "one": "⟪G0s⟫ only row containing {X} is row ⟪answer⟫."},
          declared="The rows of ⟪g0⟫ containing {X} are: ⟪answer⟫."),
    ), coords=True),
    Spec("n_rows_containing", 1, "number", s_n_rows_containing, (
        P("interrogative", "How many rows of ⟪g0⟫ contain at least one {X} cell?",
          "The number of rows of ⟪g0⟫ containing at least one {X} cell is ⟪answer⟫."),
        P("imperative", "Count the rows of ⟪g0⟫ in which {X} appears.", "{Xc} appears in ⟪answer⟫ {rows} of ⟪g0⟫.",
          declared="The number of rows of ⟪g0⟫ in which {X} appears is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of rows with {X} in them?", "⟪G0⟫ has ⟪answer⟫ {rows} with {X} in {them}.",
          declared="The number of rows of ⟪g0⟫ with {X} in them is ⟪answer⟫."),
    )),
    Spec("first_row_containing", 1, "token", s_first_row, (
        P("interrogative", "Which is the first row of ⟪g0⟫, counting from the top, that contains {X}? If {X} does "
                           "not occur, the answer is none. {ZS}",
          {"found": "The first row of ⟪g0⟫, counting from the top, that contains {X} is row ⟪answer⟫.",
           "none": "No row of ⟪g0⟫ contains {X}, so the answer is ⟪answer⟫."},
          declared="The first row of ⟪g0⟫, counting from the top, that contains {X} is ⟪answer⟫."),
        P("imperative", "Give the number of the topmost row of ⟪g0⟫ with any {X} cell in it, or none if there is "
                        "no such row. {ZS}",
          {"found": "The topmost row of ⟪g0⟫ with any {X} cell in it is row ⟪answer⟫.",
           "none": "No row of ⟪g0⟫ has any {X} cell in it, so the answer is ⟪answer⟫."},
          declared="The number of the topmost row of ⟪g0⟫ with any {X} cell in it is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of the first row containing {X} {ZP}, or none if it is absent?",
          {"found": "⟪G0s⟫ first row containing {X} is row ⟪answer⟫.",
           "none": "⟪G0⟫ has no row containing {X}, so the answer is ⟪answer⟫."},
          declared="⟪G0s⟫ first row containing {X} is ⟪answer⟫."),
    ), coords=True),
    Spec("border_count", 1, "number", s_border, (
        P("interrogative", "How many {X} cells lie on the outer border of ⟪g0⟫, meaning its first and last rows and "
                           "its first and last columns (each cell counted once)?",
          "The outer border of ⟪g0⟫ holds ⟪answer⟫ {X} {cells}.",
          declared="The number of {X} cells on the outer border of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the {X} cells in the outer ring of ⟪g0⟫, that is, in its first row, last row, first "
                        "column or last column, counting each cell once.",
          "The outer ring of ⟪g0⟫ has ⟪answer⟫ {X} {cells}.",
          declared="The count of {X} cells in the outer ring of ⟪g0⟫ is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of {X} cells on the outer border (first and last rows and columns, each cell once)?",
          "⟪G0⟫ has ⟪answer⟫ {X} {cells} on its outer border.",
          declared="On the outer border of ⟪g0⟫, the number of {X} cells is ⟪answer⟫."),
    ), weight=0.8),
    Spec("row_count", 1, "number", s_line_count("row"), (
        P("interrogative", "How many {X} cells are in row {r} of ⟪g0⟫? {ZS}", "Row {r} of ⟪g0⟫ has ⟪answer⟫ {X} {cells}.",
          declared="The number of {X} cells in row {r} of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the {X} cells in row {r} {ZP} of ⟪g0⟫.", "Row {r} of ⟪g0⟫ contains ⟪answer⟫ {X} {cells}.",
          declared="The count of {X} cells in row {r} of ⟪g0⟫ is ⟪answer⟫."),
        P("terse", "⟪G0⟫, row {r} {ZP}: number of {X} cells?", "The number of {X} cells in row {r} of ⟪g0⟫ is ⟪answer⟫."),
    ), coords=True, weight=0.8, bounds_size=True),
    Spec("col_count", 1, "number", s_line_count("col"), (
        P("interrogative", "How many {X} cells are in column {c} of ⟪g0⟫? {ZS}",
          "Column {c} of ⟪g0⟫ has ⟪answer⟫ {X} {cells}.",
          declared="The number of {X} cells in column {c} of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the {X} cells in column {c} {ZP} of ⟪g0⟫.",
          "Column {c} of ⟪g0⟫ contains ⟪answer⟫ {X} {cells}.",
          declared="The count of {X} cells in column {c} of ⟪g0⟫ is ⟪answer⟫."),
        P("terse", "⟪G0⟫, column {c} {ZP}: number of {X} cells?",
          "The number of {X} cells in column {c} of ⟪g0⟫ is ⟪answer⟫."),
    ), coords=True, weight=0.8, bounds_size=True),

    # ---- runs
    Spec("runs_row", 1, "text", s_runs_row, (
        P("imperative", "Write row {r} of ⟪g0⟫ as runs: each maximal stretch of one color becomes color@cA-cB "
                        "(color@cA for a single cell), listed left to right and separated by spaces. {ZS}",
          "Row {r} of ⟪g0⟫ as runs is ⟪answer⟫."),
        P("interrogative", "What are the runs of equal color in row {r} of ⟪g0⟫, written color@cA-cB for a run over "
                           "columns A to B and color@cA for a single cell, left to right and space-separated? {ZS}",
          "The runs in row {r} of ⟪g0⟫ are ⟪answer⟫."),
        P("terse", "⟪G0⟫, row {r} {ZP} as color@cA-cB runs (color@cA for one cell), left to right?",
          "⟪G0s⟫ row {r} as runs is ⟪answer⟫."),
    ), coords=True, names_ok=False, forbid=RUNS_VIEW, bounds_size=True),
    Spec("run_length", 1, "number", s_run("length"), (
        P("interrogative", "How long is the {dir} run that contains the cell at row {r}, column {c} of ⟪g0⟫? A run "
                           "is a maximal unbroken stretch of one color within a {line}. {ZS}",
          "The {dir} run containing the cell at row {r}, column {c} of ⟪g0⟫ is ⟪answer⟫ {cells} long.",
          declared="The length in cells of the {dir} run containing the cell at row {r}, column {c} of ⟪g0⟫ is "
                   "⟪answer⟫."),
        P("imperative", "Measure, in cells, the {dir} run of one color that passes through row {r}, column {c} of "
                        "⟪g0⟫; the run stops where the color changes or the {line} ends. {ZS}",
          "The {dir} run through row {r}, column {c} of ⟪g0⟫ measures ⟪answer⟫ {cells}.",
          declared="Measured in cells, the {dir} run through row {r}, column {c} of ⟪g0⟫ has length ⟪answer⟫."),
        P("terse", "⟪G0⟫, cell ({r},{c}) as (row,column) {ZP}: length of its {dir} same-color run?",
          "The {dir} same-color run at ({r},{c}) of ⟪g0⟫ is ⟪answer⟫ {cells} long.",
          declared="The length of the {dir} same-color run at ({r},{c}) of ⟪g0⟫ is ⟪answer⟫."),
    ), coords=True, forbid=RUNS_VIEW, weight=0.8, bounds_size=True),
    Spec("run_end", 1, "number", s_run("end"), (
        P("interrogative", "In row {r} of ⟪g0⟫, the cell in column {c} is part of a run of one color. In which column "
                           "does that run end on the right? {ZS}",
          "The run through column {c} of row {r} of ⟪g0⟫ ends on the right in column ⟪answer⟫.", tag="h"),
        P("terse", "⟪G0⟫, row {r} {ZP}: last column of the same-color run covering column {c}?",
          "In row {r} of ⟪g0⟫, the same-color run covering column {c} ends in column ⟪answer⟫.", tag="h"),
        P("interrogative", "In column {c} of ⟪g0⟫, the cell in row {r} is part of a vertical run of one color. In "
                           "which row does that run end at the bottom? {ZS}",
          "The run through row {r} of column {c} of ⟪g0⟫ ends at the bottom in row ⟪answer⟫.", tag="v"),
        P("terse", "⟪G0⟫, column {c} {ZP}: last row of the same-color run covering row {r}?",
          "In column {c} of ⟪g0⟫, the same-color run covering row {r} ends in row ⟪answer⟫.", tag="v"),
    ), coords=True, forbid=RUNS_VIEW, weight=0.8, bounds_size=True),

    # ---- whole-grid properties
    Spec("symmetric_lr", 1, "bool", s_symmetric("lr"), (
        P("interrogative", "Is ⟪g0⟫ symmetric left to right, with every row reading the same backwards?",
          YN("⟪G0⟫ is symmetric left to right", "⟪G0⟫ is not symmetric left to right"),
          declared="Asked whether ⟪g0⟫ is symmetric left to right, the answer is ⟪answer⟫."),
        P("interrogative", "Does ⟪g0⟫ look exactly the same after mirroring it left to right?",
          YN("⟪G0⟫ looks exactly the same after mirroring it left to right",
             "⟪G0⟫ does not look the same after mirroring it left to right"),
          declared="The answer to whether ⟪g0⟫ looks exactly the same after mirroring it left to right is ⟪answer⟫."),
        P("request", "Tell me whether ⟪g0⟫ has a vertical mirror line down its middle, so that its left and right "
                     "sides are reflections of each other.",
          YN("⟪G0⟫ has a vertical mirror line down its middle", "⟪G0⟫ has no vertical mirror line down its middle"),
          declared="On whether ⟪g0⟫ has a vertical mirror line down its middle, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫: left-right symmetric?", YN("⟪G0⟫ is left-right symmetric", "⟪G0⟫ is not left-right symmetric"),
          declared="The answer to whether ⟪g0⟫ is left-right symmetric is ⟪answer⟫."),
    ), weight=2.0),   # most clearly asymmetric draws are declined (yes/no balance)
    Spec("symmetric_tb", 1, "bool", s_symmetric("tb"), (
        P("interrogative", "Is ⟪g0⟫ symmetric top to bottom, with its rows reading the same in reverse order?",
          YN("⟪G0⟫ is symmetric top to bottom", "⟪G0⟫ is not symmetric top to bottom"),
          declared="Asked whether ⟪g0⟫ is symmetric top to bottom, the answer is ⟪answer⟫."),
        P("interrogative", "Does ⟪g0⟫ look exactly the same after flipping it upside down (its rows in reverse "
                           "order)?",
          YN("⟪G0⟫ looks exactly the same after flipping it upside down",
             "⟪G0⟫ does not look the same after flipping it upside down"),
          declared="The answer to whether ⟪g0⟫ looks exactly the same after flipping it upside down is ⟪answer⟫."),
        P("request", "Tell me whether ⟪g0⟫ has a horizontal mirror line across its middle, so that its upper and "
                     "lower parts are reflections of each other.",
          YN("⟪G0⟫ has a horizontal mirror line across its middle",
             "⟪G0⟫ has no horizontal mirror line across its middle"),
          declared="On whether ⟪g0⟫ has a horizontal mirror line across its middle, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫: top-bottom symmetric?", YN("⟪G0⟫ is top-bottom symmetric", "⟪G0⟫ is not top-bottom symmetric"),
          declared="The answer to whether ⟪g0⟫ is top-bottom symmetric is ⟪answer⟫."),
    ), weight=2.0),   # most clearly asymmetric draws are declined (yes/no balance)
    Spec("row_uniform", 1, "bool", s_row_uniform, (
        P("interrogative", "Is row {r} of ⟪g0⟫ a single color from end to end? {ZS}",
          YN("Row {r} of ⟪g0⟫ is a single color from end to end", "Row {r} of ⟪g0⟫ is not a single color from end to end"),
          declared="Asked whether row {r} of ⟪g0⟫ is a single color from end to end, the answer is ⟪answer⟫."),
        P("request", "Tell me whether every cell in row {r} {ZP} of ⟪g0⟫ has the same color.",
          YN("Every cell in row {r} of ⟪g0⟫ has the same color", "Not every cell in row {r} of ⟪g0⟫ has the same color"),
          declared="On whether every cell in row {r} of ⟪g0⟫ has the same color, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫, row {r} {ZP}: one color only?",
          YN("Row {r} of ⟪g0⟫ has one color only", "Row {r} of ⟪g0⟫ has more than one color"),
          declared="The answer to whether row {r} of ⟪g0⟫ has one color only is ⟪answer⟫."),
    ), coords=True, weight=0.6, bounds_size=True),
    Spec("n_uniform_rows", 1, "number", s_n_uniform_rows, (
        P("interrogative", "How many rows of ⟪g0⟫ are a single color from end to end?",
          "⟪G0⟫ has ⟪answer⟫ {rows} that {is_are} a single color from end to end.",
          declared="The number of rows of ⟪g0⟫ that are a single color from end to end is ⟪answer⟫."),
        P("imperative", "Count the rows of ⟪g0⟫ whose cells all share one color.",
          "The rows of ⟪g0⟫ whose cells all share one color number ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of rows made of one color only?", "⟪G0⟫ has ⟪answer⟫ {rows} made of one color only.",
          declared="The number of rows of ⟪g0⟫ made of one color only is ⟪answer⟫."),
    ), weight=0.6),
    Spec("row_equal", 1, "bool", s_row_equal, (
        P("interrogative", "Are rows {a} and {b} of ⟪g0⟫ identical, cell for cell? {ZS}",
          YN("Rows {a} and {b} of ⟪g0⟫ are identical, cell for cell", "Rows {a} and {b} of ⟪g0⟫ are not identical"),
          declared="Asked whether rows {a} and {b} of ⟪g0⟫ are identical, cell for cell, the answer is ⟪answer⟫."),
        P("request", "Check whether row {a} and row {b} {ZP} of ⟪g0⟫ match exactly.",
          YN("Row {a} and row {b} of ⟪g0⟫ match exactly", "Row {a} and row {b} of ⟪g0⟫ do not match exactly"),
          declared="On whether row {a} and row {b} of ⟪g0⟫ match exactly, the answer is ⟪answer⟫."),
        P("terse", "⟪G0⟫: row {a} same as row {b} {ZP}?",
          YN("In ⟪g0⟫, row {a} is the same as row {b}", "In ⟪g0⟫, row {a} is not the same as row {b}"),
          declared="The answer to whether row {a} of ⟪g0⟫ is the same as row {b} is ⟪answer⟫."),
    ), coords=True, weight=0.6, bounds_size=True),
    Spec("n_distinct_rows", 1, "number", s_n_distinct_rows, (
        P("interrogative", "How many different rows does ⟪g0⟫ have, counting rows that match cell for cell as the "
                           "same?",
          "⟪G0⟫ has ⟪answer⟫ different {rows}.",
          declared="The number of different rows of ⟪g0⟫ is ⟪answer⟫."),
        P("imperative", "Count the distinct rows of ⟪g0⟫; identical rows count once.", "⟪G0⟫ has ⟪answer⟫ distinct {rows}.",
          declared="The count of distinct rows of ⟪g0⟫ is ⟪answer⟫."),
        P("terse", "⟪G0⟫: number of distinct rows?", "The number of distinct rows in ⟪g0⟫ is ⟪answer⟫."),
    ), weight=0.6),

    # ---- edits (grid answers that change the grid)
    Spec("rotate", 1, "grid", s_rotate, (
        P("imperative", "Rotate ⟪g0⟫ {rot}.", "⟪G0⟫ rotated {rot} is:"),
        P("interrogative", "What does ⟪g0⟫ look like after {turn}?", "After {turn}, ⟪g0⟫ looks like this:"),
        P("request", "Please turn ⟪g0⟫ {rot} and show the result.", "Turned {rot}, ⟪g0⟫ becomes:"),
        P("terse", "⟪G0⟫, rotated {rot}?", "⟪G0⟫ rotated {rot} is:"),
        P("embedded", "Imagine picking ⟪g0⟫ up and turning it {rot}; write out what you would then see.",
          "Turned {rot}, ⟪g0⟫ would look like this:"),
    ), edit=True, grid_answer=True),
    Spec("reflect", 1, "grid", s_reflect, (
        P("imperative", "Reflect ⟪g0⟫ {how}, so that {effect}.", "⟪G0⟫ reflected {how} is:"),
        P("interrogative", "What is ⟪g0⟫ mirrored {how} {detail}?", "⟪G0⟫ mirrored {how} is:"),
        P("request", "Please flip ⟪g0⟫ {how} {detail} and show the result.", "Flipped {how}, ⟪g0⟫ becomes:"),
        P("terse", "⟪G0⟫, mirrored {how} {detail}?", "⟪G0⟫ mirrored {how} is:"),
    ), edit=True, grid_answer=True),
    Spec("transpose", 1, "grid", s_transpose, (
        P("imperative", "Transpose ⟪g0⟫: row i of the result is column i of ⟪g0⟫, read from top to bottom.",
          "⟪G0⟫ transposed is:"),
        P("interrogative", "What do you get when you swap the rows and columns of ⟪g0⟫, so that each column, read "
                           "from top to bottom, becomes a row?",
          "With its rows and columns swapped, ⟪g0⟫ becomes:"),
        P("terse", "⟪G0⟫, transposed (column i becomes row i)?", "⟪G0⟫ transposed is:"),
    ), edit=True, grid_answer=True, weight=0.7),
    Spec("recolour", 1, "grid", s_recolour, (
        P("imperative", "Change every {X} cell of ⟪g0⟫ to {Y}. Leave every other cell unchanged.",
          "⟪G0⟫ with every {X} cell changed to {Y} is:"),
        P("interrogative", "What does ⟪g0⟫ become if each {X} cell is recolored {Y} and every other cell stays "
                           "unchanged?",
          "With each {X} cell recolored {Y}, ⟪g0⟫ becomes:"),
        P("request", "Please repaint the {X} cells of ⟪g0⟫ in {Y}, leaving every other cell unchanged.",
          "⟪G0⟫ with its {X} cells repainted in {Y} is:"),
        P("terse", "⟪G0⟫ with {X} turned into {Y}, every other cell unchanged?", "⟪G0⟫ with {X} turned into {Y} is:"),
        P("embedded", "Take ⟪g0⟫, paint each of its {X} cells {Y}, keep every other cell unchanged, and show me the "
                      "result.",
          "Painting each {X} cell {Y}, ⟪g0⟫ turns into:"),
    ), edit=True, grid_answer=True),
    Spec("upscale", 1, "grid", s_upscale, (
        P("imperative", "Scale ⟪g0⟫ up by a factor of {k}: every cell becomes a {k}x{k} block of its color.",
          "⟪G0⟫ scaled up by a factor of {k} is:"),
        P("interrogative", "What does ⟪g0⟫ look like when each cell is replaced by a {k}x{k} block of the same color?",
          "With each cell replaced by a {k}x{k} block, ⟪g0⟫ looks like this:"),
        P("terse", "⟪G0⟫, enlarged {k} times (each cell a {k}x{k} block)?", "⟪G0⟫ enlarged {k} times is:"),
    ), edit=True, grid_answer=True, weight=0.6, bounds_size=True),
]


# --- grid-writing rate (2026-09-30, DESIGN.md section 15) -----------------------------------------------
# The reads that write a grid (a row, a column, a range of rows or of columns, a crop) weigh more than
# they did (1.3 / 1.3 / 1.0 / 1.0 / 1.3), and more still on big grids (less on tiny ones), so the rows
# written are long ones: the solver's final answer is a whole grid, where one wrong cell fails.
GRID_READ_WEIGHTS = {"row": 2.2, "col": 2.0, "rows_range": 1.8, "cols_range": 1.5, "crop": 2.2}
GRID_READ_SIZE_REF = 225            # cells at which the size factor is 1 (a 15x15 grid)
GRID_READ_SIZE_CLIP = (0.6, 1.8)    # 900 cells -> 1.8 (clipped from 2.0), 100 -> 0.67, 36 or fewer -> 0.6

SPECS = [replace(s_, weight=GRID_READ_WEIGHTS[s_.qid]) if s_.qid in GRID_READ_WEIGHTS else s_ for s_ in SPECS]


def size_weighted(specs, grid) -> list:
    """``specs`` with the grid-writing reads' weights scaled by the size of the grid they would read."""
    cells = np.asarray(grid).size
    f = float(np.clip((cells / GRID_READ_SIZE_REF) ** 0.5, *GRID_READ_SIZE_CLIP))
    return [replace(s_, weight=s_.weight * f) if s_.qid in GRID_READ_WEIGHTS else s_ for s_ in specs]
