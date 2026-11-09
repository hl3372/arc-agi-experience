"""Readable parallel runs and full bands, with positive gaps only (73/74)."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from generators.geometry_fields_71_74 import field_metadata, field_plan, render_field


ORIENTATIONS = ("h", "v", "diag_down", "diag_up")
SIGNS = {
    "h": {"above": -1, "below": 1},
    "v": {"left": -1, "right": 1},
    "diag_down": {"up-right": -1, "down-left": 1},
    "diag_up": {"up-left": -1, "down-right": 1},
}


def band_cells(h, w, orientation, key):
    # The builder uses endpoint/coordinate enumeration; the validator below
    # reconstructs whole monochrome runs directly from the observed positions.
    if orientation == "h":
        return [(key, c) for c in range(w)] if 0 <= key < h else []
    if orientation == "v":
        return [(r, key) for r in range(h)] if 0 <= key < w else []
    if orientation == "diag_down":
        return [(r, r - key) for r in range(h) if 0 <= r - key < w]
    return [(r, key - r) for r in range(h) if 0 <= key - r < w]


def _bands(h, w, orientation):
    rr, cc = np.indices((h, w))
    return {"h": rr, "v": cc, "diag_down": rr - cc, "diag_up": rr + cc}[orientation]


def _run(points, orientation):
    if len(points) < 4:
        return False
    values = sorted(c if orientation == "h" else r for r, c in points)
    return values == list(range(values[0], values[0] + len(values)))


def line_readings(grid, *, full=False):
    """All whole-color single runs and same-color parallel pairs.

    Broad field regions cannot count as one-cell-wide lines. A reused field
    color is examined globally, so a one-row remnant of the plain field above
    a line is not incorrectly split off and treated as another foreground line.
    """
    a = np.asarray(grid)
    singles, pairs = [], []
    for color in np.unique(a):
        points = list(zip(*np.nonzero(a == color)))
        for orientation in ORIENTATIONS:
            intercept = lambda r, c: r if orientation == "h" else c if orientation == "v" else r - c if orientation == "diag_down" else r + c
            groups = {}
            for r, c in points:
                groups.setdefault(int(intercept(r, c)), []).append((int(r), int(c)))
            if len(groups) not in (1, 2):
                continue
            if any(not _run(p, orientation) for p in groups.values()):
                continue
            if full and any(len(p) != int((_bands(*a.shape, orientation) == k).sum()) for k, p in groups.items()):
                continue
            entry = (orientation, tuple(sorted(groups)), str(color) if a.dtype.kind in "USO" else int(color), tuple(tuple(groups[k]) for k in sorted(groups)))
            if len(groups) == 1:
                singles.append(entry)
            elif abs(entry[1][1] - entry[1][0]) > 1:
                pairs.append(entry)
    return singles, pairs


def gap_readings(grid):
    singles, pairs = line_readings(grid)
    readings = {}
    for orientation, keys, color, points in pairs:
        readings.setdefault(keys[1] - keys[0] - 1, []).append((orientation, keys, (color, color), points))
    for left, right in combinations(singles, 2):
        if left[0] != right[0] or left[2] == right[2]:
            continue
        gap = abs(left[1][0] - right[1][0]) - 1
        if gap > 0:
            readings.setdefault(gap, []).append((left[0], (left[1][0], right[1][0]), (left[2], right[2]), (left[3][0], right[3][0])))
    return readings


def solve_parallel_gap(grid):
    readings = gap_readings(grid)
    if len(readings) != 1:
        raise ValueError("Task 73: the visible parallel lines do not give one positive gap")
    return str(next(iter(readings)))


def validate_parallel_gap(grid, output):
    if solve_parallel_gap(grid) != str(output):
        raise ValueError("Task 73: visible line gap disagrees with the answer")


def source_line(grid):
    singles, _ = line_readings(grid, full=True)
    if len(singles) != 1:
        raise ValueError("Task 74: expected one inferable full source line")
    return singles[0]


def solve_draw_parallel(grid, new_color, gap, direction):
    if gap < 1:
        raise ValueError("Task 74 uses positive gaps")
    orientation, (key,), _, _ = source_line(grid)
    if direction not in SIGNS[orientation]:
        raise ValueError("Direction is not transverse to the source line")
    new_key = key + SIGNS[orientation][direction] * (gap + 1)
    a = np.asarray(grid).copy()
    where = _bands(*a.shape, orientation) == new_key
    if int(where.sum()) < 4:
        raise ValueError("The requested parallel band is not readable inside the grid")
    a[where] = new_color
    return a.tolist()


def validate_draw_parallel(grid, output, new_color, gap, direction):
    if solve_draw_parallel(grid, new_color, gap, direction) != output:
        raise ValueError("Task 74: source line and prompt do not reproduce the output")


def _legible_bands(h, w, orientation):
    if orientation == "h":
        keys = range(h)
    elif orientation == "v":
        keys = range(w)
    elif orientation == "diag_down":
        keys = range(4 - w, h - 3)
    else:
        keys = range(3, h + w - 4)
    return {k: band_cells(h, w, orientation, k) for k in keys}


def _choose_gap(rng, values):
    modest = [g for g in values if g <= 10]
    longer = [g for g in values if g > 10]
    pool = longer if longer and (not modest or rng.random() < .35) else modest
    return int(rng.choice(pool))


def _segment(rng, cells, *, shorten=False):
    cap = len(cells) - int(shorten)
    length = int(rng.integers(max(4, len(cells) // 3), cap + 1))
    start = int(rng.integers(len(cells) - length + 1))
    return cells[start:start + length]


def _tangent(point, orientation):
    r, c = point
    return c if orientation == "h" else r if orientation == "v" else r + c if orientation == "diag_down" else r - c


def _staggered_orders(first, second, orientation, gap):
    # With horizontal/vertical lines any longitudinal separation changes the
    # nearest-cell distance. For a 45-degree pair it must exceed the band offset:
    # L1 distance is max(abs(delta(row+col)), abs(delta(row-col))).
    separation = 1 if orientation in ("h", "v") else gap + 2
    lines = [first, second]
    orders = []
    for early, late in ((0, 1), (1, 0)):
        low = _tangent(lines[early][3], orientation)
        high = _tangent(lines[late][-4], orientation) - separation
        if low <= high:
            orders.append((early, late, low, high, separation))
    return orders


def generate_measure_parallel(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    orientation = str(rng.choice(ORIENTATIONS))
    bands = _legible_bands(h, w, orientation)
    all_pairs = [(a, b) for a, b in combinations(bands, 2) if b - a > 1]
    finite_pairs = [(a, b) for a, b in all_pairs if max(len(bands[a]), len(bands[b])) > 4]
    finite = bool(finite_pairs and rng.random() < .45)
    staggered_pairs = {}
    if finite:
        for a, b in finite_pairs:
            orders = _staggered_orders(bands[a], bands[b], orientation, b - a - 1)
            if orders:
                staggered_pairs[(a, b)] = orders
    staggered = bool(staggered_pairs and rng.random() < .35)
    pairs = list(staggered_pairs) if staggered else finite_pairs if finite else all_pairs
    gaps = sorted({b - a - 1 for a, b in pairs})
    gap = _choose_gap(rng, gaps)
    choices = [(a, b) for a, b in pairs if b - a - 1 == gap]
    k1, k2 = choices[int(rng.integers(len(choices)))]
    lines = [bands[k1], bands[k2]]
    if staggered:
        orders = staggered_pairs[(k1, k2)]
        early, late, low, high, separation = orders[int(rng.integers(len(orders)))]
        cut = int(rng.integers(low, high + 1))
        prefix = [point for point in lines[early] if _tangent(point, orientation) <= cut]
        suffix = [point for point in lines[late] if _tangent(point, orientation) >= cut + separation]
        lines[early] = _segment(rng, prefix)
        lines[late] = _segment(rng, suffix)
    elif finite:
        must = int(rng.choice([i for i in range(2) if len(lines[i]) > 4]))
        lines = [_segment(rng, cells, shorten=(i == must)) for i, cells in enumerate(lines)]
    same_color = bool(rng.random() < .30)
    occupied = np.zeros((h, w), bool)
    for points in lines:
        for r, c in points:
            occupied[r, c] = True
    kind, n_accents = field_plan(rng, h, w, free=~occupied)
    colors = list(map(int, rng.choice(10, size=(2 if same_color else 3) + n_accents, replace=False)))
    bg = colors[0]
    line_colors = [colors[1], colors[1] if same_color else colors[2]]
    accents = colors[2 if same_color else 3:]
    for attempts in range(1, 101):
        field = render_field(rng, h, w, bg, accents, kind, avoid=occupied)
        if field is None:
            continue
        grid = field.copy()
        for points, color in zip(lines, line_colors):
            for r, c in points:
                grid[r, c] = color
        if set(np.unique(grid)) != {bg, *line_colors, *accents}:
            continue
        readings = gap_readings(grid)
        if set(readings) != {gap} or sum(map(len, readings.values())) != 1:
            continue
        answer = str(gap)
        validate_parallel_gap(grid, answer)
        nearest = min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in lines[0] for r2, c2 in lines[1])
        return {
            "input": grid.tolist(), "output": answer,
            "metadata": {
                "grid_h": h, "grid_w": w, "n_colors": len(np.unique(grid)),
                "planned_n_colors": (2 if same_color else 3) + n_accents,
                "bg_color": bg, "line_color_1": line_colors[0], "line_color_2": line_colors[1],
                "orientation": orientation, "family": "segments" if finite else "full_bands",
                "n_lines": 2, "same_color": same_color, "distance": gap,
                "op_endpoint_separation": staggered,
                "endpoint_separation_matters": nearest > gap + 1,
                "gap_cap": max(gaps), "line_lengths": [len(p) for p in lines],
                "line_endpoints": [[list(p[0]), list(p[-1])] for p in lines],
                "band_indices": [k1, k2], "min_visible_run": 4,
                "foreground_fraction": sum(map(len, lines)) / (h * w),
                "attempts": attempts, **field_metadata(grid, bg, kind, accents),
            },
        }
    raise RuntimeError("Task 73: the fixed line/field/color plan is not uniquely readable")


def generate_draw_parallel(rng):
    h, w = map(int, rng.integers(5, 31, size=2))
    orientation = str(rng.choice(ORIENTATIONS))
    bands = _legible_bands(h, w, orientation)
    lo, hi = min(bands), max(bands)
    two_way_cap = (hi - lo) // 2 - 1
    two_way = bool(two_way_cap >= 1 and rng.random() < .55)
    gap_cap = two_way_cap if two_way else hi - lo - 1
    gap = _choose_gap(rng, list(range(1, gap_cap + 1)))
    offset = gap + 1
    keys = [k for k in bands if int(k - offset in bands) + int(k + offset in bands) == (2 if two_way else 1)]
    old_key = int(rng.choice(keys))
    occupied = np.zeros((h, w), bool)
    for r, c in bands[old_key]:
        occupied[r, c] = True
    kind, n_accents = field_plan(rng, h, w, free=~occupied)
    colors = list(map(int, rng.choice(10, size=3 + n_accents, replace=False)))
    bg, original, new = colors[:3]
    accents = colors[3:]
    for attempts in range(1, 101):
        field = render_field(rng, h, w, bg, accents, kind, avoid=occupied)
        if field is None:
            continue
        inp = field.copy()
        for r, c in bands[old_key]:
            inp[r, c] = original
        if set(np.unique(inp)) != {bg, original, *accents}:
            continue
        try:
            source = source_line(inp)
        except ValueError:
            continue
        if source[0] != orientation or source[1] != (old_key,):
            continue
        # The input is now complete. On two-way scenes this last independent
        # draw selects the instruction, so the input cannot reveal its sign.
        targets = [k for k in (old_key - offset, old_key + offset) if k in bands]
        new_key = int(rng.choice(targets))
        sign = 1 if new_key > old_key else -1
        direction = next(name for name, value in SIGNS[orientation].items() if value == sign)
        out = inp.copy()
        for r, c in bands[new_key]:
            out[r, c] = new
        output = out.tolist()
        validate_draw_parallel(inp, output, new, gap, direction)
        return {
            "input": inp.tolist(), "output": output,
            "metadata": {
                "grid_h": h, "grid_w": w, "n_colors": len(np.unique(inp)),
                "planned_n_colors": 2 + n_accents,
                "bg_color": bg, "original_color": original, "new_color": new,
                "orientation": orientation, "family": "full_bands", "distance": gap,
                "direction": direction, "gap_cap": gap_cap,
                "two_way_feasible": two_way, "old_band": old_key, "new_band": new_key,
                "line_lengths": [len(bands[old_key]), len(bands[new_key])],
                "min_visible_run": 4, "attempts": attempts,
                "foreground_fraction": len(bands[old_key]) / (h * w),
                "output_foreground_fraction": (len(bands[old_key]) + len(bands[new_key])) / (h * w),
                "background_outside_preserved": bool(np.array_equal(
                    inp[_bands(h, w, orientation) != new_key],
                    out[_bands(h, w, orientation) != new_key])),
                **field_metadata(inp, bg, kind, accents),
            },
        }
    raise RuntimeError("Task 74: the fixed source/field/color plan is not uniquely readable")
