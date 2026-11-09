"""Slow literal reader for localized foregrounds; no generator helpers."""
import itertools
import re

import numpy as np


def _connected(cells):
    remaining = set(cells)
    if not remaining:
        return False
    todo = [remaining.pop()]
    while todo:
        r, c = todo.pop()
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if q in remaining:
                remaining.remove(q)
                todo.append(q)
    return not remaining


def _pattern(a, removed):
    h, w = a.shape
    visible = [(r, c) for r in range(h) for c in range(w) if (r, c) not in removed]
    if len({a[p] for p in visible}) < 2:
        return False
    # Necessary shift checks prune periods; literal tile reconstruction below
    # still examines every class, including cells far beyond a mask.
    ps = [p for p in range(1, h//2+1) if all(
        (r+p, c) in removed or a[r, c] == a[r+p, c] for r, c in visible if r+p < h)]
    qs = [q for q in range(1, w//2+1) if all(
        (r, c+q) in removed or a[r, c] == a[r, c+q] for r, c in visible if c+q < w)]
    for p, q in itertools.product(ps, qs):
        tile, count = {}, {}
        for r, c in visible:
            key = r % p, c % q
            if key in tile and tile[key] != a[r, c]:
                break
            tile[key] = a[r, c]
            count[key] = count.get(key, 0)+1
        else:
            if len(tile) == p*q and min(count.values()) >= 2:
                return True
    for symmetry in range(4):
        checked = set()
        for r in range(h):
            valid = True
            for c in range(w):
                if (r, c) in checked:
                    continue
                orbit = {(r, c)}
                if symmetry in (0, 2):
                    orbit.add((h-1-r, c))
                if symmetry in (1, 2):
                    orbit.add((r, w-1-c))
                if symmetry in (2, 3):
                    orbit.add((h-1-r, w-1-c))
                checked.update(orbit)
                values = {a[p] for p in orbit if p not in removed}
                if len(values) != 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return True
    return False


def outputs(grid, canvas):
    a = np.asarray(grid)
    h, w = a.shape
    groups = {color: set(map(tuple, np.argwhere(a == color))) for color in np.unique(a)}
    eligible = []
    for color, cells in groups.items():
        ys, xs = zip(*cells)
        if max(ys)-min(ys)+1 < h and max(xs)-min(xs)+1 < w:
            eligible.append(color)
    answers = {}
    for n in range(1, min(len(eligible), len(groups)-2)+1):
        for chosen in itertools.combinations(eligible, n):
            cells = set().union(*(groups[c] for c in chosen))
            ys, xs = zip(*cells)
            if max(ys)-min(ys)+1 == h or max(xs)-min(xs)+1 == w:
                continue
            if not _connected(cells) or not _pattern(a, cells):
                continue
            out = [[a[r, c].item() if (r, c) in cells else canvas for c in range(w)] for r in range(h)]
            answers[tuple(map(tuple, out))] = cells
    return answers


def solve(grid, prompt):
    canvas = re.search(r"every other output cell with color ([0-9A-Z])", prompt).group(1)
    if isinstance(grid[0][0], int):
        canvas = int(canvas)
    answers = outputs(grid, canvas)
    if len(answers) != 1:
        raise ValueError(f"Expected one extraction, got {len(answers)}")
    return list(map(list, next(iter(answers))))
