"""Both endpoint runs on a varied, independently validated simple path."""
from __future__ import annotations

import numpy as np

from generators import path_routes
from generators.path_routes import route as _route
from generators.scene_shapes_56_60 import decorate

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5
# The six templated shapes keep their proportions to one another; the four
# irregular routes (2026-09-15 feedback: "we also need more random paths")
# take 40% of the scenes between them.
FAMILY_WEIGHTS = path_routes.shares((.10, .10, .12, .14, .32, .22))
BOX_FLOORS = {"straight": 1, "elbow": 2, "U": 3, "staircase": 2, "serpentine": 3,
              "spiral": 5, **path_routes.IRREGULAR_FLOORS}


def trace(grid, color):
    a = np.asarray(grid)
    cells = set(map(tuple, np.argwhere(a == color)))
    adjacent = {p: [q for q in ((p[0]-1, p[1]), (p[0]+1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)) if q in cells] for p in cells}
    ends = sorted(p for p, ns in adjacent.items() if len(ns) == 1)
    if len(ends) != 2 or any(len(ns) not in (1, 2) for ns in adjacent.values()):
        raise ValueError("The displayed path must have two ends and no branches")
    ordered = [ends[0]]
    seen = set(ordered)
    while ordered[-1] != ends[1]:
        following = [p for p in adjacent[ordered[-1]] if p not in seen]
        if len(following) != 1:
            raise ValueError("The displayed path loops or is disconnected")
        ordered.append(following[0])
        seen.add(following[0])
    if seen != cells:
        raise ValueError("Extra path-colored cells outside the traced path")
    return ordered


def solve(grid, color, n):
    ordered = trace(grid, color)
    if not 1 <= n <= len(ordered)//2:
        raise ValueError("Endpoint runs must fit without overlapping")
    def fmt(cells):
        return "["+",".join(f"({r},{c})" for r, c in cells)+"]"
    return "first:"+fmt(ordered[:n])+", second:"+fmt(ordered[::-1][:n])


def validate_example(ex):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("Invalid path canvas")
    if solve(a, m["path_color"], m["n"]) != ex["output"]:
        raise ValueError("Path-end output disagrees with visible traversal")
    if len(trace(a, m["path_color"])) != m["path_length"] or len(np.unique(a)) != m["n_colors"]:
        raise ValueError("Path size or palette differs from its plan")


def choose_n(rng, length):
    return 1 if rng.random() < .12 else int(rng.integers(1, length//2+1))


def generate(rng):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
        if h*w >= 3:
            break
    ncolors = int(rng.integers(2, min(10, max(2, h*w//7+2))+1))
    palette = [int(c) for c in rng.permutation(10)[:ncolors]]
    bg, color = palette[:2]
    # The family is drawn before anything about the route, so neither the
    # endpoints nor n can be read off it beyond what path length implies.
    family = str(rng.choice(path_routes.FAMILIES, p=FAMILY_WEIGHTS))
    style = str(rng.choice(["solid", "lattice", "stripes", "checker", "diagonal", "split"], p=[.35, .13, .15, .12, .15, .10]))
    for attempt in range(1, 201):
        floor = BOX_FLOORS[family]
        bh = int(rng.integers(max(floor, int(h*.45)), h+1))
        bw = int(rng.integers(max(floor, int(w*.45)), w+1))
        route = _route(rng, bh, bw, family)
        if len(route) < 2 or len(route)+ncolors-1 > h*w:
            continue
        if not path_routes.acceptable(route, family):
            continue
        if rng.integers(2):
            route = [(bh-1-y, x) for y, x in route]
        if rng.integers(2):
            route = [(y, bw-1-x) for y, x in route]
        # An irregular route need not reach the sides of its box, so crop to
        # what was drawn: every position on the canvas stays reachable.
        ys, xs = zip(*route)
        route = [(y-min(ys), x-min(xs)) for y, x in route]
        bh, bw = max(ys)-min(ys)+1, max(xs)-min(xs)+1
        if rng.integers(2) and bh <= w and bw <= h:
            route = [(x, y) for y, x in route]
            bh, bw = bw, bh
        r, c = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
        route = [(y+r, x+c) for y, x in route]
        grid = np.full((h, w), bg, int)
        occupied = np.zeros(grid.shape, bool)
        for p in route:
            occupied[p] = True
        grid[occupied] = color
        try:
            decoration = decorate(rng, grid, occupied, bg, palette[2:], style)
            visible = trace(grid, color)
        except ValueError:
            continue
        if len(visible) != len(route):
            continue
        n = choose_n(rng, len(route))
        # Construction answer uses the ordered route. Validation reconstructs it.
        runs = [route[:n], route[::-1][:n]]
        runs.sort(key=lambda run: run[0])
        answer = ", ".join(label+":"+"["+",".join(f"({y},{x})" for y, x in run)+"]" for label, run in zip(("first", "second"), runs))
        ex = {"input": grid.tolist(), "output": answer, "metadata": {
            "path_color": color, "bg_color": bg, "n": n, "n_colors": ncolors,
            "path_family": family, "path_length": len(route),
            "path_turns": path_routes.turns(route),
            "path_route_kind": ("irregular" if family in path_routes.IRREGULAR_FAMILIES
                                else "templated"),
            "path_diagonal_contact": path_routes.diagonal_contact_share(route),
            "path_occupancy": len(route)/(h*w),
            "attempts": attempt, **decoration}}
        validate_example(ex)
        return ex
    raise RuntimeError(f"Task 60 cannot realize its fixed {h}x{w} {family} scene")
