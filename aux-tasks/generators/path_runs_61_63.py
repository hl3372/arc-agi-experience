"""Visible, unambiguous endpoint runs on familiar and irregular paths.

Construction shares task 60's route vocabulary (``generators.path_routes``):
templated shapes and grown irregular routes. Validation independently
reconstructs the complete chain from the final decorated image.
"""
from __future__ import annotations

import numpy as np

from generators import path_routes
from generators.path_routes import route as _route
from generators.scene_shapes_56_60 import decorate


FAMILIES = path_routes.FAMILIES
# The six templated shapes keep their proportions to one another; the four
# irregular routes (2026-09-15 feedback: "we also need more random paths")
# take 40% of the scenes between them.
FAMILY_WEIGHTS = path_routes.shares((.12, .12, .14, .16, .26, .20))
BOX_FLOORS = {"straight": 2, "elbow": 2, "U": 3, "staircase": 2, "serpentine": 5,
              "spiral": 5, **path_routes.IRREGULAR_FLOORS}
MIN_SIDE = 5
MAX_ATTEMPTS = 120


def trace_visible(grid, path_color, start_color=None):
    """Read the displayed chain, starting at the named marker when supplied."""
    a = np.asarray(grid)
    if a.ndim != 2:
        raise ValueError("The path input must be a grid")
    mask = a == path_color
    marker = None
    if start_color is not None:
        if start_color == path_color:
            raise ValueError("The start marker must have a distinct color")
        markers = np.argwhere(a == start_color)
        if len(markers) != 1:
            raise ValueError("The named start color must occur exactly once")
        marker = tuple(map(int, markers[0]))
        mask |= a == start_color
    cells = set(map(tuple, np.argwhere(mask)))
    adjacent = {
        (r, c): [q for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)) if q in cells]
        for r, c in cells
    }
    ends = sorted(p for p, neighbours in adjacent.items() if len(neighbours) == 1)
    if len(ends) != 2 or any(len(neighbours) not in (1, 2) for neighbours in adjacent.values()):
        raise ValueError("The displayed path must have two ends and no branches")
    if marker is not None and marker not in ends:
        raise ValueError("The start marker must be a path endpoint")
    ordered = [marker if marker is not None else ends[0]]
    seen = set(ordered)
    while True:
        following = [p for p in adjacent[ordered[-1]] if p not in seen]
        if not following:
            break
        if len(following) != 1:
            raise ValueError("The displayed path does not have a unique traversal")
        ordered.append(following[0])
        seen.add(following[0])
    if seen != cells:
        raise ValueError("Extra path cells are disconnected from the traced chain")
    return ordered


def solve(grid, task_id, path_color, n, *, start_color=None, highlight_color=None):
    """Literal endpoint operation on the visible chain, independent of its plan."""
    if task_id not in ("61", "62", "63"):
        raise ValueError("Unsupported path-run task")
    if task_id != "61" and start_color is None:
        raise ValueError("A marked path needs its start color")
    ordered = trace_visible(grid, path_color, start_color if task_id != "61" else None)
    cap = len(ordered)//2 if task_id == "61" else len(ordered)-1
    if not 1 <= n <= cap:
        raise ValueError("The requested runs overlap or include the start marker")
    cells = ordered[:n] + ordered[-n:] if task_id == "61" else ordered[-n:]
    if task_id == "62":
        return "[" + ",".join(f"({r},{c})" for r, c in cells) + "]"
    if highlight_color is None or np.any(np.asarray(grid) == highlight_color):
        raise ValueError("The recoloring color must be reserved for the output")
    result = np.asarray(grid).copy()
    for cell in cells:
        result[cell] = highlight_color
    return result.tolist()


def validate_example(ex, task_id):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(MIN_SIDE <= d <= 30 for d in a.shape):
        raise ValueError("Path-run canvas sides must be between five and thirty")
    expected = solve(a, task_id, m["path_color"], m["n"],
                     start_color=m.get("start_color"), highlight_color=m.get("highlight_color"))
    if expected != ex["output"]:
        raise ValueError("Path-run output disagrees with the final visible input")
    ordered = trace_visible(a, m["path_color"], m.get("start_color"))
    steps = [(b[0]-p[0], b[1]-p[1]) for p, b in zip(ordered, ordered[1:])]
    turns = sum(a != b for a, b in zip(steps, steps[1:]))
    if (tuple(a.shape) != (m["grid_h"], m["grid_w"])
            or len(ordered) != m["path_length"] or turns != m["path_turns"]
            or len(np.unique(a)) != m["n_colors"]):
        raise ValueError("Path dimensions, geometry or palette differ from metadata")
    if len(np.unique(a)) != m["planned_n_colors"]:
        raise ValueError("The requested palette was not realized")


def choose_n(rng, length, task_id):
    """Keep short requests common, with a substantial tail over the feasible run."""
    cap = length//2 if task_id == "61" else length-1
    return int(rng.integers(1, min(6, cap)+1 if rng.random() < .45 else cap+1))


def _placed_route(rng, h, w, family):
    floor = BOX_FLOORS[family]
    bh = int(rng.integers(max(floor, int(np.ceil(h*.50))), h+1))
    bw = int(rng.integers(max(floor, int(np.ceil(w*.50))), w+1))
    route = _route(rng, bh, bw, family)
    if rng.integers(2):
        route = [(bh-1-r, c) for r, c in route]
    if rng.integers(2):
        route = [(r, bw-1-c) for r, c in route]
    # Crop before placement: straight paths can occur at every row or column,
    # independently of the unused dimensions of their construction rectangle.
    ys, xs = zip(*route)
    route = [(r-min(ys), c-min(xs)) for r, c in route]
    bh, bw = max(ys)-min(ys)+1, max(xs)-min(xs)+1
    if rng.integers(2) and bh <= w and bw <= h:
        route = [(c, r) for r, c in route]
        bh, bw = bw, bh
    dr, dc = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
    route = [(r+dr, c+dc) for r, c in route]
    if rng.integers(2):
        route.reverse()
    return route


def generate(rng, task_id):
    if task_id not in ("61", "62", "63"):
        raise ValueError("Unsupported path-run task")
    h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
    semantic_count = 2 if task_id == "61" else 3
    # Small fields have room for fewer readable accents. Larger fields reach
    # every available color; recoloring tasks reserve one for their output.
    cap = min(10 if task_id == "62" else 9, semantic_count+h*w//8)
    color_counts = np.arange(semantic_count, cap+1)
    weights = np.exp(-.10*np.arange(len(color_counts)))
    ncolors = int(rng.choice(color_counts, p=weights/weights.sum()))
    palette = list(map(int, rng.permutation(10)))
    bg, path_color = palette[:2]
    start_color = palette[2] if task_id != "61" else None
    highlight_color = palette[ncolors] if task_id != "62" else None
    family = str(rng.choice(FAMILIES, p=FAMILY_WEIGHTS))
    # About half the scenes are plain or have a few drawn accents. Textures
    # occupy a minority of the field, and never reuse either semantic color.
    style = "solid"
    if ncolors > semantic_count and min(h, w) >= 8:
        style = str(rng.choice(("solid", "lattice", "stripes", "diagonal", "checker", "split"),
                               p=(.40, .26, .12, .08, .07, .07)))
    for attempt in range(1, MAX_ATTEMPTS+1):
        route = _placed_route(rng, h, w, family)
        steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(route, route[1:])]
        turns = sum(a != b for a, b in zip(steps, steps[1:]))
        # A single bend is an elbow, and a two-sweep serpent is only a U. An
        # irregular route that barely bends, or that folds against itself, is
        # likewise not what it claims. Retain the planned family rather than
        # silently relabeling a collapse.
        if (family == "staircase" and turns < 2) or (family == "serpentine" and turns < 4):
            continue
        if not path_routes.acceptable(route, family):
            continue
        if len(route)+ncolors-semantic_count+1 > h*w:
            continue
        grid = np.full((h, w), bg, int)
        protected = np.zeros((h, w), bool)
        for cell in route:
            grid[cell] = path_color
            protected[cell] = True
        if start_color is not None:
            grid[route[0]] = start_color
        try:
            decoration = decorate(rng, grid, protected, bg, palette[semantic_count:ncolors],
                                  style, density=.12, max_pattern_fraction=.28)
            # This also rejects unintended contacts introduced by the route
            # geometry. Colors and field style stay fixed on every retry.
            visible = trace_visible(grid, path_color, start_color)
        except ValueError:
            continue
        if len(visible) != len(route) or len(np.unique(grid)) != ncolors:
            continue
        n = choose_n(rng, len(route), task_id)
        cells = route[:n] + route[-n:] if task_id == "61" else route[-n:]
        if task_id == "62":
            answer = "[" + ",".join(f"({r},{c})" for r, c in cells) + "]"
        else:
            output = grid.copy()
            for cell in cells:
                output[cell] = highlight_color
            answer = output.tolist()
        decoration_cells = int(np.count_nonzero(~protected & (grid != bg)))
        meta = {
            "grid_h": h, "grid_w": w, "path_color": path_color, "bg_color": bg,
            "n": n, "n_colors": ncolors, "planned_n_colors": ncolors,
            "path_family": family, "path_length": len(route),
            "path_turns": turns,
            "path_route_kind": ("irregular" if family in path_routes.IRREGULAR_FAMILIES
                                else "templated"),
            "path_diagonal_contact": path_routes.diagonal_contact_share(route),
            "path_occupancy": len(route)/(h*w), "connectivity": 4,
            "answer_cap": len(route)//2 if task_id == "61" else len(route)-1,
            "attempts": attempt, "op_background_style": style,
            "texture_named": False, "decoration_cells": decoration_cells,
            "texture_coverage": decoration_cells/(h*w), **decoration,
        }
        if start_color is not None:
            meta["start_color"] = start_color
            meta["start_row"], meta["start_col"] = route[0]
        if highlight_color is not None:
            meta["highlight_color"] = highlight_color
        ex = {"input": grid.tolist(), "output": answer, "metadata": meta}
        validate_example(ex, task_id)
        return ex
    raise RuntimeError(f"Task {task_id} cannot realize its fixed {h}x{w} {family} scene")
