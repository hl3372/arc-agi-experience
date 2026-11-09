"""Contact questions with moderate corner-contact coverage and scalable shapes."""
from __future__ import annotations

import numpy as np

from generators.scene_shapes_56_60 import shape, halo, placement_map, place, decorate
from generators.rigid_scene import visible_objects

# user floor 2026-09-15: small sides are often ambiguous
MIN_SIDE = 5


def choose_connectivity(rng):
    return 4 if rng.random() < .70 else 8


def contact(grid, object_color, target_color, connectivity):
    a = np.asarray(grid)
    if object_color == target_color or connectivity not in (4, 8):
        raise ValueError("Invalid contact question")
    groups, _ = visible_objects(a, excluded=[c for c in np.unique(a) if c not in (object_color, target_color)], minimum=1)
    if len(groups) != 2 or {c for _, c in groups} != {object_color, target_color}:
        raise ValueError("The two named colors must each form exactly one object")
    target = set(map(tuple, np.argwhere(a == target_color)))
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (connectivity == 8 or abs(dy)+abs(dx) == 1)]
    return any((int(y)+dy, int(x)+dx) in target for y, x in np.argwhere(a == object_color) for dy, dx in steps)


def validate_example(ex):
    a, m = np.asarray(ex["input"]), ex["metadata"]
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or a.size < 6:
        raise ValueError("Invalid contact canvas")
    answer = "yes" if contact(a, m["object_color"], m["target_color"], m["contact_mode"]) else "no"
    if ex["output"] != answer:
        raise ValueError("Contact answer disagrees with the displayed cells")
    if len(np.unique(a)) != m["n_colors"]:
        raise ValueError("Contact palette was reduced")
    if int((a == m["target_color"]).sum()) != m["target_cells"] or int((a == m["object_color"]).sum()) != m["object_cells"]:
        raise ValueError("Contact object sizes differ from the plan")


def generate(rng):
    while True:
        h, w = map(int, rng.integers(MIN_SIDE, 31, size=2))
        if h*w >= 6:
            break
    ncolors = int(rng.integers(3, min(10, max(3, h*w//7+2))+1))
    palette = [int(c) for c in rng.permutation(10)[:ncolors]]
    bg, obj, tgt = palette[:3]
    # At 70/30 connectivity these proportions give approximately 50/50 answers.
    # Corner contact is deliberately a minority, as requested by the user.
    if min(h, w) == 1:
        relation = str(rng.choice(["edge", "gap", "far"], p=[.5, .25, .25]))
    else:
        relation = str(rng.choice(["edge", "corner", "gap", "far"], p=[.416, .28, .16, .144]))
    if max(h, w) < 6 and relation == "far":
        relation = "gap"
    target_class = str(rng.choice(["cell", "small", "large"], p=[.38, .25, .37]))
    if relation == "corner" and max(h, w) < 4:
        target_class = "cell"
    style = str(rng.choice(["solid", "lattice", "stripes", "checker", "split", "diagonal"], p=[.52, .12, .10, .08, .08, .10]))
    for attempt in range(1, 301):
        grid = np.full((h, w), bg, dtype=int)
        # Size retries keep the canvas, relation, palette and target class fixed.
        oh = int(rng.integers(max(1, int(h*.25)), min(h, max(2, int(h*.72)))+1))
        ow = int(rng.integers(max(1, int(w*.25)), min(w, max(2, int(w*.72)))+1))
        om, ofam = shape(rng, oh, ow)
        if om.sum() < 2:
            continue
        if target_class == "cell":
            tm, tfam = np.ones((1, 1), bool), "cell"
        else:
            if target_class == "small":
                th, tw = int(rng.integers(1, min(4, h)+1)), int(rng.integers(1, min(4, w)+1))
            else:
                th = int(rng.integers(max(1, int(h*.25)), min(h, max(2, int(h*.65)))+1))
                tw = int(rng.integers(max(1, int(w*.25)), min(w, max(2, int(w*.65)))+1))
            tm, tfam = shape(rng, th, tw)
            if tm.sum() < 2:
                continue
        if om.sum()+tm.sum()+ncolors-2 > h*w:
            continue
        tmask = place(rng, grid, tm, tgt, np.ones(grid.shape, bool))
        if tmask is None:
            continue
        ok = placement_map(om, ~tmask)
        if ok is None:
            continue
        reached = tmask.copy()
        dist = np.zeros(grid.shape, int)
        for d in range(1, max(h, w)+1):
            grown = halo(reached)
            dist[grown & ~reached] = d
            reached = grown
            if reached.all():
                break
        adjacent = halo(tmask, False) & ~tmask
        minimum = np.full(ok.shape, 99, int)
        edges = np.zeros(ok.shape, int)
        for y, x in np.argwhere(om):
            minimum = np.minimum(minimum, dist[y:y+ok.shape[0], x:x+ok.shape[1]])
            edges += adjacent[y:y+ok.shape[0], x:x+ok.shape[1]]
        if relation == "edge":
            ok &= edges > 0
            lengths = np.unique(edges[ok])
            if len(lengths):
                ok &= edges == int(rng.choice(lengths))
        elif relation == "corner":
            ok &= (minimum == 1) & (edges == 0)
        elif relation == "gap":
            ok &= minimum == 2
        else:
            ok &= minimum >= 3
        spots = np.argwhere(ok)
        if not len(spots):
            continue
        r, c = map(int, spots[int(rng.integers(len(spots)))])
        omask = np.zeros(grid.shape, bool)
        omask[r:r+om.shape[0], c:c+om.shape[1]] = om
        grid[omask] = obj
        try:
            decoration = decorate(rng, grid, omask | tmask, bg, palette[3:], style)
        except ValueError:
            continue
        mode = choose_connectivity(rng)  # complete input before instruction draw
        answer = relation == "edge" or (relation == "corner" and mode == 8)
        ex = {"input": grid.tolist(), "output": "yes" if answer else "no", "metadata": {
            "object_color": obj, "target_color": tgt, "target_kind": "cell" if tm.sum() == 1 else "object",
            "contact_mode": mode, "relation": relation, "background_color": bg, "n_colors": ncolors,
            "object_shape_family": ofam, "target_shape_family": tfam, "target_size_class": target_class,
            "object_cells": int(om.sum()), "target_cells": int(tm.sum()), "contact_length": int(edges[r, c]),
            "min_chebyshev_distance": int(minimum[r, c]), "attempts": attempt, **decoration}}
        validate_example(ex)
        return ex
    raise RuntimeError(f"Task 56 cannot realize its fixed {h}x{w} {relation} scene")
