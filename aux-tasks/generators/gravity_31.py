"""Collision-aware rigid gravity, with shared scenes across all five rules."""
from __future__ import annotations

import numpy as np

from generators.rigid_scene import capacity, draw_objects, infer_ground, visible_objects

DIRECTIONS = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
VARIANTS = ("nearest_border", "top", "bottom", "left", "right")


def prompt31_action(meta):
    """The per-example action sentence (no trailing space); prompt variants reach it as {fn:_p31_action}."""
    if meta["rule"] == "nearest_border":
        return ("Choose each object's nearest grid border from its starting position, using the smallest gap "
                "between any of its cells and that border. Break ties in this order: top, bottom, left, right.")
    return f"Move every object toward the {meta['direction']} grid border."


def prompt31(meta):
    action = prompt31_action(meta) + " "
    return ("Ignoring the background, each 4-connected same-color group (connected horizontally or vertically; "
            "cells that touch only at a corner are not connected) is one object; different colors are separate "
            "even where they touch. " + action +
            "Keep each object's chosen direction fixed. Visit objects in the order of their topmost cells, "
            "breaking ties from left to right, using their original positions for this order. "
            "On each visit, move the object one cell in its direction if all its cells stay in the grid "
            "and avoid other objects; otherwise leave it in place for that visit. Repeat these visits in "
            "the same order until no object moves. Objects keep their shapes and cannot pass through each "
            "other. The cells an object leaves behind become background. Leave every other cell unchanged.")


def _directions(objects, h, w, variant):
    names, ties = [], 0
    for cells, _ in objects:
        rows, cols = zip(*cells)
        distances = (min(rows), h - 1 - max(rows), min(cols), w - 1 - max(cols))
        ties += sum(x == min(distances) for x in distances) > 1
        names.append(list(DIRECTIONS)[int(np.argmin(distances))] if variant == "nearest_border" else variant)
    return names, ties


def _move_planned(objects, h, w, variant):
    """Construction via mutable labelled occupancy and array translations."""
    ordered = sorted(objects, key=lambda item: min(item[0]))
    names, ties = _directions(ordered, h, w, variant)
    points = [np.asarray(sorted(cells), dtype=int) for cells, _ in ordered]
    occupancy = np.full((h, w), -1, dtype=int)
    for i, cells in enumerate(points):
        occupancy[cells[:, 0], cells[:, 1]] = i
    movement = [0] * len(points)
    blocked = set()
    for _ in range((h + w) * len(points) + 1):
        changed = False
        for i, cells in enumerate(points):
            shifted = cells + DIRECTIONS[names[i]]
            if ((shifted < 0).any() or (shifted[:, 0] >= h).any() or (shifted[:, 1] >= w).any()):
                continue
            other = occupancy[shifted[:, 0], shifted[:, 1]]
            if ((other >= 0) & (other != i)).any():
                blocked.add(i)
                continue
            occupancy[cells[:, 0], cells[:, 1]] = -1
            occupancy[shifted[:, 0], shifted[:, 1]] = i
            points[i] = shifted
            movement[i] += 1
            changed = True
        if not changed:
            return [(set(map(tuple, p.tolist())), ordered[i][1]) for i, p in enumerate(points)], movement, ties, len(blocked)
    raise RuntimeError("Gravity failed to settle")


def solve_gravity(grid, variant):
    """Visible-grid re-solver using sets, independent of construction motion."""
    a = np.asarray(grid)
    h, w = a.shape
    ground = infer_ground(a)
    objects, _ = visible_objects(a, excluded=(ground,), minimum=1)
    cells = [set(item[0]) for item in objects]
    vectors = []
    for obj in cells:
        rows, cols = zip(*obj)
        distances = [min(rows), h - max(rows) - 1, min(cols), w - max(cols) - 1]
        name = tuple(DIRECTIONS)[distances.index(min(distances))] if variant == "nearest_border" else variant
        vectors.append(DIRECTIONS[name])
    while True:
        changed = False
        for i, (dy, dx) in enumerate(vectors):
            proposed = {(r + dy, c + dx) for r, c in cells[i]}
            if any(r < 0 or c < 0 or r >= h or c >= w for r, c in proposed):
                continue
            if any(proposed & other for j, other in enumerate(cells) if i != j):
                continue
            cells[i] = proposed
            changed = True
        if not changed:
            break
    out = np.full_like(a, ground)
    for (_, color), positions in zip(objects, cells):
        for r, c in positions:
            out[r, c] = color
    return out


def validate_example(ex):
    m = ex["metadata"]
    variant = m["rule"] if m["rule"] == "nearest_border" else m["direction"]
    if solve_gravity(ex["input"], variant).tolist() != ex["output"]:
        raise ValueError("Task 31 visible reconstruction differs from output")


def generate_move_to_nearest_border(rng):
    while True:
        h, w = map(int, rng.integers(1, 31, size=2))
        if h * w >= 5 and capacity(h, w):
            break
    max_objects = min(12, capacity(h, w), max(1, (h * w - 1) // 5))
    palette_size = int(rng.integers(2, min(10, max_objects + 1) + 1))
    palette = [int(x) for x in rng.permutation(10)[:palette_size]]
    n = int(rng.integers(palette_size - 1, max(palette_size - 1, min(max_objects, palette_size + 2)) + 1))
    margins = [(y, x) for y, x in ((1, 1), (1, 0), (0, 1))
               if h > 2 * y and w > 2 * x and capacity(h - 2 * y, w - 2 * x) >= n]
    if margins and rng.random() < .90:
        best = max(y + x for y, x in margins)
        choices = [(y, x) for y, x in margins if y + x == best]
        margin_r, margin_c = choices[int(rng.integers(len(choices)))]
    else:
        margin_r = margin_c = 0
    for attempt in range(1, 151):
        try:
            objects, families = draw_objects(rng, h - 2 * margin_r, w - 2 * margin_c, palette[1:], n,
                                            scale=(.50, .95) if h * w >= 150 else (.38, .85))
            if margin_r or margin_c:
                objects = [({(r + margin_r, c + margin_c) for r, c in cells}, color) for cells, color in objects]
        except ValueError:
            continue
        total = sum(len(cells) for cells, _ in objects)
        if total >= h * w * .44:
            continue
        a = np.full((h, w), palette[0], dtype=np.uint8)
        for cells, color in objects:
            for r, c in cells:
                a[r, c] = color
        # The scene is complete before selecting the instruction. Neither mode
        # rejects ties, overlaps of eventual border destinations, or identities.
        variant = VARIANTS[int(rng.choice(5, p=(.5, .125, .125, .125, .125)))]
        final, movement, ties, blocked = _move_planned(objects, h, w, variant)
        out = np.full_like(a, palette[0])
        for cells, color in final:
            for r, c in cells:
                out[r, c] = color
        ex = {"input": a.tolist(), "output": out.tolist(), "metadata": {
            "rule": "nearest_border" if variant == "nearest_border" else "directional_gravity",
            "direction": "per_object" if variant == "nearest_border" else variant,
            "n_colors": palette_size, "object_count": n, "shape_families": families,
            "object_areas": [len(cells) for cells, _ in objects], "foreground_cells": total,
            "background_color": palette[0], "background": "solid", "attempts": attempt,
            "movement": movement, "nearest_ties": ties, "blocked_objects": blocked,
            "identity": bool(np.array_equal(a, out))}}
        validate_example(ex)
        return ex
    raise RuntimeError("Task 31 could not realize its fixed scene budget")
