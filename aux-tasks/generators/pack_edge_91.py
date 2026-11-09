"""Task 91 -- pack every object against one edge of the grid.

The September 2026 rebuild replaces the inline legacy version in
``generators/misc.py``.  What the review found and what this module does about
it:

* **The prompt did the wrong half of the work.**  It said "every rigid
  4-connected object" without saying that an object is a *same-colour* group,
  without "ignoring the background", and without ever saying what happens to
  the cells an object leaves behind.  The prompt now carries the standard
  object rule (``prompts._largest_object_rule``), so sparse background texture
  is disposed of by the isolated-cell clause and the ground is inferred, never
  named; it then states the vacated-cells rule and the standing
  "Leave every other cell unchanged."  The settle-order clause is kept
  verbatim: it is load-bearing (farthest-first differs on two thirds of the
  grids) and it agrees with the "slide until nothing moves" reading on every
  emitted example, because disagreement is a rejection.
* **Coverage.**  Sides were 9..24 with no grid above 700 cells, colours were
  exactly ``n_objects + 1`` (3..6), every object had 4..8 cells and occupancy
  fell from 16% to 4.6% on the large canvases.  Sides are now drawn
  independently over 5..30, bodies come from the shared vocabulary and scale
  with the canvas, the palette is drawn over 2..10 *independently of the
  object count* (colours are assigned to the finished layout by graph
  colouring, so they may repeat), and sparse library textures appear on about
  a quarter of the eligible grids.
* **Identity examples.**  The old ``moved`` guard rejected every already-packed
  scene, so the task never showed one.  They are now planned at ~10%.
* **No independent re-solver.**  The generator planned and produced the answer
  with one function and swallowed failures with ``continue``.  Every emitted
  example is now re-solved from the visible grid by two independent
  implementations of "closest settles first" (a set-shifting one and a
  run-length one, with different tie-breaks) AND by the "slide until nothing
  moves" relaxation run from two orders.  All four must agree, and the
  validator raises when they do not.

Textures are background: they are sparse (isolated single cells), they never
sit in an object's swept path or its halo, and they are copied into the output
unchanged.  That is what makes "the cells an object leaves behind become
background" a complete statement -- nothing is ever slid over.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import apply_background, infer_background
from common.grid import NUM_COLORS, grid_to_list
# The body vocabulary, the literal object reader and the ground reader are the
# ones task 90 already uses: this module extends that scene family rather than
# copying it.  The module is bound lazily (``_b90``) because ``registry`` pulls
# both modules in and a direct ``from ... import`` fails when a caller imports
# task 90's module first.
from generators import border_objects_90 as _b90
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

MIN_BODY = 5              # the smallest body a piece is ever drawn with (task 90's floor)

MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 40
LAYOUT_TRIES = 4
TEXTURE_MIN_SIDE = 8
TEXTURE_P = 0.32          # attempt rate on eligible grids -> ~25% realised
SPECKLE_P = 0.40          # of those, the share drawn as a two/three-colour speckle
BG_MARGIN = 1.25
IDENTITY_P = 0.11         # scenes that are already packed
CONTACT_P = 0.40          # layouts that seat one object against another
LANE_P = 0.78             # placements steered into an occupied lane
BLOCK_P = 0.74            # layouts that must contain a blocked object
MAX_OCCUPANCY = 0.36

DIRECTIONS = ("top", "bottom", "left", "right")
_STEP = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The rule, three ways (no access to the plan; cells come from the picture)
# ---------------------------------------------------------------------------

def _distance(cells, h: int, w: int, direction: str) -> int:
    """How far the object would slide on an otherwise empty grid."""
    if direction == "top":
        return min(r for r, _ in cells)
    if direction == "bottom":
        return h - 1 - max(r for r, _ in cells)
    if direction == "left":
        return min(c for _, c in cells)
    return w - 1 - max(c for _, c in cells)


def _slide(cells, blocked: set, h: int, w: int, step) -> set:
    """Move a rigid cell set one step at a time until the edge or an obstacle."""
    dr, dc = step
    current = set(cells)
    while True:
        nxt = {(r + dr, c + dc) for r, c in current}
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in nxt):
            return current
        if nxt & blocked:
            return current
        current = nxt


def pack_closest_first(objects, h: int, w: int, direction: str,
                       *, reverse_ties: bool = False, order: Optional[list] = None):
    """"Objects closest to the edge settle first" -- set-shifting implementation.

    ``order`` overrides the settle order and is only used to measure the
    counterfactual readings (farthest first, reading order).
    """
    step = _STEP[direction]
    if order is None:
        def key(i):
            cells = objects[i]
            d = _distance(cells, h, w, direction)
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            return (d, -r0, -c0, -i) if reverse_ties else (d, r0, c0, i)
        order = sorted(range(len(objects)), key=key)
    occupied: set = set()
    final: list = [None] * len(objects)
    for i in order:
        cells = _slide(objects[i], occupied, h, w, step)
        final[i] = frozenset(cells)
        occupied |= cells
    return final


def pack_closest_first_runs(objects, h: int, w: int, direction: str):
    """The same rule, computed differently: the free run ahead of every cell.

    A second literal implementation with its own tie-break (distance, then the
    object's index as it comes off the picture): if the two ever disagreed, the
    settle order would not be determined by the prompt and the example is
    rejected.
    """
    dr, dc = _STEP[direction]
    blocked = [[False] * w for _ in range(h)]
    final: list = [None] * len(objects)
    order = sorted(range(len(objects)),
                   key=lambda i: (_distance(objects[i], h, w, direction), i))
    for i in order:
        shift = h + w
        for r, c in objects[i]:
            run = 0
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w and not blocked[rr][cc]:
                run += 1
                rr += dr
                cc += dc
            shift = min(shift, run)
            if shift == 0:
                break
        moved = frozenset((r + dr * shift, c + dc * shift) for r, c in objects[i])
        final[i] = moved
        for r, c in moved:
            blocked[r][c] = True
    return final


def pack_relaxation(objects, h: int, w: int, direction: str, *, reverse: bool = False):
    """"Slide every object until nothing moves any more" -- the other reading.

    Sliding only ever decreases an object's distance to the edge, so the sweep
    terminates.  The emitted examples are the ones where this reading and the
    settle-order reading agree.
    """
    step = _STEP[direction]
    pos = [set(cells) for cells in objects]
    order = list(range(len(pos)))
    if reverse:
        order.reverse()
    changed = True
    while changed:
        changed = False
        for i in order:
            others: set = set()
            for j, cells in enumerate(pos):
                if j != i:
                    others |= cells
            moved = _slide(pos[i], others, h, w, step)
            if moved != pos[i]:
                pos[i] = moved
                changed = True
    return [frozenset(cells) for cells in pos]


def consistent_packing(objects, h: int, w: int, direction: str):
    """The packed positions, or None when the readings do not agree.

    Five computations: the settle order with two tie-breaks, a second literal
    implementation of it, and the relaxation from two orders.  Anything that
    separates them is an example whose answer the prompt does not determine.
    """
    if not objects:
        return []
    base = pack_closest_first(objects, h, w, direction)
    for other in (
        pack_closest_first(objects, h, w, direction, reverse_ties=True),
        pack_closest_first_runs(objects, h, w, direction),
        pack_relaxation(objects, h, w, direction),
        pack_relaxation(objects, h, w, direction, reverse=True),
    ):
        if other != base:
            return None
    return base


def _reading_order(objects) -> list:
    """Row-major settle order: the careless reading of "closest first"."""
    return sorted(range(len(objects)),
                  key=lambda i: (min(objects[i]), i))


def _farthest_order(objects, h: int, w: int, direction: str) -> list:
    return sorted(range(len(objects)),
                  key=lambda i: (-_distance(objects[i], h, w, direction), i))


# ---------------------------------------------------------------------------
# Reading a grid back into objects (literal, from the picture only)
# ---------------------------------------------------------------------------

def read_objects(grid) -> tuple[list, list, list]:
    """(cells, colours, texture cells) of a grid under the prompt's object rule.

    ``_objects`` (task 90) drops the inferred ground and every same-colour group
    of one cell; those single cells are the background texture, which stays
    exactly where it is.
    """
    rows = [list(row) for row in grid]
    ground = _b90.read_ground(rows)
    comps = _b90._objects(rows, 4)
    cells = [frozenset(comp) for comp in comps]
    colors = [int(rows[next(iter(comp))[0]][next(iter(comp))[1]]) for comp in cells]
    covered = set().union(*cells) if cells else set()
    texture = [(r, c) for r, row in enumerate(rows) for c, v in enumerate(row)
               if v != ground and (r, c) not in covered]
    return cells, colors, texture


def _paint(h: int, w: int, ground: int, cells, colors, texture, rows) -> list:
    out = [[ground] * w for _ in range(h)]
    for body, color in zip(cells, colors):
        for r, c in body:
            out[r][c] = color
    for r, c in texture:
        out[r][c] = rows[r][c]
    return out


def solve(grid, direction: str) -> list:
    """The output grid, re-derived from the input grid and the direction."""
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    ground = _b90.read_ground(rows)
    cells, colors, texture = read_objects(rows)
    packed = consistent_packing(cells, h, w, direction)
    if packed is None:
        raise ValueError("task 91: the settle order does not determine the output")
    return _paint(h, w, ground, packed, colors, texture, rows)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _object_budget(h: int, w: int, density: float) -> int:
    """How many bodies the canvas can carry and still leave room to slide."""
    area = h * w
    return int(max(2, min(14, max(2, area // 10), max(2, int(density * area / MIN_BODY)))))


def sample_plan(rng: np.random.Generator) -> dict:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    direction = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
    # ink first, counts second: bodies grow with the canvas instead of the
    # canvas being filled with more dominoes (the task 90 lesson).
    density = float(rng.uniform(0.11, 0.26))
    budget = _object_budget(h, w, density)
    n_objects = int(rng.integers(2, budget + 1))
    hi = max(float(MIN_BODY) + 1.0, min(0.16 * area, 80.0))
    per_piece = float(np.clip(density * area / n_objects, float(MIN_BODY), hi))
    # The palette is drawn over 2..10 and truncated only by the number of
    # bodies; it is NOT n_objects + 1 any more, and the colours themselves are
    # assigned to the finished layout (see ``_assign_colors``), so two objects
    # may share one as long as they can never come to touch.
    target = int(rng.choice(np.arange(2, 11),
                            p=np.array([.07, .09, .10, .11, .11, .12, .12, .13, .15])))
    n_obj_colors = int(min(max(1, target - 1), n_objects, 9))
    # A two-colour grid means every body shares one colour, so no two of them
    # may touch in EITHER picture -- and packed bodies stack.  Such a scene is
    # therefore a small one: the palette follows the layout's real capacity
    # rather than the other way round.
    mono = n_obj_colors == 1
    if mono:
        n_objects = min(n_objects, 3)
    bg = int(rng.integers(0, NUM_COLORS))
    pool = [c for c in range(NUM_COLORS) if c != bg]
    obj_colors = [int(c) for c in rng.choice(pool, size=n_obj_colors, replace=False)]
    return {
        "h": h, "w": w, "area": area, "direction": direction,
        "density": density, "object_budget": budget, "n_objects": n_objects,
        "per_piece": per_piece, "palette_target": target, "mono": mono,
        "planned_colors": n_obj_colors, "bg": bg, "obj_colors": obj_colors,
        "identity": bool(rng.random() < IDENTITY_P),
        "want_block": bool(rng.random() < BLOCK_P) and not mono,
        "contact": bool(rng.random() < CONTACT_P) and not mono,
    }


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _window_any(vec: np.ndarray, k: int) -> np.ndarray:
    """``out[i] = vec[i:i+k].any()`` for every full window."""
    cum = np.concatenate(([0], np.cumsum(vec.astype(int))))
    n = len(vec) - k + 1
    return (cum[k:k + n] - cum[:n]) > 0


def _lane_mask(occ: np.ndarray, shape, direction: str, ok_shape) -> np.ndarray:
    """Top-left positions whose lane already holds an object.

    Two objects can only ever block each other when they share a lane (columns
    for a vertical push, rows for a horizontal one), and the settle-order
    clause is only load-bearing when they do.
    """
    fh, fw = shape
    nh, nw = ok_shape
    if direction in ("top", "bottom"):
        win = _window_any(occ.any(axis=0), fw)[:nw]
        return np.broadcast_to(win[None, :], (nh, nw))
    win = _window_any(occ.any(axis=1), fh)[:nh]
    return np.broadcast_to(win[:, None], (nh, nw))


def build_layout(rng: np.random.Generator, plan: dict) -> Optional[dict]:
    """Place the bodies.  Colours are not chosen here -- the layout comes first."""
    h, w, direction = plan["h"], plan["w"], plan["direction"]
    occ = np.zeros((h, w), dtype=bool)
    lim_a = max(1, min(h - 1, 14))
    lim_b = max(1, min(w - 1, 14))
    cells: list = []
    families: list = []
    contacts = 0
    for i in range(plan["n_objects"]):
        want_contact = plan["contact"] and i > 0 and rng.random() < 0.5
        seated = None
        for _ in range(8):
            side = float(np.sqrt(plan["per_piece"]))
            bh = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_a))
            bw = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_b))
            room = lim_a * lim_b
            floor = MIN_BODY if room > MIN_BODY else max(2, room)
            for _ in range(8):
                if bh * bw > floor:
                    break
                if bh < lim_a and (bh <= bw or bw >= lim_b):
                    bh += 1
                elif bw < lim_b:
                    bw += 1
                else:
                    break
            built = _b90._body(rng, bh, bw, min_cells=floor)
            if built is None:
                continue
            mask, fam = built
            free = (~occ) if want_contact else (~halo(occ))
            ok = placement_map(mask, free)
            if ok is None:
                continue
            if want_contact:
                if not occ.any():
                    continue
                ok = ok & _b90._shifted_any(halo(occ, diagonal=False) & ~occ, mask, *ok.shape)
            if i > 0 and not plan["mono"] and rng.random() < LANE_P:
                lane = ok & _lane_mask(occ, mask.shape, direction, ok.shape)
                if lane.any():
                    ok = lane
            spots = np.argwhere(ok)
            if not len(spots):
                continue
            r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
            body = frozenset((r + int(dr), c + int(dc)) for dr, dc in np.argwhere(mask))
            occ[r:r + mask.shape[0], c:c + mask.shape[1]] |= mask
            seated = (body, fam)
            break
        if seated is None:
            return None
        cells.append(seated[0])
        families.append(seated[1])
        if want_contact:
            contacts += 1
    if int(occ.sum()) > MAX_OCCUPANCY * plan["area"]:
        return None
    return {"cells": cells, "families": families, "contacts": contacts,
            "occupied": int(occ.sum())}


# ---------------------------------------------------------------------------
# Colours: assigned to the finished layout, so they can repeat safely
# ---------------------------------------------------------------------------

def _touch_graph(sets_a, sets_b, n: int) -> list:
    """Objects that touch (8-adjacency) in either the input or the output.

    Two objects that share a colour and touch in either picture would read as
    one body, so they are the pairs the colouring must separate.
    """
    conflicts = [set() for _ in range(n)]
    for group in (sets_a, sets_b):
        owner: dict = {}
        for i, body in enumerate(group):
            for cell in body:
                owner[cell] = i
        for cell, i in owner.items():
            r, c = cell
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    j = owner.get((r + dr, c + dc))
                    if j is not None and j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
    return conflicts


def _color_layout(rng: np.random.Generator, conflicts: list, plan: dict):
    """Colour the finished layout, widening the palette only when it must.

    The planned palette is the first try; when the layout's contacts make it
    unusable (greedy runs out of colours), one colour at a time is added rather
    than throwing the whole plan away -- discarding it would quietly re-draw
    the scene until an easy one turned up.
    """
    pool = list(plan["obj_colors"])
    spare = [c for c in range(NUM_COLORS)
             if c != plan["bg"] and c not in pool]
    widened = 0
    while True:
        for _ in range(3):
            colors = _assign_colors(rng, conflicts, pool)
            if colors is not None:
                return colors, widened
        if not spare or len(pool) >= 9:
            return None, widened
        pool.append(int(spare.pop(int(rng.integers(len(spare))))))
        widened += 1


def _assign_colors(rng: np.random.Generator, conflicts: list, pool: list) -> Optional[list]:
    """Greedy colouring that prefers a colour it has not used yet."""
    n = len(conflicts)
    colors = [-1] * n
    used: list = []
    for i in (int(v) for v in rng.permutation(n)):
        banned = {colors[j] for j in conflicts[i] if colors[j] >= 0}
        allowed = [c for c in pool if c not in banned]
        if not allowed:
            return None
        fresh = [c for c in allowed if c not in used]
        pick = fresh if fresh else allowed
        chosen = int(pick[int(rng.integers(len(pick)))])
        colors[i] = chosen
        if chosen not in used:
            used.append(chosen)
    return colors


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _swept(cells, packed, direction: str) -> set:
    """Every cell an object occupies on its way from start to rest."""
    dr, dc = _STEP[direction]
    out: set = set()
    for body, final in zip(cells, packed):
        r0 = min(r for r, _ in body)
        c0 = min(c for _, c in body)
        shift = abs(min(r for r, _ in final) - r0) + abs(min(c for _, c in final) - c0)
        for k in range(shift + 1):
            out |= {(r + dr * k, c + dc * k) for r, c in body}
    return out


def _texture(rng: np.random.Generator, grid_in: np.ndarray, bg: int, protect: np.ndarray):
    h, w = grid_in.shape
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= TEXTURE_P:
        return None, dict(_NO_TEXTURE)
    kind = "multi_speckle" if rng.random() < SPECKLE_P else None
    painted = apply_background(rng, grid_in, bg, protect, group="sparse", kind=kind)
    if painted is None:
        return None, dict(_NO_TEXTURE)
    display, info = painted
    return display, {"texture_group": info["group"], "texture_kind": info["kind"],
                     "texture_colors": [int(c) for c in info["texture_colors"]],
                     "texture_named": False, "texture_coverage": float(info["coverage"])}


# ---------------------------------------------------------------------------
# Validation -- everything below is recomputed from the emitted grids
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> None:
    grid_in, grid_out, meta = example["input"], example["output"], example["metadata"]
    h, w = len(grid_in), len(grid_in[0])
    direction = str(meta["direction"])
    if (h, w) != (len(grid_out), len(grid_out[0])):
        raise ValueError("task 91: the output does not have the input's shape")
    ground = _b90.read_ground(grid_in)
    if ground != meta["bg_color"]:
        raise ValueError("task 91: the visible ground disagrees with the plan")
    cells, colors, texture = read_objects(grid_in)
    if len(cells) != meta["n_objects"]:
        raise ValueError("task 91: the visible object count disagrees with the metadata")
    if sorted(len(body) for body in cells) != list(meta["object_sizes"]):
        raise ValueError("task 91: the object sizes disagree with the visible grid")
    if min((len(body) for body in cells), default=99) < 2:
        raise ValueError("task 91: an object has fewer than two cells")
    packed = consistent_packing(cells, h, w, direction)
    if packed is None:
        raise ValueError("task 91: the readings of the settle order disagree")
    swept = _swept(cells, packed, direction)
    if any(cell in swept for cell in texture):
        raise ValueError("task 91: a background texture cell lies in an object's path")
    expected = _paint(h, w, ground, packed, colors, texture, grid_in)
    if expected != [list(row) for row in grid_out]:
        raise ValueError("task 91: the packed grid disagrees with the literal reading")
    identity = [list(row) for row in grid_in] == [list(row) for row in grid_out]
    if identity != bool(meta["identity_output"]):
        raise ValueError("task 91: the identity flag is wrong")
    blocked = 0
    for body, final in zip(cells, packed):
        solo = _slide(body, set(), h, w, _STEP[direction])
        if frozenset(solo) != final:
            blocked += 1
    if blocked != meta["n_blocked"]:
        raise ValueError("task 91: the blocked-object count is not the recorded one")
    alt = pack_closest_first(cells, h, w, direction,
                            order=_farthest_order(cells, h, w, direction))
    if (alt != packed) != bool(meta["differs_under_farthest_first"]):
        raise ValueError("task 91: the settle-order flag is wrong")
    row_major = pack_closest_first(cells, h, w, direction, order=_reading_order(cells))
    if (row_major != packed) != bool(meta["differs_under_reading_order"]):
        raise ValueError("task 91: the reading-order flag is wrong")
    arr = np.asarray(grid_in)
    info = infer_background(arr, margin=BG_MARGIN)
    if not info["identifiable"] or int(info["color"]) != meta["bg_color"]:
        raise ValueError("task 91: the background is not identifiable")
    if meta["texture_colors"]:
        from common.backgrounds import isolated_cell_mask
        tex = np.isin(arr, meta["texture_colors"])
        if not bool(isolated_cell_mask(arr)[tex].all()):
            raise ValueError("task 91: a texture cell is not isolated")
        if set(meta["texture_colors"]) & set(meta["object_colors"] + [meta["bg_color"]]):
            raise ValueError("task 91: a texture colour is an object or ground colour")
        out_arr = np.asarray(grid_out)
        if not bool((out_arr[tex] == arr[tex]).all()):
            raise ValueError("task 91: a texture cell moved")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _color_blind_bodies(grid, ground: int) -> list:
    """4-connected components of the non-ground cells, colours merged.

    The reading a solver takes when it drops "cells of different colors are
    always separate": two touching objects become one rigid body.
    """
    rows = [list(row) for row in grid]
    h, w = len(rows), len(rows[0])
    pending = {(r, c) for r in range(h) for c in range(w) if rows[r][c] != ground}
    out = []
    while pending:
        start = min(pending)
        pending.remove(start)
        comp, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    comp.append(q)
                    stack.append(q)
        if len(comp) >= 2:
            out.append(frozenset(comp))
    return out


def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = sample_plan(rng)
        h, w, direction = plan["h"], plan["w"], plan["direction"]
        layout = None
        for _ in range(LAYOUT_TRIES):
            layout = build_layout(rng, plan)
            if layout is None:
                continue
            cells = layout["cells"]
            if plan["identity"]:
                # an already-packed scene: pack once and use the result as the
                # input, then require the rule to be a no-op on it
                seated = consistent_packing(cells, h, w, direction)
                if seated is None:
                    layout = None
                    continue
                cells = list(seated)
            packed = consistent_packing(cells, h, w, direction)
            if packed is None:
                layout = None
                continue
            if plan["identity"] and list(packed) != list(cells):
                layout = None
                continue
            if not plan["identity"] and list(packed) == list(cells):
                layout = None
                continue
            blocked = sum(1 for body, final in zip(cells, packed)
                          if frozenset(_slide(body, set(), h, w, _STEP[direction])) != final)
            if plan["want_block"] and blocked == 0:
                layout = None
                continue
            layout["cells"] = cells
            layout["packed"] = packed
            layout["n_blocked"] = blocked
            break
        if layout is None:
            continue
        cells, packed = layout["cells"], layout["packed"]

        conflicts = _touch_graph(cells, packed, len(cells))
        if plan["mono"] and any(conflicts):
            continue          # a one-colour scene whose bodies came to touch
        colors, widened = _color_layout(rng, conflicts, plan)
        if colors is None:
            continue

        grid_in = np.full((h, w), plan["bg"], dtype=np.int16)
        grid_out = np.full((h, w), plan["bg"], dtype=np.int16)
        for body, final, color in zip(cells, packed, colors):
            for r, c in body:
                grid_in[r, c] = color
            for r, c in final:
                grid_out[r, c] = color

        protect = np.zeros((h, w), dtype=bool)
        for r, c in _swept(cells, packed, direction):
            protect[r, c] = True
        display, tex = _texture(rng, grid_in, plan["bg"], halo(protect))
        if display is not None:
            tex_mask = display != grid_in
            grid_in = display
            grid_out = np.where(tex_mask, display, grid_out)

        rows_in = grid_to_list(grid_in.astype(np.int16))
        rows_out = grid_to_list(grid_out.astype(np.int16))

        # --- everything from here reads the picture, not the plan -----------
        if _b90.read_ground(rows_in) != plan["bg"]:
            continue
        seen_cells, seen_colors, seen_texture = read_objects(rows_in)
        if len(seen_cells) != len(cells) or set(seen_cells) != set(cells):
            continue          # a body merged with another or lost a cell
        info = infer_background(grid_in, margin=BG_MARGIN)
        if not info["identifiable"] or int(info["color"]) != plan["bg"]:
            continue

        identity = rows_in == rows_out
        alt = pack_closest_first(cells, h, w, direction,
                                 order=_farthest_order(cells, h, w, direction))
        row_major = pack_closest_first(cells, h, w, direction, order=_reading_order(cells))
        blind = _color_blind_bodies(rows_in, plan["bg"])
        blind_packed = consistent_packing(blind, h, w, direction)
        blind_differs = True
        if blind_packed is not None:
            blind_grid = np.full((h, w), plan["bg"], dtype=np.int16)
            for body, final in zip(blind, blind_packed):
                dr = min(r for r, _ in final) - min(r for r, _ in body)
                dc = min(c for _, c in final) - min(c for _, c in body)
                for r, c in body:
                    blind_grid[r + dr, c + dc] = grid_in[r, c]
            for r, c in seen_texture:
                blind_grid[r, c] = grid_in[r, c]
            blind_differs = grid_to_list(blind_grid) != rows_out

        sizes = sorted(len(body) for body in cells)
        present = sorted({int(v) for v in np.unique(grid_in)})
        occ_cells = sum(sizes)
        meta = {
            "grid_h": h, "grid_w": w, "n_colors": len(present),
            "direction": direction,
            "n_objects": len(cells),
            "n_blocked": layout["n_blocked"],
            "n_moved": sum(1 for body, final in zip(cells, packed) if body != final),
            "max_shift": max((_distance(body, h, w, direction) for body in cells), default=0),
            "identity_output": bool(identity),
            "object_sizes": sizes,
            "object_colors": sorted(set(colors)),
            "object_kinds": list(layout["families"]),
            "n_object_colors": len(set(colors)),
            "palette_target": plan["palette_target"],
            "palette_widened": widened,
            "mono_objects": bool(plan["mono"]),
            "colors_repeat": bool(len(set(colors)) < len(colors)),
            "has_contact": layout["contacts"] > 0,
            "differs_under_farthest_first": bool(alt != packed),
            "differs_under_reading_order": bool(row_major != packed),
            "differs_under_color_blind": bool(blind_differs),
            "object_budget": plan["object_budget"],
            "bg_color": plan["bg"],
            "background_margin": round(float(info["count"]) / max(1, int(info["runner_up_count"])), 3),
            "occupancy": round(occ_cells / plan["area"], 4),
            "object_cells": occ_cells,
            "planned_density": round(plan["density"], 4),
            "attempts": attempt,
            **tex,
        }
        example = {"input": rows_in, "output": rows_out, "metadata": meta}
        validate_example(example)
        return example
    raise RuntimeError(f"task 91: no valid scene after {MAX_ATTEMPTS} attempts")


@register("91")
def generate_pack_against_edge(rng: np.random.Generator) -> dict:
    """Task 91: slide every object against the named edge, stacking as it goes."""
    return _example(rng)
