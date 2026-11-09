"""Task 92 -- how many objects touch another object (4- or 8-connected per example).

Task 92b is merged into this task on 2026-09-15.  ``_p92(conn=8)`` was already
byte-identical to ``_p92b``, so the two ids were one rule with the connectivity
frozen into the task id -- exactly the pattern that was merged for 19/18,
21/20, 46b/46a, 48/47, 69a/68a and 69b/68b.

What the review found and what this module does about it:

* **The prompt was a single question with no object rule.**  It never said
  "ignoring the background" (treating the ground as an object changes the
  answer on 100% of grids) and never said objects are *same-colour* groups
  (a colour-blind reader differs on 87%), and it had no answer-format
  sentence.  The prompt now carries the standard object rule
  (``prompts._largest_object_rule``) followed by the question and
  "Answer with the number."  The text per connectivity is byte-identical
  across examples.
* **The 8-variant was readable off the input.**  The old generator required a
  diagonal-only touch whenever the answer was positive, but only in the
  8-branch, so an input-only classifier named the variant on 76% of examples.
  The scene is now drawn FIRST and the connectivity LAST: nothing in the
  drawing has ever looked at it, so the share of grids with a diagonal-only
  touch is the same under both readings.
* **Load-bearing.**  Scenes are built from three kinds of touching cluster --
  a side-adjacent chain (counts the same under both readings), a corner-only
  chain (0 under 4, all of it under 8) and a same-colour corner-joined pair
  with a different-coloured neighbour on each half (4 under 4, 3 under 8,
  because the two halves are one object when corners connect).  The plan
  predicts BOTH answers and the scene is accepted only when the picture
  realises both, so the acceptance test is symmetric in 4 and 8.
* **Coverage.**  Sides were 8..22 with no grid above 700 cells, a third of the
  pieces had <= 3 cells, occupancy fell from 22% to 5.5%, the answer cap was
  the hard-coded set {0,2,3,4,5,6} and there were no textures.  Sides are now
  5..30 drawn independently, every body is at least ``MIN_BODY`` cells from
  task 90's vocabulary, the cap is derived from the canvas footprint and
  reaches 12, and sparse library textures cover about 30% of eligible grids.
* **The validator re-ran the same function.**  It now re-reads the picture with
  the literal reader, counts the touching objects under both connectivities and
  the colour-blind reading, and raises on any disagreement.

The answer is never 1: touching is symmetric, so an object that touches another
comes with the object it touches.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.backgrounds import apply_background, infer_background, isolated_cell_mask
from common.grid import NUM_COLORS, grid_to_list
# Bodies, the corner join, the literal object reader and the ground reader come
# from task 90's module; this task lives in the same scene family.  The module
# is bound lazily because ``registry`` imports both and a direct
# ``from ... import`` fails when a caller imports task 90's module first.
from generators import border_objects_90 as _b90
from generators.scene_shapes_56_60 import halo, placement_map
from registry import register

MIN_SIDE, MAX_SIDE = 5, 30
MAX_ATTEMPTS = 40
LAYOUT_TRIES = 3
TEXTURE_MIN_SIDE = 8
TEXTURE_P = 0.36          # attempt rate on eligible grids -> ~30% realised
SPECKLE_P = 0.40
BG_MARGIN = 1.25
MIN_BODY = 5              # the confetti floor, task 90's
MAX_ANSWER = 12
MAX_OCCUPANCY = 0.40
ZERO_P = 0.08
EXTRA_CORNER_P = 0.45     # 4-anchored plans that also carry a corner-only chain

_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# ---------------------------------------------------------------------------
# The rule, read off the picture
# ---------------------------------------------------------------------------

def solve(grid, connectivity: int, *, color_blind: bool = False) -> int:
    """Objects with at least one cell adjacent to a cell of a DIFFERENT object.

    A literal reading of the prompt: set the inferred ground aside, group the
    rest into same-colour components under the stated connectivity, drop the
    one-cell components (background texture), and count the components that
    have a neighbour -- under that same connectivity -- belonging to another
    component.  Objects, not pairs.
    """
    comps = _b90._objects(grid, connectivity, color_blind=color_blind)
    owner: dict = {}
    for i, comp in enumerate(comps):
        for cell in comp:
            owner[cell] = i
    steps = _ORTH if connectivity == 4 else _ORTH + _DIAG
    touching: set = set()
    for (r, c), i in owner.items():
        for dr, dc in steps:
            j = owner.get((r + dr, c + dc))
            if j is not None and j != i:
                touching.add(i)
                touching.add(j)
    return len(touching)


def count_objects(grid, connectivity: int) -> int:
    return len(_b90._objects(grid, connectivity))


def has_diagonal_only_touch(grid) -> bool:
    """True when two objects touch at a corner and nowhere along a side.

    Read with the finest decomposition (4-connected, colours separate), so it
    describes the PICTURE and not a reading of it: the old generator forced
    this to be true exactly when the 8-variant had a positive answer, which is
    how the variant leaked.
    """
    comps = _b90._objects(grid, 4)
    owner: dict = {}
    for i, comp in enumerate(comps):
        for cell in comp:
            owner[cell] = i
    side: set = set()
    diag: set = set()
    for (r, c), i in owner.items():
        for dr, dc in _ORTH:
            j = owner.get((r + dr, c + dc))
            if j is not None and j != i:
                side.add((min(i, j), max(i, j)))
        for dr, dc in _DIAG:
            j = owner.get((r + dr, c + dc))
            if j is not None and j != i:
                diag.add((min(i, j), max(i, j)))
    return any(pair not in side for pair in diag)


# ---------------------------------------------------------------------------
# The plan (nothing here knows the connectivity)
# ---------------------------------------------------------------------------

def _piece_budget(h: int, w: int, density: float) -> int:
    """Bodies the canvas can carry: the ink budget, capped by the footprint.

    A ``MIN_BODY`` body plus its moat eats about twelve cells, so ``area // 12``
    is the footprint ceiling and ``density * area / MIN_BODY`` the ink one.
    """
    area = h * w
    return int(max(2, min(20, max(2, area // 12), max(2, int(density * area / MIN_BODY)))))


def _answer_cap(h: int, w: int, density: float) -> int:
    """How many objects could be touching at once on this canvas."""
    return int(min(MAX_ANSWER, max(2, (3 * _piece_budget(h, w, density)) // 4)))


def _compose(rng: np.random.Generator, target: int, anchor: int) -> Optional[list]:
    """Split the target answer into touching clusters under the anchor reading.

    A cluster is ``(kind, size)``; its contribution is ``(under 4, under 8)``:
    ``side`` k -> (k, k), ``corner`` k -> (0, k), ``merge`` 4 -> (4, 3).
    """
    clusters: list = []
    left = target
    while left > 0:
        options: list = []
        for k in range(2, min(left, 4) + 1):
            options.append(("side", k, k))
            if anchor == 8:
                options.append(("corner", k, k))
        if anchor == 4 and left >= 4:
            options += [("merge", 4, 4)] * 2
        if anchor == 8 and left >= 3:
            options += [("merge", 4, 3)] * 2
        options = [o for o in options if o[2] == left or left - o[2] >= 2]
        if not options:
            return None
        kind, size, cost = options[int(rng.integers(len(options)))]
        clusters.append((kind, size))
        left -= cost
    if anchor == 8 and clusters and all(k == "corner" for k, _ in clusters) \
            and rng.random() < 0.65:
        # an 8-anchored plan made only of corner chains answers zero under 4,
        # which would pile the 4-reading's zeros far above the ~10% budget
        i = int(rng.integers(len(clusters)))
        clusters[i] = ("side", clusters[i][1])
    return clusters


def _contribution(clusters) -> tuple[int, int]:
    a4 = a8 = 0
    for kind, size in clusters:
        if kind == "side":
            a4 += size
            a8 += size
        elif kind == "corner":
            a8 += size
        else:
            a4 += 4
            a8 += 3
    return a4, a8


def sample_plan(rng: np.random.Generator) -> Optional[dict]:
    h = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    w = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    area = h * w
    density = float(rng.uniform(0.14, 0.30))
    budget = _piece_budget(h, w, density)
    cap = _answer_cap(h, w, density)

    # Which reading the answer histogram is drawn flat over.  It is a plan-level
    # coin, not the connectivity -- that does not exist yet.
    anchor = 4 if rng.random() < 0.5 else 8
    target = 0 if rng.random() < ZERO_P else int(rng.integers(2, cap + 1))
    clusters = _compose(rng, target, anchor) if target else []
    if clusters is None:
        return None
    palette_target = int(rng.choice(np.arange(2, 11),
                                    p=np.array([.07, .09, .10, .11, .11, .12, .12, .13, .15])))
    if anchor == 4 and palette_target >= 3 and rng.random() < EXTRA_CORNER_P:
        # corner-only chains cost the 4-answer nothing and are what makes the
        # 8-reading differ on a 4-anchored plan
        for _ in range(1 if rng.random() < 0.72 else 2):
            clusters.append(("corner", int(rng.integers(2, 4))))
    used = sum(4 if kind == "merge" else size for kind, size in clusters)
    if used > budget:
        return None
    need = max([3 if kind == "merge" else size for kind, size in clusters], default=1)
    room = max(0, budget - used)
    n_decoys = int(rng.integers(0, min(6, room) + 1)) if room else 0
    pieces = used + n_decoys
    if pieces < 2:
        return None          # a lone object is not a scene

    n_obj_colors = int(min(9, max(1, pieces), max(palette_target - 1, need)))
    bg = int(rng.integers(0, NUM_COLORS))
    pool = [c for c in range(NUM_COLORS) if c != bg]
    obj_colors = [int(c) for c in rng.choice(pool, size=n_obj_colors, replace=False)]
    per_piece = float(np.clip(density * area / max(1, pieces),
                              float(MIN_BODY), max(float(MIN_BODY) + 1.0, min(0.16 * area, 80.0))))
    a4, a8 = _contribution(clusters)
    return {
        "h": h, "w": w, "area": area, "density": density, "piece_budget": budget,
        "answer_cap": cap, "answer_anchor": anchor, "target": target,
        "clusters": clusters, "n_decoys": n_decoys, "n_pieces": pieces,
        "per_piece": per_piece, "palette_target": palette_target,
        "planned_colors": n_obj_colors, "bg": bg, "obj_colors": obj_colors,
        "answer_4": a4, "answer_8": a8,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

class _Canvas:
    def __init__(self, h: int, w: int, bg: int):
        self.h, self.w, self.bg = h, w, bg
        self.grid = np.full((h, w), bg, dtype=np.int16)
        self.occ = np.zeros((h, w), dtype=bool)
        self.pieces: list = []

    def seat(self, rng, fig: np.ndarray, color: int, *,
             attach: Optional[np.ndarray] = None, corner: bool = False,
             clear8: Optional[np.ndarray] = None,
             clear4: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Place one body and return its full-grid mask.

        ``clear8`` is what the body must keep a one-cell moat from, ``clear4``
        what it may only touch at a corner, and ``attach`` what it must touch --
        along a side, or only at a corner when ``corner`` is set.
        """
        fh, fw = fig.shape
        if fh > self.h or fw > self.w:
            return None
        free = ~self.occ
        if clear8 is not None:
            free &= ~halo(clear8)
        if clear4 is not None:
            free &= ~halo(clear4, diagonal=False)
        ok = placement_map(fig, free)
        if ok is None:
            return None
        if attach is not None:
            if corner:
                field = halo(attach) & ~halo(attach, diagonal=False)
            else:
                field = halo(attach, diagonal=False) & ~attach
            field = field & free
            if not field.any():
                return None
            ok = ok & _b90._shifted_any(field, fig, *ok.shape)
        spots = np.argwhere(ok)
        if not len(spots):
            return None
        r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
        cells = np.zeros((self.h, self.w), dtype=bool)
        cells[r:r + fh, c:c + fw] = fig
        self.grid[cells] = color
        self.occ |= cells
        self.pieces.append({"mask": cells, "color": color})
        return cells


def _make_body(rng: np.random.Generator, plan: dict) -> Optional[tuple]:
    h, w = plan["h"], plan["w"]
    lim_h = max(1, min(h, 14))
    lim_w = max(1, min(w, 14))
    room = lim_h * lim_w
    floor = MIN_BODY if room > MIN_BODY else max(2, room)
    side = float(np.sqrt(plan["per_piece"]))
    bh = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_h))
    bw = int(np.clip(round(side * float(rng.uniform(0.75, 1.7))), 1, lim_w))
    for _ in range(8):
        if bh * bw > floor:
            break
        if bh < lim_h and (bh <= bw or bw >= lim_w):
            bh += 1
        elif bw < lim_w:
            bw += 1
        else:
            break
    return _b90._body(rng, bh, bw, min_cells=floor)


def _pick_colors(rng: np.random.Generator, pool: list, used: list, n: int) -> list:
    """``n`` distinct colours from the pool, preferring ones not yet used."""
    fresh = [c for c in pool if c not in used]
    stale = [c for c in pool if c in used]
    fresh = [fresh[int(i)] for i in rng.permutation(len(fresh))]
    stale = [stale[int(i)] for i in rng.permutation(len(stale))]
    chosen = (fresh + stale)[:n]
    if len(chosen) < n:
        return []
    for c in chosen:
        if c not in used:
            used.append(c)
    return [int(c) for c in chosen]


def build_scene(rng: np.random.Generator, plan: dict) -> Optional[_Canvas]:
    canvas = _Canvas(plan["h"], plan["w"], plan["bg"])
    pool = list(plan["obj_colors"])
    used: list = []
    order = list(plan["clusters"])
    order = [order[int(i)] for i in rng.permutation(len(order))]
    for kind, size in order:
        outside = canvas.occ.copy()
        if kind == "merge":
            colors = _pick_colors(rng, pool, used, 3)
            if len(colors) < 3:
                return None
            masks = []
            for _ in range(2):
                built = _make_body(rng, plan)
                if built is None:
                    return None
                masks.append(built[0])
            joined = _b90._corner_join(rng, masks)
            if joined is None:
                return None
            fig, lab = joined
            seated = canvas.seat(rng, fig, colors[0], clear8=outside)
            if seated is None:
                return None
            rows, cols = np.nonzero(seated)
            r0, c0 = int(rows.min()), int(cols.min())
            halves = []
            for idx in (1, 2):
                part = np.zeros((plan["h"], plan["w"]), dtype=bool)
                part[r0:r0 + lab.shape[0], c0:c0 + lab.shape[1]] = lab == idx
                halves.append(part)
            for half, other, color in ((halves[0], halves[1], colors[1]),
                                       (halves[1], halves[0], colors[2])):
                built = _make_body(rng, plan)
                if built is None:
                    return None
                got = canvas.seat(rng, built[0], color,
                                  attach=half, clear8=outside, clear4=other)
                if got is None:
                    return None
                outside = outside | got  # the second neighbour keeps clear of the first
            continue
        colors = _pick_colors(rng, pool, used, size)
        if len(colors) < size:
            return None
        cluster = np.zeros((plan["h"], plan["w"]), dtype=bool)
        previous = None
        for i in range(size):
            built = _make_body(rng, plan)
            if built is None:
                return None
            fig = built[0]
            if i == 0:
                got = canvas.seat(rng, fig, colors[i], clear8=outside)
            elif kind == "side":
                got = canvas.seat(rng, fig, colors[i], attach=previous, clear8=outside)
            else:
                got = canvas.seat(rng, fig, colors[i], attach=previous, corner=True,
                                  clear8=outside, clear4=cluster)
            if got is None:
                return None
            cluster = cluster | got
            previous = got
    for _ in range(plan["n_decoys"]):
        built = _make_body(rng, plan)
        if built is None:
            break
        colors = _pick_colors(rng, pool, used, 1)
        if not colors:
            break
        if canvas.seat(rng, built[0], colors[0], clear8=canvas.occ) is None:
            break          # out of room for decoys; they change no answer
    if int(canvas.occ.sum()) > MAX_OCCUPANCY * plan["area"]:
        return None
    return canvas


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

_NO_TEXTURE = {"texture_group": None, "texture_kind": None, "texture_colors": [],
               "texture_named": False, "texture_coverage": 0.0}


def _texture(rng: np.random.Generator, canvas: _Canvas):
    """Sparse library texture, one cell clear of every object.

    Only sparse kinds: their cells are isolated single cells, which the object
    rule already disposes of.  A solid band would read as an object and would
    change the answer.
    """
    h, w = canvas.h, canvas.w
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= TEXTURE_P:
        return canvas.grid, dict(_NO_TEXTURE)
    kind = "multi_speckle" if rng.random() < SPECKLE_P else None
    painted = apply_background(rng, canvas.grid, canvas.bg, halo(canvas.occ),
                               group="sparse", kind=kind)
    if painted is None:
        return canvas.grid, dict(_NO_TEXTURE)
    display, info = painted
    return display, {"texture_group": info["group"], "texture_kind": info["kind"],
                     "texture_colors": [int(c) for c in info["texture_colors"]],
                     "texture_named": False, "texture_coverage": float(info["coverage"])}


# ---------------------------------------------------------------------------
# Validation -- everything recomputed from the emitted grid
# ---------------------------------------------------------------------------

def validate_example(example: dict) -> None:
    grid, meta = example["input"], example["metadata"]
    conn = int(meta["connectivity"])
    other = 8 if conn == 4 else 4
    if _b90.read_ground(grid) != meta["bg_color"]:
        raise ValueError("task 92: the visible ground disagrees with the plan")
    answer = solve(grid, conn)
    if str(answer) != example["output"]:
        raise ValueError("task 92: the touching count disagrees with the literal reading")
    if answer != meta["n_touching"]:
        raise ValueError("task 92: the touching count disagrees with the metadata")
    if answer == 1:
        raise ValueError("task 92: touching is symmetric, so the answer is never one")
    if answer > meta["answer_cap"]:
        raise ValueError("task 92: the answer is above the canvas cap")
    if solve(grid, other) != meta["answer_other_connectivity"]:
        raise ValueError("task 92: the other-connectivity answer is not the recorded one")
    if (meta["answer_other_connectivity"] != answer) != meta["differs_under_other_connectivity"]:
        raise ValueError("task 92: the load-bearing flag is wrong")
    merged = solve(grid, conn, color_blind=True)
    if merged != meta["answer_color_blind"]:
        raise ValueError("task 92: the colour-blind answer is not the recorded one")
    if (merged != answer) != meta["differs_under_color_blind"]:
        raise ValueError("task 92: the colour-separation flag is wrong")
    comps = _b90._objects(grid, conn)
    if len(comps) != meta["n_objects"]:
        raise ValueError("task 92: the object count disagrees with the metadata")
    pieces4 = comps if conn == 4 else _b90._objects(grid, 4)
    if sorted(len(comp) for comp in pieces4) != list(meta["piece_sizes"]):
        raise ValueError("task 92: the piece sizes disagree with the visible grid")
    if has_diagonal_only_touch(grid) != meta["has_diagonal_only_touch"]:
        raise ValueError("task 92: the diagonal-only-touch flag is wrong")
    arr = np.asarray(grid)
    info = infer_background(arr, margin=BG_MARGIN)
    if not info["identifiable"] or int(info["color"]) != meta["bg_color"]:
        raise ValueError("task 92: the background is not identifiable")
    if meta["texture_colors"]:
        tex = np.isin(arr, meta["texture_colors"])
        if not bool(isolated_cell_mask(arr)[tex].all()):
            raise ValueError("task 92: a texture cell is not isolated")
        if set(meta["texture_colors"]) & set(meta["object_colors"] + [meta["bg_color"]]):
            raise ValueError("task 92: a texture colour is an object or ground colour")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _example(rng: np.random.Generator) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        plan = sample_plan(rng)
        if plan is None:
            continue
        canvas = None
        for _ in range(LAYOUT_TRIES):
            canvas = build_scene(rng, plan)
            if canvas is not None:
                break
        if canvas is None:
            continue
        display, tex = _texture(rng, canvas)
        grid = grid_to_list(display.astype(np.int16))

        # --- nothing above this line has looked at the connectivity ---------
        if _b90.read_ground(grid) != plan["bg"]:
            continue
        info = infer_background(display, margin=BG_MARGIN)
        if not info["identifiable"] or int(info["color"]) != plan["bg"]:
            continue
        answer4 = solve(grid, 4)
        answer8 = solve(grid, 8)
        # The acceptance test is symmetric in the two readings, so the scene
        # distribution cannot depend on the connectivity drawn below.
        if (answer4, answer8) != (plan["answer_4"], plan["answer_8"]):
            continue
        if max(answer4, answer8) > plan["answer_cap"]:
            continue
        # a decoy that could not be seated may leave a single object on the
        # canvas; the question needs something to compare against
        if min(count_objects(grid, 4), count_objects(grid, 8)) < 2:
            continue

        conn = 4 if rng.random() < 0.5 else 8
        other = 8 if conn == 4 else 4
        answer = answer4 if conn == 4 else answer8
        alt = answer8 if conn == 4 else answer4
        comps = _b90._objects(grid, conn)
        pieces4 = comps if conn == 4 else _b90._objects(grid, 4)
        piece_sizes = sorted(len(comp) for comp in pieces4)
        merged = solve(grid, conn, color_blind=True)
        occ_cells = int(canvas.occ.sum())
        present = sorted({int(v) for v in np.unique(display)})
        kinds = [kind for kind, _ in plan["clusters"]]
        meta = {
            "grid_h": plan["h"], "grid_w": plan["w"], "n_colors": len(present),
            "connectivity": conn,
            "n_touching": answer,
            "answer_cap": plan["answer_cap"],
            "answer_other_connectivity": alt,
            "differs_under_other_connectivity": bool(alt != answer),
            "answer_color_blind": merged,
            "differs_under_color_blind": bool(merged != answer),
            "answer_anchor": plan["answer_anchor"],
            "answer_4": answer4, "answer_8": answer8,
            "n_objects": len(comps),
            "n_objects_other_connectivity": len(_b90._objects(grid, other)),
            "n_clusters": len(plan["clusters"]),
            "cluster_kinds": kinds,
            "cluster_sizes": [size for _, size in plan["clusters"]],
            "n_decoys": plan["n_decoys"],
            "n_pieces": len(canvas.pieces),
            "has_diagonal_only_touch": has_diagonal_only_touch(grid),
            "piece_sizes": piece_sizes,
            "min_piece_cells": piece_sizes[0] if piece_sizes else 0,
            "object_colors": sorted({int(p["color"]) for p in canvas.pieces}),
            "object_sizes": sorted(int(p["mask"].sum()) for p in canvas.pieces),
            "piece_budget": plan["piece_budget"],
            "palette_target": plan["palette_target"],
            "bg_color": plan["bg"],
            "background_margin": round(float(info["count"]) / max(1, int(info["runner_up_count"])), 3),
            "occupancy": round(occ_cells / plan["area"], 4),
            "object_cells": occ_cells,
            "planned_density": round(plan["density"], 4),
            "attempts": attempt,
            **tex,
        }
        example = {"input": grid, "output": str(answer), "metadata": meta}
        validate_example(example)
        return example
    raise RuntimeError(f"task 92: no valid scene after {MAX_ATTEMPTS} attempts")


@register("92")
def generate_count_touching(rng: np.random.Generator) -> dict:
    """Task 92: how many objects touch at least one object of a different colour group."""
    return _example(rng)
