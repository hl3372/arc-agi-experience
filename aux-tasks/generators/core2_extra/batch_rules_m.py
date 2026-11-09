"""Finite rule-inference and priority-rule generators for core2 batch M."""
from __future__ import annotations

import copy
from collections import deque

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from common.objects import random_shape
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    place_shape,
    unused_colors,
)
from registry import register


Cell = tuple[int, int]


MOTION_RULES: list[dict] = [
    {"label": "motion_shift_north_2", "dr": -2, "dc": 0},
    {"label": "motion_shift_south_2", "dr": 2, "dc": 0},
    {"label": "motion_shift_west_2", "dr": 0, "dc": -2},
    {"label": "motion_shift_east_2", "dr": 0, "dc": 2},
    {"label": "motion_shift_northwest_2", "dr": -2, "dc": -2},
    {"label": "motion_shift_northeast_2", "dr": -2, "dc": 2},
    {"label": "motion_shift_southwest_2", "dr": 2, "dc": -2},
    {"label": "motion_shift_southeast_2", "dr": 2, "dc": 2},
]

MOTION_SHAPES: list[frozenset[Cell]] = [
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 0), (0, 1), (1, 1), (2, 1)}),
]

SUBSTITUTION_RULES: list[dict] = [
    {"label": "mapping_A", "role_map": [1, 0, 2], "mode": "swap_primary_secondary"},
    {"label": "mapping_B", "role_map": [1, 2, 0], "mode": "cycle_primary_secondary_accent"},
    {"label": "mapping_C", "role_map": [2, 0, 1], "mode": "cycle_primary_accent_secondary"},
    {"label": "mapping_D", "role_map": [2, 1, 0], "mode": "swap_primary_accent"},
]

BORDER_PARAMETER_RULES: list[dict] = [
    {"label": "border_width_1", "border_width": 1},
    {"label": "border_width_2", "border_width": 2},
    {"label": "border_width_3", "border_width": 3},
]

CANDIDATE_RULE_LABELS = [
    "rule_flip_horizontal",
    "rule_flip_vertical",
    "rule_rotate_180",
    "rule_transpose_main",
]

OBJECT_SELECTION_RULES: list[dict] = [
    {"label": "largest_object_recolor", "selector": "largest_object", "action": "recolor"},
    {"label": "smallest_object_recolor", "selector": "smallest_object", "action": "recolor"},
    {"label": "topmost_object_erase", "selector": "topmost_object", "action": "erase"},
    {"label": "leftmost_object_erase", "selector": "leftmost_object", "action": "erase"},
]

NEIGHBOR_PREDICATES: list[dict] = [
    {"label": "predicate_north_is_target", "kind": "north"},
    {"label": "predicate_south_is_target", "kind": "south"},
    {"label": "predicate_west_is_target", "kind": "west"},
    {"label": "predicate_east_is_target", "kind": "east"},
    {"label": "predicate_at_least_two_cardinal_targets", "kind": "at_least_two"},
]

CELL_RULE_LABELS = [
    "cell_rule_seed_cell",
    "cell_rule_next_to_seed",
    "cell_rule_foreground_cell",
]

CELL_PRIORITY_ORDERS: list[dict] = [
    {
        "label": "priority_seed_neighbor_foreground",
        "order": ["cell_rule_seed_cell", "cell_rule_next_to_seed", "cell_rule_foreground_cell"],
    },
    {
        "label": "priority_neighbor_seed_foreground",
        "order": ["cell_rule_next_to_seed", "cell_rule_seed_cell", "cell_rule_foreground_cell"],
    },
    {
        "label": "priority_foreground_seed_neighbor",
        "order": ["cell_rule_foreground_cell", "cell_rule_seed_cell", "cell_rule_next_to_seed"],
    },
]

OBJECT_RULE_LABELS = [
    "object_rule_large",
    "object_rule_wide",
    "object_rule_tall",
]


def _bbox(cells: set[Cell]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _paint(grid: np.ndarray, cells: set[Cell], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _shift(cells: set[Cell], dr: int, dc: int) -> set[Cell]:
    return {(r + dr, c + dc) for r, c in cells}


def _in_bounds(cells: set[Cell], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _place_shape_at(top: int, left: int, shape: frozenset[Cell]) -> set[Cell]:
    return {(top + r, left + c) for r, c in shape}


def _shape_size(shape: frozenset[Cell]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(set(shape))
    return r1 - r0 + 1, c1 - c0 + 1


def _components(grid: np.ndarray, bg: int, color: int | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            value = int(grid[r, c])
            if value == bg or (color is not None and value != color):
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[Cell] = set()
            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == value
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": value, "cells": cells, "bbox": _bbox(cells)})
    return sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))


def _candidate_grid_rule(label: str, grid: np.ndarray) -> np.ndarray:
    if label == "rule_flip_horizontal":
        return np.fliplr(grid).copy()
    if label == "rule_flip_vertical":
        return np.flipud(grid).copy()
    if label == "rule_rotate_180":
        return np.rot90(grid, 2).copy()
    if label == "rule_transpose_main":
        return grid.T.copy()
    raise ValueError(f"unknown candidate rule {label}")


def _make_asymmetric_square(rng: np.random.Generator, n: int, bg: int, colors: list[int], variant: int) -> np.ndarray:
    grid = make_grid(n, n, fill=bg)
    anchors = [
        (0, 1 + (variant % 2)),
        (1, n - 2),
        (2 + (variant % 2), 0),
        (n - 1, max(1, n - 3)),
        (n - 2, 2),
        (3, n - 1),
    ]
    for idx, (r, c) in enumerate(anchors):
        grid[r, c] = colors[idx % len(colors)]
    grid[n // 2, n // 2] = colors[(variant + 1) % len(colors)]
    return grid


def _all_candidate_rules_distinct(grid: np.ndarray) -> bool:
    """True when the four candidate whole-grid transforms give pairwise-distinct
    outputs, so the rule that produced any single output is unambiguous for this
    grid (flip-h vs flip-v vs rotate-180 vs transpose)."""
    outs = [_candidate_grid_rule(label, grid).tobytes() for label in CANDIDATE_RULE_LABELS]
    return len(set(outs)) == len(outs)


def _make_asymmetric_grid(rng: np.random.Generator, h: int, w: int, bg: int, colors: list[int]) -> np.ndarray | None:
    """Random h x w scene that breaks horizontal, vertical, 180 and diagonal
    symmetry so all four candidate transforms are pairwise distinct. Returns
    None if a distinguishing fill was not found within the attempt budget."""
    n_marks = max(4, (h * w) // 3)
    for _ in range(40):
        grid = make_grid(h, w, fill=bg)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        for idx in range(min(n_marks, len(cells))):
            r, c = cells[idx]
            grid[r, c] = int(colors[idx % len(colors)])
        if _all_candidate_rules_distinct(grid):
            return grid
    return None


def _mapping_tables(role_colors: list[int]) -> list[dict]:
    tables = []
    for rule in SUBSTITUTION_RULES:
        tables.append({
            "label": rule["label"],
            "mode": rule["mode"],
            "entries": [
                {"from_color": int(role_colors[src]), "to_color": int(role_colors[dst])}
                for src, dst in enumerate(rule["role_map"])
            ],
        })
    return tables


# A small bank of repeating motif tiles. Each uses all three roles (0,1,2) and
# at least one background cell (-1), but lays them out differently so the tiled
# connected-component shapes vary across samples. The cellwise three-role
# mapping is identical for every motif, so widening the bank only adds visual
# diversity, never ambiguity.
MOTIF_TILES: list[np.ndarray] = [
    np.array([[0, 1, -1], [2, 0, 1], [-1, 2, 0], [1, -1, 2]], dtype=int),
    np.array([[0, 0, 1, -1], [2, -1, 1, 1], [2, 2, -1, 0]], dtype=int),
    np.array([[0, -1, 1], [-1, 2, -1], [1, -1, 0], [-1, 2, -1]], dtype=int),
    np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=int),
    np.array([[0, 0, -1, 2], [1, -1, 2, 2], [1, 1, -1, 0]], dtype=int),
]


def _make_motif_grid(rng: np.random.Generator, h: int, w: int, bg: int, role_colors: list[int]) -> np.ndarray:
    motif = MOTIF_TILES[int(rng.integers(0, len(MOTIF_TILES)))]
    row_shift = int(rng.integers(0, motif.shape[0]))
    col_shift = int(rng.integers(0, motif.shape[1]))
    grid = make_grid(h, w, fill=bg)
    for r in range(h):
        for c in range(w):
            role = int(motif[(r + row_shift) % motif.shape[0], (c + col_shift) % motif.shape[1]])
            if role >= 0:
                grid[r, c] = role_colors[role]
    return grid


def _motif_has_all_roles(grid: np.ndarray, bg: int, role_colors: list[int]) -> bool:
    """Every role color must appear so the observed cellwise mapping is total and
    exactly one candidate table reproduces it (no tie from an absent role)."""
    present = {int(v) for v in np.unique(grid)}
    return all(int(color) in present for color in role_colors)


def _apply_substitution(grid: np.ndarray, bg: int, role_colors: list[int], role_map: list[int]) -> np.ndarray:
    out = grid.copy()
    color_to_role = {color: idx for idx, color in enumerate(role_colors)}
    for src_color, role in color_to_role.items():
        out[grid == src_color] = role_colors[role_map[role]]
    out[grid == bg] = bg
    return out


def _select_object(comps: list[dict], selector: str) -> dict:
    if selector == "largest_object":
        return max(comps, key=lambda comp: (len(comp["cells"]), -comp["bbox"][0], -comp["bbox"][1]))
    if selector == "smallest_object":
        return min(comps, key=lambda comp: (len(comp["cells"]), comp["bbox"][0], comp["bbox"][1]))
    if selector == "topmost_object":
        return min(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    if selector == "leftmost_object":
        return min(comps, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    raise ValueError(f"unknown selector {selector}")


def _object_selection_scene(
    rng: np.random.Generator,
    bg: int,
    object_color: int,
    n_objects: int,
) -> np.ndarray | None:
    """Place `n_objects` same-color objects with distinct sizes so that the
    largest, smallest, topmost and leftmost objects are distinct components.

    H and W are sampled independently across the ARC-typical small range, scaled
    up with the object count so placement keeps succeeding. Background stays
    solid: selection works over the 4-connected components of all non-background
    cells, so any texture would inject spurious objects (the task-154 lesson)."""
    base = 4 + n_objects
    h = int(rng.integers(base, base + 7))
    w = int(rng.integers(base, base + 7))
    grid = make_grid(h, w, fill=bg)
    # n distinct cell-counts so largest/smallest are unambiguous singletons
    sizes = sorted(int(s) for s in rng.choice(np.arange(2, 9), size=n_objects, replace=False))
    boxes: list[tuple[int, int, int, int]] = []
    for size in sizes:
        shape = random_shape(rng, (size, size), connectivity=4)
        cells = place_shape(grid, rng, shape, int(object_color), bg, boxes, gap=1)
        if cells is None:
            return None
    comps = _components(grid, bg, color=object_color)
    if len(comps) != n_objects:
        return None
    # Distinct sizes => largest/smallest objects are unambiguous singletons.
    if len({len(comp["cells"]) for comp in comps}) != n_objects:
        return None
    # The two erase rules (topmost, leftmost) share an action, so their outputs
    # only differ when they select different objects. Requiring this guarantees
    # all four candidate rules produce pairwise-distinct outputs in this example,
    # which makes the cross-example inference uniquely solvable. (largest/smallest
    # are recolor and always differ from each other and from the erase rules.)
    topmost = _select_object(comps, "topmost_object")
    leftmost = _select_object(comps, "leftmost_object")
    if topmost is leftmost:
        return None
    return grid


def _apply_object_selection(
    grid: np.ndarray,
    bg: int,
    object_color: int,
    action_color: int,
    rule: dict,
) -> np.ndarray:
    comps = _components(grid, bg, color=object_color)
    selected = _select_object(comps, rule["selector"])
    out = grid.copy()
    replacement = bg if rule["action"] == "erase" else action_color
    _paint(out, selected["cells"], replacement)
    return out


def _predicate_mask(grid: np.ndarray, target_color: int, true_color: int, false_color: int, kind: str) -> np.ndarray:
    h, w = grid.shape
    out = make_grid(h, w, fill=false_color)
    for r in range(h):
        for c in range(w):
            neighbors = {
                "north": r > 0 and int(grid[r - 1, c]) == target_color,
                "south": r + 1 < h and int(grid[r + 1, c]) == target_color,
                "west": c > 0 and int(grid[r, c - 1]) == target_color,
                "east": c + 1 < w and int(grid[r, c + 1]) == target_color,
            }
            if kind == "at_least_two":
                ok = sum(neighbors.values()) >= 2
            else:
                ok = neighbors[kind]
            if ok:
                out[r, c] = true_color
    return out


def _make_cell_rules(bg: int, seed_color: int, seed_result: int, neighbor_result: int, foreground_result: int) -> list[dict]:
    return [
        {
            "label": "cell_rule_seed_cell",
            "condition": "cell_is_seed_color",
            "match_color": int(seed_color),
            "output_color": int(seed_result),
        },
        {
            "label": "cell_rule_next_to_seed",
            "condition": "has_cardinal_neighbor_seed_color",
            "neighbor_color": int(seed_color),
            "output_color": int(neighbor_result),
        },
        {
            "label": "cell_rule_foreground_cell",
            "condition": "cell_is_not_background",
            "background_color": int(bg),
            "output_color": int(foreground_result),
        },
    ]


def _cell_rule_matches(grid: np.ndarray, r: int, c: int, rule: dict, bg: int, seed_color: int) -> bool:
    if rule["label"] == "cell_rule_seed_cell":
        return int(grid[r, c]) == seed_color
    if rule["label"] == "cell_rule_next_to_seed":
        h, w = grid.shape
        return any(
            0 <= r + dr < h and 0 <= c + dc < w and int(grid[r + dr, c + dc]) == seed_color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )
    if rule["label"] == "cell_rule_foreground_cell":
        return int(grid[r, c]) != bg
    raise ValueError(f"unknown cell rule {rule['label']}")


def _apply_cell_priority(grid: np.ndarray, bg: int, seed_color: int, ordered_rules: list[dict]) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            for rule in ordered_rules:
                if _cell_rule_matches(grid, r, c, rule, bg, seed_color):
                    out[r, c] = rule["output_color"]
                    break
    return out


def _cell_priority_conflicts(grid: np.ndarray, bg: int, seed_color: int) -> int:
    h, w = grid.shape
    rules = [
        {"label": "cell_rule_seed_cell"},
        {"label": "cell_rule_next_to_seed"},
        {"label": "cell_rule_foreground_cell"},
    ]
    conflicts = 0
    for r in range(h):
        for c in range(w):
            count = sum(
                _cell_rule_matches(grid, r, c, rule, bg, seed_color) for rule in rules
            )
            if count > 1:
                conflicts += 1
    return conflicts


def _cell_priority_scene(
    rng: np.random.Generator,
    bg: int,
    seed_color: int,
    other_color: int,
    seed_result: int,
    neighbor_result: int,
    foreground_result: int,
) -> np.ndarray | None:
    """Randomized scene with seed and other-foreground cells such that all three
    candidate priority orders yield pairwise-distinct outputs and at least three
    cells satisfy more than one rule.

    H and W are sampled independently across the ARC-typical small range; the
    seed/other-foreground populations scale with the area so larger grids stay
    dense enough to keep producing rule conflicts (the binding constraint)."""
    h = int(rng.integers(6, 16))
    w = int(rng.integers(6, 16))
    grid = make_grid(h, w, fill=bg)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)

    # Guarantee at least one adjacent seed pair (distinguishes seed-first vs
    # neighbor-first orders) by seeding a horizontal or vertical domino.
    r0 = int(rng.integers(0, h - 1))
    c0 = int(rng.integers(0, w - 1))
    if rng.random() < 0.5:
        seed_cells = {(r0, c0), (r0, c0 + 1)}
    else:
        seed_cells = {(r0, c0), (r0 + 1, c0)}

    area = h * w
    n_extra_seed = int(rng.integers(1, max(4, area // 12 + 1)))
    n_other = int(rng.integers(4, max(8, area // 6 + 1)))
    extra_seed = 0
    other_cells: set[Cell] = set()
    for cell in cells:
        if cell in seed_cells:
            continue
        if extra_seed < n_extra_seed:
            seed_cells.add(cell)
            extra_seed += 1
        elif len(other_cells) < n_other:
            other_cells.add(cell)
        else:
            break

    for cell in seed_cells:
        grid[cell] = seed_color
    for cell in other_cells:
        grid[cell] = other_color

    if _cell_priority_conflicts(grid, bg, seed_color) < 3:
        return None

    all_rules = _make_cell_rules(bg, seed_color, seed_result, neighbor_result, foreground_result)
    outputs = set()
    for order in CELL_PRIORITY_ORDERS:
        ordered = _ordered_cell_rules(all_rules, order["order"])
        outputs.add(_apply_cell_priority(grid, bg, seed_color, ordered).tobytes())
    if len(outputs) != len(CELL_PRIORITY_ORDERS):
        return None
    return grid


def _ordered_cell_rules(all_rules: list[dict], order: list[str]) -> list[dict]:
    by_label = {rule["label"]: rule for rule in all_rules}
    return [by_label[label] for label in order]


def _make_object_rules(large_color: int, wide_color: int, tall_color: int) -> list[dict]:
    return [
        {
            "label": "object_rule_large",
            "condition": "cell_count_at_least",
            "threshold": 5,
            "output_color": int(large_color),
        },
        {
            "label": "object_rule_wide",
            "condition": "bounding_box_width_exceeds_height",
            "output_color": int(wide_color),
        },
        {
            "label": "object_rule_tall",
            "condition": "bounding_box_height_exceeds_width",
            "output_color": int(tall_color),
        },
    ]


def _object_rule_matches(comp: dict, rule: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if rule["label"] == "object_rule_large":
        return len(comp["cells"]) >= int(rule["threshold"])
    if rule["label"] == "object_rule_wide":
        return width > height
    if rule["label"] == "object_rule_tall":
        return height > width
    raise ValueError(f"unknown object rule {rule['label']}")


def _apply_object_priority(grid: np.ndarray, bg: int, ordered_rules: list[dict]) -> np.ndarray:
    out = grid.copy()
    for comp in _components(grid, bg):
        for rule in ordered_rules:
            if _object_rule_matches(comp, rule):
                _paint(out, comp["cells"], rule["output_color"])
                break
    return out


def _priority_object_shape(rng: np.random.Generator, kind: str) -> frozenset[Cell]:
    """Build a relative-cell shape with a controlled large/wide/tall classification.

    Classification recomputed by the caller; these builders only aim for it.
      large_wide  : >=5 cells, bbox width  > height
      large_tall  : >=5 cells, bbox height > width
      large_square: >=5 cells, bbox width == height (neither wide nor tall)
      wide_small  : <5 cells, width  > height
      tall_small  : <5 cells, height > width
      no_rule     : <5 cells, width == height (matches no rule)
    """
    if kind == "large_wide":
        h = int(rng.integers(2, 4))
        w = int(rng.integers(h + 1, h + 4))
        return frozenset((r, c) for r in range(h) for c in range(w))
    if kind == "large_tall":
        w = int(rng.integers(2, 4))
        h = int(rng.integers(w + 1, w + 4))
        return frozenset((r, c) for r in range(h) for c in range(w))
    if kind == "large_square":
        n = int(rng.integers(3, 5))  # solid n x n, n>=3 -> >=9 cells
        return frozenset((r, c) for r in range(n) for c in range(n))
    if kind == "wide_small":
        w = int(rng.integers(2, 5))  # 1 x w line, <5 cells
        return frozenset((0, c) for c in range(w))
    if kind == "tall_small":
        h = int(rng.integers(2, 5))  # h x 1 line, <5 cells
        return frozenset((r, 0) for r in range(h))
    if kind == "no_rule":
        return frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})  # 2x2, <5 cells, square
    raise ValueError(f"unknown object kind {kind}")


def _object_priority_scene(
    rng: np.random.Generator,
    bg: int,
    object_color: int,
    n_objects: int,
) -> np.ndarray | None:
    """`n_objects` same-color objects with at least two that match more than one
    rule (a large object that is also wide/tall), so priority order is
    observable. H and W are sampled independently; the canvas grows with the
    object count so placement keeps succeeding. Background stays solid: the rule
    classifies every non-background 4-connected object, so any texture would add
    spurious components and corrupt the object set."""
    base = 4 + 2 * n_objects
    h = int(rng.integers(base, base + 5))
    w = int(rng.integers(base, base + 5))
    grid = make_grid(h, w, fill=bg)
    # Always include the two conflicting kinds (large+wide, large+tall); fill the
    # remaining slots from a randomized pool of unambiguous kinds.
    pool = ["large_square", "wide_small", "tall_small", "no_rule"]
    kinds = ["large_wide", "large_tall"]
    kinds += [pool[int(rng.integers(0, len(pool)))] for _ in range(n_objects - 2)]
    rng.shuffle(kinds)
    boxes: list[tuple[int, int, int, int]] = []
    for kind in kinds:
        shape = _priority_object_shape(rng, kind)
        cells = place_shape(grid, rng, shape, int(object_color), bg, boxes, gap=1)
        if cells is None:
            return None
    comps = _components(grid, bg, color=object_color)
    if len(comps) != n_objects:
        return None
    rules = _make_object_rules(0, 0, 0)
    conflicts = sum(
        1 for comp in comps if sum(_object_rule_matches(comp, rule) for rule in rules) > 1
    )
    if conflicts < 2:
        return None
    return grid


@register("221")
def generate_infer_motion_model_from_pair(rng: np.random.Generator) -> dict:
    """Infer a finite translation label from before/after named grids."""
    rule = MOTION_RULES[int(rng.integers(0, len(MOTION_RULES)))]
    dr, dc = int(rule["dr"]), int(rule["dc"])
    for _ in range(MAX_RETRIES):
        # H and W are sampled independently from the ARC-typical small span; the
        # lower bound fits a 3x3 shape plus the 2-cell shift plus a 1-cell border
        # on both the source and the shifted target. Background stays solid: the
        # rule translates *every* non-background cell, so any patterned texture
        # would itself have to move and would break the single-shift recovery.
        h = int(rng.integers(8, 19))
        w = int(rng.integers(8, 19))
        n_objects = int(rng.integers(3, 6))
        bg = int(rng.integers(0, NUM_COLORS))
        object_colors = unused_colors(rng, n_objects, bg)
        before = make_grid(h, w, fill=bg)
        after = make_grid(h, w, fill=bg)
        blocked: set[Cell] = set()
        placements: list[dict] = []
        for idx in range(n_objects):
            shape = MOTION_SHAPES[(idx + int(rng.integers(0, len(MOTION_SHAPES)))) % len(MOTION_SHAPES)]
            sh, sw = _shape_size(shape)
            placed = False
            for _attempt in range(200):
                top = int(rng.integers(1, h - sh))
                left = int(rng.integers(1, w - sw))
                src = _place_shape_at(top, left, shape)
                dst = _shift(src, dr, dc)
                if not _in_bounds(src, h, w) or not _in_bounds(dst, h, w):
                    continue
                if src & dst or src & blocked or dst & blocked:
                    continue
                color = object_colors[idx]
                _paint(before, src, color)
                _paint(after, dst, color)
                blocked |= src | dst
                placements.append({"color": int(color), "source_cells": sorted(map(list, src)), "target_cells": sorted(map(list, dst))})
                placed = True
                break
            if not placed:
                break
        if len(placements) == n_objects:
            return {
                "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
                "output": rule["label"],
                "metadata": {
                    "background_color": bg,
                    "motion_family": "translate_all_non_background_cells",
                    "rule_label": rule["label"],
                    "offset_vector": [dr, dc],
                    "candidate_motion_labels": [item["label"] for item in MOTION_RULES],
                    "placements": placements,
                },
            }
    raise RuntimeError("Task 221: failed to place translated objects")


@register("222")
def generate_infer_motif_substitution_from_pair(rng: np.random.Generator) -> dict:
    """Infer a finite motif substitution table label from one before/after pair."""
    rule = SUBSTITUTION_RULES[int(rng.integers(0, len(SUBSTITUTION_RULES)))]
    bg = int(rng.integers(0, NUM_COLORS))
    role_colors = unused_colors(rng, 3, bg)
    # H and W are sampled independently across the small ARC range; the motif
    # must show all three role colors so the single pair pins the table.
    for _attempt in range(MAX_RETRIES):
        h = int(rng.integers(5, 19))
        w = int(rng.integers(5, 19))
        before = _make_motif_grid(rng, h, w, bg, role_colors)
        if _motif_has_all_roles(before, bg, role_colors):
            break
    after = _apply_substitution(before, bg, role_colors, rule["role_map"])
    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": rule["label"],
        "metadata": {
            "background_color": bg,
            "rule_label": rule["label"],
            "substitution_family": "three_role_motif_color_table",
            "role_colors": [int(c) for c in role_colors],
            "mapping_tables": _mapping_tables(role_colors),
        },
    }


@register("244")
def generate_infer_shared_border_width(rng: np.random.Generator) -> dict:
    """Infer the shared hidden border width used in several crop examples."""
    rule = BORDER_PARAMETER_RULES[int(rng.integers(0, len(BORDER_PARAMETER_RULES)))]
    k = int(rule["border_width"])
    bg = int(rng.integers(0, NUM_COLORS))
    colors = [bg] + unused_colors(rng, 4, bg)
    examples = []
    # Grid sizes are sampled independently of k from a fixed envelope so the
    # input dimensions never leak the hidden border width (the answer must be
    # read off the input/output size *difference*, which equals 2*k, not the
    # absolute size). Lower bound 2*K_MAX+2 keeps the crop non-trivial (output
    # side >= 2) for every candidate width; H and W are drawn separately.
    # Upper bound keeps three simultaneously-maximal examples inside the
    # ~2900-token bundle budget (3 examples are packed into one prompt here).
    k_max = max(int(item["border_width"]) for item in BORDER_PARAMETER_RULES)
    lo = 2 * k_max + 2
    for idx in range(3):
        h = int(rng.integers(lo, 18))
        w = int(rng.integers(lo, 18))
        grid = np.array(rng.choice(colors, size=(h, w)), dtype=np.uint8)
        grid[k, k] = colors[(idx + 1) % len(colors)]
        out = grid[k : h - k, k : w - k].copy()
        examples.append({"input": grid_to_list(grid), "output": grid_to_list(out)})
    return {
        "input": {"examples": examples},
        "output": rule["label"],
        "metadata": {
            "family_label": "remove_equal_width_outer_border",
            "rule_label": rule["label"],
            "border_width": k,
            "candidate_parameter_labels": [item["label"] for item in BORDER_PARAMETER_RULES],
            "background_color": bg,
        },
    }


@register("245")
def generate_choose_candidate_rule_from_examples(rng: np.random.Generator) -> dict:
    """Choose the single candidate grid rule consistent with multiple examples."""
    target_label = CANDIDATE_RULE_LABELS[int(rng.integers(0, len(CANDIDATE_RULE_LABELS)))]
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        colors = unused_colors(rng, 5, bg)
        examples = []
        # Each example draws H and W independently from the small ARC range; a
        # transpose example may therefore be non-square (output side-swapped).
        # Every grid is forced to break all four symmetries so each pair on its
        # own pins the rule, and the answer is the rule shared by all examples.
        for _idx in range(3):
            h = int(rng.integers(3, 13))
            w = int(rng.integers(3, 13))
            grid = _make_asymmetric_grid(rng, h, w, bg, colors)
            if grid is None:
                break
            out = _candidate_grid_rule(target_label, grid)
            examples.append({"input": grid_to_list(grid), "output": grid_to_list(out)})
        if len(examples) == 3:
            break
    else:
        raise RuntimeError("Task 245: failed to build distinguishing grid examples")
    labels = list(CANDIDATE_RULE_LABELS)
    rng.shuffle(labels)
    return {
        "input": {"examples": examples, "rule_labels": labels},
        "output": target_label,
        "metadata": {
            "candidate_rule_labels": labels,
            "correct_rule_label": target_label,
            "background_color": bg,
            "rule_family": "whole_grid_geometric_transform",
        },
    }


@register("246")
def generate_infer_shared_object_selection_rule(rng: np.random.Generator) -> dict:
    """Infer shared object-selection and action label from examples."""
    rule = OBJECT_SELECTION_RULES[int(rng.integers(0, len(OBJECT_SELECTION_RULES)))]
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, action_color = unused_colors(rng, 2, bg)
    examples = []
    attempts = 0
    # The object count varies per example (3-5 distinct-size objects) so the
    # scene's component count is not fixed across the task distribution.
    while len(examples) < 3 and attempts < MAX_RETRIES * 4:
        attempts += 1
        n_objects = int(rng.integers(3, 6))
        grid = _object_selection_scene(rng, bg, object_color, n_objects)
        if grid is None:
            continue
        out = _apply_object_selection(grid, bg, object_color, action_color, rule)
        examples.append({"input": grid_to_list(grid), "output": grid_to_list(out)})
    if len(examples) < 3:
        raise RuntimeError("Task 246: failed to build object-selection scenes")
    return {
        "input": {"examples": examples},
        "output": rule["label"],
        "metadata": {
            "background_color": bg,
            "object_color": int(object_color),
            "action_color": int(action_color),
            "rule_label": rule["label"],
            "candidate_rule_labels": [item["label"] for item in OBJECT_SELECTION_RULES],
            "object_selection_rules": copy.deepcopy(OBJECT_SELECTION_RULES),
            "connectivity": 4,
        },
    }


@register("247")
def generate_infer_shared_mapping_table(rng: np.random.Generator) -> dict:
    """Infer the shared mapping table label from several substitution examples."""
    rule = SUBSTITUTION_RULES[int(rng.integers(0, len(SUBSTITUTION_RULES)))]
    bg = int(rng.integers(0, NUM_COLORS))
    role_colors = unused_colors(rng, 3, bg)
    examples = []
    # H and W are sampled independently across the small ARC range. Each example
    # is required to contain all three role colors so the observed mapping is
    # complete and uniquely identifies the table.
    for _idx in range(3):
        for _attempt in range(MAX_RETRIES):
            h = int(rng.integers(5, 15))
            w = int(rng.integers(5, 15))
            before = _make_motif_grid(rng, h, w, bg, role_colors)
            if _motif_has_all_roles(before, bg, role_colors):
                break
        after = _apply_substitution(before, bg, role_colors, rule["role_map"])
        examples.append({"input": grid_to_list(before), "output": grid_to_list(after)})
    return {
        "input": {"examples": examples},
        "output": rule["label"],
        "metadata": {
            "background_color": bg,
            "rule_label": rule["label"],
            "mapping_type": "cellwise_three_role_color_mapping",
            "role_colors": [int(c) for c in role_colors],
            "mapping_tables": _mapping_tables(role_colors),
        },
    }


@register("248")
def generate_infer_local_neighborhood_predicate(rng: np.random.Generator) -> dict:
    """Infer a finite local-neighborhood predicate label from one example pair."""
    rule = NEIGHBOR_PREDICATES[int(rng.integers(0, len(NEIGHBOR_PREDICATES)))]
    for _ in range(MAX_RETRIES):
        # H and W are sampled independently across the ARC-typical small range
        # (>=5 so the seeded adjacencies fit; up to 18 for a single grid).
        h = int(rng.integers(5, 19))
        w = int(rng.integers(5, 19))
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, true_color, false_color = unused_colors(rng, 3, bg)
        grid = make_grid(h, w, fill=bg)
        density = float(rng.uniform(0.22, 0.38))
        for r in range(h):
            for c in range(w):
                if rng.random() < density:
                    grid[r, c] = target_color
        # Guarantee an adjacent target pair plus an isolated target so the five
        # candidate masks reliably differ even on the smaller grids.
        pr = int(rng.integers(0, h))
        pc = int(rng.integers(0, w - 1))
        grid[pr, pc] = target_color
        grid[pr, pc + 1] = target_color
        grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = target_color
        # Sparse dotted/striped background of inert colors (never the target,
        # true or false colors): the predicate only tests target_color adjacency,
        # so these cells never change the computed mask. Recorded in metadata.
        bg_style = add_patterned_background(
            grid,
            rng,
            bg=bg,
            exclude_colors=[target_color, true_color, false_color],
            protected_cells={
                (r, c)
                for r in range(h)
                for c in range(w)
                if int(grid[r, c]) == target_color
            },
            solid_probability=0.4,
        )
        masks = {
            item["label"]: _predicate_mask(grid, target_color, true_color, false_color, item["kind"])
            for item in NEIGHBOR_PREDICATES
        }
        chosen = masks[rule["label"]]
        if len({mask.tobytes() for mask in masks.values()}) != len(masks):
            continue
        if np.all(chosen == true_color) or np.all(chosen == false_color):
            continue
        return {
            "input": {"example_input": grid_to_list(grid), "example_output": grid_to_list(chosen)},
            "output": rule["label"],
            "metadata": {
                "background_style": bg_style,
                "background_color": bg,
                "target_color": int(target_color),
                "true_color": int(true_color),
                "false_color": int(false_color),
                "mode": "return_predicate_label",
                "predicate_label": rule["label"],
                "candidate_predicate_labels": [item["label"] for item in NEIGHBOR_PREDICATES],
            },
        }
    raise RuntimeError("Task 248: failed to generate unique predicate example")


@register("257")
def generate_apply_ordered_cell_priority_rules(rng: np.random.Generator) -> dict:
    """Apply a provided ordered list of cell-level rules with conflicts."""
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        seed_color, other_color, seed_result, neighbor_result, foreground_result = unused_colors(rng, 5, bg)
        grid = _cell_priority_scene(
            rng, bg, seed_color, other_color, seed_result, neighbor_result, foreground_result
        )
        if grid is not None:
            break
    else:
        raise RuntimeError("Task 257: failed to build a conflicting cell-priority scene")
    all_rules = _make_cell_rules(bg, seed_color, seed_result, neighbor_result, foreground_result)
    priority = CELL_PRIORITY_ORDERS[int(rng.integers(0, len(CELL_PRIORITY_ORDERS)))]
    ordered_rules = _ordered_cell_rules(all_rules, priority["order"])
    out = _apply_cell_priority(grid, bg, seed_color, ordered_rules)
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "seed_color": int(seed_color),
            "other_foreground_color": int(other_color),
            "priority_label": priority["label"],
            "ordered_cell_rules": ordered_rules,
            "default_action": "leave_cell_unchanged",
            "conflict_policy": "first_satisfied_rule_wins",
        },
    }


@register("258")
def generate_infer_cell_priority_order_from_pair(rng: np.random.Generator) -> dict:
    """Infer the priority order label from one input/output pair and rule set."""
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        seed_color, other_color, seed_result, neighbor_result, foreground_result = unused_colors(rng, 5, bg)
        before = _cell_priority_scene(
            rng, bg, seed_color, other_color, seed_result, neighbor_result, foreground_result
        )
        if before is not None:
            break
    else:
        raise RuntimeError("Task 258: failed to build a distinguishing cell-priority scene")
    all_rules = _make_cell_rules(bg, seed_color, seed_result, neighbor_result, foreground_result)
    priority = CELL_PRIORITY_ORDERS[int(rng.integers(0, len(CELL_PRIORITY_ORDERS)))]
    ordered_rules = _ordered_cell_rules(all_rules, priority["order"])
    after = _apply_cell_priority(before, bg, seed_color, ordered_rules)
    return {
        "input": {"input": grid_to_list(before), "output": grid_to_list(after)},
        "output": priority["label"],
        "metadata": {
            "background_color": bg,
            "seed_color": int(seed_color),
            "other_foreground_color": int(other_color),
            "cell_rules": all_rules,
            "candidate_priority_orders": copy.deepcopy(CELL_PRIORITY_ORDERS),
            "priority_label": priority["label"],
            "conflict_policy": "first_satisfied_rule_wins",
        },
    }


@register("259")
def generate_apply_ordered_object_priority_rules(rng: np.random.Generator) -> dict:
    """Apply ordered object-level conditional rules with highest-priority wins."""
    bg = int(rng.integers(0, NUM_COLORS))
    object_color, large_color, wide_color, tall_color = unused_colors(rng, 4, bg)
    all_rules = _make_object_rules(large_color, wide_color, tall_color)
    order = list(OBJECT_RULE_LABELS)
    rng.shuffle(order)
    by_label = {rule["label"]: rule for rule in all_rules}
    ordered_rules = [by_label[label] for label in order]
    n_objects = int(rng.integers(4, 8))
    for _ in range(MAX_RETRIES):
        grid = _object_priority_scene(rng, bg, object_color, n_objects)
        if grid is not None:
            break
    else:
        raise RuntimeError("Task 259: failed to build a conflicting object-priority scene")
    out = _apply_object_priority(grid, bg, ordered_rules)
    return {
        "input": grid_to_list(grid),
        "output": grid_to_list(out),
        "metadata": {
            "background_color": bg,
            "object_color": int(object_color),
            "ordered_object_rules": ordered_rules,
            "default_action": "leave_object_unchanged",
            "conflict_policy": "first_satisfied_object_rule_wins",
            "connectivity": 4,
        },
    }
