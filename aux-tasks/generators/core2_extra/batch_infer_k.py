"""Finite-family inference/meta task generators for core2 batch K."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid
from common.objects import random_polyomino
from generators.core2_extra.utils import (
    MAX_RETRIES,
    add_patterned_background,
    boxes_overlap,
    unused_colors,
)
from registry import register


Coord = tuple[int, int]

TRANSFORM_LABELS = [
    "identity",
    "rotate_cw_90",
    "rotate_180",
    "rotate_ccw_90",
    "reflect_vertical",
    "reflect_horizontal",
]

CANDIDATE_LABELS = ["candidate_A", "candidate_B", "candidate_C"]
MAPPING_LABELS = ["mapping_A", "mapping_B", "mapping_C"]

GEOM_OPS = {
    "translate_up_1": (-1, 0),
    "translate_down_1": (1, 0),
    "translate_left_1": (0, -1),
    "translate_right_1": (0, 1),
}
GEOM_LABELS = [
    f"{scope}:{op}"
    for scope in ("all_foreground", "selected_object")
    for op in GEOM_OPS
]

SELECTION_RULES = [
    "largest_area",
    "smallest_area",
    "topmost",
    "bottommost",
    "leftmost",
    "rightmost",
]
SELECTION_ACTIONS = ["recolor", "erase", "mark_first_cell"]
SELECTION_LABELS = [
    f"{rule}:{action}"
    for rule in SELECTION_RULES
    for action in SELECTION_ACTIONS
]

LINE_LABELS = [
    f"{axis}:{operation}:{condition}"
    for axis in ("rows", "columns")
    for operation in ("erase", "recolor")
    for condition in ("contains_marker", "lacks_marker")
]

SYMMETRY_LABELS = [
    "vertical_axis:add_from_left",
    "vertical_axis:remove_right_extra",
    "vertical_axis:recolor_right_to_left",
    "horizontal_axis:add_from_top",
    "horizontal_axis:remove_bottom_extra",
    "horizontal_axis:recolor_bottom_to_top",
]

CONNECTIVITY_LABELS = ["connectivity_4", "connectivity_8"]
DISTANCE_LABELS = [
    "manhattan:distance_1",
    "manhattan:distance_2",
    "chebyshev:distance_1",
    "chebyshev:distance_2",
]


def _texture_bg(
    rng: np.random.Generator,
    grid: np.ndarray,
    *,
    bg: int,
    exclude_colors,
    protected_cells,
    solid_probability: float = 0.5,
) -> dict:
    """Add a sparse/striped background texture in place, but keep ``bg`` the genuine
    modal color so ``metadata['background_color']`` stays honest. Rejects the rare
    dense texture (e.g. period-2 checkerboard on a tiny grid) and falls back to a
    solid background. The texture palette excludes ``exclude_colors`` and never
    touches ``protected_cells``. Returns the same info dict as the underlying call
    (``{"style","colors","period"}``)."""
    snapshot = grid.copy()
    for _ in range(6):
        grid[:, :] = snapshot
        info = add_patterned_background(
            grid,
            rng,
            bg=bg,
            exclude_colors=list(exclude_colors),
            protected_cells=protected_cells,
            solid_probability=solid_probability,
        )
        if int((grid == bg).sum()) * 2 > grid.size:
            return info
    grid[:, :] = snapshot
    return {"style": "solid", "colors": [], "period": 0}


def _apply_transform(grid: np.ndarray, label: str) -> np.ndarray:
    if label == "identity":
        return grid.copy()
    if label == "rotate_cw_90":
        return np.rot90(grid, -1).copy()
    if label == "rotate_180":
        return np.rot90(grid, 2).copy()
    if label == "rotate_ccw_90":
        return np.rot90(grid, 1).copy()
    if label == "reflect_vertical":
        return np.fliplr(grid).copy()
    if label == "reflect_horizontal":
        return np.flipud(grid).copy()
    raise ValueError(f"unknown transform label {label}")


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in grid.tolist())


def _unique_transform_grid(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Return an asymmetric square grid where every whole-grid transform label
    yields a distinct result.

    Diversity: the side length is sampled over a wide square range (ARC is
    dominated by small grids and 62% square; rotations require H==W so squareness
    is structural here), the foreground is a per-seed mix of single cells and a few
    small polyominoes (so the connected-component shape vocabulary is not a single
    1-cell shape), and an optional sparse/striped background texture is mixed in.
    Every configuration is gated by the all-six-distinct check below, so any layout
    whose symmetry would collapse two transform labels (e.g. a symmetric texture or
    a self-symmetric shape placement) is rejected and resampled -> uniqueness of the
    answer label is preserved across the entire widened distribution.
    """
    for _ in range(MAX_RETRIES):
        size = int(rng.integers(5, 15))
        bg = int(rng.integers(0, NUM_COLORS))
        # 4..8 colored foreground cells, optionally clustered into small shapes.
        n_fg = int(rng.integers(4, 9))
        colors = unused_colors(rng, n_fg, bg)
        grid = make_grid(size, size, fill=bg)
        placed: set[Coord] = set()
        # With prob, grow one or two small polyominoes to enrich the shape vocab;
        # the rest are scattered single cells. All stay inside the grid.
        n_shapes = int(rng.integers(0, 3))
        ci = 0
        for _s in range(n_shapes):
            if ci >= n_fg:
                break
            ssize = int(rng.integers(2, 5))
            shape = random_polyomino(rng, ssize, connectivity=4)
            rows = [r for r, _ in shape]
            cols = [c for _, c in shape]
            sh = max(rows) + 1
            sw = max(cols) + 1
            if sh > size or sw > size:
                continue
            r0 = int(rng.integers(0, size - sh + 1))
            c0 = int(rng.integers(0, size - sw + 1))
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if cells & placed:
                continue
            color = colors[ci]
            ci += 1
            for cell in cells:
                grid[cell] = color
                placed.add(cell)
        # Remaining colors as scattered single cells on still-empty positions.
        free = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) not in placed
        ]
        remaining = n_fg - ci
        if remaining > len(free):
            continue
        if remaining > 0:
            pick = rng.choice(len(free), size=remaining, replace=False)
            for idx, color in zip(pick, colors[ci:]):
                cell = free[int(idx)]
                grid[cell] = color
                placed.add(cell)
        if not placed:
            continue
        # Optional background texture (sparse dotted / stripes). It is part of the
        # whole grid that the transform acts on, so it does not change semantics; it
        # only adds visual diversity. The uniqueness gate below rejects any texture
        # whose symmetry would make two transform labels coincide.
        _texture_bg(
            rng,
            grid,
            bg=bg,
            exclude_colors=colors,
            protected_cells=placed,
            solid_probability=0.5,
        )
        keys = {_grid_key(_apply_transform(grid, label)) for label in TRANSFORM_LABELS}
        if len(keys) == len(TRANSFORM_LABELS):
            return grid, bg
    raise RuntimeError("failed to build unique transform grid")


def _stamp(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _shift(cells: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _component_cells(grid: np.ndarray, color: int) -> list[set[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return sorted(comps, key=lambda cells: _bbox(cells))


@register("121")
def generate_classify_primary_transform(rng: np.random.Generator) -> dict:
    inp, bg = _unique_transform_grid(rng)
    label = str(rng.choice(TRANSFORM_LABELS))
    out = _apply_transform(inp, label)
    return {
        "input": {"input": grid_to_list(inp), "output": grid_to_list(out)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "classification_family": "whole_grid_transform",
            "allowed_labels": list(TRANSFORM_LABELS),
        },
    }


@register("122")
def generate_choose_candidate_for_rule(rng: np.random.Generator) -> dict:
    inp, bg = _unique_transform_grid(rng)
    rule_label = str(rng.choice(TRANSFORM_LABELS))
    decoys = [label for label in TRANSFORM_LABELS if label != rule_label]
    rng.shuffle(decoys)
    candidate_ops = [rule_label, decoys[0], decoys[1]]
    rng.shuffle(candidate_ops)
    payload = {"input": grid_to_list(inp)}
    answer = ""
    for label, op in zip(CANDIDATE_LABELS, candidate_ops):
        payload[label] = grid_to_list(_apply_transform(inp, op))
        if op == rule_label:
            answer = label
    return {
        "input": payload,
        "output": answer,
        "metadata": {
            "background_color": bg,
            "rule_label": rule_label,
            "candidate_labels": list(CANDIDATE_LABELS),
            "allowed_rule_labels": list(TRANSFORM_LABELS),
            "allowed_labels": list(CANDIDATE_LABELS),
        },
    }


@register("123")
def generate_candidate_satisfies_rule_bool(rng: np.random.Generator) -> dict:
    inp, bg = _unique_transform_grid(rng)
    rule_label = str(rng.choice(TRANSFORM_LABELS))
    is_true = bool(rng.integers(0, 2))
    if is_true:
        candidate = _apply_transform(inp, rule_label)
    else:
        decoys = [label for label in TRANSFORM_LABELS if label != rule_label]
        rng.shuffle(decoys)
        candidate = _apply_transform(inp, decoys[0])
    return {
        "input": {"input": grid_to_list(inp), "candidate": grid_to_list(candidate)},
        "output": "true" if is_true else "false",
        "metadata": {
            "background_color": bg,
            "rule_label": rule_label,
            "allowed_rule_labels": list(TRANSFORM_LABELS),
            "allowed_labels": ["true", "false"],
        },
    }


@register("204")
def generate_infer_changed_color_mapping(rng: np.random.Generator) -> dict:
    bg = int(rng.integers(0, NUM_COLORS))
    source_colors = unused_colors(rng, 3, bg)
    target_colors = unused_colors(rng, 3, bg, *source_colors)
    rng.shuffle(source_colors)
    rng.shuffle(target_colors)
    changed_sources = source_colors[:2]
    correct_targets = target_colors[:2]

    # Independent H/W over the ARC small-grid envelope; the rule is purely about
    # which colors change, so size only adds diversity. Enough cells must fit
    # 3 color blobs of >=1 cell each (a per-color count, randomized for spread).
    per_color = [int(rng.integers(2, 6)) for _ in range(3)]
    need = sum(per_color)
    # bg must stay the strict majority so metadata['background_color'] is honest:
    # require bg cells (h*w - need) > need, i.e. h*w > 2*need. Span runs 4..15 so
    # the H/W distribution is genuinely wide, not merely shifted.
    h = int(rng.integers(4, 16))
    w = int(rng.integers(4, 16))
    while h * w <= 2 * need:
        h = int(rng.integers(4, 16))
        w = int(rng.integers(4, 16))
    grid = make_grid(h, w, fill=bg)
    free = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(free)
    pos = 0
    placed: set[Coord] = set()
    for color, count in zip(source_colors, per_color):
        for _ in range(count):
            r, c = free[pos]
            pos += 1
            grid[r, c] = color
            placed.add((r, c))

    correct_pairs = [
        {"from_color": int(changed_sources[0]), "to_color": int(correct_targets[0])},
        {"from_color": int(changed_sources[1]), "to_color": int(correct_targets[1])},
    ]
    decoy_pairs = [
        [
            {"from_color": int(changed_sources[0]), "to_color": int(correct_targets[1])},
            {"from_color": int(changed_sources[1]), "to_color": int(correct_targets[0])},
        ],
        [
            {"from_color": int(changed_sources[0]), "to_color": int(correct_targets[0])},
            {"from_color": int(changed_sources[1]), "to_color": int(target_colors[2])},
        ],
    ]

    out = grid.copy()
    for pair in correct_pairs:
        out[grid == pair["from_color"]] = pair["to_color"]

    # Optional background texture, applied identically to input and output so it is
    # an UNCHANGED color (it never appears in the changed-color mapping). Its palette
    # excludes every source and target color, so it can never collide with a mapped
    # color and break the per-color uniqueness the verifier relies on.
    info = _texture_bg(
        rng,
        grid,
        bg=bg,
        exclude_colors=[*source_colors, *target_colors],
        protected_cells=placed,
        solid_probability=0.5,
    )
    for color in info["colors"]:
        out[grid == color] = color

    labels = list(MAPPING_LABELS)
    rng.shuffle(labels)
    mapping_items = [{"label": labels[0], "pairs": correct_pairs}]
    mapping_items.extend(
        {"label": label, "pairs": pairs}
        for label, pairs in zip(labels[1:], decoy_pairs)
    )
    mapping_items.sort(key=lambda item: item["label"])
    return {
        "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
        "output": labels[0],
        "metadata": {
            "background_color": bg,
            "candidate_mappings": mapping_items,
            "allowed_labels": list(MAPPING_LABELS),
            "mapping_scope": "only colors whose value changes are listed",
        },
    }


def _shape_extent(shape: frozenset[Coord]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _place_geom_objects(
    rng: np.random.Generator, h: int, w: int, n: int
) -> list[set[Coord]] | None:
    """Place ``n`` random small polyominoes at non-adjacent positions with a 1-cell
    interior margin (so any single-cell shift stays in bounds) and a >=2 gap (so the
    objects never touch even after each moves 1 cell, keeping them distinct
    components). Returns the placed cell-sets or None on failure."""
    boxes: list[tuple[int, int, int, int]] = []
    placed: list[set[Coord]] = []
    for _ in range(n):
        ok = False
        for _try in range(120):
            shape = random_polyomino(rng, int(rng.integers(2, 6)), connectivity=4)
            sh, sw = _shape_extent(shape)
            if sh > h - 2 or sw > w - 2:
                continue
            # 1-cell margin on every side so any 1-cell shift stays in bounds.
            r0 = int(rng.integers(1, h - sh))
            c0 = int(rng.integers(1, w - sw))
            box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            if any(boxes_overlap(box, other, gap=2) for other in boxes):
                continue
            boxes.append(box)
            placed.append({(r0 + r, c0 + c) for r, c in shape})
            ok = True
            break
        if not ok:
            return None
    return placed


@register("205")
def generate_infer_scope_and_geometric_operation(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        # 2..4 objects (was always 2); each gets a distinct color and a random shape.
        n_obj = int(rng.integers(2, 5))
        colors = unused_colors(rng, n_obj, bg)
        h = int(rng.integers(8, 19))
        w = int(rng.integers(8, 19))
        placed = _place_geom_objects(rng, h, w, n_obj)
        if placed is None:
            continue
        grid = make_grid(h, w, fill=bg)
        for cells, color in zip(placed, colors):
            _stamp(grid, cells, color)

        label = str(rng.choice(GEOM_LABELS))
        scope, op = label.split(":", 1)
        target_idx = int(rng.integers(0, n_obj))
        moving = (
            list(zip(placed, colors))
            if scope == "all_foreground"
            else [(placed[target_idx], colors[target_idx])]
        )

        dr, dc = GEOM_OPS[op]
        out = grid.copy()
        for cells, _color in moving:
            for r, c in cells:
                out[r, c] = bg
        for cells, color in moving:
            _stamp(out, _shift(cells, dr, dc), color)

        # Uniqueness guard: exactly one geometric label must reproduce this output.
        if not _geom_label_unique(grid, bg, out, label):
            continue

        meta = {
            "background_color": bg,
            "object_colors": [int(c) for c in colors],
            "n_objects": int(n_obj),
            "allowed_labels": list(GEOM_LABELS),
        }
        return {
            "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
            "output": label,
            "metadata": meta,
        }
    raise RuntimeError("failed to build unique scope/geometry scene")


def _apply_geom_label(
    grid: np.ndarray, bg: int, label: str, comp: set[Coord] | None
) -> np.ndarray:
    scope, op = label.split(":", 1)
    dr, dc = GEOM_OPS[op]
    if scope == "all_foreground":
        moving = [c for c in _foreground_components(grid, bg)]
    else:
        moving = [comp] if comp is not None else []
    out = grid.copy()
    for cells in moving:
        for r, c in cells:
            out[r, c] = bg
    for cells in moving:
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                out[nr, nc] = grid[r, c]
    return out


def _foreground_components(grid: np.ndarray, bg: int) -> list[set[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _geom_label_unique(grid: np.ndarray, bg: int, out: np.ndarray, label: str) -> bool:
    target = _grid_key(out)
    comps = _foreground_components(grid, bg)
    matches = []
    for cand in GEOM_LABELS:
        scope = cand.split(":", 1)[0]
        if scope == "all_foreground":
            if _grid_key(_apply_geom_label(grid, bg, cand, None)) == target:
                matches.append(cand)
        else:
            if any(_grid_key(_apply_geom_label(grid, bg, cand, comp)) == target for comp in comps):
                matches.append(cand)
    return matches == [label]


def _selection_invariants_ok(comps: list[set[Coord]]) -> bool:
    """Each of the 6 selection rules must have a strict, unique winning object."""
    if len(comps) < 2:
        return False
    boxes = [_bbox(c) for c in comps]
    areas = [len(c) for c in comps]

    def winner(values, want_max):
        target = max(values) if want_max else min(values)
        idxs = [i for i, v in enumerate(values) if v == target]
        return idxs[0] if len(idxs) == 1 else None

    win_idx = [
        winner(areas, True),  # largest_area
        winner(areas, False),  # smallest_area
        winner([b[0] for b in boxes], False),  # topmost (min row)
        winner([b[2] for b in boxes], True),  # bottommost (max row)
        winner([b[1] for b in boxes], False),  # leftmost (min col)
        winner([b[3] for b in boxes], True),  # rightmost (max col)
    ]
    if any(i is None for i in win_idx):
        return False
    # smallest object keeps >=2 cells so mark_first_cell differs from recolor.
    if min(areas) < 2:
        return False
    # each rule must select a DISTINCT object, otherwise rule_X:action and
    # rule_Y:action would produce identical outputs (ambiguous label).
    return len(set(win_idx)) == len(win_idx)


def _selection_winner(comps: list[set[Coord]], rule: str) -> set[Coord] | None:
    """Return the unique winning object for a selection rule, or None if tied."""
    boxes = [_bbox(c) for c in comps]
    if rule == "largest_area":
        vals = [len(c) for c in comps]; want_max = True
    elif rule == "smallest_area":
        vals = [len(c) for c in comps]; want_max = False
    elif rule == "topmost":
        vals = [b[0] for b in boxes]; want_max = False
    elif rule == "bottommost":
        vals = [b[2] for b in boxes]; want_max = True
    elif rule == "leftmost":
        vals = [b[1] for b in boxes]; want_max = False
    else:  # rightmost
        vals = [b[3] for b in boxes]; want_max = True
    target = max(vals) if want_max else min(vals)
    idxs = [i for i, v in enumerate(vals) if v == target]
    return comps[idxs[0]] if len(idxs) == 1 else None


def _roles_are_winners(
    comps: list[set[Coord]], role_cells: dict[str, set[Coord]]
) -> bool:
    """Confirm each rule's unique winner over ALL objects is exactly the intended
    role object (so added fillers never silently steal a selection)."""
    for rule, cells in role_cells.items():
        if _selection_winner(comps, rule) != cells:
            return False
    return True


def _place_in_window(
    rng: np.random.Generator,
    shape: frozenset[Coord],
    r_lo: int,
    r_hi: int,
    c_lo: int,
    c_hi: int,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], set[Coord]] | None:
    """Place shape so its bbox lies within [r_lo,r_hi]x[c_lo,c_hi] (inclusive),
    gap>=1 from existing boxes. Returns (box, cells) or None."""
    from common.objects import bounding_box

    sr0, sc0, sr1, sc1 = bounding_box(shape)
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    if sh > (r_hi - r_lo + 1) or sw > (c_hi - c_lo + 1):
        return None
    for _ in range(60):
        r0 = int(rng.integers(r_lo, r_hi - sh + 2))
        c0 = int(rng.integers(c_lo, c_hi - sw + 2))
        box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        if any(boxes_overlap(box, other, gap=1) for other in boxes):
            continue
        cells = {(r0 + r - sr0, c0 + c - sc0) for r, c in shape}
        return box, cells
    return None


def _selection_scene(
    rng: np.random.Generator, object_color: int, bg: int
) -> tuple[np.ndarray, dict[str, set[Coord]]] | None:
    """Build a random scene with six separated same-color objects such that every
    selection rule (largest/smallest area, topmost/bottommost/leftmost/rightmost)
    has a strict unique winner that is a DISTINCT object. Geometry is randomized
    per seed by placing the four positional roles in disjoint border bands and the
    two area roles in the interior, then verifying the strict-uniqueness invariants."""
    h = int(rng.integers(13, 23))
    w = int(rng.integers(13, 23))
    # Disjoint bands so the positional extremes are distinct objects by construction.
    # top band rows [0, t-1]; bottom band rows [b+1, h-1]; interior rows [t, b].
    t = int(rng.integers(2, 4))
    b = h - 1 - int(rng.integers(2, 4))
    l = int(rng.integers(2, 4))
    r = w - 1 - int(rng.integers(2, 4))
    if b - t < 4 or r - l < 4:
        return None
    # Interior region for area objects, kept strictly inside the bands so they can
    # never become a positional extreme.
    ir_lo, ir_hi = t + 1, b - 1
    ic_lo, ic_hi = l + 1, r - 1
    if ir_hi - ir_lo < 1 or ic_hi - ic_lo < 1:
        return None

    # Six distinct areas; the two interior objects take the extreme areas.
    big, small = 7, 2
    mids = list(range(3, 7))  # {3,4,5,6}
    rng.shuffle(mids)
    mid_sizes = mids[:4]

    boxes: list[tuple[int, int, int, int]] = []
    cells_list: list[set[Coord]] = []
    # role -> (size, window). Positional objects use mid sizes so they never tie the
    # area extremes; interior objects carry the big/small areas.
    role_specs = [
        ("topmost", mid_sizes[0], (0, t - 1, ic_lo, ic_hi)),
        ("bottommost", mid_sizes[1], (b + 1, h - 1, ic_lo, ic_hi)),
        ("leftmost", mid_sizes[2], (ir_lo, ir_hi, 0, l - 1)),
        ("rightmost", mid_sizes[3], (ir_lo, ir_hi, r + 1, w - 1)),
        ("largest_area", big, (ir_lo, ir_hi, ic_lo, ic_hi)),
        ("smallest_area", small, (ir_lo, ir_hi, ic_lo, ic_hi)),
    ]
    role_cells: dict[str, set[Coord]] = {}
    for role, size, (r_lo, r_hi, c_lo, c_hi) in role_specs:
        placed = None
        for _ in range(40):
            shape = random_polyomino(rng, size, connectivity=4)
            res = _place_in_window(rng, shape, r_lo, r_hi, c_lo, c_hi, boxes)
            if res is not None:
                placed = res
                break
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cells_list.append(cells)
        role_cells[role] = cells

    # Add 0..4 interior FILLER objects to vary the object count. They live strictly
    # inside the bands (never a positional extreme) and carry mid areas in [3, 6]
    # (never the reserved smallest=2 / largest=7), so they can never become any
    # rule's winner. The invariant + winner cross-check below still guarantee each
    # rule's winner stays the intended role object.
    n_filler = int(rng.integers(0, 5))
    for _ in range(n_filler):
        size = int(rng.integers(3, 7))
        for _try in range(40):
            shape = random_polyomino(rng, size, connectivity=4)
            res = _place_in_window(rng, shape, ir_lo, ir_hi, ic_lo, ic_hi, boxes)
            if res is not None:
                box, cells = res
                boxes.append(box)
                cells_list.append(cells)
                break

    if not _selection_invariants_ok(cells_list):
        return None
    # Cross-check: the role object must be the unique winner of its own rule even
    # after fillers were added (defends the generator output == verifier answer).
    if not _roles_are_winners(cells_list, role_cells):
        return None

    grid = make_grid(h, w, fill=bg)
    for cells in cells_list:
        _stamp(grid, cells, object_color)
    return grid, role_cells


@register("206")
def generate_infer_selection_rule_and_action(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        object_color, action_color = unused_colors(rng, 2, bg)
        scene = _selection_scene(rng, object_color, bg)
        if scene is None:
            continue
        grid, objects = scene
        break
    else:
        raise RuntimeError("failed to build unique selection scene")
    rule = str(rng.choice(SELECTION_RULES))
    action = str(rng.choice(SELECTION_ACTIONS))
    selected = objects[rule]

    out = grid.copy()
    if action == "recolor":
        _stamp(out, selected, action_color)
    elif action == "erase":
        _stamp(out, selected, bg)
    else:
        out[min(selected)] = action_color

    return {
        "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
        "output": f"{rule}:{action}",
        "metadata": {
            "background_color": bg,
            "object_color": int(object_color),
            "action_color": int(action_color),
            "allowed_selection_rules": list(SELECTION_RULES),
            "allowed_actions": list(SELECTION_ACTIONS),
            "allowed_labels": list(SELECTION_LABELS),
        },
    }


def _apply_line_solver(
    grid: np.ndarray, label: str, marker_color: int, bg: int, line_color: int
) -> np.ndarray:
    """Generator-side mirror of the verifier's line-label application, used by the
    uniqueness gate so it does not depend on the test module."""
    axis, operation, condition = label.split(":")
    outer = grid.shape[0] if axis == "rows" else grid.shape[1]
    marker_lines = set()
    for idx in range(outer):
        line = grid[idx, :] if axis == "rows" else grid[:, idx]
        if np.any(line == marker_color):
            marker_lines.add(idx)
    selected = marker_lines if condition == "contains_marker" else set(range(outer)) - marker_lines
    replacement = bg if operation == "erase" else line_color
    out = grid.copy()
    for idx in selected:
        if axis == "rows":
            out[idx, :] = replacement
        else:
            out[:, idx] = replacement
    return out


def _line_label_unique(
    grid: np.ndarray, out: np.ndarray, label: str, marker_color: int, bg: int, line_color: int
) -> bool:
    target = _grid_key(out)
    matches = [
        cand
        for cand in LINE_LABELS
        if _grid_key(_apply_line_solver(grid, cand, marker_color, bg, line_color)) == target
    ]
    return matches == [label]


@register("207")
def generate_infer_line_axis_operation_condition(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        marker_color, line_color, data_a, data_b = unused_colors(rng, 4, bg)
        axis = str(rng.choice(["rows", "columns"]))
        operation = str(rng.choice(["erase", "recolor"]))
        condition = str(rng.choice(["contains_marker", "lacks_marker"]))
        label = f"{axis}:{operation}:{condition}"

        # Independent H/W over the ARC small-grid envelope. The field is a 2-color
        # data noise pattern (not a solid bg) into which markers are stamped.
        h = int(rng.integers(5, 16))
        w = int(rng.integers(5, 16))
        grid = np.array(
            rng.choice([data_a, data_b], size=(h, w), replace=True), dtype=np.uint8
        )

        outer = h if axis == "rows" else w
        cross = w if axis == "rows" else h
        # Mark a random NON-EMPTY PROPER subset of lines on the answer axis so both
        # contains/lacks select something and differ; >=1 marker per marked line.
        n_marked = int(rng.integers(1, outer))
        marked = rng.choice(outer, size=n_marked, replace=False)
        for idx in marked:
            k = int(rng.integers(1, max(2, cross // 2 + 1)))
            cols = rng.choice(cross, size=k, replace=False)
            for j in cols:
                if axis == "rows":
                    grid[int(idx), int(j)] = marker_color
                else:
                    grid[int(j), int(idx)] = marker_color

        out = _apply_line_solver(grid, label, marker_color, bg, line_color)
        # Reject degenerate / ambiguous scenes: the chosen condition must change at
        # least one line, and exactly one label must reproduce the output.
        if _grid_key(out) == _grid_key(grid):
            continue
        if not _line_label_unique(grid, out, label, marker_color, bg, line_color):
            continue

        return {
            "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
            "output": label,
            "metadata": {
                "background_color": bg,
                "marker_color": int(marker_color),
                "line_color": int(line_color),
                "allowed_labels": list(LINE_LABELS),
                "selection_conditions": ["contains_marker", "lacks_marker"],
            },
        }
    raise RuntimeError("failed to build unique line-rule scene")


def _mirror_vertical(cell: Coord, size: int) -> Coord:
    return cell[0], size - 1 - cell[1]


def _mirror_horizontal(cell: Coord, size: int) -> Coord:
    return size - 1 - cell[0], cell[1]


def _symmetry_label_unique(grid: np.ndarray, out: np.ndarray, bg: int, label: str) -> bool:
    """Exactly one symmetry label must transform grid into out."""
    target = _grid_key(out)
    matches = [
        cand for cand in SYMMETRY_LABELS
        if _grid_key(_apply_symmetry_solver(grid, cand, bg)) == target
    ]
    return matches == [label]


def _apply_symmetry_solver(grid: np.ndarray, label: str, bg: int) -> np.ndarray:
    """Mirror of the verifier's _apply_symmetry_label, kept in the generator so the
    uniqueness guard does not depend on the test module."""
    size = grid.shape[0]
    mid = size // 2
    out = grid.copy()
    if label == "vertical_axis:add_from_left":
        for r in range(size):
            for c in range(mid):
                mc = size - 1 - c
                if grid[r, c] != bg and grid[r, mc] == bg:
                    out[r, mc] = grid[r, c]
    elif label == "vertical_axis:remove_right_extra":
        for r in range(size):
            for c in range(mid + 1, size):
                mc = size - 1 - c
                if grid[r, c] != bg and grid[r, mc] == bg:
                    out[r, c] = bg
    elif label == "vertical_axis:recolor_right_to_left":
        for r in range(size):
            for c in range(mid):
                mc = size - 1 - c
                if grid[r, c] != bg and grid[r, mc] != bg and grid[r, mc] != grid[r, c]:
                    out[r, mc] = grid[r, c]
    elif label == "horizontal_axis:add_from_top":
        for r in range(mid):
            mr = size - 1 - r
            for c in range(size):
                if grid[r, c] != bg and grid[mr, c] == bg:
                    out[mr, c] = grid[r, c]
    elif label == "horizontal_axis:remove_bottom_extra":
        for r in range(mid + 1, size):
            mr = size - 1 - r
            for c in range(size):
                if grid[r, c] != bg and grid[mr, c] == bg:
                    out[r, c] = bg
    else:  # horizontal_axis:recolor_bottom_to_top
        for r in range(mid):
            mr = size - 1 - r
            for c in range(size):
                if grid[r, c] != bg and grid[mr, c] != bg and grid[mr, c] != grid[r, c]:
                    out[mr, c] = grid[r, c]
    return out


def _pick_symmetry_refs(
    rng: np.random.Generator, size: int, axis: str, n: int = 2
) -> list[Coord] | None:
    """Pick n source-half reference cells with distinct rows and distinct cols, all
    off both center axes, so the symmetry repair stays unambiguous."""
    mid = size // 2
    if axis == "vertical":
        # source = left half cols [0, mid-1]; keep rows off the center row.
        rows = [r for r in range(size) if r != mid]
        cols = list(range(0, mid))
    else:
        # source = top half rows [0, mid-1]; keep cols off the center col.
        rows = list(range(0, mid))
        cols = [c for c in range(size) if c != mid]
    if len(rows) < n or len(cols) < n:
        return None
    chosen_rows = rng.choice(rows, size=n, replace=False)
    chosen_cols = rng.choice(cols, size=n, replace=False)
    return [(int(chosen_rows[i]), int(chosen_cols[i])) for i in range(n)]


def _add_symmetric_pairs(
    rng: np.random.Generator,
    grid: np.ndarray,
    out: np.ndarray,
    size: int,
    axis: str,
    bg: int,
    used: set[Coord],
    colors: list[int],
    k: int,
) -> None:
    """Stamp ``k`` already-symmetric source/mirror cell pairs into BOTH grid and out
    (same color on each side). These pairs are inert under every one of the six
    repair operations (add never fills a non-bg mirror, remove only deletes cells
    whose mirror is bg, recolor only acts where the two sides differ), so they do
    not change the answer. Their purpose is to decorrelate the input component count
    from the repair category — that was the n_comp leakage. Skips any cell already
    in ``used`` or on a center axis."""
    mid = size // 2
    mirror = _mirror_vertical if axis == "vertical" else _mirror_horizontal
    placed = 0
    for _ in range(80):
        if placed >= k:
            break
        if axis == "vertical":
            r = int(rng.integers(0, size))
            c = int(rng.integers(0, mid))
            if r == mid:
                continue
        else:
            r = int(rng.integers(0, mid))
            c = int(rng.integers(0, size))
            if c == mid:
                continue
        cell = (r, c)
        mcell = mirror(cell, size)
        if cell in used or mcell in used:
            continue
        if grid[cell] != bg or grid[mcell] != bg:
            continue
        color = int(colors[int(rng.integers(0, len(colors)))])
        for g in (grid, out):
            g[cell] = color
            g[mcell] = color
        used.add(cell)
        used.add(mcell)
        placed += 1


@register("208")
def generate_infer_symmetry_repair_label(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # Wider odd-size envelope; squareness is structural (the symmetry axis is the
        # grid's own center row/column), so H==W is required and hw coupling is kept.
        size = int(rng.choice([7, 9, 11, 13, 15, 17, 19]))
        bg = int(rng.integers(0, NUM_COLORS))
        color_a, color_b, wrong_color = unused_colors(rng, 3, bg)
        palette = [color_a, color_b]
        label = str(rng.choice(SYMMETRY_LABELS))
        axis = "vertical" if label.startswith("vertical") else "horizontal"
        mirror = _mirror_vertical if axis == "vertical" else _mirror_horizontal

        # Vary the number of reference cells (was fixed at 2). Combined with the
        # variable symmetric-pair count below, this smears the per-category input
        # component count so n_comp no longer predicts the repair label.
        mid = size // 2
        max_refs = min(3, mid) if axis == "vertical" else min(3, mid)
        if max_refs < 2:
            continue
        n_refs = int(rng.integers(2, max_refs + 1))
        refs = _pick_symmetry_refs(rng, size, axis, n=n_refs)
        if refs is None:
            continue
        colors = [palette[i % len(palette)] for i in range(n_refs)]

        grid = make_grid(size, size, fill=bg)
        out = make_grid(size, size, fill=bg)
        action = label.split(":", 1)[1]
        used: set[Coord] = set()

        if action in ("add_from_left", "add_from_top"):
            for cell, color in zip(refs, colors):
                grid[cell] = color
                out[cell] = color
                out[mirror(cell, size)] = color
                used.add(cell)
                used.add(mirror(cell, size))
        elif action in ("remove_right_extra", "remove_bottom_extra"):
            for cell, color in zip(refs, colors):
                out[cell] = color
                out[mirror(cell, size)] = color
                used.add(cell)
                used.add(mirror(cell, size))
            grid[:, :] = out
            # place an extra cell in the OTHER half whose source-half mirror is bg.
            extra = _free_extra_cell(rng, out, size, axis, bg)
            if extra is None:
                continue
            grid[extra] = wrong_color
            used.add(extra)
            # Protect the extra's mirror: it must stay bg so the 'remove' repair
            # (which deletes the extra because its mirror is bg) keeps working.
            used.add(mirror(extra, size))
        else:  # recolor_right_to_left / recolor_bottom_to_top
            for cell, color in zip(refs, colors):
                out[cell] = color
                out[mirror(cell, size)] = color
                used.add(cell)
                used.add(mirror(cell, size))
            grid[:, :] = out
            # recolor the mirror of one ref to a different (wrong) color.
            target = mirror(refs[0], size)
            wc = wrong_color if wrong_color != colors[0] else color_b
            if wc == colors[0]:
                continue
            grid[target] = wc

        # Inert symmetric decoy pairs (0..3) to decorrelate component count.
        _add_symmetric_pairs(
            rng, grid, out, size, axis, bg, used, palette, int(rng.integers(0, 4))
        )

        if not _symmetry_label_unique(grid, out, bg, label):
            continue

        return {
            "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
            "output": label,
            "metadata": {
                "background_color": bg,
                "allowed_labels": list(SYMMETRY_LABELS),
                "symmetry_reference": "center row or center column of the square grid",
            },
        }
    raise RuntimeError("failed to build unique symmetry-repair scene")


def _free_extra_cell(
    rng: np.random.Generator, out: np.ndarray, size: int, axis: str, bg: int
) -> Coord | None:
    """Pick a bg cell in the OTHER half whose source-half mirror is also bg, so that
    adding it makes exactly the 'remove extra' repair restore symmetry."""
    mid = size // 2
    mirror = _mirror_vertical if axis == "vertical" else _mirror_horizontal
    candidates = []
    for r in range(size):
        for c in range(size):
            if axis == "vertical":
                if c <= mid:
                    continue
            else:
                if r <= mid:
                    continue
            if out[r, c] != bg:
                continue
            if out[mirror((r, c), size)] != bg:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _reachable_targets(
    grid: np.ndarray,
    seed_color: int,
    target_color: int,
    connectivity: int,
) -> set[Coord]:
    starts = [tuple(p) for p in np.argwhere(grid == seed_color)]
    if len(starts) != 1:
        raise ValueError("expected one seed")
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = grid.shape
    q = deque([starts[0]])
    seen = {starts[0]}
    reached: set[Coord] = set()
    while q:
        r, c = q.popleft()
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) != target_color:
                continue
            seen.add((nr, nc))
            reached.add((nr, nc))
            q.append((nr, nc))
    return reached


def _build_connectivity_scene(
    rng: np.random.Generator, h: int, w: int, seed_color: int, target_color: int, bg: int
) -> tuple[np.ndarray, Coord] | None:
    """Place a seed and target cells such that the 4-connected and 8-connected
    reachable sets differ (a diagonal-only bridge connects to extra targets)."""
    grid = make_grid(h, w, fill=bg)
    occupied: set[Coord] = set()

    def free(r, c):
        return 0 <= r < h and 0 <= c < w and (r, c) not in occupied

    # seed away from the right/bottom edges so a diagonal bridge fits.
    sr = int(rng.integers(1, h - 3))
    sc = int(rng.integers(1, w - 3))
    seed = (sr, sc)
    occupied.add(seed)
    cur = seed

    # an orthogonal trunk of a few targets reachable by BOTH connectivities.
    trunk_len = int(rng.integers(1, 3))
    for _ in range(trunk_len):
        opts = [(cur[0] + dr, cur[1] + dc) for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]]
        opts = [o for o in opts if free(*o)]
        if not opts:
            break
        nxt = opts[int(rng.integers(0, len(opts)))]
        occupied.add(nxt)
        cur = nxt

    # diagonal-only bridge from cur: the diagonal cell must be free AND the two
    # orthogonal cells linking cur to it must be bg (so 4-connectivity cannot cross).
    diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    rng.shuffle(diag_dirs)
    bridge = None
    for dr, dc in diag_dirs:
        dcell = (cur[0] + dr, cur[1] + dc)
        between1 = (cur[0] + dr, cur[1])
        between2 = (cur[0], cur[1] + dc)
        if free(*dcell) and free(*between1) and free(*between2):
            bridge = (dcell, between1, between2)
            break
    if bridge is None:
        return None
    dcell, b1, b2 = bridge
    occupied.add(dcell)  # b1, b2 stay bg so they break the 4-path

    # one extra target hanging orthogonally off the bridge cell (8-only reachable).
    extra_opts = [(dcell[0] + dr, dcell[1] + dc) for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]]
    extra_opts = [o for o in extra_opts if free(*o) and o not in (b1, b2)]
    if extra_opts:
        occupied.add(extra_opts[int(rng.integers(0, len(extra_opts)))])

    # a few isolated decoy targets reachable by neither (kept apart from everything).
    # They never touch the structure, so they leave reach4/reach8 unchanged and only
    # add visual variety; the count scales loosely with the grid area.
    n_decoy = int(rng.integers(0, max(3, (h * w) // 12)))
    for _ in range(n_decoy):
        for _try in range(20):
            cell = (int(rng.integers(0, h)), int(rng.integers(0, w)))
            if cell in occupied:
                continue
            # keep decoys from accidentally bridging to the structure.
            neighborhood = {
                (cell[0] + dr, cell[1] + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            }
            if neighborhood & occupied:
                continue
            occupied.add(cell)
            break

    grid[seed] = seed_color
    for cell in occupied:
        if cell != seed:
            grid[cell] = target_color
    return grid, seed


@register("209")
def generate_infer_connectivity_from_pair(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        seed_color, target_color, mark_color = unused_colors(rng, 3, bg)
        label = str(rng.choice(CONNECTIVITY_LABELS))
        connectivity = 4 if label == "connectivity_4" else 8
        # Independent H/W over the ARC small-grid envelope (was a fixed 7..9 square-ish
        # window). The seed needs a 1-cell top/left margin and room for the diagonal
        # bridge, so the structural minimum is 5.
        h = int(rng.integers(7, 17))
        w = int(rng.integers(7, 17))

        scene = _build_connectivity_scene(rng, h, w, seed_color, target_color, bg)
        if scene is None:
            continue
        grid, _seed = scene

        reach4 = _reachable_targets(grid, seed_color, target_color, 4)
        reach8 = _reachable_targets(grid, seed_color, target_color, 8)
        # uniqueness: the two labels must produce different outputs.
        if reach4 == reach8:
            continue

        out = grid.copy()
        for cell in _reachable_targets(grid, seed_color, target_color, connectivity):
            out[cell] = mark_color

        return {
            "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
            "output": label,
            "metadata": {
                "background_color": bg,
                "seed_color": int(seed_color),
                "target_color": int(target_color),
                "mark_color": int(mark_color),
                "allowed_labels": list(CONNECTIVITY_LABELS),
            },
        }
    raise RuntimeError("failed to build unique connectivity scene")


def _distance_ring(source: Coord, metric: str, distance: int, h: int, w: int) -> set[Coord]:
    ring: set[Coord] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) == source:
                continue
            if metric == "manhattan":
                d = abs(r - source[0]) + abs(c - source[1])
            else:
                d = max(abs(r - source[0]), abs(c - source[1]))
            if d == distance:
                ring.add((r, c))
    return ring


@register("210")
def generate_infer_distance_metric_and_parameter(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        bg = int(rng.integers(0, NUM_COLORS))
        source_color, mark_color = unused_colors(rng, 2, bg)
        label = str(rng.choice(DISTANCE_LABELS))
        metric, distance_text = label.split(":", 1)
        distance = int(distance_text.rsplit("_", 1)[1])

        # Independent H/W over the ARC small-grid envelope (was a fixed 7..10 window).
        h = int(rng.integers(7, 19))
        w = int(rng.integers(7, 19))
        # keep the source >=2 cells from every edge so all four rings are complete
        # and therefore mutually distinct (no clipping collapses two rings).
        sr = int(rng.integers(2, h - 2))
        sc = int(rng.integers(2, w - 2))
        source = (sr, sc)

        rings = {
            lbl: _distance_ring(source, lbl.split(":", 1)[0], int(lbl.rsplit("_", 1)[1]), h, w)
            for lbl in DISTANCE_LABELS
        }
        # uniqueness: the chosen ring must differ from every other label's ring.
        chosen = rings[label]
        if any(other != label and rings[other] == chosen for other in DISTANCE_LABELS):
            continue
        if not chosen:
            continue

        grid = make_grid(h, w, fill=bg)
        grid[source] = source_color

        # Distractor objects of unrelated colors (not source/mark) add element-count
        # and shape variety. They are pure geometry-irrelevant noise: the answer ring
        # is defined by distance from the unique source cell, which the verifier
        # recomputes from geometry alone, so distractors never change the ring sets
        # and uniqueness is unaffected. A distractor that happens to lie on the chosen
        # ring is simply overwritten by the mark (consistent with the verifier).
        distractor_palette = [
            c for c in range(NUM_COLORS) if c not in (bg, int(source_color), int(mark_color))
        ]
        boxes: list[tuple[int, int, int, int]] = [(sr, sc, sr, sc)]
        n_distract = int(rng.integers(0, max(2, (h * w) // 18)))
        for _ in range(n_distract):
            if not distractor_palette:
                break
            color = int(distractor_palette[int(rng.integers(0, len(distractor_palette)))])
            ssize = int(rng.integers(1, 5))
            for _try in range(30):
                shape = random_polyomino(rng, ssize, connectivity=4)
                sh, sw = _shape_extent(shape)
                if sh > h or sw > w:
                    continue
                r0 = int(rng.integers(0, h - sh + 1))
                c0 = int(rng.integers(0, w - sw + 1))
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                # keep distractors off the source cell (gap=1 around source) and apart
                # from each other so they read as separate objects.
                if any(boxes_overlap(box, other, gap=1) for other in boxes):
                    continue
                for r, c in shape:
                    grid[r0 + r, c0 + c] = color
                boxes.append(box)
                break

        out = grid.copy()
        for cell in chosen:
            out[cell] = mark_color

        return {
            "input": {"input": grid_to_list(grid), "output": grid_to_list(out)},
            "output": label,
            "metadata": {
                "background_color": bg,
                "source_color": int(source_color),
                "mark_color": int(mark_color),
                "allowed_labels": list(DISTANCE_LABELS),
            },
        }
    raise RuntimeError("failed to build unique distance scene")
