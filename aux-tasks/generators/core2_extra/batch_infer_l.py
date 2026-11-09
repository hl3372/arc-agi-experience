"""Rule-inference generators for core2 batch L."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import grid_to_list, make_grid
from common.objects import (
    bounding_box,
    connected_components,
    normalize_shape,
    random_polyomino,
)
from generators.core2_extra.utils import (
    add_patterned_background,
    boxes_overlap,
    place_shape,
    unused_colors,
)
from registry import register


Coord = tuple[int, int]
Shape = frozenset[Coord]

DELTAS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

RAY_LABELS = [
    "ray_until_wall_E",
    "ray_until_wall_NE",
    "ray_until_edge_S",
    "ray_hit_wall_SW",
]

OVERLAY_LABELS = [
    "overlay_top_priority",
    "overlay_bottom_priority",
    "overlay_intersection_only",
    "overlay_union_recolor",
]

MARKER_LABELS = [
    "copy_to_markers_top_left",
    "copy_to_markers_center",
    "move_to_marker_top_left",
    "move_to_marker_center",
    "stamp_cross_on_markers_center",
]

SCALE_LABELS = [
    "scale_2x",
    "tile_2x2",
    "repeat_rows_3",
    "block_compress_2x",
]

EXTRACT_LABELS = [
    "extract_bbox_target_color",
    "extract_inside_frame",
    "extract_marker_box_interior",
    "extract_top_left_quadrant",
]

MASK_LABELS = [
    "mask_union",
    "mask_intersection",
    "mask_xor",
    "mask_a_minus_b",
    "mask_b_minus_a",
]

PROPERTY_LABELS = [
    "property_size_rank",
    "property_bbox_width",
    "property_touching_border",
    "property_hole_count",
]

MORPH_LABELS = [
    "dilate_1",
    "dilate_2",
    "erode_1",
    "open_1",
    "close_1",
]

REGION_LABELS = [
    "fill_enclosed_regions",
    "fill_border_connected_background",
    "fill_marker_adjacent_region",
    "fill_smallest_background_region",
]

PACK_LABELS = [
    "pack_left_to_right_by_area",
    "pack_left_to_right_by_width",
    "pack_top_to_bottom_by_area",
    "pack_top_to_bottom_by_width",
]

MOVE_SHAPE: Shape = frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)})
STAMP_CROSS: Shape = frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)})


def _stamp(grid: np.ndarray, cells: set[Coord] | Shape, color: int, *, top: int = 0, left: int = 0) -> set[Coord]:
    placed = {(top + r, left + c) for r, c in cells}
    for r, c in placed:
        grid[r, c] = color
    return placed


def _shape_size(shape: set[Coord] | Shape) -> tuple[int, int]:
    r0, c0, r1, c1 = bounding_box(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_cells(box: tuple[int, int, int, int]) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _random_overlapping_rects(
    rng: np.random.Generator, h: int, w: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Sample two axis-aligned rectangles (inclusive bboxes) on an h*w grid such that
    they overlap, but neither contains the other.

    Guaranteeing a non-empty intersection AND non-empty a-only / b-only cells keeps
    all overlay outcomes (top/bottom/intersection/union) and all mask ops
    (union/intersection/xor/a-b/b-a) pairwise distinct, so the rule stays uniquely
    inferable while the geometry varies per seed.
    """
    for _ in range(500):
        ar0 = int(rng.integers(0, h - 1))
        ar1 = int(rng.integers(ar0 + 1, h))
        ac0 = int(rng.integers(0, w - 1))
        ac1 = int(rng.integers(ac0 + 1, w))
        br0 = int(rng.integers(0, h - 1))
        br1 = int(rng.integers(br0 + 1, h))
        bc0 = int(rng.integers(0, w - 1))
        bc1 = int(rng.integers(bc0 + 1, w))
        a = (ar0, ac0, ar1, ac1)
        b = (br0, bc0, br1, bc1)
        a_cells = _bbox_cells(a)
        b_cells = _bbox_cells(b)
        inter = a_cells & b_cells
        if not inter:
            continue
        if not (a_cells - b_cells) or not (b_cells - a_cells):
            continue
        return a, b
    # Deterministic fallback (still satisfies the contract) if sampling exhausts.
    return (1, 1, h - 2, w // 2), (h // 2, w // 3, h - 1, w - 1)


def _line_cells(start: Coord, direction: Coord, count: int) -> list[Coord]:
    r, c = start
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, count + 1)]


def _grid_from_mask(mask: set[Coord], h: int, w: int, bg: int, fg: int) -> np.ndarray:
    grid = make_grid(h, w, bg)
    for r, c in mask:
        grid[r, c] = fg
    return grid


def _pattern(rng: np.random.Generator, h: int, w: int, colors: list[int]) -> np.ndarray:
    grid = make_grid(h, w, colors[0])
    offset = int(rng.integers(0, len(colors)))
    for r in range(h):
        for c in range(w):
            grid[r, c] = colors[(r * w + c + 2 * r + offset) % len(colors)]
    return grid


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    comps = connected_components(grid, connectivity=4, background=None)
    return [comp for comp in comps if comp["color"] == color]


def _hole_count(grid: np.ndarray, comp: dict, bg: int) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(sub[r, c]) != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                touches_edge |= cr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(sub[nr, nc]) == bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _rank_values(values: list[int]) -> dict[int, int]:
    return {value: rank for rank, value in enumerate(sorted(set(values)), start=1)}


def _property_values(
    before: np.ndarray, comps: list[dict], bg: int, property_label: str
) -> list[int]:
    if property_label == "property_size_rank":
        sizes = [len(comp["cells"]) for comp in comps]
        rank = _rank_values(sizes)
        return [rank[size] for size in sizes]
    if property_label == "property_bbox_width":
        return [comp["bbox"][3] - comp["bbox"][1] + 1 for comp in comps]
    if property_label == "property_touching_border":
        h, w = before.shape
        return [
            int(any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp["cells"]))
            for comp in comps
        ]
    return [_hole_count(before, comp, bg) for comp in comps]


def _property_partition(
    before: np.ndarray, comps: list[dict], bg: int, property_label: str
) -> frozenset[frozenset[int]]:
    """Group component indices by equal property value (a partition, ignoring the
    concrete value/color labels)."""
    values = _property_values(before, comps, bg, property_label)
    groups: dict[int, set[int]] = {}
    for idx, value in enumerate(values):
        groups.setdefault(value, set()).add(idx)
    return frozenset(frozenset(group) for group in groups.values())


def _apply_property_recolor(
    before: np.ndarray,
    comps: list[dict],
    bg: int,
    property_label: str,
    colors: list[int],
) -> tuple[np.ndarray, list[dict]]:
    out = before.copy()
    values = _property_values(before, comps, bg, property_label)

    lookup = {value: colors[i] for i, value in enumerate(sorted(set(values)))}
    for comp, value in zip(comps, values):
        for r, c in comp["cells"]:
            out[r, c] = lookup[value]
    mapping = [{"property_value": int(value), "color": int(lookup[value])} for value in sorted(lookup)]
    return out, mapping


def _dilate(mask: set[Coord], h: int, w: int, radius: int) -> set[Coord]:
    out = set(mask)
    frontier = set(mask)
    for _ in range(radius):
        expanded = set(frontier)
        for r, c in frontier:
            for dr, dc in DELTAS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
        frontier = expanded
        out = expanded
    return out


def _erode(mask: set[Coord], h: int, w: int, radius: int) -> set[Coord]:
    out = set(mask)
    for _ in range(radius):
        out = {
            (r, c)
            for r, c in out
            if all((r + dr, c + dc) in out for dr, dc in [(0, 0), *DELTAS_4])
        }
    return out


def _morph(mask: set[Coord], h: int, w: int, label: str) -> set[Coord]:
    if label == "dilate_1":
        return _dilate(mask, h, w, 1)
    if label == "dilate_2":
        return _dilate(mask, h, w, 2)
    if label == "erode_1":
        return _erode(mask, h, w, 1)
    if label == "open_1":
        return _dilate(_erode(mask, h, w, 1), h, w, 1)
    if label == "close_1":
        return _erode(_dilate(mask, h, w, 1), h, w, 1)
    raise ValueError(label)


def _background_components(grid: np.ndarray, bg: int) -> list[set[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in DELTAS_4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return sorted(comps, key=lambda cells: (min(r for r, _ in cells), min(c for _, c in cells)))


def _pack_objects(grid: np.ndarray, bg: int, label: str) -> np.ndarray:
    comps = [comp for comp in connected_components(grid, connectivity=4, background=bg)]

    def area(comp: dict) -> int:
        return len(comp["cells"])

    def width(comp: dict) -> int:
        return comp["bbox"][3] - comp["bbox"][1] + 1

    key_fn = area if label.endswith("_by_area") else width
    ordered = sorted(comps, key=lambda comp: (-key_fn(comp), comp["bbox"][0], comp["bbox"][1]))
    out = make_grid(*grid.shape, fill=bg)
    cursor = 1
    horizontal = label.startswith("pack_left_to_right")
    for comp in ordered:
        shape = normalize_shape(comp["cells"])
        if horizontal:
            top, left = 1, cursor
            _, sw = _shape_size(shape)
            cursor += sw + 1
        else:
            top, left = cursor, 1
            sh, _ = _shape_size(shape)
            cursor += sh + 1
        _stamp(out, shape, int(comp["color"]), top=top, left=left)
    return out


_RAY_SPEC = {
    "ray_until_wall_E": {"dir": (0, 1), "mode": "wall_draw"},
    "ray_until_wall_NE": {"dir": (-1, 1), "mode": "wall_draw"},
    "ray_until_edge_S": {"dir": (1, 0), "mode": "edge_draw"},
    "ray_hit_wall_SW": {"dir": (1, -1), "mode": "wall_hit"},
}


def _ray_transform(
    before: np.ndarray, bg: int, source: int, wall: int, ray: int, label: str
) -> np.ndarray | None:
    """Independent re-implementation of the ray rule (mirrors the verifier): trace
    from every source cell in the label's direction; require a clear background path;
    draw the path / hit the wall per the label's mode. Returns None if the label is
    inconsistent with `before` (e.g. the path is blocked or the required wall absent)."""
    spec = _RAY_SPEC[label]
    dr, dc = spec["dir"]
    mode = spec["mode"]
    sources = sorted(map(tuple, np.argwhere(before == source).tolist()))
    if not sources:
        return None
    h, w = before.shape
    out = before.copy()
    for sr, sc in sources:
        r, c = sr + dr, sc + dc
        path: list[Coord] = []
        hit_wall: Coord | None = None
        while 0 <= r < h and 0 <= c < w:
            if int(before[r, c]) == wall:
                hit_wall = (r, c)
                break
            if int(before[r, c]) != bg:
                return None
            path.append((r, c))
            r += dr
            c += dc
        if mode == "edge_draw":
            if hit_wall is not None:
                return None
            for cell in path:
                out[cell] = ray
        elif mode == "wall_draw":
            if hit_wall is None:
                return None
            for cell in path:
                out[cell] = ray
        else:  # wall_hit
            if hit_wall is None:
                return None
            out[hit_wall] = ray
    return out


def _ray_unique(before: np.ndarray, after: np.ndarray, bg, source, wall, ray, label) -> bool:
    matches = [
        lbl
        for lbl in RAY_LABELS
        if (cand := _ray_transform(before, bg, source, wall, ray, lbl)) is not None
        and cand.tolist() == after.tolist()
    ]
    return matches == [label]


def _place_ray_system(
    rng: np.random.Generator,
    h: int,
    w: int,
    label: str,
    reserved: set[Coord],
) -> tuple[Coord, Coord | None, list[Coord]] | None:
    """Choose one source (+ wall, for wall-based labels) consistent with `label` on an
    h*w grid, avoiding `reserved` cells. Returns (source_cell, wall_cell_or_None,
    path_cells) or None on failure. For edge_draw the south column below the source
    must contain no wall, so the wall is None and the path runs to the bottom edge."""
    spec = _RAY_SPEC[label]
    dr, dc = spec["dir"]
    mode = spec["mode"]
    for _ in range(300):
        sr = int(rng.integers(0, h))
        sc = int(rng.integers(0, w))
        if (sr, sc) in reserved:
            continue
        if mode == "edge_draw":
            if sr >= h - 1:
                continue
            path = [(rr, sc) for rr in range(sr + 1, h)]
            if any(cell in reserved for cell in path):
                continue
            return (sr, sc), None, path
        # wall-based: choose a distance so the wall lands in-bounds along the ray.
        max_steps = 0
        rr, cc = sr + dr, sc + dc
        while 0 <= rr < h and 0 <= cc < w:
            max_steps += 1
            rr += dr
            cc += dc
        if max_steps < 2:
            continue
        distance = int(rng.integers(2, max_steps + 1))
        path = [(sr + dr * k, sc + dc * k) for k in range(1, distance)]
        wall_cell = (sr + dr * distance, sc + dc * distance)
        if (sr, sc) in reserved or wall_cell in reserved or any(cell in reserved for cell in path):
            continue
        return (sr, sc), wall_cell, path


def _build_ray_scene(
    rng: np.random.Generator, bg: int, source: int, wall: int, ray: int, label: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a variable-size ray scene with 1-3 independent same-rule systems whose
    sources, walls and ray paths are mutually clear, then accept only if the chosen
    label is the unique match (gated by the verifier-mirror `_ray_unique`)."""
    h = int(rng.integers(8, 21))
    w = int(rng.integers(8, 21))
    n_systems = int(rng.integers(1, 4))
    before = make_grid(h, w, bg)
    # `reserved` blocks all cells used by any system (source, wall, every path cell,
    # plus a south guard so an edge_draw column never collides) so systems cannot
    # interfere with one another's traces.
    reserved: set[Coord] = set()
    edge_columns: set[int] = set()
    edge_sources: list[Coord] = []
    wall_seeds: list[tuple[Coord, Coord]] = []  # (wall cell, ray direction)
    for _ in range(n_systems):
        placed = _place_ray_system(rng, h, w, label, reserved)
        if placed is None:
            return None
        src, wall_cell, path = placed
        before[src] = source
        reserved.add(src)
        reserved.update(path)
        if wall_cell is not None:
            before[wall_cell] = wall
            reserved.add(wall_cell)
            wall_seeds.append((wall_cell, _RAY_SPEC[label]["dir"]))
        else:
            # edge_draw: forbid any later wall/source on this column at-or-below the
            # source so the south ray genuinely reaches the edge.
            for rr in range(src[0], h):
                reserved.add((rr, src[1]))
            edge_columns.add(src[1])
            edge_sources.append(src)

    # Decoy walls: keep the wall color present (and its cell-count varying) regardless
    # of label, so the number of distinct colors no longer separates edge_draw (which
    # otherwise has no wall) from the wall-based rules. Decoys are placed off every
    # ray path/column so they never affect any trace; uniqueness is still re-checked.
    n_decoys = int(rng.integers(1, 3)) if label == "ray_until_edge_S" else int(rng.integers(0, 3))
    for _ in range(n_decoys):
        for _ in range(200):
            dr_ = int(rng.integers(0, h))
            dc_ = int(rng.integers(0, w))
            if (dr_, dc_) in reserved:
                continue
            # Never on a south ray column at-or-below an edge source (would block it).
            if dc_ in edge_columns and any(dr_ >= sr for sr, sc in edge_sources if sc == dc_):
                continue
            before[dr_, dc_] = wall
            reserved.add((dr_, dc_))
            wall_seeds.append(((dr_, dc_), (0, 0)))
            break

    # Thicken some walls into short bars so the shape vocabulary is not purely
    # single cells. Growth only extends onto reserved-free background cells and, for
    # real (ray-stopping) walls, only perpendicular to that ray so the near face the
    # ray hits is unchanged. Any resulting ambiguity is rejected by _ray_unique below.
    for (wr, wc), (rdir, cdir) in wall_seeds:
        if rng.random() < 0.5:
            continue
        if (rdir, cdir) == (0, 0):
            grow_dirs = [DELTAS_4[int(rng.integers(0, 4))]]
        else:
            # Perpendicular directions to the ray (anti-axis / anti-diagonal).
            grow_dirs = [(-cdir, rdir), (cdir, -rdir)]
        gr, gc = grow_dirs[int(rng.integers(0, len(grow_dirs)))]
        extent = int(rng.integers(1, 3))
        rr, cc = wr, wc
        for _ in range(extent):
            rr, cc = rr + gr, cc + gc
            if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in reserved:
                break
            if cc in edge_columns and any(rr >= sr for sr, sc in edge_sources if sc == cc):
                break
            before[rr, cc] = wall
            reserved.add((rr, cc))

    after = _ray_transform(before, bg, source, wall, ray, label)
    if after is None:
        return None
    if not _ray_unique(before, after, bg, source, wall, ray, label):
        return None
    return before, after


@register("211")
def generate_infer_ray_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(RAY_LABELS))
    bg = int(rng.integers(0, 10))
    source, wall, ray = unused_colors(rng, 3, bg)

    scene: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(200):
        scene = _build_ray_scene(rng, bg, source, wall, ray, label)
        if scene is not None:
            break
    if scene is None:
        # Deterministic single-system fallback on a 9x9 grid.
        h = w = 9
        before = make_grid(h, w, bg)
        if label == "ray_until_edge_S":
            before[1, 4] = source
        elif label == "ray_until_wall_E":
            before[4, 1] = source
            before[4, 6] = wall
        elif label == "ray_until_wall_NE":
            before[7, 1] = source
            before[3, 5] = wall
        else:
            before[1, 7] = source
            before[5, 3] = wall
        after = _ray_transform(before, bg, source, wall, ray, label)
        if after is None:
            after = before.copy()
        scene = (before, after)

    before, after = scene

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "source_color": source,
            "wall_color": wall,
            "ray_color": ray,
            "candidate_labels": list(RAY_LABELS),
        },
    }


def _random_mask(rng: np.random.Generator, h: int, w: int, max_rects: int) -> set[Coord]:
    """A foreground mask = union of 1..max_rects axis-aligned rectangles (their count
    and adjacency make the component count vary across seeds)."""
    n = int(rng.integers(1, max_rects + 1))
    mask: set[Coord] = set()
    for _ in range(n):
        rh = int(rng.integers(1, max(2, h // 2) + 1))
        rw = int(rng.integers(1, max(2, w // 2) + 1))
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        mask |= {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}
    return mask


def _overlay_composite(
    a_mask: set[Coord], b_mask: set[Coord], h: int, w: int, bg, a_color, b_color, result_color, label
) -> np.ndarray:
    grid = make_grid(h, w, bg)
    if label == "overlay_top_priority":
        for r, c in b_mask:
            grid[r, c] = b_color
        for r, c in a_mask:
            grid[r, c] = a_color
    elif label == "overlay_bottom_priority":
        for r, c in a_mask:
            grid[r, c] = a_color
        for r, c in b_mask:
            grid[r, c] = b_color
    elif label == "overlay_intersection_only":
        for r, c in a_mask & b_mask:
            grid[r, c] = result_color
    else:
        for r, c in a_mask | b_mask:
            grid[r, c] = result_color
    return grid


def _overlay_unique(
    a_mask: set[Coord], b_mask: set[Coord], h: int, w: int, bg, a_color, b_color, result_color, label
) -> bool:
    target = _overlay_composite(a_mask, b_mask, h, w, bg, a_color, b_color, result_color, label)
    matches = [
        lbl
        for lbl in OVERLAY_LABELS
        if _overlay_composite(a_mask, b_mask, h, w, bg, a_color, b_color, result_color, lbl).tolist()
        == target.tolist()
    ]
    return matches == [label]


@register("212")
def generate_infer_overlay_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(OVERLAY_LABELS))
    bg = int(rng.integers(0, 10))
    a_color, b_color, result_color = unused_colors(rng, 3, bg)

    h = w = 6
    a_mask: set[Coord] = set()
    b_mask: set[Coord] = set()
    found = False
    for _ in range(300):
        # Independent, ARC-small grid sizing.
        h = int(rng.integers(5, 17))
        w = int(rng.integers(5, 17))
        cand_a = _random_mask(rng, h, w, max_rects=3)
        cand_b = _random_mask(rng, h, w, max_rects=3)
        # The three regions must all be non-empty so top/bottom (overlap recolor) and
        # intersection/union are mutually distinguishable; uniqueness re-checked below.
        if not (cand_a & cand_b) or not (cand_a - cand_b) or not (cand_b - cand_a):
            continue
        if not _overlay_unique(cand_a, cand_b, h, w, bg, a_color, b_color, result_color, label):
            continue
        a_mask, b_mask, found = cand_a, cand_b, True
        break

    if not found:
        # Deterministic fallback: two overlapping rectangles on a 6x6 grid.
        h = w = 6
        a_box, b_box = _random_overlapping_rects(rng, h, w)
        a_mask = _bbox_cells(a_box)
        b_mask = _bbox_cells(b_box)

    source_a = make_grid(h, w, bg)
    source_b = make_grid(h, w, bg)
    for cell in a_mask:
        source_a[cell] = a_color
    for cell in b_mask:
        source_b[cell] = b_color
    composite = _overlay_composite(a_mask, b_mask, h, w, bg, a_color, b_color, result_color, label)

    return {
        "input": {
            "source_a": grid_to_list(source_a),
            "source_b": grid_to_list(source_b),
            "composite": grid_to_list(composite),
        },
        "output": label,
        "metadata": {
            "background_color": bg,
            "source_a_color": a_color,
            "source_b_color": b_color,
            "result_color": result_color,
            "candidate_labels": list(OVERLAY_LABELS),
        },
    }


def _free_for(grid: np.ndarray, cells: set[Coord] | Shape, bg: int) -> bool:
    """All cells are in-bounds and currently background."""
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != bg:
            return False
    return True


def _build_marker_scene(
    rng: np.random.Generator,
    bg: int,
    object_color: int,
    marker_color: int,
    stamp_color: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Place the marker-task scene on an independently-sized grid and return
    (before, after) or None if the sampled layout fails the contract.

    Grid H/W are sampled independently and skew small (ARC-typical). The number
    of markers/stamps is varied per seed so that background coverage no longer
    cleanly separates the label classes (the prior 9x9 leakage), while keeping
    every label uniquely inferable: stamps/objects are placed with a >=1 cell gap
    so they never collide with each other, the source, or any marker.
    """
    h = int(rng.integers(9, 19))
    w = int(rng.integers(9, 19))
    before = make_grid(h, w, bg)
    sh, sw = _shape_size(MOVE_SHAPE)  # 3 x 3

    if label == "stamp_cross_on_markers_center":
        # 2-3 markers, each interior (cross needs a 1-cell margin) and >=3 apart in
        # Chebyshev distance so the stamped crosses never overlap.
        n_markers = int(rng.integers(2, 4))
        markers: list[Coord] = []
        for _ in range(n_markers):
            for _ in range(200):
                mr = int(rng.integers(1, h - 1))
                mc = int(rng.integers(1, w - 1))
                if any(max(abs(mr - r), abs(mc - c)) < 3 for r, c in markers):
                    continue
                cross = {(mr + dr, mc + dc) for dr, dc in STAMP_CROSS}
                if not _free_for(before, cross | {(mr, mc)}, bg):
                    continue
                markers.append((mr, mc))
                before[mr, mc] = marker_color
                break
            else:
                return None
        if len(markers) < 2:
            return None
        after = before.copy()
        for mr, mc in markers:
            for dr, dc in STAMP_CROSS:
                after[mr + dr, mc + dc] = stamp_color
        return before, after

    landmark = "center" if label.endswith("_center") else "top_left"
    # copy uses 2-3 target markers, move uses exactly 1 (keeps the two classes
    # distinguishable by marker count, as the verifier requires).
    n_targets = int(rng.integers(2, 4)) if label.startswith("copy_") else 1

    # Place the source object first.
    boxes: list[tuple[int, int, int, int]] = []
    source_cells = place_shape(before, rng, MOVE_SHAPE, int(object_color), bg, boxes, gap=1)
    if source_cells is None:
        return None
    after = before.copy()

    # Place each target: choose a stamp top-left so the stamped object footprint is
    # free in `after` (gap-separated from everything) and the marker landmark cell
    # is free in `before`.
    markers: list[Coord] = []
    target_tops: list[Coord] = []
    for _ in range(n_targets):
        for _ in range(300):
            top = int(rng.integers(0, h - sh + 1))
            left = int(rng.integers(0, w - sw + 1))
            shape_cells = {(top + r, left + c) for r, c in MOVE_SHAPE}
            box = (top, left, top + sh - 1, left + sw - 1)
            marker = (top + 1, left + 1) if landmark == "center" else (top, left)
            # Footprint must be clear in after, separated from placed boxes, and the
            # marker cell clear in before (it is one of the shape cells, fine).
            if any(boxes_overlap(box, other, gap=1) for other in boxes):
                continue
            if not _free_for(after, shape_cells, bg):
                continue
            target_tops.append((top, left))
            markers.append(marker)
            boxes.append(box)
            break
        else:
            return None
    if len(markers) != n_targets:
        return None

    # Markers are drawn in BOTH before and after (they are part of the input cue).
    for mr, mc in markers:
        before[mr, mc] = marker_color
        after[mr, mc] = marker_color
    # move removes the source object from after.
    if label.startswith("move_"):
        for cell in source_cells:
            after[cell] = bg
    # Stamp the object at each target in after (over background; marker at the
    # landmark cell is overwritten by object color, exactly as the verifier expects).
    for top, left in target_tops:
        _stamp(after, MOVE_SHAPE, int(object_color), top=top, left=left)
    return before, after


@register("213")
def generate_infer_marker_relative_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(MARKER_LABELS))
    bg = int(rng.integers(0, 10))
    object_color, marker_color, stamp_color = unused_colors(rng, 3, bg)

    scene: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(200):
        scene = _build_marker_scene(rng, bg, object_color, marker_color, stamp_color, label)
        if scene is not None:
            break
    if scene is None:
        # Deterministic fallback to a known-valid 9x9 configuration.
        before = make_grid(9, 9, bg)
        if label == "stamp_cross_on_markers_center":
            markers = [(2, 2), (2, 6)]
            for marker in markers:
                before[marker] = marker_color
            after = before.copy()
            for mr, mc in markers:
                for dr, dc in STAMP_CROSS:
                    after[mr + dr, mc + dc] = stamp_color
        else:
            source_cells = _stamp(before, MOVE_SHAPE, object_color, top=1, left=1)
            after = before.copy()
            landmark = "center" if label.endswith("_center") else "top_left"
            target_tops = [(1, 6), (5, 1), (5, 5)]
            target_tops = target_tops[:2] if label.startswith("copy_") else target_tops[:1]
            for tr, tc in target_tops:
                marker = (tr + 1, tc + 1) if landmark == "center" else (tr, tc)
                before[marker] = marker_color
                after[marker] = marker_color
            if label.startswith("move_"):
                for cell in source_cells:
                    after[cell] = bg
            for top in target_tops:
                _stamp(after, MOVE_SHAPE, object_color, top=top[0], left=top[1])
        scene = (before, after)

    before, after = scene
    # Optional sparse texture in colors the verifier ignores (it reconstructs from
    # the object/marker/stamp colors only), decorrelating background coverage from
    # the label without changing the answer. Applied identically to before/after so
    # the transform still round-trips.
    protected = set(map(tuple, np.argwhere(before != bg).tolist()))
    protected |= set(map(tuple, np.argwhere(after != bg).tolist()))
    bg_meta = add_patterned_background(
        before,
        rng,
        bg=bg,
        exclude_colors=[object_color, marker_color, stamp_color],
        protected_cells=protected,
        solid_probability=0.4,
    )
    if bg_meta["colors"]:
        for r, c in zip(*np.where(before != bg)):
            if (int(r), int(c)) not in protected:
                after[r, c] = before[r, c]

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "object_color": object_color,
            "marker_color": marker_color,
            "stamp_color": stamp_color,
            "candidate_labels": list(MARKER_LABELS),
        },
    }


def _scale_outputs(base: np.ndarray) -> dict[str, np.ndarray]:
    """The four geometric transforms applied to a base tile (block_compress takes the
    2x upscale as its `before`, so its `after` is the base itself)."""
    return {
        "scale_2x": np.repeat(np.repeat(base, 2, axis=0), 2, axis=1),
        "tile_2x2": np.tile(base, (2, 2)),
        "repeat_rows_3": np.repeat(base, 3, axis=0),
        "block_compress_2x": base,
    }


def _random_tile(rng: np.random.Generator, h: int, w: int, colors: list[int]) -> np.ndarray:
    """Fill an h*w tile with a per-cell random color drawn from `colors`."""
    grid = make_grid(h, w, colors[0])
    pal = np.array(colors, dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            grid[r, c] = int(pal[int(rng.integers(0, len(pal)))])
    return grid


def _scale_candidate(before: np.ndarray, label: str) -> np.ndarray | None:
    """Mirror of the verifier's per-label transform of `before`, used to gate
    uniqueness over the widened distribution."""
    if label == "scale_2x":
        return np.repeat(np.repeat(before, 2, axis=0), 2, axis=1)
    if label == "tile_2x2":
        return np.tile(before, (2, 2))
    if label == "repeat_rows_3":
        return np.repeat(before, 3, axis=0)
    if before.shape[0] % 2 or before.shape[1] % 2:
        return None
    compressed = before[::2, ::2]
    if np.repeat(np.repeat(compressed, 2, axis=0), 2, axis=1).tolist() != before.tolist():
        return None
    return compressed


def _scale_unique(before: np.ndarray, after: np.ndarray, label: str) -> bool:
    matches = [
        lbl
        for lbl in SCALE_LABELS
        if (cand := _scale_candidate(before, lbl)) is not None
        and cand.tolist() == after.tolist()
    ]
    return matches == [label]


@register("214")
def generate_infer_scale_tile_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(SCALE_LABELS))
    bg = int(rng.integers(0, 10))

    before: np.ndarray | None = None
    after: np.ndarray | None = None
    for _ in range(300):
        # Independent base dimensions (skewed small, ARC-typical). The base is the
        # `before` for 3 of the 4 ops and the 2x upscale of it is the `before` for
        # block_compress, so randomizing base H/W independently decorrelates the
        # primary-grid height from the label (the prior fixed-3x4 leakage). The
        # operation itself stays the only signal.
        bh = int(rng.integers(2, 7))
        bw = int(rng.integers(2, 8))
        n_colors = int(rng.integers(2, 6))
        colors = unused_colors(rng, n_colors, bg)
        if len(colors) < 2:
            continue
        base = _random_tile(rng, bh, bw, colors)
        if label == "block_compress_2x":
            cand_before = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)
            cand_after = base
        else:
            cand_before = base
            cand_after = _scale_outputs(base)[label]
        # Gate on exact uniqueness against ALL four candidate transforms (covers the
        # scale==tile degeneracy and the "base is itself a 2x block-upscale" case
        # that would let block_compress also match a scale/tile before).
        if not _scale_unique(cand_before, cand_after, label):
            continue
        before, after = cand_before, cand_after
        break

    if before is None:
        colors = unused_colors(rng, 5, bg)
        base = _pattern(rng, 3, 4, colors)
        if label == "block_compress_2x":
            before = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)
            after = base
        else:
            before = base
            after = _scale_outputs(base)[label]

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "candidate_labels": list(SCALE_LABELS),
        },
    }


def _extract_candidate(source: np.ndarray, meta: dict, label: str) -> np.ndarray | None:
    """Independent re-implementation of each extractor (mirrors the verifier)."""
    if label == "extract_bbox_target_color":
        cells = set(map(tuple, np.argwhere(source == meta["target_color"]).tolist()))
        if not cells:
            return None
        r0, c0, r1, c1 = _bbox_of(cells)
        return source[r0:r1 + 1, c0:c1 + 1]
    if label == "extract_inside_frame":
        cells = set(map(tuple, np.argwhere(source == meta["frame_color"]).tolist()))
        if not cells:
            return None
        r0, c0, r1, c1 = _bbox_of(cells)
        return source[r0 + 1:r1, c0 + 1:c1]
    if label == "extract_marker_box_interior":
        cells = sorted(map(tuple, np.argwhere(source == meta["marker_color"]).tolist()))
        if len(cells) != 2:
            return None
        rows = sorted([cells[0][0], cells[1][0]])
        cols = sorted([cells[0][1], cells[1][1]])
        return source[rows[0] + 1:rows[1], cols[0] + 1:cols[1]]
    return source[: source.shape[0] // 2, : source.shape[1] // 2]


def _bbox_of(cells) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_unique(source: np.ndarray, meta: dict, label: str) -> bool:
    out: list[str] = []
    for lbl in EXTRACT_LABELS:
        cand = _extract_candidate(source, meta, lbl)
        if cand is None or cand.size == 0:
            if lbl == label:
                return False
            continue
        target = _extract_candidate(source, meta, label)
        if target is None or target.size == 0:
            return False
        if cand.shape == target.shape and cand.tolist() == target.tolist():
            out.append(lbl)
    return out == [label]


def _build_extract_scene(
    rng: np.random.Generator,
    bg: int,
    target: int,
    frame: int,
    marker: int,
    data_colors: list[int],
) -> tuple[np.ndarray, dict] | None:
    """Place the four extractable structures on an independently-sized grid:
      - a multi-color pattern filling the whole top-left quadrant,
      - a hollow frame box with a pattern inside,
      - a single target-color shape (its bbox is the extraction),
      - exactly two marker cells defining a box with a pattern inside.
    Returns (source, meta_colors) or None. The caller textures and gates uniqueness."""
    h = int(rng.integers(8, 19))
    w = int(rng.integers(8, 19))
    qh, qw = h // 2, w // 2
    if qh < 2 or qw < 2:
        return None
    source = make_grid(h, w, bg)
    boxes: list[tuple[int, int, int, int]] = []

    # Top-left quadrant: fully random multi-color pattern.
    source[:qh, :qw] = _random_tile(rng, qh, qw, data_colors)
    boxes.append((0, 0, qh - 1, qw - 1))

    def _free_box(bh: int, bw: int, gap: int = 1) -> tuple[int, int] | None:
        for _ in range(200):
            r0 = int(rng.integers(0, h - bh + 1))
            c0 = int(rng.integers(0, w - bw + 1))
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if any(boxes_overlap(box, o, gap=gap) for o in boxes):
                continue
            return r0, c0
        return None

    # Frame: a hollow rectangle (>=3x3 so it has an interior) with an inside pattern.
    fh = int(rng.integers(3, 5))
    fw = int(rng.integers(3, 5))
    pos = _free_box(fh, fw)
    if pos is None:
        return None
    fr, fc = pos
    fbox = (fr, fc, fr + fh - 1, fc + fw - 1)
    for r, c in _bbox_cells(fbox):
        if r in (fbox[0], fbox[2]) or c in (fbox[1], fbox[3]):
            source[r, c] = frame
    inner = source[fr + 1:fr + fh - 1, fc + 1:fc + fw - 1]
    inner[:, :] = _random_tile(rng, inner.shape[0], inner.shape[1], data_colors[::-1])
    boxes.append(fbox)

    # Target shape: a random small polyomino in target color.
    tshape = random_polyomino(rng, int(rng.integers(3, 6)), connectivity=4)
    tcells = place_shape(source, rng, tshape, int(target), bg, boxes, gap=1)
    if tcells is None:
        return None

    # Marker box: two opposite corners, interior >=1 cell, filled with a pattern.
    mh = int(rng.integers(3, 5))
    mw = int(rng.integers(3, 5))
    pos = _free_box(mh, mw)
    if pos is None:
        return None
    mr, mc = pos
    mbox = (mr, mc, mr + mh - 1, mc + mw - 1)
    interior = source[mr + 1:mr + mh - 1, mc + 1:mc + mw - 1]
    interior[:, :] = _random_tile(rng, interior.shape[0], interior.shape[1], data_colors)
    source[mr, mc] = marker
    source[mr + mh - 1, mc + mw - 1] = marker
    boxes.append(mbox)

    return source, {"target_color": target, "frame_color": frame, "marker_color": marker}


@register("215")
def generate_infer_extraction_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(EXTRACT_LABELS))
    bg = int(rng.integers(0, 10))
    target, frame, marker = unused_colors(rng, 3, bg)
    data_colors = unused_colors(rng, 4, bg, target, frame, marker)

    source: np.ndarray | None = None
    meta_colors = {"target_color": target, "frame_color": frame, "marker_color": marker}
    for _ in range(200):
        scene = _build_extract_scene(rng, bg, target, frame, marker, data_colors)
        if scene is None:
            continue
        cand_source, meta_colors = scene
        meta = {"background_color": bg, **meta_colors}
        # Optional texture in colors disjoint from the selector colors; the extractors
        # read the final source, so texture is carried into the extracted subgrid
        # consistently and never changes which extractor is the unique match.
        protected = {
            (int(r), int(c))
            for r, c in np.argwhere(np.isin(cand_source, [target, frame, marker]))
        }
        add_patterned_background(
            cand_source,
            rng,
            bg=bg,
            exclude_colors=[target, frame, marker, *data_colors],
            protected_cells=protected,
            solid_probability=0.4,
        )
        # The 4 extractions must be pairwise distinct and the chosen one unique.
        if not all(
            (e := _extract_candidate(cand_source, meta, l)) is not None and e.size > 0
            for l in EXTRACT_LABELS
        ):
            continue
        if not _extract_unique(cand_source, meta, label):
            continue
        source = cand_source
        break

    if source is None:
        # Deterministic fallback to the original frozen-but-valid 10x10 configuration.
        target, frame, marker = unused_colors(rng, 3, bg)
        data_colors = unused_colors(rng, 4, bg, target, frame, marker)
        source = make_grid(10, 10, bg)
        source[:5, :5] = _pattern(rng, 5, 5, data_colors)
        frame_box = (1, 6, 4, 9)
        for r, c in _bbox_cells(frame_box):
            if r in (frame_box[0], frame_box[2]) or c in (frame_box[1], frame_box[3]):
                source[r, c] = frame
        source[2:4, 7:9] = _pattern(rng, 2, 2, data_colors[::-1])
        _stamp(source, {(0, 0), (0, 1), (1, 0), (2, 0), (2, 2)}, target, top=6, left=1)
        source[6:9, 6:9] = _pattern(rng, 3, 3, data_colors)
        source[5, 5] = marker
        source[9, 9] = marker
        meta_colors = {"target_color": target, "frame_color": frame, "marker_color": marker}

    meta = {"background_color": bg, **meta_colors}
    extracted = _extract_candidate(source, meta, label)

    return {
        "input": {
            "source": grid_to_list(source),
            "extracted": grid_to_list(extracted),
        },
        "output": label,
        "metadata": {
            "background_color": bg,
            **meta_colors,
            "candidate_labels": list(EXTRACT_LABELS),
        },
    }


def _mask_result(a_mask: set[Coord], b_mask: set[Coord], label: str) -> set[Coord]:
    if label == "mask_union":
        return a_mask | b_mask
    if label == "mask_intersection":
        return a_mask & b_mask
    if label == "mask_xor":
        return a_mask ^ b_mask
    if label == "mask_a_minus_b":
        return a_mask - b_mask
    return b_mask - a_mask


@register("216")
def generate_infer_mask_operation(rng: np.random.Generator) -> dict:
    label = str(rng.choice(MASK_LABELS))
    bg = int(rng.integers(0, 10))
    a_color, b_color, result_color = unused_colors(rng, 3, bg)

    h = w = 6
    a_mask: set[Coord] = set()
    b_mask: set[Coord] = set()
    found = False
    for _ in range(300):
        h = int(rng.integers(6, 17))
        w = int(rng.integers(6, 17))
        cand_a = _random_mask(rng, h, w, max_rects=3)
        cand_b = _random_mask(rng, h, w, max_rects=3)
        # Non-empty intersection / a-only / b-only makes all five set ops pairwise
        # distinct (union, intersection, xor, a-b, b-a), so the chosen op is unique.
        if not (cand_a & cand_b) or not (cand_a - cand_b) or not (cand_b - cand_a):
            continue
        results = {lbl: frozenset(_mask_result(cand_a, cand_b, lbl)) for lbl in MASK_LABELS}
        if any(not res for res in results.values()):
            continue
        if len(set(results.values())) != len(MASK_LABELS):
            continue
        a_mask, b_mask, found = cand_a, cand_b, True
        break

    if not found:
        h = w = 8
        a_box, b_box = _random_overlapping_rects(rng, h, w)
        a_mask = _bbox_cells(a_box)
        b_mask = _bbox_cells(b_box)

    source_a = _grid_from_mask(a_mask, h, w, bg, a_color)
    source_b = _grid_from_mask(b_mask, h, w, bg, b_color)
    combined = _grid_from_mask(_mask_result(a_mask, b_mask, label), h, w, bg, result_color)

    return {
        "input": {
            "source_a": grid_to_list(source_a),
            "source_b": grid_to_list(source_b),
            "combined": grid_to_list(combined),
        },
        "output": label,
        "metadata": {
            "background_color": bg,
            "source_a_color": a_color,
            "source_b_color": b_color,
            "result_color": result_color,
            "candidate_labels": list(MASK_LABELS),
        },
    }


def _random_property_shape(rng: np.random.Generator) -> Shape:
    """Sample a varied object shape: either a (possibly holed) rectangle or a random
    polyomino, giving variety across size, bbox width, hole count, and footprint."""
    if rng.random() < 0.55:
        ph = int(rng.integers(2, 5))
        pw = int(rng.integers(1, 5))
        cells = {(r, c) for r in range(ph) for c in range(pw)}
        # Punch interior holes when the rectangle is big enough.
        if ph >= 3 and pw >= 3 and rng.random() < 0.6:
            interior = [
                (r, c) for r in range(1, ph - 1) for c in range(1, pw - 1)
            ]
            rng.shuffle(interior)
            n_holes = int(rng.integers(1, min(3, len(interior)) + 1))
            for cell in interior[:n_holes]:
                cells.discard(cell)
        if len(cells) >= 2:
            return normalize_shape(cells)
    size = int(rng.integers(3, 8))
    return random_polyomino(rng, size, connectivity=4)


def _build_property_scene(
    rng: np.random.Generator, bg: int, object_color: int
) -> tuple[np.ndarray, list[dict]] | None:
    """Scatter several single-color objects on a randomized grid. Returns
    (before, comps) or None if placement fails."""
    h = int(rng.integers(10, 21))
    w = int(rng.integers(10, 22))
    n_objects = int(rng.integers(3, 8))
    before = make_grid(h, w, bg)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_objects):
        for _ in range(40):
            shape = _random_property_shape(rng)
            cells = place_shape(before, rng, shape, int(object_color), bg, boxes, gap=1)
            if cells is not None:
                break
        else:
            return None
    comps = _components_for_color(before, object_color)
    if len(comps) < 2:
        return None
    return before, comps


@register("217")
def generate_infer_property_recolor_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(PROPERTY_LABELS))
    bg = int(rng.integers(0, 10))
    object_color = unused_colors(rng, 1, bg)[0]
    recolor_colors = unused_colors(rng, 4, bg, object_color)

    before: np.ndarray | None = None
    comps: list[dict] = []
    for _ in range(400):
        scene = _build_property_scene(rng, bg, object_color)
        if scene is None:
            continue
        cand_before, cand_comps = scene
        chosen_partition = _property_partition(cand_before, cand_comps, bg, label)
        # The chosen property must split the objects into >=2 groups, with a number
        # of distinct values the recolor palette can represent...
        if not (2 <= len(chosen_partition) <= len(recolor_colors)):
            continue
        # ...and its grouping must NOT be reproducible by any other candidate
        # property, so the recoloring uniquely identifies the chosen property.
        others = [
            _property_partition(cand_before, cand_comps, bg, other)
            for other in PROPERTY_LABELS
            if other != label
        ]
        if any(other == chosen_partition for other in others):
            continue
        before, comps = cand_before, cand_comps
        break

    if before is None:
        # Deterministic fallback to the original frozen-but-valid configuration.
        before = make_grid(13, 15, bg)
        if label == "property_size_rank":
            placements = [
                (frozenset({(0, 0), (0, 1), (0, 2)}), 1, 1),
                (frozenset({(0, 0), (1, 0), (1, 1)}), 1, 6),
                (frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}), 5, 1),
                (MOVE_SHAPE, 5, 6),
                (frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)}), 3, 12),
            ]
        elif label == "property_bbox_width":
            placements = [
                (frozenset({(0, 0), (1, 0), (2, 0)}), 1, 1),
                (frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}), 1, 4),
                (frozenset({(0, 0), (1, 0), (1, 1)}), 1, 7),
                (frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}), 5, 1),
                (frozenset({(0, 0), (0, 1), (0, 2)}), 5, 6),
                (MOVE_SHAPE, 8, 10),
            ]
        elif label == "property_touching_border":
            placements = [
                (frozenset({(0, 0), (0, 1), (0, 2)}), 0, 1),
                (frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}), 4, 0),
                (frozenset({(0, 0), (1, 0), (2, 0)}), 4, 5),
                (frozenset({(0, 0), (1, 0), (1, 1)}), 8, 8),
            ]
        else:
            ring = frozenset(
                (r, c)
                for r in range(3)
                for c in range(3)
                if r in (0, 2) or c in (0, 2)
            )
            no_hole = frozenset((r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0))
            placements = [
                (ring, 1, 1),
                (no_hole, 1, 6),
                (ring, 6, 1),
                (no_hole, 6, 6),
            ]
        for shape, top, left in placements:
            _stamp(before, shape, object_color, top=top, left=left)
        comps = _components_for_color(before, object_color)

    # Optional sparse texture in colors disjoint from the object and recolor palette.
    # It must stay OUTSIDE every object's bounding box, because the hole-count property
    # inspects each object's bbox interior and would miscount if texture landed there;
    # protecting the full bboxes keeps all four properties (and the recolor) intact,
    # and the texture is carried unchanged into `after`.
    object_cells = {cell for comp in comps for cell in comp["cells"]}
    object_boxes = [tuple(comp["bbox"]) for comp in comps]
    add_patterned_background(
        before,
        rng,
        bg=bg,
        exclude_colors=[object_color, *recolor_colors],
        protected_cells=object_cells,
        protected_boxes=object_boxes,
        solid_probability=0.4,
    )

    after, mapping = _apply_property_recolor(before, comps, bg, label, recolor_colors)

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "object_color": object_color,
            "property_to_color": mapping,
            "candidate_labels": list(PROPERTY_LABELS),
        },
    }


def _one_blob(
    rng: np.random.Generator, h: int, w: int, box: tuple[int, int, int, int]
) -> set[Coord]:
    """A solid sub-blob inside `box` with punched interior holes and short spurs that
    stay within the box (so the caller can keep multiple blobs and the dilate-2 margin
    separate)."""
    br0, bc0, br1, bc1 = box
    bh = br1 - br0 + 1
    bw = bc1 - bc0 + 1
    blob = {(r, c) for r in range(br0, br1 + 1) for c in range(bc0, bc1 + 1)}
    interior = [
        (r, c) for r in range(br0 + 1, br1) for c in range(bc0 + 1, bc1)
    ]
    rng.shuffle(interior)
    n_holes = int(rng.integers(1, 4)) if interior else 0
    for cell in interior[:n_holes]:
        blob.discard(cell)
    # Spurs that remain inside the box footprint.
    n_spurs = int(rng.integers(1, 4))
    boundary = sorted(blob)
    for _ in range(n_spurs):
        br, bc = boundary[int(rng.integers(0, len(boundary)))]
        dr, dc = DELTAS_4[int(rng.integers(0, 4))]
        length = int(rng.integers(1, 3))
        for k in range(1, length + 1):
            nr, nc = br + dr * k, bc + dc * k
            if br0 <= nr <= br1 and bc0 <= nc <= bc1:
                blob.add((nr, nc))
            else:
                break
    return blob


def _random_morph_mask(rng: np.random.Generator, h: int, w: int) -> set[Coord]:
    """Build a randomized foreground mask of 1-3 solid blobs (each holed + spurred),
    all kept clear of the border by margin 2 so dilate_2 never clips, and separated
    by a >=2 cell gap so they read as distinct components in the input.

    Caller verifies that the 5 morphology ops yield pairwise-distinct results."""
    margin = 2
    n_blobs = int(rng.integers(1, 4))
    boxes: list[tuple[int, int, int, int]] = []
    mask: set[Coord] = set()
    for _ in range(n_blobs):
        for _ in range(60):
            bh = int(rng.integers(3, min(h - 2 * margin, 7) + 1))
            bw = int(rng.integers(3, min(w - 2 * margin, 7) + 1))
            if bh < 3 or bw < 3:
                break
            r0 = int(rng.integers(margin, h - margin - bh + 1))
            c0 = int(rng.integers(margin, w - margin - bw + 1))
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            # Keep a 2-cell gap so blobs stay separate components (and don't merge
            # under erode/open re-checks ambiguously).
            if any(boxes_overlap(box, o, gap=2) for o in boxes):
                continue
            boxes.append(box)
            mask |= _one_blob(rng, h, w, box)
            break
    return mask


@register("218")
def generate_infer_morphology_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(MORPH_LABELS))
    bg = int(rng.integers(0, 10))
    fg = unused_colors(rng, 1, bg)[0]

    mask: set[Coord] | None = None
    h = w = 11
    for _ in range(300):
        h = int(rng.integers(9, 19))
        w = int(rng.integers(9, 19))
        cand = _random_morph_mask(rng, h, w)
        if not cand:
            continue
        results = [frozenset(_morph(cand, h, w, lbl)) for lbl in MORPH_LABELS]
        # All five ops must produce mutually distinct, non-empty masks that each
        # differ from the input, so the chosen op is uniquely inferable.
        if any(not res for res in results):
            continue
        if any(res == cand for res in results):
            continue
        if len(set(results)) != len(MORPH_LABELS):
            continue
        mask = cand
        break

    if mask is None:
        # Deterministic single-blob fallback on an 11x11 grid.
        h = w = 11
        mask = {(r, c) for r in range(3, 8) for c in range(3, 8)}
        mask.discard((5, 5))

    before = _grid_from_mask(mask, h, w, bg, fg)
    after = _grid_from_mask(_morph(mask, h, w, label), h, w, bg, fg)

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "foreground_color": fg,
            "connectivity": 4,
            "candidate_labels": list(MORPH_LABELS),
        },
    }


def _region_selection(
    before: np.ndarray, bg: int, marker_cells: set[Coord], label: str
) -> set[Coord]:
    comps = _background_components(before, bg)
    h, w = before.shape
    selected: set[Coord] = set()
    if label == "fill_enclosed_regions":
        for comp in comps:
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                selected |= comp
    elif label == "fill_border_connected_background":
        for comp in comps:
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                selected |= comp
    elif label == "fill_marker_adjacent_region":
        for comp in comps:
            if any((r + dr, c + dc) in marker_cells for r, c in comp for dr, dc in DELTAS_4):
                selected |= comp
    else:
        selected = set(min(comps, key=len))
    return selected


def _build_region_scene(
    rng: np.random.Generator, bg: int, wall: int, marker: int
) -> tuple[np.ndarray, set[Coord]] | None:
    """Randomize grid size, border thickness, and the enclosed background pockets,
    placing one marker inside a pocket. Returns (before, marker_cells) or None if the
    sampled layout fails the determinability contract below.

    The caller verifies (via _region_selection) that all 4 region rules produce
    mutually distinct fills, that there is a unique smallest background component,
    and that exactly one component is marker-adjacent.
    """
    h = int(rng.integers(10, 21))
    w = int(rng.integers(10, 21))
    thickness = int(rng.integers(1, 3))
    before = make_grid(h, w, wall)
    before[:thickness, :] = bg
    before[-thickness:, :] = bg
    before[:, :thickness] = bg
    before[:, -thickness:] = bg

    # Interior wall region in which pockets may be carved, leaving a 1-cell wall
    # gap to the border so pockets stay enclosed (not border-connected).
    lo_r, hi_r = thickness + 1, h - thickness - 1
    lo_c, hi_c = thickness + 1, w - thickness - 1
    if hi_r - lo_r < 2 or hi_c - lo_c < 2:
        return None

    n_pockets = int(rng.integers(2, 5))
    pocket_boxes: list[tuple[int, int, int, int]] = []
    pockets: list[set[Coord]] = []
    for _ in range(n_pockets):
        for _ in range(60):
            ph = int(rng.integers(1, 4))
            pw = int(rng.integers(1, 4))
            r0 = int(rng.integers(lo_r, hi_r - ph + 1))
            c0 = int(rng.integers(lo_c, hi_c - pw + 1))
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            # Keep a 1-cell wall separation between pockets so they stay distinct
            # enclosed components.
            if any(boxes_overlap(box, other, gap=1) for other in pocket_boxes):
                continue
            cells = _bbox_cells(box)
            pocket_boxes.append(box)
            pockets.append(cells)
            break
    if len(pockets) < 2:
        return None
    for cells in pockets:
        for r, c in cells:
            before[r, c] = bg

    # Drop the marker on a bg cell inside one pocket that has >=2 cells, so that
    # pocket still keeps at least one bg cell adjacent to the marker.
    big_pockets = [p for p in pockets if len(p) >= 2]
    if not big_pockets:
        return None
    chosen = big_pockets[int(rng.integers(0, len(big_pockets)))]
    mr, mc = sorted(chosen)[int(rng.integers(0, len(chosen)))]
    before[mr, mc] = marker
    return before, {(mr, mc)}


@register("219")
def generate_infer_background_region_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(REGION_LABELS))
    bg = int(rng.integers(0, 10))
    wall, marker, fill = unused_colors(rng, 3, bg)

    before: np.ndarray | None = None
    marker_cells: set[Coord] = set()
    for _ in range(300):
        scene = _build_region_scene(rng, bg, wall, marker)
        if scene is None:
            continue
        cand_before, cand_marker = scene
        comps = _background_components(cand_before, bg)
        # Unique smallest background component (no size tie).
        sizes = sorted(len(comp) for comp in comps)
        if len(sizes) < 2 or sizes[0] == sizes[1]:
            continue
        # Exactly one marker-adjacent component.
        marker_adj = sum(
            1
            for comp in comps
            if any((r + dr, c + dc) in cand_marker for r, c in comp for dr, dc in DELTAS_4)
        )
        if marker_adj != 1:
            continue
        # All four region rules must yield mutually distinct, non-empty fills.
        sels = [
            frozenset(_region_selection(cand_before, bg, cand_marker, lbl))
            for lbl in REGION_LABELS
        ]
        if any(not sel for sel in sels):
            continue
        if len(set(sels)) != len(REGION_LABELS):
            continue
        before, marker_cells = cand_before, cand_marker
        break

    if before is None:
        # Deterministic fallback to the original frozen-but-valid configuration.
        before = make_grid(12, 12, wall)
        before[0, :] = bg
        before[-1, :] = bg
        before[:, 0] = bg
        before[:, -1] = bg
        before[2:5, 2:5] = bg
        before[2:4, 7:9] = bg
        before[7:9, 2:5] = bg
        before[3, 5] = marker
        marker_cells = {(3, 5)}

    selected = _region_selection(before, bg, marker_cells, label)
    after = before.copy()
    for cell in selected:
        after[cell] = fill

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "wall_color": wall,
            "marker_color": marker,
            "fill_color": fill,
            "candidate_labels": list(REGION_LABELS),
        },
    }


def _sample_pack_shapes(rng: np.random.Generator) -> list[Shape] | None:
    """Sample a varying number (3-5) of distinct-area, distinct-width polyomino shapes
    whose area ordering differs from their width ordering (so by_area vs by_width
    packing is distinguishable). Returns normalized shapes or None on failure.

    The size range spans 3-7 cells so up to 5 mutually distinct areas are available,
    letting the object count vary across {3,4,5} instead of being pinned near 3-4."""
    n = int(rng.integers(3, 6))
    for _ in range(160):
        shapes: list[Shape] = []
        areas: set[int] = set()
        widths: set[int] = set()
        ok = True
        for _ in range(n):
            placed = False
            for _ in range(60):
                size = int(rng.integers(3, 8))
                shape = random_polyomino(rng, size, connectivity=4)
                sh, sw = _shape_size(shape)
                if sh > 6 or sw > 6:
                    continue
                area = len(shape)
                if area in areas or sw in widths:
                    continue
                shapes.append(shape)
                areas.add(area)
                widths.add(sw)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok or len(shapes) != n:
            continue
        # Require the area ordering to differ from the width ordering.
        by_area = [id(s) for s in sorted(shapes, key=lambda s: len(s))]
        by_width = [id(s) for s in sorted(shapes, key=lambda s: _shape_size(s)[1])]
        if by_area == by_width:
            continue
        return shapes
    return None


@register("220")
def generate_infer_pack_rule(rng: np.random.Generator) -> dict:
    label = str(rng.choice(PACK_LABELS))
    bg = int(rng.integers(0, 10))

    before: np.ndarray | None = None
    for _ in range(200):
        shapes = _sample_pack_shapes(rng)
        if shapes is None:
            continue
        n = len(shapes)
        dims = [_shape_size(s) for s in shapes]
        heights = [d[0] for d in dims]
        widths = [d[1] for d in dims]
        # Grid must hold every one of the 4 packed layouts (extent is order
        # independent: sum of dimension+gap), plus a little slack for scatter.
        need_h = max(max(heights) + 1, sum(heights) + n) + 2
        need_w = max(max(widths) + 1, sum(widths) + n) + 2
        gh = int(need_h + rng.integers(0, 3))
        gw = int(need_w + rng.integers(0, 3))
        colors = unused_colors(rng, n, bg)
        cand = make_grid(gh, gw, bg)
        boxes: list[tuple[int, int, int, int]] = []
        placed_all = True
        for shape, color in zip(shapes, colors):
            cells = place_shape(cand, rng, shape, int(color), bg, boxes, gap=1)
            if cells is None:
                placed_all = False
                break
        if not placed_all:
            continue
        # All 4 packed layouts must fit and be pairwise distinct & uniquely matched.
        packed = {lbl: _pack_objects(cand, bg, lbl) for lbl in PACK_LABELS}
        signatures = {tuple(map(tuple, grid.tolist())) for grid in packed.values()}
        if len(signatures) != len(PACK_LABELS):
            continue
        before = cand
        break

    if before is None:
        # Deterministic fallback to the original frozen-but-valid configuration.
        colors = unused_colors(rng, 4, bg)
        before = make_grid(16, 16, bg)
        fb_shapes = [
            frozenset({(0, 0), (0, 1), (0, 2)}),
            frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
            frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}),
            frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)}),
        ]
        placements = [(8, 1), (1, 10), (1, 1), (5, 7)]
        object_colors = [colors[2], colors[0], colors[3], colors[1]]
        for shape, (top, left), color in zip(fb_shapes, placements, object_colors):
            _stamp(before, shape, color, top=top, left=left)

    after = _pack_objects(before, bg, label)

    return {
        "input": {"before": grid_to_list(before), "after": grid_to_list(after)},
        "output": label,
        "metadata": {
            "background_color": bg,
            "candidate_labels": list(PACK_LABELS),
            "packing_gap": 1,
        },
    }
