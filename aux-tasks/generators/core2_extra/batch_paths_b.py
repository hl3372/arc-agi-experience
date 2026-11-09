"""Path, ray, projection, and distance generators for core2 batch paths B."""
from __future__ import annotations

from collections import deque

import numpy as np

from common.grid import NUM_COLORS, grid_to_list, make_grid, pick_color
from common.objects import bounding_box, random_polyomino, random_shape
from generators.core2_extra.utils import MAX_RETRIES, add_patterned_background, unused_colors
from registry import register


Coord = tuple[int, int]


def _sample_hw(
    rng: np.random.Generator,
    lo: int,
    hi: int,
    *,
    area_cap: int | None = None,
    square_frac: float = 0.25,
) -> tuple[int, int]:
    """Sample H and W independently from [lo, hi].

    Decorrelates the two dimensions (ARC has no H==W bias for these rules) while
    keeping a minority of square grids, and rejects samples whose area exceeds
    ``area_cap`` so grid->grid bundles stay inside the token budget. Falls back to
    the smallest square within the cap if rejection never succeeds.
    """
    for _ in range(200):
        if rng.random() < square_frac:
            side = int(rng.integers(lo, hi + 1))
            h = w = side
        else:
            h = int(rng.integers(lo, hi + 1))
            w = int(rng.integers(lo, hi + 1))
        if area_cap is None or h * w <= area_cap:
            return h, w
    return lo, lo


def _scatter_distractor_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int],
    n_objects: int,
    forbidden: set[Coord],
    max_size: int = 5,
    gap: int = 1,
) -> list[set[Coord]]:
    """Stamp a few varied background-color distractor shapes onto the grid.

    Shapes never touch each other, never touch ``forbidden`` cells (the semantic
    cells of the task), and are drawn only over background. They are decorative —
    the task rule must reference colors, not "every non-background object" — so
    they widen the object-count and shape-vocabulary distributions without
    changing the answer. Returns the list of placed cell sets (possibly fewer
    than requested if the grid is crowded).
    """
    h, w = grid.shape
    placed: list[set[Coord]] = []
    occupied = set(forbidden)
    if not colors or n_objects <= 0:
        return placed
    for _ in range(n_objects * 12):
        if len(placed) >= n_objects:
            break
        size = int(rng.integers(1, max_size + 1))
        shape = random_shape(rng, size_range=(size, size), connectivity=4)
        r0, c0, r1, c1 = bounding_box(shape)
        sh = r1 - r0 + 1
        sw = c1 - c0 + 1
        if sh > h or sw > w:
            continue
        tr = int(rng.integers(0, h - sh + 1))
        tc = int(rng.integers(0, w - sw + 1))
        cells = {(tr + r - r0, tc + c - c0) for r, c in shape}
        halo = {
            (r + dr, c + dc)
            for r, c in cells
            for dr in range(-gap, gap + 1)
            for dc in range(-gap, gap + 1)
        }
        if any(cell in occupied for cell in halo):
            continue
        if any(grid[cell] != bg for cell in cells):
            continue
        color = int(colors[len(placed) % len(colors)])
        for cell in cells:
            grid[cell] = color
        occupied |= halo
        placed.append(cells)
    return placed

DIRECTIONS: dict[str, Coord] = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
    "northwest": (-1, -1),
    "northeast": (-1, 1),
    "southwest": (1, -1),
    "southeast": (1, 1),
}


def _neighbors(connectivity: int) -> list[Coord]:
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return deltas


def _in_bounds(grid: np.ndarray, cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def _shortest_path(grid: np.ndarray, start: Coord, goal: Coord, wall_color: int, connectivity: int) -> list[Coord] | None:
    q = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in _neighbors(connectivity):
            nb = (r + dr, c + dc)
            if not _in_bounds(grid, nb) or nb in parent or int(grid[nb]) == int(wall_color):
                continue
            parent[nb] = (r, c)
            q.append(nb)
    if goal not in parent:
        return None
    path = []
    cur: Coord | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _place_noise_cells(
    rng: np.random.Generator,
    grid: np.ndarray,
    colors: list[int],
    count: int,
    allowed_mask: np.ndarray,
) -> list[tuple[Coord, int]]:
    coords = [tuple(map(int, p)) for p in np.argwhere(allowed_mask)]
    if count <= 0 or not colors or not coords:
        return []
    chosen = rng.choice(len(coords), size=min(count, len(coords)), replace=False)
    placements: list[tuple[Coord, int]] = []
    for i, idx in enumerate(np.atleast_1d(chosen).tolist()):
        cell = coords[int(idx)]
        color = int(colors[i % len(colors)])
        grid[cell] = color
        placements.append((cell, color))
    return placements


def _carve_dead_end_open_cells(
    rng: np.random.Generator,
    grid: np.ndarray,
    path: list[Coord],
    wall_color: int,
    background_color: int,
) -> list[Coord]:
    open_cells = set(path)
    carved: list[Coord] = []
    path_indices = list(range(1, max(1, len(path) - 1)))
    target = int(rng.integers(2, 5))
    for _ in range(300):
        if len(carved) >= target or not path_indices:
            break
        base = path[int(rng.choice(path_indices))]
        directions = _neighbors(4)
        rng.shuffle(directions)
        for dr, dc in directions:
            length = int(rng.integers(1, 3))
            branch: list[Coord] = []
            prev = base
            cur = (base[0] + dr, base[1] + dc)
            ok = True
            for _step in range(length):
                occupied = open_cells | set(branch)
                touching = {
                    (cur[0] + nr, cur[1] + nc)
                    for nr, nc in _neighbors(4)
                    if (cur[0] + nr, cur[1] + nc) in occupied
                }
                if (
                    not _in_bounds(grid, cur)
                    or int(grid[cur]) != int(wall_color)
                    or touching != {prev}
                ):
                    ok = False
                    break
                branch.append(cur)
                prev = cur
                cur = (cur[0] + dr, cur[1] + dc)
            if not ok:
                continue
            for cell in branch:
                grid[cell] = background_color
            open_cells.update(branch)
            carved.extend(branch)
            break
    return carved


def _carve_isolated_pockets(
    rng: np.random.Generator,
    grid: np.ndarray,
    wall_color: int,
    background_color: int,
    occupied: set[Coord],
) -> list[Coord]:
    """Open up a few small wall-enclosed pockets disconnected from ``occupied``.

    Each carved cell is required to have no open (non-wall) 8-neighbor outside the
    pocket itself, so the pockets never connect to the start/goal corridor and the
    unique-shortest-path check is unaffected. They are unreachable open cells that
    the rule leaves unchanged. Returns the carved cells.
    """
    h, w = grid.shape
    target = int(rng.integers(0, 4))
    carved: list[Coord] = []
    reserved = set(occupied)
    for _ in range(120):
        if len(carved) >= target:
            break
        blob_size = int(rng.integers(1, 4))
        seed = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        blob: list[Coord] = []
        cur = seed
        ok = True
        for _step in range(blob_size):
            r, c = cur
            ring8 = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - {(r, c)}
            # Every neighbour must be a wall or already part of this blob, and the
            # cell must not be reserved (corridor/decoy) — keeps the pocket isolated.
            if (r, c) in reserved or int(grid[r, c]) != int(wall_color):
                ok = False
                break
            blocked = False
            for nb in ring8:
                if not _in_bounds(grid, nb):
                    continue
                if nb in reserved or (nb in carved):
                    blocked = True
                    break
                if int(grid[nb]) != int(wall_color) and nb not in blob:
                    blocked = True
                    break
            if blocked:
                ok = False
                break
            blob.append((r, c))
            options = [nb for nb in _neighbors(4) if _in_bounds(grid, (r + nb[0], c + nb[1]))]
            dr, dc = options[int(rng.integers(0, len(options)))]
            cur = (r + dr, c + dc)
        if not ok or not blob:
            continue
        for cell in blob:
            grid[cell] = background_color
        carved.extend(blob)
        reserved |= {
            (cell[0] + dr, cell[1] + dc)
            for cell in blob
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        }
    return carved


def _path_from_waypoints(points: list[Coord]) -> list[Coord]:
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        dr = int(np.sign(r1 - r0))
        dc = int(np.sign(c1 - c0))
        if (r0 != r1 and c0 != c1) and abs(r1 - r0) != abs(c1 - c0):
            raise ValueError("waypoints must be horizontal, vertical, or 45-degree diagonal")
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            if (r, c) != path[-1]:
                path.append((r, c))
    if len(set(path)) != len(path):
        raise ValueError("path repeats a cell")
    return path


def _orthogonal_path(rng: np.random.Generator, h: int, w: int) -> tuple[list[Coord], str]:
    shape = str(rng.choice(["horizontal", "vertical", "single_turn", "two_turn_horizontal", "two_turn_vertical"]))
    if shape == "horizontal":
        r = int(rng.integers(1, h - 1))
        c0 = int(rng.integers(1, max(2, w // 2)))
        c1 = int(rng.integers(max(c0 + 3, w // 2), w - 1))
        return _path_from_waypoints([(r, c0), (r, c1)]), shape
    if shape == "vertical":
        c = int(rng.integers(1, w - 1))
        r0 = int(rng.integers(1, max(2, h // 2)))
        r1 = int(rng.integers(max(r0 + 3, h // 2), h - 1))
        return _path_from_waypoints([(r0, c), (r1, c)]), shape
    if shape == "single_turn":
        r0, r1 = sorted(rng.choice(np.arange(1, h - 1), size=2, replace=False).tolist())
        c0, c1 = sorted(rng.choice(np.arange(1, w - 1), size=2, replace=False).tolist())
        if rng.random() < 0.5:
            return _path_from_waypoints([(r0, c0), (r0, c1), (r1, c1)]), shape
        return _path_from_waypoints([(r0, c0), (r1, c0), (r1, c1)]), shape
    if shape == "two_turn_horizontal":
        rows = sorted(rng.choice(np.arange(1, h - 1), size=2, replace=False).tolist())
        if abs(rows[1] - rows[0]) < 2:
            rows[1] = min(h - 2, rows[0] + 2)
        c0 = int(rng.integers(1, max(2, w // 3)))
        cm = int(rng.integers(max(c0 + 2, w // 3), max(c0 + 3, (2 * w) // 3)))
        c1 = int(rng.integers(max(cm + 2, (2 * w) // 3), w - 1))
        return _path_from_waypoints([(rows[0], c0), (rows[0], cm), (rows[1], cm), (rows[1], c1)]), shape
    cols = sorted(rng.choice(np.arange(1, w - 1), size=2, replace=False).tolist())
    if abs(cols[1] - cols[0]) < 2:
        cols[1] = min(w - 2, cols[0] + 2)
    r0 = int(rng.integers(1, max(2, h // 3)))
    rm = int(rng.integers(max(r0 + 2, h // 3), max(r0 + 3, (2 * h) // 3)))
    r1 = int(rng.integers(max(rm + 2, (2 * h) // 3), h - 1))
    return _path_from_waypoints([(r0, cols[0]), (rm, cols[0]), (rm, cols[1]), (r1, cols[1])]), shape


def _path_turns(path: list[Coord]) -> int:
    turns = 0
    prev = None
    for a, b in zip(path, path[1:]):
        step = (b[0] - a[0], b[1] - a[1])
        if prev is not None and step != prev:
            turns += 1
        prev = step
    return turns


def _distance(a: Coord, b: Coord, metric: str) -> int:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return dr + dc if metric == "manhattan" else max(dr, dc)


@register("109")
def generate_shortest_path_avoiding_walls(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 8, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        wall_color, start_color, goal_color, path_color = unused_colors(rng, 4, bg)
        try:
            path, shape = _orthogonal_path(rng, h, w)
        except ValueError:
            continue
        grid = make_grid(h, w, fill=wall_color)
        for r, c in path:
            grid[r, c] = bg
        start, goal = path[0], path[-1]
        grid[start] = start_color
        grid[goal] = goal_color
        decoys = _carve_dead_end_open_cells(rng, grid, path, int(wall_color), bg)
        if not decoys:
            continue
        # Optional isolated open pockets, fully walled off from the corridor, that
        # are unreachable from the start and so never affect the shortest path. The
        # rule already preserves every non-path cell (open or wall), so they only
        # widen the open-region count without changing the answer.
        _carve_isolated_pockets(rng, grid, int(wall_color), bg, set(path) | set(decoys))
        if _shortest_path(grid, start, goal, int(wall_color), 4) != path:
            continue
        out = grid.copy()
        for r, c in path:
            out[r, c] = path_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "wall_color": int(wall_color),
                "start_color": int(start_color),
                "goal_color": int(goal_color),
                "path_color": int(path_color),
                "connectivity": 4,
                "path_shape": shape,
                "draw_rule": "color every cell on the shortest path, including start and goal",
            },
        }
    raise RuntimeError("Task 109: failed to generate valid example")


@register("110")
def generate_path_measure(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        # grid->text task: H and W are independent over a wide span.
        h, w = _sample_hw(rng, 8, 22)
        bg = int(rng.integers(0, NUM_COLORS))
        path_color = pick_color(rng, [bg])
        try:
            path, shape = _orthogonal_path(rng, h, w)
        except ValueError:
            continue
        measure = str(rng.choice(["length", "turns", "both"]))
        length = len(path)
        turns = _path_turns(path)
        if measure in {"turns", "both"} and turns == 0:
            continue
        grid = make_grid(h, w, fill=bg)
        for r, c in path:
            grid[r, c] = path_color
        distractor_pool = unused_colors(rng, int(rng.integers(1, 3)), bg, int(path_color))
        placements = _place_noise_cells(
            rng,
            grid,
            distractor_pool,
            int(rng.integers(2, 6)),
            grid == bg,
        )
        if not placements:
            continue
        if measure == "length":
            out = str(length)
        elif measure == "turns":
            out = str(turns)
        else:
            out = f"length={length};turns={turns}"
        return {
            "input": grid_to_list(grid),
            "output": out,
            "metadata": {
                "background_color": bg,
                "path_color": int(path_color),
                "connectivity": 4,
                "path_shape": shape,
                "measure": measure,
                "distractor_colors": sorted({color for _cell, color in placements}),
                "length_unit": "number of path cells",
                "turn_rule": "count direction changes while walking from one endpoint to the other",
            },
        }
    raise RuntimeError("Task 110: failed to generate valid example")


@register("112")
def generate_cast_rays_until_stop(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 7, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        source_color, ray_color, stop_color = unused_colors(rng, 3, bg)
        grid = make_grid(h, w, fill=bg)
        source = (int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3)))
        grid[source] = source_color
        direction_sets = [
            ["north", "south", "east", "west"],
            ["northwest", "northeast", "southwest", "southeast"],
            ["north", "east", "southwest"],
            ["south", "west", "northeast"],
        ]
        directions = list(direction_sets[int(rng.integers(0, len(direction_sets)))])
        stop_condition = str(rng.choice(["grid_edge", "stop_color"]))
        stop_cells: set[Coord] = set()
        for name in directions:
            dr, dc = DIRECTIONS[name]
            r, c = source
            if stop_condition == "stop_color":
                max_steps = 0
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w:
                    max_steps += 1
                    rr += dr
                    cc += dc
                stop_step = int(rng.integers(2, max_steps + 1))
                stop_pos = (r + dr * stop_step, c + dc * stop_step)
                grid[stop_pos] = stop_color
                stop_cells.add(stop_pos)
        ray_cells: set[Coord] = set()
        for name in directions:
            dr, dc = DIRECTIONS[name]
            r, c = source[0] + dr, source[1] + dc
            while 0 <= r < h and 0 <= c < w:
                if stop_condition == "stop_color" and grid[r, c] == stop_color:
                    break
                ray_cells.add((r, c))
                r += dr
                c += dc
        # Distractor objects of varied shapes/colors, placed strictly off the
        # source, rays, and stop cells so the rays never overwrite them and they
        # stay preserved in the output. They widen the shape vocabulary and object
        # count while the prompt restricts recoloring to ray cells only.
        protected = {source} | ray_cells | stop_cells
        distractor_pool = unused_colors(
            rng, int(rng.integers(2, 4)), bg, int(source_color), int(ray_color), int(stop_color)
        )
        distractor_cells = _scatter_distractor_objects(
            rng, grid, bg=bg, colors=distractor_pool,
            n_objects=int(rng.integers(2, 5)), forbidden=protected,
        )
        distractor_colors = sorted({int(grid[cell]) for cells in distractor_cells for cell in cells})
        if not distractor_colors:
            continue
        out = grid.copy()
        for cell in ray_cells:
            out[cell] = ray_color
        metadata = {
            "background_color": bg,
            "source_color": int(source_color),
            "ray_color": int(ray_color),
            "directions": directions,
            "stop_condition": stop_condition,
            "distractor_colors": distractor_colors,
            "source_policy": "do not recolor the source cell",
        }
        if stop_condition == "stop_color":
            metadata["stop_color"] = int(stop_color)
            metadata["stop_policy"] = "do not recolor stop-color cells"
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": metadata,
        }
    raise RuntimeError("Task 112: failed to generate valid example")


@register("115")
def generate_line_network_endpoint_branchpoint_count(rng: np.random.Generator) -> dict:
    # grid->text task: H and W are sampled independently (the centred network
    # fits any grid >= 9 on a side), so the grid stops being forced square.
    h, w = _sample_hw(rng, 9, 22, square_frac=0.3)
    bg = int(rng.integers(0, NUM_COLORS))
    line_color = pick_color(rng, [bg])
    grid = make_grid(h, w, fill=bg)
    connectivity = int(rng.choice([4, 8]))
    cr, cc = h // 2, w // 2
    variant = str(rng.choice(["chain", "tee", "plus", "fork", "double_tee", "diagonal_x", "diagonal_y"]))
    if variant == "chain":
        orientation = str(rng.choice(["horizontal", "vertical"]))
        span = min(7, (w if orientation == "horizontal" else h) - 2)
        length = int(rng.integers(4, max(5, span)))
        if orientation == "horizontal":
            start = cc - length // 2
            for step in range(length):
                grid[cr, start + step] = line_color
        else:
            start = cr - length // 2
            for step in range(length):
                grid[start + step, cc] = line_color
    elif variant == "double_tee":
        left = cc - 2
        right = cc + 2
        for c in range(left - 1, right + 2):
            grid[cr, c] = line_color
        for c in [left, right]:
            for r in range(cr - 2, cr + 1):
                grid[r, c] = line_color
    elif connectivity == 4:
        variant = str(rng.choice(["tee", "plus", "fork"]))
        arms = {
            "tee": [(0, -1), (-1, 0), (0, 1)],
            "plus": [(-1, 0), (1, 0), (0, -1), (0, 1)],
            "fork": [(-1, 0), (1, 0), (0, 1)],
        }[variant]
        grid[cr, cc] = line_color
        lengths = [int(rng.integers(2, 5)) for _ in arms]
        for (dr, dc), length in zip(arms, lengths):
            for step in range(1, length + 1):
                grid[cr + dr * step, cc + dc * step] = line_color
    else:
        variant = str(rng.choice(["diagonal_x", "diagonal_y"]))
        arms = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if variant == "diagonal_x" else [(-1, -1), (-1, 1), (1, 0)]
        grid[cr, cc] = line_color
        lengths = [int(rng.integers(2, 5)) for _ in arms]
        for (dr, dc), length in zip(arms, lengths):
            for step in range(1, length + 1):
                grid[cr + dr * step, cc + dc * step] = line_color
    distractor_pool = unused_colors(rng, int(rng.integers(1, 3)), bg, int(line_color))
    placements = _place_noise_cells(
        rng,
        grid,
        distractor_pool,
        int(rng.integers(2, 6)),
        grid == bg,
    )
    cells = [tuple(p) for p in np.argwhere(grid == line_color)]
    endpoint_count = 0
    branchpoint_count = 0
    cell_set = set(cells)
    for r, c in cells:
        degree = sum((r + dr, c + dc) in cell_set for dr, dc in _neighbors(connectivity))
        endpoint_count += int(degree == 1)
        branchpoint_count += int(degree >= 3)
    return {
        "input": grid_to_list(grid),
        "output": f"endpoints={endpoint_count};branchpoints={branchpoint_count}",
        "metadata": {
            "background_color": bg,
            "line_color": int(line_color),
            "connectivity": connectivity,
            "network_variant": variant,
            "distractor_colors": sorted({color for _cell, color in placements}),
            "endpoint_rule": "line cells with exactly one neighbor under the stated adjacency",
            "branchpoint_rule": "line cells with three or more neighbors under the stated adjacency",
        },
    }


@register("140")
def generate_draw_aligned_segment(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 6, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        marker_color, line_color = unused_colors(rng, 2, bg)
        alignment = str(rng.choice(["horizontal", "vertical", "main_diagonal", "anti_diagonal"]))
        if alignment == "horizontal":
            if w < 6:
                continue
            r = int(rng.integers(1, h - 1))
            c0 = int(rng.integers(1, w - 4))
            c1 = int(rng.integers(c0 + 3, w - 1))
            points = [(r, c0), (r, c1)]
        elif alignment == "vertical":
            if h < 6:
                continue
            c = int(rng.integers(1, w - 1))
            r0 = int(rng.integers(1, h - 4))
            r1 = int(rng.integers(r0 + 3, h - 1))
            points = [(r0, c), (r1, c)]
        elif alignment == "main_diagonal":
            if min(h, w) < 5:
                continue
            step = int(rng.integers(3, min(h, w) - 1))
            r0 = int(rng.integers(0, h - step))
            c0 = int(rng.integers(0, w - step))
            points = [(r0, c0), (r0 + step, c0 + step)]
        else:
            if min(h, w) < 5:
                continue
            step = int(rng.integers(3, min(h, w) - 1))
            r0 = int(rng.integers(0, h - step))
            c0 = int(rng.integers(step, w))
            points = [(r0, c0), (r0 + step, c0 - step)]
        segment = _path_from_waypoints(points)
        grid = make_grid(h, w, fill=bg)
        for p in points:
            grid[p] = marker_color
        # Decorative distractor objects in colors other than the marker/line color.
        # They never touch the segment, so they cannot create a second aligned
        # marker pair and the output leaves them unchanged.
        distractor_pool = unused_colors(rng, int(rng.integers(1, 4)), bg, int(marker_color), int(line_color))
        forbidden = set(segment) | {
            (r + dr, c + dc) for r, c in segment for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        }
        distractor_cells = _scatter_distractor_objects(
            rng, grid, bg=bg, colors=distractor_pool,
            n_objects=int(rng.integers(2, 5)), forbidden=forbidden,
        )
        distractor_colors = sorted({int(grid[cell]) for cells in distractor_cells for cell in cells})
        # Optional patterned background over cells the rule ignores (off-segment,
        # off-marker, off-distractor); the segment is drawn last over the texture.
        bg_info = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[int(marker_color), int(line_color), *distractor_colors],
            protected_cells=forbidden,
        )
        out = grid.copy()
        for p in segment:
            out[p] = line_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "endpoint_color": int(marker_color),
                "line_color": int(line_color),
                "alignment": alignment,
                "distractor_colors": distractor_colors,
                "background_style": bg_info["style"],
                "draw_rule": "color the full straight segment including both endpoints",
                "ignore_rule": "ignore all other colors; leave every non-segment cell unchanged",
            },
        }
    raise RuntimeError("Task 140: failed to generate valid example")


@register("141")
def generate_project_cells_to_border(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 8, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, projection_color = unused_colors(rng, 2, bg)
        grid = make_grid(h, w, fill=bg)
        source_type = str(rng.choice(["single_object", "scattered_cells"]))
        if source_type == "single_object":
            # Interior box the object must fit inside so its projection onto every
            # border stays well-defined (no target cell ever lands on a border).
            max_h = h - 4
            max_w = w - 4
            shape = None
            for _ in range(50):
                cand = random_polyomino(rng, int(rng.integers(4, 7)))
                _, _, mdr, mdc = bounding_box(cand)
                if mdr < max_h and mdc < max_w:
                    shape = cand
                    break
            if shape is None:
                # Fallback to a small fixed tetromino guaranteed to fit (h,w >= 8).
                shape = frozenset({(0, 0), (1, 0), (1, 1), (2, 1)})
            _, _, max_dr, max_dc = bounding_box(shape)
            r0 = int(rng.integers(2, h - 2 - max_dr))
            c0 = int(rng.integers(2, w - 2 - max_dc))
            cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
        else:
            count = int(rng.integers(4, 8))
            candidates = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
            idxs = rng.choice(len(candidates), size=count, replace=False).tolist()
            cells = {candidates[int(i)] for i in idxs}
        for r, c in cells:
            grid[r, c] = target_color
        # Optional patterned background. The projection only references target_color
        # columns/rows, so texture in other colors never changes which border cells
        # are marked; protect the target cells and the chosen border line.
        border = str(rng.choice(["top", "bottom", "left", "right"]))
        if border in ("top", "bottom"):
            border_line = {(0 if border == "top" else h - 1, c) for c in range(w)}
        else:
            border_line = {(r, 0 if border == "left" else w - 1) for r in range(h)}
        bg_info = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[int(target_color), int(projection_color)],
            protected_cells=set(cells) | border_line,
        )
        out = grid.copy()
        if border in ("top", "bottom"):
            rr = 0 if border == "top" else h - 1
            for c in sorted({c for _, c in cells}):
                out[rr, c] = projection_color
        else:
            cc = 0 if border == "left" else w - 1
            for r in sorted({r for r, _ in cells}):
                out[r, cc] = projection_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": int(target_color),
                "projection_color": int(projection_color),
                "border": border,
                "source_type": source_type,
                "background_style": bg_info["style"],
                "projection_rule": "mark occupied target columns on top/bottom borders or occupied target rows on left/right borders",
            },
        }
    raise RuntimeError("Task 141: failed to generate valid example")


@register("142")
def generate_count_then_draw_bar(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 8, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        target_color, distractor_color, anchor_color, bar_color = unused_colors(rng, 4, bg)
        grid = make_grid(h, w, fill=bg)
        condition = str(rng.choice(["target_cells", "target_objects"]))
        count = int(rng.integers(3, 7))
        occupied: set[Coord] = set()
        if condition == "target_cells":
            candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            for i in rng.choice(len(candidates), size=count, replace=False).tolist():
                p = candidates[int(i)]
                grid[p] = target_color
                occupied.add(p)
        else:
            placed = 0
            for _attempt in range(300):
                if placed == count:
                    break
                orientation_obj = str(rng.choice(["horizontal", "vertical"]))
                if orientation_obj == "horizontal":
                    r = int(rng.integers(1, h - 1))
                    c = int(rng.integers(1, w - 2))
                    cells = [(r, c), (r, c + 1)]
                else:
                    r = int(rng.integers(1, h - 2))
                    c = int(rng.integers(1, w - 1))
                    cells = [(r, c), (r + 1, c)]
                halo = {
                    (rr + dr, cc + dc)
                    for rr, cc in cells
                    for dr, dc in _neighbors(4)
                }
                if any(p in occupied for p in cells) or any(p in occupied for p in halo):
                    continue
                for p in cells:
                    grid[p] = target_color
                    occupied.add(p)
                placed += 1
            if placed != count:
                continue
        for _i in range(int(rng.integers(2, 5))):
            p = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
            if p not in occupied:
                grid[p] = distractor_color
        orientation = str(rng.choice(["horizontal", "vertical"]))
        direction = str(rng.choice(["east", "west"] if orientation == "horizontal" else ["south", "north"]))
        if direction == "east":
            anchor = (h - 1, 0)
            bar = [(anchor[0], anchor[1] + i) for i in range(count)]
        elif direction == "west":
            anchor = (h - 1, w - 1)
            bar = [(anchor[0], anchor[1] - i) for i in range(count)]
        elif direction == "south":
            anchor = (0, w - 1)
            bar = [(anchor[0] + i, anchor[1]) for i in range(count)]
        else:
            anchor = (h - 1, w - 1)
            bar = [(anchor[0] - i, anchor[1]) for i in range(count)]
        grid[anchor] = anchor_color
        # Optional patterned background. The count only uses color target_color and
        # the bar is drawn over the border frame, so texture in other colors leaves
        # both the count and the bar unchanged. Protect the whole frame (anchor +
        # bar live there) plus the target and distractor cells.
        frame = (
            {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)}
            | {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
        )
        bg_info = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[int(target_color), int(distractor_color), int(anchor_color), int(bar_color)],
            protected_cells=occupied | frame,
        )
        out = grid.copy()
        for p in bar:
            out[p] = bar_color
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "target_color": int(target_color),
                "distractor_color": int(distractor_color),
                "anchor_color": int(anchor_color),
                "bar_color": int(bar_color),
                "condition": condition,
                "orientation": orientation,
                "direction": direction,
                "connectivity": 4,
                "background_style": bg_info["style"],
                "draw_rule": "draw a bar whose length equals the count, starting at and including the anchor cell",
            },
        }
    raise RuntimeError("Task 142: failed to generate valid example")


@register("149")
def generate_exact_distance_ring(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 7, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        source_color, ring_color = unused_colors(rng, 2, bg)
        grid = make_grid(h, w, fill=bg)
        source_type = str(rng.choice(["single_cell", "two_cell_source"]))
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(2, w - 2))
        sources = {(r, c)}
        if source_type == "two_cell_source":
            sources.add((r, min(w - 2, c + 1)))
        for p in sources:
            grid[p] = source_color
        metric = str(rng.choice(["manhattan", "chebyshev"]))
        distance = int(rng.integers(2, 5))
        # Decorative distractors + optional texture. They are non-background, so
        # under the "background cells only" rule they are simply not eligible for
        # the ring; the answer stays the cells that remain background at exactly
        # `distance`. Keep them clear of the sources so the source set is intact.
        source_halo = {
            (sr + dr, sc + dc) for sr, sc in sources for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        }
        distractor_pool = unused_colors(rng, int(rng.integers(1, 4)), bg, int(source_color), int(ring_color))
        distractor_cells = _scatter_distractor_objects(
            rng, grid, bg=bg, colors=distractor_pool,
            n_objects=int(rng.integers(1, 4)), forbidden=source_halo,
        )
        distractor_colors = sorted({int(grid[cell]) for cells in distractor_cells for cell in cells})
        bg_info = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[int(source_color), int(ring_color), *distractor_colors],
            protected_cells=source_halo,
        )
        out = grid.copy()
        marked = 0
        for rr in range(h):
            for cc in range(w):
                if grid[rr, cc] != bg:
                    continue
                d = min(_distance((rr, cc), src, metric) for src in sources)
                if d == distance:
                    out[rr, cc] = ring_color
                    marked += 1
        if marked == 0:
            continue
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "source_color": int(source_color),
                "ring_color": int(ring_color),
                "source_type": source_type,
                "metric": metric,
                "distance": distance,
                "distractor_colors": distractor_colors,
                "background_style": bg_info["style"],
                "eligible_condition": "background cells only",
            },
        }
    raise RuntimeError("Task 149: failed to generate valid example")


@register("159")
def generate_recolor_path_with_sequence(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 8, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        path_color, start_color = unused_colors(rng, 2, bg)
        seq_len = int(rng.choice([2, 3, 4]))
        sequence = unused_colors(rng, seq_len, bg, path_color, start_color)
        try:
            path, shape = _orthogonal_path(rng, h, w)
        except ValueError:
            continue
        if rng.random() < 0.5:
            path = list(reversed(path))
        grid = make_grid(h, w, fill=bg)
        for p in path:
            grid[p] = path_color
        grid[path[0]] = start_color
        # Decorative distractors + optional texture in colors disjoint from the
        # path, endpoint, and sequence colors, kept clear of the path so the
        # {path_color, start_color} cells still trace exactly one 1-wide path.
        reserved = [int(path_color), int(start_color), *(int(c) for c in sequence)]
        path_halo = {
            (r + dr, c + dc) for r, c in path for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        }
        distractor_pool = unused_colors(rng, int(rng.integers(1, 3)), bg, *reserved)
        distractor_cells = _scatter_distractor_objects(
            rng, grid, bg=bg, colors=distractor_pool,
            n_objects=int(rng.integers(1, 4)), forbidden=path_halo,
        )
        distractor_colors = sorted({int(grid[cell]) for cells in distractor_cells for cell in cells})
        bg_info = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[*reserved, *distractor_colors],
            protected_cells=path_halo,
        )
        out = grid.copy()
        for i, p in enumerate(path):
            out[p] = sequence[i % len(sequence)]
        meta = {
            "background_color": bg,
            "path_color": int(path_color),
            "start_endpoint_color": int(start_color),
            "sequence_colors": [int(c) for c in sequence],
            "sequence_length": len(sequence),
            "connectivity": 4,
            "path_shape": shape,
            "distractor_colors": distractor_colors,
            "background_style": bg_info["style"],
            "draw_rule": "recolor every path cell including the marked endpoint, advancing one sequence color per cell",
        }
        for i, color in enumerate(sequence, start=1):
            meta[f"sequence_color_{i}"] = int(color)
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": meta,
        }
    raise RuntimeError("Task 159: failed to generate valid example")


@register("170")
def generate_equidistant_bisector(rng: np.random.Generator) -> dict:
    for _ in range(MAX_RETRIES):
        h, w = _sample_hw(rng, 6, 18, area_cap=324)
        bg = int(rng.integers(0, NUM_COLORS))
        marker_color, bisector_color = unused_colors(rng, 2, bg)
        metric = str(rng.choice(["manhattan", "chebyshev"]))
        p1 = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        p2 = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        if p1 == p2:
            continue
        grid = make_grid(h, w, fill=bg)
        grid[p1] = marker_color
        grid[p2] = marker_color
        # Decorative distractors + optional texture. Being non-background they are
        # not eligible under the "background cells only" bisector rule, so the
        # answer is unchanged; keep them off the two markers.
        marker_halo = {
            (mr + dr, mc + dc) for mr, mc in (p1, p2) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        }
        distractor_pool = unused_colors(rng, int(rng.integers(1, 4)), bg, int(marker_color), int(bisector_color))
        distractor_cells = _scatter_distractor_objects(
            rng, grid, bg=bg, colors=distractor_pool,
            n_objects=int(rng.integers(1, 4)), forbidden=marker_halo,
        )
        distractor_colors = sorted({int(grid[cell]) for cells in distractor_cells for cell in cells})
        bg_info = add_patterned_background(
            grid, rng, bg=bg,
            exclude_colors=[int(marker_color), int(bisector_color), *distractor_colors],
            protected_cells=marker_halo,
        )
        out = grid.copy()
        marked = 0
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg:
                    continue
                if _distance((r, c), p1, metric) == _distance((r, c), p2, metric):
                    out[r, c] = bisector_color
                    marked += 1
        if marked == 0:
            continue
        return {
            "input": grid_to_list(grid),
            "output": grid_to_list(out),
            "metadata": {
                "background_color": bg,
                "marker_color": int(marker_color),
                "bisector_color": int(bisector_color),
                "metric": metric,
                "distractor_colors": distractor_colors,
                "background_style": bg_info["style"],
                "eligible_condition": "background cells only",
            },
        }
    raise RuntimeError("Task 170: failed to generate valid example")
