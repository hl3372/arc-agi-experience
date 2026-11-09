"""Path family generators: tasks 23, 24, 60, 61, 62, 63."""
from __future__ import annotations

import numpy as np

from common.grid import (
    NUM_COLORS,
    grid_to_list,
    make_grid,
    pick_color,
    pick_colors,
    random_background,
    random_grid_size,
)
from common.backgrounds import apply_background
from common.objects import bounding_box, random_shape
from common.shapes import sample_shape, shape_size
from generators.core2_extra.utils import add_patterned_background
from registry import register

MAX_RETRIES = 20


def _scatter_distractors(
    rng: np.random.Generator,
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int],
    n_objects: int,
    forbidden: set[tuple[int, int]],
    max_size: int = 5,
    gap: int = 1,
) -> list[set[tuple[int, int]]]:
    """Stamp a few separated decorative shapes onto background cells.

    Shapes are drawn only over *bg* cells, never touch each other and never
    touch *forbidden* (the task's semantic cells). They are decorative detail
    in colors disjoint from the semantic palette: because the task rule keys
    off a specific colour (the path / trace colour), these distractors widen
    the object-count and shape-vocabulary distributions without changing the
    answer. Returns the placed cell sets (possibly fewer than requested).
    """
    h, w = grid.shape
    placed: list[set[tuple[int, int]]] = []
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


def _decorate_path_background(
    rng: np.random.Generator,
    grid: np.ndarray,
    *,
    bg: int,
    semantic_colors: list[int],
    protected: set[tuple[int, int]],
) -> None:
    """Add distractor objects + patterned texture around an already-drawn path.

    All decorative colours are disjoint from *semantic_colors* (background,
    path, marker, highlight) so the path remains the unique connected line in
    its own colour. Semantic cells in *protected* are never overwritten. The
    decoration is a solid+patterned MIX (``add_patterned_background`` leaves a
    fraction of scenes solid), so the object-count spread spans from a clean
    single-path scene up to a richly textured one.
    """
    forbid = [int(c) for c in semantic_colors]
    distractor_colors = pick_colors(
        rng, int(rng.integers(1, 4)), exclude=forbid,
    )
    distractor_cells = _scatter_distractors(
        rng, grid, bg=bg, colors=[int(c) for c in distractor_colors],
        n_objects=int(rng.integers(0, 4)), forbidden=protected,
    )
    protected = set(protected)
    for cells in distractor_cells:
        protected |= cells
    used_distractor = sorted({int(grid[cell]) for cells in distractor_cells for cell in cells})
    add_patterned_background(
        grid, rng, bg=bg,
        exclude_colors=[c for c in forbid if c != bg] + used_distractor,
        protected_cells=protected,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_path(
    rng: np.random.Generator,
    grid: np.ndarray,
    path_color: int,
    bg_color: int,
    min_len: int = 8,
    max_len: int = 20,
) -> list[tuple[int, int]]:
    """Generate a random non-branching, non-looping 4-connected path on *grid*.

    The walk is performed on cells that currently hold *bg_color*.
    Returns the ordered list of (r, c) cells forming the path, or an empty
    list if a path of the required length could not be built.
    """
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    h, w = grid.shape

    bg_positions = list(zip(*np.where(grid == bg_color)))
    if not bg_positions:
        return []

    best_path: list[tuple[int, int]] = []
    target_len = int(rng.integers(min_len, max_len + 1))

    for _attempt in range(40):
        start_idx = int(rng.integers(0, len(bg_positions)))
        sr, sc = bg_positions[start_idx]
        path: list[tuple[int, int]] = [(sr, sc)]
        visited: set[tuple[int, int]] = {(sr, sc)}

        while len(path) < target_len:
            r, c = path[-1]
            neighbors: list[tuple[int, int, int]] = []
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != bg_color:
                    continue

                adj_visited = 0
                future_options = 0
                for dr2, dc2 in deltas:
                    ar, ac = nr + dr2, nc + dc2
                    if (ar, ac) in visited:
                        adj_visited += 1
                    elif 0 <= ar < h and 0 <= ac < w and grid[ar, ac] == bg_color:
                        future_options += 1

                # A valid path cell should only touch the existing path at the tip.
                if adj_visited <= 1:
                    neighbors.append((future_options, nr, nc))

            if not neighbors:
                break

            rng.shuffle(neighbors)
            neighbors.sort(key=lambda item: item[0], reverse=True)
            _, nr, nc = neighbors[0]
            path.append((nr, nc))
            visited.add((nr, nc))

        if len(path) > len(best_path):
            best_path = path
        if len(path) >= min_len:
            return path

    return best_path if len(best_path) >= min_len else []


def _draw_path(grid: np.ndarray, path: list[tuple[int, int]], color: int) -> None:
    """Paint all cells in *path* with *color* on *grid* (in-place)."""
    for r, c in path:
        grid[r, c] = color


def _pick_colors(rng: np.random.Generator, n: int = 2) -> list[int]:
    """Pick *n* distinct colors from 0..NUM_COLORS-1."""
    return list(rng.choice(NUM_COLORS, size=n, replace=False).astype(int))


def _format_coords(cells: list[tuple[int, int]]) -> str:
    """Serialize a coordinate list in the canonical task format."""
    return "[" + ",".join(f"({r},{c})" for r, c in cells) + "]"


def _format_path_end_runs(
    path: list[tuple[int, int]],
    n: int,
) -> str:
    """Serialize the two endpoint runs in a canonical order.

    Each run is listed from the endpoint inward. The two runs are then
    ordered by the endpoint coordinate in row-major order so the answer
    is unique without a marked start cell.
    """
    runs = [
        path[:n],
        list(reversed(path[-n:])),
    ]
    runs.sort(key=lambda run: run[0])
    labels = ["first", "second"]
    parts = []
    for label, run in zip(labels, runs):
        parts.append(f"{label}:{_format_coords(run)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Tasks 23 / 24 -- bouncing diagonals (rewritten 2026-09-07)
# ---------------------------------------------------------------------------
# Geometry shared by both tasks.  A diagonal path starts at a cell with a
# direction (dr, dc) in {-1, 1}^2 and reflects off the grid walls: a step that
# would leave through the top or bottom wall flips dr, one that would leave
# through the left or right wall flips dc.  The reflection is located ON the
# wall cell (the cell whose next step would leave the grid), and its visible
# signature is a direction change at that cell.  Corner cells are never part
# of a trace: a corner hit flips both components at once and is not a
# single-wall reflection, so a reader could not tell how to count it.
#
# Task 23 draws the path: it reflects exactly n times, then runs straight to
# the wall cell it reaches and stops there (the path may overlap itself).
# Task 24 shows a path and asks for the reflection count: its trace is a
# simple diagonal chain (no cell diagonal-adjacent to a non-consecutive trace
# cell, no revisit) so the trace can be followed from the marked start cell.
#
# Review findings (2026-09-07) fixed here: the task-23 marker could overwrite
# a scene cell (two singleton colours, 6 / 10k undecidable) and the prompt
# never said how the marker is identified; task 24 drew high answers from
# thin-grid size pools, so the grid size leaked the answer (spikes at 8 and
# 20).  Now: the marker sits on a background cell with a one-cell halo and its
# colour is named; task 24 samples the size uniformly among the sizes whose
# feasibility table admits the drawn answer; both tasks re-derive the answer
# from the visible input and raise on mismatch; textures and distractors come
# from the shared libraries; colours reach 10.

_DIRS = {"NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)}
DIRECTION_GLOSS = {"NE": "up-right", "NW": "up-left", "SE": "down-right", "SW": "down-left"}

T23_MAX_BOUNCES = 6
T23_BOUNCE_WEIGHTS = (0.08, 0.155, 0.155, 0.155, 0.155, 0.15, 0.15)   # n = 0..6
T23_SCENE_P = 0.80
T23_TEXTURE_P = 0.45
# Task 24 answer prior, applied AFTER the size is drawn and restricted to the answers the
# feasibility table admits at that size (renormalised).  The higher counts get more weight
# because only thin sizes admit them; drawing the answer first and then a size within its
# feasible set concentrated 6 and 7 on those sizes (the size leaked the answer).
T24_BOUNCE_WEIGHTS = (0.06, 0.12, 0.12, 0.12, 0.12, 0.13, 0.16, 0.22, 0.07)  # b = 0..8, trimmed by the table
T24_TEXTURE_P = 0.50
T24_EXTEND_P = {0: 1.0, 1: 0.55, 2: 0.55, 3: 0.6, 4: 0.75, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
T23_SCENE_COLOR_WEIGHTS = (0.18, 0.17, 0.16, 0.15, 0.13, 0.11, 0.10)      # 1..7 scene colours (input tops out at 9: the trace colour is reserved)
T24_DISTRACTOR_COLOR_WEIGHTS = (0.14, 0.16, 0.16, 0.16, 0.14, 0.13, 0.11)  # 0..6 distractor colours
MAX_ATTEMPTS_23 = 60
MAX_ATTEMPTS_24 = 60
TEXTURE_MIN_SIDE = 8


def _corners(h: int, w: int) -> set[tuple[int, int]]:
    return {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _diag_path(
    h: int, w: int, r: int, c: int, dr: int, dc: int, *, max_len: int = 400, stop_on_revisit: bool,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Cells of the bouncing diagonal from (r, c) and the indices of its reflections.

    The first step must stay in bounds (callers guarantee it).  ``turns[i]`` is the
    index of the wall cell at which the (i + 1)-th reflection happens; a reflection
    is only recorded once the cell after it exists in the path."""
    if not (0 <= r + dr < h and 0 <= c + dc < w):
        raise ValueError("first step leaves the grid")
    cells = [(r, c)]
    seen = {(r, c)}
    turns: list[int] = []
    while len(cells) < max_len:
        nr, nc = r + dr, c + dc
        turned = False
        if not 0 <= nr < h:
            dr = -dr
            turned = True
        if not 0 <= nc < w:
            dc = -dc
            turned = True
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):   # cannot happen after the flips; safety
            break
        if stop_on_revisit and (nr, nc) in seen:
            break
        cells.append((nr, nc))
        seen.add((nr, nc))
        if turned:
            turns.append(len(cells) - 2)
        r, c = nr, nc
    return cells, turns


def _clean_prefix_len(cells: list[tuple[int, int]], h: int, w: int) -> int:
    """Length of the longest prefix that is a simple diagonal chain avoiding corners.

    Simple: every cell's diagonal neighbours inside the prefix are exactly its path
    neighbours (no cell diagonal-adjacent to a non-consecutive trace cell, no revisit).
    Both properties are monotone in the prefix length."""
    corners = _corners(h, w)
    inside: set[tuple[int, int]] = set()
    length = 0
    for i, (r, c) in enumerate(cells):
        if (r, c) in corners or (r, c) in inside:
            break
        prev = cells[i - 1] if i else None
        if any((r + dr, c + dc) in inside and (r + dr, c + dc) != prev
               for dr in (-1, 1) for dc in (-1, 1)):
            break
        inside.add((r, c))
        length = i + 1
    return length


def _min_len(b: int) -> int:
    return 5 if b == 0 else 6


def _prefix_ends(turns: list[int], clean_len: int, b: int) -> tuple[list[int], int | None]:
    """End indices of clean prefixes showing exactly *b* reflections.

    Returns ``(extended_ends, wall_end)``: extended prefixes stop in the middle of a
    straight run at least two cells after the last reflection; the wall-terminated
    prefix stops on the wall cell where the next reflection would happen."""
    if b > len(turns):
        return [], None
    t_prev = turns[b - 1] if b else -1
    t_next = turns[b] if b < len(turns) else None
    lo = max(t_prev + 2, _min_len(b) - 1)
    hi = clean_len - 1 if t_next is None else min(t_next - 1, clean_len - 1)
    extended = list(range(lo, hi + 1))
    wall = None
    if t_next is not None and t_next <= clean_len - 1 and t_next >= _min_len(b) - 1:
        wall = t_next
    return extended, wall


def _max_clean_bounces(h: int, w: int) -> int:
    """Largest reflection count some clean trace on an h x w grid can show (exhaustive)."""
    best = -1
    for r in range(h):
        for c in range(w):
            for dr, dc in _DIRS.values():
                if not (0 <= r + dr < h and 0 <= c + dc < w):
                    continue
                cells, turns = _diag_path(h, w, r, c, dr, dc, stop_on_revisit=True)
                clean = _clean_prefix_len(cells, h, w)
                for b in range(len(turns), best, -1):
                    ext, wall = _prefix_ends(turns, clean, b)
                    if ext or wall is not None:
                        best = b
                        break
    return best


# Per-size feasibility table for task 24: ``T24_FEASIBLE[(h, w)]`` is the largest
# reflection count a clean trace can show on an h x w grid, h and w in 5..30.
# Computed once with ``_max_clean_bounces`` (exhaustive over starts and directions)
# and stored as a literal; ``tests/test_paths_23_24.py`` spot-checks it.
_T24_TABLE_ROWS = """
23334444555566667777888899
33343455545666567776788878
33344445566556677667788778
34444454556666566777767788
43444445555666666667777777
44444444555566666666677777
45454444455556666666666777
45545444445555666666666667
55555544444555566666666666
54655554444455556666666666
55665555444445555666666666
56566555544444555566666666
66566655554444455556666666
66666665555444445555666666
65656666555544444555566666
66766666655554444455556666
77766666665555444445555666
77676666666555544444555566
77676666666655554444455556
76777666666665555444445555
87777666666666555544444555
88867766666666655554444455
88877766666666665555444445
88777776666666666555544444
97787776666666666655554444
98887777666666666665555444
""".strip().split("\n")
T24_FEASIBLE: dict[tuple[int, int], int] = {
    (5 + i, 5 + j): int(ch, 16) for i, row in enumerate(_T24_TABLE_ROWS) for j, ch in enumerate(row)
}
T24_MAX_BOUNCES = 7   # answers above this would come from too few sizes (< 50)
# The table gives the MAX clean count per size; the lower counts are all reachable too,
# except these (a straight run of >= 5 corner-free cells does not fit): exhaustive sweep,
# 2026-09-07, tested in tests/test_paths_23_24.py.
T24_INFEASIBLE: frozenset[tuple[int, int, int]] = frozenset({(5, 5, 0), (5, 5, 1), (5, 6, 0), (6, 5, 0)})


def t24_feasible_answers(h: int, w: int) -> list[int]:
    """The reflection counts a clean trace can show on an h x w grid (answer prior support)."""
    top = min(T24_FEASIBLE[(h, w)], T24_MAX_BOUNCES)
    return [b for b in range(top + 1) if (h, w, b) not in T24_INFEASIBLE]


def _first_step_ok(h: int, w: int, r: int, c: int, d: tuple[int, int]) -> bool:
    return 0 <= r + d[0] < h and 0 <= c + d[1] < w


def _place_shapes(rng: np.random.Generator, grid: np.ndarray, bg: int, colors: list[int],
                  n_objects: int, max_cells: int, *, forbidden: np.ndarray | None = None,
                  gap: int = 1, occupancy_cap: float = 0.35) -> tuple[np.ndarray, list[str]]:
    """Stamp up to *n_objects* library shapes on background cells, one cell apart from
    each other and from *forbidden*.  Returns (occupied mask, families)."""
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool) if forbidden is None else forbidden.copy()
    families: list[str] = []
    budget = int(occupancy_cap * h * w)
    for _ in range(n_objects * 10):
        if len(families) >= n_objects or int(occupied.sum()) >= budget:
            break
        shape, fam = sample_shape(rng, max_cells, min(h, 12), min(w, 12))
        sh, sw = shape_size(shape)
        if sh > h or sw > w or len(shape) + int(occupied.sum()) > budget:
            continue
        top = int(rng.integers(0, h - sh + 1))
        left = int(rng.integers(0, w - sw + 1))
        cells = [(top + r, left + c) for r, c in shape]
        if any(blocked[r, c] for r, c in cells):
            continue
        color = int(colors[len(families) % len(colors)])
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        for r, c in cells:
            blocked[max(0, r - gap):r + gap + 1, max(0, c - gap):c + gap + 1] = True
        families.append(fam)
    return occupied, families


def _scene_budget(rng: np.random.Generator, h: int, w: int) -> tuple[int, int]:
    """(number of objects, max cells per object) scaling with the canvas."""
    area = h * w
    n = int(rng.integers(1, max(2, min(12, 2 + area // 70)) + 1))
    max_cells = int(rng.integers(3, max(4, min(40, 4 + area // 25)) + 1))
    return n, max_cells


def _paint_texture(rng: np.random.Generator, grid: np.ndarray, bg: int, occupied: np.ndarray,
                   exclude: list[int], p: float, **kw) -> tuple[np.ndarray, dict]:
    """Shared-library texture on the free cells (or the plain grid)."""
    none = {"texture_group": None, "texture_kind": None, "texture_colors": [], "texture_coverage": 0.0}
    h, w = grid.shape
    if min(h, w) < TEXTURE_MIN_SIDE or rng.random() >= p:
        return grid, none
    painted = apply_background(rng, grid, bg, occupied, exclude=exclude, **kw)
    if painted is None:
        return grid, none
    disp, info = painted
    return disp, {"texture_group": info["group"], "texture_kind": info["kind"],
                  "texture_colors": [int(c) for c in info["texture_colors"]],
                  "texture_coverage": float(info["coverage"])}


def _weighted_int(rng: np.random.Generator, weights) -> int:
    p = np.asarray(weights, dtype=float)
    return int(rng.choice(len(p), p=p / p.sum()))


# ---------------------------------------------------------------------------
# Task 23 -- draw the bouncing diagonal (grid -> grid)
# ---------------------------------------------------------------------------

def _simulate_23(h: int, w: int, start: tuple[int, int], direction: str, n_bounces: int) -> list[tuple[int, int]] | None:
    """The task-23 trace: *n_bounces* reflections, then straight to the next wall cell.
    None when the path would touch a corner or the first step leaves the grid."""
    d = _DIRS[direction]
    if not _first_step_ok(h, w, start[0], start[1], d):
        return None
    cells, turns = _diag_path(h, w, start[0], start[1], d[0], d[1], max_len=600, stop_on_revisit=False)
    if len(turns) < n_bounces + 1:
        return None
    trace = cells[: turns[n_bounces] + 1]
    if any(cell in _corners(h, w) for cell in trace):
        return None
    if len(trace) < (4 if n_bounces == 0 else 5):
        return None
    return trace


def validate_23(inp: np.ndarray, out: np.ndarray, start_color: int, trace_color: int,
                direction: str, n_bounces: int) -> None:
    """Re-derive the output from the visible input and the prompt parameters; raise on any
    mismatch (marker not unique, corner hit, drawn output differs)."""
    inp = np.asarray(inp)
    out = np.asarray(out)
    marks = np.argwhere(inp == start_color)
    if len(marks) != 1:
        raise ValueError(f"task 23: marker colour occurs {len(marks)} times")
    if (inp == trace_color).any():
        raise ValueError("task 23: trace colour already present in the input")
    h, w = inp.shape
    start = (int(marks[0][0]), int(marks[0][1]))
    trace = _simulate_23(h, w, start, direction, n_bounces)
    if trace is None:
        raise ValueError("task 23: trace is not drawable from the prompt")
    expect = inp.copy()
    for r, c in trace:
        expect[r, c] = trace_color
    expect[start] = start_color
    if not np.array_equal(expect, out):
        raise ValueError("task 23: output differs from the re-drawn trace")


@register("23")
def generate_extend_bouncing_diagonal(rng: np.random.Generator) -> dict:
    """Task 23: from the single marked cell, draw the diagonal that reflects exactly n
    times and stops on the next wall cell.  The marker sits on a background cell with a
    one-cell halo of background; scene objects and textures come from the shared
    libraries; the output is re-derived from the visible input (``validate_23``)."""
    n_bounces = _weighted_int(rng, T23_BOUNCE_WEIGHTS)     # kept across retries
    for attempt in range(MAX_ATTEMPTS_23):
        h, w = random_grid_size(rng, min_dim=4, max_dim=30)
        bg, trace_color, start_color = _pick_colors(rng, 3)
        grid = make_grid(h, w, fill=bg)
        occupied = np.zeros((h, w), dtype=bool)
        families: list[str] = []
        scene_colors: list[int] = []
        if rng.random() < T23_SCENE_P:
            n_obj, max_cells = _scene_budget(rng, h, w)
            n_col = _weighted_int(rng, T23_SCENE_COLOR_WEIGHTS) + 1
            scene_colors = [int(c) for c in pick_colors(rng, n_col, exclude=[bg, trace_color, start_color])]
            occupied, families = _place_shapes(rng, grid, bg, scene_colors, max(n_obj, n_col), max_cells)
        grid, tex = _paint_texture(rng, grid, bg, occupied, [bg, trace_color, start_color], T23_TEXTURE_P)
        # marker: a background cell whose whole 8-neighbourhood is background
        free = (grid == bg)
        halo = free.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                shifted = np.ones_like(free)
                src = free[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
                shifted[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)] = src
                halo &= shifted
        spots = np.argwhere(halo)
        if len(spots) == 0:
            continue
        order = rng.permutation(len(spots))
        trace = None
        direction = ""
        for idx in order[:40]:
            start = (int(spots[idx][0]), int(spots[idx][1]))
            for direction in rng.permutation(list(_DIRS)):
                trace = _simulate_23(h, w, start, str(direction), n_bounces)
                if trace is not None:
                    break
            if trace is not None:
                break
        if trace is None:
            continue
        direction = str(direction)
        inp = grid.copy()
        inp[start] = start_color
        out = inp.copy()
        for r, c in trace:
            out[r, c] = trace_color
        out[start] = start_color
        validate_23(inp, out, start_color, trace_color, direction, n_bounces)
        return {
            "input": grid_to_list(inp),
            "output": grid_to_list(out),
            "metadata": {
                "direction": direction,
                "n_bounces": int(n_bounces),
                "start_color": int(start_color),
                "trace_color": int(trace_color),
                "bg_color": int(bg),
                "start_row": int(start[0]), "start_col": int(start[1]),
                "trace_len": int(len(trace)),
                "trace_overlaps": int(len(trace) - len(set(trace))),
                "n_scene_objects": int(len(families)),
                "scene_families": families,
                "scene_colors": scene_colors,
                "n_colors": int(len(np.unique(inp))),
                "grid_h": int(h), "grid_w": int(w),
                "attempts": int(attempt + 1),
                **tex,
            },
        }
    raise RuntimeError(f"task 23: no valid example in {MAX_ATTEMPTS_23} attempts (n_bounces={n_bounces})")


# ---------------------------------------------------------------------------
# Task 24 -- count the reflections (grid -> text)
# ---------------------------------------------------------------------------

def count_reflections_from_grid(grid: np.ndarray, start_color: int, trace_color: int) -> int:
    """Follow the trace from the marked cell and count its direction changes; raise when the
    marker is not unique, the trace does not form one chain, or trace cells are left over."""
    grid = np.asarray(grid)
    marks = np.argwhere(grid == start_color)
    if len(marks) != 1:
        raise ValueError(f"task 24: marker colour occurs {len(marks)} times")
    trace_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == trace_color)}
    cur = (int(marks[0][0]), int(marks[0][1]))
    prev = None
    visited = {cur}
    direction = None
    bounces = 0
    while True:
        nxt = [(cur[0] + dr, cur[1] + dc) for dr in (-1, 1) for dc in (-1, 1)
               if (cur[0] + dr, cur[1] + dc) in trace_cells and (cur[0] + dr, cur[1] + dc) != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise ValueError("task 24: trace branches")
        step = (nxt[0][0] - cur[0], nxt[0][1] - cur[1])
        if direction is not None and step != direction:
            bounces += 1
        direction = step
        prev, cur = cur, nxt[0]
        if cur in visited:
            raise ValueError("task 24: trace revisits a cell")
        visited.add(cur)
    if visited - {(int(marks[0][0]), int(marks[0][1]))} != trace_cells:
        raise ValueError("task 24: trace cells left over")
    return bounces


def _t24_candidate(rng: np.random.Generator, h: int, w: int, b: int) -> tuple[tuple[int, int], list[tuple[int, int]], bool] | None:
    """A clean trace with exactly *b* reflections on an h x w grid: random starts first,
    then the exhaustive sweep (the feasibility table guarantees one exists)."""
    dirs = list(_DIRS.values())
    tried: set[tuple[int, int, int]] = set()

    def attempt(r: int, c: int, k: int):
        dr, dc = dirs[k]
        if not _first_step_ok(h, w, r, c, (dr, dc)):
            return None
        cells, turns = _diag_path(h, w, r, c, dr, dc, stop_on_revisit=True)
        clean = _clean_prefix_len(cells, h, w)
        ext, wall = _prefix_ends(turns, clean, b)
        if not ext and wall is None:
            return None
        use_ext = bool(ext) and (wall is None or rng.random() < T24_EXTEND_P[b])
        end = int(rng.choice(ext)) if use_ext else wall
        return (r, c), cells[: end + 1], (not use_ext)

    for _ in range(300):
        r, c, k = int(rng.integers(0, h)), int(rng.integers(0, w)), int(rng.integers(0, 4))
        if (r, c, k) in tried:
            continue
        tried.add((r, c, k))
        got = attempt(r, c, k)
        if got is not None:
            return got
    found = []
    for r in range(h):
        for c in range(w):
            for k in range(4):
                if (r, c, k) in tried:
                    continue
                got = attempt(r, c, k)
                if got is not None:
                    found.append(got)
    if not found:
        return None
    return found[int(rng.integers(0, len(found)))]


@register("24")
def generate_count_reflections(rng: np.random.Generator) -> dict:
    """Task 24: count the reflections of a drawn diagonal trace.  The grid size is drawn
    first, flat and independent on 5..30 per side, then the answer from the prior
    restricted to the counts the feasibility table admits at that size, so the size
    carries no signal beyond the feasibility itself; the answer is re-derived by
    following the trace on the visible grid."""
    h, w = int(rng.integers(5, 31)), int(rng.integers(5, 31))          # kept across retries
    support = t24_feasible_answers(h, w)
    b = support[_weighted_int(rng, [T24_BOUNCE_WEIGHTS[k] for k in support])]   # kept across retries
    for attempt in range(MAX_ATTEMPTS_24):
        cand = _t24_candidate(rng, h, w, b)
        if cand is None:
            continue
        start, trace, at_wall = cand
        bg, trace_color, start_color = _pick_colors(rng, 3)
        grid = make_grid(h, w, fill=bg)
        protected = np.zeros((h, w), dtype=bool)
        for r, c in trace:
            grid[r, c] = trace_color
            protected[r, c] = True
        grid[start] = start_color
        # distractor shapes one cell clear of the trace, then a shared-library texture
        n_col = _weighted_int(rng, T24_DISTRACTOR_COLOR_WEIGHTS)
        families: list[str] = []
        occupied = protected.copy()
        if n_col:
            colors = [int(c) for c in pick_colors(rng, n_col, exclude=[bg, trace_color, start_color])]
            n_obj, max_cells = _scene_budget(rng, h, w)
            halo = protected.copy()
            for r, c in trace + [start]:
                halo[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
            placed, families = _place_shapes(rng, grid, bg, colors, max(n_obj, n_col), max_cells,
                                             forbidden=halo, occupancy_cap=0.30)
            occupied |= placed
        grid, tex = _paint_texture(rng, grid, bg, occupied, [bg, trace_color, start_color], T24_TEXTURE_P)
        answer = count_reflections_from_grid(grid, start_color, trace_color)
        if answer != b:
            raise ValueError(f"task 24: grid shows {answer} reflections, intended {b}")
        return {
            "input": grid_to_list(grid),
            "output": str(answer),
            "metadata": {
                "start_color": int(start_color),
                "trace_color": int(trace_color),
                "bg_color": int(bg),
                "n_bounces": int(answer),
                "ends_at_wall": bool(at_wall),
                "trace_len": int(len(trace)),
                "n_distractors": int(len(families)),
                "distractor_families": families,
                "n_colors": int(len(np.unique(grid))),
                "grid_h": int(h), "grid_w": int(w),
                "attempts": int(attempt + 1),
                **tex,
            },
        }
    raise RuntimeError(f"task 24: no valid example in {MAX_ATTEMPTS_24} attempts (b={b})")


# ---------------------------------------------------------------------------
# Task 60 — Coordinates of last n cells on both path ends (grid -> text)
# ---------------------------------------------------------------------------

@register("60")
def generate_path_end_coords(rng: np.random.Generator) -> dict:
    from generators.path_ends_60 import generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 61 — Recolor last n cells on both path ends (grid -> grid)
# ---------------------------------------------------------------------------

@register("61")
def generate_recolor_path_ends(rng: np.random.Generator) -> dict:
    from generators.path_runs_61_63 import generate
    return generate(rng, "61")


# ---------------------------------------------------------------------------
# Task 62 — Coordinates of last n cells from marked start (grid -> text)
# ---------------------------------------------------------------------------

@register("62")
def generate_path_far_end_coords(rng: np.random.Generator) -> dict:
    from generators.path_runs_61_63 import generate
    return generate(rng, "62")


# ---------------------------------------------------------------------------
# Task 63 — Recolor last n cells from marked start (grid -> grid)
# ---------------------------------------------------------------------------

@register("63")
def generate_recolor_path_far_end(rng: np.random.Generator) -> dict:
    from generators.path_runs_61_63 import generate
    return generate(rng, "63")
