"""Translation-only fitting into visible, closed holes (task 28).

Construction and verification deliberately use different representations.  The
builder carves geometric bodies and records a proposed placement; the verifier
starts again from the rendered grid, discovers objects and holes, and searches
all translated placements jointly.  Only distinct COMPLETE colored outputs are
limited.  No local cover list, object dimension or planted role is supplied to
the solver.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Iterable

import numpy as np

from common.backgrounds import apply_background
from common.shapes import rect

Cell = tuple[int, int]
Shape = frozenset[Cell]
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COHERENT_SPARSE_KINDS = ('lattice', 'dotted_stripes', 'sparse_checker', 'dotted_border', 'dotted_diagonal')
# A connected host enclosing a two-cell cavity needs at least nine wall cells.
# The smallest compact frames below satisfy both the empty exterior and the
# existing frequency/connected-ground guards; pairs are sampled uniformly.
FEASIBLE_SIZES = tuple((h, w) for h in range(3, 31) for w in range(3, 31)
                      if min(h, w) >= 5 or (min(h, w) == 4 and max(h, w) >= 6)
                      or (min(h, w) == 3 and max(h, w) >= 8))


def prompt28(metadata=None) -> str:
    return (
        "Ignoring the background, each object is a same-color group of at least two cells connected "
        "horizontally or vertically. Objects are separated by background; cells that touch only at a corner "
        "are not connected. "
        "Isolated single cells, whatever their color, are background texture and are ignored. "
        "Objects that enclose holes are fixed hosts; the other objects are loose pieces. "
        "A hole is an empty region completely enclosed by one host; empty cells connect horizontally or vertically. "
        "Translate loose pieces, without rotating or reflecting them, to fill every hole that can be tiled completely. "
        "One piece or several pieces may exactly tile a hole without overlaps; use each piece at most once. "
        "Leave a hole empty if no piece or combination of pieces fits it exactly; never fill a hole partially. "
        "All holes that have an exact fit can be filled simultaneously. "
        "Moved pieces keep every cell's color, and their original cells become the plain background color. "
        "Unused loose pieces stay where they are. "
        "There is exactly one possible final colored grid. Leave every other cell unchanged."
    )


def _normal(cells: Iterable[Cell]) -> Shape:
    cells = frozenset(cells)
    if not cells:
        return cells
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return frozenset((r-r0, c-c0) for r, c in cells)


def _extent(cells: Iterable[Cell]) -> tuple[int, int]:
    cells = list(cells)
    return max(r for r, _ in cells)-min(r for r, _ in cells)+1, max(c for _, c in cells)-min(c for _, c in cells)+1


def _connected(cells: Iterable[Cell]) -> bool:
    left = set(cells)
    if not left:
        return False
    todo = [left.pop()]
    while todo:
        r, c = todo.pop()
        for dr, dc in D4:
            p = r+dr, c+dc
            if p in left:
                left.remove(p)
                todo.append(p)
    return not left


def _components(grid: np.ndarray) -> list[tuple[int, Shape]]:
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    answer = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            todo = [(r, c)]
            seen[r, c] = True
            cells = []
            while todo:
                rr, cc = todo.pop()
                cells.append((rr, cc))
                for dr, dc in D4:
                    nr, nc = rr+dr, cc+dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        todo.append((nr, nc))
            answer.append((color, frozenset(cells)))
    return answer


def _enclosed(cells: Shape) -> list[Shape]:
    """Finite complement regions of one connected body, with 4-neighbor air."""
    r0, r1 = min(r for r, _ in cells)-1, max(r for r, _ in cells)+1
    c0, c1 = min(c for _, c in cells)-1, max(c for _, c in cells)+1
    remaining = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)} - cells
    holes = []
    while remaining:
        start = remaining.pop()
        todo = [start]
        region = {start}
        edge = False
        while todo:
            r, c = todo.pop()
            edge |= r in (r0, r1) or c in (c0, c1)
            for dr, dc in D4:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    region.add(p)
                    todo.append(p)
        if not edge:
            holes.append(frozenset(region))
    return sorted(holes, key=lambda x: min(x))


@dataclass(frozen=True)
class Object:
    cells: Shape

    @property
    def shape(self) -> Shape:
        return _normal(self.cells)


@dataclass(frozen=True)
class Scene:
    background: int
    hosts: tuple[Object, ...]
    holes: tuple[Shape, ...]
    pieces: tuple[Object, ...]
    texture: Shape
    background_margin: float


def parse_scene(input_grid) -> Scene:
    """Infer every semantic role from the visible grid, without metadata."""
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not grid.size:
        raise ValueError("task28: expected a nonempty grid")
    components = _components(grid)
    counts = Counter(map(int, grid.flat))
    if len(counts) < 2:
        raise ValueError("task28: no identifiable background and objects")
    ordered = counts.most_common()
    background, amount = ordered[0]
    runner = ordered[1][1]
    margin = amount / runner
    if amount <= runner or margin < 1.25:
        raise ValueError("task28: background is not visually separated")
    largest_ground = max(len(cells) for color, cells in components if color == background)
    largest_object = max(len(cells) for color, cells in components if color != background)
    if largest_ground <= largest_object:
        raise ValueError("task28: ground region is not identifiable")
    remaining = set(zip(*np.where(grid != background)))
    objects, texture = [], set()
    while remaining:
        start = remaining.pop()
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in D4:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    todo.append(p)
        if len(cells) == 1:
            texture.update(cells)
        else:
            if len({int(grid[p]) for p in cells}) != 1:
                raise ValueError("task28: input objects must be monochrome")
            objects.append(frozenset(cells))
    if objects and largest_ground <= max(map(len, objects)):
        raise ValueError("task28: ground region is smaller than a spatial object")
    texture = frozenset(texture)
    hosts, holes, pieces = [], [], []
    for cells in sorted(objects, key=min):
        voids = _enclosed(cells)
        obj = Object(cells)
        if voids:
            for void in voids:
                if any(int(grid[p]) != background for p in void):
                    raise ValueError("task28: a host contains nonempty enclosed material")
            hosts.append(obj)
            holes.extend(voids)
        else:
            pieces.append(obj)
    if not hosts or not pieces:
        raise ValueError("task28: missing a host or loose pieces")
    host_colors = {int(grid[next(iter(obj.cells))]) for obj in hosts}
    piece_colors = {int(grid[next(iter(obj.cells))]) for obj in pieces}
    if host_colors & piece_colors:
        raise ValueError("task28: loose pieces must have different colors from hosts")
    return Scene(background, tuple(hosts), tuple(sorted(holes, key=min)), tuple(pieces), texture, margin)


def enumerate_final_outputs(input_grid, limit: int = 2, *, stats: dict | None = None) -> list[np.ndarray]:
    """Enumerate distinct complete colored grids, retaining every legal cover.

    Pieces of the same normalized shape AND color form an interchangeable class
    while filling cavities. Class counts replace permutations of physical IDs.
    At a completed cover, every possible subset of erased sources is restored:
    using only some identical pieces still gives different final input erasures.
    Thus this quotient removes duplicate work without conflating colored grids.
    """
    if limit < 2:
        raise ValueError("a uniqueness search must retain at least two outputs")
    grid = np.asarray(input_grid)
    scene = parse_scene(grid)
    destinations = sorted(set().union(*scene.holes))
    cell_index = {p: i for i, p in enumerate(destinations)}
    grouped = defaultdict(list)
    for obj in scene.pieces:
        grouped[(obj.shape, int(grid[next(iter(obj.cells))]))].append(obj)
    classes = list(grouped)
    members = [grouped[key] for key in classes]
    initial = tuple(map(len, members))
    areas = tuple(len(shape) for shape, _ in classes)
    colors = sorted({color for _, color in classes})
    color_ids = [colors.index(color) for _, color in classes]
    placements = []
    for kind, (shape, _) in enumerate(classes):
        sh, sw = _extent(shape)
        for hole in scene.holes:
            if len(shape) > len(hole):
                continue
            for r0 in range(min(r for r, _ in hole), max(r for r, _ in hole)-sh+2):
                for c0 in range(min(c for _, c in hole), max(c for _, c in hole)-sw+2):
                    moved = frozenset((r+r0, c+c0) for r, c in shape)
                    if moved <= hole:
                        placements.append((kind, sum(1 << cell_index[p] for p in moved)))
    at_cell = [[] for _ in destinations]
    for idx, (_, mask) in enumerate(placements):
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            at_cell[bit.bit_length()-1].append(idx)
            remaining ^= bit
    hole_masks = [sum(1 << cell_index[p] for p in hole) for hole in scene.holes]
    feasibility_nodes = nodes = 0

    @lru_cache(maxsize=None)
    def area_possible(area, left):
        reachable = 1
        for size, count in zip(areas, left):
            for _ in range(count):
                reachable |= reachable << size
        return bool((reachable >> area) & 1)

    def branch(remaining, left):
        choices = None
        cells = remaining
        while cells:
            bit = cells & -cells
            available = [idx for idx in at_cell[bit.bit_length()-1]
                         if left[placements[idx][0]]
                         and placements[idx][1] & remaining == placements[idx][1]]
            if not available:
                return []
            if choices is None or len(available) < len(choices):
                choices = available
                if len(choices) == 1:
                    break
            cells ^= bit
        return choices

    @lru_cache(maxsize=None)
    def can_fill(remaining, left):
        nonlocal feasibility_nodes
        feasibility_nodes += 1
        if not remaining:
            return True
        if not area_possible(remaining.bit_count(), left):
            return False
        for idx in branch(remaining, left):
            kind, mask = placements[idx]
            after = list(left)
            after[kind] -= 1
            if can_fill(remaining ^ mask, tuple(after)):
                return True
        return False

    # This existence query stops at its first witness. The subsequent joint
    # search starts from every placement, never from truncated local cover lists.
    fillable = [can_fill(mask, initial) for mask in hole_masks]
    all_cells = sum(mask for mask, fits in zip(hole_masks, fillable) if fits)
    outputs, seen = {}, set()

    def render_sources(left, paint):
        output = grid.copy()
        for color, mask in zip(colors, paint):
            while mask:
                bit = mask & -mask
                output[destinations[bit.bit_length()-1]] = color
                mask ^= bit

        def choose_sources(kind):
            if len(outputs) >= limit:
                return
            if kind == len(classes):
                outputs.setdefault(output.tobytes(), output.copy())
                return
            count = initial[kind]-left[kind]
            for subset in combinations(range(initial[kind]), count):
                for index in subset:
                    for p in members[kind][index].cells:
                        output[p] = scene.background
                choose_sources(kind+1)
                for index in subset:
                    for p in members[kind][index].cells:
                        output[p] = grid[p]
                if len(outputs) >= limit:
                    return

        choose_sources(0)

    def search(remaining, left, paint):
        nonlocal nodes
        nodes += 1
        if len(outputs) >= limit:
            return
        state = remaining, left, paint
        if state in seen:
            return
        seen.add(state)
        if not remaining:
            render_sources(left, paint)
            return
        if not area_possible(remaining.bit_count(), left):
            return
        for idx in branch(remaining, left):
            kind, mask = placements[idx]
            after, colored = list(left), list(paint)
            after[kind] -= 1
            colored[color_ids[kind]] |= mask
            search(remaining ^ mask, tuple(after), tuple(colored))
            if len(outputs) >= limit:
                return

    search(all_cells, initial, (0,)*len(colors))
    if stats is not None:
        stats.update(search_nodes=nodes, feasibility_nodes=feasibility_nodes,
                     placements=len(placements), piece_classes=len(classes),
                     distinct_outputs=len(outputs), fillable_holes=sum(fillable),
                     unfilled_holes=len(fillable)-sum(fillable))
    return list(outputs.values())


def validate_fit_holes(input_grid, output_grid) -> dict:
    stats = {}
    outputs = enumerate_final_outputs(input_grid, stats=stats)
    if len(outputs) != 1:
        raise ValueError(f"task28: expected one final colored output, found {len(outputs)}")
    if not np.array_equal(outputs[0], np.asarray(output_grid)):
        raise ValueError("task28: output disagrees with the independent visible-grid solver")
    return stats


def _halo(cells: Iterable[Cell], radius=1) -> set[Cell]:
    return {(r+dr, c+dc) for r, c in cells
            for dr in range(-radius, radius+1) for dc in range(-radius, radius+1)}


def _geometric(rng, h: int, w: int, family: str, *, host=False) -> Shape:
    cells = set(rect(rng, h, w))
    if family in ("chamfer", "diamond"):
        cut = max(1, min(h, w)//(3 if family == "diamond" else 5))
        cells = {(r, c) for r, c in cells
                 if min(r+c, r+w-1-c, h-1-r+c, h-1-r+w-1-c) >= cut}
    elif family == "L":
        th, tw = (min(h, max(4, 2*h//3)), min(w, max(4, 2*w//3))) if host else (max(1, h//2), max(1, w//2))
        cells = {(r, c) for r, c in cells if r < th or c < tw}
    elif family == "T":
        th = min(h, max(4, 2*h//3)) if host else max(1, (h+1)//2)
        tw = min(w, max(3, 2*w//3)) if host else max(1, (w+1)//2)
        c0 = (w-tw)//2
        cells = {(r, c) for r, c in cells if r < th or c0 <= c < c0+tw}
    elif family == "plus":
        dh, dw = max(1, h//4), max(1, w//4)
        cells = {(r, c) for r, c in cells if dh <= r < h-dh or dw <= c < w-dw}
    elif family == "step":
        # Three broad steps, not a random boundary walk.
        bite = max(1, w//4)
        cells = {(r, c) for r, c in cells if c < w-bite*(2 if r < h//3 else 1 if r < 2*h//3 else 0)}
    elif family == "union":
        dh, dw = max(1, h//4), max(1, w//4)
        cells = {(r, c) for r, c in cells if not (r < dh and c < dw) and not (r >= h-dh and c >= w-dw)}
    if host and min(h, w) <= 5 and family in ("chamfer", "diamond"):
        # A broad corner remains around the smallest cavity.
        cells.add((0, 0))
    return _normal(cells)


def _split(rng, shape: Shape, count: int) -> list[Shape] | None:
    parts = [shape]
    while len(parts) < count:
        choices = []
        for index, part in enumerate(parts):
            for axis in (0, 1):
                lo, hi = min(p[axis] for p in part), max(p[axis] for p in part)
                for cut in range(lo+1, hi+1):
                    first = frozenset(p for p in part if p[axis] < cut)
                    second = part-first
                    if min(len(first), len(second)) < 2 or not _connected(first) or not _connected(second):
                        continue
                    if _enclosed(first) or _enclosed(second):
                        continue
                    choices.append((index, first, second))
        if not choices:
            return None
        index, first, second = choices[int(rng.integers(len(choices)))]
        parts[index:index+1] = [first, second]
    return parts


def _colored_objects(rng, objects: list[Shape], palette: list[int], n_hosts=1) -> list[dict[Cell, int]]:
    """Monochrome objects, with disjoint parent/piece color roles.

    Several pieces may reuse a color, including moved and unused pieces. Color
    does not indicate which cavity fits. Palette capacity comes from objects,
    never from splitting one object into multicolor bands.
    """
    n_pieces = len(objects) - n_hosts
    if not 2 <= len(palette) <= len(objects) or n_pieces < 1:
        raise ValueError("task28: infeasible monochrome palette budget")
    low = max(1, len(palette) - n_pieces)
    host_count = int(rng.integers(low, min(n_hosts, len(palette) - 1) + 1))
    colors = []
    for pool, count in ((palette[:host_count], n_hosts), (palette[host_count:], n_pieces)):
        assigned = pool + [int(rng.choice(pool)) for _ in range(count - len(pool))]
        rng.shuffle(assigned)
        colors.extend(assigned)
    return [{p: color for p in shape} for shape, color in zip(objects, colors)]


@lru_cache(maxsize=512)
def _positions(shape: Shape, height: int, width: int, inset: int = 0):
    """All actual footprint placements, with a clipped full one-cell moat."""
    sh, sw = _extent(shape)
    positions = []
    base = sum(1 << (r*width+c) for r, c in shape)
    left_edge = sum(1 << (r*width) for r in range(height))
    right_edge = left_edge << (width-1)
    canvas = (1 << (height*width))-1
    for r0 in range(inset, height-sh-inset+1):
        for c0 in range(inset, width-sw-inset+1):
            mask = base << (r0*width+c0)
            horizontal = mask | ((mask & ~left_edge) >> 1) | ((mask & ~right_edge) << 1)
            moat = (horizontal | (horizontal << width) | (horizontal >> width)) & canvas
            positions.append((r0, c0, mask, moat))
    return tuple(positions)


def _place_joint(rng, shapes, height, width, *, n_hosts=0, inset=0, restarts=8, edge_last=False):
    """Place hulls and inserts in one shared free field using actual footprints.

    No region is assigned to a role. Large/restrictive shapes go first; random
    offsets and shared moats leave staggered and interleaved placements. The
    reserved hull includes its cavities, so a loose source cannot be nested in it.
    """
    options = [_positions(shape, height, width, inset if i < n_hosts else 0)
               for i, shape in enumerate(shapes)]
    if edge_last:
        sh, sw = _extent(shapes[-1])
        options[-1] = [p for p in options[-1] if p[0] in (0, height-sh) or p[1] in (0, width-sw)]
    if any(not positions for positions in options):
        return None
    order = sorted(range(len(shapes)), key=lambda i: (-len(shapes[i]), len(options[i])))
    for _ in range(restarts):
        blocked = 0
        answer = [None]*len(shapes)
        for i in order:
            legal = [p for p in options[i] if not p[2] & blocked]
            if not legal:
                break
            # Sharing exterior moats packs more cleanly than repeatedly taking
            # random central sites. Random sampling retains a broad range of
            # coordinates instead of selecting a lattice of attachment points.
            if blocked and rng.random() < .72:
                subset = [legal[j] for j in rng.choice(len(legal), min(18, len(legal)), replace=False)]
                scores = [(p[3] & blocked).bit_count()+float(rng.uniform(0, 5)) for p in subset]
                chosen = subset[int(np.argmax(scores))]
            else:
                chosen = legal[int(rng.integers(len(legal)))]
            r0, c0, _, moat = chosen
            answer[i] = frozenset((r+r0, c+c0) for r, c in shapes[i])
            blocked |= moat
        else:
            return answer
    return None


def _shelf_fits(boxes, height, width):
    """A sufficient footprint proof used only to select feasible count budgets.

    Each box includes a moat on its bottom/right. These shelves never become
    generated layouts: final geometry uses the free actual-footprint packer.
    """
    for transpose in (False, True):
        bounds = (width+1, height+1) if transpose else (height+1, width+1)
        oriented = [(w+1, h+1) if transpose else (h+1, w+1) for h, w in boxes]
        for axis in (0, 1):
            shelves = []
            for h, w in sorted(oriented, key=lambda p: (-p[axis], -p[1-axis])):
                if h > bounds[0] or w > bounds[1]:
                    break
                fits = [(space-w, i) for i, (sh, space) in enumerate(shelves) if h <= sh and w <= space]
                if fits:
                    _, index = min(fits)
                    sh, space = shelves[index]
                    shelves[index] = sh, space-w
                elif sum(sh for sh, _ in shelves)+h <= bounds[0]:
                    shelves.append((h, bounds[1]-w))
                else:
                    break
            else:
                return True
    return False


@lru_cache(maxsize=None)
def _count_capacity(height, width):
    """Feasible hole/parent counts from concrete minimum hull/source footprints.

    A 2x2 cavity has a full surrounding wall. Two 2x2 source boxes per cavity
    budget more space than the usual pair of dominoes; their moats are included.
    A few boundary cases use the separate narrow-envelope construction.
    """
    inset = int(height >= 9)
    pairs = []
    for holes in range(1, 13):
        for hosts in range(1, min(6, holes)+1):
            distribution = [holes//hosts+int(i < holes % hosts) for i in range(hosts)]
            for aspect in (.55, 1., 1.8):
                boxes = []
                for count in distribution:
                    options = [(3*rows+1, 3*((count+rows-1)//rows)+1) for rows in range(1, count+1)]
                    options = [(h, w) for h, w in options if h <= height-2*inset and w <= width-2*inset]
                    if not options:
                        break
                    boxes.append(min(options, key=lambda p: p[0]*p[1]*(1+.3*abs(np.log((p[0]/p[1])/aspect)))))
                else:
                    if _shelf_fits(boxes+[(2, 2)]*(2*holes), height-2*inset, width-2*inset):
                        pairs.append((holes, hosts))
                        break
    if not pairs:
        raise ValueError("task28: canvas has no proven multi-object footprint capacity")
    return tuple(pairs)


@lru_cache(maxsize=None)
def _background_capacity(height, width, distribution, sources):
    """A reserved 3x3 exterior patch has nine real sites for sparse detail."""
    inset = int(height >= 9)
    for aspect in (.55, 1., 1.8):
        boxes = []
        for count in distribution:
            options = [(3*rows+1, 3*((count+rows-1)//rows)+1) for rows in range(1, count+1)]
            options = [(h, w) for h, w in options if h <= height-2*inset and w <= width-2*inset]
            if not options:
                break
            boxes.append(min(options, key=lambda p: p[0]*p[1]*(1+.3*abs(np.log((p[0]/p[1])/aspect)))))
        else:
            if _shelf_fits(boxes+[(2, 2)]*sources+[(3, 3)], height-2*inset, width-2*inset):
                return True
    return False


def _draw_plan(rng, H, W):
    area = H*W
    narrow = H <= 4
    small = not narrow and H <= 7 and W <= 8
    capacity = ((1, 1),) if narrow or small else _count_capacity(H, W)
    max_holes = max(k for k, _ in capacity)
    upper = min(3, max_holes) if rng.random() < .28 else max_holes
    n_holes = int(rng.integers(1, upper+1))
    parent_choices = [n for k, n in capacity if k == n_holes]
    n_hosts = int(rng.choice(parent_choices))
    distribution = [1]*n_hosts
    for _ in range(n_holes-n_hosts):
        # Randomized balanced loads support distinct parent sizes without making
        # one overfull body defeat the count that was selected for this canvas.
        least = min(distribution)
        choices = [i for i, n in enumerate(distribution) if n <= least]
        distribution[int(rng.choice(choices))] += 1
    pieces_per_hole = [1]*n_holes
    if not narrow and not small:
        for i in range(n_holes):
            pieces_per_hole[i] = 2 if rng.random() < .48 else 1
            if area >= 240 and n_holes <= 6 and rng.random() < .13:
                pieces_per_hole[i] = 3 if rng.random() < .78 else 4
        if n_holes >= 8 and rng.random() < .18:
            pieces_per_hole = [2]*n_holes
        if n_holes >= 2 and rng.random() < .32:
            count = 2 if n_holes >= 5 and rng.random() < .23 else 1
            for i in rng.choice(n_holes, count, replace=False):
                pieces_per_hole[int(i)] = 0
    strays = 0
    if not small and not narrow and area >= 75 and rng.random() < .43:
        strays = int(rng.integers(1, 3 if area < 300 else 4))
    if narrow and area >= 60 and rng.random() < .35:
        strays = 1
    while sum(pieces_per_hole)+strays > 24:
        index = max(range(n_holes), key=lambda i: pieces_per_hole[i])
        pieces_per_hole[index] -= 1
    colored_fits = bool(not narrow and not small and H >= 7 and area/n_holes >= 65 and n_holes <= 5
                        and max(pieces_per_hole) >= 2 and rng.random() < .38)
    color_shapes = min(sum(n > 0 for n in pieces_per_hole), 1+int(area/n_holes >= 100)+int(area/n_holes >= 250))
    if colored_fits:
        color_shapes = max(2, color_shapes)
    max_palette = min(10, 1+n_hosts+color_shapes+strays)
    palette_size = int(rng.integers(4 if colored_fits else 3, max_palette+1))
    return dict(hosts=n_hosts, holes=n_holes, holes_per_host=distribution,
                pieces_per_hole=pieces_per_hole, strays=strays, palette_size=palette_size,
                inset=int(H >= 9 and rng.random() < .9),
                texture_colors=int(H >= 8 and _background_capacity(H, W, tuple(distribution),
                                                                sum(pieces_per_hole)+strays) and rng.random() < .30),
                texture_kind=str(rng.choice(COHERENT_SPARSE_KINDS)),
                capacity_holes=max_holes, capacity_hosts=max(n for _, n in capacity),
                narrow=narrow, small=small, colored_fits=colored_fits,
                broad_cavities=bool(n_holes == 1 and H >= 12 and area >= 240
                                    and strays == 0 and rng.random() < .45))


def _make_host(rng, shapes, H, W, target_area):
    """Place actual cavity footprints, then choose a hull that encloses them.

    Candidate sizes have a concrete cavity-and-moat capacity witness. The
    cavities themselves are freely packed, so those witnesses do not impose
    shelves, regular grids or an order on the rendered composition.
    """
    boxes = [_extent(shape) for shape in shapes]
    largest_h, largest_w = max(h for h, _ in boxes)+2, max(w for _, w in boxes)+2
    aspect = float(rng.uniform(.6, 1.65))*min(1.4, H/W+.5)
    sizes = [(h, w) for h in range(largest_h, H+1) for w in range(largest_w, W+1)
             if _shelf_fits(boxes, h-2, w-2)]
    sizes.sort(key=lambda p: abs(np.log(p[0]*p[1]/target_area))+.28*abs(np.log((p[0]/p[1])/aspect)))
    for h, w in sizes[:6]:
        cluster = _place_joint(rng, shapes, h-2, w-2, restarts=14)
        if cluster is None:
            continue
        cavities = [frozenset((r+1, c+1) for r, c in shape) for shape in cluster]
        empty = set().union(*cavities)
        families = ['rect', 'chamfer', 'diamond', 'L', 'T', 'plus', 'step', 'union']
        weights = [.10, .26, .07, .12, .11, .09, .13, .12]
        choices, probabilities = [], []
        for family, probability in zip(families, weights):
            hull = _geometric(rng, h, w, family, host=True)
            if len(hull) == h*w and family != 'rect':
                continue
            if not empty <= hull:
                continue
            body = hull-empty
            if _connected(body) and set(_enclosed(frozenset(body))) == set(cavities):
                choices.append((hull, cavities, family))
                probabilities.append(probability)
        if choices:
            probability = np.asarray(probabilities)/sum(probabilities)
            return choices[int(rng.choice(len(choices), p=probability))]
    return None


def _shape_tiling_exists(hole, shapes):
    """Construction-only local check; source IDs/colors are deliberately absent."""
    inventory = Counter(shapes)
    kinds, counts = tuple(inventory), tuple(inventory.values())
    cells = {p: i for i, p in enumerate(sorted(hole))}
    moves = []
    for i, shape in enumerate(kinds):
        anchor = min(shape)
        for r0, c0 in hole:
            moved = {(r+r0-anchor[0], c+c0-anchor[1]) for r, c in shape}
            if moved <= hole:
                moves.append((i, sum(1 << cells[p] for p in moved)))
    by_cell = [[] for _ in hole]
    for kind, mask in moves:
        for index in range(len(hole)):
            if mask & (1 << index):
                by_cell[index].append((kind, mask))

    @lru_cache(maxsize=None)
    def fill(empty, left):
        if not empty:
            return True
        first = (empty & -empty).bit_length()-1
        for kind, mask in by_cell[first]:
            if left[kind] and mask & empty == mask:
                after = list(left)
                after[kind] -= 1
                if fill(empty ^ mask, tuple(after)):
                    return True
        return False

    return fill((1 << len(hole))-1, counts)


def _build(rng, H: int, W: int, plan: dict) -> tuple[np.ndarray, np.ndarray, dict] | None:
    area = H*W
    # A selected count is held fixed; sizes vary around a canvas-scaled budget.
    # Few-cavity scenes receive broad cavities and bodies even on the largest
    # grids. High counts use smaller individual shapes, with no tiny-cell noise.
    cavity_fraction = .18 if plan['broad_cavities'] else .10
    unit = np.sqrt(cavity_fraction*area/plan['holes'])
    compact_multi = (unit < 2.8 and sum(plan['pieces_per_hole'])+plan['strays'] > 2*plan['holes'])
    holes, parts, hole_families = [], [], []
    for count in plan['pieces_per_hole']:
        sh = max(2, int(round(unit*float(rng.uniform(.7, 1.2)))))
        sw = max(2, int(round(unit*float(rng.uniform(.7, 1.3)))))
        if count >= 3:
            sh, sw = max(3, sh), max(3 if count == 3 else 4, sw)
        sh, sw = min(H-2, sh), min(W-2, sw)
        family = str(rng.choice(['rect', 'L', 'step', 'chamfer', 'T', 'union'],
                                p=[.34, .18, .15, .12, .12, .09]))
        if compact_multi and count >= 3 and not plan['colored_fits']:
            # Three/four dominoes fit readable 2x3/2x4 cavities. Requiring a
            # 3x3/3x4 box inflated both the hull and loose-source footprints in
            # plans whose source count already exceeds two per cavity.
            sh, sw, family = 2, count, 'rect'
            if rng.random() < .3:
                sh, sw = sw, sh
        if plan['colored_fits'] and count > 1:
            sh, sw = max(3, sh), max(3, sw)
            family = str(rng.choice(['L', 'step', 'T', 'union'], p=[.35, .30, .20, .15]))
        shape = _geometric(rng, sh, sw, family)
        if len(shape) < max(3, 2*count):
            shape, family = _geometric(rng, sh, sw, "rect"), "rect"
        if len(shape) < max(3, 2*count) or not _connected(shape) or _enclosed(shape):
            return None
        split = _split(rng, shape, count) if count > 1 else [shape] if count else []
        if split is None:
            return None
        holes.append(shape)
        parts.append(split)
        hole_families.append('rect' if len(shape) == sh*sw else family)
    target_shapes = []
    target_holes = []
    for i, split in enumerate(parts):
        for part in split:
            target_shapes.append(_normal(part))
            target_holes.append(i)
    extras = []
    compact_stray = None
    if plan['strays'] and unit < 2.8:
        # Count capacity reserves 2x2 source footprints. A 3x3 plus overbooks
        # tight canvases, especially with two strays. Clean L trominoes stay
        # inside the budget and remain plausible translated-piece candidates.
        # One shared orientation avoids making complementary extra pairs, and
        # an intended source's shape cannot be reused without erasure ambiguity.
        square = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
        choices = [square-{corner} for corner in sorted(square)
                   if square-{corner} not in target_shapes]
        if not choices:
            return None
        compact_stray = choices[int(rng.integers(len(choices)))]
    for _ in range(plan['strays']):
        if compact_stray is not None:
            extras.append(compact_stray)
            continue
        sh = max(2, int(round(unit*float(rng.uniform(.55, 1.2)))))
        sw = max(2, int(round(unit*float(rng.uniform(.6, 1.2)))))
        family = str(rng.choice(['rect', 'L', 'T', 'step', 'union']))
        shape = _geometric(rng, min(sh, H-2), min(sw, W-2), family)
        if len(shape) < 2 or not _connected(shape) or _enclosed(shape):
            return None
        extras.append(shape)
    if any(shape in target_shapes for shape in extras):
        return None
    source_shapes = target_shapes+extras
    # Reserve genuinely unfillable cavities before spending work on placement.
    # With many small inserts an arbitrary empty rectangle is usually tileable;
    # simple crosses, bars and broad corners provide readable impossible fits.
    for index, count in enumerate(plan['pieces_per_hole']):
        if count:
            continue
        choices = [(holes[index], hole_families[index])]
        for _ in range(18):
            sh = min(H-2, max(2, int(round(unit*float(rng.uniform(.75, 1.5))))))
            sw = min(W-2, max(2, int(round(unit*float(rng.uniform(.75, 1.5))))))
            family = str(rng.choice(['plus', 'L', 'T', 'rect', 'step', 'chamfer']))
            if family == 'plus':
                sh, sw = max(3, sh), max(3, sw)
            candidate = _geometric(rng, sh, sw, family)
            if len(candidate) >= 3 and _connected(candidate) and not _enclosed(candidate):
                choices.append((candidate, family))
        for candidate, family in choices:
            if not _shape_tiling_exists(candidate, source_shapes):
                holes[index], hole_families[index] = candidate, family
                break
        else:
            return None
    hulls, host_holes, families = [], [], []
    offset = 0
    for count in plan['holes_per_host']:
        local = holes[offset:offset+count]
        target_area = max(sum(map(len, local))/(.58 if plan['broad_cavities'] else .30),
                          .31*area*count/plan['holes'])
        target_area *= float(rng.uniform(.92, 1.12))
        built = _make_host(rng, local, H-2*plan['inset'], W-2*plan['inset'], target_area)
        if built is None:
            return None
        hull, cavities, family = built
        hulls.append(hull)
        host_holes.append(cavities)
        families.append(family)
        offset += count
    reserved = [frozenset((r, c) for r in range(3) for c in range(3))] if plan['texture_colors'] else []
    placed = _place_joint(rng, hulls+source_shapes+reserved, H, W, n_hosts=plan['hosts'],
                          inset=plan['inset'], edge_last=bool(reserved))
    if placed is None:
        return None
    hulls_world = placed[:plan['hosts']]
    sources = placed[plan['hosts']:plan['hosts']+len(source_shapes)]
    holes_world, bodies = [], []
    for hull, cavities in zip(hulls_world, host_holes):
        r0, c0 = min(r for r, _ in hull), min(c for _, c in hull)
        shifted = [frozenset((r+r0, c+c0) for r, c in cavity) for cavity in cavities]
        holes_world.extend(shifted)
        bodies.append(hull-set().union(*shifted))
    foreground = sum(map(len, bodies))+sum(map(len, sources))
    if not .20 <= foreground/area <= .47:
        return None

    # Identical cavities or interchangeable split parts share a fill color.
    # Otherwise exchanging two identical pieces would visibly swap their colors.
    # This necessary condition leaves the visible solver responsible for
    # every possible alternative tiling (including composite alternatives).
    if plan['colored_fits']:
        group_keys = list(dict.fromkeys(target_shapes))
        piece_groups = [group_keys.index(shape) for shape in target_shapes]
        piece_groups.extend(range(len(group_keys), len(group_keys)+len(extras)))
        group_count = len(group_keys)+len(extras)
        multicolor_choices = [sorted({piece_groups[j] for j, hole in enumerate(target_holes) if hole == i})
                              for i in range(len(holes))]
        multicolor_choices = [groups for groups in multicolor_choices if len(groups) >= 2]
        if not multicolor_choices:
            return None
    else:
        links = list(range(len(holes)))

        def representative(index):
            while links[index] != index:
                links[index] = links[links[index]]
                index = links[index]
            return index

        seen_cavities, seen_parts = {}, {}
        for index, split in enumerate(parts):
            if not split:
                continue
            matches = []
            if holes[index] in seen_cavities:
                matches.append(seen_cavities[holes[index]])
            seen_cavities[holes[index]] = index
            for part in split:
                shape = _normal(part)
                if shape in seen_parts:
                    matches.append(seen_parts[shape])
                seen_parts[shape] = index
            for previous in matches:
                links[representative(previous)] = representative(index)
        group_keys = sorted({representative(i) for i in target_holes})
        piece_groups = [group_keys.index(representative(i)) for i in target_holes]
        piece_groups.extend(range(len(group_keys), len(group_keys)+len(extras)))
        group_count = len(group_keys)+len(extras)
    want = plan['palette_size']-1
    lower = max(1, want-group_count)
    upper = min(plan['hosts'], want-(2 if plan['colored_fits'] else 1))
    if lower > upper:
        return None
    host_color_count = int(rng.integers(lower, upper+1))
    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    host_pool = palette[1:host_color_count+1]
    piece_pool = palette[host_color_count+1:want+1]
    host_colors = host_pool+[int(rng.choice(host_pool)) for _ in range(plan['hosts']-host_color_count)]
    rng.shuffle(host_colors)
    if plan['colored_fits']:
        pair = rng.choice(multicolor_choices[int(rng.integers(len(multicolor_choices)))], 2, replace=False)
        group_colors = [None]*group_count
        for index, color in zip(pair, piece_pool[:2]):
            group_colors[int(index)] = color
        free = [i for i, color in enumerate(group_colors) if color is None]
        rng.shuffle(free)
        for index, color in zip(free, piece_pool[2:]):
            group_colors[index] = color
        group_colors = [int(rng.choice(piece_pool)) if color is None else color for color in group_colors]
    else:
        group_colors = piece_pool+[int(rng.choice(piece_pool)) for _ in range(group_count-len(piece_pool))]
        rng.shuffle(group_colors)
    source_colors = [group_colors[index] for index in piece_groups]
    grid = np.full((H, W), bg, dtype=np.uint8)
    output = grid.copy()
    for body, color in zip(bodies, host_colors):
        for p in body:
            grid[p] = output[p] = color
    for index, (source, color) in enumerate(zip(sources, source_colors)):
        for p in source:
            grid[p] = color
            if index >= len(target_shapes):
                output[p] = color
    source_index = 0
    for i, split in enumerate(parts):
        r0 = min(r for r, _ in holes_world[i])
        c0 = min(c for _, c in holes_world[i])
        for part in split:
            for r, c in part:
                output[r+r0, c+c0] = source_colors[source_index]
            source_index += 1
    texture_info = {'group': 'solid', 'kind': 'uniform', 'texture_colors': []}
    if plan['texture_colors']:
        occupied = np.zeros((H, W), bool)
        for r, c in _halo(set().union(*hulls_world, *sources)):
            if 0 <= r < H and 0 <= c < W:
                occupied[r, c] = True
        blank = np.full_like(grid, bg)
        texture = apply_background(rng, blank, bg, occupied, group='sparse',
                                   kind=plan['texture_kind'], colors=[int(rng.choice(piece_pool))])
        if texture is None:
            return None
        textured, texture_info = texture
        changed = textured != blank
        output[changed] = grid[changed] = textured[changed]
    metadata = dict(n_hosts=len(hulls), n_holes=len(holes), n_loose_pieces=len(sources),
        n_moved_pieces=len(target_shapes), n_multi_piece_holes=sum(n > 1 for n in plan['pieces_per_hole']),
        n_strays=plan['strays'], host_families=families, hole_families=hole_families,
        host_cells=list(map(len, bodies)), hole_areas=list(map(len, holes)), piece_areas=list(map(len, source_shapes)),
        background=texture_info['group'], background_kind=texture_info['kind'],
        texture_colors=texture_info['texture_colors'], background_color=bg, foreground_fraction=foreground/area,
        connectivity=4, no_local_cover_truncation=True, multicolor_objects=0,
        semantic_colors=want, requested_palette_size=plan['palette_size'], pieces_per_hole=plan['pieces_per_hole'],
        placement='joint_actual_footprints',
        n_multicolor_filled_holes=sum(len({int(output[p]) for p in cavity}) > 1 for cavity in holes_world))
    return grid, output, metadata


def _small_scene(rng, H, W, plan):
    """The real small-envelope family: a thin frame and its two/three-cell insert."""
    hole_w = int(rng.integers(2, min(3, W-2)+1))
    host_w = hole_w+2
    colors = list(map(int, rng.permutation(10)))
    bg = colors[0]
    grid = np.full((H, W), bg, dtype=np.uint8)
    r0 = int(rng.integers(0, H-4))
    c0 = int(rng.integers(0, W-host_w+1))
    pr = int(rng.integers(r0+4, H))
    pc = int(rng.integers(0, W-hole_w+1))
    body = frozenset((r, c) for r in range(r0, r0+3) for c in range(c0, c0+host_w)
                     if r != r0+1 or c in (c0, c0+host_w-1))
    chamfer = rng.random() < .35
    if chamfer:
        body = body-{(r0, c0)}
    piece = frozenset((0, c) for c in range(hole_w))
    painted = _colored_objects(rng, [body, piece], colors[1:plan['palette_size']])
    for p, color in painted[0].items():
        grid[p] = color
    for (r, c), color in painted[1].items():
        grid[pr+r, pc+c] = color
    output = grid.copy()
    output[pr, pc:pc+hole_w] = bg
    for (r, c), color in painted[1].items():
        output[r0+1+r, c0+1+c] = color
    metadata = dict(n_hosts=1, n_holes=1, n_loose_pieces=1, n_moved_pieces=1,
        n_multi_piece_holes=0, n_strays=0, host_families=['chamfer' if chamfer else 'rect'], hole_families=['bar'],
        host_cells=[len(body)], hole_areas=[hole_w], piece_areas=[hole_w],
        background='solid', background_kind='uniform', texture_colors=[], background_color=bg,
        foreground_fraction=(len(body)+hole_w)/(H*W), connectivity=4, no_local_cover_truncation=True,
        multicolor_objects=sum(len(set(pixels.values())) > 1 for pixels in painted),
        semantic_colors=plan['palette_size']-1, requested_palette_size=plan['palette_size'], pieces_per_hole=[1])
    return grid, output, metadata


def _narrow_scene(rng, H, W, plan):
    """Three/four-row frames with scaled rectangular cavities and loose inserts."""
    chamfer = H*W == 24 or rng.random() < .35
    minimum = 2 if H == 3 else 1
    maximum = min((W-3)//2, (H*W-4*H+int(chamfer)-2*plan['strays']-1)//(2*H))
    if maximum < minimum:
        return None
    high = min(maximum, max(minimum, int(.44*W)-2))
    low = min(high, max(minimum, int(.30*W)-2))
    width = int(rng.integers(low, high+1))
    hole = frozenset((r, c) for r in range(1, H-1) for c in range(1, width+1))
    hull = frozenset((r, c) for r in range(H) for c in range(width+2))
    body = hull-hole
    if chamfer:
        body = body-{(0, 0)}
    specs = [_normal(hole)]+[frozenset({(0, 0), (0, 1)}) for _ in range(plan['strays'])]
    sources, blocked = [], _halo(hull)
    for shape in specs:
        sh, sw = _extent(shape)
        candidates = []
        for r0 in range(H-sh+1):
            for c0 in range(width+3, W-sw+1):
                placed = frozenset((r+r0, c+c0) for r, c in shape)
                if not placed & blocked:
                    candidates.append(placed)
        if not candidates:
            return None
        placed = candidates[int(rng.integers(len(candidates)))]
        sources.append(placed)
        blocked |= _halo(placed)
    colors = list(map(int, rng.permutation(10)))
    bg = colors[0]
    painted = _colored_objects(rng, [body]+specs, colors[1:plan['palette_size']])
    grid = np.full((H, W), bg, dtype=np.uint8)
    for p, color in painted[0].items():
        grid[p] = color
    for pixels, source in zip(painted[1:], sources):
        r0, c0 = min(r for r, _ in source), min(c for _, c in source)
        for (r, c), color in pixels.items():
            grid[r+r0, c+c0] = color
    output = grid.copy()
    for p in sources[0]:
        output[p] = bg
    for (r, c), color in painted[1].items():
        output[r+1, c+1] = color
    foreground = len(body)+sum(map(len, specs))
    metadata = dict(n_hosts=1, n_holes=1, n_loose_pieces=len(specs), n_moved_pieces=1,
        n_multi_piece_holes=0, n_strays=plan['strays'], host_families=['chamfer' if chamfer else 'rect'],
        hole_families=['bar' if min(H-2, width) == 1 else 'rect'], host_cells=[len(body)],
        hole_areas=[len(hole)], piece_areas=list(map(len, specs)), background='solid', background_kind='uniform',
        texture_colors=[], background_color=bg, foreground_fraction=foreground/(H*W), connectivity=4,
        no_local_cover_truncation=True, multicolor_objects=sum(len(set(p.values())) > 1 for p in painted),
        semantic_colors=plan['palette_size']-1, requested_palette_size=plan['palette_size'], pieces_per_hole=[1])
    return grid, output, metadata


def generate_fit_holes(rng: np.random.Generator) -> dict:
    """Draw dimensions once and prove the final visible colored answer unique."""
    h, w = FEASIBLE_SIZES[int(rng.integers(len(FEASIBLE_SIZES)))]
    H, W = min(h, w), max(h, w)
    plan = _draw_plan(rng, H, W)
    narrow, small = plan["narrow"], plan["small"]
    failures = Counter()
    for attempt in range(1, 161):
        built = _narrow_scene(rng, H, W, plan) if narrow else _small_scene(rng, H, W, plan) if small else _build(rng, H, W, plan)
        if built is None:
            failures['geometry'] += 1
            continue
        grid, output, metadata = built
        if h > w:
            grid, output = grid.T.copy(), output.T.copy()
        if rng.random() < .5:
            grid, output = grid[::-1, :].copy(), output[::-1, :].copy()
        if rng.random() < .5:
            grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
        try:
            parsed = parse_scene(grid)
            stats = validate_fit_holes(grid, output)
        except ValueError as exc:
            failures[str(exc)] += 1
            continue
        metadata.setdefault('n_multicolor_filled_holes', 0)
        metadata.update(attempts=attempt, size_attempts=1, height=h, width=w,
                        requested_hosts=plan["hosts"], requested_holes=plan["holes"],
                        requested_background="sparse" if plan["texture_colors"] else "solid",
                        colored_fit_budget=plan["colored_fits"], broad_cavity_budget=plan["broad_cavities"],
                        requested_pieces=sum(plan["pieces_per_hole"])+plan["strays"],
                        capacity_holes=plan["capacity_holes"], capacity_hosts=plan["capacity_hosts"],
                        n_filled_holes=stats['fillable_holes'], n_unfilled_holes=stats['unfilled_holes'],
                        n_colors=int(len(np.unique(grid))), background_margin=parsed.background_margin,
                        host_colors=[int(grid[next(iter(o.cells))]) for o in parsed.hosts],
                        piece_colors=[int(grid[next(iter(o.cells))]) for o in parsed.pieces],
                        hosts_touching_border=sum(any(r in (0, h-1) or c in (0, w-1) for r, c in o.cells) for o in parsed.hosts),
                        verification=stats, failures=dict(failures))
        return {'input': grid.tolist(), 'output': output.tolist(), 'metadata': metadata}
    raise RuntimeError(f"task28: cannot build the fixed {h}x{w} plan after 160 attempts: {dict(failures)}; {plan}")
