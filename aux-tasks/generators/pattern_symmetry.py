"""Pattern & symmetry family generators: tasks 32, 33, 34, 38, 64, 65."""
from __future__ import annotations

import numpy as np

from common.grid import (
    NUM_COLORS,
    checkered_background,
    grid_to_list,
    make_grid,
    noisy_background,
    pick_color,
    pick_colors,
    random_grid_size,
    striped_background,
)
from common.objects import (
    random_shape,
    normalize_shape,
    bounding_box,
    stamp_rectangle,
)
from registry import register

MAX_RETRIES = 20

SQUARE_ONLY_SYMMETRY_FAMILIES = {
    "main_diagonal_mirror",
    "anti_diagonal_mirror",
    "full_square_symmetry",
}

PATTERN_FAMILY_CHOICES = (
    "periodic",
    "horizontal_mirror",
    "vertical_mirror",
    "both_axis_mirror",
    "rotational_180",
    "main_diagonal_mirror",
    "anti_diagonal_mirror",
    "full_square_symmetry",
)

# Keep full-square symmetry as a rare stress variant; the remaining mass matches
# the old 3:1:1:1:1:1:1 periodic/symmetry balance.
PATTERN_FAMILY_WEIGHTS = (
    0.3233333333333333,
    0.10777777777777778,
    0.10777777777777778,
    0.10777777777777778,
    0.10777777777777778,
    0.10777777777777778,
    0.10777777777777778,
    0.03,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_tile(
    rng: np.random.Generator, h: int, w: int, palette: list[int]
) -> np.ndarray:
    """Create a random tile of size h x w using colors from *palette*.

    Guarantees at least 2 distinct colors so the tile is not trivially constant.
    """
    for _ in range(MAX_RETRIES):
        tile = np.array(
            [[int(rng.choice(palette)) for _ in range(w)] for _ in range(h)],
            dtype=np.uint8,
        )
        if len(np.unique(tile)) >= 2:
            return tile
    # Force a second colour if stuck
    tile[0, 0] = palette[0]
    if tile[0, 0] == palette[-1]:
        tile[-1, -1] = palette[1] if len(palette) > 1 else (palette[0] + 1) % NUM_COLORS
    else:
        tile[-1, -1] = palette[-1]
    return tile


def _generate_structured_tile(
    rng: np.random.Generator,
    h: int,
    w: int,
    palette: list[int],
) -> np.ndarray:
    """Create a tile with a bit more internal structure than pure iid noise."""
    tile = np.full((h, w), int(rng.choice(palette)), dtype=np.uint8)
    mode = str(rng.choice(["checker", "stripes", "diag", "cross", "blocks", "random"]))
    if mode == "random":
        return _generate_tile(rng, h, w, palette)
    if mode == "checker":
        for r in range(h):
            for c in range(w):
                tile[r, c] = palette[(r + c) % len(palette)]
    elif mode == "stripes":
        vertical = bool(rng.integers(0, 2))
        for r in range(h):
            for c in range(w):
                idx = c if vertical else r
                tile[r, c] = palette[idx % len(palette)]
    elif mode == "diag":
        offset = int(rng.integers(0, max(1, w)))
        for r in range(h):
            for c in range(w):
                tile[r, c] = palette[(r + c + offset) % len(palette)]
    elif mode == "blocks":
        split_r = int(rng.integers(1, h + 1))
        split_c = int(rng.integers(1, w + 1))
        tile[:split_r, :split_c] = palette[0]
        tile[:split_r, split_c:] = palette[min(1, len(palette) - 1)]
        tile[split_r:, :split_c] = palette[min(2, len(palette) - 1)]
        tile[split_r:, split_c:] = palette[min(3, len(palette) - 1)]
    else:
        mid_r = h // 2
        mid_c = w // 2
        for r in range(h):
            tile[r, mid_c] = palette[r % len(palette)]
        for c in range(w):
            tile[mid_r, c] = palette[c % len(palette)]
    if len(np.unique(tile)) >= 2:
        return tile
    return _generate_tile(rng, h, w, palette)


def _generate_clear_background_tile(
    rng: np.random.Generator,
    h: int,
    w: int,
    palette: list[int],
) -> np.ndarray:
    """Create a simple repeating tile that stays legible after object removal."""
    mode = str(rng.choice(["checker", "stripes", "diag", "blocks"]))
    tile = np.full((h, w), palette[0], dtype=np.uint8)
    if mode == "checker":
        block = int(rng.integers(1, 3))
        for r in range(h):
            for c in range(w):
                tile[r, c] = palette[((r // block) + (c // block)) % len(palette)]
    elif mode == "stripes":
        vertical = bool(rng.integers(0, 2))
        width = int(rng.integers(1, 3))
        for r in range(h):
            for c in range(w):
                idx = c if vertical else r
                tile[r, c] = palette[(idx // width) % len(palette)]
    elif mode == "diag":
        for r in range(h):
            for c in range(w):
                tile[r, c] = palette[(r + c) % len(palette)]
    else:
        split_r = int(rng.integers(1, h + 1))
        split_c = int(rng.integers(1, w + 1))
        tile[:split_r, :split_c] = palette[0]
        tile[:split_r, split_c:] = palette[min(1, len(palette) - 1)]
        tile[split_r:, :split_c] = palette[min(2, len(palette) - 1)]
        tile[split_r:, split_c:] = palette[min(3, len(palette) - 1)]
    if len(np.unique(tile)) >= 2:
        return tile
    return _generate_tile(rng, h, w, palette)


def _tile_grid(tile: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Repeat *tile* to fill a grid of size (grid_h, grid_w) using np.tile."""
    th, tw = tile.shape
    reps_h = (grid_h + th - 1) // th
    reps_w = (grid_w + tw - 1) // tw
    big = np.tile(tile, (reps_h, reps_w))
    return big[:grid_h, :grid_w].copy()


def _minimize_tile(tile: np.ndarray) -> np.ndarray:
    """Return the smallest sub-tile of *tile* that reproduces it by tiling.

    Tries every (th, tw) where th divides tile height and tw divides tile
    width, in area-ascending order, and returns the first that works.
    """
    h, w = tile.shape
    for th in range(1, h + 1):
        if h % th != 0:
            continue
        for tw in range(1, w + 1):
            if w % tw != 0:
                continue
            sub = tile[:th, :tw]
            # Check if sub tiles the full tile
            if np.array_equal(_tile_grid(sub, h, w), tile):
                return sub
    return tile  # should never happen


def _random_palette(rng: np.random.Generator, n: int, exclude: set[int] | None = None) -> list[int]:
    """Pick *n* distinct colors from 0..9, excluding any in *exclude*."""
    if exclude is None:
        exclude = set()
    available = [c for c in range(NUM_COLORS) if c not in exclude]
    chosen = rng.choice(available, size=min(n, len(available)), replace=False)
    return [int(c) for c in chosen]


def _unique_full_line_count(grid: np.ndarray, bg_color: int) -> int:
    """Count full rows/columns that are a single non-background color.

    Used to keep the divider in tasks 34/38 the *only* uniform full line so a
    solver can still locate it after background texture is added.
    """
    h, w = grid.shape
    count = 0
    for r in range(h):
        if grid[r, 0] != bg_color and len(set(grid[r, :].tolist())) == 1:
            count += 1
    for c in range(w):
        if grid[0, c] != bg_color and len(set(grid[:, c].tolist())) == 1:
            count += 1
    return count


def _add_symmetric_bg_texture(
    rng: np.random.Generator,
    grid: np.ndarray,
    output: np.ndarray,
    orientation: str,
    div_pos: int,
    bg_color: int,
    forbidden: set[int],
    *,
    density: float = 0.3,
) -> bool:
    """Stamp a patterned background that is mirror-symmetric across the divider.

    The texture is drawn from a palette DISJOINT from every semantic colour
    (background, divider, object palette) and is placed identically on both
    sides of the divider. Because it is already mirror-symmetric, the task-34
    rule ("mirror every coloured shape") leaves it unchanged, so the unique
    answer is preserved; for task 38 the symmetric texture keeps the two halves
    equal and therefore does not change the true/false verdict.

    Only cells that are background in BOTH *grid* and *output* are textured, so
    objects, the divider and every mirror target stay untouched. The same cells
    are written to *grid* and *output* so the texture is identical on input and
    answer. Returns True if any texture was added (both grids mutated in place).
    """
    h, w = grid.shape
    avail = [c for c in range(NUM_COLORS) if c not in forbidden]
    if not avail:
        return False
    rng.shuffle(avail)
    tex_palette = avail[: int(rng.integers(1, min(3, len(avail)) + 1))]

    # Build a patterned source via a real bg-variety helper, then re-colour its
    # non-bg cells into the disjoint texture palette.
    style = str(rng.choice(["checkered", "striped", "noisy"]))
    if style == "checkered":
        source, _ = checkered_background(
            rng, h, w, bg_color=bg_color, cell_size=int(rng.integers(1, 3))
        )
    elif style == "striped":
        source, _ = striped_background(rng, h, w, bg_color=bg_color)
    else:
        source, _ = noisy_background(
            rng, h, w, bg_color=bg_color, noise_rate=float(rng.uniform(0.12, 0.3))
        )

    if orientation == "horizontal":
        first_side = [(r, c) for r in range(div_pos) for c in range(w)]

        def mirror(r: int, c: int) -> tuple[int, int]:
            return (2 * div_pos - r, c)
    else:
        first_side = [(r, c) for r in range(h) for c in range(div_pos)]

        def mirror(r: int, c: int) -> tuple[int, int]:
            return (r, 2 * div_pos - c)

    # Keep the background colour strictly dominant so a solver can still detect
    # it; never texture more than ~30% of the available background cells.
    bg_cells = int(np.count_nonzero(grid == bg_color))
    budget = max(1, int(bg_cells * 0.3))
    added = 0
    for r, c in first_side:
        if added >= budget:
            break
        if source[r, c] == bg_color:
            continue
        if rng.random() > density:
            continue
        mr, mc = mirror(r, c)
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        if mr == r and mc == c:  # cells on the divider itself
            continue
        # Both endpoints must be background in input AND output: protect every
        # object, the divider and every mirror target.
        if grid[r, c] != bg_color or grid[mr, mc] != bg_color:
            continue
        if output[r, c] != bg_color or output[mr, mc] != bg_color:
            continue
        color = int(rng.choice(tex_palette))
        for rr, cc in {(r, c), (mr, mc)}:
            grid[rr, cc] = color
            output[rr, cc] = color
            added += 1
    return added > 0


def _symmetry_orbit(
    family: str,
    r: int,
    c: int,
    h: int,
    w: int,
) -> tuple[tuple[int, int], ...]:
    """Return the cell-equivalence orbit for one explicit symmetry family."""
    if family == "horizontal_mirror":
        orbit = {(r, c), (h - 1 - r, c)}
    elif family == "vertical_mirror":
        orbit = {(r, c), (r, w - 1 - c)}
    elif family == "rotational_180":
        orbit = {(r, c), (h - 1 - r, w - 1 - c)}
    elif family == "both_axis_mirror":
        orbit = {
            (r, c),
            (h - 1 - r, c),
            (r, w - 1 - c),
            (h - 1 - r, w - 1 - c),
        }
    elif family == "main_diagonal_mirror":
        if h != w:
            raise ValueError(f"{family} requires a square grid")
        orbit = {(r, c), (c, r)}
    elif family == "anti_diagonal_mirror":
        if h != w:
            raise ValueError(f"{family} requires a square grid")
        orbit = {(r, c), (w - 1 - c, h - 1 - r)}
    elif family == "full_square_symmetry":
        if h != w:
            raise ValueError(f"{family} requires a square grid")
        n = h
        orbit = {
            (r, c),
            (c, n - 1 - r),
            (n - 1 - r, n - 1 - c),
            (n - 1 - c, r),
            (n - 1 - r, c),
            (r, n - 1 - c),
            (c, r),
            (n - 1 - c, n - 1 - r),
        }
    else:
        raise ValueError(f"unknown symmetry family: {family}")
    return tuple(sorted(orbit))


def _grid_satisfies_family(grid: np.ndarray, family: str) -> bool:
    h, w = grid.shape
    if family == "horizontal_mirror":
        return bool(np.array_equal(grid, np.flipud(grid)))
    if family == "vertical_mirror":
        return bool(np.array_equal(grid, np.fliplr(grid)))
    if family == "both_axis_mirror":
        return bool(np.array_equal(grid, np.flipud(grid)) and np.array_equal(grid, np.fliplr(grid)))
    if family == "rotational_180":
        return bool(np.array_equal(grid, np.rot90(grid, 2)))
    if family == "main_diagonal_mirror":
        return h == w and bool(np.array_equal(grid, grid.T))
    if family == "anti_diagonal_mirror":
        return h == w and bool(np.array_equal(grid, np.flipud(np.fliplr(grid)).T))
    if family == "full_square_symmetry":
        if h != w:
            return False
        return bool(
            np.array_equal(grid, np.rot90(grid, 1))
            and np.array_equal(grid, np.rot90(grid, 2))
            and np.array_equal(grid, np.rot90(grid, 3))
            and np.array_equal(grid, np.flipud(grid))
            and np.array_equal(grid, np.fliplr(grid))
            and np.array_equal(grid, grid.T)
            and np.array_equal(grid, np.flipud(np.fliplr(grid)).T)
        )
    if family == "periodic":
        return True
    raise ValueError(f"unknown pattern family: {family}")


def _valid_periodic_mask(
    masked_grid: np.ndarray,
    mask_color: int,
    tile_h: int,
    tile_w: int,
) -> bool:
    """Check that every tile position has at least one visible representative."""
    grid_h, grid_w = masked_grid.shape
    for tr in range(tile_h):
        for tc in range(tile_w):
            visible = False
            for rr in range(tr, grid_h, tile_h):
                for cc in range(tc, grid_w, tile_w):
                    if masked_grid[rr, cc] != mask_color:
                        visible = True
                        break
                if visible:
                    break
            if not visible:
                return False
    return True


def _periodic_equivalents(
    r: int,
    c: int,
    h: int,
    w: int,
    tile_h: int,
    tile_w: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (rr, cc)
        for rr in range(r % tile_h, h, tile_h)
        for cc in range(c % tile_w, w, tile_w)
    )


def _equivalence_cells(
    meta: dict[str, object],
    r: int,
    c: int,
    h: int,
    w: int,
) -> tuple[tuple[int, int], ...]:
    family = str(meta["pattern_family"])
    if family == "periodic":
        return _periodic_equivalents(r, c, h, w, int(meta["tile_h"]), int(meta["tile_w"]))
    return _symmetry_orbit(family, r, c, h, w)


def _valid_mask_for_meta(
    masked_grid: np.ndarray,
    mask_color: int,
    meta: dict[str, object],
) -> bool:
    """Check that every masked equivalence class keeps at least one visible cell."""
    h, w = masked_grid.shape
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(h):
        for c in range(w):
            orbit = _equivalence_cells(meta, r, c, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)
            if all(masked_grid[rr, cc] == mask_color for rr, cc in orbit):
                return False
    return True


def _occluder_preserves_inference(
    meta: dict[str, object],
    occ_r: int,
    occ_c: int,
    occ_h: int,
    occ_w: int,
    grid_h: int,
    grid_w: int,
) -> bool:
    """Check that an occluder leaves enough repeated/symmetric evidence."""
    hidden = {
        (r, c)
        for r in range(occ_r, occ_r + occ_h)
        for c in range(occ_c, occ_c + occ_w)
    }
    family = str(meta["pattern_family"])
    for r, c in hidden:
        equivalents = _equivalence_cells(meta, r, c, grid_h, grid_w)
        if all(cell in hidden for cell in equivalents):
            return False
    return True


def _sample_reflection_shape(
    rng: np.random.Generator,
    max_size: int,
) -> frozenset[tuple[int, int]]:
    """Sample a somewhat richer object for tasks 34/38."""
    size_lo = 4 if max_size >= 4 else 3
    size_hi = max(size_lo, min(10, max_size))
    shape = random_shape(rng, size_range=(size_lo, size_hi), connectivity=4)
    return shape


def _place_structured_cells_on_one_side(
    rng: np.random.Generator,
    grid: np.ndarray,
    orientation: str,
    div_pos: int,
    palette: list[int],
    bg_color: int,
    *,
    min_objects: int = 2,
    max_objects: int = 4,
) -> list[tuple[int, int, int]]:
    """Place a few regular or irregular objects on one side of the divider."""
    placed: list[tuple[int, int, int]] = []
    h, w = grid.shape
    target_objects = int(rng.integers(min_objects, max_objects + 1))
    for _ in range(target_objects * 5):
        if target_objects <= 0:
            break
        color = int(rng.choice(palette))
        kind = str(rng.choice(["shape", "rect", "line"]))
        if kind == "shape":
            cells = _sample_reflection_shape(rng, 12)
            r0, c0, r1, c1 = bounding_box(cells)
            sh = r1 - r0 + 1
            sw = c1 - c0 + 1
            if orientation == "horizontal":
                if sh > div_pos or sw > w:
                    continue
                base_r = int(rng.integers(0, div_pos - sh + 1))
                base_c = int(rng.integers(0, w - sw + 1))
            else:
                if sh > h or sw > div_pos:
                    continue
                base_r = int(rng.integers(0, h - sh + 1))
                base_c = int(rng.integers(0, div_pos - sw + 1))
            placed_cells = []
            ok = True
            for sr, sc in cells:
                rr = base_r + sr - r0
                cc = base_c + sc - c0
                if grid[rr, cc] != bg_color:
                    ok = False
                    break
                placed_cells.append((rr, cc))
            if not ok:
                continue
            for rr, cc in placed_cells:
                grid[rr, cc] = color
                placed.append((rr, cc, color))
            target_objects -= 1
        elif kind == "rect":
            rh = int(rng.integers(2, 6))
            rw = int(rng.integers(2, 6))
            if orientation == "horizontal":
                if rh > div_pos or rw > w:
                    continue
                r0 = int(rng.integers(0, div_pos - rh + 1))
                c0 = int(rng.integers(0, w - rw + 1))
            else:
                if rh > h or rw > div_pos:
                    continue
                r0 = int(rng.integers(0, h - rh + 1))
                c0 = int(rng.integers(0, div_pos - rw + 1))
            region = grid[r0:r0 + rh, c0:c0 + rw]
            if np.any(region != bg_color):
                continue
            grid[r0:r0 + rh, c0:c0 + rw] = color
            for rr in range(r0, r0 + rh):
                for cc in range(c0, c0 + rw):
                    placed.append((rr, cc, color))
            target_objects -= 1
        else:
            if orientation == "horizontal":
                row = int(rng.integers(0, div_pos))
                c0 = int(rng.integers(0, max(1, w - 4)))
                c1 = int(rng.integers(c0 + 2, w))
                if np.any(grid[row, c0:c1] != bg_color):
                    continue
                grid[row, c0:c1] = color
                for cc in range(c0, c1):
                    placed.append((row, cc, color))
            else:
                col = int(rng.integers(0, div_pos))
                r0 = int(rng.integers(0, max(1, h - 4)))
                r1 = int(rng.integers(r0 + 2, h))
                if np.any(grid[r0:r1, col] != bg_color):
                    continue
                grid[r0:r1, col] = color
                for rr in range(r0, r1):
                    placed.append((rr, col, color))
            target_objects -= 1
    return placed


# ---------------------------------------------------------------------------
# Task 32 — Complete a masked repeating pattern  (grid -> grid)
# ---------------------------------------------------------------------------

@register("32")
def generate_complete_masked_pattern(rng: np.random.Generator) -> dict:
    from generators.pattern_completion_32_33 import generate_pattern as generate_impl
    return generate_impl(rng, "32")


# ---------------------------------------------------------------------------
# Task 33 — Reveal content under one occluding rectangle  (grid -> grid)
# ---------------------------------------------------------------------------

@register("33")
def generate_reveal_occluded(rng: np.random.Generator) -> dict:
    from generators.pattern_completion_32_33 import generate_pattern as generate_impl
    return generate_impl(rng, "33")


# ---------------------------------------------------------------------------
# Task 34 — Mirror object across divider  (grid -> grid)
# ---------------------------------------------------------------------------

@register("34")
def generate_mirror_across_divider(rng: np.random.Generator) -> dict:
    from generators.mirror_divider_34 import generate_mirror_across_divider as generate_impl
    return generate_impl(rng)


# ---------------------------------------------------------------------------
# Task 38 — Is scene mirrored across divider?  (grid -> text)
# ---------------------------------------------------------------------------

@register("38")
def generate_is_mirrored(rng: np.random.Generator) -> dict:
    from generators.mirror_check_38 import generate_is_mirrored as generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 64 — Extract object from patterned background  (grid -> grid)
# ---------------------------------------------------------------------------

@register("64")
def generate_extract_from_pattern(rng: np.random.Generator) -> dict:
    """Task 64: Extract a uniquely identifiable object from a patterned field."""
    from generators.extract_pattern_64 import generate
    return generate(rng)


# ---------------------------------------------------------------------------
# Task 65 — Infer the smallest repeating tile  (grid -> grid)
# ---------------------------------------------------------------------------

@register("65")
def generate_describe_pattern(rng: np.random.Generator) -> dict:
    """Task 65: Infer a minimum tile, allowing partial copies at the edges."""
    from generators.repeating_tile_65 import generate_example
    return generate_example(rng)
