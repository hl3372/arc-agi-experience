"""Rectangle family generators: tasks 49, 50, 51, 52 (including former 53), 67."""
from __future__ import annotations

import numpy as np

from common.grid import (
    NUM_COLORS,
    grid_to_list,
    make_grid,
    pick_colors,
    random_background,
    random_grid_size,
)
from common.objects import stamp_rectangle
from registry import register

MAX_RETRIES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_colors(rng: np.random.Generator, n: int, bg: int) -> list[int]:
    """Pick *n* distinct non-background colors."""
    return pick_colors(rng, n, exclude=[bg])


def _rects_overlap(r1: tuple[int, int, int, int],
                   r2: tuple[int, int, int, int],
                   spacing: int = 1) -> bool:
    """Check whether two (row, col, h, w) rects overlap (with spacing margin)."""
    r1r, r1c, r1h, r1w = r1
    r2r, r2c, r2h, r2w = r2
    # Expand r1 by spacing on each side and check intersection
    if (r1r + r1h + spacing <= r2r or r2r + r2h + spacing <= r1r):
        return False
    if (r1c + r1w + spacing <= r2c or r2c + r2w + spacing <= r1c):
        return False
    return True


# ---------------------------------------------------------------------------
# Task 49 — Hollow filled rectangles
# ---------------------------------------------------------------------------

@register("49")
def generate_hollow_filled_rects(rng: np.random.Generator) -> dict:
    from generators.rectangle_edits_49_50 import generate
    return generate(rng, fill=False)


@register("50")
def generate_fill_hollow_rects(rng: np.random.Generator) -> dict:
    from generators.rectangle_edits_49_50 import generate
    return generate(rng, fill=True)


# ---------------------------------------------------------------------------
# Task 51 — Border thickness of the thickest rectangle
# ---------------------------------------------------------------------------

@register("51")
def generate_border_thickness(rng: np.random.Generator) -> dict:
    from generators.frame_measurement_51_52 import generate
    return generate(rng, thickness=True)


@register("52")
def generate_cells_extreme_rect(rng: np.random.Generator) -> dict:
    """Largest/smallest frame, all interior cells or a requested color."""
    from generators.frame_measurement_51_52 import generate
    return generate(rng, thickness=False)


# ---------------------------------------------------------------------------
# Task 67 — Swap occlusion order of overlapping rectangle groups
# ---------------------------------------------------------------------------


@register("67")
def generate_swap_occlusion(rng: np.random.Generator) -> dict:
    from generators.rectangle_layers_67 import generate
    return generate(rng)
