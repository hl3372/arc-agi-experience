"""Lattice family generators: tasks 81, 87, 88, 89.

Task 81 delegates to ``generators.compartments_81``; tasks 87, 88 and 89 delegate to
``generators.lattice_blocks_87_89``, which holds the shared lattice family (independent
row / column spacing, cut-off partial lattices with clipped edge blocks, 1-3 cell thick
lines) and one independent re-solver per task.
"""
from __future__ import annotations

import numpy as np

from registry import register


# ---------------------------------------------------------------------------
# Task 81 — Color largest and smallest compartments (grid -> grid)
# ---------------------------------------------------------------------------


@register("81")
def generate_color_compartments(rng: np.random.Generator) -> dict:
    """Task 81: Recolor extrema in readable four/eight-connected floor plans."""
    from generators.compartments_81 import generate_color_compartments as generate

    return generate(rng)


# ---------------------------------------------------------------------------
# Task 87 — Paint one lattice block by block index (grid -> grid)
# ---------------------------------------------------------------------------


@register("87")
def generate_paint_lattice_block(rng: np.random.Generator) -> dict:
    """Task 87: paint the block at a given 0-based (block row, block column) index."""
    from generators.lattice_blocks_87_89 import generate_paint_block

    return generate_paint_block(rng)


# ---------------------------------------------------------------------------
# Task 88 — Fill blocks between marked blocks (grid -> grid)
# ---------------------------------------------------------------------------


@register("88")
def generate_fill_between_blocks(rng: np.random.Generator) -> dict:
    """Task 88: propagate every marker colour between its aligned marked blocks."""
    from generators.lattice_blocks_87_89 import generate_fill_between

    return generate_fill_between(rng)


# ---------------------------------------------------------------------------
# Task 89 — Count completely filled lattice blocks (grid -> text)
# ---------------------------------------------------------------------------


@register("89")
def generate_count_colored_blocks(rng: np.random.Generator) -> dict:
    """Task 89: count the blocks that are completely filled with the named color."""
    from generators.lattice_blocks_87_89 import generate_count_filled

    return generate_count_filled(rng)
