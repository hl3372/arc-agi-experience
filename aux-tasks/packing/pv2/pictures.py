"""Pictures: input grids borrowed from the reviewed generators, without their questions.

Library questions and ARC bundles need grids to ask about.  Borrowing the generators' inputs keeps
them exactly as varied as the tasks (sizes 1-30, ten colours, drawn shapes, textured grounds).
Only single-grid inputs are used (14 is text, 41 and 78 are two-part inputs).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from pv2.paths import ensure_generator_package

ensure_generator_package()
from registry import generate, list_tasks  # noqa: E402

EXCLUDED = {"14", "41", "78"}


def try_generate(tid: str, seed: int):
    """``registry.generate`` for a freshly DRAWN seed, or None when the generator raises.

    The reviewed generators raise instead of falling back when a plan cannot be built (a rare seed;
    task 28 raised on 0 of 1500 random seeds, but v2 draws many).  Call sites that draw their own
    seed skip such a seed and draw another; re-rendering a recorded seed must still raise, so those
    call ``generate`` directly.
    """
    try:
        return generate(tid, seed)
    except Exception:  # noqa: BLE001
        return None


def _number(task_id: str) -> int:
    return int(re.match(r"\d+", task_id).group(0))


def picture_tasks() -> list[str]:
    """Task ids 1-102 whose input is one grid."""
    return [t for t in list_tasks() if _number(t) <= 102 and t not in EXCLUDED]


def sample_picture(rng: np.random.Generator, *, max_cells: int = 900, min_side: int = 1,
                   solid: bool = False, tries: int = 50) -> tuple[list[list[int]], str, int]:
    """(grid, task_id, seed) of a borrowed input grid.

    ``solid``: the most frequent colour covers at least half the grid and no cell of another colour
    is isolated (every non-majority cell has a same-colour 4-neighbour) -- a plain ground, for
    transforms whose result depends on what the background is.
    """
    tasks = picture_tasks()
    for _ in range(tries):
        tid = tasks[int(rng.integers(len(tasks)))]
        seed = int(rng.integers(0, 2 ** 31))
        ex = try_generate(tid, seed)
        if ex is None:
            continue
        grid = ex["input"]
        h, w = len(grid), len(grid[0])
        if h * w > max_cells or min(h, w) < min_side:
            continue
        if solid and not is_solid(grid):
            continue
        return grid, tid, seed
    raise RuntimeError("no picture satisfied the constraints")


def is_solid(grid) -> bool:
    arr = np.asarray(grid)
    counts = Counter(arr.ravel().tolist())
    ground, n = counts.most_common(1)[0]
    if n < 0.5 * arr.size:
        return False
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            v = arr[r, c]
            if v == ground:
                continue
            if not any(0 <= r + dr < h and 0 <= c + dc < w and arr[r + dr, c + dc] == v
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                return False
    return True
