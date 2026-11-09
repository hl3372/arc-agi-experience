"""Central registry mapping task IDs to generator functions."""
from __future__ import annotations

from typing import Callable

import numpy as np

GeneratorFn = Callable[[np.random.Generator], dict]

_REGISTRY: dict[str, GeneratorFn] = {}

# Retired task ids and the active tasks that cover their operations.
# ``generate`` raises for them instead of silently aliasing, so stale references fail loudly.
RETIRED: dict[str, str] = {"19": "18", "21": "20", "43": "42", "46b": "46a", "48": "47", "53": "52", "69a": "68a", "69b": "68b", "92b": "92", "100": "99"}
RETIRED_ON = "2026-09-07"


def register(task_id: str):
    """Decorator to register a generator function for a task ID."""

    def decorator(fn: GeneratorFn) -> GeneratorFn:
        if task_id in _REGISTRY:
            raise ValueError(f"Duplicate task_id: {task_id}")
        _REGISTRY[task_id] = fn
        return fn

    return decorator


def list_tasks() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_generator(task_id: str) -> GeneratorFn:
    """Return the registered generator function for a task ID."""
    if task_id in RETIRED:
        if task_id == "100":
            raise KeyError("task 100 was merged into task 99 on 2026-09-15; "
                           "generate('99', seed) draws the symmetry (four mirror axes "
                           "or a rotation) per example")
        if task_id == "92b":
            raise KeyError("task 92b was merged into task 92 on 2026-09-15; "
                           "generate('92', seed) draws 4/8 connectivity per example")
        if task_id in ("69a", "69b"):
            raise KeyError(f"task {task_id} was merged into task {RETIRED[task_id]} on 2026-09-13; "
                           f"generate('{RETIRED[task_id]}', seed) draws 4/8 connectivity per example")
        if task_id == "53":
            raise KeyError("task 53 was merged into task 52 on 2026-09-11; "
                           "generate('52', seed) draws largest/smallest and total/color counting per example")
        if task_id in ("46b", "48"):
            raise KeyError(f"task {task_id} was merged into task {RETIRED[task_id]} on 2026-09-11; "
                           f"generate('{RETIRED[task_id]}', seed) draws the connectivity per example")
        if task_id == "43":
            raise KeyError("task 43 was merged into task 42 on 2026-09-11; "
                           "generate('42', seed) draws the shared/different color encoding per example")
        raise KeyError(f"task {task_id} was merged into task {RETIRED[task_id]} on {RETIRED_ON}; "
                       f"generate('{RETIRED[task_id]}', seed) draws the connectivity per example")
    if task_id not in _REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id}")
    return _REGISTRY[task_id]


def generate(task_id: str, seed: int) -> dict:
    """Generate a single example for the given task.

    Returns dict with keys: input, output, task_id, and optionally metadata.
    """
    rng = np.random.default_rng(seed)
    result = get_generator(task_id)(rng)
    assert "input" in result and "output" in result
    result["task_id"] = task_id
    return result


# Import generator modules to trigger @register decorators
import generators.metadata  # noqa: F401
import generators.global_transforms  # noqa: F401
import generators.object_counting  # noqa: F401
import generators.background_restore  # noqa: F401
import generators.color_filtering  # noqa: F401
import generators.rectangles  # noqa: F401
import generators.object_measurement  # noqa: F401
import generators.topology_tasks  # noqa: F401
import generators.count_holes  # noqa: F401
import generators.local_transforms  # noqa: F401
import generators.lines_diamonds  # noqa: F401
import generators.paths  # noqa: F401
import generators.binary_neighborhood  # noqa: F401
import generators.pattern_symmetry  # noqa: F401
import generators.construction_tiling  # noqa: F401
import generators.lattice  # noqa: F401
import generators.misc  # noqa: F401
import generators.border_objects_90  # noqa: F401
import generators.pack_edge_91  # noqa: F401
import generators.touching_92  # noqa: F401
import generators.join_pieces  # noqa: F401
import generators.target_objects_93_94  # noqa: F401
import generators.bbox_outline_95  # noqa: F401
import generators.frame_extract_96  # noqa: F401
import generators.odd_shape_97  # noqa: F401
import generators.sorted_strip_98  # noqa: F401
import generators.symmetry_fill_99  # noqa: F401
import generators.symmetry_violations_101  # noqa: F401
import generators.symmetry_report_102  # noqa: F401
import generators.grid_slices_77_79_86  # noqa: F401
import generators.core2_simple  # noqa: F401
import generators.core2_extra  # noqa: F401


if __name__ == "__main__":
    for tid in list_tasks():
        for seed in range(3):
            ex = generate(tid, seed)
            inp = ex["input"]
            if isinstance(inp, dict):
                shape_desc = ", ".join(
                    f"{name}={len(grid)}x{len(grid[0])}"
                    for name, grid in inp.items()
                    if isinstance(grid, list) and grid
                )
            else:
                shape_desc = f"grid {len(inp)}x{len(inp[0])}"
            print(
                f"Task {tid:>3s} seed={seed}: "
                f"{shape_desc}, "
                f"output={str(ex['output'])[:80]}"
            )
            if "metadata" in ex:
                print(f"         metadata={ex['metadata']}")
