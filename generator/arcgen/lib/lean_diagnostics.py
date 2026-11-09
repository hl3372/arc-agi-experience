"""Objective format checks and non-gating authorship diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Any, Iterable, Sequence
import json

from .validation import validate_arc_task


PAIR_ECONOMY_FLAG_CODES = {
    "over_90_percent_identical_inputs",
    "dihedral_equivalent_inputs",
    "repeated_component_or_subgrid",
}


def _grid_key(grid: Sequence[Sequence[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def _dimensions(grid: Sequence[Sequence[int]]) -> list[int]:
    return [len(grid), len(grid[0])]


def _background(grid: Sequence[Sequence[int]]) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return min(counts, key=lambda color: (-counts[color], color))


def _occupied_ratio(grid: Sequence[Sequence[int]]) -> float:
    bg = _background(grid)
    total = len(grid) * len(grid[0])
    occupied = sum(cell != bg for row in grid for cell in row)
    return round(occupied / total, 6)


def _margins(grid: Sequence[Sequence[int]]) -> list[int]:
    bg = _background(grid)
    height, width = len(grid), len(grid[0])
    occupied = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != bg
    ]
    if not occupied:
        return [height, height, width, width]
    rows = [item[0] for item in occupied]
    columns = [item[1] for item in occupied]
    return [min(rows), height - 1 - max(rows), min(columns), width - 1 - max(columns)]


def _rot90(grid: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*grid[::-1])]


def _dihedral_keys(grid: Sequence[Sequence[int]]) -> set[str]:
    current = [list(row) for row in grid]
    variants: list[list[list[int]]] = []
    for _ in range(4):
        variants.append(current)
        variants.append([list(reversed(row)) for row in current])
        current = _rot90(current)
    return {_grid_key(item) for item in variants}


def _cell_similarity(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> float | None:
    if _dimensions(left) != _dimensions(right):
        return None
    total = len(left) * len(left[0])
    equal = sum(
        left[row][column] == right[row][column]
        for row in range(len(left))
        for column in range(len(left[0]))
    )
    return equal / total


def _components(grid: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """Approximate monochrome, four-connected non-background components."""

    bg = _background(grid)
    height, width = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    result: list[dict[str, Any]] = []
    for row in range(height):
        for column in range(width):
            color = grid[row][column]
            if color == bg or (row, column) in visited:
                continue
            queue = deque([(row, column)])
            visited.add((row, column))
            cells: list[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                cells.append(current)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = current[0] + dr, current[1] + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and (nr, nc) not in visited
                        and grid[nr][nc] == color
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            min_r = min(cell[0] for cell in cells)
            max_r = max(cell[0] for cell in cells)
            min_c = min(cell[1] for cell in cells)
            max_c = max(cell[1] for cell in cells)
            subgrid = [
                [grid[r][c] for c in range(min_c, max_c + 1)]
                for r in range(min_r, max_r + 1)
            ]
            result.append(
                {
                    "color": color,
                    "cell_count": len(cells),
                    "bounds": [min_r, min_c, max_r, max_c],
                    "signature": _grid_key(subgrid),
                }
            )
    return result


def _all_pairs(task: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for split in ("train", "test"):
        for index, pair in enumerate(task[split], 1):
            yield f"{split}_{index}", pair


def diagnose_arc_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return hard validation plus transparent semantic-artifact flags.

    Only errors returned by :func:`validate_arc_task` are hard failures.
    Everything under ``flags`` is evidence for the technical verifier.
    """

    validation = validate_arc_task(task, require_test_outputs=True)
    flags: list[dict[str, Any]] = []
    train = task.get("train", []) if isinstance(task, dict) else []
    test = task.get("test", []) if isinstance(task, dict) else []
    hard_issues = [
        issue for issue in validation.issues if issue.severity == "error" and issue.code != "duplicate_pair"
    ]
    duplicate_pair_issues = [
        issue for issue in validation.issues if issue.code == "duplicate_pair"
    ]
    if hard_issues:
        return {
            "hard_valid": False,
            "hard_errors": [issue.render() for issue in hard_issues],
            "format_warnings": list(validation.warnings),
            "flags": [],
            "metrics": {},
        }
    for issue in duplicate_pair_issues:
        flags.append(
            {
                "code": "exact_duplicate_pair",
                "locations": [issue.path],
                "detail": issue.message,
            }
        )

    train_inputs = [pair["input"] for pair in train]
    all_inputs = [
        *((f"train_{index + 1}", pair["input"]) for index, pair in enumerate(train)),
        *((f"test_{index + 1}", pair["input"]) for index, pair in enumerate(test)),
    ]
    seen_train_inputs: dict[str, int] = {}
    for index, grid in enumerate(train_inputs, 1):
        key = _grid_key(grid)
        if key in seen_train_inputs:
            flags.append(
                {
                    "code": "exact_duplicate_training_input",
                    "locations": [f"train_{seen_train_inputs[key]}", f"train_{index}"],
                    "detail": "training inputs are exactly identical",
                }
            )
        else:
            seen_train_inputs[key] = index

    pairwise: list[dict[str, Any]] = []
    for left in range(len(train_inputs)):
        for right in range(left + 1, len(train_inputs)):
            score = _cell_similarity(train_inputs[left], train_inputs[right])
            pairwise.append(
                {
                    "left": f"train_{left + 1}",
                    "right": f"train_{right + 1}",
                    "same_dimensions": score is not None,
                    "cell_identity_fraction": None if score is None else round(score, 6),
                }
            )
            if score is not None and 0.9 < score < 1.0:
                flags.append(
                    {
                        "code": "over_90_percent_identical_inputs",
                        "locations": [f"train_{left + 1}", f"train_{right + 1}"],
                        "detail": f"inputs share {score:.2%} of cells",
                    }
                )
            if (
                _grid_key(train_inputs[right]) != _grid_key(train_inputs[left])
                and _grid_key(train_inputs[right]) in _dihedral_keys(train_inputs[left])
            ):
                flags.append(
                    {
                        "code": "dihedral_equivalent_inputs",
                        "locations": [f"train_{left + 1}", f"train_{right + 1}"],
                        "detail": "one whole input is a rotation or reflection of the other",
                    }
                )

    input_locations: defaultdict[str, list[str]] = defaultdict(list)
    for label, grid in all_inputs:
        input_locations[_grid_key(grid)].append(label)
    for locations in input_locations.values():
        if any(item.startswith("test_") for item in locations) and len(locations) > 1:
            flags.append(
                {
                    "code": "repeated_train_test_grid",
                    "locations": locations,
                    "detail": "a test input repeats an earlier whole input verbatim",
                }
            )

    component_occurrences: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    component_counts: dict[str, int] = {}
    for label, grid in all_inputs:
        components = _components(grid)
        component_counts[label] = len(components)
        for component in components:
            if component["cell_count"] >= 2:
                component_occurrences[component["signature"]].append(
                    {
                        "pair": label,
                        "color": component["color"],
                        "cell_count": component["cell_count"],
                        "bounds": component["bounds"],
                    }
                )
    repeated = [
        occurrences
        for occurrences in component_occurrences.values()
        if len({item["pair"] for item in occurrences}) >= 2
    ]
    if repeated:
        flags.append(
            {
                "code": "repeated_component_or_subgrid",
                "locations": sorted(
                    {item["pair"] for group in repeated for item in group}
                ),
                "detail": (
                    f"{len(repeated)} exact non-background component bounding "
                    "subgrid signature(s) recur across episode inputs"
                ),
                "examples": repeated[:10],
            }
        )

    margins = {label: _margins(grid) for label, grid in all_inputs}
    if len(margins) > 1 and len({tuple(value) for value in margins.values()}) == 1:
        common = next(iter(margins.values()))
        if any(common):
            flags.append(
                {
                    "code": "uniform_canvas_margins",
                    "locations": list(margins),
                    "detail": f"all episode inputs use the same margins {common}",
                }
            )

    color_counts: dict[str, dict[str, int]] = {}
    occupied: dict[str, float] = {}
    input_sizes: dict[str, list[int]] = {}
    output_sizes: dict[str, list[int]] = {}
    for label, pair in _all_pairs(task):
        grid = pair["input"]
        input_sizes[label] = _dimensions(grid)
        occupied[label] = _occupied_ratio(grid)
        color_counts[label] = {
            str(color): count
            for color, count in sorted(
                Counter(cell for row in grid for cell in row).items()
            )
        }
        if "output" in pair:
            output_sizes[label] = _dimensions(pair["output"])

    return {
        "hard_valid": True,
        "hard_errors": [],
        "format_warnings": list(validation.warnings),
        "flags": flags,
        "metrics": {
            "pairwise_training_input_similarity": pairwise,
            "uniform_margins_by_input": margins,
            "color_counts_by_input": color_counts,
            "occupied_area_ratio_by_input": occupied,
            "input_sizes": input_sizes,
            "input_size_variety": len({tuple(value) for value in input_sizes.values()}),
            "output_sizes": output_sizes,
            "output_size_variety": len({tuple(value) for value in output_sizes.values()}),
            "approx_connected_component_counts": component_counts,
        },
    }


def combine_public_and_hidden(
    task: dict[str, Any], expected_test_outputs: Sequence[Sequence[Sequence[int]]]
) -> dict[str, Any]:
    complete = deepcopy(task)
    if len(complete.get("test", [])) != len(expected_test_outputs):
        raise ValueError("hidden-output count does not match test-input count")
    for pair, output in zip(complete["test"], expected_test_outputs, strict=True):
        if "output" in pair:
            raise ValueError("public test pair already contains an output")
        pair["output"] = deepcopy(output)
    return complete


__all__ = [
    "PAIR_ECONOMY_FLAG_CODES",
    "combine_public_and_hidden",
    "diagnose_arc_task",
]
