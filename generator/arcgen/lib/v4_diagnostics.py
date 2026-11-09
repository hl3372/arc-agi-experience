"""V4 deterministic diagnostics and concrete-storyboard fidelity checks.

None of the canvas statistics in this module is a semantic quality gate.  They
exist so a verifier and a human report can distinguish compact economy from an
underused or artificially enlarged canvas.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from typing import Any, Mapping, Sequence

from .lean_v3_diagnostics import diagnose_arc_task_v3
from .validation import validate_arc_task


class V4PlanFidelityError(ValueError):
    pass


def _dimensions(grid: Sequence[Sequence[int]]) -> list[int]:
    return [len(grid), len(grid[0])]


def _conservative_background(grid: Sequence[Sequence[int]]) -> int:
    """Choose the most frequent border color, breaking ties globally then by ID."""

    height, width = len(grid), len(grid[0])
    border = []
    for row in range(height):
        for column in range(width):
            if row in {0, height - 1} or column in {0, width - 1}:
                border.append(grid[row][column])
    border_counts = Counter(border)
    all_counts = Counter(cell for row in grid for cell in row)
    return min(
        border_counts,
        key=lambda color: (-border_counts[color], -all_counts[color], color),
    )


def _active_bbox(grid: Sequence[Sequence[int]], background: int) -> tuple[int, int, int, int] | None:
    occupied = [
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value != background
    ]
    if not occupied:
        return None
    rows = [item[0] for item in occupied]
    columns = [item[1] for item in occupied]
    return min(rows), min(columns), max(rows), max(columns)


def _components(grid: Sequence[Sequence[int]], background: int) -> int:
    height, width = len(grid), len(grid[0])
    remaining = {
        (row, column)
        for row in range(height)
        for column in range(width)
        if grid[row][column] != background
    }
    count = 0
    while remaining:
        count += 1
        queue = deque([remaining.pop()])
        while queue:
            row, column = queue.popleft()
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return count


def grid_canvas_metrics(grid: Sequence[Sequence[int]]) -> dict[str, Any]:
    height, width = len(grid), len(grid[0])
    background = _conservative_background(grid)
    occupied = sum(cell != background for row in grid for cell in row)
    bbox = _active_bbox(grid, background)
    if bbox is None:
        bbox_area = 0
        bbox_value = None
    else:
        top, left, bottom, right = bbox
        bbox_area = (bottom - top + 1) * (right - left + 1)
        bbox_value = {
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
        }
    active_rows = sum(any(cell != background for cell in row) for row in grid)
    active_columns = sum(
        any(grid[row][column] != background for row in range(height))
        for column in range(width)
    )
    return {
        "dimensions": [height, width],
        "area": height * width,
        "maximum_dimension": max(height, width),
        "conservative_background_color": background,
        "color_counts": {str(color): count for color, count in sorted(Counter(cell for row in grid for cell in row).items())},
        "non_background_cells": occupied,
        "non_background_occupancy": round(occupied / (height * width), 6),
        "active_bounding_box": bbox_value,
        "active_bounding_box_area": bbox_area,
        "active_bounding_box_fraction": round(bbox_area / (height * width), 6),
        "rows_with_relevant_content_fraction": round(active_rows / height, 6),
        "columns_with_relevant_content_fraction": round(active_columns / width, 6),
        "approximate_connected_component_count": _components(grid, background),
    }


def _episode_grids(task: Mapping[str, Any]) -> list[tuple[str, Sequence[Sequence[int]]]]:
    result: list[tuple[str, Sequence[Sequence[int]]]] = []
    for index, pair in enumerate(task["train"], 1):
        result.extend(((f"train_{index}_input", pair["input"]), (f"train_{index}_output", pair["output"])))
    for index, pair in enumerate(task["test"], 1):
        result.append((f"test_{index}_input", pair["input"]))
        if "output" in pair:
            result.append((f"test_{index}_output", pair["output"]))
    return result


def _canvas_summary(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = [tuple(item["dimensions"]) for item in metrics]
    input_dimensions = [
        tuple(item["dimensions"]) for item in metrics if item["grid"].endswith("_input")
    ]
    output_dimensions = [
        tuple(item["dimensions"]) for item in metrics if item["grid"].endswith("_output")
    ]
    maximum = max((item["maximum_dimension"] for item in metrics), default=0)
    return {
        "maximum_dimension": maximum,
        "maximum_total_area": max((item["area"] for item in metrics), default=0),
        "minimum_non_background_occupancy": min(
            (item["non_background_occupancy"] for item in metrics), default=0
        ),
        "maximum_non_background_occupancy": max(
            (item["non_background_occupancy"] for item in metrics), default=0
        ),
        "grid_size_variation": len(set(dimensions)),
        "input_size_variation": len(set(input_dimensions)),
        "output_size_variation": len(set(output_dimensions)),
        "has_dimension_above_15": maximum > 15,
        "has_dimension_above_20": maximum > 20,
        "has_dimension_above_25": maximum > 25,
        "interpretation": "Descriptive only; grid size, occupancy, and component counts are not universal quality gates.",
    }


def diagnose_arc_task_v4(task: dict[str, Any]) -> dict[str, Any]:
    base = diagnose_arc_task_v3(task)
    metrics = [
        {"grid": label, **grid_canvas_metrics(grid)}
        for label, grid in _episode_grids(task)
    ] if base.get("hard_valid") else []
    return {
        **base,
        "diagnostic_version": "arcgen-v4-control-4.0.0",
        "canvas_metrics": metrics,
        "canvas_summary": _canvas_summary(metrics),
    }


def diagnose_public_arc_task_v4(task: dict[str, Any]) -> dict[str, Any]:
    """Run input-facing diagnostics without ever receiving a hidden answer.

    The inherited V3 diagnostic validates complete authoring episodes even
    though all of its semantic-artifact checks inspect public inputs.  V4 uses
    a disposable, input-cloned test output solely to satisfy that legacy
    validator, then removes every surrogate-output statistic before returning.
    No hidden grid is passed to this function.
    """

    validation = validate_arc_task(task, require_test_outputs=False)
    if not validation.valid:
        return {
            "hard_valid": False,
            "hard_errors": list(validation.errors),
            "format_warnings": list(validation.warnings),
            "flags": [],
            "metrics": {},
            "diagnostic_version": "arcgen-v4-control-public-4.0.0",
            "canvas_metrics": [],
            "canvas_summary": _canvas_summary([]),
            "hidden_output_material_used": False,
        }
    surrogate = deepcopy(task)
    for pair in surrogate["test"]:
        pair["output"] = deepcopy(pair["input"])
    result = diagnose_arc_task_v3(surrogate)
    metrics = deepcopy(result.get("metrics", {}))
    if "output_sizes" in metrics:
        metrics["output_sizes"] = {
            key: value for key, value in metrics["output_sizes"].items()
            if key.startswith("train_")
        }
        metrics["output_size_variety"] = len(
            {tuple(value) for value in metrics["output_sizes"].values()}
        )
    canvas = [
        {"grid": label, **grid_canvas_metrics(grid)}
        for label, grid in _episode_grids(task)
    ]
    return {
        **result,
        "hard_valid": True,
        "hard_errors": [],
        "format_warnings": list(validation.warnings),
        "metrics": metrics,
        "diagnostic_version": "arcgen-v4-control-public-4.0.0",
        "canvas_metrics": canvas,
        "canvas_summary": _canvas_summary(canvas),
        "hidden_output_material_used": False,
    }


def _expected_pair_dimensions(plan: Mapping[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    def pair(value: Mapping[str, Any]) -> list[int]:
        return [int(value["height"]), int(value["width"])]

    train = [
        [pair(item["input_dimensions"]), pair(item["output_dimensions"])]
        for item in plan["training_pairs"]
    ]
    test = [
        [pair(item["input_dimensions"]), pair(item["output_dimensions"])]
        for item in plan["test_pairs"]
    ]
    return train, test


def check_plan_fidelity(plan: Mapping[str, Any], realization: Mapping[str, Any]) -> dict[str, Any]:
    """Check every objectively frozen structural decision before semantic review."""

    if realization.get("status") != "realized":
        return {
            "passed": False,
            "objective_failures": ["realizer refused the plan"],
            "declared_deviations": deepcopy(realization.get("deviations", [])),
            "semantic_review_required": False,
        }
    task = realization["task"]
    hidden = realization["authoring_metadata"]["expected_test_outputs"]
    failures: list[str] = []
    declared = list(realization.get("deviations", []))
    if len(task["train"]) != plan["episode_overview"]["training_pair_count"]:
        failures.append("training pair count differs from the frozen plan")
    if len(task["test"]) != plan["episode_overview"]["test_pair_count"]:
        failures.append("test pair count differs from the frozen plan")
    if len(hidden) != len(task["test"]):
        failures.append("hidden output cardinality differs from test count")
    planned_train, planned_test = _expected_pair_dimensions(plan)
    actual_train = [[_dimensions(pair["input"]), _dimensions(pair["output"])] for pair in task["train"]]
    actual_test = [
        [_dimensions(pair["input"]), _dimensions(hidden[index])]
        for index, pair in enumerate(task["test"])
        if index < len(hidden)
    ]
    if actual_train != planned_train:
        failures.append("one or more training grid dimensions differ from the frozen plan")
    if actual_test != planned_test:
        failures.append("one or more test grid dimensions differ from the frozen plan")
    trace = realization["authoring_metadata"].get("pair_construction_trace", [])
    labels = {item.get("pair") for item in trace}
    expected_labels = {
        *(f"train_{index}" for index in range(1, len(task["train"]) + 1)),
        *(f"test_{index}" for index in range(1, len(task["test"]) + 1)),
    }
    if labels != expected_labels:
        failures.append("construction witness does not cover every pair exactly")
    if declared:
        failures.append("realizer declared deviations from the frozen plan")
    fidelity = realization["authoring_metadata"].get("plan_fidelity", {})
    if not fidelity.get("statement", "").strip():
        failures.append("realizer omitted its plan-fidelity statement")
    if fidelity.get("deviations") != declared:
        failures.append("top-level and plan-fidelity deviation ledgers disagree")
    return {
        "passed": not failures,
        "objective_failures": failures,
        "declared_deviations": declared,
        "semantic_review_required": not failures,
        "actual_train_dimensions": actual_train,
        "actual_test_dimensions": actual_test,
        "planned_train_dimensions": planned_train,
        "planned_test_dimensions": planned_test,
    }


def public_task(realization: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(realization["task"])


def complete_task(realization: Mapping[str, Any]) -> dict[str, Any]:
    task = public_task(realization)
    outputs = realization["authoring_metadata"]["expected_test_outputs"]
    if len(outputs) != len(task["test"]):
        raise V4PlanFidelityError("hidden output cardinality mismatch")
    for index, output in enumerate(outputs):
        task["test"][index]["output"] = deepcopy(output)
    return task


def hidden_outputs_match(derived: Sequence[Any], realization: Mapping[str, Any]) -> bool:
    return list(derived) == realization["authoring_metadata"]["expected_test_outputs"]


__all__ = [
    "V4PlanFidelityError",
    "check_plan_fidelity",
    "complete_task",
    "diagnose_arc_task_v4",
    "diagnose_public_arc_task_v4",
    "grid_canvas_metrics",
    "hidden_outputs_match",
    "public_task",
]
