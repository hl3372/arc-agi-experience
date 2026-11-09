"""Prompt templates for core2 batch paths B tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


DIRECTION_DELTAS = {
    "north": "north (row -1, column 0)",
    "south": "south (row +1, column 0)",
    "west": "west (row 0, column -1)",
    "east": "east (row 0, column +1)",
    "northwest": "northwest (row -1, column -1)",
    "northeast": "northeast (row -1, column +1)",
    "southwest": "southwest (row +1, column -1)",
    "southeast": "southeast (row +1, column +1)",
}


def _direction_list(meta: dict) -> str:
    return ", ".join(DIRECTION_DELTAS.get(str(d), str(d)) for d in meta.get("directions", []))


def _adjacency_phrase(connectivity: str) -> str:
    if str(connectivity) == "8":
        return "8-adjacency, where neighbors share an edge or a corner"
    return "4-adjacency, where neighbors share an edge"


def _sequence_text(meta: dict) -> str:
    parts = []
    for i in range(1, int(meta.get("sequence_length", 0)) + 1):
        parts.append(_ci(meta, f"sequence_color_{i}"))
    return ", ".join(parts)


@register_prompt("109")
def _p109(m):
    return (
        f"Find a shortest path from the color {_ci(m, 'start_color')} start cell to the color "
        f"{_ci(m, 'goal_color')} goal cell using 4-adjacency, where each step changes either the "
        f"zero-indexed row or the zero-indexed column by 1. Do not enter color {_ci(m, 'wall_color')} "
        f"wall cells. Recolor every cell on that path to color {_ci(m, 'path_color')}, including "
        "the start and goal cells. Leave wall cells and every non-path cell unchanged, even if a "
        "non-path cell is not a wall."
    )


@register_prompt("110")
def _p110(m):
    measure = _m(m, "measure")
    base = (
        f"The color {_ci(m, 'path_color')} cells alone form one one-cell-wide path using "
        "4-adjacency, where neighboring path cells share an edge. Ignore every other color. "
        "A path endpoint is a path cell with exactly one edge-adjacent path neighbor. "
        "Length means the number of path cells, including both endpoints. Turns means the "
        "number of times the step direction changes while walking from one endpoint to the other. "
    )
    if measure == "length":
        return base + "Return only the integer length."
    if measure == "turns":
        return base + "Return only the integer turn count."
    return (
        base
        + "Return both measurements in the exact format length=N;turns=M with no spaces."
    )


@register_prompt("112")
def _p112(m):
    stop = _m(m, "stop_condition")
    if stop == "stop_color":
        stop_text = (
            f"Stop before entering the first color {_ci(m, 'stop_color')} stop cell in that "
            "direction; stop cells are not ray cells."
        )
        suffix = "Leave the source cell, every stop cell, and every cell outside the cast rays unchanged."
    else:
        stop_text = "Continue through the border cell, then stop when the next step would leave the grid."
        suffix = "Leave the source cell and every cell outside the cast rays unchanged."
    return (
        f"From the single color {_ci(m, 'source_color')} source cell, cast one ray in each of these "
        f"directions: {_direction_list(m)}. A ray starts at the adjacent cell, not at the "
        f"source cell. Color ray cells with color {_ci(m, 'ray_color')}. {stop_text} "
        f"{suffix}"
    )


@register_prompt("115")
def _p115(m):
    adjacency = _adjacency_phrase(_m(m, "connectivity"))
    return (
        f"Only color {_ci(m, 'line_color')} cells belong to the one-cell-wide line network; ignore "
        f"every other color. Using {adjacency}, count each line cell's neighboring line-color cells. "
        "An endpoint has exactly one such neighbor. A branchpoint has three or more such neighbors; "
        "cells with exactly two neighbors are neither. Return the exact text endpoints=N;branchpoints=M "
        "with no spaces."
    )


@register_prompt("140")
def _p140(m):
    alignment_text = {
        "horizontal": "horizontally",
        "vertical": "vertically",
        "main_diagonal": "along the main diagonal",
        "anti_diagonal": "along the anti-diagonal",
    }.get(_m(m, "alignment"), str(_m(m, "alignment")))
    return (
        f"Exactly two color {_ci(m, 'endpoint_color')} endpoint cells are aligned {alignment_text}. "
        f"Draw the full straight segment connecting them in color {_ci(m, 'line_color')}, including both endpoints. "
        "Ignore every other color and leave every cell that is not on that segment unchanged."
    )


@register_prompt("141")
def _p141(m):
    border = _m(m, "border")
    if border in ("top", "bottom"):
        axis = "column"
    else:
        axis = "row"
    return (
        f"Project all color {_ci(m, 'target_color')} cells onto the {border} border. "
        f"Mark each border cell whose {axis} contains at least one color {_ci(m, 'target_color')} "
        f"target cell with color {_ci(m, 'projection_color')}. "
        "Ignore every other color and leave all other cells unchanged."
    )


@register_prompt("142")
def _p142(m):
    if _m(m, "condition") == "target_objects":
        counted = f"4-connected color {_ci(m, 'target_color')} objects"
    else:
        counted = f"color {_ci(m, 'target_color')} cells"
    return (
        f"Count the {counted}, ignoring every other color. From the color {_ci(m, 'anchor_color')} anchor cell, draw a "
        f"{_m(m, 'orientation')} bar toward {_m(m, 'direction')} in color {_ci(m, 'bar_color')}. "
        "The bar length equals the count and includes the anchor cell."
    )


@register_prompt("149")
def _p149(m):
    return (
        f"Using {_m(m, 'metric')} distance to the nearest color {_ci(m, 'source_color')} source cell, "
        f"mark every background color {_ci(m, 'background_color')} cell at exact distance {_m(m, 'distance')} "
        f"with color {_ci(m, 'ring_color')}."
    )


@register_prompt("159")
def _p159(m):
    return (
        f"The one-cell-wide 4-connected path is made of the color {_ci(m, 'path_color')} cells together with "
        f"the single color {_ci(m, 'start_endpoint_color')} endpoint cell; ignore every other color. "
        f"Starting at the color {_ci(m, 'start_endpoint_color')} endpoint, walk along that path and recolor every "
        f"path cell using this repeating color sequence: {_sequence_text(m)}. "
        "Leave every cell that is not on the path unchanged."
    )


@register_prompt("170")
def _p170(m):
    return (
        f"Two cells have marker color {_ci(m, 'marker_color')}. Using {_m(m, 'metric')} distance, "
        f"mark every background color {_ci(m, 'background_color')} cell that is equidistant from the two markers "
        f"with color {_ci(m, 'bisector_color')}."
    )
