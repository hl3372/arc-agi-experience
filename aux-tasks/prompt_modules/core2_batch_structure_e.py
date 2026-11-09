"""Prompt templates for structure batch E tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _contact_convention(value: str) -> str:
    return {
        "object_a_side_cells": "object A cells that touch object B",
        "object_b_side_cells": "object B cells that touch object A",
        "both_objects_side_cells": "cells from both objects that touch across the interface",
    }.get(value, value.replace("_", " "))


def _contact_output_mode(value: str, bg_label: str) -> str:
    return {
        "preserve_scene": "preserve the scene and recolor the marked contact cells",
        "contact_mask": (
            f"return a mask filled with the input background color {bg_label} that contains "
            f"only the marked contact cells"
        ),
    }.get(value, value.replace("_", " "))


def _line_output_mode(value: str) -> str:
    return {
        "recolor_entire_line": "recolor the entire selected line",
        "mark_line_endpoints": "mark only the two endpoints of the selected line",
    }.get(value, value.replace("_", " "))


@register_prompt("116")
def _p116(m):
    return (
        f"Return only the top-left copy of the smallest rectangular tile: the tile has "
        f"height {_m(m, 'tile_height')} and width {_m(m, 'tile_width')}, and repeating it "
        f"by translation recreates the whole input grid. Use zero-indexed input rows "
        f"0 through {int(_m(m, 'tile_height')) - 1} and columns 0 through "
        f"{int(_m(m, 'tile_width')) - 1}, preserving their original order and colors."
    )


@register_prompt("117")
def _p117(m):
    mode = _m(m, "period_mode")
    if mode == "row_wise":
        detail = (
            f"for each row independently, cells in columns with the same zero-indexed "
            f"column remainder modulo {_m(m, 'period_length')} have the same color"
        )
    elif mode == "column_wise":
        detail = (
            f"for each column independently, cells in rows with the same zero-indexed "
            f"row remainder modulo {_m(m, 'period_length')} have the same color"
        )
    else:
        detail = (
            f"the whole grid repeats a rectangular tile of height {_m(m, 'tile_height')} "
            f"and width {_m(m, 'tile_width')}; cells with the same zero-indexed "
            f"(row modulo {_m(m, 'tile_height')}, column modulo {_m(m, 'tile_width')}) "
            f"have the same color"
        )
    return (
        f"Replace every unknown placeholder cell of color {_ci(m, 'unknown_color')} using "
        f"this {mode.replace('_', '-')} periodic rule: {detail}. Use the known cells as "
        f"anchors; equivalent known cells agree. Preserve all known cells; do not overwrite them. "
        f"Output the full repaired grid."
    )


@register_prompt("119")
def _p119(m):
    kind = _m(m, "pattern_type").replace("_", " ")
    extra = ""
    if _m(m, "pattern_type") == "diagonal_stripes":
        extra = f" running {_m(m, 'diagonal_direction').replace('_', ' ')}"
    return (
        f"Repair the missing cells in the equally spaced {kind}{extra}: every stripe or lattice "
        f"cell belongs to the pattern color {_ci(m, 'pattern_color')}, and the background is "
        f"{_ci(m, 'background_color')}. Use spacing {_m(m, 'spacing')}."
    )


@register_prompt("155")
def _p155(m):
    mode = _contact_output_mode(_m(m, "output_mode"), _ci(m, "background_color"))
    convention = _contact_convention(_m(m, "contact_convention"))
    return (
        f"Find the 4-neighbor contact interface between object A of color {_ci(m, 'object_a_color')} "
        f"and object B of color {_ci(m, 'object_b_color')}. Mark {convention} with "
        f"{_ci(m, 'contact_color')}; {mode}. Ignore cells of any other color when locating the "
        f"interface (they are background detail)."
    )


@register_prompt("168")
def _p168(m):
    rule = _m(m, "representative_rule")
    if rule == "middle":
        representative = "single middle representative cell; for even-length runs use the earlier top-or-left middle cell"
    elif rule == "first":
        representative = "first cell (the earliest one in the run, top-or-left end)"
    elif rule == "last":
        representative = "last cell (the latest one in the run, bottom-or-right end)"
    elif rule == "endpoints":
        representative = "two endpoint cells (the first and last cells of the run)"
    else:
        representative = f"{rule} representative cell set"
    return (
        f"For every maximal {_m(m, 'axis')} run of target color {_ci(m, 'target_color')}, "
        f"replace the run by its {representative} in marker color {_ci(m, 'marker_color')}; "
        f"the other cells from that run become "
        f"{_ci(m, 'background_color')}. Cells of any other color are background detail: leave "
        f"them unchanged and do not treat them as part of a run."
    )


@register_prompt("169")
def _p169(m):
    policy = _m(m, "tie_policy")
    tie_phrase = {
        "lexicographically_first_pair": (
            "mark only one (A-cell, B-cell) pair: among the closest pairs choose the one whose "
            "object-A cell comes first in row-major order (top-to-bottom, then left-to-right), "
            "breaking any remaining tie by the row-major order of its object-B cell, and mark "
            "both cells of that single chosen pair"
        ),
        "object_a_minimizers": "mark every object-A cell that participates in a closest pair",
        "object_b_minimizers": "mark every object-B cell that participates in a closest pair",
        "all_minimizing_cells": "mark every cell (from either object) that participates in a closest pair",
    }.get(policy, f"apply tie policy {policy.replace('_', ' ')}")
    return (
        f"Between object A of color {_ci(m, 'object_a_color')} and object B of color "
        f"{_ci(m, 'object_b_color')}, find the cell pairs that minimize "
        f"{_m(m, 'distance_metric')} distance; {tie_phrase}, using marker color "
        f"{_ci(m, 'marker_color')}. Measure distances only between object-A and object-B cells; "
        f"ignore cells of every other color and leave them unchanged."
    )


@register_prompt("174")
def _p174(m):
    return (
        f"Exactly three cells of corner color {_ci(m, 'corner_color')} mark three corners of one "
        f"axis-aligned rectangle. Add the missing fourth corner in color {_ci(m, 'corner_color')} so all "
        f"four corners are present. Any cells of other colors are background detail: leave every "
        f"other cell exactly as it is and change only the single missing-corner cell."
    )


@register_prompt("177")
def _p177(m):
    return (
        f"For each target object of color {_ci(m, 'target_color')}, mark every cell of that object "
        f"whose immediate neighbor in the {_m(m, 'direction')} direction is not part of the same "
        f"object (i.e. that neighbor is outside the grid or not a cell of this object), using edge "
        f"color {_ci(m, 'edge_color')}. This includes cells along interior gaps, not only the "
        f"single extremal cell of each line. Objects use {_m(m, 'connectivity')} connectivity. "
        f"Consider only cells of the target color as object cells; cells of any other color are "
        f"background detail that you leave unchanged."
    )


@register_prompt("181")
def _p181(m):
    axis_singular = {"rows": "row", "columns": "column"}.get(_m(m, "axis"), _m(m, "axis"))
    return (
        f"For every {axis_singular} that contains at least one trigger cell of color "
        f"{_ci(m, 'trigger_color')}, recolor the whole {axis_singular} to band color "
        f"{_ci(m, 'band_color')}, overwriting every cell in it (including the trigger cell and any "
        f"other non-background cells). Leave the other {axis_singular}s unchanged."
    )


@register_prompt("182")
def _p182(m):
    count_rule = _m(m, "count_rule")
    if count_rule == "target_color_count":
        count_detail = f"count of target color {_ci(m, 'target_color')} cells"
    elif count_rule == "non_background_count":
        count_detail = f"count of cells that are not background color {_ci(m, 'background_color')}"
    else:
        count_detail = "count of distinct colors present"
    return (
        f"Among all {_m(m, 'axis')}s, select the one with the {_m(m, 'extremum')} "
        f"{count_detail}. Use tie policy {_m(m, 'tie_policy')} and "
        f"{_line_output_mode(_m(m, 'output_mode'))} with highlight color "
        f"{_ci(m, 'highlight_color')}."
    )
