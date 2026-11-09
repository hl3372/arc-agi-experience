"""Prompt templates for the core2 pattern batch D tasks."""
from __future__ import annotations

from prompts import _ci, _m, _pattern_literal, _symbol_mode, register_prompt, render_pattern_grid


def _render_table(meta: dict) -> str:
    palette = meta.get("__color_palette__")
    parts = []
    for entry in meta.get("replacement_table", []):
        motif = render_pattern_grid(entry["motif"], color_palette=palette, row_sep="//", compact=True)
        parts.append(f"{_ci({**meta, 'color': entry['color']}, 'color')}->{motif}")
    return "; ".join(parts)


def _render_color(meta: dict, value: int) -> str:
    return _ci({**meta, "_color": value}, "_color")


def _singular_axis(value: str) -> str:
    return {"rows": "row", "columns": "column"}.get(value, value)


def _side_phrase(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(part) for part in value)
    return str(value)


def _axis_phrase(value: str) -> str:
    return {
        "row": "rows",
        "column": "columns",
        "both": "rows and columns",
    }.get(value, value)


@register_prompt("131")
def _p131(m):
    return (
        "Replace every exact occurrence of source motif "
        f"{_pattern_literal(m, 'source_motif')} with target motif "
        f"{_pattern_literal(m, 'target_motif')}. Apply all replacements simultaneously, "
        "and leave every other cell (including any scattered background detail) unchanged."
    )


@register_prompt("132")
def _p132(m):
    return (
        "Replace each input cell by the motif assigned to its color, preserving row and column order. "
        f"Use this color-to-motif table: {_render_table(m)}."
    )


@register_prompt("139")
def _p139(m):
    return (
        f"Using {_m(m, 'connectivity', '4')}-connectivity, find each target object of color "
        f"{_ci(m, 'target_color')}. On the selected {_axis_phrase(_m(m, 'axis'))}, fill every background "
        f"color {_ci(m, 'background_color')} cell strictly between two cells of the same target "
        f"object on that row or column with color {_ci(m, 'fill_color')}. Cells of any other color are "
        "background detail: only background cells are ever filled, and every other cell stays unchanged."
    )


@register_prompt("150")
def _p150(m):
    return (
        f"The two color {_ci(m, 'corner_marker_color')} cells mark opposite corners of a rectangle. "
        f"Draw the {_m(m, 'rectangle_mode')} rectangle using color {_ci(m, 'draw_color')}; "
        "the rectangle may overwrite cells inside it, including the corner markers. Leave every cell "
        "outside the rectangle unchanged, including any scattered background detail."
    )


@register_prompt("160")
def _p160(m):
    return (
        f"Inside the rectangle outlined by color {_ci(m, 'region_border_color')}, fill the blank "
        f"color {_ci(m, 'blank_color')} region with a checkerboard using colors "
        f"{_ci(m, 'anchor_color')} and {_ci(m, 'other_checker_color')}. The existing color "
        f"{_ci(m, 'anchor_color')} cell is the checkerboard anchor phase. Leave everything outside the "
        "outlined rectangle unchanged, including any scattered background detail."
    )


@register_prompt("163")
def _p163(m):
    return (
        f"Output the subgrid induced by rows containing row marker color {_ci(m, 'row_marker_color')} "
        f"and columns containing column marker color {_ci(m, 'column_marker_color')}. Preserve the "
        "selected row and column order."
    )


@register_prompt("165")
def _p165(m):
    rule = _m(m, "summary_rule")
    axis = _m(m, "axis")
    if axis == "rows":
        scan_phrase = "scanning each row from left to right (starting at column 0)"
        orientation = "Output a one-column strip with one cell per input row, in row order."
    else:
        scan_phrase = "scanning each column from top to bottom (starting at row 0)"
        orientation = "Output a one-row strip with one cell per input column, in column order."
    if rule == "majority_color":
        detail = f"the most frequent color; if several colors tie for most frequent, use background color {_ci(m, 'background_color')}"
    elif rule == "first_non_background":
        detail = f"the first non-background color when {scan_phrase}, or background color {_ci(m, 'background_color')} if none exists"
    else:
        detail = f"color {_ci(m, 'true_color')} if target color {_ci(m, 'target_color')} appears, otherwise color {_ci(m, 'false_color')}"
    return f"Compress each {_singular_axis(axis)} into one output cell using this summary rule: {detail}. {orientation}"


@register_prompt("166")
def _p166(m):
    return (
        f"Count color {_ci(m, 'target_color')} cells in each {_singular_axis(_m(m, 'count_axis'))}. Output a blank "
        f"same-size grid with background color {_ci(m, 'background_color')}, then draw a color "
        f"{_ci(m, 'bar_color')} bar of that count from the {_m(m, 'edge')} edge for each line."
    )


@register_prompt("171")
def _p171(m):
    ordering = _m(m, "ordering")
    if ordering == "frequency_desc":
        detail = "from most frequent to least frequent, breaking ties by first appearance in row-major scan order"
    else:
        detail = "by first appearance in row-major scan order"
    return f"Output the distinct colors present as a one-{_m(m, 'output_orientation')} palette ordered {detail}."


@register_prompt("180")
def _p180(m):
    return (
        f"Remove the {_m(m, 'thickness')}-cell-thick outer padding of color {_ci(m, 'padding_color')} "
        f"from these side(s): {_side_phrase(m.get('sides', ''))}. Return the remaining grid."
    )
