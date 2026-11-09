"""Prompt templates for symmetry/comparison second-batch slice F."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _color(meta: dict, value: int) -> str:
    m = dict(meta)
    m["_render_color"] = value
    return _ci(m, "_render_color")


def _color_list(meta: dict, values: list[int]) -> str:
    return ", ".join(_color(meta, int(v)) for v in values)


def _legend_entries(meta: dict) -> str:
    dimension = _m(meta, "measure_dimension", "height")
    label = "cell count" if dimension == "cell_count" else dimension
    return ", ".join(
        f"{label} {item.get('measure_value', item.get('bar_length'))} -> color {_color(meta, int(item['output_color']))}"
        for item in meta.get("legend_entries", [])
    )


def _mapping_sources(meta: dict) -> str:
    values = [int(item["from_color"]) for item in meta.get("color_mapping", [])]
    return _color_list(meta, values)


@register_prompt("103")
def _p103(m):
    measure_dimension = _m(m, "measure_dimension", "height")
    if measure_dimension == "width":
        measure_rule = "width"
        measure_definition = "Width means the number of columns the object occupies."
        legend_measure = "width"
    elif measure_dimension == "cell_count":
        measure_rule = "number of cells"
        measure_definition = "Cell count means the total number of cells in the object."
        legend_measure = "cell count"
    else:
        measure_rule = "height"
        measure_definition = "Height means the number of rows the object occupies."
        legend_measure = "height"
    if _m(m, "separator_axis") == "vertical":
        if _m(m, "legend_side") == "right":
            split = f"A full column of color {_ci(m, 'separator_color')} splits the grid into a scene on the left and a legend on the right."
            scene_phrase = "In the scene on the left"
        else:
            split = f"A full column of color {_ci(m, 'separator_color')} splits the grid into a legend on the left and a scene on the right."
            scene_phrase = "In the scene on the right"
    else:
        if _m(m, "legend_side") == "bottom":
            split = f"A full row of color {_ci(m, 'separator_color')} splits the grid into a scene above and a legend below."
            scene_phrase = "In the scene above"
        else:
            split = f"A full row of color {_ci(m, 'separator_color')} splits the grid into a legend above and a scene below."
            scene_phrase = "In the scene below"
    return (
        f"{split} The legend maps each color {_ci(m, 'object_color')} object {legend_measure} to an output color: "
        f"{_legend_entries(m)}. {scene_phrase}, look only at objects of color "
        f"{_ci(m, 'object_color')}. Recolor each one by its {measure_rule} using that legend. "
        f"{measure_definition} Leave everything else "
        "unchanged and return the full grid."
    )


@register_prompt("104")
def _p104(m):
    return (
        "The named grids before and after show a one-to-one color replacement rule. "
        f"Infer the permutation for source colors {_mapping_sources(m)} and apply it "
        "to every cell of the query grid, preserving the query grid dimensions. Return "
        "only the transformed query grid."
    )


@register_prompt("107")
def _p107(m):
    def source_text(source: str) -> str:
        if source == "row_header":
            return "the row-header color at row r, column 0"
        return "the column-header color at row 0, column c"

    return (
        f"Fill every blank interior cell of color {_ci(m, 'blank_color')} in the table. "
        "Use zero-indexed rows and columns: row 0 contains column headers, column 0 "
        "contains row headers, and interior cells have row > 0 and column > 0. For each "
        "blank interior cell at row r, column c, use the parity of r + c: when r + c "
        f"is even, place {source_text(_m(m, 'even_parity_source'))}; when r + c is odd, "
        f"place {source_text(_m(m, 'odd_parity_source'))}. Leave header cells unchanged. "
        "Return the completed grid."
    )


@register_prompt("120")
def _p120(m):
    return (
        "Compare the named grids left and right cell by cell. Output a grid with color "
        f"{_ci(m, 'match_color')} where the two cells match, and color "
        f"{_ci(m, 'difference_color')} where they differ."
    )


@register_prompt("127")
def _p127(m):
    return (
        f"Every object made from color {_ci(m, 'object_color')} is the same shape shown at "
        "a different rotation. On exactly one of them, two of its cells are already painted "
        f"color {_ci(m, 'decoration_color')}; these two decorated cells sit at specific "
        "positions within that object's shape. In every other (plain) object, paint the two "
        "cells that occupy those same positions within the shape with color "
        f"{_ci(m, 'decoration_color')}. Equivalently, the decoration is fixed to the shape "
        "itself, so within each object it lands on whichever cells the decorated object's "
        "decoration maps to once that object's shape is rotated to match this object's "
        "rotation. Determine each object's rotation from its own cells (the shape looks "
        "different at each of the four rotations); do not rotate by the difference between "
        "this object and the decorated one. Leave all other cells unchanged."
    )


@register_prompt("130")
def _p130(m):
    return (
        f"The input contains a source shape of color {_ci(m, 'source_color')} on the left "
        f"and a target shape of color {_ci(m, 'target_color')} on the right, separated by "
        f"color {_ci(m, 'divider_color')}. Compare the two panels cell by cell using the "
        f"same row and the same within-panel column: the source panel occupies columns 0 "
        f"through {int(_m(m, 'panel_width')) - 1} and the target panel occupies the columns "
        "after the divider, and source-panel cell (r, c) is compared against target-panel "
        "cell (r, c). Output an edit mask with the same size as one "
        f"panel: use color {_ci(m, 'add_color')} where a cell must be added to the source, "
        f"color {_ci(m, 'remove_color')} where a source cell must be removed, and color "
        f"{_ci(m, 'background_color')} elsewhere."
    )


@register_prompt("281")
def _p281(m):
    labels = ", ".join(m.get("allowed_symmetries", []))
    return (
        f"A rectangular frame of color {_ci(m, 'frame_color')} marks the region to inspect. "
        "Ignore the frame itself and everything outside it. Look only at the cells strictly "
        "inside the frame. For each allowed symmetry below, decide whether it is present in "
        "that framed interior. horizontal_reflection means row r, column c matches row "
        "height - 1 - r, column c; vertical_reflection means row r, column c matches row r, "
        "column width - 1 - c; main_diagonal_reflection means the framed interior is square "
        "and row r, column c matches row c, column r; anti_diagonal_reflection means the "
        "framed interior is square and row r, column c matches row width - 1 - c, column "
        "height - 1 - r; rotational_180 means row r, column c matches row height - 1 - r, "
        f"column width - 1 - c. Use this exact order: {labels}. Return exactly one line for "
        "each label in that order, formatted as label: yes or label: no. Do not add extra "
        "lines or extra text."
    )
