"""Prompt templates for graph/count batch I tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _color_for_entry(meta: dict, entry: dict) -> str:
    return _ci({**meta, "_entry_color": entry["color"]}, "_entry_color")


def _ignore_texture(meta: dict) -> str:
    """Clause telling the solver to ignore scattered background decoration,
    present only when the example actually carries it. The decoration may be
    single dots or small shapes (always a colour disjoint from every meaningful
    colour), so the wording is deliberately shape-agnostic."""
    if meta.get("background_texture_color") is None:
        return ""
    return (
        f" Ignore the scattered color {_ci(meta, 'background_texture_color')} "
        "background decoration; it is not part of any object."
    )


def _order_colors(meta: dict) -> str:
    return ", ".join(_ci({**meta, "_order_color": color}, "_order_color") for color in meta["marker_order"])


def _hole_mapping(meta: dict) -> str:
    return "; ".join(
        f"{entry['hole_count']} hole{'' if entry['hole_count'] == 1 else 's'} -> "
        f"color {_color_for_entry(meta, entry)}"
        for entry in meta["hole_count_to_color"]
    )


def _degree_mapping(meta: dict) -> str:
    return "; ".join(
        f"degree {entry['degree']} -> color {_color_for_entry(meta, entry)}"
        for entry in meta["degree_to_color"]
    )


def _property_mapping(meta: dict) -> str:
    return "; ".join(
        f"value {entry['property_value']} -> color {_color_for_entry(meta, entry)}"
        for entry in meta["property_to_color"]
    )


def _count_phrase(meta: dict, set_name: str) -> str:
    color_key = f"set_{set_name}_color"
    type_key = f"set_{set_name}_count_type"
    if _m(meta, type_key) == "cells":
        return f"color {_ci(meta, color_key)} cells"
    return f"4-connected color {_ci(meta, color_key)} objects"


def _operation_phrase(operation: str) -> str:
    return {
        "sum": "A plus B",
        "absolute_difference": "the absolute difference between A and B",
        "a_minus_b_floor0": "A minus B, floored at zero",
        "b_minus_a_floor0": "B minus A, floored at zero",
        "product": "A times B",
    }[operation]


def _action_phrase(meta: dict, action_key: str, color_key: str) -> str:
    action = _m(meta, action_key)
    if action == "recolor_object":
        return f"recolor the whole target object to color {_ci(meta, color_key)}"
    if action == "mark_first_cell":
        return (
            "recolor only the target object's first cell in row-major order (the cell "
            "with the smallest row index, breaking ties by smallest column index) to "
            f"color {_ci(meta, color_key)}, leaving that object's remaining cells with "
            f"their original color {_ci(meta, 'target_color')}"
        )
    return f"erase the target object to background color {_ci(meta, 'background_color')}"


@register_prompt("197")
def _p197(m):
    return (
        f"The input is a regular panel grid separated by divider color {_ci(m, 'divider_color')}. "
        f"The unique row marker of color {_ci(m, 'row_marker_color')} and unique column marker of "
        f"color {_ci(m, 'column_marker_color')} select one block at their intersection. Return only "
        "that block's interior, excluding marker cells and divider cells."
    )


@register_prompt("198")
def _p198(m):
    axis = _m(m, "axis")
    return (
        f"Reorder the marked {axis} by moving each full {axis[:-1]} so the header markers appear "
        f"in this order: {_order_colors(m)}. Preserve every cell within each moved {axis[:-1]}."
    )


@register_prompt("199")
def _p199(m):
    axis = _m(m, "axis")
    coord = "row coordinate" if axis == "row" else "column coordinate"
    fixed = "column coordinate" if axis == "row" else "row coordinate"
    return (
        f"Translate every color {_ci(m, 'object_color')} object by changing only its {coord} "
        f"so its {_m(m, 'landmark').replace('_', ' ')} aligns with the target line of color "
        f"{_ci(m, 'target_line_color')}. Keep each object's {fixed} fixed. Wherever a moved object "
        f"covers the target line, the object color replaces the line.{_ignore_texture(m)}"
    )


@register_prompt("200")
def _p200(m):
    return (
        f"For each color {_ci(m, 'object_color')} object, count enclosed background holes using "
        f"{_m(m, 'hole_connectivity')}-connectivity for background regions. Recolor the object's "
        f"cells by this mapping: {_hole_mapping(m)}. Leave hole cells as background color "
        f"{_ci(m, 'background_color')}.{_ignore_texture(m)}"
    )


def _relation_phrase(meta: dict) -> str:
    """Concrete definition of the edge relation (the raw metadata key is an
    internal enum, never surfaced verbatim)."""
    relation = _m(meta, "adjacency_relation")
    if relation == "same_row_band":
        return (
            "their bounding boxes overlap in rows, that is there is at least one "
            "row index spanned by both objects' bounding boxes"
        )
    if relation == "same_column_band":
        return (
            "their bounding boxes overlap in columns, that is there is at least one "
            "column index spanned by both objects' bounding boxes"
        )
    return (
        "the smallest Manhattan distance between any cell of one object and any "
        f"cell of the other is at most {_m(meta, 'gap_threshold')} (the Manhattan "
        "distance between two cells is the sum of the absolute differences of "
        "their row and column indices)"
    )


@register_prompt("201")
def _p201(m):
    return (
        f"Build a graph whose nodes are the 4-connected color {_ci(m, 'object_color')} "
        f"objects. Connect two objects with an edge when {_relation_phrase(m)}. The degree "
        "of an object is the number of edges joining it to other objects. Recolor each "
        f"object by its graph degree using: {_degree_mapping(m)}."
        f"{_ignore_texture(m)}"
    )


@register_prompt("202")
def _p202(m):
    alignment = "row" if _m(m, "pairing_axis") == "horizontal" else "column"
    return (
        f"Pair each color {_ci(m, 'object_a_color')} object with the unique color "
        f"{_ci(m, 'object_b_color')} object aligned on the same {alignment}. "
        f"For every pair, fill the straight background gap between them with color {_ci(m, 'fill_color')}; "
        f"leave the two endpoint objects unchanged.{_ignore_texture(m)}"
    )


@register_prompt("203")
def _p203(m):
    return (
        f"For each color {_ci(m, 'object_color')} object, compute its "
        f"{_m(m, 'bbox_property').replace('_', ' ')}. Recolor the object using this mapping: "
        f"{_property_mapping(m)}.{_ignore_texture(m)}"
    )


@register_prompt("227")
def _p227(m):
    return (
        f"For each target object of color {_ci(m, 'target_color')}, decide whether at least one "
        f"reference object of color {_ci(m, 'reference_color')} is equivalent under "
        f"{_m(m, 'equivalence_relation').replace('_', ' ')}. If matched, "
        f"{_action_phrase(m, 'matched_action', 'matched_color')}; if unmatched, "
        f"{_action_phrase(m, 'unmatched_action', 'unmatched_color')}. Leave reference objects unchanged."
        f"{_ignore_texture(m)}"
    )


@register_prompt("228")
def _p228(m):
    return (
        f"Let A be the count of {_count_phrase(m, 'a')}. Let B be the count of "
        f"{_count_phrase(m, 'b')}. Return the numeric result of "
        f"{_operation_phrase(_m(m, 'operation'))} as plain arithmetic text."
        f"{_ignore_texture(m)}"
    )


@register_prompt("229")
def _p229(m):
    return (
        f"Let A be the count of {_count_phrase(m, 'a')}. Let B be the count of "
        f"{_count_phrase(m, 'b')}. Compute {_operation_phrase(_m(m, 'operation'))}. Starting at "
        f"the anchor cell of color {_ci(m, 'anchor_color')}, draw a {_m(m, 'bar_direction')} bar "
        f"of that length in color {_ci(m, 'bar_color')}, including the anchor cell."
        f"{_ignore_texture(m)}"
    )
