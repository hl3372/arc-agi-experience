"""Prompt templates for object-focused second-batch slice A."""
from __future__ import annotations

from prompts import (
    _ci,
    _connectivity_phrase,
    _largest_object_rule,
    _m,
    _symbol_mode,
    register_prompt,
)


def _color(meta: dict, value: int) -> str:
    m = dict(meta)
    m["_render_color"] = value
    return _ci(m, "_render_color")


def _color_list(meta: dict, values: list[int]) -> str:
    return ", ".join(_color(meta, int(v)) for v in values)


def _rank_mapping(meta: dict) -> str:
    return ", ".join(
        f"rank {item['rank']} -> color {_color(meta, int(item['color']))}"
        for item in meta.get("rank_to_color_mapping", [])
    )


def _index_mapping(meta: dict) -> str:
    return ", ".join(
        f"index {item['index']} -> color {_color(meta, int(item['color']))}"
        for item in meta.get("index_to_color_mapping", [])
    )


def _sort_key_phrase(meta: dict) -> str:
    """Task 98's sort key, phrased to read after "in ascending order of"."""
    key = _m(meta, "sort_key")
    if key == "cell_count":
        return "the number of cells each object contains"
    if key == "bbox_height":
        return "bounding-box height, counted in rows"
    if key == "bbox_width":
        return "bounding-box width, counted in columns"
    return key.replace("_", " ")


def _size_measure_phrase(meta: dict) -> str:
    measure = _m(meta, "size_measure")
    if measure == "cell_count":
        return "object cell count"
    if measure == "bounding_box_area":
        return "bounding-box area, computed as height in rows times width in columns"
    return measure.replace("_", " ")


def _rank_order_phrase(meta: dict) -> str:
    order = _m(meta, "rank_order")
    if order == "smallest_to_largest":
        return "smallest to largest"
    if order == "largest_to_smallest":
        return "largest to smallest"
    return order.replace("_", " ")


def _distance_metric_phrase(meta: dict) -> str:
    metric = _m(meta, "distance_metric")
    if metric == "min_manhattan":
        return (
            "minimum Manhattan distance from the marker cell to any cell in the object "
            "(absolute row difference plus absolute column difference)"
        )
    if metric == "top_left_manhattan":
        return (
            "Manhattan distance from the marker cell to the object's top-left bounding-box cell "
            "(absolute row difference plus absolute column difference)"
        )
    return metric.replace("_", " ")


def _association_rule_phrase(meta: dict) -> str:
    rule = _m(meta, "association_rule")
    if rule == "nearest_min_manhattan":
        return (
            "the object with the smallest minimum Manhattan distance from the marker cell to "
            "any of the object's cells (absolute row difference plus absolute column difference)"
        )
    if rule == "nearest_top_left_manhattan":
        return (
            "the object whose top-left bounding-box cell has the smallest Manhattan distance to "
            "the marker cell (absolute row difference plus absolute column difference)"
        )
    return rule.replace("_", " ")


def _landmark_phrase(meta: dict) -> str:
    landmark = _m(meta, "landmark")
    if landmark == "medoid":
        return (
            "medoid cell (the cell minimizing total Manhattan distance to all the object's "
            "cells; if several cells tie, the row-major first one)"
        )
    return f"{landmark.replace('_', '-')} bounding-box corner"


def _ordering_phrase(meta: dict) -> str:
    ordering = _m(meta, "object_ordering")
    if ordering == "top_to_bottom_left_to_right":
        return "top bounding-box row, then left bounding-box column"
    if ordering == "left_to_right_top_to_bottom":
        return "left bounding-box column, then top bounding-box row"
    if ordering == "bottom_to_top_left_to_right":
        return "bottom bounding-box row from bottom to top, then left bounding-box column"
    return ordering.replace("_", " ")


@register_prompt("97")
def _p97(m):
    # Rebuilt on 2026-09-15 (generators/odd_shape_97.py), reworked 2026-09-15
    # after the follow-up review.  The connectivity and the criterion are drawn
    # per example; the two criteria can never single out DIFFERENT objects (same
    # shape means same cell count), so the criterion decides exactly where the
    # other reading finds no unique object -- about 30% of examples, which is
    # what keeps the scene from naming the criterion.  The standard object rule
    # supplies "Ignoring the background" (never named), the at-least-two-cells
    # clause, the isolated-cell clause and the colour clause.  The parenthesis
    # closes the over-reading "all the others are alike": the non-odd objects
    # come in groups that share a count, not one uniform family.
    conn = _connectivity_phrase(m, default="4")
    if _m(m, "equivalence_rule") == "same_cell_count":
        criterion = (
            f"Among the color {_ci(m, 'object_color')} objects, and ignoring every object of "
            "another color, exactly one has a different number of cells from all the others "
            "(the other objects need not all match each other): count the cells that belong "
            "to each object."
        )
    else:
        criterion = (
            f"Among the color {_ci(m, 'object_color')} objects, and ignoring every object of "
            "another color, exactly one has a different shape from all the others (the other "
            "objects need not all match each other): two objects have the same shape only when "
            "one can be moved onto the other by a translation, so do not rotate or reflect them."
        )
    return (
        f"{_largest_object_rule(m, conn)} {criterion} "
        f"Recolor every cell of that object to color {_ci(m, 'target_color')}. "
        "Leave every other cell unchanged."
    )


@register_prompt("98")
def _p98(m):
    # Rebuilt on 2026-09-15 (generators/sorted_strip_98.py).  The standard object
    # rule replaces "all 4-connected objects": it says same-color, it supplies
    # "Ignoring the background" without naming it, and its isolated-cell clause
    # disposes of the sparse texture, which the output drops.  The layout rule is
    # stated completely -- axis, start cell, gap, alignment, output dimensions and
    # what fills the rest -- because "Leave all other cells empty" was undefined.
    # The gap is the number of empty columns (rows) BETWEEN the two bounding
    # boxes: "K cells to the right of" also read as "shift the left edge by K".
    conn = _connectivity_phrase(m, default="4")
    key = _sort_key_phrase(m)
    axis = _m(m, "strip_axis")
    gap = _m(m, "gap")
    if axis == "horizontal":
        unit = "column" if str(gap) == "1" else "columns"
        step = (
            "and place each later object's bounding box to the right of the previous "
            f"object's bounding box, leaving exactly {gap} empty {unit} between the two "
            "bounding boxes, with its top edge on the same row"
        )
    else:
        unit = "row" if str(gap) == "1" else "rows"
        step = (
            "and place each later object's bounding box below the previous object's "
            f"bounding box, leaving exactly {gap} empty {unit} between the two bounding "
            "boxes, with its left edge in the same column"
        )
    return (
        f"{_largest_object_rule(m, conn)} Sort the objects in ascending order of {key}; when two "
        "objects have the same value, the one whose bounding box starts on the smaller row comes "
        "first, and if those are equal the one whose bounding box starts on the smaller column "
        "comes first. The output has the same dimensions as the input. Place the first object's "
        f"bounding box with its top-left corner at row {_m(m, 'strip_start_row')}, column "
        f"{_m(m, 'strip_start_col')}, {step}; each object keeps its own shape and color. Every "
        "cell that no placed object covers is plain background, without any of the input's "
        "background texture. Rows and columns are 0-indexed from the top-left corner."
    )


@register_prompt("105")
def _p105(m):
    return (
        f"For each color {_ci(m, 'target_color')} 4-connected object, find the nearest "
        f"marker cell whose color is one of {_color_list(m, m.get('marker_colors', []))}. "
        "Ignore and preserve objects of other colors. Use this distance: "
        f"{_distance_metric_phrase(m)}. If two markers tie for nearest, choose the marker "
        "with the earliest row-major position, meaning smaller row first and then smaller "
        "column. Recolor the whole target object to the chosen marker's color; leave marker "
        "cells unchanged."
    )


@register_prompt("106")
def _p106(m):
    return (
        f"Rank the color {_ci(m, 'target_color')} 4-connected objects by "
        f"{_size_measure_phrase(m)} from {_rank_order_phrase(m)}. Ignore and preserve objects "
        "of other colors. Rank 1 is the first object in that ordered list. The measured sizes "
        f"are unique in this example. Recolor each target object using this mapping: {_rank_mapping(m)}."
    )


@register_prompt("108")
def _p108(m):
    return (
        f"Treat color {_ci(m, 'background_color')} as background. Count the cells of each "
        "non-background color in the input grid, then rank those colors from most frequent "
        "to least frequent. The non-background color frequencies are unique in this example. "
        "Rank 1 is the most frequent color. "
        f"Recolor every cell of each ranked color simultaneously using this mapping: {_rank_mapping(m)}. "
        "Preserve background cells and return the full recolored grid."
    )


@register_prompt("143")
def _p143(m):
    return (
        f"Order the color {_ci(m, 'target_color')} 4-connected objects by "
        f"{_ordering_phrase(m)}. "
        f"Ignore objects of other colors. Recolor each target object according to its 1-based "
        f"index using: {_index_mapping(m)}. Leave other-color objects unchanged."
    )


@register_prompt("144")
def _p144(m):
    return (
        f"For each color {_ci(m, 'target_color')} 4-connected object, mark its medoid "
        f"cell with color {_ci(m, 'marker_color')}. The medoid minimizes total Manhattan "
        "distance to all cells of that object; if tied, choose the row-major first cell. "
        "Ignore and preserve objects of other colors."
    )


@register_prompt("152")
def _p152(m):
    return (
        f"Find the color {_ci(m, 'target_color')} object associated with the color "
        f"{_ci(m, 'marker_color')} marker cell. The associated object is "
        f"{_association_rule_phrase(m)}. "
        "Ignore objects of other colors. "
        "If objects tie, choose the one with row-major top-left bounding-box corner. "
        "Keep that object and the marker cell, and replace the other target objects with "
        f"background color {_ci(m, 'background_color')}. Leave other-color objects unchanged."
    )


@register_prompt("161")
def _p161(m):
    if _m(m, "color_rule") == "marker_color":
        color_phrase = f"color {_ci(m, 'marker_color')}"
    else:
        color_phrase = "the object's original color"
    return (
        "Replace every non-background 4-connected object with a single cell at its "
        f"{_landmark_phrase(m)}, using {color_phrase}. All other cells become "
        f"background color {_ci(m, 'background_color')}."
    )


@register_prompt("172")
def _p172(m):
    return (
        "Replace every non-background 4-connected object with a straight bar in the "
        f"{_m(m, 'direction')} direction, starting from its {_landmark_phrase(m)}. "
        "The bar length equals that object's cell count and the bar uses the object's "
        f"original color. All other cells become background color {_ci(m, 'background_color')}."
    )
