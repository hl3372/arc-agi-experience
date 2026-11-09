"""Prompt templates for core2 batch N correspondence tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _color(meta: dict, value: int) -> str:
    local = dict(meta)
    local["_render_color"] = int(value)
    return _ci(local, "_render_color")


def _color_list(meta: dict, values: list[int]) -> str:
    return ", ".join(_color(meta, int(v)) for v in values)


def _transform(meta: dict, transform: dict) -> str:
    if transform.get("type") == "erase":
        return f"erase it to background color {_ci(meta, 'background_color')}"
    if transform.get("type") == "recolor":
        return f"recolor it to color {_color(meta, int(transform['color']))}"
    return str(transform)


def _condition(meta: dict) -> str:
    condition = _m(meta, "property_condition")
    if condition == "cell_count_at_least":
        return f"has at least {_m(meta, 'threshold')} cells"
    if condition == "bbox_wider_than_tall":
        return "has a bounding box wider than it is tall"
    if condition == "cell_count_even":
        return "has an even number of cells"
    return condition.replace("_", " ")


def _mapping(meta: dict, key: str, label_key: str = "value") -> str:
    parts = []
    for item in meta.get(key, []):
        label = item[label_key].replace("_", " ")
        parts.append(f"{label} -> {_transform(meta, item['transform'])}")
    return "; ".join(parts)


def _relation(meta: dict) -> str:
    relation = _m(meta, "matching_rule")
    return {
        "same_normalized_shape": "the same arrangement of cells after shifting each object to its own top-left corner",
        "same_shape_under_rotation": "the same shape after any rotation",
        "same_shape_under_reflection": "the same shape after a horizontal or vertical reflection",
    }.get(relation, relation.replace("_", " "))


_MEDOID_DEFINITION = (
    "the object's medoid cell (the cell minimizing total Manhattan distance to all "
    "the object's cells; if tied, choose the smallest row then smallest column)"
)


def _association_relation(meta: dict) -> str:
    relation = _m(meta, "association_relation")
    return {
        "nearest_marker_by_min_manhattan": "the marker with the smallest Manhattan distance to any cell of the object",
        "nearest_marker_by_medoid": f"the marker with the smallest Manhattan distance to {_MEDOID_DEFINITION}",
    }.get(relation, relation.replace("_", " "))


@register_prompt("193")
def _p193(m):
    return (
        f"Count the marker cells of color {_ci(m, 'condition_color')}. Use that count as a "
        f"one-based index into the target objects of color {_ci(m, 'target_color')}, ordered by "
        "top-left bounding-box position from top to bottom, then left to right. "
        f"Apply this action to the indexed target object: {_transform(m, m['action_transform'])}. "
        "Leave every other cell unchanged."
    )


@register_prompt("223")
def _p223(m):
    return (
        f"Label the objects of color {_ci(m, 'object_color')} by top-left bounding-box order. "
        f"For each object, test whether it {_condition(m)}. If the condition is true, apply "
        f"transform A: {_transform(m, m['transform_A'])}. Otherwise apply transform B: "
        f"{_transform(m, m['transform_B'])}. Leave all non-object cells unchanged."
    )


@register_prompt("224")
def _p224(m):
    return (
        f"Label the objects of color {_ci(m, 'object_color')} by top-left bounding-box order. "
        "Classify each object by bounding-box orientation: wide, tall, or square. "
        f"Apply the property-value mapping: {_mapping(m, 'value_to_transform')}. "
        "Leave all non-object cells unchanged."
    )


@register_prompt("225")
def _p225(m):
    return (
        f"For each object of color {_ci(m, 'object_color')}, associate the marker cell whose color is "
        f"one of {_color_list(m, m.get('marker_colors', []))}. Use {_association_relation(m)}. "
        "Recolor each object to its associated marker's color. Leave marker and background cells unchanged."
    )


@register_prompt("226")
def _p226(m):
    return (
        f"The divider cells of color {_ci(m, 'divider_color')} split the grid into four regions. "
        f"Assign each object of color {_ci(m, 'object_color')} to a region by {_MEDOID_DEFINITION}. "
        f"Apply the region mapping: {_mapping(m, 'region_to_transform', label_key='region')}. "
        "Leave divider and background cells unchanged."
    )


@register_prompt("232")
def _p232(m):
    return (
        "The input contains named grids input_grid and output_grid. In input_grid, the selected "
        f"input object has color {_ci(m, 'selected_color')}. Label input objects input1, input2, ... "
        "and output objects output1, output2, ... by top-left bounding-box order. Match objects by "
        f"{_relation(m)}. Return a copy of output_grid with the corresponding output object recolored "
        f"to highlight color {_ci(m, 'highlight_color')}."
    )


@register_prompt("233")
def _p233(m):
    return (
        "The named grids input_grid and output_grid contain the same number of objects. Label input "
        "objects input1, input2, ... and output objects output1, output2, ... by top-left bounding-box "
        f"order. Match each input object to the output object with {_relation(m)}. Return only the "
        "semicolon-separated correspondence, for example input1->output2;input2->output1."
    )


@register_prompt("234")
def _p234(m):
    return (
        "The named grids input_grid and output_grid contain equivalent objects. Label input objects "
        "input1, input2, ... and output objects output1, output2, ... by top-left bounding-box order. "
        f"Use the matching rule {_relation(m)} and return only the semicolon-separated correspondence."
    )


@register_prompt("235")
def _p235(m):
    return (
        "The named grids input_grid and output_grid may have different object counts. Label input "
        "objects input1, input2, ... and output objects output1, output2, ... by top-left bounding-box "
        f"order. Compare objects by {_relation(m)}. Classify each input as matched when exactly one "
        "output matches it, deleted when none match it, and copied when multiple outputs match it; "
        "classify any output with no input match as new. Return exactly "
        "matched=...;deleted=...;copied=...;new=... with these per-section formats: the matched "
        "section lists comma-separated inputI->outputJ pairs (one pair per matched input); the "
        "copied section lists comma-separated inputI->outputJ pairs, repeating the same input "
        "label once for each output it matches; the deleted section lists comma-separated bare "
        "input labels (inputI); the new section lists comma-separated bare output labels (outputJ). "
        "Order the labels in each section by ascending label number, and leave a section empty after "
        "its '=' if it has no members. For example: "
        "matched=input2->output5,input4->output1;deleted=input3;"
        "copied=input1->output2,input1->output3;new=output4."
    )
