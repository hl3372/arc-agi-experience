"""Prompt templates for core2 batch J fit and candidate tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _transforms(m: dict) -> str:
    return ", ".join(str(t).replace("_", " ") for t in m.get("allowed_transforms", []))


def _unselected_clause(m: dict) -> str:
    """Pin down output handling of the source object and the unselected scaffolding.

    The placed object is stamped at exactly one selected anchor; every other marked
    cell in the scene (the other candidate anchor markers, the guide/contact cell,
    and any distractor cells) is left exactly as in the input. In move mode the
    object's original cells are cleared to the background; in copy mode they stay.
    """
    if _m(m, "copy_move_mode") == "move":
        source = (
            f"Clear the object's original cells to the background color {_ci(m, 'background_color')}. "
        )
    else:
        source = "Keep the object at its original location. "
    return (
        source
        + "Apply only this single placement: leave every other cell unchanged, including all "
        "candidate anchor markers you did not select and every other marked or distractor cell."
    )


def _label_text(m: dict) -> str:
    parts = []
    for item in m.get("candidate_labels", []):
        key = f"_label_{item['label']}_color"
        m[key] = item["marker_color"]
        parts.append(f"{item['label']} is marked by {_ci(m, key)}")
    return "; ".join(parts)


@register_prompt("230")
def _p230(m):
    action = (
        "recolor to " + _ci(m, "action_color")
        if _m(m, "action_type") == "recolor"
        else f"erase to the plain background color {_ci(m, 'background_color')} "
        "(set each such cell to the plain background color, even where the background carries a pattern; "
        "do not extend or restore any background pattern into the erased cell)"
    )
    return (
        f"Count cells in set A color {_ci(m, 'set_a_color')} and set B color {_ci(m, 'set_b_color')}. "
        f"Select the {_m(m, 'target_category').replace('_', ' ')} category; if that category is empty, "
        f"use fallback category {_m(m, 'fallback_category').replace('_', ' ')}. "
        f"Apply the action: {action}. Leave all unselected cells unchanged."
    )


@register_prompt("231")
def _p231(m):
    form = _m(m, "output_form")
    axis = _m(m, "axis")
    if form == "summary_strip":
        shape = "Return a one-column summary strip with one cell per input row." if axis == "row" else "Return a one-row summary strip with one cell per input column."
    elif form == "paint_full_lines":
        shape = f"Return a same-size grid where each full {axis} is painted with its comparison result color."
    else:
        start = "leftmost cell of its row" if axis == "row" else "top cell of its column"
        shape = (
            f"Return a same-size grid with each result color written only at the {start}, "
            f"and fill every other cell with the plain background color {_ci(m, 'background_color')} "
            "(do not reproduce any background pattern from the input)."
        )
    return (
        f"For each {axis}, count color {_ci(m, 'set_a_color')} cells and color "
        f"{_ci(m, 'set_b_color')} cells. If A is greater use {_ci(m, 'greater_color')}; "
        f"if equal use {_ci(m, 'equal_color')}; if less use {_ci(m, 'less_color')}. "
        f"{shape}"
    )


@register_prompt("249")
def _p249(m):
    return (
        f"{_m(m, 'copy_move_mode').capitalize()} the movable object of color {_ci(m, 'object_color')} "
        f"to the unique candidate anchor marked {_ci(m, 'anchor_color')} where it fits. "
        f"Allowed transforms: {_transforms(m)}. Anchor landmark: {_m(m, 'anchor_landmark')}. "
        f"Fit rule: {_m(m, 'fit_constraint')}. Overwrite/collision behavior: {_m(m, 'overwrite_behavior')}. "
        f"{_unselected_clause(m)}"
    )


@register_prompt("250")
def _p250(m):
    if _m(m, "region_type") == "mask_region":
        region_desc = f"unique mask region made of {_ci(m, 'mask_color')} cells"
    else:
        region_desc = (
            f"unique {_m(m, 'region_type').replace('_', ' ')} marked by {_ci(m, 'mask_color')} "
            f"cells and enclosed by frame color {_ci(m, 'frame_color')}"
        )
    return (
        f"{_m(m, 'copy_move_mode').capitalize()} the movable object of color {_ci(m, 'object_color')} "
        f"into the {region_desc}. "
        f"Allowed transforms: {_transforms(m)}. Fit rule: {_m(m, 'fit_constraint')}. "
        f"Overwrite/collision behavior: {_m(m, 'overwrite_behavior')}."
    )


@register_prompt("251")
def _p251(m):
    return (
        f"{_m(m, 'copy_move_mode').capitalize()} the movable object of color {_ci(m, 'object_color')} "
        f"to the unique candidate anchor marked {_ci(m, 'anchor_color')} satisfying alignment constraint "
        f"{_m(m, 'alignment_constraint').replace('_', ' ')} with the guide cell {_ci(m, 'guide_color')}. "
        f"Allowed transforms: {_transforms(m)}. Anchor landmark: {_m(m, 'anchor_landmark')}. "
        f"Overwrite/collision behavior: {_m(m, 'overwrite_behavior')}. "
        f"{_unselected_clause(m)}"
    )


@register_prompt("252")
def _p252(m):
    return (
        f"{_m(m, 'copy_move_mode').capitalize()} the movable object of color {_ci(m, 'object_color')} "
        f"to the unique candidate anchor marked {_ci(m, 'anchor_color')} satisfying this contact constraint: "
        f"{_m(m, 'contact_constraint')}, where the contact object has color {_ci(m, 'contact_color')}. "
        f"Allowed transforms: {_transforms(m)}. Anchor landmark: {_m(m, 'anchor_landmark')}. "
        f"Overwrite/collision behavior: {_m(m, 'overwrite_behavior')}. "
        f"{_unselected_clause(m)}"
    )


@register_prompt("253")
def _p253(m):
    return (
        f"Use the reference object of color {_ci(m, 'reference_color')} and choose the labeled candidate "
        f"object of color {_ci(m, 'candidate_color')} that matches by relation "
        f"{_m(m, 'equivalence_relation').replace('_', ' ')}. Candidate labels: {_label_text(m)}. "
        "Return only the label A, B, or C."
    )


@register_prompt("254")
def _p254(m):
    return (
        f"Compare the reference patch with the labeled candidate regions using match relation "
        f"{_m(m, 'match_relation').replace('_', ' ')}. Candidate labels: {_label_text(m)}. "
        "Return only the matching label A, B, or C."
    )


@register_prompt("255")
def _p255(m):
    relation = _m(m, "spatial_relation")
    metric = ""
    if relation in ("nearest_to_reference", "farthest_from_reference"):
        metric = "Distances are measured in Manhattan (row+column) steps. "
    return (
        f"Use the reference marker color {_ci(m, 'reference_color')} and choose the labeled candidate "
        f"cell of color {_ci(m, 'candidate_color')} satisfying spatial relation "
        f"{relation.replace('_', ' ')}. {metric}Only cells of the candidate color are candidates; "
        f"ignore any other scattered cells. Candidate labels: {_label_text(m)}. "
        "Return only the label A, B, or C."
    )


@register_prompt("256")
def _p256(m):
    return (
        f"Use the target slot made of {_ci(m, 'target_color')} cells and choose the labeled piece "
        f"of color {_ci(m, 'piece_color')} that exactly covers it under allowed transforms: {_transforms(m)}. "
        f"Candidate labels: {_label_text(m)}. Return only the label A, B, or C."
    )
