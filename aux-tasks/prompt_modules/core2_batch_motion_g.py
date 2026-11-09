"""Prompt templates for motion and multi-grid second-batch slice G."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _transform_phrase(name: str) -> str:
    return {
        "rotate_cw": "rotate the whole grid 90 degrees clockwise",
        "rotate_ccw": "rotate the whole grid 90 degrees counterclockwise",
        "flip_horizontal": "flip the whole grid left-right",
        "flip_vertical": "flip the whole grid top-bottom",
        "transpose_main": "transpose across the main diagonal",
        "transpose_anti": "transpose across the anti-diagonal",
    }.get(name, name.replace("_", " "))


def _sequence(meta: dict, key: str) -> str:
    return "; then ".join(_transform_phrase(str(name)) for name in meta.get(key, []))


def _landmark_phrase(meta: dict) -> str:
    return _m(meta, "landmark").replace("_", "-")


@register_prompt("128")
def _p128(m):
    return (
        f"The cells of color {_ci(m, 'slot_marker_color')} mark slots in a "
        f"{_m(m, 'cycle_direction')} visible cycle. Move each 2 by 2 payload block "
        "from its current marked slot to the next marked slot in that cycle, wrapping "
        "from the last slot to the first. Keep the slot marker cells unchanged and "
        f"turn the old payload cells into background color {_ci(m, 'background_color')}."
    )


@register_prompt("129")
def _p129(m):
    return (
        "The input grid was produced from an original square grid by this sequence: "
        f"{_sequence(m, 'applied_sequence')}. Recover the original by undoing those "
        "transformations in reverse order."
    )


@register_prompt("148")
def _p148(m):
    rule = _m(m, "overlay_rule")
    if rule == "top_priority":
        collision = "when both grids have foreground at a cell, use the top grid's color"
    elif rule == "bottom_priority":
        collision = "when both grids have foreground at a cell, use the bottom grid's color"
    else:
        collision = f"when both grids have foreground at a cell, use collision color {_ci(m, 'collision_color')}"
    return (
        f"Overlay the named grids bottom and top. Background is color {_ci(m, 'background_color')}; "
        f"any other cell is foreground. If exactly one grid has foreground, use that cell. {collision}. "
        "If neither grid has foreground, output background."
    )


@register_prompt("175")
def _p175(m):
    op = "copy" if _m(m, "operation") == "copy" else "move"
    return (
        f"Find the offset vector from marker A color {_ci(m, 'marker_a_color')} to marker B color "
        f"{_ci(m, 'marker_b_color')}. Apply that same offset to every object of target color "
        f"{_ci(m, 'target_color')}: {op} each target object by the vector. Marker cells remain unchanged; "
        "the generated shifted targets fit on background cells."
    )


@register_prompt("176")
def _p176(m):
    return (
        f"Move the target object of color {_ci(m, 'target_color')} so its "
        f"{_landmark_phrase(m)} landmark aligns with the anchor marker cell of color "
        f"{_ci(m, 'anchor_color')}. Clear the object's old cells to background color "
        f"{_ci(m, 'background_color')}; the moved object overwrites the anchor marker."
    )


@register_prompt("179")
def _p179(m):
    if _m(m, "copy_mode") == "frame":
        return (
            f"Find the source frame of color {_ci(m, 'source_frame_color')} and the same-size "
            f"target frame of color {_ci(m, 'target_frame_color')}. Copy the source frame's "
            "interior contents into the target frame interior. Keep both frame outlines unchanged."
        )
    return (
        f"The corner cells of source marker color {_ci(m, 'source_frame_color')} and target "
        f"marker color {_ci(m, 'target_frame_color')} identify same-size rectangular regions. "
        "Copy the full source rectangle, including its corner cells, into the target rectangle."
    )


@register_prompt("188")
def _p188(m):
    cond = _m(m, "mask_condition")
    if cond == "mask_equals_color":
        phrase = f"where the mask cell is color {_ci(m, 'mask_color')}"
    elif cond == "mask_non_background":
        phrase = f"where the mask cell is not mask background color {_ci(m, 'mask_background_color')}"
    else:
        phrase = f"where the mask cell is not color {_ci(m, 'mask_color')}"
    return (
        f"Use the same-sized source and mask grids. Output the source cell {phrase}; "
        f"otherwise output fill color {_ci(m, 'fill_color')}."
    )


@register_prompt("189")
def _p189(m):
    rule = _m(m, "consensus_rule")
    if rule == "unanimous_non_background":
        text = f"use a color only when all grids agree on the same non-background color; otherwise use background color {_ci(m, 'background_color')}"
    elif rule == "majority_non_background":
        text = f"use a non-background color only when it appears in more than half of the grids; otherwise use background color {_ci(m, 'background_color')}"
    else:
        text = (
            f"count every grid's value including the background color {_ci(m, 'background_color')}, "
            f"then use the unique most frequent value; if there is a tie for most frequent, use tie color {_ci(m, 'tie_color')}"
        )
    return f"Combine the {_m(m, 'grid_count')} same-sized named grids cell by cell using this consensus rule: {text}."


@register_prompt("194")
def _p194(m):
    return (
        f"Treat every non-background cell except the anchor marker color {_ci(m, 'anchor_color')} "
        f"as one scene. Translate the whole scene so its {_landmark_phrase(m)} landmark aligns "
        f"with the anchor marker cell. The translated scene overwrites the anchor marker; all old "
        f"scene cells become background color {_ci(m, 'background_color')}."
    )


@register_prompt("195")
def _p195(m):
    return (
        f"Use the single non-marker prototype pattern and copy it to every marker cell of color "
        f"{_ci(m, 'marker_color')}. Align the prototype's {_landmark_phrase(m)} landmark to each "
        "marker cell. Keep the original prototype and overwrite marker cells with the copied pattern."
    )
