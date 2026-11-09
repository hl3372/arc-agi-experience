"""Prompt templates for core2 batch L rule-inference tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _labels(m: dict) -> str:
    return ", ".join(str(label) for label in m.get("candidate_labels", []))


@register_prompt("211")
def _p211(m):
    return (
        f"Compare the before and after grids. Source cells have color {_ci(m, 'source_color')}, "
        f"wall cells have color {_ci(m, 'wall_color')}, and drawn or hit ray cells use color "
        f"{_ci(m, 'ray_color')}. Infer the ray direction, draw-versus-hit mode, and stopping "
        f"condition. Return only one label from: {_labels(m)}."
    )


@register_prompt("212")
def _p212(m):
    return (
        f"The input shows source_a with foreground color {_ci(m, 'source_a_color')}, source_b with "
        f"foreground color {_ci(m, 'source_b_color')}, and their composite. Some composite rules may "
        f"use result color {_ci(m, 'result_color')}. Infer the overlay rule and return only one "
        f"label from: {_labels(m)}."
    )


@register_prompt("213")
def _p213(m):
    return (
        f"Compare the marker-relative before and after pair. Marker cells use color "
        f"{_ci(m, 'marker_color')}; a movable source object, when present, uses color "
        f"{_ci(m, 'object_color')}; stamp marks use color {_ci(m, 'stamp_color')}. Infer whether "
        f"the rule copies, moves, or stamps, and which marker landmark is used. Return only one "
        f"label from: {_labels(m)}."
    )


@register_prompt("214")
def _p214(m):
    return (
        f"Compare before and after. Background-like filler may include color {_ci(m, 'background_color')}, "
        "but the decision is the geometric array operation: scaling, tiling, row repetition, or "
        f"block compression. Return only one label from: {_labels(m)}."
    )


@register_prompt("215")
def _p215(m):
    return (
        f"The source grid contains selector cues: target color {_ci(m, 'target_color')}, frame color "
        f"{_ci(m, 'frame_color')}, and marker color {_ci(m, 'marker_color')}. The extracted grid is "
        f"a subgrid of the source. Infer the extraction rule and return only one label from: {_labels(m)}."
    )


@register_prompt("216")
def _p216(m):
    return (
        f"Treat non-background cells in source_a and source_b as foreground masks over background color "
        f"{_ci(m, 'background_color')}. source_a uses {_ci(m, 'source_a_color')}, source_b uses "
        f"{_ci(m, 'source_b_color')}, and the combined mask uses {_ci(m, 'result_color')}. Infer the "
        f"mask operation and return only one label from: {_labels(m)}."
    )


@register_prompt("217")
def _p217(m):
    return (
        f"All objects begin in source object color {_ci(m, 'object_color')} on background color "
        f"{_ci(m, 'background_color')}. Compare how whole objects are recolored in the after grid. "
        f"Infer which object property controls the property-to-color mapping. Return only one label "
        f"from: {_labels(m)}."
    )


@register_prompt("218")
def _p218(m):
    return (
        f"Compare the before and after foreground mask. Foreground has color {_ci(m, 'foreground_color')} "
        f"on background color {_ci(m, 'background_color')}, using {_m(m, 'connectivity')} connected "
        f"neighbors for the shape operation. Infer the morphological operation and parameter. Return "
        f"only one label from: {_labels(m)}."
    )


@register_prompt("219")
def _p219(m):
    return (
        f"Compare the background regions before and after. Background is color {_ci(m, 'background_color')}, "
        f"walls are color {_ci(m, 'wall_color')}, marker cues use {_ci(m, 'marker_color')}, and selected "
        f"regions are recolored to {_ci(m, 'fill_color')}. Infer the background region-selection rule. "
        f"Return only one label from: {_labels(m)}."
    )


@register_prompt("220")
def _p220(m):
    return (
        f"Compare the object arrangement before and after on background color {_ci(m, 'background_color')}. "
        f"Objects are moved without changing their shapes or colors and packed with gap {_m(m, 'packing_gap')}. "
        f"Infer the ordering property and horizontal or vertical packing layout. Return only one label "
        f"from: {_labels(m)}."
    )
