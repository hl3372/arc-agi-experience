"""Prompt templates for core2_extra region and boundary tasks."""
from __future__ import annotations

from prompts import register_prompt, _ci, _m


def _color_list(meta: dict, key: str) -> str:
    colors = meta.get(key, [])
    labels = [_ci({**meta, "_tmp_color": int(color)}, "_tmp_color") for color in colors]
    if not labels:
        return ""
    if len(labels) == 1:
        return f"color {labels[0]}"
    return "colors " + ", ".join(labels)


def _connectivity_phrase(meta: dict) -> str:
    connectivity = int(_m(meta, "connectivity"))
    if connectivity == 8:
        return "8-connectivity, where one step may go to any edge-neighbor or diagonal corner-neighbor"
    return "4-connectivity, where one step may go only to a row/column edge-neighbor"


@register_prompt("111")
def _p111(m):
    return (
        "Create a same-size output grid. "
        f"Start from the unique seed cell of color {_ci(m, 'seed_color')}. Using "
        f"{_connectivity_phrase(m)}, move only through background color "
        f"{_ci(m, 'background_color')} cells and the seed cell. Wall color "
        f"{_ci(m, 'wall_color')} and distractor cells of {_color_list(m, 'distractor_colors')} "
        "are barriers: do not enter, cross, or recolor them. Recolor every reachable "
        f"fillable cell, including the seed, to color {_ci(m, 'fill_color')}; leave "
        "unreachable cells and all non-fillable cells unchanged."
    )


@register_prompt("113")
def _p113(m):
    return (
        "Create a same-size output grid. "
        f"Find every cell of target color {_ci(m, 'target_color')}; those target cells have "
        f"distance 0. Expand simultaneously up to distance {_m(m, 'distance')} using "
        f"{_connectivity_phrase(m)}. Protected {_color_list(m, 'protected_colors')} are "
        "barriers: never enter, cross, or overwrite them. Every reached non-protected cell, "
        f"including background cells and non-protected distractor cells of {_color_list(m, 'distractor_colors')}, "
        f"becomes target color {_ci(m, 'target_color')}. Leave unreached cells unchanged."
    )


@register_prompt("114")
def _p114(m):
    condition = "area is at most" if _m(m, "area_condition") == "at_most" else "area is at least"
    return (
        "Create a same-size output grid. "
        f"Treat background color {_ci(m, 'background_color')} as empty space. Using "
        f"{_connectivity_phrase(m)}, find background components that are not reachable from "
        "the grid border. A component is a target-enclosed hole only if every non-background "
        f"edge-neighbor touching it has target color {_ci(m, 'target_color')}; ignore holes "
        f"enclosed by distractor cells of {_color_list(m, 'distractor_colors')} or any other color. "
        f"Measure a hole's area by its number of cells. Fill every target-enclosed hole whose "
        f"{condition} {_m(m, 'area_threshold')} cells with color {_ci(m, 'fill_color')}. "
        "Leave exterior background, non-qualifying holes, target cells, and other colors unchanged."
    )


@register_prompt("124")
def _p124(m):
    if _m(m, "negative_space_mode") == "inside_frame":
        rule = (
            f"the enclosed background shape inside the foreground color "
            f"{_ci(m, 'foreground_color')} frame"
        )
    else:
        rule = (
            f"the background shape strictly between the two foreground color "
            f"{_ci(m, 'foreground_color')} objects on their shared rows"
        )
    extra = ""
    if _m(m, "distractor_colors"):
        extra = (
            f" Ignore any other {_color_list(m, 'distractor_colors')} objects; they are "
            "not part of the negative space."
        )
    return (
        f"Extract {rule}.{extra} Output a same-size grid with that negative-space shape in color "
        f"{_ci(m, 'output_color')} and all other cells as background color "
        f"{_ci(m, 'background_color')}."
    )


@register_prompt("125")
def _p125(m):
    extra = ""
    if _m(m, "distractor_colors"):
        extra = (
            f" Consider only the color {_ci(m, 'target_color')} object; ignore any "
            f"{_color_list(m, 'distractor_colors')} objects."
        )
    return (
        f"In the color {_ci(m, 'target_color')} object, using {_m(m, 'connectivity')}-connectivity, "
        f"mark every connector cell whose removal would split the object into multiple disconnected "
        f"components.{extra} Recolor connector cells to color {_ci(m, 'connector_color')}; "
        "leave all other cells unchanged."
    )


@register_prompt("126")
def _p126(m):
    return (
        f"Consider the 4-connected cell-edge boundary of all color {_ci(m, 'target_color')} cells. "
        f"Recolor target cells at outward 90-degree boundary corners to color "
        f"{_ci(m, 'outer_corner_color')} and target cells at inward notch corners to color "
        f"{_ci(m, 'inner_corner_color')}. If a cell qualifies as both, use the inward notch color. "
        "Leave non-corner cells unchanged."
    )


@register_prompt("151")
def _p151(m):
    return (
        f"Treat {_color_list(m, 'barrier_colors')} as barriers. For background color "
        f"{_ci(m, 'background_color')} cells, use {_m(m, 'connectivity')}-connectivity to find those "
        f"reachable from the grid border without crossing barriers. Recolor reachable background cells "
        f"to color {_ci(m, 'outside_color')} and all other background cells to color "
        f"{_ci(m, 'inside_color')}; leave barriers unchanged."
    )


@register_prompt("156")
def _p156(m):
    return (
        f"Find each enclosed 4-connected background color {_ci(m, 'background_color')} hole. "
        "If all object cells touching that hole have a single shared color, fill the entire hole "
        "with that surrounding object color. Leave holes touching multiple object colors unchanged, "
        "and leave the surrounding objects and exterior background unchanged."
    )


@register_prompt("167")
def _p167(m):
    extra = ""
    if _m(m, "distractor_colors"):
        extra = (
            f" Use only the color {_ci(m, 'target_color')} object; ignore any "
            f"{_color_list(m, 'distractor_colors')} objects."
        )
    return (
        f"For the color {_ci(m, 'target_color')} object, compute boundary layers using "
        f"{_m(m, 'connectivity')}-connectivity by repeatedly removing current boundary cells. "
        f"Recolor cells in layer {_m(m, 'layer_index')} to color {_ci(m, 'layer_color')}; "
        f"leave all other cells unchanged.{extra}"
    )


@register_prompt("173")
def _p173(m):
    condition = "exactly" if _m(m, "condition") == "exactly" else "at least"
    neighbor_count = int(_m(m, "neighbor_count"))
    neighbor_word = "neighbor" if neighbor_count == 1 else "neighbors"
    return (
        f"Consider only eligible cells of color {_ci(m, 'eligible_color')}. Count each eligible cell's "
        f"{_m(m, 'connectivity')}-neighbors that have color {_ci(m, 'neighbor_color')}. "
        f"Simultaneously recolor every eligible cell with {condition} {neighbor_count} "
        f"such {neighbor_word} to color {_ci(m, 'marker_color')}; leave all other cells unchanged."
    )
