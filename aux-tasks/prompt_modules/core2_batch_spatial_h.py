"""Prompt templates for core2 batch spatial H tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt, render_pattern_grid


def _color_value(meta: dict, value: int) -> str:
    wrapped = dict(meta)
    wrapped["__tmp_color"] = value
    return _ci(wrapped, "__tmp_color")


_ROTATION_PHRASE = {
    "rot0": "the prototype with no rotation (0 degrees)",
    "rot90": "the prototype rotated 90 degrees clockwise",
    "rot180": "the prototype rotated 180 degrees",
    "rot270": "the prototype rotated 270 degrees clockwise",
}


def _rotation_phrase(transform: str) -> str:
    return _ROTATION_PHRASE.get(transform, str(transform).replace("_", " "))


def _orientation_map(meta: dict) -> str:
    parts = []
    for item in meta.get("orientation_color_map", []):
        parts.append(
            f"{_rotation_phrase(item['transform'])} -> color {_color_value(meta, item['color'])}"
        )
    return "; ".join(parts)


def _motif_map(meta: dict) -> str:
    palette = meta.get("__color_palette__")
    parts = []
    for item in meta.get("motif_color_map", []):
        motif = render_pattern_grid(item["motif"], color_palette=palette, row_sep="//", compact=True)
        parts.append(f"{motif} -> color {_color_value(meta, item['color'])}")
    return "; ".join(parts)


def _depth_map(meta: dict) -> str:
    parts = []
    for item in meta.get("depth_color_map", []):
        parts.append(f"depth {item['depth']} -> color {_color_value(meta, item['color'])}")
    return "; ".join(parts)


def _marker_colors(meta: dict) -> str:
    return ", ".join(_color_value(meta, c) for c in meta.get("marker_colors", []))


def _source_colors(meta: dict) -> str:
    return ", ".join(_color_value(meta, c) for c in meta.get("source_colors", []))


@register_prompt("178")
def _p178(m):
    return (
        f"The color {_ci(m, 'prototype_color')} shape is the asymmetric prototype. "
        f"Each color {_ci(m, 'target_object_color')} 4-connected target object is the prototype rotated by one of "
        "0, 90, 180, or 270 degrees clockwise (all rotations are clockwise; the prototype is deliberately "
        "asymmetric, so exactly one of these four rotations reproduces each target object's exact shape). "
        f"Recolor each target object by its matching rotation using this legend: {_orientation_map(m)}. "
        "Leave the prototype and background unchanged."
    )


@register_prompt("183")
def _p183(m):
    return (
        f"The color {_ci(m, 'path_color')} cells form one 4-connected, one-cell-wide, non-branching path. "
        f"Recolor every turn cell with color {_ci(m, 'turn_color')}. A turn cell is a path cell whose two path "
        "neighbors lie in perpendicular directions. Leave the background and any other-color cells unchanged."
    )


@register_prompt("184")
def _p184(m):
    return (
        f"Find the 4-connected color {_ci(m, 'target_object_color')} target object, ignoring other-color objects. Within its bounding box, "
        f"recolor every background color {_ci(m, 'background_color')} cell that is not part of the object "
        f"with color {_ci(m, 'gap_color')}. Preserve the object cells, other-color objects, and everything outside the bounding box."
    )


@register_prompt("185")
def _p185(m):
    return (
        f"For each background color {_ci(m, 'background_color')} cell, find the nearest source cell among colors "
        f"{_source_colors(m)} using {_m(m, 'metric')} distance. Recolor the background cell with the nearest "
        "source's color. If two or more sources tie for nearest, leave that background cell unchanged; source "
        "cells keep their colors and any other-color cells are left unchanged."
    )


@register_prompt("186")
def _p186(m):
    return (
        f"From the color {_ci(m, 'source_color')} source cell, cast rays in these directions: "
        f"{', '.join(m.get('directions', []))}. Along each ray, recolor only the first color "
        f"{_ci(m, 'hit_color')} hit cell with color {_ci(m, 'mark_color')}. Rays with no hit leave the grid unchanged."
    )


@register_prompt("187")
def _p187(m):
    return (
        f"The color {_ci(m, 'line_color')} cells form a one-cell-wide 4-connected line network. "
        f"Recolor endpoint cells, with exactly one 4-neighbor in the network, using color "
        f"{_ci(m, 'endpoint_marker_color')}. Recolor branchpoint cells, with three or more 4-neighbors, "
        f"using color {_ci(m, 'branchpoint_marker_color')}. Leave the background and any other-color cells unchanged."
    )


@register_prompt("190")
def _p190(m):
    return (
        f"Partition the whole input grid into {_m(m, 'block_height')}x{_m(m, 'block_width')} blocks from the "
        f"top-left corner. Replace every block by a solid block whose color comes from this motif legend: "
        f"{_motif_map(m)}."
    )


@register_prompt("191")
def _p191(m):
    return (
        f"The color {_ci(m, 'object_color')} objects are nested one-cell-thick rectangular frames. "
        "For each object, compute its containment depth as the number of other object bounding boxes that "
        f"strictly contain its bounding box, then recolor the object using this mapping: {_depth_map(m)}. "
        "Leave background unchanged."
    )


@register_prompt("192")
def _p192(m):
    return (
        f"Treat color {_ci(m, 'wall_color')} cells as walls. The non-wall cells form 4-connected regions containing "
        f"background color {_ci(m, 'background_color')} cells and possible marker cells of colors {_marker_colors(m)}. "
        "For each region containing exactly one marker cell, fill only its background cells with that marker cell's color. "
        "Leave regions with zero or two-or-more marker cells unchanged."
    )


@register_prompt("196")
def _p196(m):
    return (
        f"The color {_ci(m, 'divider_color')} cells form one straight {_m(m, 'divider_orientation')} divider line. "
        f"Recolor background color {_ci(m, 'background_color')} cells strictly on the {_m(m, 'side')} side of the "
        f"divider with color {_ci(m, 'side_color')}. Preserve divider cells and all non-background cells."
    )
