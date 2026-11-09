"""Prompt templates for correspondence/change batch O tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _render_color(meta: dict, value: int) -> str:
    m = dict(meta)
    m["_render_color"] = value
    return _ci(m, "_render_color")


def _labels(meta: dict, key: str) -> str:
    return ", ".join(str(label) for label in meta.get(key, []))


def _correspondence(meta: dict) -> str:
    inputs = meta.get("input_labels", [])
    outputs = meta.get("output_labels", [])
    return ", ".join(f"{a}->{b}" for a, b in zip(inputs, outputs))


def _scene_intro(meta: dict, *, include_correspondence_pairs: bool = True) -> str:
    top = int(meta.get("object_top_row", 3))
    intro = (
        "The input has named grids input_scene and output_scene. Ignore background color "
        f"{_ci(meta, 'background_color')}, divider color {_ci(meta, 'divider_color')}, "
        f"and the top label-marker color {_ci(meta, 'label_marker_color')} when comparing objects. "
        f"Each slot holds exactly one object, located in rows from index {top} downward; any "
        f"other colored cells above that (a decorative top band) are not objects and must be ignored. "
        f"Input objects are labelled {_labels(meta, 'input_labels')} from left to right by their "
        f"marked slots; output objects are labelled {_labels(meta, 'output_labels')} from left to right. "
        "Correspond by same slot order."
    )
    if include_correspondence_pairs:
        intro += f" The full correspondence is {_correspondence(meta)}."
    return intro


def _property_list(meta: dict) -> str:
    return ", ".join(str(label) for label in meta.get("property_labels", []))


def _changed_property_list(meta: dict) -> str:
    return ", ".join(str(label) for label in meta.get("changed_property_labels", []))


def _global_property_list(meta: dict) -> str:
    return ", ".join(str(label) for label in meta.get("global_property_labels", []))


def _category_mapping(meta: dict) -> str:
    return "; ".join(
        f"{entry['category']} -> color {_render_color(meta, int(entry['color']))}"
        for entry in meta.get("category_to_color", [])
    )


def _relation_definition(candidate: str) -> str:
    return {
        "left_to_right_order": "the left-to-right order of corresponding objects is unchanged",
        "top_row_order": "every pair of corresponding objects keeps the same top-row ordering",
        "cell_count_order": "every pair of corresponding objects keeps the same object-size ordering",
        "color_equality_pattern": "pairs that had equal colors before have equal colors after, and unequal pairs stay unequal",
    }[candidate]


@register_prompt("236")
def _p236(m):
    return (
        f"{_scene_intro(m)} Classify each input object as unchanged if its corresponding output "
        "object has the same color, shape, size, and position; otherwise classify it as changed. "
        "Return labels in input-object order as A:changed;B:unchanged with no extra text."
    )


@register_prompt("237")
def _p237(m):
    labels = ", ".join(str(label) for label in m.get("change_labels", []))
    return (
        f"{_scene_intro(m)} For each input object, identify the one-to-one change type. "
        f"Use only these labels when applicable: {labels}. Return input-object order as "
        "A:translated;B:recolored with no extra text."
    )


@register_prompt("238")
def _p238(m):
    return (
        f"{_scene_intro(m, include_correspondence_pairs=False)} The selected input object is {_m(m, 'selected_input_label')}. "
        "Identify its corresponding output label and the parameterized change. Use descriptors "
        "unchanged, translated(dr=R,dc=C), recolored(to_output_color), or scaled_2x(factor=2). "
        f"Return exactly {_m(m, 'selected_input_label')}->{_m(m, 'selected_output_label')}:descriptor for the selected object."
    )


@register_prompt("239")
def _p239(m):
    return (
        f"{_scene_intro(m)} Return a same-size grid preserving the divider and top label-marker "
        f"cells. Recolor object cells by lifecycle/change category using: {_category_mapping(m)}. "
        "Use the changed category at the output object's cells for translated, recolored, and "
        "scaled_2x correspondences; use deleted at the input object's old cells; use new at the "
        f"output object's cells. Leave all other cells (including any decorative top-band cells) "
        f"as background color {_ci(m, 'background_color')}."
    )


@register_prompt("240")
def _p240(m):
    return (
        f"{_scene_intro(m, include_correspondence_pairs=False)} The selected input object is {_m(m, 'selected_input_label')}. "
        f"Compare it with its corresponding output object. Position means the top-left cell of the "
        f"object's bounding box. Candidate preserved-property labels are: "
        f"{_property_list(m)}. Candidate changed-property labels are: {_changed_property_list(m)}. "
        f"Return exactly {_m(m, 'selected_input_label')}->{_m(m, 'selected_output_label')}:preserved=label,label;"
        "changed=label,label, using none for an empty side."
    )


@register_prompt("241")
def _p241(m):
    return (
        f"{_scene_intro(m)} Identify which property labels are preserved for every corresponding "
        f"object pair. Position means the top-left cell of the object's bounding box. Candidate "
        f"labels, in output order, are: {_property_list(m)}. Return only the "
        "semicolon-separated preserved labels in that order, or none."
    )


@register_prompt("242")
def _p242(m):
    return (
        f"{_scene_intro(m)} The output_scene may use a different uniform background fill color or a "
        "different overall grid size than input_scene; treat whatever color uniformly fills the empty "
        "area (everything other than label-marker cells, divider lines, objects, and any decorative "
        "top-band cells) as that scene's background. Identify which global scene-property labels are preserved from input_scene to "
        "output_scene. background_color_preserved means the background fill color is the same in "
        "both scenes; grid_size_preserved means both scenes have the same number of rows and "
        f"columns. Candidate labels, in output order, are: {_global_property_list(m)}. "
        "Return only the semicolon-separated preserved labels in that order, or none."
    )


@register_prompt("243")
def _p243(m):
    candidate = _m(m, "candidate_relation")
    return (
        f"{_scene_intro(m)} Test candidate relational property {candidate}: "
        f"{_relation_definition(candidate)}. Return exactly relation_preserved or "
        "relation_not_preserved."
    )
