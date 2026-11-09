"""Prompt templates for core2 batch K finite inference tasks."""
from __future__ import annotations

from prompts import _ci, _m, register_prompt


def _allowed(m: dict) -> str:
    return ", ".join(str(label) for label in m.get("allowed_labels", []))


def _mapping_pair(m: dict, pair: dict) -> str:
    local = {
        **m,
        "_from_color": pair["from_color"],
        "_to_color": pair["to_color"],
    }
    return f"{_ci(local, '_from_color')} -> {_ci(local, '_to_color')}"


def _mapping_options(m: dict) -> str:
    parts = []
    for item in m.get("candidate_mappings", []):
        pairs = ", ".join(_mapping_pair(m, pair) for pair in item.get("pairs", []))
        parts.append(f"{item['label']}: {pairs}")
    return "; ".join(parts)


@register_prompt("121")
def _p121(m):
    return (
        "The named grids show an input grid and its transformed output grid. "
        "Classify the whole-grid transformation. "
        f"Return exactly one label from this allowed set: {_allowed(m)}."
    )


@register_prompt("122")
def _p122(m):
    return (
        f"Apply the stated whole-grid rule {_m(m, 'rule_label')} to the named input grid. "
        "Choose the candidate output grid that exactly matches the rule result. "
        f"Return exactly one candidate label from: {_allowed(m)}."
    )


@register_prompt("123")
def _p123(m):
    return (
        f"Apply the stated whole-grid rule {_m(m, 'rule_label')} to the named input grid. "
        "Decide whether the named candidate grid exactly matches that result. "
        "Return exactly true or false."
    )


@register_prompt("204")
def _p204(m):
    return (
        "Compare the named input and output grids. Infer which listed mapping accounts for "
        "only the colors that changed; all unlisted colors stay unchanged. "
        f"Mapping choices: {_mapping_options(m)}. Return exactly one mapping label from: {_allowed(m)}."
    )


@register_prompt("205")
def _p205(m):
    return (
        f"Compare the named input and output grids. Foreground means any cell that is not "
        f"background color {_ci(m, 'background_color')}. Infer the changed scope and the "
        f"one-cell geometric operation. Return exactly one label from: {_allowed(m)}."
    )


@register_prompt("206")
def _p206(m):
    return (
        f"The input contains separated objects of color {_ci(m, 'object_color')}. "
        f"The output applies one action to exactly one selected object; action color is "
        f"{_ci(m, 'action_color')} and erase uses background color {_ci(m, 'background_color')}. "
        f"The mark_first_cell action recolors only the top-left (row-major first) cell of "
        f"the selected object to the action color, leaving its other cells unchanged. "
        f"Infer the selection rule and action. Return exactly one label from: {_allowed(m)}."
    )


@register_prompt("207")
def _p207(m):
    return (
        f"Compare the named input and output grids. Row or column selection is based on whether "
        f"a line contains marker color {_ci(m, 'marker_color')}. The erase operation writes "
        f"background color {_ci(m, 'background_color')}; the recolor operation writes "
        f"{_ci(m, 'line_color')}. Infer axis, operation, and condition. "
        f"Return exactly one label from: {_allowed(m)}."
    )


@register_prompt("208")
def _p208(m):
    return (
        "Compare the named input and output grids. The changed cells repair a symmetry relation "
        "using the center row or center column as reference. Infer the symmetry type, reference "
        f"side, and repair action. Return exactly one label from: {_allowed(m)}."
    )


@register_prompt("209")
def _p209(m):
    return (
        f"Starting from the seed cell of color {_ci(m, 'seed_color')}, traverse target cells "
        f"of color {_ci(m, 'target_color')} using either 4-connected or 8-connected adjacency. "
        f"The output recolors reachable target cells to {_ci(m, 'mark_color')}. "
        f"Infer the adjacency rule and return exactly one label from: {_allowed(m)}."
    )


@register_prompt("210")
def _p210(m):
    return (
        f"The input has a source cell of color {_ci(m, 'source_color')}. The output marks "
        f"background cells at one exact distance from the source using color {_ci(m, 'mark_color')}. "
        f"Infer the metric and distance parameter. Return exactly one label from: {_allowed(m)}."
    )
