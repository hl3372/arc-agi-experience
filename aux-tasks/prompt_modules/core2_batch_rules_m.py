"""Prompt templates for core2 batch M finite rule tasks."""
from __future__ import annotations

from prompts import _ci, register_prompt


def _color_value(m: dict, key: str, value: int) -> str:
    m[key] = int(value)
    return _ci(m, key)


def _mapping_tables(m: dict) -> str:
    parts = []
    for table in m.get("mapping_tables", []):
        entries = []
        for idx, entry in enumerate(table.get("entries", [])):
            src = _color_value(m, f"_map_{table['label']}_{idx}_from", entry["from_color"])
            dst = _color_value(m, f"_map_{table['label']}_{idx}_to", entry["to_color"])
            entries.append(f"{src}->{dst}")
        parts.append(f"{table['label']}: " + ", ".join(entries))
    return "; ".join(parts)


def _candidate_list(m: dict, key: str) -> str:
    return ", ".join(str(label) for label in m.get(key, []))


def _cell_rule_phrase(m: dict, rule: dict, idx: int) -> str:
    out = _color_value(m, f"_cell_rule_{idx}_out", rule["output_color"])
    label = rule["label"]
    if label == "cell_rule_seed_cell":
        match = _color_value(m, f"_cell_rule_{idx}_match", rule["match_color"])
        return f"{label}: if the current cell is {match}, write {out}"
    if label == "cell_rule_next_to_seed":
        neighbor = _color_value(m, f"_cell_rule_{idx}_neighbor", rule["neighbor_color"])
        return f"{label}: if a 4-neighbor is {neighbor}, write {out}"
    if label == "cell_rule_foreground_cell":
        bg = _color_value(m, f"_cell_rule_{idx}_bg", rule["background_color"])
        return f"{label}: if the current cell is not background {bg}, write {out}"
    return f"{label}: write {out}"


def _cell_rules(m: dict, key: str) -> str:
    return "; ".join(
        f"{idx + 1}. {_cell_rule_phrase(m, rule, idx)}"
        for idx, rule in enumerate(m.get(key, []))
    )


def _priority_orders(m: dict) -> str:
    parts = []
    for order in m.get("candidate_priority_orders", []):
        parts.append(f"{order['label']}: " + " > ".join(order.get("order", [])))
    return "; ".join(parts)


def _object_rule_phrase(m: dict, rule: dict, idx: int) -> str:
    out = _color_value(m, f"_object_rule_{idx}_out", rule["output_color"])
    label = rule["label"]
    if label == "object_rule_large":
        return f"{label}: if the object has at least {rule['threshold']} cells, recolor it to {out}"
    if label == "object_rule_wide":
        return f"{label}: if its bounding box is wider than tall, recolor it to {out}"
    if label == "object_rule_tall":
        return f"{label}: if its bounding box is taller than wide, recolor it to {out}"
    return f"{label}: recolor it to {out}"


def _object_rules(m: dict) -> str:
    return "; ".join(
        f"{idx + 1}. {_object_rule_phrase(m, rule, idx)}"
        for idx, rule in enumerate(m.get("ordered_object_rules", []))
    )


@register_prompt("221")
def _p221(m):
    return (
        f"The named grids before and after show every non-background cell translated together; "
        f"background is {_ci(m, 'background_color')}. Infer the motion model and parameters. "
        f"Return only one label from: {_candidate_list(m, 'candidate_motion_labels')}."
    )


@register_prompt("222")
def _p222(m):
    return (
        "The before and after grids show a motif color-substitution table. "
        f"Background {_ci(m, 'background_color')} is unchanged. Candidate table labels are: "
        f"{_mapping_tables(m)}. Return only the matching mapping label."
    )


@register_prompt("244")
def _p244(m):
    return (
        "Each example pair comes from the same family: remove an equal-width outer border "
        "from the input grid to get the output grid. Infer the shared hidden parameter. "
        f"Return only one label from: {_candidate_list(m, 'candidate_parameter_labels')}."
    )


@register_prompt("245")
def _p245(m):
    return (
        "Use all example input/output pairs to choose the single candidate whole-grid rule "
        "that is consistent with every pair. The rule labels are supplied in the input. "
        "Meanings: rule_flip_horizontal flips left-right; rule_flip_vertical flips top-bottom; "
        "rule_rotate_180 rotates half a turn; rule_transpose_main swaps rows and columns. "
        "Return only the chosen rule label."
    )


@register_prompt("246")
def _p246(m):
    return (
        f"Each example contains separate 4-connected objects of color {_ci(m, 'object_color')}. "
        f"In every output, one selected object is either recolored to {_ci(m, 'action_color')} "
        f"or erased to background {_ci(m, 'background_color')}. Infer the shared selection/action rule. "
        f"Return only one label from: {_candidate_list(m, 'candidate_rule_labels')}."
    )


@register_prompt("247")
def _p247(m):
    return (
        "All example pairs use the same cellwise color mapping type. "
        f"Background {_ci(m, 'background_color')} is unchanged. Candidate mapping labels are: "
        f"{_mapping_tables(m)}. Return only the shared mapping label."
    )


@register_prompt("248")
def _p248(m):
    return (
        f"The input contains one example_input grid and its example_output mask. The mask uses "
        f"{_ci(m, 'true_color')} for cells where a local predicate is true and "
        f"{_ci(m, 'false_color')} otherwise; the target neighbor color is {_ci(m, 'target_color')}. "
        "Only the target color matters; any other colors are background and are ignored. "
        "Return the predicate label for this shown example; do not apply it to another grid. "
        f"Return only one label from: {_candidate_list(m, 'candidate_predicate_labels')}."
    )


@register_prompt("257")
def _p257(m):
    return (
        "Apply this ordered list of cell-level rules to the input grid. For each cell, test rules "
        "against the original input cell and original neighbors; the first satisfied rule wins. "
        f"Rules: {_cell_rules(m, 'ordered_cell_rules')}. "
        "If no rule is satisfied, leave the cell unchanged."
    )


@register_prompt("258")
def _p258(m):
    return (
        "The named input and output grids were produced by applying the listed cell-level rules "
        "with first-satisfied-rule priority. Infer which candidate priority order was used. "
        f"Rules: {_cell_rules(m, 'cell_rules')}. "
        f"Candidate orders: {_priority_orders(m)}. Return only the priority label."
    )


@register_prompt("259")
def _p259(m):
    return (
        f"Find every non-background 4-connected object; background is {_ci(m, 'background_color')}. "
        "Apply this ordered list of object-level conditional rules to each object. "
        "For an object, the first satisfied rule wins. "
        f"Rules: {_object_rules(m)}. If no rule is satisfied, leave that object unchanged."
    )
