"""Public verification payloads shared by the factory and calibration replay."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any


TECHNICAL_PLAN_PROJECTION = {
    "version": "technical-public-plan-v1",
    "applies_to": ["gpt-5.6-sol", "gpt-6-astra"],
    "withheld_top_level_fields": [
        "semantic_core", "cross_pair_concrete_audit", "deterministic_realization_procedure",
        "axis_self_assessment", "abort_reason",
    ],
    "retained_test_fields": ["pair_number", "input_dimensions", "exact_input_construction"],
    "withheld_test_fields": [
        "output_dimensions", "exact_expected_output_construction",
        "precise_placements_quantities_relations_or_states", "output_derivation",
        "what_is_new_relative_to_training", "training_support",
        "test_only_semantics_audit", "why_one_output_is_best_supported",
    ],
    "withheld_other_fields": [
        "test canvas descriptions", "unrecognized fields",
        "free-text constraints referring to tests, outputs, answers, constructions, dimensions, or grid literals",
    ],
    "intended_rule_supplied_separately": True,
    "public_task_test_outputs_forbidden": True,
}

_TRAINING_FIELDS = (
    "pair_number", "pair_purpose", "input_dimensions", "output_dimensions",
    "exact_input_construction", "exact_output_construction",
    "precise_placements_quantities_relations_or_states", "output_derivation",
    "what_it_helps_communicate", "deliberate_overlap_with_other_pairs", "why_overlap_is_harmless",
)
_CANVAS_FIELDS = (
    "pair", "why_dimensions_are_needed", "functionally_used_areas",
    "empty_space_role", "smaller_grid_audit", "larger_grid_audit",
)
_ANSWER_TEXT = re.compile(r"\b(?:outputs?|answers?|expected|results?)\b|\[\s*\[", re.IGNORECASE)
_PRIVATE_CONSTRAINT = re.compile(
    r"\b(?:tests?|outputs?|answers?|construction|derive[ds]?|derivation|procedure|"
    r"placements?|coordinates?|dimensions?|shape|height|width|canvas)\b|\[\s*\[",
    re.IGNORECASE,
)


def _fields(value: Mapping[str, Any], names: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {name: deepcopy(value[name]) for name in names if name in value}


def technical_public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Withhold authored test answers and answer-bearing plan prose.

    This is an allowlisted *input projection*, not a different authoring schema.
    The intended rule and actual public grids are separate verifier inputs.
    Public training constructions remain inspectable. Test descriptions expose
    input construction only; private test dimensions, output procedures and
    global narrative audits are deliberately absent in both matched controls.
    The full plan is retained in private candidate state and dossiers.
    """
    result = _fields(plan, ("candidate_id", "status"))
    overview = plan.get("episode_overview")
    if isinstance(overview, Mapping):
        result["episode_overview"] = _fields(overview, (
            "training_pair_count", "test_pair_count", "background_color", "functional_color_roles",
        ))
    result["training_pairs"] = [
        _fields(pair, _TRAINING_FIELDS)
        for pair in plan.get("training_pairs", []) if isinstance(pair, Mapping)
    ]
    result["test_pairs"] = []
    for pair in plan.get("test_pairs", []):
        if not isinstance(pair, Mapping):
            continue
        public = _fields(pair, TECHNICAL_PLAN_PROJECTION["retained_test_fields"])
        text = public.get("exact_input_construction")
        if not isinstance(text, str) or _ANSWER_TEXT.search(text):
            public.pop("exact_input_construction", None)
        result["test_pairs"].append(public)
    result["canvas_plan"] = [
        _fields(pair, _CANVAS_FIELDS) for pair in plan.get("canvas_plan", [])
        if isinstance(pair, Mapping) and re.fullmatch(r"train_\d+", str(pair.get("pair", "")))
    ]
    for name in ("frozen_decisions", "allowed_realizer_freedom"):
        result[name] = [
            text for text in plan.get(name, [])
            if isinstance(text, str) and not _PRIVATE_CONSTRAINT.search(text)
        ]
    return result
