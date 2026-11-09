"""Strict structured-output contracts for ARCGEN V4 and its isolated studies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
import re

from jsonschema import Draft202012Validator

from .lean_v31_schemas import STRICT_OUTPUT_SCHEMAS as V31_SCHEMAS
from .schemas import SchemaError, parse_grid, parse_json_object


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}
CONFIDENCE = {"type": "string", "enum": ["low", "medium", "high"]}
CONCEPT_AXIS = {"type": "string", "enum": ["weak", "solid", "standout"]}
COMPOSITION_AXIS = {"type": "string", "enum": ["thin", "solid", "standout"]}
GRID = {
    "type": "array",
    "minItems": 1,
    "maxItems": 30,
    "items": {
        "type": "array",
        "minItems": 1,
        "maxItems": 30,
        "items": {"type": "integer", "minimum": 0, "maximum": 9},
    },
}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: Any, *, minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "array",
        "items": deepcopy(items),
        "minItems": minimum,
    }
    if maximum is not None:
        value["maxItems"] = maximum
    return value


DIMENSIONS = _object(
    {
        "height": {"type": "integer", "minimum": 1, "maximum": 30},
        "width": {"type": "integer", "minimum": 1, "maximum": 30},
    }
)
COLOR_ROLE = _object(
    {
        "role": TEXT,
        "color": {"type": "integer", "minimum": 0, "maximum": 9},
        "functional_meaning": TEXT,
    }
)
SEMANTIC_CORE = _object(
    {
        "central_rule": TEXT,
        "central_conceptual_insight": TEXT,
        "why_worth_realizing": TEXT,
        "what_must_remain_visible_after_rendering": TEXT,
    }
)
EPISODE_OVERVIEW = _object(
    {
        "training_pair_count": {"type": "integer", "minimum": 1, "maximum": 10},
        "test_pair_count": {"type": "integer", "minimum": 1, "maximum": 3},
        "intended_difficulty_and_inferability": TEXT,
        "background_treatment": TEXT,
        "background_color": {
            "anyOf": [
                {"type": "integer", "minimum": 0, "maximum": 9},
                {"type": "null"},
            ]
        },
        "functional_color_roles": _array(COLOR_ROLE, minimum=1, maximum=10),
    }
)
TRAINING_STORYBOARD = _object(
    {
        "pair_number": {"type": "integer", "minimum": 1, "maximum": 10},
        "pair_purpose": {
            "type": "string",
            "enum": ["establish", "extend", "reinforce", "vary", "edge_case", "other"],
        },
        "input_dimensions": DIMENSIONS,
        "output_dimensions": DIMENSIONS,
        "exact_input_construction": TEXT,
        "exact_output_construction": TEXT,
        "precise_placements_quantities_relations_or_states": TEXT,
        "output_derivation": TEXT,
        "what_it_helps_communicate": TEXT,
        "deliberate_overlap_with_other_pairs": MAYBE_TEXT,
        "why_overlap_is_harmless": MAYBE_TEXT,
    }
)
TEST_STORYBOARD = _object(
    {
        "pair_number": {"type": "integer", "minimum": 1, "maximum": 3},
        "input_dimensions": DIMENSIONS,
        "output_dimensions": DIMENSIONS,
        "exact_input_construction": TEXT,
        "exact_expected_output_construction": TEXT,
        "precise_placements_quantities_relations_or_states": TEXT,
        "output_derivation": TEXT,
        "what_is_new_relative_to_training": TEXT,
        "training_support": TEXT,
        "test_only_semantics_audit": TEXT,
        "why_one_output_is_best_supported": TEXT,
    }
)
SHORTCUT = _object(
    {
        "rule": TEXT,
        "fits_every_training_pair": BOOLEAN,
        "comparably_simple": BOOLEAN,
        "predicts_a_different_test_output": BOOLEAN,
        "evidence_that_defeats_it": TEXT,
    }
)
CROSS_PAIR_AUDIT = _object(
    {
        "output_relevant_properties": _array(TEXT, minimum=1, maximum=20),
        "incidental_properties": _array(TEXT, maximum=20),
        "relevant_and_incidental_variation": TEXT,
        "strongest_simple_shortcuts": _array(SHORTCUT, maximum=10),
        "remaining_ambiguity_risk": TEXT,
    }
)
CANVAS_PAIR = _object(
    {
        "pair": TEXT,
        "why_dimensions_are_needed": TEXT,
        "functionally_used_areas": TEXT,
        "empty_space_role": TEXT,
        "smaller_grid_audit": TEXT,
        "larger_grid_audit": TEXT,
    }
)
AXIS_SELF_ASSESSMENT = _object(
    {
        "novel_concept": CONCEPT_AXIS,
        "functional_composition": COMPOSITION_AXIS,
    }
)
CONCRETE_PLAN = _object(
    {
        "candidate_id": TEXT,
        "status": {"type": "string", "enum": ["ready", "abort"]},
        "abort_reason": MAYBE_TEXT,
        "semantic_core": {"anyOf": [SEMANTIC_CORE, {"type": "null"}]},
        "episode_overview": {"anyOf": [EPISODE_OVERVIEW, {"type": "null"}]},
        "training_pairs": _array(TRAINING_STORYBOARD, maximum=10),
        "test_pairs": _array(TEST_STORYBOARD, maximum=3),
        "cross_pair_concrete_audit": {"anyOf": [CROSS_PAIR_AUDIT, {"type": "null"}]},
        "canvas_plan": _array(CANVAS_PAIR, maximum=13),
        "frozen_decisions": _array(TEXT, maximum=30),
        "allowed_realizer_freedom": _array(TEXT, maximum=20),
        "deterministic_realization_procedure": MAYBE_TEXT,
        "axis_self_assessment": {"anyOf": [AXIS_SELF_ASSESSMENT, {"type": "null"}]},
    }
)

ARC_TRAIN_PAIR = _object({"input": GRID, "output": GRID})
ARC_TEST_INPUT = _object({"input": GRID})
PUBLIC_TASK = _object(
    {
        "train": _array(ARC_TRAIN_PAIR, minimum=1, maximum=10),
        "test": _array(ARC_TEST_INPUT, minimum=1, maximum=3),
    }
)
PAIR_TRACE = _object({"pair": TEXT, "trace": TEXT})
DEVIATION = _object(
    {
        "plan_field": TEXT,
        "deviation": TEXT,
        "semantic_or_evidential": BOOLEAN,
        "reason": TEXT,
    }
)
PLAN_FIDELITY = _object(
    {
        "statement": TEXT,
        "preserved_frozen_decisions": _array(TEXT, maximum=30),
        "allowed_freedoms_used": _array(TEXT, maximum=20),
        "deviations": _array(DEVIATION, maximum=20),
    }
)
REALIZER_METADATA = _object(
    {
        "intended_rule": TEXT,
        "expected_test_outputs": _array(GRID, minimum=1, maximum=3),
        "construction_witness": TEXT,
        "pair_construction_trace": _array(PAIR_TRACE, minimum=2, maximum=13),
        "plan_fidelity": PLAN_FIDELITY,
        "concrete_shortcut_audit": TEXT,
        "canvas_use_classification": {
            "type": "string",
            "enum": [
                "compact_and_appropriate",
                "large_and_functionally_used",
                "large_but_partially_underused",
                "artificially_inflated",
            ],
        },
    }
)
TASK_REALIZER = _object(
    {
        "candidate_id": TEXT,
        "status": {"type": "string", "enum": ["realized", "refused"]},
        "refusal_reason": MAYBE_TEXT,
        "task": {"anyOf": [PUBLIC_TASK, {"type": "null"}]},
        "authoring_metadata": {"anyOf": [REALIZER_METADATA, {"type": "null"}]},
        "deviations": _array(DEVIATION, maximum=20),
    }
)

TRAINING_TRACE = _object(
    {
        "pair": TEXT,
        "follows_rule_exactly": BOOLEAN,
        "trace": TEXT,
        "issues": _array(TEXT, maximum=20),
    }
)
DIAGNOSTIC_ASSESSMENT = _object(
    {
        "flag_code": TEXT,
        "justified": BOOLEAN,
        "evidentially_necessary": BOOLEAN,
        "assessment": TEXT,
    }
)
TECHNICAL_VERIFIER = _object(
    {
        "task_id": TEXT,
        "verifier_id": TEXT,
        "status": {"type": "string", "enum": ["pass", "repairable", "fatal", "uncertain"]},
        "derived_test_outputs": _array(GRID, minimum=1, maximum=3),
        "training_pair_traces": _array(TRAINING_TRACE, minimum=1, maximum=10),
        "mechanically_correct": BOOLEAN,
        "test_supported": BOOLEAN,
        "one_answer_supported": BOOLEAN,
        "plan_fidelity": {"type": "string", "enum": ["pass", "concern", "fail"]},
        "unapproved_realizer_decisions": _array(TEXT, maximum=20),
        "evidence_sufficiency": {"type": "string", "enum": ["sufficient", "thin", "unsupported"]},
        "redundancy_assessment": TEXT,
        "pair_economy": {"type": "string", "enum": ["pass", "concern", "fail"]},
        "composition_preserved": BOOLEAN,
        "serious_concrete_shortcuts": _array(TEXT, maximum=20),
        "canvas_use": {"type": "string", "enum": ["appropriate", "underused", "inflated", "uncertain"]},
        "arbitrary_exceptions": _array(TEXT, maximum=20),
        "unsupported_test_semantics": _array(TEXT, maximum=20),
        "diagnostic_flag_assessments": _array(DIAGNOSTIC_ASSESSMENT, maximum=100),
        "repair_instruction": MAYBE_TEXT,
        "confidence": CONFIDENCE,
    }
)

EDITORIAL_ENTRY = _object(
    {
        "task_id": TEXT,
        "novel_concept": CONCEPT_AXIS,
        "functional_composition": COMPOSITION_AXIS,
        "overall_quality": {"type": "string", "enum": ["below_real_bar", "real_quality", "exceptional"]},
        "discoverability": TEXT,
        "conceptual_compression": TEXT,
        "pair_sufficiency": TEXT,
        "composition_grid_evidence": TEXT,
        "scale_and_canvas_observation": TEXT,
        "same_minimal_procedure": BOOLEAN,
        "theme_is_common": BOOLEAN,
        "addition_is_excessive": BOOLEAN,
        "intrinsic_recommendation": {"type": "string", "enum": ["core_quality", "reserve_quality", "rejected_quality"]},
        "portfolio_recommendation": {"type": "string", "enum": ["include_now", "hold_crowded", "same_procedure_duplicate"]},
        "reason": TEXT,
    }
)
EDITORIAL_BOARD = _object(
    {
        "board_id": TEXT,
        "judgments": _array(EDITORIAL_ENTRY, minimum=7, maximum=12),
        "batch_observations": MAYBE_TEXT,
    }
)

CRITIC_ISSUE = _object(
    {
        "category": {
            "type": "string",
            "enum": [
                "semantic_decision_left_to_realizer",
                "test_not_uniquely_supported",
                "incidental_correlation",
                "simpler_rule",
                "harmful_repetition",
                "insight_buried",
                "canvas_not_justified",
                "simplification_available",
                "other_concrete_defect",
            ],
        },
        "concrete_evidence": TEXT,
        "minimal_action": TEXT,
    }
)
PLAN_CRITIC_ENTRY = _object(
    {
        "candidate_id": TEXT,
        "verdict": {"type": "string", "enum": ["ready_unchanged", "concrete_defect_found", "drop_recommended"]},
        "issues": _array(CRITIC_ISSUE, maximum=8),
        "why_no_change_is_preferred_or_not": TEXT,
        "confidence": CONFIDENCE,
    }
)
PLAN_CRITIC = _object(
    {
        "review_id": TEXT,
        "judgments": _array(PLAN_CRITIC_ENTRY, minimum=1, maximum=10),
        "editing_tendency_audit": TEXT,
    }
)
PLAN_REVISION = _object(
    {
        "candidate_id": TEXT,
        "status": {"type": "string", "enum": ["revised", "unchanged", "drop"]},
        "revision_summary": TEXT,
        "revised_plan": {"anyOf": [CONCRETE_PLAN, {"type": "null"}]},
    }
)

ANSWER_AWARE = _object(
    {
        "task_id": TEXT,
        "reviewer_id": TEXT,
        "review_mode": {"type": "string", "enum": ["answer_aware", "adversarial"]},
        "verdict": {"type": "string", "enum": ["well_supported", "underdetermined", "unsupported"]},
        "coherent_supporting_rule": TEXT,
        "specific_training_evidence": _array(TEXT, minimum=1, maximum=20),
        "strongest_competing_rule": TEXT,
        "competing_rule_fits_every_training_pair": BOOLEAN,
        "competing_rule_predictions": _array(GRID, minimum=1, maximum=3),
        "competing_rule_predicts_different_output": BOOLEAN,
        "test_only_assumptions": _array(TEXT, maximum=20),
        "retrospective_rationalization_risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "repair_advice": MAYBE_TEXT,
        "reason": TEXT,
        "confidence": CONFIDENCE,
    }
)

CURATION_ENTRY = _object(
    {
        "candidate_id": TEXT,
        "recommendation": {"type": "string", "enum": ["promote", "hold", "flag_duplicate"]},
        "reason": TEXT,
    }
)
PRELIMINARY_CURATION = _object(
    {
        "curation_id": TEXT,
        "assessments": _array(CURATION_ENTRY, minimum=1, maximum=48),
        "ranking": _array(TEXT, minimum=1, maximum=48),
        "portfolio_observation": TEXT,
    }
)


STRICT_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "v4_idea_search": deepcopy(V31_SCHEMAS["lean_v31_idea_search"]),
    "v4_shard_selector": deepcopy(V31_SCHEMAS["lean_v31_shard_selector"]),
    "v4_global_selector": deepcopy(V31_SCHEMAS["lean_v31_global_selector"]),
    "v4_concrete_plan": CONCRETE_PLAN,
    "v4_task_realizer": TASK_REALIZER,
    "v4_technical_verifier": TECHNICAL_VERIFIER,
    "v4_blind_solver": deepcopy(V31_SCHEMAS["lean_v31_blind_solver"]),
    "v4_ambiguity_adjudicator": deepcopy(V31_SCHEMAS["lean_v31_ambiguity_adjudicator"]),
    "v4_editorial_board": EDITORIAL_BOARD,
    "v4_plan_critic": PLAN_CRITIC,
    "v4_plan_revision": PLAN_REVISION,
    "v4_answer_aware": ANSWER_AWARE,
    "v4_preliminary_curation": PRELIMINARY_CURATION,
}

# A targeted repair returns a potentially revised plan and a fresh realization.
STRICT_OUTPUT_SCHEMAS["v4_targeted_repair"] = _object(
    {
        "candidate_id": TEXT,
        "repair_type": {"type": "string", "enum": ["mechanical", "ambiguity"]},
        "status": {"type": "string", "enum": ["repaired", "unrepairable"]},
        "unrepairable_reason": MAYBE_TEXT,
        "repaired_plan": {"anyOf": [CONCRETE_PLAN, {"type": "null"}]},
        "repaired_realization": {"anyOf": [TASK_REALIZER, {"type": "null"}]},
        "changes": _array(TEXT, maximum=10),
        "why_rule_and_test_purpose_are_preserved": MAYBE_TEXT,
    }
)


def _transport_schema(value: Any) -> Any:
    unsupported = {
        "minLength", "maxLength", "minimum", "maximum", "minItems",
        "maxItems", "uniqueItems",
    }
    if isinstance(value, dict):
        return {key: _transport_schema(item) for key, item in value.items() if key not in unsupported}
    if isinstance(value, list):
        return [_transport_schema(item) for item in value]
    return value


OUTPUT_SCHEMAS = {stage: _transport_schema(schema) for stage, schema in STRICT_OUTPUT_SCHEMAS.items()}


def _schema_validate(stage: str, value: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(STRICT_OUTPUT_SCHEMAS[stage]).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)


def _validate_plan(value: dict[str, Any], *, path: str = "$") -> None:
    ready = value["status"] == "ready"
    sections = (
        value["semantic_core"], value["episode_overview"],
        value["cross_pair_concrete_audit"], value["axis_self_assessment"],
    )
    if not ready:
        if not value["abort_reason"].strip():
            raise SchemaError(f"{path}.abort_reason", "abort requires a reason")
        return
    if value["abort_reason"].strip() or any(item is None for item in sections):
        raise SchemaError(path, "ready concrete plan has an abort reason or missing section")
    overview = value["episode_overview"]
    if overview["training_pair_count"] != len(value["training_pairs"]):
        raise SchemaError(f"{path}.training_pairs", "count differs from episode overview")
    if overview["test_pair_count"] != len(value["test_pairs"]):
        raise SchemaError(f"{path}.test_pairs", "count differs from episode overview")
    if [item["pair_number"] for item in value["training_pairs"]] != list(range(1, len(value["training_pairs"]) + 1)):
        raise SchemaError(f"{path}.training_pairs", "pair numbers must be contiguous")
    if [item["pair_number"] for item in value["test_pairs"]] != list(range(1, len(value["test_pairs"]) + 1)):
        raise SchemaError(f"{path}.test_pairs", "pair numbers must be contiguous")
    if not value["frozen_decisions"] or not value["deterministic_realization_procedure"].strip():
        raise SchemaError(path, "ready plan must freeze decisions and give an executable procedure")
    axes = value["axis_self_assessment"]
    # Local broad-arm q2 revision: retain the honest composition self-rating.
    # Every structural/public/evidence check and the novelty floor is unchanged.
    if axes["novel_concept"] == "weak":
        raise SchemaError(f"{path}.axis_self_assessment", "ready plan must have at least solid novelty")
    canonicalize_plan_canvas_pair_labels(value)
    expected_canvas = {
        *(f"train_{index}" for index in range(1, len(value["training_pairs"]) + 1)),
        *(f"test_{index}" for index in range(1, len(value["test_pairs"]) + 1)),
    }
    actual_canvas = {item["pair"] for item in value["canvas_plan"]}
    if actual_canvas != expected_canvas or len(value["canvas_plan"]) != len(expected_canvas):
        raise SchemaError(f"{path}.canvas_plan", "must cover every planned pair exactly once")


def canonicalize_plan_canvas_pair_labels(value: dict[str, Any]) -> list[dict[str, str]]:
    """Canonicalize purely referential canvas-plan labels.

    The pair references are dynamic, so JSON Schema cannot enumerate them.
    Models naturally emit ``Training 1`` as well as ``train_1``.  These labels
    do not encode task semantics; normalize only the unambiguous spelling
    variants and leave every other value untouched for strict validation.
    The returned ledger lets interrupted-run recovery record each change.
    """

    changes: list[dict[str, str]] = []
    for item in value.get("canvas_plan", []):
        original = item.get("pair")
        if not isinstance(original, str):
            continue
        match = re.fullmatch(
            r"\s*(train(?:ing)?|test)[\s_-]*(\d+)\s*",
            original,
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        prefix = "train" if match.group(1).casefold().startswith("train") else "test"
        canonical = f"{prefix}_{int(match.group(2))}"
        if canonical != original:
            item["pair"] = canonical
            changes.append({"from": original, "to": canonical})
    return changes


def _validate_realizer(value: dict[str, Any], *, path: str = "$") -> None:
    if value["status"] == "refused":
        if not value["refusal_reason"].strip() or value["task"] is not None or value["authoring_metadata"] is not None:
            raise SchemaError(path, "refused realization must explain itself and contain no task")
        return
    if value["refusal_reason"].strip() or value["task"] is None or value["authoring_metadata"] is None:
        raise SchemaError(path, "realized response is incomplete or contains a refusal reason")
    task = value["task"]
    metadata = value["authoring_metadata"]
    if len(metadata["expected_test_outputs"]) != len(task["test"]):
        raise SchemaError(f"{path}.authoring_metadata.expected_test_outputs", "must match test count")
    grids: list[tuple[Any, str]] = []
    for split in ("train", "test"):
        for index, pair in enumerate(task[split]):
            grids.append((pair["input"], f"{path}.task.{split}[{index}].input"))
            if split == "train":
                grids.append((pair["output"], f"{path}.task.train[{index}].output"))
    grids.extend(
        (grid, f"{path}.authoring_metadata.expected_test_outputs[{index}]")
        for index, grid in enumerate(metadata["expected_test_outputs"])
    )
    for grid, grid_path in grids:
        parse_grid(grid, grid_path)
    labels = {item["pair"] for item in metadata["pair_construction_trace"]}
    expected = {
        *(f"train_{index}" for index in range(1, len(task["train"]) + 1)),
        *(f"test_{index}" for index in range(1, len(task["test"]) + 1)),
    }
    if labels != expected or len(metadata["pair_construction_trace"]) != len(expected):
        raise SchemaError(f"{path}.authoring_metadata.pair_construction_trace", "must cover every pair exactly once")


def _validate_output_grids(stage: str, value: dict[str, Any]) -> None:
    if stage == "v4_technical_verifier":
        for index, grid in enumerate(value["derived_test_outputs"]):
            parse_grid(grid, f"$.derived_test_outputs[{index}]")
    elif stage == "v4_answer_aware":
        for index, grid in enumerate(value["competing_rule_predictions"]):
            parse_grid(grid, f"$.competing_rule_predictions[{index}]")
    elif stage == "v4_blind_solver":
        for index, grid in enumerate(value["predicted_test_outputs"]):
            parse_grid(grid, f"$.predicted_test_outputs[{index}]")
        for alternative_index, alternative in enumerate(value["strongest_plausible_alternatives"]):
            for index, grid in enumerate(alternative["predicted_test_outputs"]):
                parse_grid(
                    grid,
                    f"$.strongest_plausible_alternatives[{alternative_index}].predicted_test_outputs[{index}]",
                )


def parse_stage_output(stage: str, data: str | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    if stage not in STRICT_OUTPUT_SCHEMAS:
        raise SchemaError("$.stage", f"unknown V4 stage {stage!r}")
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else dict(data)
    _schema_validate(stage, value)
    if stage == "v4_idea_search":
        ids = [item["candidate_id"] for item in value["ideas"]]
        if len(ids) != len(set(ids)) or any(item["search_mode"] != value["search_mode"] for item in value["ideas"]):
            raise SchemaError("$.ideas", "idea IDs must be unique and preserve search mode")
    elif stage in {"v4_shard_selector", "v4_global_selector"}:
        assessments = value["candidate_assessments"]
        assessed = [item["candidate_id"] for item in assessments]
        selected = [item["candidate_id"] for item in value["selected"]]
        expected = {item["candidate_id"] for item in assessments if item["decision"] == "select"}
        if len(assessed) != len(set(assessed)) or len(selected) != len(set(selected)) or set(selected) != expected:
            raise SchemaError("$", "selector IDs or select decisions are inconsistent")
        if sorted(item["development_priority"] for item in value["selected"]) != list(range(1, len(selected) + 1)):
            raise SchemaError("$.selected", "development priorities must be contiguous")
    elif stage == "v4_concrete_plan":
        _validate_plan(value)
    elif stage == "v4_task_realizer":
        _validate_realizer(value)
    elif stage == "v4_targeted_repair":
        if value["status"] == "unrepairable":
            if not value["unrepairable_reason"].strip() or value["repaired_plan"] is not None or value["repaired_realization"] is not None:
                raise SchemaError("$", "unrepairable response must contain only a reason")
        else:
            if value["repaired_plan"] is None or value["repaired_realization"] is None:
                raise SchemaError("$", "repaired response needs a plan and realization")
            _validate_plan(value["repaired_plan"], path="$.repaired_plan")
            _validate_realizer(value["repaired_realization"], path="$.repaired_realization")
    elif stage == "v4_plan_critic":
        ids = [item["candidate_id"] for item in value["judgments"]]
        if len(ids) != len(set(ids)):
            raise SchemaError("$.judgments", "candidate IDs must be unique")
        for index, item in enumerate(value["judgments"]):
            if item["verdict"] == "ready_unchanged" and item["issues"]:
                raise SchemaError(f"$.judgments[{index}]", "ready_unchanged must not invent issues")
            if item["verdict"] != "ready_unchanged" and not item["issues"]:
                raise SchemaError(f"$.judgments[{index}]", "a defect verdict requires a concrete issue")
    elif stage == "v4_plan_revision":
        if value["status"] == "drop" and value["revised_plan"] is not None:
            raise SchemaError("$.revised_plan", "drop must not return a plan")
        if value["status"] in {"revised", "unchanged"}:
            if value["revised_plan"] is None:
                raise SchemaError("$.revised_plan", "non-drop revision requires a plan")
            _validate_plan(value["revised_plan"], path="$.revised_plan")
    elif stage == "v4_preliminary_curation":
        ids = [item["candidate_id"] for item in value["assessments"]]
        if len(ids) != len(set(ids)) or set(ids) != set(value["ranking"]) or len(value["ranking"]) != len(ids):
            raise SchemaError("$", "curation ranking must cover each assessed candidate exactly once")
    _validate_output_grids(stage, value)
    return value


__all__ = [
    "OUTPUT_SCHEMAS", "STRICT_OUTPUT_SCHEMAS",
    "canonicalize_plan_canvas_pair_labels", "parse_stage_output",
]
