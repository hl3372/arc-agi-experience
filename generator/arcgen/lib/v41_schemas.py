"""Strict structured outputs for non-gating ARCGEN V4.1 experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .schemas import SchemaError, parse_json_object
from .v4_schemas import (
    OUTPUT_SCHEMAS as V4_OUTPUT_SCHEMAS,
    STRICT_OUTPUT_SCHEMAS as V4_STRICT_OUTPUT_SCHEMAS,
    parse_stage_output as parse_v4_stage_output,
)


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}
BOOLEAN = {"type": "boolean"}


def _object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": deepcopy(dict(properties)),
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: Any, *, minimum: int = 0, maximum: int | None = None) -> dict[str, Any]:
    value = {"type": "array", "items": deepcopy(items), "minItems": minimum}
    if maximum is not None:
        value["maxItems"] = maximum
    return value


SUPPORTED_ASSESSMENT = _object(
    {
        "status": {"type": "string", "enum": ["genuine", "cosmetic", "absent", "unsupported"]},
        "location": MAYBE_TEXT,
        "invariant": MAYBE_TEXT,
        "training_support": MAYBE_TEXT,
        "shortcut_defeated": MAYBE_TEXT,
        "test_only_semantics": MAYBE_TEXT,
        "reason": TEXT,
    }
)
JOINT_ASSESSMENT = _object(
    {
        "status": {"type": "string", "enum": ["genuine", "weak", "absent", "invalid"]},
        "jointly_inferred_fact": MAYBE_TEXT,
        "single_pair_insufficiency": MAYBE_TEXT,
        "joint_support": MAYBE_TEXT,
        "order_independence": MAYBE_TEXT,
        "test_determination": MAYBE_TEXT,
        "reason": TEXT,
    }
)
ABSTRACTION_ASSESSMENT = _object(
    {
        "status": {"type": "string", "enum": ["genuine", "partial", "merely_formal", "absent", "invalid"]},
        "latent_representation": MAYBE_TEXT,
        "visible_grounding": MAYBE_TEXT,
        "operational_necessity": MAYBE_TEXT,
        "compression": MAYBE_TEXT,
        "generalization": MAYBE_TEXT,
        "reason": TEXT,
    }
)
MECHANISM_JUDGMENT = _object(
    {
        "task_id": TEXT,
        "supported_departure": {"anyOf": [SUPPORTED_ASSESSMENT, {"type": "null"}]},
        "episode_joint_inference": {"anyOf": [JOINT_ASSESSMENT, {"type": "null"}]},
        "abstraction_led": {"anyOf": [ABSTRACTION_ASSESSMENT, {"type": "null"}]},
        "open_ended_mechanism_signatures": _object(
            {
                "supported_departure": MAYBE_TEXT,
                "episode_joint_inference": MAYBE_TEXT,
                "abstraction_led": MAYBE_TEXT,
            }
        ),
    }
)
MECHANISM_AUDIT = _object(
    {
        "audit_id": TEXT,
        "requested_properties": _array(
            {"type": "string", "enum": ["supported_departure", "episode_joint_inference", "abstraction_led"]},
            minimum=1,
            maximum=3,
        ),
        "judgments": _array(MECHANISM_JUDGMENT, minimum=1, maximum=16),
        "non_gating_confirmation": BOOLEAN,
    }
)

CLUSTER = _object(
    {
        "cluster_id": TEXT,
        "open_ended_description": TEXT,
        "member_task_ids": _array(TEXT, minimum=1, maximum=32),
        "representative_task_id": TEXT,
        "same_minimal_procedure_within_cluster": BOOLEAN,
    }
)
SAME_PROCEDURE_PAIR = _object(
    {"task_id_a": TEXT, "task_id_b": TEXT, "shared_minimal_procedure": TEXT}
)
DIVERSITY_AUDIT = _object(
    {
        "audit_id": TEXT,
        "property": {"type": "string", "enum": ["supported_departure", "episode_joint_inference", "abstraction_led"]},
        "clusters": _array(CLUSTER, minimum=1, maximum=32),
        "same_procedure_pairs": _array(SAME_PROCEDURE_PAIR, maximum=100),
        "surface_template_concentration": TEXT,
        "largest_cluster_is_more_than_half": BOOLEAN,
        "overall_diversity_assessment": TEXT,
    }
)

PALETTE_PAIR = _object(
    {
        "pair_id": TEXT,
        "task_a_id": TEXT,
        "task_b_id": TEXT,
        "global_bijection_exists": BOOLEAN,
        "full_episode_consistent": BOOLEAN,
        "rule_unchanged": BOOLEAN,
        "inferability_unchanged": BOOLEAN,
        "preferred_presentation": {"type": "string", "enum": ["a", "b", "neither"]},
        "substantive_difference": BOOLEAN,
        "reason": TEXT,
    }
)
PALETTE_EQUIVALENCE = _object(
    {
        "review_id": TEXT,
        "judgments": _array(PALETTE_PAIR, minimum=1, maximum=12),
        "overall_observation": TEXT,
    }
)

DEPARTURE_CONTRACT = _object(
    {
        "location": TEXT,
        "apparent_regularity": TEXT,
        "concrete_departure": TEXT,
        "invariant": TEXT,
        "training_support": TEXT,
        "shortcut_defeated": TEXT,
        "uniqueness": TEXT,
    }
)
SUPPORTED_PLAN = _object(
    {
        "candidate_id": TEXT,
        "treatment_decision": {"type": "string", "enum": ["use", "decline", "abort"]},
        "decision_reason": TEXT,
        "supported_departure_contract": {"anyOf": [DEPARTURE_CONTRACT, {"type": "null"}]},
        "concrete_plan": {"anyOf": [deepcopy(V4_STRICT_OUTPUT_SCHEMAS["v4_concrete_plan"]), {"type": "null"}]},
    }
)

JOINT_CONTRACT = _object(
    {
        "jointly_inferred_fact": TEXT,
        "single_pair_insufficiency": TEXT,
        "joint_support": TEXT,
        "order_independence": TEXT,
        "test_determination": TEXT,
    }
)
ABSTRACTION_CONTRACT = _object(
    {
        "latent_representation": TEXT,
        "grounding": TEXT,
        "necessity": TEXT,
        "compression": TEXT,
        "generalization": TEXT,
        "arbitrariness_check": TEXT,
        "simpler_rule_check": TEXT,
    }
)


def _grounded_plan(contract: Mapping[str, Any]) -> dict[str, Any]:
    return _object(
        {
            "candidate_id": TEXT,
            "property_status": {"type": "string", "enum": ["retained", "dropped", "abort"]},
            "decision_reason": TEXT,
            "property_contract": {"anyOf": [deepcopy(dict(contract)), {"type": "null"}]},
            "concrete_plan": {"anyOf": [deepcopy(V4_STRICT_OUTPUT_SCHEMAS["v4_concrete_plan"]), {"type": "null"}]},
        }
    )


STRICT_OUTPUT_SCHEMAS = {
    **deepcopy(V4_STRICT_OUTPUT_SCHEMAS),
    "v41_mechanism_audit": MECHANISM_AUDIT,
    "v41_diversity_audit": DIVERSITY_AUDIT,
    "v41_palette_equivalence": PALETTE_EQUIVALENCE,
    "v41_supported_plan": SUPPORTED_PLAN,
    "v41_episode_joint_plan": _grounded_plan(JOINT_CONTRACT),
    "v41_abstraction_plan": _grounded_plan(ABSTRACTION_CONTRACT),
}


def _transport(value: Any) -> Any:
    unsupported = {"minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "uniqueItems"}
    if isinstance(value, dict):
        return {key: _transport(item) for key, item in value.items() if key not in unsupported}
    if isinstance(value, list):
        return [_transport(item) for item in value]
    return value


OUTPUT_SCHEMAS = {
    **deepcopy(V4_OUTPUT_SCHEMAS),
    **{
        stage: _transport(schema)
        for stage, schema in STRICT_OUTPUT_SCHEMAS.items()
        if stage.startswith("v41_")
    },
}


def _validate_new_stage(stage: str, value: dict[str, Any]) -> None:
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
    if stage == "v41_mechanism_audit":
        requested = set(value["requested_properties"])
        if len(requested) != len(value["requested_properties"]):
            raise SchemaError("$.requested_properties", "properties must be unique")
        if value["non_gating_confirmation"] is not True:
            raise SchemaError("$.non_gating_confirmation", "auditor must confirm non-gating status")
        field = {
            "supported_departure": "supported_departure",
            "episode_joint_inference": "episode_joint_inference",
            "abstraction_led": "abstraction_led",
        }
        for index, judgment in enumerate(value["judgments"]):
            for prop, key in field.items():
                if (judgment[key] is not None) != (prop in requested):
                    raise SchemaError(f"$.judgments[{index}].{key}", "must be present exactly when requested")
    elif stage == "v41_diversity_audit":
        members = [item for cluster in value["clusters"] for item in cluster["member_task_ids"]]
        if len(members) != len(set(members)):
            raise SchemaError("$.clusters", "a task may belong to only one mechanism cluster")
        for index, cluster in enumerate(value["clusters"]):
            if cluster["representative_task_id"] not in cluster["member_task_ids"]:
                raise SchemaError(f"$.clusters[{index}]", "representative must be a cluster member")
    elif stage == "v41_palette_equivalence":
        pair_ids = [item["pair_id"] for item in value["judgments"]]
        if len(pair_ids) != len(set(pair_ids)):
            raise SchemaError("$.judgments", "pair IDs must be unique")
    elif stage == "v41_supported_plan":
        use = value["treatment_decision"] == "use"
        abort = value["treatment_decision"] == "abort"
        if (value["supported_departure_contract"] is not None) != use:
            raise SchemaError("$.supported_departure_contract", "contract is required only when used")
        if (value["concrete_plan"] is None) != abort:
            raise SchemaError("$.concrete_plan", "only abort may omit its plan")
        if value["concrete_plan"] is not None:
            parse_v4_stage_output("v4_concrete_plan", value["concrete_plan"])
    elif stage in {"v41_episode_joint_plan", "v41_abstraction_plan"}:
        retained = value["property_status"] == "retained"
        abort = value["property_status"] == "abort"
        if (value["property_contract"] is not None) != retained:
            raise SchemaError("$.property_contract", "contract is required only when retained")
        if (value["concrete_plan"] is None) != abort:
            raise SchemaError("$.concrete_plan", "only abort may omit its plan")
        if value["concrete_plan"] is not None:
            parse_v4_stage_output("v4_concrete_plan", value["concrete_plan"])


def parse_stage_output(stage: str, data: str | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    if stage in V4_STRICT_OUTPUT_SCHEMAS:
        return parse_v4_stage_output(stage, data)
    if stage not in STRICT_OUTPUT_SCHEMAS:
        raise SchemaError("$.stage", f"unknown V4.1 stage {stage!r}")
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else dict(data)
    _validate_new_stage(stage, value)
    return value


__all__ = ["OUTPUT_SCHEMAS", "STRICT_OUTPUT_SCHEMAS", "parse_stage_output"]
