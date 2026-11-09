"""Structured output contracts for the V3.1 scaled authoring branch."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .lean_v3_schemas import (
    STRICT_OUTPUT_SCHEMAS as V3_SCHEMAS,
    parse_stage_output as parse_v3_stage_output,
)
from .schemas import SchemaError, parse_json_object


V31_TO_V3_STAGE = {
    "lean_v31_idea_search": "lean_v3_idea_search",
    "lean_v31_shard_selector": "lean_v3_batch_selector",
    "lean_v31_episode_architect": "lean_v3_episode_architect",
    "lean_v31_task_realizer": "lean_v3_task_realizer",
    "lean_v31_technical_verifier": "lean_v3_technical_verifier",
    "lean_v31_blind_solver": "lean_v3_blind_solver",
    "lean_v31_ambiguity_adjudicator": "lean_v3_ambiguity_adjudicator",
    "lean_v31_targeted_repair": "lean_v3_targeted_repair",
}

STRICT_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    stage: deepcopy(V3_SCHEMAS[v3_stage])
    for stage, v3_stage in V31_TO_V3_STAGE.items()
}

# The global selector receives the finalist union rather than a fixed shard.
global_selector = deepcopy(V3_SCHEMAS["lean_v3_batch_selector"])
global_selector["properties"]["candidate_assessments"].update(
    {"minItems": 1, "maxItems": 25}
)
global_selector["properties"]["candidate_assessments"]["items"]["properties"][
    "duplicate_or_sibling_of"
]["maxItems"] = 24
global_selector["properties"]["within_batch_duplicate_groups"]["maxItems"] = 25
global_selector["properties"]["within_batch_duplicate_groups"]["items"][
    "properties"
]["candidate_ids"]["maxItems"] = 25
global_selector["properties"]["selected"]["maxItems"] = 10
global_selector["properties"]["selected"]["items"]["properties"][
    "development_priority"
]["maximum"] = 10
STRICT_OUTPUT_SCHEMAS["lean_v31_global_selector"] = global_selector

editorial = deepcopy(V3_SCHEMAS["lean_v3_editorial_board"])
editorial["properties"]["judgments"].update({"minItems": 7, "maxItems": 14})
entry = editorial["properties"]["judgments"]["items"]
entry["properties"]["composition_grid_evidence"] = {
    "type": "string",
    "minLength": 1,
}
entry["properties"]["scale_and_embodiment_observation"] = {
    "type": "string",
    "minLength": 1,
}
entry["required"].extend(
    ["composition_grid_evidence", "scale_and_embodiment_observation"]
)
# A missing task judgment is always an error. An empty batch-level aside is not
# worth an expensive full-panel parse retry.
editorial["properties"]["batch_observations"] = {"type": "string"}
STRICT_OUTPUT_SCHEMAS["lean_v31_editorial_board"] = editorial


def _transport_schema(value: Any) -> Any:
    unsupported = {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    if isinstance(value, dict):
        return {
            key: _transport_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_transport_schema(item) for item in value]
    return value


OUTPUT_SCHEMAS = {
    stage: _transport_schema(schema)
    for stage, schema in STRICT_OUTPUT_SCHEMAS.items()
}


def _schema_validate(stage: str, value: dict[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(STRICT_OUTPUT_SCHEMAS[stage]).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = "$" + "".join(
        f"[{item}]" if isinstance(item, int) else f".{item}"
        for item in error.absolute_path
    )
    raise SchemaError(path, error.message)


def _validate_global_selector(value: dict[str, Any]) -> None:
    assessments = value["candidate_assessments"]
    assessed = [item["candidate_id"] for item in assessments]
    if len(assessed) != len(set(assessed)):
        raise SchemaError("$.candidate_assessments", "candidate IDs must be unique")
    assessed_set = set(assessed)
    selected = value["selected"]
    selected_ids = [item["candidate_id"] for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise SchemaError("$.selected", "selected candidate IDs must be unique")
    decisions = {item["candidate_id"]: item["decision"] for item in assessments}
    expected = {candidate_id for candidate_id, decision in decisions.items() if decision == "select"}
    if set(selected_ids) != expected:
        raise SchemaError(
            "$.selected", "selected briefs must match assessment select decisions"
        )
    priorities = sorted(item["development_priority"] for item in selected)
    if priorities != list(range(1, len(selected) + 1)):
        raise SchemaError(
            "$.selected", "development priorities must be contiguous from one"
        )
    if sum(item["optional_development_suggestion"] is not None for item in selected) > 2:
        raise SchemaError("$.selected", "at most two optional suggestions are allowed")
    strongest_by_candidate: dict[str, list[int]] = {}
    for group_index, group in enumerate(value["within_batch_duplicate_groups"]):
        strongest_by_candidate.setdefault(group["strongest_candidate_id"], []).append(
            group_index
        )
    for index, item in enumerate(assessments):
        siblings = set(item["duplicate_or_sibling_of"])
        if not siblings <= assessed_set - {item["candidate_id"]}:
            raise SchemaError(
                f"$.candidate_assessments[{index}].duplicate_or_sibling_of",
                "siblings must name other assessed finalists",
            )
        if (
            item["similarity_scope"] == "same_minimal_procedure"
            and item["derivative_of"] == "none"
        ):
            # The strongest representative of an explicit same-batch group is
            # not derivative of its weaker siblings. Every other member must
            # still identify the relevant source (normally ``same_batch``).
            eligible_groups = strongest_by_candidate.get(item["candidate_id"], [])
            is_strongest_representative = any(
                (
                    set(
                        value["within_batch_duplicate_groups"][group_index][
                            "candidate_ids"
                        ]
                    )
                    - {item["candidate_id"]}
                )
                <= siblings
                for group_index in eligible_groups
            )
            if not is_strongest_representative:
                raise SchemaError(
                    f"$.candidate_assessments[{index}].derivative_of",
                    "same-procedure similarity must identify a derivative source unless this candidate is the strongest representative of an explicit duplicate group",
                )
    for index, group in enumerate(value["within_batch_duplicate_groups"]):
        members = set(group["candidate_ids"])
        if not members <= assessed_set or group["strongest_candidate_id"] not in members:
            raise SchemaError(
                f"$.within_batch_duplicate_groups[{index}]",
                "duplicate group must reference assessed candidates and its strongest member",
            )


def parse_stage_output(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STRICT_OUTPUT_SCHEMAS:
        raise SchemaError("$.stage", f"unknown Lean V3.1 stage {stage!r}")
    value = (
        parse_json_object(data)
        if isinstance(data, (str, bytes, bytearray))
        else dict(data)
    )
    _schema_validate(stage, value)
    if stage == "lean_v31_global_selector":
        _validate_global_selector(value)
    elif stage == "lean_v31_editorial_board":
        pass
    else:
        parse_v3_stage_output(V31_TO_V3_STAGE[stage], value)
    return value


__all__ = ["OUTPUT_SCHEMAS", "STRICT_OUTPUT_SCHEMAS", "parse_stage_output"]
