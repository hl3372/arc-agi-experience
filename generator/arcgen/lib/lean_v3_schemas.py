"""V3 output contracts: V1 schemas plus two narrow editorial extensions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .lean_schemas import (
    STRICT_OUTPUT_SCHEMAS as V1_STRICT_OUTPUT_SCHEMAS,
    parse_stage_output as parse_v1_stage_output,
)
from .schemas import SchemaError, parse_json_object


TEXT = {"type": "string", "minLength": 1}
BOOLEAN = {"type": "boolean"}
CONCEPT_AXIS = {"type": "string", "enum": ["weak", "solid", "standout"]}
COMPOSITION_AXIS = {"type": "string", "enum": ["thin", "solid", "standout"]}


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


V3_TO_V1_STAGE = {
    "lean_v3_idea_search": "lean_idea_search",
    "lean_v3_batch_selector": "lean_batch_selector",
    "lean_v3_episode_architect": "lean_episode_architect",
    "lean_v3_task_realizer": "lean_task_realizer",
    "lean_v3_technical_verifier": "lean_technical_verifier",
    "lean_v3_blind_solver": "lean_blind_solver",
    "lean_v3_ambiguity_adjudicator": "lean_ambiguity_adjudicator",
    "lean_v3_targeted_repair": "lean_targeted_repair",
    "lean_v3_editorial_board": "lean_editorial_board",
}


STRICT_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    v3_stage: deepcopy(V1_STRICT_OUTPUT_SCHEMAS[v1_stage])
    for v3_stage, v1_stage in V3_TO_V1_STAGE.items()
}

# The selector keeps the V1 contract and adds only the distinction needed for
# archive tiers plus one nullable, candidate-specific development suggestion.
selector = STRICT_OUTPUT_SCHEMAS["lean_v3_batch_selector"]
assessment = selector["properties"]["candidate_assessments"]["items"]
assessment["properties"]["similarity_scope"] = {
    "type": "string",
    "enum": [
        "none",
        "broad_theme",
        "close_sibling",
        "same_minimal_procedure",
    ],
}
assessment["required"].append("similarity_scope")
brief = selector["properties"]["selected"]["items"]
brief["properties"]["optional_development_suggestion"] = {
    "anyOf": [{"type": "null"}, TEXT],
}
brief["required"].append("optional_development_suggestion")

EDITORIAL_ENTRY = _object(
    {
        "task_id": TEXT,
        "novel_concept": CONCEPT_AXIS,
        "functional_composition": COMPOSITION_AXIS,
        "overall_quality": {
            "type": "string",
            "enum": ["below_real_bar", "real_quality", "exceptional"],
        },
        "same_minimal_procedure": BOOLEAN,
        "theme_is_common": BOOLEAN,
        "addition_is_excessive": BOOLEAN,
        "intrinsic_recommendation": {
            "type": "string",
            "enum": ["core_quality", "reserve_quality", "rejected_quality"],
        },
        "portfolio_recommendation": {
            "type": "string",
            "enum": [
                "include_now",
                "hold_crowded",
                "same_procedure_duplicate",
            ],
        },
        "reason": TEXT,
    }
)
STRICT_OUTPUT_SCHEMAS["lean_v3_editorial_board"] = _object(
    {
        "board_id": TEXT,
        "judgments": _array(EDITORIAL_ENTRY, minimum=7, maximum=10),
        "batch_observations": TEXT,
    }
)


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
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)


def parse_stage_output(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in STRICT_OUTPUT_SCHEMAS:
        raise SchemaError("$.stage", f"unknown Lean V3 stage {stage!r}")
    value = (
        parse_json_object(data)
        if isinstance(data, (str, bytes, bytearray))
        else dict(data)
    )
    _schema_validate(stage, value)

    if stage == "lean_v3_batch_selector":
        # Reuse every V1 semantic validation after removing V3-only fields.
        v1_value = deepcopy(value)
        for item in v1_value["candidate_assessments"]:
            item.pop("similarity_scope")
        for item in v1_value["selected"]:
            item.pop("optional_development_suggestion")
        parse_v1_stage_output("lean_batch_selector", v1_value)
        suggestion_count = sum(
            item["optional_development_suggestion"] is not None
            for item in value["selected"]
        )
        if suggestion_count > 2:
            raise SchemaError(
                "$.selected",
                "at most two briefs may contain optional development suggestions",
            )
        return value

    if stage == "lean_v3_editorial_board":
        return value

    parse_v1_stage_output(V3_TO_V1_STAGE[stage], value)
    return value


__all__ = ["OUTPUT_SCHEMAS", "STRICT_OUTPUT_SCHEMAS", "parse_stage_output"]
