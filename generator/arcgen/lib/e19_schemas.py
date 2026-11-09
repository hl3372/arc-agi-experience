"""Structured outputs for the E19 dominant-answer calibration experiment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .schemas import SchemaError, parse_grid, parse_json_object
from .v41_schemas import OUTPUT_SCHEMAS as V41_OUTPUT_SCHEMAS
from .v41_schemas import parse_stage_output as parse_v41_stage_output


TEXT = {"type": "string"}
NONEMPTY_TEXT = {"type": "string", "minLength": 1}
BOOLEAN = {"type": "boolean"}
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


def _array(items: Any, *, maximum: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": "array", "items": deepcopy(items)}
    if maximum is not None:
        value["maxItems"] = maximum
    return value


PAIR_EVIDENCE = _object({"pair": NONEMPTY_TEXT, "evidence": NONEMPTY_TEXT})
COMPETITOR = _object(
    {
        "rule": NONEMPTY_TEXT,
        "fits_every_training_pair": BOOLEAN,
        "per_pair_fit": _array(PAIR_EVIDENCE, maximum=10),
        "predicted_test_outputs": _array(GRID, maximum=3),
        "simplicity_comparison": {
            "type": "string",
            "enum": ["proposal_simpler", "comparable", "competitor_simpler"],
        },
        "naturalness_comparison": {
            "type": "string",
            "enum": ["proposal_more_natural", "comparable", "competitor_more_natural"],
        },
        "extra_assumptions_or_exceptions": _array(TEXT, maximum=10),
        "missing_distinguishing_evidence": NONEMPTY_TEXT,
    }
)

DOMINANT_REVIEW = _object(
    {
        "task_id": NONEMPTY_TEXT,
        "reviewer_id": NONEMPTY_TEXT,
        "verdict": {
            "type": "string",
            "enum": ["dominant_answer", "comparable_different_answer", "inconclusive"],
        },
        "supporting_rule": TEXT,
        "supporting_pair_evidence": _array(PAIR_EVIDENCE, maximum=10),
        "proposed_output_derivation": TEXT,
        "strongest_competitor": {"anyOf": [COMPETITOR, {"type": "null"}]},
        "why_proposal_is_or_is_not_dominant": NONEMPTY_TEXT,
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    }
)

ADJUDICATION = _object(
    {
        "task_id": NONEMPTY_TEXT,
        "adjudicator_id": NONEMPTY_TEXT,
        "decision": {
            "type": "string",
            "enum": ["clear_dominant", "repairable", "quarantine", "inconclusive"],
        },
        "supporting_rule": TEXT,
        "supporting_pair_evidence": _array(PAIR_EVIDENCE, maximum=10),
        "proposed_output_derivation": TEXT,
        "surviving_competitor": {"anyOf": [COMPETITOR, {"type": "null"}]},
        "why_proposal_is_or_is_not_dominant": NONEMPTY_TEXT,
        "repair_instruction": {"anyOf": [NONEMPTY_TEXT, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    }
)

OUTPUT_SCHEMAS = {
    **deepcopy(V41_OUTPUT_SCHEMAS),
    "e19_dominant_review": DOMINANT_REVIEW,
    "e19_dominant_adjudication": ADJUDICATION,
}


def _validate(stage: str, value: Mapping[str, Any]) -> dict[str, Any]:
    schema = OUTPUT_SCHEMAS[stage]
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path)
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(f"[{part!r}]" for part in error.path)
        raise SchemaError(path, error.message)
    for field in ("strongest_competitor", "surviving_competitor"):
        competitor = value.get(field)
        if isinstance(competitor, Mapping):
            for index, grid in enumerate(competitor["predicted_test_outputs"]):
                parse_grid(grid, f"$.{field}.predicted_test_outputs[{index}]")
    return dict(value)


def parse_stage_output(
    stage: str, data: str | bytes | bytearray | Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in {"e19_dominant_review", "e19_dominant_adjudication"}:
        return parse_v41_stage_output(stage, data)
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else dict(data)
    return _validate(stage, value)


def _filled(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _covers_all_pairs(value: Any, count: int) -> bool:
    if not isinstance(value, list):
        return False
    expected = {f"train_{index}" for index in range(1, count + 1)}
    actual = {
        str(item.get("pair", "")).strip().casefold()
        for item in value
        if isinstance(item, Mapping) and _filled(item.get("evidence"))
    }
    return actual == expected


def concrete_competitor(
    value: Any,
    *,
    proposed_outputs: Sequence[Any],
    training_pair_count: int,
) -> tuple[bool, list[str]]:
    """Return whether a response contains the required serious competitor."""

    if not isinstance(value, Mapping):
        return False, ["no competitor supplied"]
    reasons: list[str] = []
    if value.get("fits_every_training_pair") is not True:
        reasons.append("competitor not affirmed to fit every training pair")
    if not _covers_all_pairs(value.get("per_pair_fit"), training_pair_count):
        reasons.append("competitor evidence does not cover every training pair")
    outputs = value.get("predicted_test_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(proposed_outputs):
        reasons.append("competitor output is incomplete")
    elif list(outputs) == list(proposed_outputs):
        reasons.append("competitor output is not different")
    if value.get("simplicity_comparison") not in {"comparable", "competitor_simpler"}:
        reasons.append("competitor is not comparably simple")
    if value.get("naturalness_comparison") not in {"comparable", "competitor_more_natural"}:
        reasons.append("competitor is not comparably natural")
    assumptions = value.get("extra_assumptions_or_exceptions")
    if not isinstance(assumptions, list) or any(_filled(item) for item in assumptions):
        reasons.append("competitor requires extra assumptions or exceptions")
    if not _filled(value.get("missing_distinguishing_evidence")):
        reasons.append("missing distinguishing evidence was not identified")
    return not reasons, reasons


def normalize_dominant_review(
    response: Mapping[str, Any],
    *,
    proposed_outputs: Sequence[Any],
    training_pair_count: int,
) -> dict[str, Any]:
    raw = str(response["verdict"])
    competitor_ok, competitor_reasons = concrete_competitor(
        response.get("strongest_competitor"),
        proposed_outputs=proposed_outputs,
        training_pair_count=training_pair_count,
    )
    support_ok = (
        _filled(response.get("supporting_rule"))
        and _filled(response.get("proposed_output_derivation"))
        and _covers_all_pairs(response.get("supporting_pair_evidence"), training_pair_count)
    )
    normalized = raw
    reasons: list[str] = []
    if raw == "comparable_different_answer" and not competitor_ok:
        normalized = "inconclusive"
        reasons = competitor_reasons
    elif raw == "dominant_answer" and competitor_ok:
        normalized = "comparable_different_answer"
        reasons = ["a complete comparably simple and natural different answer was supplied"]
    elif raw == "dominant_answer" and not support_ok:
        normalized = "inconclusive"
        reasons = ["dominance claim does not support every training pair and the proposed output"]
    return {
        "raw_verdict": raw,
        "normalized_verdict": normalized,
        "support_evidence_complete": support_ok,
        "concrete_competitor": competitor_ok,
        "normalization_reasons": reasons,
    }


def normalize_adjudication(
    response: Mapping[str, Any],
    *,
    proposed_outputs: Sequence[Any],
    training_pair_count: int,
) -> dict[str, Any]:
    raw = str(response["decision"])
    competitor_ok, competitor_reasons = concrete_competitor(
        response.get("surviving_competitor"),
        proposed_outputs=proposed_outputs,
        training_pair_count=training_pair_count,
    )
    support_ok = (
        _filled(response.get("supporting_rule"))
        and _filled(response.get("proposed_output_derivation"))
        and _covers_all_pairs(response.get("supporting_pair_evidence"), training_pair_count)
    )
    normalized = raw
    reasons: list[str] = []
    if raw == "clear_dominant" and (not support_ok or competitor_ok):
        normalized = "inconclusive"
        reasons.append("clear decision lacks complete support or retains a serious competitor")
    elif raw in {"repairable", "quarantine"} and not competitor_ok:
        normalized = "inconclusive"
        reasons.extend(competitor_reasons)
    elif raw == "repairable" and not _filled(response.get("repair_instruction")):
        normalized = "inconclusive"
        reasons.append("repairable decision lacks a concrete repair instruction")
    elif raw != "repairable" and response.get("repair_instruction") is not None:
        normalized = "inconclusive"
        reasons.append("repair instruction is only valid for a repairable decision")
    return {
        "raw_decision": raw,
        "normalized_decision": normalized,
        "support_evidence_complete": support_ok,
        "concrete_competitor": competitor_ok,
        "normalization_reasons": reasons,
    }


__all__ = [
    "ADJUDICATION",
    "DOMINANT_REVIEW",
    "OUTPUT_SCHEMAS",
    "concrete_competitor",
    "normalize_adjudication",
    "normalize_dominant_review",
    "parse_stage_output",
]

