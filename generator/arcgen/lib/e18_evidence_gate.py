"""Live dominant-answer evidence gate for the E18 candidate path."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .e2_live_streaming import IdempotentSlateBudget, _blind_one
from .e18_calls import E18AttemptError
from .e19_dominant_answer import corrected_blind_trigger
from .v41_experiments import V41Runtime
from .v41_supported_departure import _blind_call, _record_stage, _wrap_blind

EVIDENCE_CLEAR = "evidence_clear"
EVIDENCE_CLEAR_AFTER_REVIEW = "evidence_clear_after_review"
EVIDENCE_HOLD = "evidence_hold"
EVIDENCE_AMBIGUOUS = "ambiguous"
EVIDENCE_POLICY = 'grounded-difficulty-retention-v1'


class E18EvidenceError(RuntimeError):
    """Raised when a blind record cannot support a safe disposition."""


def _hidden_outputs(record: Mapping[str, Any]) -> list[Any]:
    realization = record.get("realization")
    metadata = (
        realization.get("authoring_metadata")
        if isinstance(realization, Mapping)
        else None
    )
    outputs = (
        metadata.get("expected_test_outputs") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(outputs, list) or not outputs:
        raise E18EvidenceError("candidate has no hidden expected outputs")
    return deepcopy(outputs)


def _response(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    nested = value.get("response")
    if isinstance(nested, Mapping):
        return nested
    if "predicted_test_outputs" in value:
        return value
    return None


def _gate(record: Mapping[str, Any], wrapped: Mapping[str, Any]) -> dict[str, Any]:
    response = _response(wrapped)
    if response is None:
        return {
            "old_serious_predicate": False,
            "primary_output_present": False,
            "primary_differs_from_hidden": False,
            "raw_serious_flag": False,
            "unsupported_guessing_flag": False,
            "proposed_review_required": True,
            "trigger_reasons": ["missing_blind_response"],
            "automatic_rejection": False,
        }
    value = corrected_blind_trigger(response, _hidden_outputs(record))
    value['evidence_policy'] = EVIDENCE_POLICY
    if value['primary_differs_from_hidden']:
        value['proposed_review_required'] = True
        value['trigger_reasons'] = [*value['trigger_reasons'], 'primary_answer_mismatch']
    return value


def _persist_metadata(
    runtime: V41Runtime,
    task_id: str,
    *,
    evidence_status: str,
    route: Mapping[str, Any],
    force_terminal_hold: bool = False,
) -> None:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        prior = record.get("answer_evidence")
        value = {
            "status": evidence_status,
            "corrected_e19_trigger": True,
            "evidence_policy": EVIDENCE_POLICY,
            **deepcopy(dict(route)),
        }
        if prior is not None and prior != value:
            # Evidence may progress from review_required to a terminal result;
            # retain every earlier value in a separate history.
            record.setdefault("answer_evidence_history", []).append(deepcopy(prior))
        record["answer_evidence"] = value
        if force_terminal_hold:
            prior_status = record.get("status")
            record["pre_evidence_hold_status"] = prior_status
            record["status"] = EVIDENCE_HOLD
            record.setdefault("stage_history", []).append(
                {
                    "stage": "answer_evidence",
                    "status": EVIDENCE_HOLD,
                    "at_epoch_seconds": __import__("time").time(),
                }
            )
        runtime.save()


def run_primary(runtime: V41Runtime, task_id: str) -> bool:
    """Run or reuse the primary blind call and apply the corrected trigger.

    Returns true when no conditional blind work remains.
    """

    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("blind_results"):
        return record.get("status") != "blind_needs_resolution"
    try:
        response = _blind_call(runtime, task_id, "blind-primary")
    except E18AttemptError as exc:
        _record_stage(
            runtime,
            task_id,
            "blind",
            status="blind_operational_failure",
            blind_error=str(exc),
        )
        _persist_metadata(
            runtime,
            task_id,
            evidence_status=EVIDENCE_HOLD,
            route={"trigger_reasons": ["primary_blind_operational_failure"], "hold_kind": "operational"},
            force_terminal_hold=True,
        )
        return True
    record = deepcopy(runtime.state["semantic_records"][task_id])
    wrapped = _wrap_blind(record, response)
    gate = _gate(record, wrapped)
    wrapped["e18_corrected_evidence_trigger"] = deepcopy(gate)
    needs_review = gate["proposed_review_required"] is True
    status = "blind_needs_resolution" if needs_review else "blind_pass"
    _record_stage(runtime, task_id, "blind", status=status, blind_results=[wrapped])
    _persist_metadata(
        runtime,
        task_id,
        evidence_status="review_required" if needs_review else EVIDENCE_CLEAR,
        route={
            "primary_gate": gate,
            "conditional_review_used": False,
            "resolution": None if needs_review else "ordinary_primary_clear",
        },
    )
    return not needs_review


def _terminal_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    results = record.get("blind_results")
    if not isinstance(results, list) or not results:
        return {
            "proposed_review_required": True,
            "trigger_reasons": ["missing_terminal_blind_result"],
        }
    return _gate(record, results[-1])


def resolve(runtime, task_id, budget):
    from arcgen.batch.gate import resolve as revised
    return revised(runtime, task_id, budget)


def evidence_is_clear(record: Mapping[str, Any]) -> bool:
    evidence = record.get("answer_evidence")
    return isinstance(evidence, Mapping) and evidence.get("status") in {
        EVIDENCE_CLEAR,
        EVIDENCE_CLEAR_AFTER_REVIEW,
    }


def terminal_ambiguity_profile(record: Mapping[str, Any]) -> dict[str, Any]:
    """Separate preliminary blind alarms from terminal ambiguity evidence.

    Historical reports must retain every raw alarm, but a completed conditional
    review is the authoritative aggregate disposition.  Older records without
    terminal evidence retain the legacy raw-alarm fallback explicitly.
    """

    blind = [
        row for row in (record.get("blind_results") or []) if isinstance(row, Mapping)
    ]
    raw_alarms = [
        bool((row.get("response") or {}).get("serious_different_output_alternative"))
        and row.get("primary_matches_hidden") is False
        for row in blind
    ]
    evidence = record.get("answer_evidence") or {}
    adjudication = (
        evidence.get("ambiguity_adjudication")
        or record.get("ambiguity_adjudication")
        or {}
    )
    evidence_status = str(evidence.get("status") or "")
    resolution = str(evidence.get("resolution") or "")
    decision = str(adjudication.get("decision") or "")
    comparable = adjudication.get("comparably_simple_different_output_rule")

    if (
        evidence_status == EVIDENCE_AMBIGUOUS
        and resolution == "concrete_ambiguity_confirmed"
    ) or comparable is True or decision == "ambiguous":
        effective: bool | None = True
        basis = "terminal_ambiguity_confirmed"
    elif (
        evidence_status in {EVIDENCE_CLEAR, EVIDENCE_CLEAR_AFTER_REVIEW}
        or (decision == "clear" and comparable is False)
    ):
        effective = False
        basis = "terminal_clear"
    elif evidence_status in {EVIDENCE_HOLD, EVIDENCE_AMBIGUOUS}:
        effective = None
        basis = "terminal_unresolved"
    elif raw_alarms:
        effective = any(raw_alarms)
        basis = "legacy_raw_alarm_fallback"
    else:
        effective = False
        basis = "no_alarm_or_terminal_ambiguity"

    second = None
    if len(blind) >= 2:
        second_response = blind[1].get("response") or {}
        second = {
            "serious_different_output_alternative": second_response.get(
                "serious_different_output_alternative"
            ),
            "primary_matches_hidden": blind[1].get("primary_matches_hidden"),
        }
    return {
        "raw_blind_ambiguity_alarm": any(raw_alarms),
        "raw_blind_alarm_count": sum(raw_alarms),
        "second_review_outcome": second,
        "adjudication_outcome": deepcopy(dict(adjudication))
        if isinstance(adjudication, Mapping)
        else None,
        "effective_terminal_ambiguity": effective,
        "effective_basis": basis,
        "evidence_status": evidence_status or None,
        "evidence_resolution": resolution or None,
    }


__all__ = [
    "EVIDENCE_AMBIGUOUS",
    "EVIDENCE_CLEAR",
    "EVIDENCE_CLEAR_AFTER_REVIEW",
    "EVIDENCE_HOLD",
    "E18EvidenceError",
    "evidence_is_clear",
    "terminal_ambiguity_profile",
    "resolve",
    "run_primary",
]
