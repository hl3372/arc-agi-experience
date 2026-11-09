"""Conservative, non-mutating ``basic_valid`` curriculum classification.

``basic_valid`` is deliberately independent of intrinsic editorial quality.
It identifies an editorially non-core/non-reserve episode that is mechanically
valid, exactly verified, routinely solved from scratch, and rejected only for
editorial thinness or familiarity.  It must never turn a correctness,
inferability, duplicate, or operational failure into accepted work.

Historical pipeline statuses remain frozen.  New experiments may attach the
returned ``curriculum_status`` beside the existing ``intrinsic_quality`` and
``portfolio_placement`` fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .validation import validate_arc_task
from .factory_review_policy import required_boards
from .factory_policy import validate_review_policy

BASIC_VALID = "basic_valid"
NOT_BASIC_VALID = "not_basic_valid"
NOT_APPLICABLE = "not_applicable"

BASIC_VALID_CURRICULUM_STATUSES = frozenset(
    {BASIC_VALID, NOT_BASIC_VALID, NOT_APPLICABLE}
)


def _empty_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and not value


def _strict_technical_clear(record: Mapping[str, Any]) -> bool:
    results = record.get("technical_results")
    if not isinstance(results, list) or not results:
        return False
    terminal = results[-1]
    if not isinstance(terminal, Mapping):
        return False
    response = terminal.get("response")
    if not isinstance(response, Mapping):
        return False
    return (
        terminal.get("hidden_output_match") is True
        and response.get("status") == "pass"
        and response.get("mechanically_correct") is True
        and response.get("test_supported") is True
        and response.get("one_answer_supported") is True
        and response.get("plan_fidelity") == "pass"
        and response.get("evidence_sufficiency") == "sufficient"
        and response.get("pair_economy") != "fail"
        and response.get("composition_preserved") is True
        and _empty_sequence(response.get("arbitrary_exceptions"))
        and _empty_sequence(response.get("unsupported_test_semantics"))
        and _empty_sequence(response.get("unapproved_realizer_decisions"))
        and _empty_sequence(terminal.get("unresolved_pair_economy_flags", []))
    )


def _routine_blind_clear(record: Mapping[str, Any]) -> bool:
    """Require the ordinary first blind opinion to solve exactly and cleanly.

    A later repair, adjudication, or second opinion may establish ordinary task
    validity, but it does not establish the deliberately stricter
    routine-solvability proxy used for the basic/easier collection.
    """

    results = record.get("blind_results")
    if not isinstance(results, list) or not results:
        return False
    primary = results[0]
    terminal = results[-1]
    if not isinstance(primary, Mapping) or not isinstance(terminal, Mapping):
        return False
    primary_response = primary.get("response")
    terminal_response = terminal.get("response")
    if not isinstance(primary_response, Mapping) or not isinstance(
        terminal_response, Mapping
    ):
        return False
    return (
        len(results) == 1
        and record.get("repair_count", 0) == 0
        and primary.get("primary_matches_hidden") is True
        and primary.get("serious_different_output_alternative") is False
        and primary_response.get("serious_different_output_alternative") is False
        and primary_response.get("test_requires_unsupported_guessing") is False
        and primary_response.get("confidence") == "high"
        and terminal.get("primary_matches_hidden") is True
        and terminal.get("serious_different_output_alternative") is False
        and terminal_response.get("serious_different_output_alternative") is False
        and terminal_response.get("test_requires_unsupported_guessing") is False
    )


def _editorial_thinness_only(record: Mapping[str, Any], *, review_policy=None) -> bool:
    judgments = record.get("editorial_judgments")
    if not required_boards(record, review_policy):
        return False
    if any(not isinstance(item, Mapping) for item in judgments):
        return False
    if any(
        item.get("same_minimal_procedure") is True
        or item.get("portfolio_recommendation") == "same_procedure_duplicate"
        for item in judgments
    ):
        return False
    if any(
        item.get("overall_quality") not in {"real_quality", "exceptional"}
        for item in judgments
    ):
        return False
    if any(
        item.get("intrinsic_recommendation")
        not in {"core_quality", "reserve_quality"}
        for item in judgments
    ):
        return False
    return any(
        item.get("novel_concept") == "weak"
        or item.get("functional_composition") == "thin"
        for item in judgments
    )


def assess_basic_valid(record: Mapping[str, Any], *, review_policy=None) -> dict[str, Any]:
    """Return a parallel curriculum disposition without mutating ``record``."""

    validate_review_policy(review_policy)
    intrinsic = record.get("intrinsic_quality")
    if intrinsic != "rejected_quality":
        return {
            "curriculum_status": NOT_APPLICABLE,
            "intrinsic_quality_preserved": intrinsic,
            "routine_solvability_proxy": False,
            "reasons": [
                "basic_valid is only a parallel disposition for editorially rejected work"
            ],
        }

    failures: list[str] = []
    task = record.get("complete_task")
    validation = validate_arc_task(task, require_test_outputs=True)
    if not validation.valid:
        failures.append("ARC task is mechanically invalid")
    if record.get("status") != "editorial_complete":
        failures.append("task lacks terminal editorial completion")
    if (record.get("deterministic_diagnostics") or {}).get("hard_valid") is not True:
        failures.append("deterministic diagnostics are not hard-valid")
    if (record.get("plan_fidelity") or {}).get("passed") is not True:
        failures.append("plan fidelity did not pass")
    if not _strict_technical_clear(record):
        failures.append("strict exact technical verification did not pass")
    routine = _routine_blind_clear(record)
    if not routine:
        failures.append("primary blind solve was not exact, high-confidence, and clear")
    if not _editorial_thinness_only(record, review_policy=review_policy):
        failures.append(
            "editorial rejection was not solely a real-quality thinness/familiarity case"
        )

    return {
        "curriculum_status": BASIC_VALID if not failures else NOT_BASIC_VALID,
        "intrinsic_quality_preserved": intrinsic,
        "routine_solvability_proxy": routine,
        "reasons": failures,
    }


__all__ = [
    "BASIC_VALID",
    "BASIC_VALID_CURRICULUM_STATUSES",
    "NOT_APPLICABLE",
    "NOT_BASIC_VALID",
    "assess_basic_valid",
]

