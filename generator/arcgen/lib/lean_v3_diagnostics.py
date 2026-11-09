"""V1 diagnostics with the narrowly refined repeated-component semantics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .lean_diagnostics import combine_public_and_hidden, diagnose_arc_task


PAIR_ECONOMY_FLAG_CODES = {
    "over_90_percent_identical_inputs",
    "dihedral_equivalent_inputs",
    "repeated_training_evidence",
    "train_test_clone_or_leak",
}
INFORMATIONAL_CODES = {
    "repeated_internal_vocabulary",
    "uniform_canvas_margins",
}


def diagnose_arc_task_v3(
    task: dict[str, Any],
    *,
    repeated_evidence_min_component_cells: int = 9,
) -> dict[str, Any]:
    """Split vocabulary recurrence from evidence reuse without changing hard gates."""

    base = diagnose_arc_task(task)
    if not base["hard_valid"]:
        return {
            **base,
            "diagnostic_version": "lean-v3.0.0",
            "escalation_trigger_codes": [],
        }

    refined: list[dict[str, Any]] = []
    for flag in base["flags"]:
        code = flag["code"]
        if code == "repeated_component_or_subgrid":
            locations = list(flag.get("locations", []))
            examples = flag.get("examples", [])
            refined.append(
                {
                    **deepcopy(flag),
                    "code": "repeated_internal_vocabulary",
                    "severity": "informational",
                    "detail": (
                        "exact component vocabulary recurs within the episode; "
                        "this is informational unless separate evidence-reuse "
                        "or leakage diagnostics apply"
                    ),
                }
            )
            training_groups = []
            for group in examples:
                train_occurrences = [
                    item
                    for item in group
                    if str(item.get("pair", "")).startswith("train_")
                ]
                if (
                    len({item["pair"] for item in train_occurrences}) >= 2
                    and max(
                        (
                            int(item.get("cell_count", 0))
                            for item in train_occurrences
                        ),
                        default=0,
                    )
                    >= repeated_evidence_min_component_cells
                ):
                    training_groups.append(train_occurrences)
            if training_groups:
                refined.append(
                    {
                        "code": "repeated_training_evidence",
                        "locations": sorted(
                            {
                                item["pair"]
                                for group in training_groups
                                for item in group
                            }
                        ),
                        "severity": "semantic_review",
                        "detail": (
                            f"{len(training_groups)} large component signature(s) "
                            "recur across training inputs and require a semantic "
                            "pair-economy assessment"
                        ),
                        "examples": training_groups[:10],
                    }
                )
            continue
        if code == "repeated_train_test_grid":
            refined.append(
                {
                    **deepcopy(flag),
                    "code": "train_test_clone_or_leak",
                    "severity": "strong_concern",
                    "detail": "a test input is a verbatim copy of another episode input",
                }
            )
            continue
        if code in {"exact_duplicate_pair", "exact_duplicate_training_input"}:
            refined.append(
                {
                    **deepcopy(flag),
                    "code": "repeated_training_evidence",
                    "severity": "semantic_review",
                }
            )
            continue
        refined.append(
            {
                **deepcopy(flag),
                "severity": (
                    "strong_concern"
                    if code
                    in {
                        "over_90_percent_identical_inputs",
                        "dihedral_equivalent_inputs",
                    }
                    else "informational"
                ),
            }
        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for flag in refined:
        identity = (
            flag["code"],
            tuple(flag.get("locations", [])),
            flag.get("detail", ""),
        )
        if identity not in seen:
            unique.append(flag)
            seen.add(identity)
    codes = {flag["code"] for flag in unique}
    return {
        **base,
        "flags": unique,
        "diagnostic_version": "lean-v3.0.0",
        "flag_category_counts": {
            "repeated_internal_vocabulary": sum(
                flag["code"] == "repeated_internal_vocabulary" for flag in unique
            ),
            "repeated_training_evidence": sum(
                flag["code"] == "repeated_training_evidence" for flag in unique
            ),
            "train_test_clone_or_leak": sum(
                flag["code"] == "train_test_clone_or_leak" for flag in unique
            ),
        },
        "pair_economy_review_codes": sorted(codes & PAIR_ECONOMY_FLAG_CODES),
        # The standing primary verifier sees every flag. A diagnostic alone
        # never schedules an extra model call.
        "escalation_trigger_codes": [],
    }


def internal_vocabulary_only(diagnostics: dict[str, Any]) -> bool:
    codes = {flag["code"] for flag in diagnostics.get("flags", [])}
    return bool(codes) and codes <= INFORMATIONAL_CODES


__all__ = [
    "INFORMATIONAL_CODES",
    "PAIR_ECONOMY_FLAG_CODES",
    "combine_public_and_hidden",
    "diagnose_arc_task_v3",
    "internal_vocabulary_only",
]
