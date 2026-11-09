"""Non-gating coverage and promotion analysis for ARCGEN V4.1.

These helpers never decide whether an ARC task is intrinsically good.  They
only summarize whether an isolated experimental mechanism has enough authentic,
diverse, non-inferior evidence to be presented to the user for a promotion
decision.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence


class V41AuditError(ValueError):
    pass


POSITIVE_STATUS = {
    "supported_departure": "genuine",
    "episode_joint_inference": "genuine",
    "abstraction_led": "genuine",
}


def property_status(judgment: Mapping[str, Any], property_name: str) -> str:
    if property_name not in POSITIVE_STATUS:
        raise V41AuditError(f"unknown semantic property {property_name!r}")
    assessment = judgment.get(property_name)
    if not isinstance(assessment, Mapping) or not isinstance(assessment.get("status"), str):
        raise V41AuditError(f"missing {property_name} assessment")
    return str(assessment["status"])


def genuine_task_ids(
    judgments: Sequence[Mapping[str, Any]],
    property_name: str,
) -> list[str]:
    """Return audited positives, independent of source or experimental arm."""

    wanted = POSITIVE_STATUS.get(property_name)
    if wanted is None:
        raise V41AuditError(f"unknown semantic property {property_name!r}")
    ids = [
        str(item["task_id"])
        for item in judgments
        if property_status(item, property_name) == wanted
    ]
    if len(ids) != len(set(ids)):
        raise V41AuditError("mechanism audit contains duplicate task IDs")
    return ids


def _components(
    ids: Sequence[str], same_procedure_pairs: Sequence[Mapping[str, Any]]
) -> list[set[str]]:
    parent = {task_id: task_id for task_id in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for pair in same_procedure_pairs:
        left, right = str(pair["task_id_a"]), str(pair["task_id_b"])
        if left in parent and right in parent:
            union(left, right)
    groups: dict[str, set[str]] = {}
    for task_id in ids:
        groups.setdefault(find(task_id), set()).add(task_id)
    return list(groups.values())


def mechanism_diversity_summary(
    *,
    positive_task_ids: Sequence[str],
    diversity_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Count mechanisms without treating same-procedure siblings as successes."""

    positive = set(positive_task_ids)
    clusters = diversity_audit.get("clusters")
    pairs = diversity_audit.get("same_procedure_pairs", [])
    if not isinstance(clusters, list) or not clusters:
        raise V41AuditError("diversity audit has no clusters")
    cluster_members = [set(map(str, item["member_task_ids"])) for item in clusters]
    flattened = set().union(*cluster_members)
    if flattened != positive:
        raise V41AuditError("diversity clusters must cover exactly the audited positives")
    if sum(len(item) for item in cluster_members) != len(flattened):
        raise V41AuditError("a positive task belongs to multiple clusters")
    components = _components(sorted(positive), pairs)
    representatives = [min(component) for component in components]
    representative_set = set(representatives)
    effective_cluster_sizes = [len(members & representative_set) for members in cluster_members]
    effective_cluster_sizes = [size for size in effective_cluster_sizes if size]
    unique_count = len(representatives)
    largest = max(effective_cluster_sizes, default=0)
    return {
        "raw_positive_count": len(positive),
        "same_procedure_sibling_count": len(positive) - unique_count,
        "unique_positive_mechanism_instances": unique_count,
        "unique_representative_task_ids": sorted(representatives),
        "material_cluster_count": len(effective_cluster_sizes),
        "effective_cluster_sizes": effective_cluster_sizes,
        "largest_cluster_count": largest,
        "largest_cluster_fraction": round(largest / unique_count, 6) if unique_count else None,
        "same_procedure_tasks_counted_as_separate_successes": False,
        "open_ended_cluster_descriptions": [
            str(item["open_ended_description"]) for item in clusters
        ],
    }


def matched_noninferiority(
    *,
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the provisional V4.1 quality guards to matched aggregate results."""

    required = {
        "deterministic_validity_rate",
        "technical_pass_rate",
        "hidden_output_exact_match_rate",
        "serious_ambiguity_rate",
        "core_quality_yield",
        "core_or_reserve_yield",
        "below_real_bar_count",
        "completed_task_count",
    }
    for arm, value in (("control", control), ("treatment", treatment)):
        missing = required - set(value)
        if missing:
            raise V41AuditError(f"{arm} non-inferiority metrics missing {sorted(missing)}")
    ambiguity_count_control = round(
        float(control["serious_ambiguity_rate"]) * int(control["completed_task_count"])
    )
    ambiguity_count_treatment = round(
        float(treatment["serious_ambiguity_rate"]) * int(treatment["completed_task_count"])
    )
    ambiguity_count_increase = ambiguity_count_treatment - ambiguity_count_control
    ambiguity_rate_increase = (
        float(treatment["serious_ambiguity_rate"])
        - float(control["serious_ambiguity_rate"])
    )
    gates = {
        "deterministic_validity": float(treatment["deterministic_validity_rate"])
        + 1e-9 >= float(control["deterministic_validity_rate"]),
        "technical_pass": float(treatment["technical_pass_rate"])
        + 1e-9 >= float(control["technical_pass_rate"]),
        "hidden_output_exact_match": float(treatment["hidden_output_exact_match_rate"])
        + 1e-9 >= float(control["hidden_output_exact_match_rate"]),
        "core_yield_80_percent": float(treatment["core_quality_yield"])
        + 1e-9 >= 0.8 * float(control["core_quality_yield"]),
        "retained_yield_90_percent": float(treatment["core_or_reserve_yield"])
        + 1e-9 >= 0.9 * float(control["core_or_reserve_yield"]),
        "serious_ambiguity": ambiguity_count_increase <= 1
        and ambiguity_rate_increase <= 0.10 + 1e-9,
        "below_real_bar_concentration": int(treatment["below_real_bar_count"])
        <= int(control["below_real_bar_count"]) + 1,
    }
    return {
        "control": deepcopy(dict(control)),
        "treatment": deepcopy(dict(treatment)),
        "ambiguity_count_increase": ambiguity_count_increase,
        "ambiguity_rate_increase": round(ambiguity_rate_increase, 6),
        "gates": gates,
        "all_provisional_model_gates_passed": all(gates.values()),
        "human_review_required": True,
    }


def semantic_promotion_assessment(
    *,
    property_name: str,
    independent_slate_count: int,
    treatment_authoring_candidate_count: int,
    matched_control_completed_count: int,
    judgments: Sequence[Mapping[str, Any]],
    diversity_audit: Mapping[str, Any],
    noninferiority: Mapping[str, Any],
    human_review_complete: bool,
    human_control_materially_preferred: bool | None,
    minimums: Mapping[str, Any],
) -> dict[str, Any]:
    positives = genuine_task_ids(judgments, property_name)
    diversity = mechanism_diversity_summary(
        positive_task_ids=positives, diversity_audit=diversity_audit
    )
    requirements = {
        "independent_slates": independent_slate_count
        >= int(minimums["minimum_independent_slates"]),
        "treatment_authoring_candidates": treatment_authoring_candidate_count
        >= int(minimums["minimum_treatment_authoring_candidates"]),
        "genuine_unique_completed": diversity["unique_positive_mechanism_instances"]
        >= int(minimums["minimum_genuine_completed_tasks"]),
        "matched_control_completed": matched_control_completed_count
        >= int(minimums["minimum_matched_control_completed_tasks"]),
        "mechanism_clusters": diversity["material_cluster_count"]
        >= int(minimums["minimum_mechanism_clusters"]),
        "cluster_concentration": diversity["largest_cluster_fraction"] is not None
        and diversity["largest_cluster_fraction"]
        <= float(minimums["maximum_largest_cluster_fraction"]),
        "quality_noninferiority": bool(noninferiority.get("all_provisional_model_gates_passed")),
        "human_review_complete": human_review_complete,
        "human_no_clear_control_preference": human_review_complete
        and human_control_materially_preferred is False,
    }
    if not human_review_complete:
        recommendation = "inconclusive"
    elif all(requirements.values()):
        recommendation = "candidate_for_user_acceptance"
    elif not requirements["quality_noninferiority"] or human_control_materially_preferred:
        recommendation = "reject"
    else:
        recommendation = "continue_testing"
    return {
        "property": property_name,
        "positive_status": POSITIVE_STATUS[property_name],
        "diversity": diversity,
        "requirements": requirements,
        "recommendation": recommendation,
        "ordinary_task_acceptance_affected": False,
        "automatic_promotion": False,
    }


def apply_specialized_source_ceilings(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_limit: int,
    episode_joint_maximum: int,
    abstraction_led_maximum: int,
    total_specialized_maximum: int,
) -> dict[str, Any]:
    """Apply source ceilings after qualitative ranking, never as a quota/bonus."""

    accepted: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    counts = Counter()
    for candidate in ranked_candidates:
        row = deepcopy(dict(candidate))
        source = row.get("experimental_source", "ordinary")
        specialized = source in {"episode_joint_inference", "abstraction_led"}
        exceeds = (
            specialized and counts["specialized"] >= total_specialized_maximum
        ) or (
            source == "episode_joint_inference" and counts[source] >= episode_joint_maximum
        ) or (
            source == "abstraction_led" and counts[source] >= abstraction_led_maximum
        )
        if exceeds or len(accepted) >= selected_limit:
            held.append({**row, "ceiling_reason": "specialized source ceiling" if exceeds else "slate size"})
            continue
        accepted.append(row)
        if specialized:
            counts["specialized"] += 1
            counts[source] += 1
    return {
        "selected": accepted,
        "held": held,
        "counts": dict(counts),
        "qualitative_rank_reordered": False,
        "ceiling_is_not_a_quota": True,
    }


__all__ = [
    "POSITIVE_STATUS",
    "V41AuditError",
    "apply_specialized_source_ceilings",
    "genuine_task_ids",
    "matched_noninferiority",
    "mechanism_diversity_summary",
    "property_status",
    "semantic_promotion_assessment",
]
