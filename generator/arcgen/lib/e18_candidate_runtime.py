"""Candidate-level E18 planning, realization, and evidence verification.

The module is deliberately additive.  It reuses the frozen V4 planner,
realizer, deterministic diagnostics, technical verifier, conditional repair,
and blind-solver implementations, while replacing their orchestration with
the invocation-aware E2 scheduler and the corrected E19 answer-evidence gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

from .factory_contracts import is_canonical_plan
from .e2_live_streaming import IdempotentSlateBudget, _realize_one, _technical_one
from .e2_live_streaming_v2 import _technical_primary
from .e2_streaming_scheduler_v2 import (
    PersistentCandidateSchedulerV2,
    validate_dependency_events_v2,
)
from .e3_live_overlap import SharedCallPool, _LimiterInvoker
from .e18_calls import E18AttemptError, E18AttemptInvoker
from .e18_evidence_gate import evidence_is_clear
from .e18_evidence_gate import resolve as resolve_evidence
from .e18_evidence_gate import run_primary as run_evidence_primary
from .e18_supply import ReservoirLedger
from .util import atomic_write_json, content_hash
from .v4_pipeline import V4ControlPipeline
from .v41_experiments import V41Runtime
from .v41_integration import _committed_distribution, _palette_target
from .v41_palette import schedule_backgrounds
from .v41_supported_departure import _record_stage
from .v42_logical_leap import validate_v42_plan_response
from .v44_abstraction_led import _normalize_palette_claims
from .v44_scale_fresh_integrated import load_fresh_scale_integrated_config
from .v48_evidence_scale_paired import _install_evidence_extension

SOL = "gpt-5.6-sol"
SPECIAL_ROUTES = frozenset({"logical_leap", "episode_joint_inference"})
PLANNER_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "ordinary_current": ("v44_quality_plan",),
    "logical_leap": ("v44_quality_plan", "logical_leap_planner_addendum"),
    "episode_joint_inference": ("v44_quality_plan", "v43_episode_joint_plan"),
    "evidence_preserving_scale": ("v44_quality_plan", "evidence_scale_addendum"),
}
TERMINAL_OPERATIONAL = frozenset(
    {
        "planning_operational_failure",
        "realization_operational_failure",
        "technical_operational_failure",
        "blind_operational_failure",
        "escalation_failure",
        "evidence_hold",
    }
)
TERMINAL_SEMANTIC = frozenset(
    {
        "plan_aborted",
        "realization_failure",
        "rejected_technical",
        "ambiguous",
    }
)


class E18CandidateError(RuntimeError):
    """Raised when downstream semantics or persistence changes."""


def _task_id(config: Mapping[str, Any], source_id: str) -> str:
    return (
        "e18"
        + content_hash(
            {"run_id": config["run"]["id"], "source_candidate_id": source_id}
        )[:12]
    )


def assign_routes(
    config: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Assign source-native routes and at most one optional scale invitation."""

    rows = [deepcopy(dict(item)) for item in selected]
    if not rows:
        raise E18CandidateError("cannot assign routes to an empty selected slate")
    for row in rows:
        source = str(row["private_source"])
        row["private_route"] = (
            source if source in SPECIAL_ROUTES else "ordinary_current"
        )

    deliberate = sum(row["private_route"] in SPECIAL_ROUTES for row in rows)
    minimum_free = ceil(
        len(rows) * float(config["design"]["minimum_invitation_free_fraction"])
    )
    maximum_deliberate = len(rows) - minimum_free
    if deliberate > maximum_deliberate:
        raise E18CandidateError("selection crossed the 70% invitation-free floor")

    ordinary = sorted(
        (row for row in rows if row["private_route"] == "ordinary_current"),
        key=lambda item: (
            int(item["source_development_priority"]),
            str(item["source_candidate_id"]),
        ),
    )
    # Functional scale is a conservative minority invitation and may be
    # declined by the planner.  It is never stacked with a semantic source.
    if ordinary and deliberate + 1 <= maximum_deliberate:
        ordinary[0]["private_route"] = "evidence_preserving_scale"

    invitations = sum(row["private_route"] != "ordinary_current" for row in rows)
    if len(rows) - invitations < minimum_free:
        raise E18CandidateError("route assignment violated invitation-free floor")
    if sum(row["private_route"] == "evidence_preserving_scale" for row in rows) > 1:
        raise E18CandidateError("more than one scale invitation in a slate")
    return rows


def _repair_parser(runtime: V41Runtime) -> None:
    """Normalize only demonstrably equivalent repair deviation ledgers.

    The repair is defined relative to the newly returned repaired plan.  When
    both ledgers describe the same authorized plan revision, leaving those
    descriptions in the realizer's *deviation from repaired plan* fields is a
    serialization mistake.  We normalize that narrow case to empty ledgers;
    a one-sided ledger, mismatched semantic flag, or malformed repair remains
    a parser failure and receives only the ordinary bounded retry.
    """

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        value = original(stage, data)
        if stage != "v4_targeted_repair" or value.get("status") != "repaired":
            return value
        realization = value.get("repaired_realization")
        if not isinstance(realization, dict):
            return value
        metadata = realization.get("authoring_metadata")
        fidelity = (
            metadata.get("plan_fidelity") if isinstance(metadata, Mapping) else None
        )
        if not isinstance(fidelity, dict):
            return value
        top = realization.get("deviations")
        nested = fidelity.get("deviations")
        if top == nested:
            return value
        if (
            not isinstance(top, list)
            or not isinstance(nested, list)
            or not top
            or not nested
        ):
            raise E18CandidateError(
                "repair returned one-sided or empty/nonempty deviation ledgers"
            )
        if len(top) != len(nested):
            raise E18CandidateError("repair deviation ledger cardinalities differ")
        top_flags = [
            item.get("semantic_or_evidential")
            for item in top
            if isinstance(item, Mapping)
        ]
        nested_flags = [
            item.get("semantic_or_evidential")
            for item in nested
            if isinstance(item, Mapping)
        ]
        if len(top_flags) != len(top) or sorted(top_flags) != sorted(nested_flags):
            raise E18CandidateError(
                "repair deviation ledgers are not semantically aligned"
            )
        if any(
            not str(item.get("plan_field") or "").strip()
            or not str(item.get("deviation") or "").strip()
            or not str(item.get("reason") or "").strip()
            for item in [*top, *nested]
        ):
            raise E18CandidateError("repair deviation ledger is incomplete")
        realization["deviations"] = []
        fidelity["deviations"] = []
        value["e18_repair_ledger_normalization"] = {
            "applied": True,
            "reason": (
                "both ledgers described the authorized revision from the old plan, "
                "not a deviation from the returned repaired plan"
            ),
            "original_top_level": top,
            "original_nested": nested,
        }
        return value

    runtime.invoker.output_parser = parse


def create_runtime(
    config: Mapping[str, Any], pool: SharedCallPool, source_manifest: Mapping[str, Any]
) -> V41Runtime:
    from .v4_integrated_validation import _install_extensions

    resolved = load_fresh_scale_integrated_config(
        config["_references"]["downstream_adapter"]
    )
    resolved = deepcopy(resolved)
    resolved["seed"] = int(config["seed"]) + 18_031
    resolved["paths"] = deepcopy(resolved["paths"])
    resolved["paths"]["runs"] = str(Path(config["_output_directory"]))
    resolved["paths"]["committed_task_registry"] = str(
        Path(config["_paths"]["active_registry"]).resolve()
    )
    resolved["paths"]["cache"] = str(Path(config["_paths"]["cache"]).resolve())
    resolved["run"] = {
        **deepcopy(resolved["run"]),
        "id": "downstream",
        "mock": bool(config["run"]["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    resolved["model"] = deepcopy(resolved["model"])
    resolved["model"]["concurrency"] = int(config["scheduling"]["global_workers"])
    resolved["selection"] = deepcopy(resolved["selection"])
    resolved["selection"]["max_plans"] = int(
        config["design"]["maximum_plans_per_slate"]
    )
    resolved["selection"]["max_realizations"] = int(
        config["design"]["maximum_realizations_per_slate"]
    )
    resolved["repair_policy"] = {
        "allow_mechanical_test_input_restoration": True,
        "require_consistent_deviation_ledgers": True,
    }
    runtime = V41Runtime(resolved, source_manifest)
    _install_extensions(runtime)
    _install_evidence_extension(runtime)
    _repair_parser(runtime)
    runtime.invoker = E18AttemptInvoker(
        _LimiterInvoker(runtime.invoker, pool, "downstream"),
        runtime.run_dir / "attempt_journal.json",
        maximum_attempts=int(config["scheduling"]["maximum_attempts_per_logical_call"]),
    )
    roles = resolved["model"]["roles"]
    expected = {
        "concrete_planner": (SOL, "max"),
        "task_realizer": (SOL, "max"),
        "technical_verifier": (SOL, "high"),
        "blind_solver": (SOL, "xhigh"),
        "ambiguity_adjudicator": (SOL, "high"),
        "targeted_repair": (SOL, "max"),
        "editorial_board": (SOL, "max"),
    }
    for role, wanted in expected.items():
        actual = (roles[role]["model"], roles[role]["reasoning_effort"])
        if actual != wanted:
            raise E18CandidateError(f"downstream route changed for {role}: {actual}")
    return runtime


def add_records(
    config: Mapping[str, Any],
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
) -> list[str]:
    routed = assign_routes(config, selected)
    task_ids: list[str] = []
    with runtime.lock:
        records = runtime.state.setdefault("semantic_records", {})
        for row in routed:
            source = str(row["source_candidate_id"])
            task_id = _task_id(config, source)
            task_ids.append(task_id)
            record = {
                "task_id": task_id,
                "source_candidate_id": source,
                "lineage_id": row["lineage_id"],
                "arm": "single",
                "private_source": row["private_source"],
                "private_conditioned": bool(row["private_conditioned"]),
                "private_source_call_id": row["private_source_call_id"],
                "private_idea_effort": row["private_effort"],
                "private_route": row["private_route"],
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(row["source_development_priority"]),
                "search_mode": row["search_mode"],
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "novelty": deepcopy(row.get("novelty") or {}),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
                "candidate_revision": 0,
            }
            prior = records.get(task_id)
            if prior is None:
                records[task_id] = record
            else:
                for field, expected_value in record.items():
                    if field in {
                        "status",
                        "stage_history",
                        "repair_count",
                        "candidate_revision",
                    }:
                        continue
                    if prior.get(field) != expected_value:
                        raise E18CandidateError(
                            f"resumed candidate changed: {task_id}:{field}"
                        )
        runtime.state["e18_candidate_graph"] = {
            "candidate_level_streaming": True,
            "global_workers": int(config["scheduling"]["global_workers"]),
            "corrected_e19_trigger": True,
            "deferred_boards": True,
        }
        runtime.save()
    if len(task_ids) != len(set(task_ids)):
        raise E18CandidateError("E18 task ID collision")
    return sorted(task_ids)


def plan_one(runtime: V41Runtime, task_id: str) -> None:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("status") != "selected":
        return
    route = str(record["private_route"])
    brief = deepcopy(record["editorial_brief"])
    if route not in {"logical_leap"}:
        brief["optional_development_suggestion"] = None
    values = {
        "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
        "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
        "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
        "CANDIDATE_ID": task_id,
    }
    try:
        if route == "ordinary_current":
            plan = runtime.call(
                prompt_name="concrete_planner",
                composed_with=PLANNER_COMPOSITIONS[route],
                call_id=f"{task_id}-concrete-plan",
                values=values,
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="anonymous-authoring",
                expected={"candidate_id": task_id},
            )
            result = {
                "candidate_id": task_id,
                "decision": "abort"
                if plan.get("status") == "abort"
                else "ordinary_current",
                "decision_reason": (
                    str(plan.get("abort_reason") or "ordinary planner aborted")
                    if plan.get("status") == "abort"
                    else "ordinary V4.4 quality planning"
                ),
                "concrete_plan": plan,
            }
        elif route == "logical_leap":
            def validate_logical(value: dict[str, Any]) -> None:
                validate_v42_plan_response(
                    task_id=task_id, route="logical_leap", value=value
                )

            result = runtime.call(
                prompt_name="concrete_planner",
                composed_with=PLANNER_COMPOSITIONS[route],
                call_id=f"{task_id}-concrete-plan",
                values=values,
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="anonymous-authoring",
                expected={"candidate_id": task_id},
                validator=validate_logical,
            )
            result["structural_transfer_contract"] = None
            validate_v42_plan_response(
                task_id=task_id, route="logical_leap", value=result
            )
        elif route == "episode_joint_inference" or route == "evidence_preserving_scale":
            result = runtime.call(
                prompt_name="concrete_planner",
                composed_with=PLANNER_COMPOSITIONS[route],
                call_id=f"{task_id}-concrete-plan",
                values=values,
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="anonymous-authoring",
                expected={"candidate_id": task_id},
            )
        else:
            raise E18CandidateError(f"unknown E18 planning route {route}")
        plan = result.get("concrete_plan")
        if plan is not None:
            if plan.get("candidate_id") != task_id:
                raise E18CandidateError("planner changed nested candidate ID")
            if not (getattr(runtime, "contract_bindings", None) and is_canonical_plan(plan)):
                plan = _normalize_palette_claims(plan)
            result["concrete_plan"] = plan
        if plan is not None and plan.get("status") == "abort":
            decision = (
                result.get("decision")
                or result.get("property_status")
                or result.get("scale_decision")
            )
            if decision != "abort":
                raise E18CandidateError("aborted plan did not use abort decision")
    except E18AttemptError as exc:
        _record_stage(
            runtime,
            task_id,
            "planning",
            status="planning_operational_failure",
            planning_error=str(exc),
        )
        return
    status = (
        "plan_aborted"
        if plan is None or plan.get("status") == "abort"
        else "plan_ready"
    )
    _record_stage(
        runtime,
        task_id,
        "planning",
        status=status,
        planning_result=result,
        concrete_plan=plan,
    )
    destination = runtime.run_dir / "plans" / "single"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / f"{task_id}.json", result)


def palette_assignments(
    config: Mapping[str, Any],
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
    *,
    slate: int,
) -> dict[str, int | None]:
    path = runtime.run_dir / "palette_assignments.dynamic.json"
    target, reference = _palette_target(runtime.config)
    committed = _committed_distribution(runtime.config)
    value = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {
            "schema_version": 1,
            "assignments": {},
            "by_source_candidate": {},
            "slates": {},
            "target_distribution": {str(key): item for key, item in target.items()},
            "real_reference": reference,
            "committed_distribution": committed,
        }
    )
    source_ids = sorted(str(item["source_candidate_id"]) for item in selected)
    existing = value["by_source_candidate"]
    missing = [source for source in source_ids if source not in existing]
    if missing:
        counts = {
            int(key): int(item)
            for key, item in committed["counts"].items()
            if str(key).isdigit() and 0 <= int(key) <= 9
        }
        for color in existing.values():
            if color is not None:
                counts[int(color)] = counts.get(int(color), 0) + 1
        existing.update(
            schedule_backgrounds(
                missing,
                seed=int(config["seed"]) + slate * 180_271,
                current_counts=counts,
                target_distribution=target,
                eligible={source: True for source in missing},
            )
        )
    assignments = {_task_id(config, source): existing[source] for source in source_ids}
    for task_id, color in assignments.items():
        if task_id in value["assignments"] and value["assignments"][task_id] != color:
            raise E18CandidateError("palette assignment changed on resume")
        value["assignments"][task_id] = color
    value["slates"][str(slate)] = {
        "source_candidate_ids": source_ids,
        "task_ids": sorted(assignments),
    }
    atomic_write_json(path, value)
    return {str(key): item for key, item in value["assignments"].items()}


def _slate_records(runtime: V41Runtime, task_id: str) -> list[dict[str, Any]]:
    slate = int(runtime.state["semantic_records"][task_id]["slate_index"])
    return [
        row
        for row in runtime.state["semantic_records"].values()
        if int(row["slate_index"]) == slate
    ]


def _cannot_reach_technical(record: Mapping[str, Any]) -> bool:
    return str(record.get("status")) in {
        "plan_aborted",
        "planning_operational_failure",
        "realization_failure",
        "realization_operational_failure",
    }


def _technical_primary_known(record: Mapping[str, Any]) -> bool:
    return (
        _cannot_reach_technical(record)
        or bool(record.get("technical_results"))
        or str(record.get("status"))
        in {"technical_operational_failure", "rejected_technical", "escalation_failure"}
    )


def _technical_pending(record: Mapping[str, Any]) -> bool:
    return str(record.get("status")) == "technical_needs_resolution" or (
        str(record.get("status")) == "realized"
        and bool(record.get("technical_results"))
        and int(record.get("repair_count") or 0) > 0
    )


def _blind_primary_known(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status"))
    evidence = record.get("answer_evidence")
    if isinstance(evidence, Mapping) and evidence.get("status") in {
        "evidence_hold",
        "ambiguous",
    }:
        return True
    if _cannot_reach_technical(record) or status in {
        "technical_operational_failure",
        "rejected_technical",
        "escalation_failure",
    }:
        return True
    return bool(record.get("blind_results")) or status == "blind_operational_failure"


def _blind_pending(record: Mapping[str, Any]) -> bool:
    return str(record.get("status")) in {
        "blind_needs_resolution",
        "blind_second_complete",
    } or (
        str(record.get("status")) == "realized"
        and bool(record.get("blind_results"))
        and int(record.get("repair_count") or 0) > 0
    )


def _priority(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["source_development_priority"]), str(record["task_id"])


def can_dispatch(runtime: V41Runtime, task_id: str, stage: str) -> bool:
    # Slate barriers held follow-up reviews until every sibling had its primary result and then
    # ran them one at a time; they protected per-slate budgets, which are now a shared pool.
    profile = (getattr(runtime, "factory_config", None) or {}).get("_experiment_profile") or {}
    if profile.get("slate_barriers", True) is False:
        return True
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        slate = _slate_records(runtime, task_id)
        if stage == "technical" and _technical_pending(record):
            if not all(_technical_primary_known(item) for item in slate):
                return False
            pending = sorted(
                (item for item in slate if _technical_pending(item)), key=_priority
            )
            return bool(pending) and pending[0]["task_id"] == task_id
        if stage == "blind" and _blind_pending(record):
            if any(_technical_pending(item) for item in slate):
                return False
            if not all(_blind_primary_known(item) for item in slate):
                return False
            pending = sorted(
                (item for item in slate if _blind_pending(item)), key=_priority
            )
            return bool(pending) and pending[0]["task_id"] == task_id
        return True


def next_stage(runtime: V41Runtime, task_id: str) -> str | None:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        status = str(record.get("status"))
        if status == "selected":
            return "planning"
        if status == "plan_ready":
            return "realization"
        if status == "realized":
            return "blind" if record.get("blind_results") else "technical"
        if status == "technical_needs_resolution":
            return "technical"
        if status == "technical_pass":
            return "blind"
        if status in {"blind_needs_resolution", "blind_second_complete"}:
            return "blind"
        return None


def reconcile_active(runtime: V41Runtime, task_id: str, stage: str) -> str:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        current = next_stage(runtime, task_id)
        if stage == "technical" and _technical_pending(record):
            return "substep"
        if stage == "blind" and _blind_pending(record):
            return "substep"
        if current == stage:
            return "requeue"
        if current is None:
            return "terminal"
        return "completed"


def run_slate(
    config: Mapping[str, Any],
    runtime: V41Runtime,
    *,
    slate: int,
    task_ids: Sequence[str],
    assignments: Mapping[str, int | None],
) -> dict[str, Any]:
    budget = IdempotentSlateBudget(
        runtime, slate, int(config["scheduling"]["escalation_calls_per_slate"])
    )

    def run_stage(task_id: str, stage: str) -> bool:
        if stage == "planning":
            plan_one(runtime, task_id)
            return True
        if stage == "realization":
            _realize_one(runtime, task_id, assignments)
            return True
        record = deepcopy(runtime.state["semantic_records"][task_id])
        if stage == "technical":
            public = (
                runtime.state["semantic_records"][task_id]
                .get("realization", {})
                .get("task")
            )
            if not isinstance(public, Mapping) or any(
                "output" in pair for pair in public.get("test", [])
            ):
                raise E18CandidateError(
                    "hidden output entered technical-verifier public task"
                )
            if not record.get("technical_results"):
                return _technical_primary(runtime, task_id)
            _technical_one(runtime, task_id, budget)
            return True
        if stage == "blind":
            if not record.get("blind_results"):
                return run_evidence_primary(runtime, task_id)
            resolve_evidence(runtime, task_id, budget)
            return True
        raise E18CandidateError(f"unknown E18 candidate stage {stage}")

    scheduler = PersistentCandidateSchedulerV2(
        state_path=runtime.run_dir
        / "candidate_dependency_states"
        / f"slate-{slate:02d}.json",
        candidate_ids=task_ids,
        workers=int(config["scheduling"]["global_workers"]),
        next_stage=lambda task_id: next_stage(runtime, task_id),
        run_stage=run_stage,
        can_dispatch=lambda task_id, stage: can_dispatch(runtime, task_id, stage),
        reconcile_active=lambda task_id, stage: reconcile_active(
            runtime, task_id, stage
        ),
        max_downstream_before_new_plan=int(
            config["scheduling"]["max_downstream_before_new_plan"]
        ),
    )
    state = scheduler.run()
    validation = validate_dependency_events_v2(state)
    if not validation["valid"]:
        raise E18CandidateError(
            f"slate {slate} dependency graph failed: {validation['errors']}"
        )
    return state


def finalize_reservoir_for_slate(
    runtime: V41Runtime,
    ledger: ReservoirLedger,
    task_ids: Sequence[str],
) -> None:
    for task_id in task_ids:
        record = runtime.state["semantic_records"][task_id]
        source = str(record["source_candidate_id"])
        status = str(record.get("status"))
        if record.get("complete_task") is not None and evidence_is_clear(record):
            outcome = "completed_uncommitted"
            reason = "candidate completed exact verification with clear answer evidence"
        elif status in TERMINAL_OPERATIONAL:
            outcome = "recoverable"
            reason = f"operational or unresolved evidence outcome: {status}"
        else:
            outcome = "rejected"
            reason = f"terminal semantic disposition: {status}"
        ledger.finalize(source, outcome=outcome, reason=reason)
    with ledger.lock:
        slate = (
            int(runtime.state["semantic_records"][str(task_ids[0])]["slate_index"])
            if task_ids
            else None
        )
        if slate is not None and slate not in ledger.value["finalized_slates"]:
            ledger.value["finalized_slates"].append(slate)
            ledger.value["events"].append(
                {
                    "event": "slate_finalized",
                    "slate_index": slate,
                    "at_epoch_seconds": round(__import__("time").time(), 3),
                }
            )
            ledger.save()
            ledger.validate()


__all__ = [
    "PLANNER_COMPOSITIONS",
    "E18CandidateError",
    "add_records",
    "assign_routes",
    "can_dispatch",
    "create_runtime",
    "finalize_reservoir_for_slate",
    "next_stage",
    "palette_assignments",
    "plan_one",
    "reconcile_active",
    "run_slate",
]
