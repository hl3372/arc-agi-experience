"""Repaired live runner for the isolated E2 candidate-streaming experiment.

The frozen V4.8 authoring and review stack is reused verbatim.  This module
changes only orchestration, persistence, reservation validation, and timing:

* development slates execute independently;
* candidates stream through their own dependency chains;
* conditional work waits for deterministic per-slate allocation;
* all selected mechanisms hold run-local recoverable leases; and
* every physical model attempt is retained and reported.

The runner is experimental.  It cannot commit tasks, mutate the production
reservoir/bank, or promote E2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ConfigError
from .e2_attempt_ledger_v2 import build_attempt_ledger
from .e2_live_streaming import (
    EXPECTED_WINNER_ID,
    IdempotentSlateBudget,
    _aliases,
    _assert_downstream_prompt_blinding,
    _initialize_records,
    _load_or_freeze_post_blind,
    _load_selected,
    _materialize,
    _next_stage,
    _palette_assignments,
    _plan_one,
    _read_yaml_or_json,
    _realize_one,
    _resolve,
    _safe_summary,
    _sha,
    _snapshot,
    _validated_ref,
    _write_immutable,
)
from .e2_reservations_v2 import (
    RunLocalReservationLeases,
    validate_recoverable_handoff,
)
from .e2_stage_replay_v2 import (
    build_stage_ledger,
    reconcile_attempts_with_stages,
    replay_pooled_and_independent,
)
from .e2_streaming_scheduler_v2 import (
    PersistentCandidateSchedulerV2,
    validate_dependency_events_v2,
)
from .model import CallResult
from .util import read_json
from .v41_experiments import V41Runtime
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_supported_departure import (
    _blind_call,
    _record_stage,
    _technical_call,
    _technical_evaluation,
    _wrap_blind,
)
from .v41_test_challenge import _board as _editorial_boards
from .v44_abstraction_led import _install_extensions as _install_v44_extensions
from .v44_scale_config import load_v44_scale_config
from .v48_evidence_scale_paired import _install_evidence_extension

PIPELINE = "arcgen-e2-live-candidate-streaming-v2"
VERSION = "2.0"


class E2LiveV2Error(RuntimeError):
    """Raised when E2 v2 changes semantics or loses timing provenance."""


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != VERSION:
        raise ConfigError(f"configuration must use {PIPELINE} {VERSION}")
    root = source.parent.parent if source.parent.name == "config" else source.parent
    refs = {
        name: raw.get(name)
        for name in (
            "base_scale_config",
            "experimental_winner_manifest",
            "selected_cohort_manifest",
        )
    }
    if not all(isinstance(value, Mapping) for value in refs.values()):
        raise ConfigError("base, winner, and selected-cohort references are required")
    base = _validated_ref(
        refs["base_scale_config"], root=root, name="base_scale_config"
    )
    winner_path = _validated_ref(
        refs["experimental_winner_manifest"],
        root=root,
        name="experimental_winner_manifest",
    )
    selected_path = _validated_ref(
        refs["selected_cohort_manifest"],
        root=root,
        name="selected_cohort_manifest",
    )
    winner = read_json(winner_path)
    if (
        winner.get("baseline_id") != EXPECTED_WINNER_ID
        or winner.get("status") != "frozen_experimental_candidate_continue_testing"
        or winner.get("production_policy", {}).get("production_promotion_authorized")
        is not False
    ):
        raise ConfigError("E2 v2 must use the frozen unpromoted V4.8 winner")
    settings = raw.get("scheduler")
    run = raw.get("run")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        raise ConfigError("scheduler.enabled must be true")
    for field in (
        "candidate_level_streaming",
        "independent_slates",
        "unchanged_editorial_boards",
        "whole_slate_reserved_before_execution",
        "attempt_complete_accounting",
        "deterministic_escalation_allocation",
        "reconstruct_barrier_counterfactual",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"scheduler.{field} must be true")
    if settings.get("global_workers") != 8:
        raise ConfigError("the isolated E2 qualification retains exactly 8 workers")
    if settings.get("escalation_calls_per_slate") != 5:
        raise ConfigError("E2 retains the exact five-call escalation ceiling")
    if (
        type(settings.get("max_downstream_before_new_plan")) is not int
        or int(settings["max_downstream_before_new_plan"]) < 1
    ):
        raise ConfigError("scheduler starvation bound must be positive")
    minimum = settings.get("minimum_selected")
    maximum = settings.get("maximum_selected")
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or minimum < 1
        or maximum < minimum
    ):
        raise ConfigError("selected bounds are invalid")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str):
        raise ConfigError("run.id is required")
    if run.get("mock") not in {True, False}:
        raise ConfigError("run.mock must be boolean")
    for field in (
        "automatic_task_commit",
        "production_state_mutation_allowed",
        "production_promotion_allowed",
    ):
        if run.get(field) is not False:
            raise ConfigError(f"run.{field} must be false")
    if type(raw.get("seed")) is not int:
        raise ConfigError("seed must be an integer")
    output_value = raw.get("output_directory")
    if not isinstance(output_value, str) or not output_value:
        raise ConfigError("output_directory is required")

    config = deepcopy(load_v44_scale_config(base))
    repository_root = Path(config["_repository_root"]).resolve()
    if repository_root != root.resolve():
        raise ConfigError("base configuration resolved another repository root")
    config["seed"] = int(raw["seed"])
    config["model"] = deepcopy(config["model"])
    config["model"]["concurrency"] = 8
    config["functional_scale"] = deepcopy(config["functional_scale"])
    config["functional_scale"]["escalation_calls_per_slate"] = 5
    config["run"] = {
        **deepcopy(config["run"]),
        "id": str(run["id"]),
        "mock": bool(run["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    output = _resolve(root, output_value)
    if output.name != str(run["id"]):
        raise ConfigError("output directory basename must equal run.id")
    config["paths"] = deepcopy(config["paths"])
    config["paths"]["runs"] = str(output.parent)
    config["scheduler"] = deepcopy(dict(settings))
    config["_e2_v2_config_path"] = str(source)
    config["_e2_output_directory"] = str(output)
    config["_e2_base_config_path"] = str(base)
    config["_e2_winner_manifest_path"] = str(winner_path)
    config["_e2_selected_cohort_path"] = str(selected_path)
    return config


def _implementation_manifest(
    config: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    names = (
        "arcgen.lib/e2_live_streaming_v2.py",
        "arcgen.lib/e2_streaming_scheduler_v2.py",
        "arcgen.lib/e2_attempt_ledger_v2.py",
        "arcgen.lib/e2_stage_replay_v2.py",
        "arcgen.lib/e2_reservations_v2.py",
        "arcgen.lib/e2_live_streaming.py",
        "arcgen.lib/v41_experiments.py",
        "arcgen.lib/v41_supported_departure.py",
        "arcgen.lib/v41_test_challenge.py",
        "arcgen.lib/v48_evidence_scale_paired.py",
    )
    value = {
        "schema_version": 2,
        "pipeline": PIPELINE,
        "version": VERSION,
        "scheduling_only_change": True,
        "files": {name: _sha(root / name) for name in names},
        "config": {
            "path": config["_e2_v2_config_path"],
            "sha256": _sha(config["_e2_v2_config_path"]),
        },
    }
    _write_immutable(
        run_dir / "e2_v2_implementation_manifest.json",
        value,
        label="E2 v2 implementation manifest",
    )
    return value


class _AttemptPreservingInvoker:
    """Archive exhausted attempts before an explicit semantic retry.

    ``ModelInvoker`` numbers attempts from one on each invocation.  If a prior
    invocation exhausted all retries, an ordinary resume would overwrite those
    files.  Successful or recoverable partial calls are deliberately left
    untouched and remain content-addressed/idempotent.
    """

    def __init__(self, inner: Any, run_dir: Path) -> None:
        self.inner = inner
        self.run_dir = run_dir
        self.lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def invoke(self, *, stage: str, call_id: str, **kwargs: Any) -> CallResult:
        log_dir = self.run_dir / "model_calls" / stage / call_id
        prior_count = len(
            list((log_dir / "attempt_history").rglob("attempt-*.timing.json"))
        )
        with self.lock:
            result = log_dir / "result.json"
            recoverable_partial = (log_dir / "parsed_response.json").is_file() and (
                log_dir / "raw_response.txt"
            ).is_file()
            attempts = sorted(log_dir.glob("attempt-*.timing.json"))
            if attempts and not result.is_file() and not recoverable_partial:
                history = log_dir / "attempt_history"
                index = 1
                while (history / f"session-{index:03d}").exists():
                    index += 1
                destination = history / f"session-{index:03d}"
                destination.mkdir(parents=True)
                for pattern in (
                    "attempt-*.timing.json",
                    "attempt-*.failure.log",
                    "attempt-*.raw_response.txt",
                    "parse_failures.json",
                ):
                    for path in log_dir.glob(pattern):
                        shutil.move(str(path), destination / path.name)
                prior_count += len(attempts)
        result = self.inner.invoke(stage=stage, call_id=call_id, **kwargs)
        return replace(result, attempts=prior_count + int(result.attempts))


def _technical_primary(runtime: V41Runtime, task_id: str) -> bool:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("technical_results"):
        return record.get("status") != "technical_needs_resolution"
    try:
        response = _technical_call(runtime, task_id, "technical-primary")
    except Exception as exc:  # noqa: BLE001 - semantic operational outcome
        _record_stage(
            runtime,
            task_id,
            "technical",
            status="technical_operational_failure",
            technical_error=str(exc),
        )
        return True
    record = deepcopy(runtime.state["semantic_records"][task_id])
    evaluation = _technical_evaluation(record, response)
    status = (
        "technical_pass"
        if evaluation["passed"] and response["confidence"] != "low"
        else "technical_needs_resolution"
    )
    _record_stage(
        runtime,
        task_id,
        "technical",
        status=status,
        technical_results=[evaluation],
    )
    return status != "technical_needs_resolution"


def _blind_primary(runtime: V41Runtime, task_id: str) -> bool:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("blind_results"):
        return record.get("status") not in {
            "blind_needs_resolution",
            "blind_second_complete",
        }
    try:
        response = _blind_call(runtime, task_id, "blind-primary")
    except Exception as exc:  # noqa: BLE001 - semantic operational outcome
        _record_stage(
            runtime,
            task_id,
            "blind",
            status="blind_operational_failure",
            blind_error=str(exc),
        )
        return True
    record = deepcopy(runtime.state["semantic_records"][task_id])
    wrapped = _wrap_blind(record, response)
    status = (
        "blind_pass"
        if not wrapped["serious_different_output_alternative"]
        else "blind_needs_resolution"
    )
    _record_stage(
        runtime,
        task_id,
        "blind",
        status=status,
        blind_results=[wrapped],
    )
    return status != "blind_needs_resolution"


def _slate_records(runtime: V41Runtime, task_id: str) -> list[dict[str, Any]]:
    slate = int(runtime.state["semantic_records"][task_id]["slate_index"])
    return [
        record
        for record in runtime.state["semantic_records"].values()
        if int(record["slate_index"]) == slate
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
        in {"technical_operational_failure", "rejected_technical"}
    )


def _technical_resolution_pending(record: Mapping[str, Any]) -> bool:
    return str(record.get("status")) == "technical_needs_resolution" or (
        str(record.get("status")) == "realized"
        and bool(record.get("technical_results"))
        and int(record.get("repair_count") or 0) > 0
    )


def _blind_primary_known(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status"))
    if _cannot_reach_technical(record) or status in {
        "technical_operational_failure",
        "rejected_technical",
    }:
        return True
    return bool(record.get("blind_results")) or status == "blind_operational_failure"


def _blind_resolution_pending(record: Mapping[str, Any]) -> bool:
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


def _can_dispatch(runtime: V41Runtime, task_id: str, stage: str) -> bool:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        slate = _slate_records(runtime, task_id)
        if stage == "technical" and _technical_resolution_pending(record):
            if not all(_technical_primary_known(row) for row in slate):
                return False
            unresolved = sorted(
                (row for row in slate if _technical_resolution_pending(row)),
                key=_priority,
            )
            return bool(unresolved) and unresolved[0]["task_id"] == task_id
        if stage == "blind" and _blind_resolution_pending(record):
            if any(_technical_resolution_pending(row) for row in slate):
                return False
            if not all(_blind_primary_known(row) for row in slate):
                return False
            unresolved = sorted(
                (row for row in slate if _blind_resolution_pending(row)),
                key=_priority,
            )
            return bool(unresolved) and unresolved[0]["task_id"] == task_id
        return True


def _reconcile_active(runtime: V41Runtime, task_id: str, stage: str) -> str:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        current = _next_stage(runtime, task_id)
        if stage == "technical" and _technical_resolution_pending(record):
            return "substep"
        if stage == "blind" and _blind_resolution_pending(record):
            return "substep"
        if current == stage:
            return "requeue"
        if current is None:
            return "terminal"
        return "completed"


def _filter_attempts_for_slate(
    attempt_ledger: Mapping[str, Any], slate_index: int
) -> dict[str, Any]:
    value = deepcopy(dict(attempt_ledger))
    value["logical_calls"] = [
        row
        for row in attempt_ledger.get("logical_calls", [])
        if int(row.get("slate_index") or 0) == slate_index
    ]
    return value


def _selection_production_binding(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    current = _protected_hashes(config)
    source = selection.get("protected_production_inputs") or {}
    fields = ("committed_task_registry", "reservoir")
    comparisons = {}
    for field in fields:
        selected_row = source.get(field) or {}
        current_row = current.get(field) or {}
        comparisons[field] = {
            "selected_sha256": selected_row.get("sha256"),
            "current_sha256": current_row.get("sha256"),
            "matched": bool(selected_row.get("sha256"))
            and selected_row.get("sha256") == current_row.get("sha256"),
        }
    return {
        "passed": all(row["matched"] for row in comparisons.values()),
        "comparisons": comparisons,
        "accepted_bank_snapshot_id": selection.get("accepted_bank_snapshot_id"),
    }


def _aggregate_scheduler_report(
    runtime: V41Runtime, scheduler_states: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    attempts = build_attempt_ledger(runtime.state, run_dir=runtime.run_dir)
    slates = []
    for state in scheduler_states:
        stage_ledger = build_stage_ledger(state, runtime.state)
        candidate_ids = list(state.get("candidate_ids") or [])
        if not candidate_ids:
            raise E2LiveV2Error("scheduler state has no candidates")
        slate_indices = {
            int(runtime.state["semantic_records"][task_id]["slate_index"])
            for task_id in candidate_ids
        }
        if len(slate_indices) != 1:
            raise E2LiveV2Error("one E2 v2 scheduler crossed slate boundaries")
        slate_index = next(iter(slate_indices))
        reconciliation = reconcile_attempts_with_stages(
            stage_ledger, _filter_attempts_for_slate(attempts, slate_index)
        )
        replay = (
            replay_pooled_and_independent(stage_ledger)
            if stage_ledger["coverage_complete"] and reconciliation["passed"]
            else None
        )
        slates.append(
            {
                "slate_index": slate_index,
                "candidate_count": len(state["candidate_ids"]),
                "dependency_validation": validate_dependency_events_v2(state),
                "stage_ledger": stage_ledger,
                "attempt_stage_reconciliation": reconciliation,
                "replay": replay,
            }
        )
    barrier = sum(
        float(
            row["replay"]["pooled_continuous_factory"]["barrier_counterfactual"][
                "makespan_seconds"
            ]
        )
        for row in slates
        if row["replay"] is not None
    )
    observed = sum(
        float(row["stage_ledger"]["observed_scheduler_wall_seconds"] or 0)
        for row in slates
    )
    integrity = {
        "attempt_coverage_complete": attempts["summary"]["coverage_complete"] is True,
        "every_slate_dependency_valid": all(
            row["dependency_validation"]["valid"] for row in slates
        ),
        "every_slate_stage_complete": all(
            row["stage_ledger"]["coverage_complete"] for row in slates
        ),
        "every_attempt_bound_to_stage": all(
            row["attempt_stage_reconciliation"]["passed"] for row in slates
        ),
        "every_slate_primary_timing_qualified": all(
            row["stage_ledger"]["primary_timing_qualified"] for row in slates
        ),
    }
    return {
        "schema_version": 2,
        "independent_slates_executed": True,
        "slates": slates,
        "attempt_accounting": attempts,
        "aggregate_observed_streaming_seconds": round(observed, 6),
        "aggregate_barrier_counterfactual_seconds": round(barrier, 6),
        "aggregate_observed_reduction_fraction": (
            round(1 - observed / barrier, 6) if barrier else None
        ),
        "integrity_checks": integrity,
        "live_timing_qualified": all(integrity.values()),
        "continuous_factory_pooling_used": False,
    }


def validate(config: Mapping[str, Any]) -> dict[str, Any]:
    source = read_json(Path(config["_e2_selected_cohort_path"]))
    handoff = validate_recoverable_handoff(source)
    binding = _selection_production_binding(config, source)
    return {
        "valid": source.get("completed") is True
        and handoff["validated"]
        and binding["passed"],
        "structurally_valid": source.get("completed") is True and handoff["validated"],
        "live_execution_eligible": binding["passed"],
        "selection_production_binding": binding,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "selected_slate_count": len(source.get("slates", [])),
        "global_workers": 8,
        "independent_slates": True,
        "editorial_boards_retained": True,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }


def run(config: Mapping[str, Any], *, authorize_model_calls: bool) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E2LiveV2Error("live E2 v2 requires --authorize-model-calls")
    run_dir = Path(config["_e2_output_directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    protected = _protected_hashes(config)
    source_selection = read_json(Path(config["_e2_selected_cohort_path"]))
    handoff = validate_recoverable_handoff(source_selection)
    binding = _selection_production_binding(config, source_selection)
    if not binding["passed"]:
        raise E2LiveV2Error(
            "selected cohort was frozen against a different bank or reservoir; "
            "generate a fresh current-state selection cohort"
        )
    snapshot = {
        "schema_version": 2,
        "pipeline": PIPELINE,
        "version": VERSION,
        "config": _snapshot(config),
        "config_path": config["_e2_v2_config_path"],
        "config_sha256": _sha(config["_e2_v2_config_path"]),
        "protected_production_state": protected,
        "reservation_handoff": handoff,
        "selection_production_binding": binding,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }
    _write_immutable(
        run_dir / "e2_v2_config.snapshot.json",
        snapshot,
        label="E2 v2 config snapshot",
    )
    _implementation_manifest(config, run_dir)
    selected = _load_selected(config, run_dir)
    selected_ids = sorted(str(row["source_candidate_id"]) for row in selected)
    if selected_ids != sorted(handoff["selected_proposal_ids"]):
        raise E2LiveV2Error(
            "selected bounds truncated or changed the atomically reserved cohort"
        )
    aliases = _aliases(selected, int(config["seed"]))
    selected_with_ids = [
        {**deepcopy(row), "task_id": aliases[str(row["source_candidate_id"])]}
        for row in selected
    ]
    leases = RunLocalReservationLeases(
        path=run_dir / "run_local_reservation_leases.json",
        owner_run_id=str(config["run"]["id"]),
        selected=selected_with_ids,
        handoff=handoff,
        lease_seconds=int(
            config["scheduler"].get("reservation_lease_seconds") or 259200
        ),
    )
    leases.assert_all_active(selected_ids)
    runtime = V41Runtime(
        deepcopy(dict(config)),
        {
            "schema_version": 2,
            "pipeline": PIPELINE,
            "selected_cohort_sha256": _sha(config["_e2_selected_cohort_path"]),
            "reservation_handoff_hash": handoff["handoff_hash"],
            "scheduling_only_change": True,
            "automatic_task_commit": False,
        },
    )
    runtime.invoker = _AttemptPreservingInvoker(runtime.invoker, runtime.run_dir)
    _install_v44_extensions(runtime)
    _install_evidence_extension(runtime)
    _initialize_records(runtime, selected, aliases)
    if runtime.state.get("completed") is True:
        report_path = runtime.run_dir / "report.json"
        if not report_path.is_file():
            raise E2LiveV2Error("completed runtime lacks report.json")
        _assert_hashes_unchanged(protected, _protected_hashes(config))
        return read_json(report_path)
    assignments = _palette_assignments(runtime)
    budgets = {
        slate: IdempotentSlateBudget(runtime, slate, 5)
        for slate in sorted({int(row["slate_index"]) for row in selected})
    }
    scheduler_states = []
    for slate in sorted({int(row["slate_index"]) for row in selected}):
        task_ids = sorted(
            task_id
            for task_id, record in runtime.state["semantic_records"].items()
            if int(record["slate_index"]) == slate
        )
        leases.heartbeat()
        leases.assert_active(
            [
                runtime.state["semantic_records"][task_id]["source_candidate_id"]
                for task_id in task_ids
            ]
        )

        def run_stage(task_id: str, stage: str, *, slate_index: int = slate) -> bool:
            if stage == "planning":
                _plan_one(runtime, task_id)
                return True
            if stage == "realization":
                _realize_one(runtime, task_id, assignments)
                return True
            record = runtime.state["semantic_records"][task_id]
            if stage == "technical":
                if not record.get("technical_results"):
                    return _technical_primary(runtime, task_id)
                from .e2_live_streaming import _technical_one

                _technical_one(runtime, task_id, budgets[slate_index])
                return True
            if stage == "blind":
                if not record.get("blind_results"):
                    return _blind_primary(runtime, task_id)
                from .e2_live_streaming import _blind_one

                _blind_one(runtime, task_id, budgets[slate_index])
                return True
            raise E2LiveV2Error(f"unknown stage {stage}")

        scheduler = PersistentCandidateSchedulerV2(
            state_path=run_dir
            / "candidate_dependency_states"
            / f"slate-{slate:02d}.json",
            candidate_ids=task_ids,
            workers=8,
            next_stage=lambda task_id: _next_stage(runtime, task_id),
            can_dispatch=lambda task_id, stage: _can_dispatch(runtime, task_id, stage),
            run_stage=run_stage,
            reconcile_active=lambda task_id, stage: _reconcile_active(
                runtime, task_id, stage
            ),
            max_downstream_before_new_plan=int(
                config["scheduler"]["max_downstream_before_new_plan"]
            ),
        )
        state = scheduler.run()
        validation = validate_dependency_events_v2(state)
        if not validation["valid"]:
            raise E2LiveV2Error(
                f"slate {slate} dependency validation failed: {validation['errors']}"
            )
        scheduler_states.append(state)

    post_blind = _load_or_freeze_post_blind(runtime)
    leases.heartbeat()
    if not (runtime.run_dir / "editorial_board_results.json").is_file():
        _editorial_boards(runtime)
    prompt_blinding = _assert_downstream_prompt_blinding(runtime)
    if not prompt_blinding["passed"]:
        raise E2LiveV2Error("private E2 labels leaked into downstream prompts")
    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda row: (row["slate_index"], row["source_development_priority"]),
    )
    tasks = _materialize(runtime)
    summary = _safe_summary(records)
    scheduler_report = _aggregate_scheduler_report(runtime, scheduler_states)
    for record in records:
        leases.resolve(
            str(record["source_candidate_id"]),
            str(record.get("intrinsic_quality") or record.get("status") or "terminal"),
        )
    report = {
        "schema_version": 2,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "completed": True,
        "summary": summary,
        "tasks": tasks,
        "scheduler": scheduler_report,
        "post_blind_freeze": {
            "path": str((runtime.run_dir / "post_blind.private.json").resolve()),
            "sha256": _sha(runtime.run_dir / "post_blind.private.json"),
            "record_count": len(post_blind["records"]),
        },
        "reservation_ledger": {
            "path": str(leases.path.resolve()),
            "sha256": _sha(leases.path),
            "all_terminal": all(
                row["state"] == "terminal" for row in leases.state["leases"].values()
            ),
        },
        "prompt_blinding": prompt_blinding,
        "editorial_boards_retained": True,
        "continuous_factory_pooling_used": False,
        "experimental_winner_remains_unpromoted": True,
        "automatic_task_commit_occurred": False,
        "production_state_mutated": False,
    }
    _assert_hashes_unchanged(protected, _protected_hashes(config))
    reduction = scheduler_report["aggregate_observed_reduction_fraction"]
    markdown = "\n".join(
        [
            "# E2 v2 repaired candidate-streaming qualification",
            "",
            f"Selected: {summary['selected']}",
            f"Realized: {summary['realized']}",
            f"Technical pass: {summary['technical_pass']}",
            f"Blind clear: {summary['blind_clear']}",
            f"Core / reserve / ambiguous: {summary['core']} / {summary['reserve']} / {summary['ambiguous']}",
            f"Independent slates: {len(scheduler_states)}",
            f"Observed candidate-graph wall: {scheduler_report['aggregate_observed_streaming_seconds'] / 60:.2f} minutes",
            f"Stage-identical barrier counterfactual: {scheduler_report['aggregate_barrier_counterfactual_seconds'] / 60:.2f} minutes",
            (
                f"Measured reduction: {float(reduction):.1%}"
                if reduction is not None
                else "Measured reduction: unqualified"
            ),
            "",
            "Editorial boards were retained. No task was committed and E2 was not promoted.",
            "",
        ]
    )
    return runtime.finish(report, markdown)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "report"))
    parser.add_argument("--config")
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise E2LiveV2Error("--run is required")
            print(json.dumps(read_json(Path(args.run) / "report.json"), indent=2))
            return 0
        if not args.config:
            raise E2LiveV2Error("--config is required")
        config = load_config(args.config)
        value = (
            validate(config)
            if args.command == "validate"
            else run(config, authorize_model_calls=args.authorize_model_calls)
        )
        print(json.dumps(value, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI is fail closed
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PIPELINE",
    "VERSION",
    "E2LiveV2Error",
    "load_config",
    "run",
    "validate",
]
