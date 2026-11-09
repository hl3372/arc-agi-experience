"""Live candidate-streaming qualification on the frozen V4.8 authoring winner.

Only scheduling changes.  Selected kernels, palette assignment, prompts,
schemas, model routes, escalation ceilings, repair policy, blind information
boundary, editorial boards, and final classification are inherited from the
frozen V4 stack.  The run is experimental and cannot mutate production state
or commit generated tasks.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any

from .config import ConfigError
from .e2_scheduler_replay import POST_SELECTION_ROLES, E2ReplayError, replay_state
from .e2_streaming_scheduler import (
    PersistentCandidateScheduler,
    validate_dependency_events,
)
from .util import atomic_write_json, content_hash, read_json
from .v4_diagnostics import (
    complete_task,
    diagnose_public_arc_task_v4,
)
from .v4_pipeline import V4ControlPipeline
from .v41_experiments import V41Runtime, _anonymous_id
from .v41_integration import (
    _assert_hashes_unchanged,
    _protected_hashes,
    transform_realization_palette,
)
from .v41_supported_departure import (
    _blind_call,
    _palette_assignments,
    _record_stage,
    _repair_call,
    _store_repaired_realization,
    _technical_call,
    _technical_evaluation,
    _validate_realization,
    _wrap_blind,
)
from .v41_test_challenge import _board as _editorial_boards
from .v44_abstraction_led import _install_extensions as _install_v44_extensions
from .v44_abstraction_led import _normalize_palette_claims
from .v44_functional_scale import (
    _arm_summary as _original_arm_summary,
)
from .v44_functional_scale import (
    _private_leaks,
)
from .v44_scale_config import load_v44_scale_config
from .v48_evidence_scale_paired import (
    EVIDENCE_PROMPT,
    _install_evidence_extension,
)
from .validation import validate_arc_task

PIPELINE = "arcgen-e2-live-candidate-streaming"
VERSION = "1.0"
EXPECTED_WINNER_ID = (
    "arcgen-v4.8-evidence-preserving-scale-experimental-winner-20260812"
)
PRIVATE_LABELS = (
    "evidence-preserving scale",
    "evidence preserving scale",
    "functional-scale experiment",
    "streaming scheduler",
    "experimental winner",
    "treatment arm",
    "control arm",
)


class E2LiveError(RuntimeError):
    """Raised when E2 changes more than scheduling or loses provenance."""


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"E2 configuration must be a mapping: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _validated_ref(
    raw: Mapping[str, Any], *, root: Path, name: str
) -> Path:
    path_value = raw.get("path")
    digest = raw.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ConfigError(f"{name} must freeze path and sha256")
    path = _resolve(root, path_value)
    if not path.is_file() or _sha(path) != digest:
        raise ConfigError(f"{name} is missing or changed: {path}")
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != VERSION:
        raise ConfigError(f"configuration must use {PIPELINE} {VERSION}")
    root = source.parent.parent if source.parent.name == "config" else source.parent
    base_ref = raw.get("base_scale_config")
    winner_ref = raw.get("experimental_winner_manifest")
    selected_ref = raw.get("selected_cohort_manifest")
    if not all(isinstance(item, Mapping) for item in (base_ref, winner_ref, selected_ref)):
        raise ConfigError("base, winner, and selected-cohort references are required")
    base = _validated_ref(base_ref, root=root, name="base_scale_config")
    winner_path = _validated_ref(
        winner_ref, root=root, name="experimental_winner_manifest"
    )
    selected_path = _validated_ref(
        selected_ref, root=root, name="selected_cohort_manifest"
    )
    winner = read_json(winner_path)
    if (
        winner.get("baseline_id") != EXPECTED_WINNER_ID
        or winner.get("status")
        != "frozen_experimental_candidate_continue_testing"
        or winner.get("production_policy", {}).get(
            "production_promotion_authorized"
        )
        is not False
    ):
        raise ConfigError("E2 must reference the frozen, unpromoted V4.8 winner")
    settings = raw.get("scheduler")
    run = raw.get("run")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        raise ConfigError("scheduler.enabled must be true")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str):
        raise ConfigError("run.id is required")
    for field in (
        "candidate_level_streaming",
        "unchanged_editorial_boards",
        "whole_slate_reserved_before_execution",
        "reconstruct_barrier_counterfactual",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"scheduler.{field} must be true")
    for field in (
        "automatic_task_commit",
        "production_state_mutation_allowed",
        "production_promotion_allowed",
    ):
        if run.get(field) is not False:
            raise ConfigError(f"run.{field} must be false")
    workers = settings.get("global_workers")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ConfigError("scheduler.global_workers must be in [1,16]")
    starvation = settings.get("max_downstream_before_new_plan")
    if type(starvation) is not int or starvation < 1:
        raise ConfigError("scheduler starvation bound must be positive")
    min_selected = settings.get("minimum_selected")
    max_selected = settings.get("maximum_selected")
    if (
        type(min_selected) is not int
        or type(max_selected) is not int
        or min_selected < 1
        or max_selected < min_selected
    ):
        raise ConfigError("scheduler selected bounds are invalid")
    escalation = settings.get("escalation_calls_per_slate")
    if type(escalation) is not int or escalation != 5:
        raise ConfigError("E2 retains the exact five-call escalation ceiling")
    if run.get("mock") not in {True, False}:
        raise ConfigError("run.mock must be boolean")
    if type(raw.get("seed")) is not int:
        raise ConfigError("seed must be an integer")

    config = load_v44_scale_config(base)
    repository_root = Path(config["_repository_root"]).resolve()
    if repository_root != root.resolve():
        raise ConfigError("base configuration resolved a different repository root")
    config = deepcopy(config)
    config["seed"] = int(raw["seed"])
    config["model"] = deepcopy(config["model"])
    config["model"]["concurrency"] = int(workers)
    config["functional_scale"] = deepcopy(config["functional_scale"])
    config["functional_scale"]["escalation_calls_per_slate"] = int(escalation)
    config["run"] = {
        **deepcopy(config["run"]),
        "id": str(run["id"]),
        "mock": bool(run["mock"]),
        "automatically_commit_generated_tasks": False,
    }
    output_value = raw.get("output_directory")
    if not isinstance(output_value, str) or not output_value:
        raise ConfigError("output_directory is required")
    config["paths"] = deepcopy(config["paths"])
    output = _resolve(root, output_value)
    if output.name != str(run["id"]):
        raise ConfigError("output_directory basename must equal run.id")
    config["paths"]["runs"] = str(output.parent)
    config["scheduler"] = deepcopy(dict(settings))
    config["_e2_config_path"] = str(source)
    config["_e2_output_directory"] = str(output)
    config["_e2_base_config_path"] = str(base)
    config["_e2_winner_manifest_path"] = str(winner_path)
    config["_e2_selected_cohort_path"] = str(selected_path)
    return config


def _snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value) for key, value in config.items() if not key.startswith("_")
    }


def _write_immutable(path: Path, value: Any, *, label: str) -> None:
    if path.exists():
        if read_json(path) != value:
            raise E2LiveError(f"{label} changed during resume")
        return
    atomic_write_json(path, value)


def _implementation_manifest(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    files = (
        "arcgen.lib/e2_live_streaming.py",
        "arcgen.lib/e2_streaming_scheduler.py",
        "arcgen.lib/e2_scheduler_replay.py",
        "arcgen.lib/v41_experiments.py",
        "arcgen.lib/v41_supported_departure.py",
        "arcgen.lib/v41_test_challenge.py",
        "arcgen.lib/v44_functional_scale.py",
        "arcgen.lib/v48_evidence_scale_paired.py",
        f"prompts/{EVIDENCE_PROMPT}",
    )
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "files": {
            name: _sha(root / name) for name in files
        },
        "config": {
            "path": config["_e2_config_path"],
            "sha256": _sha(config["_e2_config_path"]),
        },
        "experimental_winner": {
            "path": config["_e2_winner_manifest_path"],
            "sha256": _sha(config["_e2_winner_manifest_path"]),
        },
    }
    _write_immutable(
        run_dir / "e2_implementation_manifest.json",
        value,
        label="E2 implementation manifest",
    )
    return value


def _load_selected(config: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    frozen_path = run_dir / "selected_kernel_manifest.private.json"
    if frozen_path.exists():
        return deepcopy(read_json(frozen_path)["development_sample"])
    source_path = Path(config["_e2_selected_cohort_path"])
    source = read_json(source_path)
    if (
        source.get("completed") is not True
        or source.get("selection_only") is not True
        or source.get("stopped_before_planning") is not True
    ):
        raise E2LiveError("selected cohort did not stop before planning")
    rows = []
    artifacts = []
    for slate in source.get("slates", []):
        ref = slate.get("selector_source_manifest")
        if not isinstance(ref, Mapping):
            raise E2LiveError("selected slate lacks its selector-source manifest")
        path = Path(str(ref.get("path", "")))
        if not path.is_file() or _sha(path) != ref.get("sha256"):
            raise E2LiveError("selector-source manifest is missing or changed")
        selected = read_json(path)
        if selected.get("selection_stopped_before_planning") is not True:
            raise E2LiveError("selector-source manifest crossed planning boundary")
        artifacts.append({"path": str(path), "sha256": _sha(path)})
        for raw in selected.get("selected", []):
            rows.append(
                {
                    "source_candidate_id": str(raw["candidate_id"]),
                    "slate_index": int(raw["slate_index"]),
                    "source_development_priority": int(
                        raw["selector_global_priority"]
                    ),
                    "search_mode": str(raw["search_mode"]),
                    "private_kernel_author_provider": str(
                        raw["private_kernel_author_provider"]
                    ),
                    "kernel": deepcopy(raw["kernel"]),
                    "editorial_brief": deepcopy(raw["editorial_brief"]),
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["slate_index"]),
            int(row["source_development_priority"]),
        )
    )
    ids = [row["source_candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise E2LiveError("selected cohort repeats a candidate")
    maximum = int(config["scheduler"]["maximum_selected"])
    rows = rows[:maximum]
    if len(rows) < int(config["scheduler"]["minimum_selected"]):
        raise E2LiveError(
            f"only {len(rows)} quality-selected kernels; do not lower selection quality"
        )
    frozen = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "selection_completed_before_scheduling": True,
        "whole_selected_slate_reserved_upstream": True,
        "selected_cohort": {"path": str(source_path), "sha256": _sha(source_path)},
        "selector_source_artifacts": artifacts,
        "development_sample": rows,
    }
    _write_immutable(frozen_path, frozen, label="selected-kernel manifest")
    atomic_write_json(
        run_dir / "selected_kernel_manifest.public.json",
        {
            "schema_version": 1,
            "selected_count": len(rows),
            "slate_count": len({row["slate_index"] for row in rows}),
            "search_mode_counts": dict(
                sorted(Counter(row["search_mode"] for row in rows).items())
            ),
            "provider_labels_hidden_downstream": True,
        },
    )
    return rows


def _aliases(selected: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    # Preserve the frozen V4.8 treatment alias stream. Besides keeping opaque
    # IDs reproducible, this permits a no-new-model-call cache replay of the
    # completed V4.8 cohort as an end-to-end scheduler preflight.
    rng = Random(seed + 480045)
    used: set[str] = set()
    result = {}
    for row in sorted(selected, key=lambda item: str(item["source_candidate_id"])):
        source = str(row["source_candidate_id"])
        _anonymous_id(rng, used)  # paired-control slot in the frozen V4.8 stream
        result[source] = _anonymous_id(rng, used)
    return result


def _initialize_records(
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            expected = {aliases[str(row["source_candidate_id"])] for row in selected}
            if set(runtime.state["semantic_records"]) != expected:
                raise E2LiveError("resumed semantic records target another selected slate")
            return
        records = {}
        for row in selected:
            source = str(row["source_candidate_id"])
            task_id = aliases[source]
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source,
                "lineage_id": content_hash({"source_candidate_id": source}),
                "arm": "single",
                "private_route": "evidence_preserving_scale",
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(
                    row["source_development_priority"]
                ),
                "search_mode": row["search_mode"],
                "private_kernel_author_provider": row[
                    "private_kernel_author_provider"
                ],
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["e2_scheduler"] = {
            "candidate_level_streaming": True,
            "global_workers": int(runtime.config["scheduler"]["global_workers"]),
            "editorial_barrier_retained": True,
        }
        runtime.save()


class IdempotentSlateBudget:
    """Persist the unchanged per-slate call ceiling with resume-safe keys."""

    def __init__(self, runtime: V41Runtime, slate: int, limit: int) -> None:
        self.runtime = runtime
        self.slate = str(slate)
        self.limit = int(limit)
        self.lock = threading.Lock()
        with runtime.lock:
            root = runtime.state.setdefault("e2_escalation_budgets", {})
            root.setdefault(self.slate, {"limit": self.limit, "used": 0, "reservations": {}})
            if root[self.slate]["limit"] != self.limit:
                raise E2LiveError("persisted escalation ceiling changed")
            runtime.save()

    def reserve_once(self, key: str, count: int) -> bool:
        with self.lock, self.runtime.lock:
            row = self.runtime.state["e2_escalation_budgets"][self.slate]
            prior = row["reservations"].get(key)
            if prior is not None:
                if int(prior) != count:
                    raise E2LiveError("escalation reservation changed size on resume")
                return True
            if int(row["used"]) + count > self.limit:
                return False
            row["reservations"][key] = count
            row["used"] = int(row["used"]) + count
            self.runtime.save()
            return True

    @property
    def remaining(self) -> int:
        with self.runtime.lock:
            row = self.runtime.state["e2_escalation_budgets"][self.slate]
            return self.limit - int(row["used"])


def _budgets(runtime: V41Runtime) -> dict[int, IdempotentSlateBudget]:
    limit = int(runtime.config["functional_scale"]["escalation_calls_per_slate"])
    slates = sorted(
        {int(row["slate_index"]) for row in runtime.state["semantic_records"].values()}
    )
    return {slate: IdempotentSlateBudget(runtime, slate, limit) for slate in slates}


def _plan_one(runtime: V41Runtime, task_id: str) -> None:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("status") != "selected":
        return
    brief = deepcopy(record["editorial_brief"])
    brief["optional_development_suggestion"] = None
    values = {
        "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
        "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
        "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
        "CANDIDATE_ID": task_id,
    }
    try:
        result = runtime.call(
            prompt_name="concrete_planner",
            composed_with=("v44_quality_plan", "evidence_scale_addendum"),
            call_id=f"{task_id}-concrete-plan",
            values=values,
            parent_ids=[task_id],
            role="concrete_planner",
            arm_id="single-authoring",
            expected={"candidate_id": task_id},
        )
        plan = result.get("concrete_plan")
        if plan is not None:
            if plan.get("candidate_id") != task_id:
                raise E2LiveError("planner changed nested candidate ID")
            plan = _normalize_palette_claims(plan)
            result["concrete_plan"] = plan
            leaks = _private_leaks(plan)
            if leaks:
                raise E2LiveError(f"planner leaked private labels: {leaks}")
        if (
            plan is not None
            and plan.get("status") == "abort"
            and result.get("scale_decision") != "abort"
        ):
            raise E2LiveError("aborted plan did not use abort scale decision")
    except Exception as exc:  # noqa: BLE001
        _record_stage(
            runtime,
            task_id,
            "planning",
            status="planning_operational_failure",
            planning_error=str(exc),
        )
        return
    status = "plan_aborted" if plan is None or plan.get("status") == "abort" else "plan_ready"
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


def _realize_one(
    runtime: V41Runtime, task_id: str, assignments: Mapping[str, int | None]
) -> None:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("status") != "plan_ready" or record.get("realization") is not None:
        return
    plan = record.get("concrete_plan")
    if not plan or plan.get("status") != "ready":
        _record_stage(
            runtime,
            task_id,
            "realization",
            status="realization_failure",
            realization_error="plan is not ready",
        )
        return
    try:
        realization = runtime.call(
            prompt_name="task_realizer",
            call_id=f"{task_id}-realization",
            values={"CONCRETE_PLAN": plan, "CANDIDATE_ID": task_id},
            parent_ids=[task_id],
            role="task_realizer",
            arm_id="single-authoring",
            expected={"candidate_id": task_id},
            forbidden_prompt_fragments=PRIVATE_LABELS,
        )
        materialize = (getattr(runtime, "contract_bindings", None) or {}).get("materialize_palette")
        if materialize is not None:
            final_plan, final_realization, palette = materialize(
                plan, realization, assignments[task_id], runtime.config["palette_experiment"],
            )
        else:
            final_plan, final_realization, palette = transform_realization_palette(
                candidate_id=task_id,
                plan=plan,
                realization=realization,
                target_background=assignments[task_id],
                palette_config=runtime.config["palette_experiment"],
            )
    except Exception as exc:  # noqa: BLE001
        _record_stage(
            runtime,
            task_id,
            "realization",
            status="realization_operational_failure",
            realization_error=str(exc),
        )
        return
    okay, reason, diagnostics, fidelity = _validate_realization(
        task_id, final_plan, final_realization
    )
    if okay and callable(getattr(runtime, "episode_defect", None)):
        # A runtime may refuse a valid episode before any verification (the v1
        # batch: degenerate test pairs). No repair can change frozen test pairs.
        reason = runtime.episode_defect(complete_task(final_realization))
        okay = not reason
    if not okay:
        _record_stage(
            runtime,
            task_id,
            "realization",
            status="realization_failure",
            realization=final_realization,
            concrete_plan=final_plan,
            palette_treatment=palette,
            plan_fidelity=fidelity,
            realization_error=reason,
        )
        return
    complete = complete_task(final_realization)
    _record_stage(
        runtime,
        task_id,
        "realization",
        status="realized",
        realization=final_realization,
        concrete_plan=final_plan,
        complete_task=complete,
        palette_treatment=palette,
        plan_fidelity=fidelity,
        deterministic_diagnostics=diagnostics,
        technical_diagnostics=diagnose_public_arc_task_v4(final_realization["task"]),
    )
    destination = runtime.run_dir / "tasks" / "single"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / f"{task_id}.arc.json", complete)
    atomic_write_json(
        destination / f"{task_id}.authoring.json",
        {
            "concrete_plan": final_plan,
            "realization": final_realization,
            "palette_treatment": palette,
        },
    )


def _technical_one(
    runtime: V41Runtime, task_id: str, budget: IdempotentSlateBudget
) -> None:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    # Resume immediately after a mechanical repair was stored.
    if (
        record.get("status") == "realized"
        and record.get("technical_results")
        and record.get("repair_count", 0) > 0
        and not record.get("blind_results")
    ):
        try:
            fresh = _technical_call(runtime, task_id, "technical-after-repair")
            latest = deepcopy(runtime.state["semantic_records"][task_id])
            evaluation = _technical_evaluation(latest, fresh)
            history = list(record["technical_results"]) + [evaluation]
            status = (
                "technical_pass"
                if evaluation["passed"] and fresh["confidence"] != "low"
                else "rejected_technical"
            )
            _record_stage(
                runtime, task_id, "technical", status=status, technical_results=history
            )
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "technical_escalation",
                status="escalation_failure",
                escalation_error=str(exc),
            )
        return
    if not record.get("technical_results"):
        try:
            response = _technical_call(runtime, task_id, "technical-primary")
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "technical",
                status="technical_operational_failure",
                technical_error=str(exc),
            )
            return
        record = deepcopy(runtime.state["semantic_records"][task_id])
        evaluation = _technical_evaluation(record, response)
        status = (
            "technical_pass"
            if evaluation["passed"] and response["confidence"] != "low"
            else "technical_needs_resolution"
        )
        _record_stage(
            runtime, task_id, "technical", status=status, technical_results=[evaluation]
        )
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("status") == "technical_pass":
        return
    if record.get("status") != "technical_needs_resolution":
        return
    primary = record["technical_results"][-1]["response"]
    if (
        primary["status"] == "repairable"
        and primary["repair_instruction"].strip()
        and record.get("repair_count", 0) == 0
        and budget.reserve_once(f"{task_id}:technical-mechanical-repair", 2)
    ):
        try:
            repair = _repair_call(
                runtime,
                task_id,
                repair_type="mechanical",
                instruction=primary["repair_instruction"],
                evidence=record["technical_results"][-1],
            )
            if repair["status"] == "repaired":
                okay, reason = _store_repaired_realization(
                    runtime,
                    task_id,
                    repair["repaired_plan"],
                    repair["repaired_realization"],
                    repair_type="mechanical",
                )
                if okay:
                    fresh = _technical_call(runtime, task_id, "technical-after-repair")
                    latest = deepcopy(runtime.state["semantic_records"][task_id])
                    evaluation = _technical_evaluation(latest, fresh)
                    history = list(record["technical_results"]) + [evaluation]
                    status = (
                        "technical_pass"
                        if evaluation["passed"] and fresh["confidence"] != "low"
                        else "rejected_technical"
                    )
                    _record_stage(
                        runtime,
                        task_id,
                        "technical",
                        status=status,
                        technical_results=history,
                    )
                    return
                _record_stage(
                    runtime,
                    task_id,
                    "mechanical_repair_validation",
                    status="technical_needs_resolution",
                    repair_failure_reason=reason,
                )
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "technical_escalation",
                status="escalation_failure",
                escalation_error=str(exc),
            )
            return
    elif budget.reserve_once(f"{task_id}:technical-second", 1):
        try:
            second = _technical_call(runtime, task_id, "technical-second")
            evaluation = _technical_evaluation(record, second)
            history = list(record["technical_results"]) + [evaluation]
            status = (
                "technical_pass"
                if evaluation["passed"] and second["confidence"] != "low"
                else "rejected_technical"
            )
            _record_stage(
                runtime, task_id, "technical", status=status, technical_results=history
            )
            return
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "technical_escalation",
                status="escalation_failure",
                escalation_error=str(exc),
            )
            return
    if runtime.state["semantic_records"][task_id].get("status") == "technical_needs_resolution":
        _record_stage(runtime, task_id, "technical", status="evidence_hold",
                      answer_evidence={"status":"evidence_hold", "hold_kind":"budget",
                                       "reason":"required_technical_review_unfunded"})


def _after_ambiguity_repair(runtime: V41Runtime, task_id: str) -> bool:
    """Resume or execute unchanged technical/blind validation after repair."""

    technical = _technical_call(runtime, task_id, "technical-after-ambiguity-repair")
    current = deepcopy(runtime.state["semantic_records"][task_id])
    evaluation = _technical_evaluation(current, technical)
    if not evaluation["passed"] or technical["confidence"] == "low":
        return False
    blind_response = _blind_call(runtime, task_id, "blind-after-ambiguity-repair")
    wrapped = _wrap_blind(current, blind_response)
    if wrapped["serious_different_output_alternative"]:
        return False
    prior_technical = list(current.get("technical_results") or [])
    prior_blind = list(current.get("blind_results") or [])
    _record_stage(
        runtime,
        task_id,
        "blind",
        status="blind_pass",
        technical_results=prior_technical + [evaluation],
        blind_results=prior_blind + [wrapped],
        ambiguity_reason="one targeted evidence repair removed the alternative",
    )
    return True


def _blind_one(
    runtime: V41Runtime, task_id: str, budget: IdempotentSlateBudget
) -> None:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    # Resume after an ambiguity repair was persisted but before its two checks.
    if (
        record.get("status") == "realized"
        and record.get("blind_results")
        and record.get("repair_count", 0) > 0
    ):
        try:
            if _after_ambiguity_repair(runtime, task_id):
                return
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "ambiguity_repair",
                status="escalation_failure",
                escalation_error=str(exc),
            )
            return
        _record_stage(
            runtime,
            task_id,
            "blind",
            status="ambiguous",
            ambiguity_reason="repaired episode did not pass technical and blind rechecks",
        )
        return
    if not record.get("blind_results"):
        try:
            response = _blind_call(runtime, task_id, "blind-primary")
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "blind",
                status="blind_operational_failure",
                blind_error=str(exc),
            )
            return
        record = deepcopy(runtime.state["semantic_records"][task_id])
        wrapped = _wrap_blind(record, response)
        status = (
            "blind_pass"
            if not wrapped["serious_different_output_alternative"]
            else "blind_needs_resolution"
        )
        _record_stage(
            runtime, task_id, "blind", status=status, blind_results=[wrapped]
        )
    record = deepcopy(runtime.state["semantic_records"][task_id])
    if record.get("status") == "blind_pass":
        return
    if record.get("status") not in {"blind_needs_resolution", "blind_second_complete"}:
        return
    results = list(record["blind_results"])
    if record.get("status") == "blind_second_complete" and len(results) >= 2:
        second = results[-1]
    else:
        if not budget.reserve_once(f"{task_id}:blind-second", 1):
            _record_stage(
                runtime,
                task_id,
                "blind",
                status="ambiguous",
                ambiguity_reason="slate escalation budget exhausted",
            )
            return
        try:
            second_response = _blind_call(runtime, task_id, "blind-second")
            second = _wrap_blind(record, second_response)
            results = results + [second]
            _record_stage(
                runtime,
                task_id,
                "blind_second",
                status="blind_second_complete",
                blind_results=results,
            )
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "blind",
                status="blind_operational_failure",
                blind_error=str(exc),
            )
            return
    if second["serious_different_output_alternative"]:
        latest = deepcopy(runtime.state["semantic_records"][task_id])
        if (
            latest.get("repair_count", 0) == 0
            and budget.reserve_once(f"{task_id}:blind-direct-repair", 3)
        ):
            alternatives = [
                item["response"]["strongest_plausible_alternatives"]
                for item in results
            ]
            instruction = (
                "Adjust or replace one training pair to discriminate the intended rule from "
                "these concrete different-output alternatives without changing the test or core task: "
                + json.dumps(alternatives, ensure_ascii=False)
            )
            try:
                repair = _repair_call(
                    runtime,
                    task_id,
                    repair_type="ambiguity",
                    instruction=instruction,
                    evidence=results,
                )
                if repair["status"] == "repaired":
                    okay, reason = _store_repaired_realization(
                        runtime,
                        task_id,
                        repair["repaired_plan"],
                        repair["repaired_realization"],
                        repair_type="ambiguity",
                    )
                    if okay and _after_ambiguity_repair(runtime, task_id):
                        return
                    if not okay:
                        _record_stage(
                            runtime,
                            task_id,
                            "ambiguity_repair_validation",
                            status="blind_needs_resolution",
                            repair_failure_reason=reason,
                        )
            except Exception as exc:  # noqa: BLE001
                _record_stage(
                    runtime,
                    task_id,
                    "ambiguity_repair",
                    status="escalation_failure",
                    escalation_error=str(exc),
                )
                return
        _record_stage(
            runtime,
            task_id,
            "blind",
            status="ambiguous",
            ambiguity_reason="two blind solvers retained serious different-output alternatives",
        )
        return

    latest = deepcopy(runtime.state["semantic_records"][task_id])
    adjudication = latest.get("ambiguity_adjudication")
    if adjudication is None:
        if not budget.reserve_once(f"{task_id}:ambiguity-adjudication", 1):
            _record_stage(
                runtime,
                task_id,
                "blind",
                status="ambiguous",
                ambiguity_reason="unresolved blind disagreement",
            )
            return
        try:
            adjudication = runtime.call(
                prompt_name="ambiguity_adjudicator",
                call_id=f"{task_id}-ambiguity-adjudication",
                values={
                    "TASK_ID": task_id,
                    "AUTHOR_SPECIFICATION": {
                        "intended_rule": latest["realization"]["authoring_metadata"][
                            "intended_rule"
                        ],
                        "concrete_plan": latest["concrete_plan"],
                    },
                    "COMPLETE_TASK": latest["complete_task"],
                    "BLIND_RESULTS": latest["blind_results"],
                },
                parent_ids=[task_id],
                role="ambiguity_adjudicator",
                arm_id="private-single",
                expected={"task_id": task_id},
                forbidden_prompt_fragments=PRIVATE_LABELS,
            )
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "blind",
                status="blind_operational_failure",
                blind_error=str(exc),
            )
            return
    if adjudication["decision"] == "clear":
        _record_stage(
            runtime,
            task_id,
            "blind",
            status="blind_pass",
            ambiguity_adjudication=adjudication,
        )
        return
    latest = deepcopy(runtime.state["semantic_records"][task_id])
    if (
        adjudication["decision"] == "repairable"
        and latest.get("repair_count", 0) == 0
        and budget.reserve_once(f"{task_id}:blind-adjudicated-repair", 3)
    ):
        try:
            repair = _repair_call(
                runtime,
                task_id,
                repair_type="ambiguity",
                instruction=adjudication["repair_instruction"],
                evidence=latest["blind_results"],
            )
            if repair["status"] == "repaired":
                okay, reason = _store_repaired_realization(
                    runtime,
                    task_id,
                    repair["repaired_plan"],
                    repair["repaired_realization"],
                    repair_type="ambiguity",
                )
                if okay and _after_ambiguity_repair(runtime, task_id):
                    _record_stage(
                        runtime,
                        task_id,
                        "blind",
                        status="blind_pass",
                        ambiguity_adjudication=adjudication,
                        ambiguity_reason="adjudicated local ambiguity was repaired",
                    )
                    return
                if not okay:
                    _record_stage(
                        runtime,
                        task_id,
                        "ambiguity_repair_validation",
                        status="blind_second_complete",
                        repair_failure_reason=reason,
                    )
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                runtime,
                task_id,
                "ambiguity_repair",
                status="escalation_failure",
                escalation_error=str(exc),
            )
            return
    _record_stage(
        runtime,
        task_id,
        "blind",
        status="ambiguous",
        ambiguity_adjudication=adjudication,
        ambiguity_reason=adjudication["reason"],
    )


def _next_stage(runtime: V41Runtime, task_id: str) -> str | None:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        status = record.get("status")
        if status == "selected":
            return "planning"
        if status == "plan_ready":
            return "realization"
        if status == "realized":
            if record.get("blind_results"):
                return "blind"
            return "technical"
        if status == "technical_needs_resolution":
            return "technical"
        if status == "technical_pass":
            return "blind"
        if status in {"blind_needs_resolution", "blind_second_complete"}:
            return "blind"
        return None


def _assert_downstream_prompt_blinding(runtime: V41Runtime) -> dict[str, Any]:
    leaks = []
    root = runtime.run_dir / "model_calls"
    for path in root.glob("*/*/prompt.md"):
        request_path = path.with_name("request.json")
        request = read_json(request_path) if request_path.is_file() else {}
        # The intervention is intentionally visible to the planner and only
        # the planner. The assertion covers every later stage.
        if request.get("stage") == "v44_functional_scale_plan":
            continue
        text = path.read_text(encoding="utf-8").casefold()
        found = [fragment for fragment in PRIVATE_LABELS if fragment in text]
        if found:
            leaks.append({"path": str(path), "fragments": found})
    return {"passed": not leaks, "leaks": leaks}


def _freeze_post_blind(runtime: V41Runtime) -> dict[str, Any]:
    path = runtime.run_dir / "post_blind.private.json"
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "frozen_before_editorial_boards": True,
        "records": deepcopy(runtime.state["semantic_records"]),
        "call_ledger": deepcopy(runtime.state.get("call_ledger", {})),
        "editorial_call_count": sum(
            row.get("role") == "editorial_board"
            for row in runtime.state.get("call_ledger", {}).values()
        ),
    }
    if value["editorial_call_count"]:
        raise E2LiveError("editorial work crossed the post-blind freeze")
    _write_immutable(path, value, label="post-blind freeze")
    return value


def _load_or_freeze_post_blind(runtime: V41Runtime) -> dict[str, Any]:
    """Return the immutable pre-board seam without rewriting it on resume.

    A crash after editorial review has started leaves the live runtime records
    in a later state than the intentionally frozen post-blind snapshot.  In
    that case, recreating the snapshot from current records would both be
    conceptually wrong and make a safe resume impossible.  Validate the
    immutable seam against the task payloads that editorial review cannot
    change, then reuse it verbatim.
    """

    path = runtime.run_dir / "post_blind.private.json"
    if not path.is_file():
        return _freeze_post_blind(runtime)
    value = read_json(path)
    if (
        value.get("pipeline") != PIPELINE
        or value.get("frozen_before_editorial_boards") is not True
        or value.get("editorial_call_count") != 0
        or not isinstance(value.get("records"), Mapping)
    ):
        raise E2LiveError("invalid immutable post-blind freeze on resume")
    frozen_records = value["records"]
    current_records = runtime.state.get("semantic_records", {})
    if set(frozen_records) != set(current_records):
        raise E2LiveError("post-blind candidate set changed during resume")
    for task_id, frozen in frozen_records.items():
        current = current_records[task_id]
        # Boards may add judgments and terminal disposition, but they cannot
        # change the completed episode or any pre-board verification evidence.
        for field in (
            "complete_task",
            "concrete_plan",
            "realization",
            "deterministic_diagnostics",
            "technical_results",
            "blind_results",
            "repair_count",
        ):
            if frozen.get(field) != current.get(field):
                raise E2LiveError(
                    f"post-blind semantic evidence changed during resume: {task_id}:{field}"
                )
    return value


def _safe_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clean = []
    for raw in records:
        row = deepcopy(dict(raw))
        row["concrete_plan"] = row.get("concrete_plan") or {}
        row["deterministic_diagnostics"] = row.get("deterministic_diagnostics") or {}
        clean.append(row)
    return _original_arm_summary(clean)


def _materialize(runtime: V41Runtime) -> list[dict[str, Any]]:
    rows = []
    root = runtime.run_dir / "final_state_exports"
    for record in sorted(
        runtime.state["semantic_records"].values(),
        key=lambda row: (row["slate_index"], row["source_development_priority"]),
    ):
        task_id = str(record["task_id"])
        complete = record.get("complete_task")
        task_path = None
        if complete is not None:
            validation = validate_arc_task(complete, require_test_outputs=True)
            if not validation.valid:
                raise E2LiveError(
                    f"invalid terminal task {task_id}: {'; '.join(validation.errors)}"
                )
            task_path = root / "tasks" / f"{task_id}.arc.json"
            atomic_write_json(task_path, complete)
        authoring_path = root / "authoring" / f"{task_id}.json"
        atomic_write_json(authoring_path, deepcopy(record))
        rows.append(
            {
                "task_id": task_id,
                "source_candidate_id": record["source_candidate_id"],
                "lineage_id": record["lineage_id"],
                "slate_index": record["slate_index"],
                "search_mode": record["search_mode"],
                "status": record.get("status"),
                "realized": complete is not None,
                "technical_pass": bool(
                    (record.get("technical_results") or [{}])[-1].get("passed")
                ),
                "blind_clear": record.get("status") in {
                    "blind_pass",
                    "editorial_complete",
                },
                "repair_count": int(record.get("repair_count") or 0),
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "task_json": str(task_path) if task_path else None,
                "authoring_json": str(authoring_path),
            }
        )
    return rows


def _timing_replay(runtime: V41Runtime, workers: int) -> dict[str, Any]:
    semantic = [
        row
        for row in runtime.state.get("call_ledger", {}).values()
        if row.get("role") in POST_SELECTION_ROLES
    ]
    timed = [
        row
        for row in semantic
        if isinstance((row.get("timing") or {}).get("total_seconds"), (int, float))
        and isinstance(
            (row.get("timing") or {}).get("queued_at_epoch_seconds"), (int, float)
        )
    ]
    if not timed and semantic and all(row.get("cache_source") for row in semantic):
        return {
            "available": False,
            "reason": "cache_only_end_to_end_preflight",
            "semantic_call_count": len(semantic),
            "timed_semantic_call_count": 0,
            "all_calls_reused_from_content_addressed_cache": True,
            "live_timing_qualification": False,
        }
    try:
        value = replay_state(runtime.state_path, workers)
    except E2ReplayError as exc:
        raise E2LiveError(f"exact-call timing replay failed: {exc}") from exc
    return {"available": True, **value, "live_timing_qualification": True}


def run(config: Mapping[str, Any], *, authorize_model_calls: bool) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise E2LiveError("live E2 execution requires --authorize-model-calls")
    run_dir = Path(config["_e2_output_directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    protected = _protected_hashes(config)
    snapshot = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "config": _snapshot(config),
        "config_path": config["_e2_config_path"],
        "config_sha256": _sha(config["_e2_config_path"]),
        "experimental_winner_manifest": {
            "path": config["_e2_winner_manifest_path"],
            "sha256": _sha(config["_e2_winner_manifest_path"]),
        },
        "selected_cohort_manifest": {
            "path": config["_e2_selected_cohort_path"],
            "sha256": _sha(config["_e2_selected_cohort_path"]),
        },
        "protected_production_state": protected,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }
    _write_immutable(
        run_dir / "e2_config.snapshot.json",
        snapshot,
        label="E2 configuration snapshot",
    )
    _implementation_manifest(config, run_dir)
    selected = _load_selected(config, run_dir)
    aliases = _aliases(selected, int(config["seed"]))
    _write_immutable(
        run_dir / "candidate_aliases.private.json",
        {
            "schema_version": 1,
            "aliases": aliases,
            "provider_and_route_hidden_downstream": True,
        },
        label="candidate aliases",
    )
    runtime = V41Runtime(
        deepcopy(dict(config)),
        {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "selected_manifest_sha256": _sha(
                run_dir / "selected_kernel_manifest.private.json"
            ),
            "experimental_winner_manifest_sha256": _sha(
                config["_e2_winner_manifest_path"]
            ),
            "scheduling_only_change": True,
            "automatic_task_commit": False,
        },
    )
    _install_v44_extensions(runtime)
    _install_evidence_extension(runtime)
    _initialize_records(runtime, selected, aliases)
    if runtime.state.get("completed") is True:
        report_path = runtime.run_dir / "report.json"
        if not report_path.is_file():
            raise E2LiveError("completed runtime lacks report.json")
        _assert_hashes_unchanged(protected, _protected_hashes(config))
        return read_json(report_path)
    assignments = _palette_assignments(runtime)
    budgets = _budgets(runtime)
    task_ids = sorted(runtime.state["semantic_records"])

    def run_stage(task_id: str, stage: str) -> None:
        if stage == "planning":
            _plan_one(runtime, task_id)
        elif stage == "realization":
            _realize_one(runtime, task_id, assignments)
        elif stage == "technical":
            record = runtime.state["semantic_records"][task_id]
            _technical_one(runtime, task_id, budgets[int(record["slate_index"])])
        elif stage == "blind":
            record = runtime.state["semantic_records"][task_id]
            _blind_one(runtime, task_id, budgets[int(record["slate_index"])])
        else:
            raise E2LiveError(f"unknown stage {stage}")

    scheduler = PersistentCandidateScheduler(
        state_path=run_dir / "candidate_dependency_state.json",
        candidate_ids=task_ids,
        workers=int(config["scheduler"]["global_workers"]),
        next_stage=lambda task_id: _next_stage(runtime, task_id),
        run_stage=run_stage,
        max_downstream_before_new_plan=int(
            config["scheduler"]["max_downstream_before_new_plan"]
        ),
    )
    scheduler_state = scheduler.run()
    dependency_check = validate_dependency_events(scheduler_state)
    if not dependency_check["valid"]:
        raise E2LiveError("dependency event validation failed")
    post_blind = _load_or_freeze_post_blind(runtime)
    # Editorial review is the unchanged semantic synchronization point. It is
    # deliberately outside the candidate scheduler and remains in the run.
    if not runtime.state.get("completed"):
        _editorial_boards(runtime)
    prompt_blinding = _assert_downstream_prompt_blinding(runtime)
    if not prompt_blinding["passed"]:
        raise E2LiveError("private E2 labels leaked into downstream prompts")
    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda row: (row["slate_index"], row["source_development_priority"]),
    )
    tasks = _materialize(runtime)
    summary = _safe_summary(records)
    replay = _timing_replay(
        runtime, int(config["scheduler"]["global_workers"])
    )
    scheduler_elapsed = float(scheduler_state["completed_at_epoch_seconds"]) - float(
        scheduler_state["started_at_epoch_seconds"]
    )
    barrier = (
        float(replay["barrier_replay"]["makespan_seconds"])
        if replay["available"]
        else None
    )
    actual_reduction = (
        1 - scheduler_elapsed / barrier if barrier is not None and barrier else None
    )
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "version": VERSION,
        "run_id": config["run"]["id"],
        "completed": True,
        "summary": summary,
        "tasks": tasks,
        "scheduler": {
            "worker_count": int(config["scheduler"]["global_workers"]),
            "actual_pre_board_elapsed_seconds": round(scheduler_elapsed, 6),
            "exact_call_barrier_counterfactual_seconds": barrier,
            "actual_vs_barrier_reduction_fraction": (
                round(actual_reduction, 6) if actual_reduction is not None else None
            ),
            "replay_streaming_makespan_seconds": (
                replay["candidate_streaming_replay"]["makespan_seconds"]
                if replay["available"]
                else None
            ),
            "replay_estimated_reduction_fraction": (
                replay["estimated_wall_reduction_fraction"]
                if replay["available"]
                else None
            ),
            "measured_effective_concurrency": (
                replay["candidate_streaming_replay"]["effective_concurrency"]
                if replay["available"]
                else None
            ),
            "live_timing_qualification": replay["live_timing_qualification"],
            "dependency_check": dependency_check,
            "state": {
                "path": str((run_dir / "candidate_dependency_state.json").resolve()),
                "sha256": _sha(run_dir / "candidate_dependency_state.json"),
            },
        },
        "post_blind_freeze": {
            "path": str((runtime.run_dir / "post_blind.private.json").resolve()),
            "sha256": _sha(runtime.run_dir / "post_blind.private.json"),
            "record_count": len(post_blind["records"]),
        },
        "prompt_blinding": prompt_blinding,
        "exact_call_replay": replay,
        "editorial_boards_retained": True,
        "experimental_winner_remains_unpromoted": True,
        "automatic_task_commit_occurred": False,
        "production_state_mutated": False,
    }
    _assert_hashes_unchanged(protected, _protected_hashes(config))
    markdown = "\n".join(
        [
            "# E2 live candidate-streaming qualification",
            "",
            f"Selected: {summary['selected']}",
            f"Realized: {summary['realized']}",
            f"Technical pass: {summary['technical_pass']}",
            f"Blind clear: {summary['blind_clear']}",
            f"Core / reserve / ambiguous: {summary['core']} / {summary['reserve']} / {summary['ambiguous']}",
            f"Actual pre-board candidate graph: {scheduler_elapsed / 60:.2f} minutes",
            (
                f"Exact-call barrier counterfactual: {barrier / 60:.2f} minutes"
                if barrier is not None
                else "Exact-call barrier counterfactual: unavailable (cache-only preflight)"
            ),
            (
                f"Measured reduction: {actual_reduction:.1%}"
                if actual_reduction is not None
                else "Measured reduction: not qualified by cache-only preflight"
            ),
            "",
            "The experimental winner remains unpromoted. No task was committed.",
            "",
        ]
    )
    return runtime.finish(report, markdown)


def validate(config: Mapping[str, Any]) -> dict[str, Any]:
    selected = read_json(Path(config["_e2_selected_cohort_path"]))
    return {
        "valid": selected.get("completed") is True,
        "pipeline": PIPELINE,
        "run_id": config["run"]["id"],
        "selected_slate_count": len(selected.get("slates", [])),
        "global_workers": config["scheduler"]["global_workers"],
        "experimental_winner": EXPECTED_WINNER_ID,
        "editorial_boards_retained": True,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }


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
                raise E2LiveError("--run is required")
            print(json.dumps(read_json(Path(args.run) / "report.json"), indent=2))
            return 0
        if not args.config:
            raise E2LiveError("--config is required")
        config = load_config(args.config)
        result = (
            validate(config)
            if args.command == "validate"
            else run(config, authorize_model_calls=args.authorize_model_calls)
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


# --- r23 overlay (materialized): shared review budget and release hook ---
from arcgen.batch.review_budget import SharedSlateBudget
IdempotentSlateBudget=SharedSlateBudget
_frozen_technical_one=_technical_one


def _technical_one(runtime,task_id,budget):
    result=_frozen_technical_one(runtime,task_id,budget)
    # A pause/interruption propagates before here, preserving unfinished packages.
    budget.release_task(task_id)
    return result
