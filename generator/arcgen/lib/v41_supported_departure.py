"""Paired ARCGEN V4.1 supported-departure experiment.

The experiment freezes three ordinary V4 selector slates, plans the same four
kernels from each slate in control and treatment arms, and then applies the
unchanged V4 realization, verification, blind, and editorial policy.  The
treatment instruction is visible only to its planner.  Every later stage sees
an opaque task ID and an ordinary concrete plan.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import math
import threading
import time

from .factory_contracts import plan_fidelity, display_value
from .embeddings import build_embedding_provider
from .lean_pipeline import serious_blind_alternative
from .lean_v3_pipeline import classify_v3_editorial_results
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import atomic_write_json, atomic_write_text, content_hash, parallel_map, read_json
from .v4_config import load_v4_config
from .v4_diagnostics import (
    check_plan_fidelity,
    complete_task,
    diagnose_arc_task_v4,
    diagnose_public_arc_task_v4,
)
from .v4_pipeline import V4ControlPipeline, V4ReservoirProducer, v4_call_accounting
from .v4_reservoir import load_reservoir, reservoir_status
from .validation import validate_arc_task
from .v41_audits import (
    genuine_task_ids,
    matched_noninferiority,
    mechanism_diversity_summary,
    semantic_promotion_assessment,
)
from .v41_experiments import (
    V41ExperimentError,
    V41Runtime,
    _anchors,
    _anonymous_id,
    _portfolio_digest,
    _task_metrics,
)
from .v41_integration import (
    _committed_distribution,
    _palette_target,
    _protected_hashes,
    _assert_hashes_unchanged,
    transform_realization_palette,
)
from .v41_palette import schedule_backgrounds


SPECIAL_LABEL_FRAGMENTS = (
    "supported departure",
    "supported-departure",
    "curveball",
    "experimental arm",
    "treatment arm",
)


class SupportedDepartureError(V41ExperimentError):
    pass


class SlateEscalationBudget:
    def __init__(
        self,
        limit: int,
        *,
        used: int = 0,
        on_change: Callable[[int], None] | None = None,
    ) -> None:
        self.limit = int(limit)
        self.used = int(used)
        if not 0 <= self.used <= self.limit:
            raise SupportedDepartureError("persisted escalation usage is outside the slate limit")
        self.on_change = on_change
        self.lock = threading.Lock()

    def reserve(self, count: int) -> bool:
        with self.lock:
            if count < 0 or self.used + count > self.limit:
                return False
            self.used += count
            if self.on_change is not None:
                self.on_change(self.used)
            return True

    @property
    def remaining(self) -> int:
        with self.lock:
            return self.limit - self.used


def _selection_sandbox_config(
    config: Mapping[str, Any], run_dir: Path, *, slate_index: int
) -> dict[str, Any]:
    v4 = load_v4_config(config["_v4_control_config"])
    sandbox = run_dir / "selection_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    private = sandbox / "reservoir.json"
    source = Path(config["paths"]["reservoir"])
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_manifest = sandbox / "source.json"
    if private.exists():
        if read_json(source_manifest)["source_sha256"] != source_sha:
            raise SupportedDepartureError(
                "production reservoir changed after the semantic experiment snapshot was frozen"
            )
    else:
        atomic_write_json(private, read_json(source))
        atomic_write_json(
            source_manifest,
            {"source_path": str(source), "source_sha256": source_sha},
        )
    v4["paths"]["reservoir"] = str(private)
    v4["paths"]["runs"] = str((run_dir / "selection_graph").resolve())
    v4["run"]["id"] = f"selection-slate-{slate_index:02d}"
    v4["run"]["automatically_commit_generated_tasks"] = False
    return v4


def _ensure_selection_window(
    config: Mapping[str, Any], v4: dict[str, Any], run_dir: Path, *, slate_index: int
) -> dict[str, Any]:
    reservoir = load_reservoir(v4["paths"]["reservoir"])
    proposed = sum(item["state"] == "proposed" for item in reservoir["proposals"])
    required = int(v4["selection"]["eligible_window_size"])
    if proposed >= required:
        return {
            "generated": 0,
            "reason": "eligible-window supply already sufficient",
            "reservoir": reservoir_status(reservoir, v4["reservoir"]),
        }
    cycles = max(1, math.ceil((required - proposed) / 12))
    producer = V4ReservoirProducer(
        v4,
        supply_run_id=f"{config['run']['id']}-slate-{slate_index:02d}-supply",
    )
    result = producer.run(cycles=cycles, force_between_watermarks=True)
    final = load_reservoir(v4["paths"]["reservoir"])
    final_proposed = sum(item["state"] == "proposed" for item in final["proposals"])
    if final_proposed < required:
        raise SupportedDepartureError(
            f"slate {slate_index} has {final_proposed} proposals for a {required}-proposal window"
        )
    return result


def _freeze_three_selected_samples(
    config: Mapping[str, Any], run_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = run_dir / "selected_kernel_manifest.json"
    if manifest_path.exists():
        value = read_json(manifest_path)
        return value["development_sample"], value["slates"]
    selected: list[dict[str, Any]] = []
    slates = []
    for slate_index in range(1, int(config["semantic_experiments"]["minimum_independent_slates"]) + 1):
        v4 = _selection_sandbox_config(config, run_dir, slate_index=slate_index)
        supply = _ensure_selection_window(config, v4, run_dir, slate_index=slate_index)
        selector = V4ControlPipeline(v4)
        briefs = selector.select_slate()
        state = selector.state
        if len(briefs) < 4:
            raise SupportedDepartureError(
                f"V4 selector slate {slate_index} returned fewer than four candidates"
            )
        sample = []
        for brief in sorted(briefs, key=lambda item: item["development_priority"])[:4]:
            candidate_id = brief["candidate_id"]
            record = state["records"][candidate_id]
            row = {
                "source_candidate_id": candidate_id,
                "slate_index": slate_index,
                "slate_run_id": v4["run"]["id"],
                "source_development_priority": record["development_priority"],
                "kernel": {
                    key: deepcopy(record[key])
                    for key in (
                        "candidate_id", "search_mode", "central_relation", "why_coherent",
                        "novel_concept_potential", "functional_composition_potential",
                        "why_multiple_training_examples_may_be_useful",
                        "possible_test_generalization", "main_authoring_risk",
                    )
                },
                "editorial_brief": deepcopy(record["editorial_brief"]),
            }
            selected.append(row)
            sample.append(candidate_id)
        slates.append(
            {
                "slate_index": slate_index,
                "selection_run_id": v4["run"]["id"],
                "frozen_window": deepcopy(state["frozen_window"]),
                "full_selected_slate": [item["candidate_id"] for item in briefs],
                "development_sample": sample,
                "supply": supply,
                "state_path": str(selector.state_path),
                "arm_information_seen_by_selector": False,
            }
        )
    if len(selected) < int(config["semantic_experiments"]["minimum_treatment_authoring_candidates"]):
        raise SupportedDepartureError("three selected samples did not meet the 12-kernel minimum")
    value = {
        "schema_version": 1,
        "independent_slate_count": len(slates),
        "development_sample_count": len(selected),
        "control_and_treatment_use_same_kernels": True,
        "slates": slates,
        "development_sample": selected,
    }
    atomic_write_json(manifest_path, value)
    return selected, slates


def _route_guidance(route: str) -> str:
    return V4ControlPipeline._route_guidance(route)


def _private_aliases(selected: Sequence[Mapping[str, Any]], seed: int) -> dict[tuple[str, str], str]:
    rng = Random(seed + 4217)
    used: set[str] = set()
    keys = [
        (str(item["source_candidate_id"]), arm)
        for item in selected for arm in ("control", "treatment")
    ]
    rng.shuffle(keys)
    return {key: _anonymous_id(rng, used) for key in keys}


def _source_manifest(
    config: Mapping[str, Any], selected: Sequence[Mapping[str, Any]],
    slates: Sequence[Mapping[str, Any]], aliases: Mapping[tuple[str, str], str],
    protected: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "feature": "supported_departure",
        "assignment_seed": config["seed"],
        "frozen_v4_parent": deepcopy(config["_v4_parent_verification"]),
        "feature_manifest_snapshot": deepcopy(config["_feature_manifest_snapshot"]),
        "effective_accepted_baseline": deepcopy(config["_effective_features"]),
        "protected_production_state": deepcopy(dict(protected)),
        "fixed_anchor_manifest_sha256": hashlib.sha256(
            Path(config["paths"]["fixed_anchor_manifest"]).read_bytes()
        ).hexdigest(),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "selected_kernels": deepcopy(list(selected)),
        "slates": deepcopy(list(slates)),
        "private_arm_mapping": [
            {
                "source_candidate_id": source,
                "arm": arm,
                "opaque_task_id": alias,
            }
            for (source, arm), alias in sorted(aliases.items())
        ],
        "arm_mapping_is_never_bound_to_downstream_prompts": True,
    }


def _initialize_records(
    runtime: V41Runtime, selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[tuple[str, str], str],
) -> dict[str, dict[str, Any]]:
    with runtime.lock:
        if "semantic_records" not in runtime.state:
            records = {}
            for source in selected:
                source_id = str(source["source_candidate_id"])
                for arm in ("control", "treatment"):
                    alias = aliases[(source_id, arm)]
                    records[alias] = {
                        "task_id": alias,
                        "source_candidate_id": source_id,
                        "arm": arm,
                        "slate_index": int(source["slate_index"]),
                        "source_development_priority": int(source["source_development_priority"]),
                        "search_mode": source["kernel"]["search_mode"],
                        "kernel": deepcopy(source["kernel"]),
                        "editorial_brief": deepcopy(source["editorial_brief"]),
                        "status": "selected",
                        "stage_history": [],
                        "repair_count": 0,
                    }
            runtime.state["semantic_records"] = records
            runtime.save()
        return runtime.state["semantic_records"]


def _record_stage(runtime: V41Runtime, task_id: str, stage: str, **values: Any) -> None:
    with runtime.lock:
        record = runtime.state["semantic_records"][task_id]
        record.update(deepcopy(values))
        record.setdefault("stage_history", []).append(
            {"stage": stage, "at_epoch_seconds": round(time.time(), 3), "status": values.get("status")}
        )
        runtime.save()


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def plan_one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {**deepcopy(record["editorial_brief"]), "candidate_id": task_id},
            "ROUTE_GUIDANCE": _route_guidance(record["editorial_brief"]["route"]),
            "CANDIDATE_ID": task_id,
        }
        try:
            if record["arm"] == "control":
                response = runtime.call(
                    prompt_name="concrete_planner",
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="private-control",
                    expected={"candidate_id": task_id},
                )
                return task_id, {
                    "treatment_decision": "not_applicable",
                    "decision_reason": "ordinary frozen V4 planning",
                    "supported_departure_contract": None,
                    "concrete_plan": response,
                }, None

            def validate(value: dict[str, Any]) -> None:
                nested = value.get("concrete_plan")
                if nested is not None and nested.get("candidate_id") != task_id:
                    raise SupportedDepartureError("treatment plan returned a mismatched nested candidate ID")
                if nested is not None and nested.get("status") == "abort" and value["treatment_decision"] != "abort":
                    raise SupportedDepartureError("an aborted concrete plan must use the abort decision")

            response = runtime.call(
                prompt_name="concrete_planner",
                composed_with=("supported_addendum",),
                call_id=f"{task_id}-supported-plan",
                values=values,
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="private-treatment",
                expected={"candidate_id": task_id},
                validator=validate,
            )
            return task_id, response, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [task_id for task_id, record in records.items() if "planning_result" not in record]
    results = parallel_map(plan_one, pending, min(len(pending), int(runtime.config["model"]["concurrency"]))) if pending else []
    for task_id, result, error in results:
        if error:
            _record_stage(runtime, task_id, "planning", status="operational_failure", planning_error=error)
            continue
        assert result is not None
        plan = result["concrete_plan"]
        status = "plan_aborted" if plan is None or plan["status"] == "abort" else "plan_ready"
        _record_stage(runtime, task_id, "planning", status=status, planning_result=result, concrete_plan=plan)
        directory = runtime.run_dir / "plans" / records[task_id]["arm"]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / f"{task_id}.json", result)


def _palette_assignments(runtime: V41Runtime) -> dict[str, int | None]:
    path = runtime.run_dir / "palette_assignments.json"
    if path.exists():
        return read_json(path)["assignments"]
    records = runtime.state["semantic_records"]
    sources = sorted({record["source_candidate_id"] for record in records.values()})
    target, reference = _palette_target(runtime.config)
    committed = _committed_distribution(runtime.config)
    by_source = schedule_backgrounds(
        sources,
        seed=int(runtime.config["seed"]),
        current_counts=committed["counts"],
        target_distribution=target,
        eligible={item: True for item in sources},
    )
    assignments = {
        task_id: by_source[record["source_candidate_id"]]
        for task_id, record in records.items()
    }
    atomic_write_json(
        path,
        {
            "same_requested_background_for_matched_arms": True,
            "assignments": assignments,
            "by_source_candidate": by_source,
            "target_distribution": {str(key): value for key, value in target.items()},
            "real_reference": reference,
            "committed_distribution": committed,
        },
    )
    return assignments


def _validate_realization(
    task_id: str, plan: Mapping[str, Any], realization: Mapping[str, Any]
) -> tuple[bool, str, dict[str, Any] | None, dict[str, Any] | None]:
    if realization.get("status") != "realized":
        return False, realization.get("refusal_reason", "realizer refused"), None, None
    fidelity = plan_fidelity(plan, realization)
    if not fidelity["passed"]:
        return False, "; ".join(fidelity["objective_failures"]), None, fidelity
    public_validation = validate_arc_task(realization["task"], require_test_outputs=False)
    if not public_validation.valid:
        return False, "; ".join(public_validation.errors), None, fidelity
    full = complete_task(realization)
    full_validation = validate_arc_task(full, require_test_outputs=True)
    if not full_validation.valid:
        return False, "; ".join(full_validation.errors), None, fidelity
    diagnostics = diagnose_arc_task_v4(full)
    if not diagnostics["hard_valid"]:
        return False, "; ".join(diagnostics.get("errors", [])), diagnostics, fidelity
    return True, "", diagnostics, fidelity


def _realize_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]
    assignments = _palette_assignments(runtime)

    def realize_one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        plan = record.get("concrete_plan")
        if not plan or plan.get("status") != "ready":
            return task_id, None, "plan is not ready"
        try:
            response = runtime.call(
                prompt_name="task_realizer",
                call_id=f"{task_id}-realization",
                values={"CONCRETE_PLAN": plan, "CANDIDATE_ID": task_id},
                parent_ids=[task_id],
                role="task_realizer",
                arm_id=f"private-{record['arm']}",
                expected={"candidate_id": task_id},
                forbidden_prompt_fragments=SPECIAL_LABEL_FRAGMENTS,
            )
            final_plan, final_response, palette = transform_realization_palette(
                candidate_id=task_id,
                plan=plan,
                realization=response,
                target_background=assignments[task_id],
                palette_config=runtime.config["palette_experiment"],
            )
            return task_id, {
                "concrete_plan": final_plan,
                "realization": final_response,
                "palette_treatment": palette,
            }, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, record in records.items()
        if record.get("status") in {"plan_ready", "realization_operational_failure"}
        and "realization" not in record
    ]
    results = parallel_map(realize_one, pending, min(len(pending), int(runtime.config["model"]["concurrency"]))) if pending else []
    for task_id, result, error in results:
        if error or result is None:
            _record_stage(runtime, task_id, "realization", status="realization_operational_failure", realization_error=error)
            continue
        okay, reason, diagnostics, fidelity = _validate_realization(
            task_id, result["concrete_plan"], result["realization"]
        )
        if not okay:
            _record_stage(
                runtime, task_id, "realization", status="realization_failure",
                realization=result["realization"], concrete_plan=result["concrete_plan"],
                palette_treatment=result["palette_treatment"], plan_fidelity=fidelity,
                realization_error=reason,
            )
            continue
        public_diagnostics = diagnose_public_arc_task_v4(result["realization"]["task"])
        _record_stage(
            runtime, task_id, "realization", status="realized",
            realization=result["realization"], concrete_plan=result["concrete_plan"],
            complete_task=complete_task(result["realization"]),
            palette_treatment=result["palette_treatment"], plan_fidelity=fidelity,
            deterministic_diagnostics=diagnostics,
            technical_diagnostics=public_diagnostics,
        )
        arm = records[task_id]["arm"]
        directory = runtime.run_dir / "tasks" / arm
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / f"{task_id}.arc.json", complete_task(result["realization"]))
        atomic_write_json(directory / f"{task_id}.authoring.json", result)


def _technical_evaluation(record: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
    hidden_match = response["derived_test_outputs"] == hidden
    required = set(record["technical_diagnostics"].get("pair_economy_review_codes", []))
    assessed = {item["flag_code"]: item for item in response["diagnostic_flag_assessments"]}
    unresolved = [code for code in sorted(required) if code not in assessed or not assessed[code]["justified"]]
    passed = (
        response["status"] == "pass"
        and response["mechanically_correct"]
        and response["test_supported"]
        and response["one_answer_supported"]
        and response["plan_fidelity"] != "fail"
        and response["evidence_sufficiency"] == "sufficient"
        and response["pair_economy"] != "fail"
        and response["composition_preserved"]
        and not response["arbitrary_exceptions"]
        and not response["unsupported_test_semantics"]
        and not response["unapproved_realizer_decisions"]
        and hidden_match
        and not unresolved
    )
    return {
        "response": deepcopy(dict(response)),
        "hidden_output_match": hidden_match,
        "unresolved_pair_economy_flags": unresolved,
        "passed": passed,
    }


def _technical_call(runtime: V41Runtime, task_id: str, suffix: str) -> dict[str, Any]:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    verifier_id = "verifier-" + content_hash({"task": task_id, "suffix": suffix})[:12]
    return runtime.call(
        prompt_name="technical_verifier",
        call_id=f"{task_id}-{suffix}",
        values={
            "INTENDED_RULE": record["realization"]["authoring_metadata"]["intended_rule"],
            "CONCRETE_PLAN": record["concrete_plan"],
            "PUBLIC_TASK": record["realization"]["task"],
            "DIAGNOSTICS": record["technical_diagnostics"],
            "TASK_ID": task_id,
            "VERIFIER_ID": verifier_id,
        },
        parent_ids=[task_id],
        role="technical_verifier",
        arm_id=f"private-{record['arm']}",
        expected={"task_id": task_id, "verifier_id": verifier_id},
        forbidden_prompt_fragments=SPECIAL_LABEL_FRAGMENTS,
    )


def _store_repaired_realization(
    runtime: V41Runtime, task_id: str, repaired_plan: Mapping[str, Any],
    repaired: Mapping[str, Any], *, repair_type: str,
) -> tuple[bool, str]:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    original_public_test = record["realization"]["task"]["test"]
    original_hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
    repaired_public_test = repaired["task"]["test"]
    repaired_hidden = repaired["authoring_metadata"]["expected_test_outputs"]
    policy = runtime.config.get("repair_policy", {})
    allow_mechanical_restoration = bool(
        isinstance(policy, Mapping)
        and policy.get("allow_mechanical_test_input_restoration") is True
        and repair_type == "mechanical"
    )
    if repaired_hidden != original_hidden:
        return False, "repair changed the frozen expected output"
    if repaired_public_test != original_public_test:
        if not allow_mechanical_restoration:
            return False, "repair changed the frozen test input"
        if len(repaired_public_test) != len(original_public_test):
            return False, "mechanical repair changed the test-pair count"
        for original_pair, repaired_pair in zip(original_public_test, repaired_public_test):
            original_grid = original_pair.get("input")
            repaired_grid = repaired_pair.get("input")
            if not original_grid or not repaired_grid:
                return False, "mechanical repair produced a malformed test input"
            if (
                len(original_grid) != len(repaired_grid)
                or len(original_grid[0]) != len(repaired_grid[0])
            ):
                return False, "mechanical repair changed test-input dimensions"
        original_rule = record["realization"]["authoring_metadata"].get("intended_rule")
        repaired_rule = repaired["authoring_metadata"].get("intended_rule")
        if repaired_rule != original_rule:
            return False, "mechanical repair changed the intended rule"
    okay, reason, diagnostics, fidelity = _validate_realization(task_id, repaired_plan, repaired)
    if okay and callable(getattr(runtime, "episode_defect", None)):
        reason = runtime.episode_defect(complete_task(repaired))  # e.g. a restored test input equal to its answer
        okay = not reason
    if not okay:
        return False, reason
    _record_stage(
        runtime, task_id, f"{repair_type}_repair", status="realized",
        repair_count=record.get("repair_count", 0) + 1,
        concrete_plan=deepcopy(dict(repaired_plan)), realization=deepcopy(dict(repaired)),
        complete_task=complete_task(repaired), plan_fidelity=fidelity,
        deterministic_diagnostics=diagnostics,
        technical_diagnostics=diagnose_public_arc_task_v4(repaired["task"]),
    )
    return True, ""


def _repair_call(
    runtime: V41Runtime, task_id: str, *, repair_type: str,
    instruction: str, evidence: Any, call_id: str | None = None,
) -> dict[str, Any]:
    # A caller allowing several repairs gives each later repair its own call ID (default: the one-repair ID).
    record = deepcopy(runtime.state["semantic_records"][task_id])
    policy = runtime.config.get("repair_policy", {})
    if (
        isinstance(policy, Mapping)
        and policy.get("require_consistent_deviation_ledgers") is True
    ):
        instruction = (
            instruction
            + "\n\nRepair-output contract: the repaired realization's top-level deviations "
            "and authoring_metadata.plan_fidelity.deviations must be identical. Because the "
            "repaired plan and repaired grids are returned together, record no deviation when "
            "the realization follows that repaired plan. Do not describe the requested repair "
            "itself as a deviation from the repaired plan."
        )
    return runtime.call(
        prompt_name="targeted_repair",
        call_id=call_id or f"{task_id}-{repair_type}-repair",
        values={
            "CANDIDATE_ID": task_id,
            "REPAIR_TYPE": repair_type,
            "REPAIR_INSTRUCTION": instruction,
            "AUTHOR_SPECIFICATION": {
                "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concrete_plan": record["concrete_plan"],
                "realization": record["realization"],
            },
            "EVALUATOR_EVIDENCE": evidence,
        },
        parent_ids=[task_id],
        role="targeted_repair",
        arm_id=f"private-{record['arm']}",
        expected={"candidate_id": task_id, "repair_type": repair_type},
        forbidden_prompt_fragments=SPECIAL_LABEL_FRAGMENTS,
    )


def _run_technical(
    runtime: V41Runtime,
    budgets: Mapping[tuple[int, str], SlateEscalationBudget],
) -> None:
    records = runtime.state["semantic_records"]

    def primary(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            return task_id, _technical_call(runtime, task_id, "technical-primary"), None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, record in records.items()
        if record.get("status") in {"realized", "technical_operational_failure"}
        and record.get("complete_task")
        and "technical_results" not in record
    ]
    results = parallel_map(primary, pending, min(len(pending), int(runtime.config["model"]["concurrency"]))) if pending else []
    for task_id, response, error in results:
        if error or response is None:
            _record_stage(runtime, task_id, "technical", status="technical_operational_failure", technical_error=error)
            continue
        record = deepcopy(runtime.state["semantic_records"][task_id])
        evaluation = _technical_evaluation(record, response)
        status = "technical_pass" if evaluation["passed"] and response["confidence"] != "low" else "technical_needs_resolution"
        _record_stage(runtime, task_id, "technical", status=status, technical_results=[evaluation])

    unresolved = sorted(
        (
            task_id for task_id, record in runtime.state["semantic_records"].items()
            if record.get("status") == "technical_needs_resolution"
        ),
        key=lambda task_id: (
            runtime.state["semantic_records"][task_id]["slate_index"],
            runtime.state["semantic_records"][task_id]["source_development_priority"],
            runtime.state["semantic_records"][task_id]["arm"],
        ),
    )
    for task_id in unresolved:
        record = deepcopy(runtime.state["semantic_records"][task_id])
        primary_response = record["technical_results"][-1]["response"]
        budget = budgets[(record["slate_index"], record["arm"])]
        if (
            primary_response["status"] == "repairable"
            and primary_response["repair_instruction"].strip()
            and record.get("repair_count", 0) == 0
            and budget.reserve(2)
        ):
            try:
                repair = _repair_call(
                    runtime, task_id, repair_type="mechanical",
                    instruction=primary_response["repair_instruction"], evidence=record["technical_results"][-1],
                )
                if repair["status"] == "repaired":
                    okay, reason = _store_repaired_realization(
                        runtime, task_id, repair["repaired_plan"], repair["repaired_realization"],
                        repair_type="mechanical",
                    )
                    if okay:
                        fresh = _technical_call(runtime, task_id, "technical-after-repair")
                        latest = deepcopy(runtime.state["semantic_records"][task_id])
                        evaluation = _technical_evaluation(latest, fresh)
                        history = record["technical_results"] + [evaluation]
                        status = "technical_pass" if evaluation["passed"] and fresh["confidence"] != "low" else "rejected_technical"
                        _record_stage(runtime, task_id, "technical", status=status, technical_results=history)
                        continue
                    _record_stage(
                        runtime,
                        task_id,
                        "mechanical_repair_validation",
                        status="technical_needs_resolution",
                        repair_failure_reason=reason,
                    )
            except Exception as exc:
                _record_stage(runtime, task_id, "technical_escalation", status="escalation_failure", escalation_error=str(exc))
        elif (
            primary_response["status"] in {"uncertain", "pass"}
            or primary_response["confidence"] == "low"
            or not record["technical_results"][-1]["hidden_output_match"]
        ) and budget.reserve(1):
            try:
                second = _technical_call(runtime, task_id, "technical-second")
                evaluation = _technical_evaluation(record, second)
                history = record["technical_results"] + [evaluation]
                status = "technical_pass" if evaluation["passed"] and second["confidence"] != "low" else "rejected_technical"
                _record_stage(runtime, task_id, "technical", status=status, technical_results=history)
                continue
            except Exception as exc:
                _record_stage(runtime, task_id, "technical_escalation", status="escalation_failure", escalation_error=str(exc))
        if runtime.state["semantic_records"][task_id].get("status") == "technical_needs_resolution":
            _record_stage(runtime, task_id, "technical", status="rejected_technical")


def _blind_call(runtime: V41Runtime, task_id: str, suffix: str) -> dict[str, Any]:
    record = deepcopy(runtime.state["semantic_records"][task_id])
    public = deepcopy(record["realization"]["task"])
    if any("output" in pair for pair in public["test"]):
        raise SupportedDepartureError("hidden output entered blind solver input")
    solver_id = "solver-" + content_hash({"task": task_id, "suffix": suffix})[:12]
    opaque = "blind-" + content_hash({"seed": runtime.config["seed"], "task": task_id})[:12]
    return runtime.call(
        prompt_name="blind_solver",
        call_id=f"{task_id}-{suffix}",
        values={"PUBLIC_TASK": public, "TASK_ID": opaque, "SOLVER_ID": solver_id},
        parent_ids=[task_id],
        role="blind_solver",
        arm_id=f"private-{record['arm']}",
        expected={"task_id": opaque, "solver_id": solver_id},
        forbidden_prompt_fragments=SPECIAL_LABEL_FRAGMENTS,
    )


def _wrap_blind(record: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
    return {
        "response": deepcopy(dict(response)),
        "primary_matches_hidden": response["predicted_test_outputs"] == hidden,
        "serious_different_output_alternative": serious_blind_alternative(dict(response), hidden),
    }


def _run_blind(
    runtime: V41Runtime,
    budgets: Mapping[tuple[int, str], SlateEscalationBudget],
) -> None:
    records = runtime.state["semantic_records"]

    def primary(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            return task_id, _blind_call(runtime, task_id, "blind-primary"), None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, record in records.items()
        if record.get("status") in {"technical_pass", "blind_operational_failure"}
        and "blind_results" not in record
    ]
    results = parallel_map(primary, pending, min(len(pending), int(runtime.config["model"]["concurrency"]))) if pending else []
    for task_id, response, error in results:
        if error or response is None:
            _record_stage(runtime, task_id, "blind", status="blind_operational_failure", blind_error=error)
            continue
        record = deepcopy(runtime.state["semantic_records"][task_id])
        wrapped = _wrap_blind(record, response)
        status = "blind_pass" if not wrapped["serious_different_output_alternative"] else "blind_needs_resolution"
        _record_stage(runtime, task_id, "blind", status=status, blind_results=[wrapped])

    unresolved = sorted(
        (
            task_id
            for task_id, record in runtime.state["semantic_records"].items()
            if record.get("status") in {"blind_needs_resolution", "blind_second_complete"}
        ),
        key=lambda task_id: (
            runtime.state["semantic_records"][task_id]["slate_index"],
            runtime.state["semantic_records"][task_id]["source_development_priority"],
            runtime.state["semantic_records"][task_id]["arm"],
        ),
    )
    for task_id in unresolved:
        record = deepcopy(runtime.state["semantic_records"][task_id])
        budget = budgets[(record["slate_index"], record["arm"])]
        if record.get("status") == "blind_second_complete" and len(record.get("blind_results", [])) >= 2:
            results = record["blind_results"]
            second = results[-1]
        else:
            if not budget.reserve(1):
                _record_stage(runtime, task_id, "blind", status="ambiguous", ambiguity_reason="slate escalation budget exhausted")
                continue
            try:
                second_response = _blind_call(runtime, task_id, "blind-second")
                second = _wrap_blind(record, second_response)
                results = record["blind_results"] + [second]
                _record_stage(runtime, task_id, "blind_second", status="blind_second_complete", blind_results=results)
            except Exception as exc:
                _record_stage(runtime, task_id, "blind", status="blind_operational_failure", blind_error=str(exc))
                continue
        if second["serious_different_output_alternative"]:
            # A full ambiguity repair requires repair + technical + blind.  It
            # is attempted only while preserving the one-repair rule and the
            # per-slate five-call ceiling.
            latest = deepcopy(runtime.state["semantic_records"][task_id])
            if latest.get("repair_count", 0) == 0 and budget.reserve(3):
                alternatives = [item["response"]["strongest_plausible_alternatives"] for item in results]
                instruction = (
                    "Adjust or replace one training pair to discriminate the intended rule from "
                    "these concrete different-output alternatives without changing the test or core task: "
                    + json.dumps(alternatives, ensure_ascii=False)
                )
                try:
                    repair = _repair_call(
                        runtime, task_id, repair_type="ambiguity",
                        instruction=instruction, evidence=results,
                    )
                    if repair["status"] == "repaired":
                        okay, reason = _store_repaired_realization(
                            runtime, task_id, repair["repaired_plan"], repair["repaired_realization"],
                            repair_type="ambiguity",
                        )
                        if okay:
                            technical = _technical_call(runtime, task_id, "technical-after-ambiguity-repair")
                            current = deepcopy(runtime.state["semantic_records"][task_id])
                            evaluation = _technical_evaluation(current, technical)
                            blind_response = _blind_call(runtime, task_id, "blind-after-ambiguity-repair") if evaluation["passed"] else None
                            if blind_response is not None:
                                wrapped = _wrap_blind(current, blind_response)
                                if not wrapped["serious_different_output_alternative"]:
                                    _record_stage(
                                        runtime, task_id, "blind", status="blind_pass",
                                        technical_results=current.get("technical_results", []) + [evaluation],
                                        blind_results=results + [wrapped],
                                        ambiguity_reason="one targeted evidence repair removed the alternative",
                                    )
                                    continue
                        else:
                            _record_stage(
                                runtime,
                                task_id,
                                "ambiguity_repair_validation",
                                status="blind_needs_resolution",
                                repair_failure_reason=reason,
                            )
                except Exception as exc:
                    _record_stage(runtime, task_id, "ambiguity_repair", status="escalation_failure", escalation_error=str(exc))
            _record_stage(runtime, task_id, "blind", status="ambiguous", ambiguity_reason="two blind solvers retained serious different-output alternatives")
            continue

        # The second solver did not reproduce the concrete alternative.  Use
        # the unchanged V4 adjudicator only when the per-slate budget allows.
        if not budget.reserve(1):
            _record_stage(runtime, task_id, "blind", status="ambiguous", ambiguity_reason="unresolved blind disagreement")
            continue
        try:
            latest = deepcopy(runtime.state["semantic_records"][task_id])
            adjudication = runtime.call(
                prompt_name="ambiguity_adjudicator",
                call_id=f"{task_id}-ambiguity-adjudication",
                values={
                    "TASK_ID": task_id,
                    "AUTHOR_SPECIFICATION": {
                        "intended_rule": latest["realization"]["authoring_metadata"]["intended_rule"],
                        "concrete_plan": latest["concrete_plan"],
                    },
                    "COMPLETE_TASK": latest["complete_task"],
                    "BLIND_RESULTS": latest["blind_results"],
                },
                parent_ids=[task_id],
                role="ambiguity_adjudicator",
                arm_id=f"private-{latest['arm']}",
                expected={"task_id": task_id},
                forbidden_prompt_fragments=SPECIAL_LABEL_FRAGMENTS,
            )
            if adjudication["decision"] == "clear":
                _record_stage(
                    runtime, task_id, "blind", status="blind_pass",
                    ambiguity_adjudication=adjudication,
                )
                continue
            latest = deepcopy(runtime.state["semantic_records"][task_id])
            if (
                adjudication["decision"] == "repairable"
                and latest.get("repair_count", 0) == 0
                and budget.reserve(3)
            ):
                repair = _repair_call(
                    runtime, task_id, repair_type="ambiguity",
                    instruction=adjudication["repair_instruction"],
                    evidence=latest["blind_results"],
                )
                if repair["status"] == "repaired":
                    okay, reason = _store_repaired_realization(
                        runtime, task_id, repair["repaired_plan"],
                        repair["repaired_realization"], repair_type="ambiguity",
                    )
                    if okay:
                        technical = _technical_call(
                            runtime, task_id, "technical-after-ambiguity-repair"
                        )
                        current = deepcopy(runtime.state["semantic_records"][task_id])
                        evaluation = _technical_evaluation(current, technical)
                        if evaluation["passed"] and technical["confidence"] != "low":
                            blind_response = _blind_call(
                                runtime, task_id, "blind-after-ambiguity-repair"
                            )
                            wrapped = _wrap_blind(current, blind_response)
                            if not wrapped["serious_different_output_alternative"]:
                                _record_stage(
                                    runtime, task_id, "blind", status="blind_pass",
                                    ambiguity_adjudication=adjudication,
                                    technical_results=current.get("technical_results", []) + [evaluation],
                                    blind_results=latest["blind_results"] + [wrapped],
                                    ambiguity_reason="adjudicated local ambiguity was repaired",
                                )
                                continue
                    else:
                        _record_stage(
                            runtime,
                            task_id,
                            "ambiguity_repair_validation",
                            status="blind_second_complete",
                            repair_failure_reason=reason,
                        )
            _record_stage(
                runtime, task_id, "blind", status="ambiguous",
                ambiguity_adjudication=adjudication,
                ambiguity_reason=adjudication["reason"],
            )
        except Exception as exc:
            _record_stage(runtime, task_id, "blind", status="blind_operational_failure", blind_error=str(exc))


def _board_groups(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]
    anchors = _anchors(runtime.config)
    portfolio = _portfolio_digest(runtime.config)
    board_results = []
    by_source: dict[str, dict[str, str]] = {}
    for task_id, record in records.items():
        if record.get("status") in {"blind_pass", "editorial_complete"}:
            by_source.setdefault(record["source_candidate_id"], {})[record["arm"]] = task_id
    matched_sources = sorted(
        [source for source, arms in by_source.items() if set(arms) == {"control", "treatment"}],
        key=lambda source: content_hash({"seed": runtime.config["seed"], "source": source}),
    )
    source_groups = [matched_sources[index:index + 6] for index in range(0, len(matched_sources), 6)]
    for group_index, source_group in enumerate(source_groups, 1):
        for arm in ("control", "treatment"):
            group = [by_source[source][arm] for source in source_group]

            def board(board_index: int) -> dict[str, Any]:
                local_rng = Random(int(runtime.config["seed"]) + group_index * 101 + board_index * 17 + (0 if arm == "control" else 5000))
                used: set[str] = set()
                rows = [
                    (f"anchor:{anchor.task_id}", anchor.rule_description, {"train": list(anchor.train), "test": list(anchor.test)})
                    for anchor in anchors
                ]
                rows.extend(
                    (
                        f"generated:{task_id}",
                        records[task_id]["realization"]["authoring_metadata"]["intended_rule"],
                        records[task_id]["complete_task"],
                    )
                    for task_id in group
                )
                local_rng.shuffle(rows)
                mapping = {}
                package = []
                for source_id, rule, episode in rows:
                    opaque = _anonymous_id(local_rng, used)
                    mapping[opaque] = source_id
                    package.append({"task_id": opaque, "rule_summary": rule, "episode": episode})
                board_id = "board-" + content_hash({"arm_private": arm, "group": group_index, "index": board_index, "ids": sorted(mapping)})[:14]

                def validate(value: dict[str, Any]) -> None:
                    returned = [item["task_id"] for item in value["judgments"]]
                    if len(returned) != len(set(returned)) or set(returned) != set(mapping):
                        raise SupportedDepartureError("board did not judge every opaque item")

                response = runtime.call(
                    prompt_name="editorial_board",
                    call_id=board_id,
                    values={"BLINDED_TASKS": package, "PORTFOLIO_DIGEST": portfolio, "BOARD_ID": board_id},
                    parent_ids=list(group),
                    role="editorial_board",
                    arm_id=f"private-{arm}",
                    expected={"board_id": board_id},
                    validator=validate,
                    forbidden_prompt_fragments=SPECIAL_LABEL_FRAGMENTS,
                )
                return {"arm": arm, "group_index": group_index, "board_index": board_index, "source_group": source_group, "group": group, "mapping": mapping, "response": response}

            results = parallel_map(board, [1, 2], min(2, int(runtime.config["model"]["concurrency"])))
            board_results.extend(results)
            for task_id in group:
                judgments = []
                for result in results:
                    opaque = next(key for key, value in result["mapping"].items() if value == f"generated:{task_id}")
                    judgment = next(item for item in result["response"]["judgments"] if item["task_id"] == opaque)
                    judgments.append({**deepcopy(judgment), "task_id": task_id})
                intrinsic, placement, intrinsic_reason, placement_reason = classify_v3_editorial_results(judgments)
                _record_stage(
                    runtime, task_id, "editorial", status="editorial_complete",
                    editorial_judgments=judgments, intrinsic_quality=intrinsic,
                    portfolio_placement=placement,
                    editorial_reason=intrinsic_reason + "; " + placement_reason,
                )
    atomic_write_json(runtime.run_dir / "editorial_board_results.json", board_results)


def _mechanism_audit(runtime: V41Runtime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = runtime.state["semantic_records"]
    completed = [record for record in records.values() if record.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 8123)
    rng.shuffle(completed)
    all_judgments = []
    calls = []
    for index in range(0, len(completed), 16):
        chunk = completed[index:index + 16]
        audit_id = f"mechanism-audit-{index // 16 + 1:02d}"
        packages = [
            {
                "task_id": record["task_id"],
                "episode": record["complete_task"],
                "concise_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concrete_plan": record["concrete_plan"],
            }
            for record in chunk
        ]

        def validate(value: dict[str, Any]) -> None:
            ids = [item["task_id"] for item in value["judgments"]]
            if len(ids) != len(set(ids)) or set(ids) != {item["task_id"] for item in packages}:
                raise SupportedDepartureError("mechanism auditor did not cover every opaque task")

        response = runtime.call(
            prompt_name="mechanism_auditor",
            call_id=audit_id,
            values={
                "REQUESTED_PROPERTIES": ["supported_departure"],
                "TASK_PACKAGES": packages,
                "AUDIT_ID": audit_id,
            },
            parent_ids=[item["task_id"] for item in packages],
            role="editorial_board",
            arm_id="blinded-mechanism-audit",
            expected={"audit_id": audit_id, "requested_properties": ["supported_departure"]},
            validator=validate,
            forbidden_prompt_fragments=("private-control", "private-treatment", "experimental arm"),
        )
        all_judgments.extend(response["judgments"])
        calls.append(response)
    result = {"arm_hidden": True, "non_gating": True, "calls": calls, "judgments": all_judgments}
    atomic_write_json(runtime.run_dir / "mechanism_audit_report.json", result)
    return all_judgments, result


def _diversity_audit(
    runtime: V41Runtime, judgments: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    records = runtime.state["semantic_records"]
    treatment_ids = {
        task_id for task_id, record in records.items() if record["arm"] == "treatment"
    }
    positive = [
        item for item in judgments
        if item["task_id"] in treatment_ids
        and item["supported_departure"]["status"] == "genuine"
    ]
    if not positive:
        summary = {
            "raw_positive_count": 0,
            "unique_positive_mechanism_instances": 0,
            "material_cluster_count": 0,
            "largest_cluster_fraction": None,
            "same_procedure_tasks_counted_as_separate_successes": False,
        }
        atomic_write_json(runtime.run_dir / "diversity_cluster_report.json", summary)
        return None, summary
    audit_id = "supported-departure-diversity"
    signatures = {
        item["task_id"]: item["open_ended_mechanism_signatures"]["supported_departure"]
        for item in positive
    }
    embedding_report: dict[str, Any]
    nearest: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in signatures}
    try:
        ordered_ids = sorted(signatures)
        embedder = build_embedding_provider(
            runtime.config["embedding"], runtime.config["paths"]["cache"]
        )
        vectors = embedder.embed([signatures[task_id] for task_id in ordered_ids])
        pairs = []
        for left_index, left_id in enumerate(ordered_ids):
            for right_index in range(left_index + 1, len(ordered_ids)):
                right_id = ordered_ids[right_index]
                similarity = sum(
                    left * right
                    for left, right in zip(vectors[left_index], vectors[right_index])
                )
                row = {
                    "task_id_a": left_id,
                    "task_id_b": right_id,
                    "cosine_similarity": round(similarity, 6),
                }
                pairs.append(row)
                nearest[left_id].append({"task_id": right_id, "cosine_similarity": row["cosine_similarity"]})
                nearest[right_id].append({"task_id": left_id, "cosine_similarity": row["cosine_similarity"]})
        for task_id in nearest:
            nearest[task_id] = sorted(
                nearest[task_id], key=lambda item: (-item["cosine_similarity"], item["task_id"])
            )[:3]
        embedding_report = {
            "provider": runtime.config["embedding"]["provider"],
            "model": runtime.config["embedding"]["model"],
            "signature_count": len(ordered_ids),
            "pairwise_similarities": sorted(
                pairs, key=lambda item: -item["cosine_similarity"]
            ),
            "status": "complete",
        }
    except Exception as exc:
        embedding_report = {"status": "operational_failure", "error": str(exc)}
    instances = [
        {
            "task_id": item["task_id"],
            "signature": item["open_ended_mechanism_signatures"]["supported_departure"],
            "concrete_evidence": item["supported_departure"],
            "nearest_signature_neighbors": nearest[item["task_id"]],
        }
        for item in positive
    ]

    def validate(value: dict[str, Any]) -> None:
        members = [
            task_id
            for cluster in value["clusters"]
            for task_id in cluster["member_task_ids"]
        ]
        expected_ids = {item["task_id"] for item in instances}
        if len(members) != len(set(members)) or set(members) != expected_ids:
            raise SupportedDepartureError(
                "diversity audit must cluster every genuine treatment instance exactly once"
            )

    response = runtime.call(
        prompt_name="diversity_auditor",
        call_id=audit_id,
        values={"PROPERTY": "supported_departure", "INSTANCES": instances, "AUDIT_ID": audit_id},
        parent_ids=[item["task_id"] for item in instances],
        role="editorial_board",
        arm_id="blinded-diversity-audit",
        expected={"audit_id": audit_id, "property": "supported_departure"},
        validator=validate,
        forbidden_prompt_fragments=("private-control", "private-treatment", "experimental arm"),
    )
    summary = mechanism_diversity_summary(
        positive_task_ids=[item["task_id"] for item in positive],
        diversity_audit=response,
    )
    value = {
        "audit": response,
        "summary": summary,
        "semantic_embedding_analysis": embedding_report,
    }
    atomic_write_json(runtime.run_dir / "diversity_cluster_report.json", value)
    return response, summary


def _human_pack(runtime: V41Runtime) -> dict[str, Any]:
    records = runtime.state["semantic_records"]
    by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records.values():
        if record.get("complete_task"):
            by_source.setdefault(record["source_candidate_id"], {})[record["arm"]] = record
    pairs = [(source, arms) for source, arms in sorted(by_source.items()) if set(arms) == {"control", "treatment"}]
    destination = runtime.run_dir / "human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(int(runtime.config["seed"]) + 9127)
    used: set[str] = set()
    public, key, rendered, form = [], [], [], []
    for source, arms in pairs:
        pair_id = _anonymous_id(rng, used, length=8)
        order = ("control", "treatment") if rng.randrange(2) else ("treatment", "control")
        row = {"pair_id": pair_id, "variants": []}
        for label, arm in zip(("a", "b"), order):
            anonymous = _anonymous_id(rng, used)
            record = arms[arm]
            atomic_write_json(tasks_dir / f"{anonymous}.json", record["complete_task"])
            image = render_episode(
                record["complete_task"],
                title=f"Pair {pair_id.upper()} — variant {label.upper()}",
                subtitle="Arm, source, and author rule withheld",
                destination=images_dir / f"{anonymous}.png",
            )
            rendered.append((f"{pair_id}-{label}", image))
            row["variants"].append({"label": label, "anonymous_id": anonymous, "task_json": f"tasks/{anonymous}.json", "image": f"images/{anonymous}.png"})
            key.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "anonymous_id": anonymous,
                    "source_candidate_id": source,
                    "arm": arm,
                    "rule": record["realization"]["authoring_metadata"]["intended_rule"],
                    "planning_decision": record["planning_result"]["treatment_decision"],
                }
            )
        public.append(row)
        form.append(
            {
                "pair_id": pair_id,
                "preferred_task": None,
                "clear_conceptual_click_a": None,
                "clear_conceptual_click_b": None,
                "fair_a": None,
                "fair_b": None,
                "genuine_supported_departure_a": None,
                "genuine_supported_departure_b": None,
                "overcomplicated_a": None,
                "overcomplicated_b": None,
                "several_treatment_tasks_feel_templated": None,
                "notes": "",
            }
        )
    contact = build_contact_sheet(rendered, destination / "review_sheet.png")
    atomic_write_json(destination / "manifest.json", {"feature": "supported_departure", "provenance_hidden": True, "rule_withheld": True, "pair_count": len(public), "contact_sheet": "review_sheet.png" if contact else None, "pairs": public})
    atomic_write_json(destination / "review_form.json", {"responses": form})
    atomic_write_json(destination / "DO_NOT_OPEN_FIRST.answer_key.json", {"answers": key})
    atomic_write_json(
        destination / "human_review_status.json",
        {
            "feature": "supported_departure",
            "human_results_available": False,
            "pair_count": len(public),
            "instructions": "Complete review_form.json before opening the answer key.",
        },
    )
    atomic_write_text(
        destination / "review_sheet.md",
        "# Blinded supported-departure review\n\nReview each A/B pair before opening the answer key. Judge preference, conceptual click, fairness, whether a noticeable but rule-preserving departure is genuine, overcomplication, and cross-task template repetition.\n",
    )
    appendix = ["# Rule and assignment appendix", "", "Open only after blind review.", ""]
    for item in key:
        appendix.extend([f"## {item['pair_id']} / {item['label'].upper()}", "", f"Arm: `{item['arm']}`", "", f"Rule: {item['rule']}", ""])
    atomic_write_text(destination / "answer_rule_appendix.md", "\n".join(appendix))
    return {"directory": str(destination), "pair_count": len(public), "human_results_available": False, "contact_sheet": str(contact) if contact else None}


def _arm_metrics(runtime: V41Runtime, arm: str) -> dict[str, Any]:
    records = [record for record in runtime.state["semantic_records"].values() if record["arm"] == arm]
    denominator = len(records)
    deterministic = [record for record in records if record.get("complete_task")]
    technical_attempted = [record for record in deterministic if record.get("technical_results")]
    technical_pass = [record for record in records if record.get("status") in {"technical_pass", "blind_pass", "editorial_complete"} or (record.get("editorial_judgments") and record.get("technical_results", [{}])[-1].get("passed"))]
    hidden_matches = sum(
        bool(record.get("technical_results")) and record["technical_results"][-1]["hidden_output_match"]
        for record in deterministic
    )
    blind = [record for record in records if record.get("blind_results")]
    serious = sum(
        any(item["serious_different_output_alternative"] for item in record["blind_results"])
        and record.get("status") == "ambiguous"
        for record in blind
    )
    core = sum(record.get("intrinsic_quality") == "core_quality" for record in records)
    reserve = sum(record.get("intrinsic_quality") == "reserve_quality" for record in records)
    below = sum(
        judgment["overall_quality"] == "below_real_bar"
        for record in records for judgment in record.get("editorial_judgments", [])
    )
    return {
        "deterministic_validity_rate": len(deterministic) / denominator if denominator else 0.0,
        "technical_pass_rate": len(technical_pass) / denominator if denominator else 0.0,
        "hidden_output_exact_match_rate": hidden_matches / len(technical_attempted) if technical_attempted else 0.0,
        "serious_ambiguity_rate": serious / len(blind) if blind else 0.0,
        "core_quality_yield": core / denominator if denominator else 0.0,
        "core_or_reserve_yield": (core + reserve) / denominator if denominator else 0.0,
        "below_real_bar_count": below,
        "completed_task_count": len(blind),
        "authoring_candidate_count": denominator,
        "realized_count": len(deterministic),
        "technical_pass_count": len(technical_pass),
        "blind_completed_count": len(blind),
        "editorially_reviewed_count": sum(bool(record.get("editorial_judgments")) for record in records),
        "core_count": core,
        "reserve_count": reserve,
        "repair_count": sum(record.get("repair_count", 0) > 0 for record in records),
        "plan_abort_count": sum(record.get("status") == "plan_aborted" for record in records),
        "realization_failure_count": sum(record.get("status") == "realization_failure" for record in records),
    }


def _inflation(runtime: V41Runtime) -> dict[str, Any]:
    records = runtime.state["semantic_records"]
    by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records.values():
        by_source.setdefault(record["source_candidate_id"], {})[record["arm"]] = record
    rows = []
    for source, arms in sorted(by_source.items()):
        if set(arms) != {"control", "treatment"}:
            continue
        row = {"source_candidate_id": source}
        for arm, record in arms.items():
            plan = record.get("concrete_plan")
            row[arm] = {
                "plan_characters": len(json.dumps(plan, ensure_ascii=False)) if plan else None,
                "pair_count": len(record["complete_task"]["train"]) + len(record["complete_task"]["test"]) if record.get("complete_task") else None,
                "grid_metrics": _task_metrics(record["complete_task"]) if record.get("complete_task") else None,
            }
        rows.append(row)
    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None
    summary = {}
    for arm in ("control", "treatment"):
        plans = [row[arm]["plan_characters"] for row in rows if row[arm]["plan_characters"] is not None]
        areas = [row[arm]["grid_metrics"]["maximum_area"] for row in rows if row[arm]["grid_metrics"]]
        pairs = [row[arm]["pair_count"] for row in rows if row[arm]["pair_count"] is not None]
        colors = [row[arm]["grid_metrics"]["color_count"] for row in rows if row[arm]["grid_metrics"]]
        summary[arm] = {"mean_plan_characters": mean(plans), "mean_maximum_grid_area": mean(areas), "mean_pair_count": mean(pairs), "mean_color_count": mean(colors)}
    return {"matched_rows": rows, "summary": summary}


def _selection_accounting(slates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selector_rows = []
    supply_rows = []
    for slate in slates:
        path = Path(slate["state_path"])
        state = read_json(path)
        selector_rows.append(v4_call_accounting(state["call_ledger"], run_dir=path.parent))
        supply_path = slate.get("supply", {}).get("call_ledger_path")
        if supply_path and Path(supply_path).is_file():
            supply_state = read_json(Path(supply_path))
            supply_rows.append(
                v4_call_accounting(
                    supply_state.get("call_ledger", {}), run_dir=Path(supply_path).parent
                )
            )
    rows = [*selector_rows, *supply_rows]
    return {
        "substantive_logical_calls": sum(item["substantive_logical_calls"] for item in rows),
        "active_model_minutes": round(sum(item["active_model_minutes"] for item in rows), 6),
        "transport_retry_calls": sum(item["transport_retry_calls"] for item in rows),
        "selection_slates": selector_rows,
        "reservoir_supply_cycles": supply_rows,
    }


def _render_report(report: Mapping[str, Any]) -> str:
    efficacy = report["efficacy"]
    quality = report["quality_noninferiority"]
    lines = [
        "# ARCGEN V4.1 Experiment C — supported departures",
        "",
        "This is a paired, non-gating experiment. The same selected kernel received ordinary V4 planning and the isolated optional supported-departure instruction. Realization, deterministic checks, technical verification, blind solving, and fixed-anchor boards were unchanged and arm-blind.",
        "",
        "## Sample and efficacy",
        "",
        f"- Independent selector slates: {report['independent_slate_count']}",
        f"- Matched selected kernels: {report['matched_kernel_count']}",
        f"- Completed treatment tasks: {efficacy['treatment_completed_count']}",
        f"- Planner uses / declines / aborts: {efficacy['planner_use_count']} / {efficacy['planner_decline_count']} / {efficacy['planner_abort_count']}",
        f"- Independently audited genuine / cosmetic / absent / unsupported: {efficacy['audit_status_counts']}",
        "",
        "## Quality non-inferiority",
        "",
        f"- Control metrics: `{quality['control']}`",
        f"- Treatment metrics: `{quality['treatment']}`",
        f"- Provisional code/model gates all passed: **{quality['all_provisional_model_gates_passed']}**",
        "",
        "## Diversity",
        "",
        f"- Summary: `{report['diversity']}`",
        "",
        "## Human review and decision",
        "",
        f"- Review pack: `{report['human_review']['directory']}`",
        "- Human results: pending",
        f"- Recommendation: **{report['promotion_recommendation']}**",
        "",
        "No production promotion or good-task-bank mutation occurred. Episode-joint and abstraction-led experiments were not started.",
    ]
    return "\n".join(lines) + "\n"


def run_supported_departure_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config["experimental"]["feature"] != "supported_departure":
        raise SupportedDepartureError("supported-departure runner requires Experiment C config")
    feature_state = config["_feature_manifest_snapshot"]["features"]["supported_departure"]["status"]
    if feature_state not in {"proposed", "testing"}:
        raise SupportedDepartureError("supported departure is not awaiting evaluation")
    if not config["_effective_features"].get("palette_scheduler"):
        raise SupportedDepartureError("the explicitly accepted palette baseline is missing")
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "experiment_start.json"
    if not metadata_path.exists():
        atomic_write_json(metadata_path, {"started_at_epoch_seconds": round(time.time(), 3)})
    protected_before = _protected_hashes(config)
    selected, slates = _freeze_three_selected_samples(config, run_dir)
    aliases = _private_aliases(selected, int(config["seed"]))
    frozen = _source_manifest(config, selected, slates, aliases, protected_before)
    runtime = V41Runtime(config, frozen)
    if runtime.state.get("completed") and (runtime.run_dir / "report.json").exists():
        return read_json(runtime.run_dir / "report.json")
    _initialize_records(runtime, selected, aliases)
    _plan_records(runtime)
    _realize_records(runtime)
    with runtime.lock:
        runtime.state.setdefault(
            "escalation_usage_by_slate",
            {
                f"{index}/{arm}": 0
                for index in range(1, len(slates) + 1)
                for arm in ("control", "treatment")
            },
        )
        runtime.save()

    def persist_budget(index: int, arm: str, used: int) -> None:
        with runtime.lock:
            runtime.state["escalation_usage_by_slate"][f"{index}/{arm}"] = used
            runtime.save()

    budgets = {
        (index, arm): SlateEscalationBudget(
            int(config["budget"]["escalation_call_limit"]),
            used=int(runtime.state["escalation_usage_by_slate"].get(f"{index}/{arm}", 0)),
            on_change=lambda used, index=index, arm=arm: persist_budget(index, arm, used),
        )
        for index in range(1, len(slates) + 1)
        for arm in ("control", "treatment")
    }
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _board_groups(runtime)
    judgments, audit_report = _mechanism_audit(runtime)
    diversity_audit, diversity_summary = _diversity_audit(runtime, judgments)
    human = _human_pack(runtime)
    control_metrics = _arm_metrics(runtime, "control")
    treatment_metrics = _arm_metrics(runtime, "treatment")
    noninferiority = matched_noninferiority(control=control_metrics, treatment=treatment_metrics)
    treatment_ids = {task_id for task_id, record in runtime.state["semantic_records"].items() if record["arm"] == "treatment"}
    control_ids = {task_id for task_id, record in runtime.state["semantic_records"].items() if record["arm"] == "control"}
    treatment_judgments = [item for item in judgments if item["task_id"] in treatment_ids]
    control_judgments = [item for item in judgments if item["task_id"] in control_ids]
    if diversity_audit is not None:
        promotion = semantic_promotion_assessment(
            property_name="supported_departure",
            independent_slate_count=len(slates),
            treatment_authoring_candidate_count=len(treatment_ids),
            matched_control_completed_count=control_metrics["editorially_reviewed_count"],
            judgments=treatment_judgments,
            diversity_audit=diversity_audit,
            noninferiority=noninferiority,
            human_review_complete=False,
            human_control_materially_preferred=None,
            minimums=config["semantic_experiments"],
        )
    else:
        promotion = {
            "property": "supported_departure",
            "requirements": {"genuine_unique_completed": False, "human_review_complete": False},
            "recommendation": "inconclusive",
            "ordinary_task_acceptance_affected": False,
            "automatic_promotion": False,
            "diversity": diversity_summary,
        }
    planning_decisions = Counter(
        record.get("planning_result", {}).get("treatment_decision", "missing")
        for record in runtime.state["semantic_records"].values()
        if record["arm"] == "treatment"
    )
    audit_counts = Counter(
        item["supported_departure"]["status"] for item in treatment_judgments
    )
    genuine_locations = Counter(
        (
            "test"
            if item["supported_departure"]["location"].strip().casefold().startswith("test")
            else "training"
            if item["supported_departure"]["location"].strip().casefold().startswith(("train", "training"))
            else "other_or_unclear"
        )
        for item in treatment_judgments
        if item["supported_departure"]["status"] == "genuine"
    )
    control_audit_counts = Counter(
        item["supported_departure"]["status"] for item in control_judgments
    )
    records = runtime.state["semantic_records"]
    mode_yields = {}
    for arm in ("control", "treatment"):
        mode_yields[arm] = {}
        for mode in ("open", "concept_led", "composition_led"):
            rows = [record for record in records.values() if record["arm"] == arm and record["search_mode"] == mode]
            mode_yields[arm][mode] = {
                "candidate_count": len(rows),
                "core_count": sum(record.get("intrinsic_quality") == "core_quality" for record in rows),
                "core_or_reserve_count": sum(record.get("intrinsic_quality") in {"core_quality", "reserve_quality"} for record in rows),
            }
    background_counts = {
        arm: Counter(
            str(record.get("palette_treatment", {}).get("realized_background_color", "none"))
            for record in records.values()
            if record["arm"] == arm and record.get("complete_task")
        )
        for arm in ("control", "treatment")
    }
    inflation = _inflation(runtime)
    selection_accounting = _selection_accounting(slates)
    runtime_accounting = runtime.accounting()
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    completed_at = round(time.time(), 3)
    started_at = read_json(metadata_path)["started_at_epoch_seconds"]
    judgment_by_id = {item["task_id"]: item for item in judgments}
    cluster_by_id = {}
    if diversity_audit is not None:
        for cluster in diversity_audit["clusters"]:
            for task_id in cluster["member_task_ids"]:
                cluster_by_id[task_id] = cluster["cluster_id"]
    task_reports = []
    for task_id, record in sorted(records.items()):
        judgment = judgment_by_id.get(task_id)
        task_reports.append(
            {
                "task_id": task_id,
                "source_candidate_id": record["source_candidate_id"],
                "source_mode": record["search_mode"],
                "experimental_arm": record["arm"],
                "arm_hidden_from_downstream": True,
                "planner_decision": record.get("planning_result", {}).get("treatment_decision"),
                "plan_outcome": "ready" if (record.get("concrete_plan") or {}).get("status") == "ready" else record.get("status"),
                "realization_outcome": "realized" if record.get("complete_task") else record.get("status"),
                "deterministic_valid": bool(record.get("deterministic_diagnostics", {}).get("hard_valid")),
                "technical_outcome": record.get("technical_results"),
                "blind_outcome": record.get("blind_results"),
                "editorial_judgments": record.get("editorial_judgments"),
                "intrinsic_quality": record.get("intrinsic_quality"),
                "portfolio_placement": record.get("portfolio_placement"),
                "genuine_property_status": judgment["supported_departure"]["status"] if judgment else None,
                "open_ended_mechanism_signature": judgment["open_ended_mechanism_signatures"]["supported_departure"] if judgment else None,
                "cluster_assignment": cluster_by_id.get(task_id),
                "existing_mechanism_overlap": None,
                "grid_metrics": _task_metrics(record["complete_task"]) if record.get("complete_task") else None,
                "pair_count": len(record["complete_task"]["train"]) + len(record["complete_task"]["test"]) if record.get("complete_task") else None,
                "palette_treatment": record.get("palette_treatment"),
                "human_judgment": None,
                "artifact_record": deepcopy(record),
            }
        )
    report = {
        "schema_version": 1,
        "pipeline": config["pipeline"],
        "run_id": runtime.run_id,
        "feature": "supported_departure",
        "completed": True,
        "control_baseline": "frozen V4 plus explicitly accepted palette scheduler",
        "independent_slate_count": len(slates),
        "matched_kernel_count": len(selected),
        "efficacy": {
            "planner_use_count": planning_decisions["use"],
            "planner_decline_count": planning_decisions["decline"],
            "planner_abort_count": planning_decisions["abort"],
            "treatment_completed_count": treatment_metrics["realized_count"],
            "audit_status_counts": dict(sorted(audit_counts.items())),
            "genuine_completed_count": audit_counts["genuine"],
            "genuine_location_counts": dict(sorted(genuine_locations.items())),
        },
        "diversity": diversity_summary,
        "quality_noninferiority": noninferiority,
        "coverage_retention": {
            "ordinary_source_share": {"control": 1.0, "treatment": 1.0},
            "search_mode_counts": dict(sorted(Counter(item["kernel"]["search_mode"] for item in selected).items())),
            "ordinary_mode_quality_yields": mode_yields,
            "background_color_counts": {
                arm: dict(sorted(counts.items())) for arm, counts in background_counts.items()
            },
            "palette_active_in_both_arms": True,
            "supported_departure_share_control": control_audit_counts["genuine"] / control_metrics["realized_count"] if control_metrics["realized_count"] else 0.0,
            "supported_departure_share_among_completed_treatment": audit_counts["genuine"] / treatment_metrics["realized_count"] if treatment_metrics["realized_count"] else 0.0,
            "episode_joint_not_intentionally_treated": True,
            "abstraction_led_not_intentionally_treated": True,
        },
        "inflation_and_economy": inflation,
        "human_review": human,
        "promotion_assessment": promotion,
        "promotion_recommendation": "inconclusive" if not human["human_results_available"] else promotion["recommendation"],
        "production_promotion_occurred": False,
        "feature_manifest_status_after_run": config["_feature_manifest_snapshot"]["features"]["supported_departure"]["status"],
        "good_task_bank_mutated": False,
        "protected_production_state_unchanged": protected_before == protected_after,
        "call_accounting": {
            "selection": selection_accounting,
            "paired_authoring_and_review": runtime_accounting,
            "active_model_minutes_total": round(selection_accounting["active_model_minutes"] + runtime_accounting["active_model_minutes"], 6),
            "substantive_logical_calls_total": selection_accounting["substantive_logical_calls"] + runtime_accounting["experimental_calls"],
            "wall_minutes": round((completed_at - started_at) / 60, 6),
            "escalation_calls_by_slate_and_arm": {
                f"{index}/{arm}": budget.used
                for (index, arm), budget in budgets.items()
            },
        },
        "tasks": task_reports,
        "failure_mode_analysis": {
            "control_statuses": dict(sorted(Counter(record["status"] for record in runtime.state["semantic_records"].values() if record["arm"] == "control").items())),
            "treatment_statuses": dict(sorted(Counter(record["status"] for record in runtime.state["semantic_records"].values() if record["arm"] == "treatment").items())),
            "unsupported_departures": [item["task_id"] for item in treatment_judgments if item["supported_departure"]["status"] == "unsupported"],
            "cosmetic_departures": [item["task_id"] for item in treatment_judgments if item["supported_departure"]["status"] == "cosmetic"],
            "planner_label_leakage_is_a_hard_experimental_failure": True,
        },
        "unresolved_uncertainties": [
            "Blinded human review is pending.",
            "Three slates remain a small sample for a creativity intervention.",
            "No production feature status changes until explicit user authorization.",
        ],
        "next_experiment_started": False,
    }
    atomic_write_json(runtime.run_dir / "quality_noninferiority_report.json", noninferiority)
    atomic_write_json(runtime.run_dir / "coverage_retention_report.json", report["coverage_retention"])
    atomic_write_json(runtime.run_dir / "promotion_recommendation.json", {"feature": "supported_departure", "recommendation": report["promotion_recommendation"], "production_promotion_occurred": False})
    return runtime.finish(report, _render_report(report))


__all__ = ["SupportedDepartureError", "run_supported_departure_experiment"]
