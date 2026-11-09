"""Isolated randomized V4.4 planner experiment for functionally larger canvases.

The ordinary V4/V4.4 reservoir, quality selection, concrete planning, realization,
technical verification, blind solving, and editorial policy are preserved.  Only
half of already-selected kernels receive one optional planner addendum.  Each
kernel is authored once; this is a randomized cohort comparison, not paired
counterfactual task generation.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import json
import math
import time

from jsonschema import Draft202012Validator

from .lean_v3_rendering import build_contact_sheet, render_episode
from .schemas import SchemaError, parse_json_object
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
)
from .v4_diagnostics import grid_canvas_metrics
from .v4_pipeline import V4ControlPipeline, V4ReservoirProducer, v4_call_accounting
from .v4_reservoir import load_reservoir, reservoir_status
from .v4_schemas import STRICT_OUTPUT_SCHEMAS as V4_SCHEMAS
from .v41_experiments import V41ExperimentError, V41Runtime, _anonymous_id
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_supported_departure import (
    SlateEscalationBudget,
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
    _selection_sandbox_config,
)
from .v41_test_challenge import _board as _single_instance_boards
from .v44_abstraction_led import _install_extensions as _install_v44_extensions
from .v44_abstraction_led import _normalize_palette_claims
from .v44_scale_config import V44_SCALE_PIPELINE, verify_v44_baseline


class FunctionalScaleError(V41ExperimentError):
    pass


TEXT = {"type": "string", "minLength": 1}
SCALE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": TEXT,
        "scale_decision": {
            "type": "string",
            "enum": ["expanded", "kept_compact", "abort"],
        },
        "scale_reason": TEXT,
        "concrete_plan": {
            "anyOf": [deepcopy(V4_SCHEMAS["v4_concrete_plan"]), {"type": "null"}]
        },
    },
    "required": ["candidate_id", "scale_decision", "scale_reason", "concrete_plan"],
    "additionalProperties": False,
}

PRIVATE_LABELS = (
    "functional-scale",
    "functional scale experiment",
    "scale treatment",
    "treatment arm",
    "control arm",
    "experimental assignment",
)


def _transport(value: Any) -> Any:
    unsupported = {
        "minLength", "maxLength", "minimum", "maximum", "minItems",
        "maxItems", "uniqueItems",
    }
    if isinstance(value, dict):
        return {key: _transport(item) for key, item in value.items() if key not in unsupported}
    if isinstance(value, list):
        return [_transport(item) for item in value]
    return value


def parse_scale_plan(data: str | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else dict(data)
    errors = sorted(
        Draft202012Validator(SCALE_PLAN_SCHEMA).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)
    decision = value["scale_decision"]
    plan = value["concrete_plan"]
    if decision == "abort" and plan is not None:
        raise SchemaError("$.concrete_plan", "abort must not return a plan")
    if decision != "abort" and plan is None:
        raise SchemaError("$.concrete_plan", "expanded or kept_compact requires a plan")
    return value


def _install_scale_extension(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    filename = "v4_4_functional_scale_planning_addendum.md"
    text, version, digest = load_prompt(root / "prompts" / filename)
    runtime.prompts.entries["functional_scale_addendum"] = {
        "filename": filename,
        "text": text,
        "version": version,
        "hash": digest,
        "stage": "v44_functional_scale_plan",
    }
    destination = runtime.run_dir / "prompts" / "functional_scale"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FunctionalScaleError("scale prompt changed during resume")
    if not target.exists():
        atomic_write_text(target, text)
    active_config = Path(runtime.config["_v4_4_scale_config_path"]).resolve()
    try:
        active_config_relative = str(active_config.relative_to(root))
    except ValueError as exc:
        raise FunctionalScaleError("scale config must be inside the repository") from exc
    files = (
        "arcgen.lib/v44_scale_config.py",
        "arcgen.lib/v44_functional_scale.py",
        active_config_relative,
        f"prompts/{filename}",
    )
    manifest = {
        "schema_version": 1,
        "pipeline": V44_SCALE_PIPELINE,
        "prompt": {"filename": filename, "version": version, "sha256": digest},
        "implementation": {
            relative: sha256((root / relative).read_bytes()).hexdigest()
            for relative in files
        },
    }
    manifest_path = runtime.run_dir / "functional_scale_extension_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise FunctionalScaleError("scale implementation changed during resume")
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage == "v44_functional_scale_plan":
            return parse_scale_plan(data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas["v44_functional_scale_plan"] = _transport(SCALE_PLAN_SCHEMA)
    runtime.invoker._write_schemas()


def _eligible_count(reservoir: Mapping[str, Any], accepted: set[str]) -> int:
    reserved = {item["concept_hash"] for item in reservoir.get("active_reservations", [])}
    return sum(
        item.get("state") == "proposed"
        and item.get("concept_hash") not in accepted
        and item.get("concept_hash") not in reserved
        for item in reservoir.get("proposals", [])
    )


def _ensure_window(
    config: Mapping[str, Any],
    v4: dict[str, Any],
    run_dir: Path,
    *,
    slate_index: int,
    accepted_hashes: set[str],
) -> dict[str, Any]:
    required = int(config["functional_scale"]["eligible_window_size"])
    results: list[dict[str, Any]] = []
    state_paths: list[str] = []
    for attempt in range(1, 7):
        reservoir = load_reservoir(v4["paths"]["reservoir"])
        eligible = _eligible_count(reservoir, accepted_hashes)
        if eligible >= required:
            break
        supply_id = (
            f"{config['run']['id']}-scale-slate-{slate_index:02d}-topup-{attempt:02d}"
        )
        producer = V4ReservoirProducer(v4, supply_run_id=supply_id)
        results.append(producer.run(cycles=1, force_between_watermarks=True))
        state = run_dir / "selection_graph" / "v4_reservoir_supply" / supply_id / "state.json"
        if state.is_file():
            state_paths.append(str(state))
    else:
        eligible = _eligible_count(load_reservoir(v4["paths"]["reservoir"]), accepted_hashes)
    if eligible < required:
        raise FunctionalScaleError(
            f"slate {slate_index} retained only {eligible} eligible proposals; {required} required"
        )
    return {
        "generated_topup_cycles": len(results),
        "topup_results": results,
        "call_ledger_paths": state_paths,
        "retained_eligible_count": eligible,
        "reservoir": reservoir_status(load_reservoir(v4["paths"]["reservoir"]), v4["reservoir"]),
    }


def _kernel_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "candidate_id", "search_mode", "central_relation", "why_coherent",
        "novel_concept_potential", "functional_composition_potential",
        "why_multiple_training_examples_may_be_useful",
        "possible_test_generalization", "main_authoring_risk",
    )
    return {key: deepcopy(record[key]) for key in fields}


def _freeze_selected_slates(
    config: Mapping[str, Any], run_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = run_dir / "selected_kernel_manifest.private.json"
    if path.exists():
        value = read_json(path)
        return value["development_sample"], value["slates"]
    selected: list[dict[str, Any]] = []
    slates: list[dict[str, Any]] = []
    settings = config["functional_scale"]
    offset = int(settings.get("slate_index_offset", 0))
    for slate_index in range(offset + 1, offset + int(settings["independent_slates"]) + 1):
        v4 = _selection_sandbox_config(config, run_dir, slate_index=slate_index)
        v4["selection"]["eligible_window_size"] = int(settings["eligible_window_size"])
        selector = V4ControlPipeline(v4)
        supply = _ensure_window(
            config,
            v4,
            run_dir,
            slate_index=slate_index,
            accepted_hashes=selector.context.accepted_concept_hashes(),
        )
        briefs = selector.select_slate()
        ordered = sorted(briefs, key=lambda item: int(item["development_priority"]))
        maximum = int(settings["maximum_development_per_slate"])
        sample: list[dict[str, Any]] = []
        for brief in ordered[:maximum]:
            candidate_id = str(brief["candidate_id"])
            record = selector.state["records"][candidate_id]
            sample.append(
                {
                    "source_candidate_id": candidate_id,
                    "slate_index": slate_index,
                    "source_development_priority": int(record["development_priority"]),
                    "kernel": _kernel_record(record),
                    "editorial_brief": deepcopy(record["editorial_brief"]),
                }
            )
        selected.extend(sample)
        slates.append(
            {
                "slate_index": slate_index,
                "selection_run_id": v4["run"]["id"],
                "frozen_window": deepcopy(selector.state["frozen_window"]),
                "full_selected_slate": [item["candidate_id"] for item in ordered],
                "development_sample": [item["source_candidate_id"] for item in sample],
                "supply": supply,
                "state_path": str(selector.state_path),
            }
        )
    value = {
        "schema_version": 1,
        "selection_precedes_assignment": True,
        "selector_received_scale_information": False,
        "independent_slates": len(slates),
        "development_sample": selected,
        "slates": slates,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        run_dir / "selected_kernel_manifest.public.json",
        {
            "schema_version": 1,
            "selection_precedes_assignment": True,
            "selector_received_scale_information": False,
            "development_count": len(selected),
            "slates": [
                {
                    "slate_index": item["slate_index"],
                    "frozen_window_proposal_ids": item["frozen_window"]["proposal_ids"],
                    "development_sample_count": len(item["development_sample"]),
                }
                for item in slates
            ],
        },
    )
    return selected, slates


def _assignment_objective(
    rows: Sequence[Mapping[str, Any]], treatment_ids: set[str]
) -> tuple[int, int]:
    mode_totals = Counter(str(item["kernel"]["search_mode"]) for item in rows)
    mode_treatment = Counter(
        str(item["kernel"]["search_mode"])
        for item in rows
        if str(item["source_candidate_id"]) in treatment_ids
    )
    mode_imbalance = sum(
        abs(2 * mode_treatment[mode] - total) for mode, total in mode_totals.items()
    )
    priority_pairs = Counter(
        (int(item["source_development_priority"]) - 1) // 2
        for item in rows
        if str(item["source_candidate_id"]) in treatment_ids
    )
    priority_imbalance = sum(abs(priority_pairs[pair] - 1) for pair in range(4))
    return mode_imbalance, priority_imbalance


def assign_scale_arms(
    selected: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, str]:
    """Assign treatment after selection, balancing source mode and priority."""

    assignments: dict[str, str] = {}
    by_slate: dict[int, list[Mapping[str, Any]]] = {}
    for row in selected:
        by_slate.setdefault(int(row["slate_index"]), []).append(row)
    for slate_index, rows in sorted(by_slate.items()):
        target = len(rows) // 2
        ids = [str(item["source_candidate_id"]) for item in rows]
        rng = Random(seed + slate_index * 1009)
        tie = {candidate_id: rng.random() for candidate_id in ids}
        choices = []
        for picked in combinations(ids, target):
            picked_set = set(picked)
            choices.append(
                (
                    _assignment_objective(rows, picked_set),
                    sum(tie[item] for item in picked),
                    tuple(sorted(picked)),
                    picked_set,
                )
            )
        treatment = min(choices, key=lambda item: (item[0], item[1], item[2]))[3]
        for candidate_id in ids:
            assignments[candidate_id] = "scale_prompt" if candidate_id in treatment else "ordinary_control"
    return assignments


def _aliases(selected: Sequence[Mapping[str, Any]], seed: int) -> dict[str, str]:
    rng = Random(seed + 77419)
    used: set[str] = set()
    ids = [str(item["source_candidate_id"]) for item in selected]
    rng.shuffle(ids)
    return {candidate_id: _anonymous_id(rng, used) for candidate_id in ids}


def _freeze_assignment(
    config: Mapping[str, Any],
    run_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
    protected: Mapping[str, Any],
) -> dict[str, Any]:
    private = run_dir / "functional_scale_assignment.private.json"
    if private.exists():
        return read_json(private)
    assignments = assign_scale_arms(selected, seed=int(config["seed"]))
    rows = [
        {
            "source_candidate_id": str(item["source_candidate_id"]),
            "opaque_task_id": aliases[str(item["source_candidate_id"])],
            "slate_index": int(item["slate_index"]),
            "development_priority": int(item["source_development_priority"]),
            "search_mode": item["kernel"]["search_mode"],
            "private_arm": assignments[str(item["source_candidate_id"])],
        }
        for item in selected
    ]
    value = {
        "schema_version": 1,
        "pipeline": V44_SCALE_PIPELINE,
        "assignment_seed": int(config["seed"]),
        "selection_completed_before_assignment": True,
        "stratified_by_search_mode_and_priority": True,
        "one_task_instance_per_kernel": True,
        "downstream_arm_labels_hidden": True,
        "protected_production_state": deepcopy(dict(protected)),
        "frozen_v4_4_baseline": deepcopy(config["_v4_4_baseline_verification"]),
        "assignments": rows,
    }
    atomic_write_json(private, value)
    atomic_write_json(
        run_dir / "functional_scale_assignment.public.json",
        {
            "schema_version": 1,
            "assignment_seed_hash": content_hash(int(config["seed"])),
            "selection_completed_before_assignment": True,
            "stratification": "search mode and selector-priority bands",
            "counts": dict(sorted(Counter(row["private_arm"] for row in rows).items())),
            "arm_labels_hidden_downstream": True,
        },
    )
    return value


def _initialize_records(
    runtime: V41Runtime,
    selected: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        private = {
            str(item["source_candidate_id"]): item for item in assignment["assignments"]
        }
        records = {}
        for row in selected:
            source_id = str(row["source_candidate_id"])
            assigned = private[source_id]
            task_id = str(assigned["opaque_task_id"])
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source_id,
                "arm": "single",
                "scale_arm": assigned["private_arm"],
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(row["source_development_priority"]),
                "search_mode": row["kernel"]["search_mode"],
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["functional_scale_pipeline"] = V44_SCALE_PIPELINE
        runtime.save()


def _private_leaks(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False).casefold()
    return [label for label in PRIVATE_LABELS if label in text]


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        brief = deepcopy(record["editorial_brief"])
        # This experiment intentionally stacks no accepted optional planning
        # suggestion. The scale assignment is the sole planner variable.
        brief["optional_development_suggestion"] = None
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
            "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
            "CANDIDATE_ID": task_id,
        }
        try:
            if record["scale_arm"] == "ordinary_control":
                plan = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan",),
                    call_id=f"{task_id}-concrete-plan",
                    values=values,
                    parent_ids=[task_id],
                    role="concrete_planner",
                    arm_id="single-authoring",
                    expected={"candidate_id": task_id},
                    forbidden_prompt_fragments=PRIVATE_LABELS,
                )
                result = {
                    "candidate_id": task_id,
                    "scale_decision": "not_invited",
                    "scale_reason": "ordinary frozen V4.4 planning",
                    "concrete_plan": plan,
                }
            else:
                result = runtime.call(
                    prompt_name="concrete_planner",
                    composed_with=("v44_quality_plan", "functional_scale_addendum"),
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
                    raise FunctionalScaleError("nested plan changed candidate ID")
                plan = _normalize_palette_claims(plan)
                result["concrete_plan"] = plan
                leaks = _private_leaks(plan)
                if leaks:
                    raise FunctionalScaleError(f"downstream plan leaked private labels: {leaks}")
            if plan is not None and plan.get("status") == "abort" and result["scale_decision"] != "abort":
                raise FunctionalScaleError("aborted plan must use abort scale decision")
            return task_id, result, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [task_id for task_id, record in records.items() if "planning_result" not in record]
    results = parallel_map(
        one,
        pending,
        min(len(pending), int(runtime.config["model"]["concurrency"])),
    ) if pending else []
    for task_id, result, error in results:
        if error or result is None:
            _record_stage(runtime, task_id, "planning", status="operational_failure", planning_error=error)
            continue
        plan = result["concrete_plan"]
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


def _budgets(runtime: V41Runtime) -> dict[tuple[int, str], SlateEscalationBudget]:
    limit = int(runtime.config["functional_scale"]["escalation_calls_per_slate"])
    slate_indices = sorted(
        {
            int(record["slate_index"])
            for record in runtime.state.get("semantic_records", {}).values()
        }
    )
    with runtime.lock:
        usage = runtime.state.setdefault("functional_scale_escalation_usage", {})
        for slate_index in slate_indices:
            usage.setdefault(str(slate_index), 0)
        runtime.save()
    budgets: dict[tuple[int, str], SlateEscalationBudget] = {}
    for slate_index in slate_indices:
        key = str(slate_index)

        def persist(used: int, *, usage_key: str = key) -> None:
            with runtime.lock:
                runtime.state["functional_scale_escalation_usage"][usage_key] = used
                runtime.save()

        budgets[(slate_index, "single")] = SlateEscalationBudget(
            limit,
            used=int(runtime.state["functional_scale_escalation_usage"][key]),
            on_change=persist,
        )
    return budgets


def _all_grids(task: Mapping[str, Any]) -> list[list[list[int]]]:
    return [
        pair[field]
        for split in ("train", "test")
        for pair in task[split]
        for field in ("input", "output")
        if field in pair
    ]


def task_scale_metrics(task: Mapping[str, Any]) -> dict[str, Any]:
    grids = _all_grids(task)
    metrics = [grid_canvas_metrics(grid) for grid in grids]
    colors = {cell for grid in grids for row in grid for cell in row}
    maximum_dimension = max(item["maximum_dimension"] for item in metrics)
    return {
        "grid_count": len(grids),
        "maximum_dimension": maximum_dimension,
        "maximum_area": max(item["area"] for item in metrics),
        "total_episode_area": sum(item["area"] for item in metrics),
        "mean_grid_area": round(sum(item["area"] for item in metrics) / len(metrics), 6),
        "mean_active_bounding_box_area": round(
            sum(item["active_bounding_box_area"] for item in metrics) / len(metrics), 6
        ),
        "mean_active_bounding_box_fraction": round(
            sum(item["active_bounding_box_fraction"] for item in metrics) / len(metrics), 6
        ),
        "mean_non_background_occupancy": round(
            sum(item["non_background_occupancy"] for item in metrics) / len(metrics), 6
        ),
        "mean_rows_with_relevant_content_fraction": round(
            sum(item["rows_with_relevant_content_fraction"] for item in metrics) / len(metrics), 6
        ),
        "mean_columns_with_relevant_content_fraction": round(
            sum(item["columns_with_relevant_content_fraction"] for item in metrics) / len(metrics), 6
        ),
        "mean_uniform_margin_fraction": round(
            1 - sum(item["active_bounding_box_fraction"] for item in metrics) / len(metrics), 6
        ),
        "above_dimension_15": maximum_dimension > 15,
        "above_dimension_20": maximum_dimension > 20,
        "above_dimension_25": maximum_dimension > 25,
        "color_count": len(colors),
        "training_pair_count": len(task["train"]),
        "test_pair_count": len(task["test"]),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _arm_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [task_scale_metrics(item["complete_task"]) for item in records if item.get("complete_task")]
    selected = len(records)
    realized = len(metrics)
    technical = sum(
        bool(item.get("technical_results"))
        and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind = sum(item.get("status") in {"blind_pass", "editorial_complete"} for item in records)
    core = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    reserve = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    ambiguous = sum(item.get("status") == "ambiguous" or item.get("intrinsic_quality") == "ambiguous" for item in records)
    boards = [judgment for item in records for judgment in item.get("editorial_judgments", [])]
    return {
        "selected": selected,
        "plans_ready": sum(item.get("concrete_plan", {}).get("status") == "ready" for item in records),
        "realized": realized,
        "deterministic_valid": sum(bool(item.get("deterministic_diagnostics", {}).get("hard_valid")) for item in records),
        "technical_pass": technical,
        "blind_clear": blind,
        "core": core,
        "reserve": reserve,
        "ambiguous": ambiguous,
        "other": selected - core - reserve - ambiguous,
        "core_yield": core / selected if selected else None,
        "core_or_reserve_yield": (core + reserve) / selected if selected else None,
        "technical_pass_rate": technical / realized if realized else None,
        "serious_ambiguity_rate": ambiguous / selected if selected else None,
        "board_overall_quality": dict(sorted(Counter(item["overall_quality"] for item in boards).items())),
        "board_novel_concept": dict(sorted(Counter(item["novel_concept"] for item in boards).items())),
        "board_functional_composition": dict(sorted(Counter(item["functional_composition"] for item in boards).items())),
        "scale": {
            "maximum_dimension_mean": _mean(metrics, "maximum_dimension"),
            "maximum_area_mean": _mean(metrics, "maximum_area"),
            "total_episode_area_mean": _mean(metrics, "total_episode_area"),
            "active_bounding_box_area_mean": _mean(metrics, "mean_active_bounding_box_area"),
            "active_bounding_box_fraction_mean": _mean(metrics, "mean_active_bounding_box_fraction"),
            "non_background_occupancy_mean": _mean(metrics, "mean_non_background_occupancy"),
            "rows_used_fraction_mean": _mean(metrics, "mean_rows_with_relevant_content_fraction"),
            "columns_used_fraction_mean": _mean(metrics, "mean_columns_with_relevant_content_fraction"),
            "above_dimension_15_count": sum(item["above_dimension_15"] for item in metrics),
            "above_dimension_20_count": sum(item["above_dimension_20"] for item in metrics),
            "above_dimension_25_count": sum(item["above_dimension_25"] for item in metrics),
            "pair_count_mean": _mean(metrics, "training_pair_count"),
            "color_count_mean": _mean(metrics, "color_count"),
        },
    }


def _genuinely_expanded(record: Mapping[str, Any]) -> bool:
    if record.get("planning_result", {}).get("scale_decision") != "expanded":
        return False
    if not record.get("complete_task") or not record.get("technical_results"):
        return False
    realization = record.get("realization", {}).get("authoring_metadata", {})
    technical = record["technical_results"][-1]
    response = technical.get("response", {})
    return (
        realization.get("canvas_use_classification") == "large_and_functionally_used"
        and technical.get("passed") is True
        and response.get("canvas_use") == "appropriate"
    )


def _committed_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    registry_path = Path(config["paths"]["committed_task_registry"])
    registry = read_json(registry_path)
    metrics = []
    for entry in registry.get("entries", []):
        path = registry_path.parent / entry["task_path"]
        if path.is_file():
            metrics.append(task_scale_metrics(read_json(path)))
    dimensions = [item["maximum_dimension"] for item in metrics]
    return {
        "task_count": len(metrics),
        "maximum_dimension_mean": round(sum(dimensions) / len(dimensions), 6) if dimensions else None,
        "above_dimension_15_share": sum(value > 15 for value in dimensions) / len(dimensions) if dimensions else None,
        "above_dimension_20_share": sum(value > 20 for value in dimensions) / len(dimensions) if dimensions else None,
        "above_dimension_25_share": sum(value > 25 for value in dimensions) / len(dimensions) if dimensions else None,
    }


def _human_pack(runtime: V41Runtime) -> dict[str, Any]:
    destination = runtime.run_dir / "human_review"
    tasks_dir = destination / "tasks"
    images_dir = destination / "images"
    dossiers = destination / "dossiers_private"
    for directory in (tasks_dir, images_dir, dossiers):
        directory.mkdir(parents=True, exist_ok=True)
    records = [item for item in runtime.state["semantic_records"].values() if item.get("complete_task")]
    rng = Random(int(runtime.config["seed"]) + 55177)
    rng.shuffle(records)
    used: set[str] = set()
    public, key, rendered, rendered_rules, form = [], [], [], [], []
    for index, record in enumerate(records, 1):
        anonymous = _anonymous_id(rng, used)
        atomic_write_json(tasks_dir / f"{anonymous}.json", record["complete_task"])
        image = render_episode(
            record["complete_task"],
            title=f"Task {index}",
            subtitle="Source, assignment, and rule withheld",
            destination=images_dir / f"{anonymous}.png",
        )
        rule_image = render_episode(
            record["complete_task"],
            title=f"Task {index}",
            subtitle=record["realization"]["authoring_metadata"]["intended_rule"],
            destination=images_dir / f"{anonymous}_with_rule.png",
        )
        rendered.append((f"Task {index}", image))
        rendered_rules.append((f"Task {index}", rule_image))
        public.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_json": f"tasks/{anonymous}.json",
                "image": f"images/{anonymous}.png",
            }
        )
        key.append(
            {
                "review_index": index,
                "anonymous_id": anonymous,
                "task_id": record["task_id"],
                "private_arm": record["scale_arm"],
                "scale_decision": record.get("planning_result", {}).get("scale_decision"),
                "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
            }
        )
        form.append(
            {
                "review_index": index,
                "conceptual_interest_1_to_5": None,
                "functional_composition_1_to_5": None,
                "visual_legibility_1_to_5": None,
                "canvas_scale_is_functionally_justified": None,
                "task_feels_artificially_inflated": None,
                "overall_preference_notes": "",
            }
        )
        atomic_write_json(
            dossiers / f"{record['task_id']}.json",
            {
                "task_id": record["task_id"],
                "private_arm": record["scale_arm"],
                "planning_result": record.get("planning_result"),
                "task": record["complete_task"],
                "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "technical_results": record.get("technical_results"),
                "blind_results": record.get("blind_results"),
                "editorial_judgments": record.get("editorial_judgments"),
            },
        )
    contact = build_contact_sheet(rendered, destination / "contact_sheet_blind.png", columns=2) if rendered else None
    with_rules = build_contact_sheet(rendered_rules, destination / "contact_sheet_with_rules.png", columns=2) if rendered_rules else None
    atomic_write_json(destination / "review_manifest.json", {"arm_hidden": True, "tasks": public})
    atomic_write_json(destination / "answer_key.private.json", {"tasks": key})
    atomic_write_json(destination / "review_form.json", {"tasks": form})
    return {
        "task_count": len(public),
        "arm_hidden": True,
        "contact_sheet": str(contact) if contact else None,
        "contact_sheet_with_rules": str(with_rules) if with_rules else None,
        "review_manifest": str(destination / "review_manifest.json"),
        "review_form": str(destination / "review_form.json"),
        "answer_key_private": str(destination / "answer_key.private.json"),
    }


def _call_accounting(runtime: V41Runtime, slates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    downstream = runtime.accounting()
    selector_rows = []
    supply_rows = []
    seen_supply: set[str] = set()
    for slate in slates:
        path = Path(slate["state_path"])
        state = read_json(path)
        selector_rows.append(v4_call_accounting(state.get("call_ledger", {}), run_dir=path.parent))
        for raw in slate.get("supply", {}).get("call_ledger_paths", []):
            if raw in seen_supply:
                continue
            seen_supply.add(raw)
            supply_path = Path(raw)
            if supply_path.is_file():
                supply = read_json(supply_path)
                supply_rows.append(v4_call_accounting(supply.get("call_ledger", {}), run_dir=supply_path.parent))
    selector_calls = sum(item["substantive_logical_calls"] for item in selector_rows)
    supply_calls = sum(item["substantive_logical_calls"] for item in supply_rows)
    downstream_calls = downstream["experimental_calls"]
    active_selector = sum(item["active_model_minutes"] for item in selector_rows)
    active_supply = sum(item["active_model_minutes"] for item in supply_rows)
    return {
        "selector_calls": selector_calls,
        "reservoir_supply_calls": supply_calls,
        "downstream_calls": downstream_calls,
        "extra_scale_calls": 0,
        "standing_audit_calls": 0,
        "total_calls_excluding_off_path_supply": selector_calls + downstream_calls,
        "total_calls_including_supply": selector_calls + supply_calls + downstream_calls,
        "active_model_minutes_excluding_supply": round(active_selector + downstream["active_model_minutes"], 6),
        "active_model_minutes_including_supply": round(active_selector + active_supply + downstream["active_model_minutes"], 6),
        "transport_retry_calls": sum(item["transport_retry_calls"] for item in selector_rows + supply_rows) + downstream["transport_retry_calls"],
        "runtime": downstream,
        "selectors": selector_rows,
        "supply": supply_rows,
    }


def _ratio(treatment: float | None, control: float | None) -> float | None:
    if treatment is None or control is None or control == 0:
        return None
    return round(treatment / control, 6)


def _render_report(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    arms = report["quality_by_arm"]
    efficacy = report["efficacy"]
    lines = [
        "# ARCGEN V4.4 functional-scale planner experiment",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['recommendation']}**",
        "",
        "This randomized, single-instance study changed one thing only after quality selection: half of the selected kernels received an optional instruction to use a substantially larger canvas when added space remained functionally consequential. It did not create paired variants, add a judge, change acceptance, or promote tasks.",
        "",
        "## Funnel",
        "",
        f"- Independent frozen slates / developed kernels: {funnel['slates']} / {funnel['developed']}",
        f"- Ordinary control / optional-scale assignments: {funnel['ordinary_control']} / {funnel['scale_prompt']}",
        f"- Planner expanded / kept compact / aborted: {funnel['expanded']} / {funnel['kept_compact']} / {funnel['aborted']}",
        f"- Realized / technical pass / blind clear: {funnel['realized']} / {funnel['technical_pass']} / {funnel['blind_clear']}",
        f"- Core / reserve / ambiguous / other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        "",
        "## Scale efficacy",
        "",
        f"- Functionally expanded completions confirmed by planner, realizer, and technical verifier: {efficacy['functionally_expanded_count']}",
        f"- Treatment planner expansion rate: {efficacy['planner_expansion_rate']}",
        f"- Treatment/control maximum-dimension means: {arms['scale_prompt']['scale']['maximum_dimension_mean']} / {arms['ordinary_control']['scale']['maximum_dimension_mean']}",
        f"- Treatment/control maximum-area means: {arms['scale_prompt']['scale']['maximum_area_mean']} / {arms['ordinary_control']['scale']['maximum_area_mean']}",
        f"- Treatment/control >15 dimension counts: {arms['scale_prompt']['scale']['above_dimension_15_count']} / {arms['ordinary_control']['scale']['above_dimension_15_count']}",
        f"- Treatment/control >20 dimension counts: {arms['scale_prompt']['scale']['above_dimension_20_count']} / {arms['ordinary_control']['scale']['above_dimension_20_count']}",
        f"- Treatment/control >25 dimension counts: {arms['scale_prompt']['scale']['above_dimension_25_count']} / {arms['ordinary_control']['scale']['above_dimension_25_count']}",
        "",
        "## Quality preservation",
        "",
        f"- Control summary: `{arms['ordinary_control']}`",
        f"- Treatment summary: `{arms['scale_prompt']}`",
        f"- Core-yield ratio (treatment/control): `{report['noninferiority']['core_yield_ratio']}` (floor {report['noninferiority']['core_yield_ratio_floor']})",
        f"- Core-or-reserve ratio: `{report['noninferiority']['core_or_reserve_yield_ratio']}` (floor {report['noninferiority']['core_or_reserve_yield_ratio_floor']})",
        f"- Serious-ambiguity increase: `{report['noninferiority']['serious_ambiguity_rate_difference']}`",
        "",
        "Grid size is descriptive, never an acceptance gate. An expanded claim counts as functionally confirmed only when the realizer classified the canvas as large and functionally used and the technical verifier independently called it appropriate.",
        "",
        "## Calls and timing",
        "",
        f"- Extra scale-specific model calls: **{report['call_accounting']['extra_scale_calls']}**",
        f"- Selector / downstream / off-path supply calls: {report['call_accounting']['selector_calls']} / {report['call_accounting']['downstream_calls']} / {report['call_accounting']['reservoir_supply_calls']}",
        f"- Active model-minutes excluding / including supply: {report['call_accounting']['active_model_minutes_excluding_supply']} / {report['call_accounting']['active_model_minutes_including_supply']}",
        f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
        "",
        "## Tasks",
        "",
        "| Task | Mode | Private arm | Decision | Max dim | Canvas verdict | Technical | Blind | Intrinsic |",
        "|---|---|---|---|---:|---|---|---|---|",
    ]
    for item in report["tasks"]:
        lines.append(
            f"| `{item['task_id']}` | `{item['search_mode']}` | `{item['private_arm']}` | "
            f"`{item['scale_decision']}` | {item.get('scale_metrics', {}).get('maximum_dimension', 'n/a')} | "
            f"`{item.get('canvas_use_classification')}` | `{item['technical_pass']}` | "
            f"`{item['blind_clear']}` | `{item.get('intrinsic_quality') or item['terminal_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Human review and decision",
            "",
            f"- Arm-blind contact sheet: `{report['human_review']['contact_sheet']}`",
            f"- Inspection sheet with rules: `{report['human_review']['contact_sheet_with_rules']}`",
            "- Human review is pending; therefore the experiment cannot be promoted.",
            "",
            f"Recommendation: **{report['recommendation']}**. {report['recommendation_reason']}",
            "",
            "No V4.4 production file, task bank, reservoir, feature status, or task disposition was changed. No task was committed and no treatment was promoted automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def run_functional_scale_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("v4_4_scale_pipeline") != V44_SCALE_PIPELINE:
        raise FunctionalScaleError("the isolated V4.4 functional-scale configuration is required")
    settings = config["functional_scale"]
    if settings.get("automatic_promotion") is not False or settings.get("automatic_task_commit") is not False:
        raise FunctionalScaleError("the scale experiment cannot promote or commit automatically")
    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    start = run_dir / "experiment_start.json"
    if not start.exists():
        atomic_write_json(start, {"started_at_epoch_seconds": round(time.time(), 3)})
    protected_before = _protected_hashes(config)
    selected, slates = _freeze_selected_slates(config, run_dir)
    aliases = _aliases(selected, int(config["seed"]))
    assignment = _freeze_assignment(config, run_dir, selected, aliases, protected_before)
    source_manifest = {
        "schema_version": 1,
        "pipeline": V44_SCALE_PIPELINE,
        "selected_manifest_sha256": sha256((run_dir / "selected_kernel_manifest.private.json").read_bytes()).hexdigest(),
        "assignment_manifest_sha256": sha256((run_dir / "functional_scale_assignment.private.json").read_bytes()).hexdigest(),
        "frozen_v4_4_baseline": deepcopy(config["_v4_4_baseline_verification"]),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "selection_precedes_assignment": True,
        "one_instance_per_kernel": True,
        "downstream_arm_labels_hidden": True,
        "automatic_promotion": False,
        "automatic_task_commit": False,
        "continuation": deepcopy(config.get("_v4_4_scale_continuation")),
    }
    runtime = V41Runtime(config, source_manifest)
    _install_v44_extensions(runtime)
    _install_scale_extension(runtime)
    if runtime.state.get("completed") and (run_dir / "report.json").is_file():
        return read_json(run_dir / "report.json")
    _initialize_records(runtime, selected, assignment)
    _plan_records(runtime)
    _realize_records(runtime)
    budgets = _budgets(runtime)
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _single_instance_boards(runtime)
    human = _human_pack(runtime)

    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: (int(item["slate_index"]), int(item["source_development_priority"])),
    )
    groups = {
        arm: [item for item in records if item["scale_arm"] == arm]
        for arm in ("ordinary_control", "scale_prompt")
    }
    summaries = {arm: _arm_summary(rows) for arm, rows in groups.items()}
    decisions = Counter(item.get("planning_result", {}).get("scale_decision", "failed") for item in records)
    statuses = Counter(item.get("status", "unknown") for item in records)
    core = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    reserve = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    ambiguous = sum(item.get("status") == "ambiguous" or item.get("intrinsic_quality") == "ambiguous" for item in records)
    realized = sum(bool(item.get("complete_task")) for item in records)
    technical = sum(
        bool(item.get("technical_results")) and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind = sum(item.get("status") in {"blind_pass", "editorial_complete"} for item in records)
    expanded = sum(_genuinely_expanded(item) for item in records)
    accounting = _call_accounting(runtime, slates)
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    baseline_after = verify_v44_baseline(
        config["_repository_root"], config["_v4_4_baseline_verification"]["manifest_path"]
    )
    if baseline_after != config["_v4_4_baseline_verification"]:
        raise FunctionalScaleError("the frozen V4.4 baseline changed during the experiment")

    control, treatment = summaries["ordinary_control"], summaries["scale_prompt"]
    core_ratio = _ratio(treatment["core_yield"], control["core_yield"])
    useful_ratio = _ratio(treatment["core_or_reserve_yield"], control["core_or_reserve_yield"])
    ambiguity_delta = (
        round(float(treatment["serious_ambiguity_rate"]) - float(control["serious_ambiguity_rate"]), 6)
        if treatment["serious_ambiguity_rate"] is not None and control["serious_ambiguity_rate"] is not None
        else None
    )
    completed = round(time.time(), 3)
    started = read_json(start)["started_at_epoch_seconds"]
    treatment_count = len(groups["scale_prompt"])
    recommendation = "continue_testing"
    reason = (
        "The randomized model cohort is complete, but promotion requires the pending blinded human review and explicit user authorization."
    )
    report = {
        "schema_version": 1,
        "pipeline": V44_SCALE_PIPELINE,
        "run_id": runtime.run_id,
        "completed": True,
        "experimental_design": "post_selection_randomized_single_instance_planner_overlay",
        "continuation": deepcopy(config.get("_v4_4_scale_continuation")),
        "frozen_v4_4_baseline": deepcopy(config["_v4_4_baseline_verification"]),
        "funnel": {
            "slates": len(slates),
            "eligible_proposals_total": sum(len(item["frozen_window"]["proposal_ids"]) for item in slates),
            "developed": len(records),
            "ordinary_control": len(groups["ordinary_control"]),
            "scale_prompt": treatment_count,
            "expanded": decisions["expanded"],
            "kept_compact": decisions["kept_compact"],
            "aborted": decisions["abort"] + statuses["plan_aborted"],
            "realized": realized,
            "technical_pass": technical,
            "blind_clear": blind,
            "core": core,
            "reserve": reserve,
            "ambiguous": ambiguous,
            "other": len(records) - core - reserve - ambiguous,
        },
        "efficacy": {
            "functionally_expanded_count": expanded,
            "adoption_minimum_functionally_expanded": settings["adoption_minimum_functionally_expanded"],
            "planner_expansion_rate": decisions["expanded"] / treatment_count if treatment_count else None,
            "planner_decline_rate": decisions["kept_compact"] / treatment_count if treatment_count else None,
            "measurement_uses_existing_realizer_and_technical_canvas_assessments": True,
            "extra_property_judge_calls": 0,
        },
        "quality_by_arm": summaries,
        "noninferiority": {
            "core_yield_ratio": core_ratio,
            "core_yield_ratio_floor": settings["core_yield_ratio_floor"],
            "core_or_reserve_yield_ratio": useful_ratio,
            "core_or_reserve_yield_ratio_floor": settings["core_or_reserve_yield_ratio_floor"],
            "serious_ambiguity_rate_difference": ambiguity_delta,
            "human_review_complete": False,
            "provisional_gate_pass": False,
            "reason": "human review is a required gate; small randomized cohorts remain descriptive",
        },
        "committed_bank_scale_reference": _committed_reference(config),
        "search_mode_retention": {
            arm: dict(sorted(Counter(item["search_mode"] for item in rows).items()))
            for arm, rows in groups.items()
        },
        "human_review": human,
        "call_accounting": accounting,
        "timing": {
            "started_at_epoch_seconds": started,
            "completed_at_epoch_seconds": completed,
            "wall_clock_minutes": round((completed - started) / 60, 3),
        },
        "tasks": [
            {
                "task_id": item["task_id"],
                "source_candidate_id": item["source_candidate_id"],
                "slate_index": item["slate_index"],
                "development_priority": item["source_development_priority"],
                "search_mode": item["search_mode"],
                "private_arm": item["scale_arm"],
                "scale_decision": item.get("planning_result", {}).get("scale_decision"),
                "scale_reason": item.get("planning_result", {}).get("scale_reason"),
                "functionally_expanded": _genuinely_expanded(item),
                "terminal_status": item.get("status"),
                "technical_pass": bool(item.get("technical_results")) and item["technical_results"][-1].get("passed") is True,
                "blind_clear": item.get("status") in {"blind_pass", "editorial_complete"},
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "canvas_use_classification": item.get("realization", {}).get("authoring_metadata", {}).get("canvas_use_classification"),
                "scale_metrics": task_scale_metrics(item["complete_task"]) if item.get("complete_task") else None,
                "technical_results": item.get("technical_results"),
                "blind_results": item.get("blind_results"),
                "editorial_judgments": item.get("editorial_judgments"),
                "task_json": str(run_dir / "tasks" / "single" / f"{item['task_id']}.arc.json") if item.get("complete_task") else None,
                "plan_json": str(run_dir / "plans" / "single" / f"{item['task_id']}.json"),
            }
            for item in records
        ],
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "automatic_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "production_state_unchanged": protected_before == protected_after,
        "unresolved_uncertainties": [
            "Blinded human review is pending.",
            "Randomized single-instance arms estimate distributional effects but do not supply task-level counterfactuals.",
            (
                "This is the single allowed continuation slate; no further automatic continuation is authorized."
                if config.get("_v4_4_scale_continuation")
                else "A fourth independent slate is available as a later continuation only if either arm misses the completion target."
            ),
        ],
    }
    atomic_write_json(
        run_dir / "promotion_recommendation.json",
        {
            "feature": "functional_scale_planner_overlay",
            "recommendation": recommendation,
            "production_promotion_occurred": False,
            "user_authorization_required": True,
        },
    )
    return runtime.finish(report, _render_report(report))


def rebuild_functional_scale_report(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    report = read_json(directory / "report.json")
    if report.get("pipeline") != V44_SCALE_PIPELINE:
        raise FunctionalScaleError("run is not a functional-scale experiment")
    atomic_write_text(directory / "report.md", _render_report(report))
    return report


def functional_scale_human_review_status(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    report = read_json(directory / "report.json")
    return deepcopy(report["human_review"])


__all__ = [
    "FunctionalScaleError",
    "SCALE_PLAN_SCHEMA",
    "assign_scale_arms",
    "functional_scale_human_review_status",
    "parse_scale_plan",
    "rebuild_functional_scale_report",
    "run_functional_scale_experiment",
    "task_scale_metrics",
]
