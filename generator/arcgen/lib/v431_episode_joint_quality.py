"""Frozen-kernel V4.3.1 episode-joint authoring-quality pilot.

This runner intentionally performs no idea generation.  It globally reselects
the 24 checksummed V4.3 treatment kernels, strengthens only their concrete
architecture gate, and then reuses the unchanged V4.2/V4.3 realization and
review stack.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence
import json
import time

from jsonschema import Draft202012Validator

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
from .v41_experiments import V41Runtime, _task_metrics
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_supported_departure import (
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
)
from .v41_test_challenge import _board as _single_instance_boards
from .v4_pipeline import V4ControlPipeline
from .v43_config import verify_v42_baseline
from .v43_episode_joint import (
    EDITORIAL_BRIEF,
    KERNEL_ASSESSMENT,
    PRIVATE_LABELS,
    EpisodeJointError,
    _arm_quality,
    _budgets,
    _diversity_audit,
    _human_pack,
    _install_extensions as _install_v43_extensions,
    _mechanism_audit,
    _reference_packages,
)
from .v431_config import (
    V431_DESIGN,
    V431_PIPELINE,
    verify_v43_source,
)


class EpisodeJointQualityError(EpisodeJointError):
    pass


GLOBAL_ASSESSMENT = deepcopy(KERNEL_ASSESSMENT)
GLOBAL_ASSESSMENT["properties"]["same_minimal_procedure_with"]["maxItems"] = 23
GLOBAL_SELECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "selector_id": {"type": "string", "minLength": 1},
        "assessments": {
            "type": "array",
            "items": GLOBAL_ASSESSMENT,
            "minItems": 1,
            "maxItems": 24,
        },
        "quality_order": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 24,
        },
        "portfolio_observation": {"type": "string", "minLength": 1},
    },
    "required": [
        "selector_id",
        "assessments",
        "quality_order",
        "portfolio_observation",
    ],
    "additionalProperties": False,
}


def _transport(value: Any) -> Any:
    unsupported = {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
    if isinstance(value, dict):
        return {
            key: _transport(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_transport(item) for item in value]
    return value


def parse_v431_global_selector(
    data: str | bytes | bytearray | Mapping[str, Any],
) -> dict[str, Any]:
    value = (
        parse_json_object(data)
        if isinstance(data, (str, bytes, bytearray))
        else dict(data)
    )
    errors = sorted(
        Draft202012Validator(GLOBAL_SELECTOR_SCHEMA).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = "$" + "".join(
            f"[{item}]" if isinstance(item, int) else f".{item}"
            for item in error.absolute_path
        )
        raise SchemaError(path, error.message)
    assessed = [item["candidate_id"] for item in value["assessments"]]
    ordered = value["quality_order"]
    if len(assessed) != len(set(assessed)) or len(ordered) != len(set(ordered)):
        raise SchemaError("$", "selector candidate IDs must be unique")
    if set(assessed) != set(ordered):
        raise SchemaError("$.quality_order", "quality order must cover every assessment")
    assessed_set = set(assessed)
    for index, item in enumerate(value["assessments"]):
        siblings = set(item["same_minimal_procedure_with"])
        if not siblings <= assessed_set - {item["candidate_id"]}:
            raise SchemaError(
                f"$.assessments[{index}].same_minimal_procedure_with",
                "siblings must identify other candidates in the global set",
            )
        develop = item["recommendation"] == "develop"
        if (item["development_brief"] is not None) != develop:
            raise SchemaError(
                f"$.assessments[{index}].development_brief",
                "a brief is required exactly for develop recommendations",
            )
        if develop and (
            item["novel_concept_potential"] == "weak"
            or item["functional_composition_potential"] == "thin"
            or item["fair_episode_potential"] != "plausible"
        ):
            raise SchemaError(
                f"$.assessments[{index}]",
                "develop recommendations must meet both quality floors and be plausible",
            )
    return value


def _install_extensions(runtime: V41Runtime) -> None:
    _install_v43_extensions(runtime)
    root = Path(runtime.config["_repository_root"])
    specs = {
        "v431_global_selector": (
            "v4_3_1_global_quality_selector.md",
            "v431_global_selector",
        ),
        "v431_quality_plan": (
            "v4_3_1_episode_joint_quality_planning_addendum.md",
            "v43_episode_joint_plan",
        ),
    }
    destination = runtime.run_dir / "prompts" / "v431_extensions"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": V431_PIPELINE,
        "prompts": {},
        "implementation": {},
    }
    for name, (filename, stage) in specs.items():
        text, version, digest = load_prompt(root / "prompts" / filename)
        runtime.prompts.entries[name] = {
            "filename": filename,
            "text": text,
            "version": version,
            "hash": digest,
            "stage": stage,
        }
        target = destination / filename
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise EpisodeJointQualityError(f"V4.3.1 prompt changed during resume: {filename}")
        if not target.exists():
            atomic_write_text(target, text)
        manifest["prompts"][name] = {
            "filename": filename,
            "version": version,
            "sha256": digest,
            "stage": stage,
            "placeholder_names": sorted(PLACEHOLDER_RE.findall(text)),
        }
    for relative in (
        "arcgen.lib/v431_config.py",
        "arcgen.lib/v431_episode_joint_quality.py",
        "config/v4_3_1_exp_episode_joint_quality.yaml",
        "V4_3_1_AUTHORING_QUALITY_PILOT.md",
    ):
        path = root / relative
        manifest["implementation"][relative] = sha256(path.read_bytes()).hexdigest()
    path = runtime.run_dir / "v431_extension_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise EpisodeJointQualityError("V4.3.1 implementation or prompt changed during resume")
    if not path.exists():
        atomic_write_json(path, manifest)

    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage == "v431_global_selector":
            return parse_v431_global_selector(data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas["v431_global_selector"] = _transport(
        GLOBAL_SELECTOR_SCHEMA
    )
    runtime.invoker._write_schemas()


def _freeze_parent_kernels(
    config: Mapping[str, Any], run_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    private_path = run_dir / "frozen_parent_kernels.private.json"
    manifest_path = run_dir / "frozen_parent_manifest.json"
    if private_path.exists() and manifest_path.exists():
        return read_json(private_path)["kernels"], read_json(manifest_path)

    verification = verify_v43_source(config)
    source = Path(verification["source_run"])
    source_supply = read_json(source / "kernel_supply_manifest.private.json")
    source_report = read_json(source / "report.json")
    if source_report.get("pipeline") != "arcgen-v4.3-exp-episode-joint":
        raise EpisodeJointQualityError("frozen source is not a completed V4.3 run")
    arm = config["authoring_quality"]["frozen_source_arm"]
    kernels = [
        deepcopy(item)
        for item in source_supply["kernels"]
        if item.get("private_arm") == arm
    ]
    expected = int(config["authoring_quality"]["expected_kernel_count"])
    if len(kernels) != expected or len({item["candidate_id"] for item in kernels}) != expected:
        raise EpisodeJointQualityError("frozen parent kernel supply is incomplete or duplicated")
    manifest = {
        "schema_version": 1,
        "pipeline": V431_PIPELINE,
        "design": V431_DESIGN,
        "source_verification": verification,
        "source_pipeline": source_report["pipeline"],
        "source_run_id": source_report["run_id"],
        "source_arm_private": arm,
        "kernel_count": len(kernels),
        "candidate_ids": sorted(item["candidate_id"] for item in kernels),
        "no_new_idea_calls": True,
        "global_selection_input_frozen": True,
        "automatic_promotion": False,
        "automatic_task_commit": False,
    }
    atomic_write_json(private_path, {"schema_version": 1, "kernels": kernels})
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        run_dir / "frozen_parent_kernels.blinded.json",
        {
            "schema_version": 1,
            "source_labels_removed": True,
            "kernels": [
                {"candidate_id": item["candidate_id"], "kernel": item["kernel"]}
                for item in kernels
            ],
        },
    )
    return kernels, manifest


def choose_global_development(
    response: Mapping[str, Any],
    kernels: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose globally ranked developable kernels while collapsing siblings."""

    by_id = {str(item["candidate_id"]): item for item in kernels}
    assessments = {
        str(item["candidate_id"]): item for item in response["assessments"]
    }
    chosen_ids: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for candidate_id in response["quality_order"]:
        assessment = assessments[candidate_id]
        if assessment["recommendation"] != "develop":
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "reason": assessment["recommendation"],
                    "same_procedure_with_chosen": [],
                }
            )
            continue
        conflicts = [
            other
            for other in chosen_ids
            if other in assessment["same_minimal_procedure_with"]
            or candidate_id in assessments[other]["same_minimal_procedure_with"]
        ]
        if conflicts:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "same_minimal_procedure_sibling",
                    "same_procedure_with_chosen": conflicts,
                }
            )
            continue
        if len(chosen_ids) >= maximum:
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "below_global_development_cap",
                    "same_procedure_with_chosen": [],
                }
            )
            continue
        chosen_ids.append(candidate_id)
    chosen = []
    for priority, candidate_id in enumerate(chosen_ids, 1):
        source = deepcopy(dict(by_id[candidate_id]))
        source["parent_slate_index"] = int(source["slate_index"])
        source["selector_global_priority"] = priority
        source["source_development_priority"] = priority
        source["editorial_brief"] = deepcopy(
            assessments[candidate_id]["development_brief"]
        )
        source["selector_assessment"] = deepcopy(assessments[candidate_id])
        chosen.append(source)
    return chosen, exclusions


def _select_kernels(
    runtime: V41Runtime, kernels: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = runtime.run_dir / "global_selection_manifest.private.json"
    if path.exists():
        value = read_json(path)
        return value["development_sample"], value
    rows = [deepcopy(dict(item)) for item in kernels]
    Random(int(runtime.config["seed"]) + 431019).shuffle(rows)
    packages, accepted, reservations, rejected = _reference_packages(
        runtime.config, rows
    )
    selector_id = "v431-global-quality-selector"
    expected_ids = {item["candidate_id"] for item in rows}

    def validate(value: dict[str, Any]) -> None:
        returned = {item["candidate_id"] for item in value["assessments"]}
        if returned != expected_ids or set(value["quality_order"]) != expected_ids:
            raise EpisodeJointQualityError("global selector did not cover all 24 kernels")

    response = runtime.call(
        prompt_name="v431_global_selector",
        call_id=selector_id,
        values={
            "CANDIDATES_WITH_REFERENCES": packages,
            "ACCEPTED_BANK_DIGEST": accepted,
            "IN_FLIGHT_RESERVATIONS": reservations,
            "REJECTED_MEMORY_DIGEST": rejected,
            "SELECTOR_ID": selector_id,
        },
        parent_ids=sorted(expected_ids),
        role="batch_selector",
        arm_id="global-quality-selection",
        expected={"selector_id": selector_id},
        validator=validate,
        forbidden_prompt_fragments=PRIVATE_LABELS,
    )
    selected, exclusions = choose_global_development(
        response,
        rows,
        maximum=int(runtime.config["authoring_quality"]["maximum_global_selection"]),
    )
    if not selected:
        raise EpisodeJointQualityError("global quality selection found no developable kernel")
    value = {
        "schema_version": 1,
        "source_labels_visible_to_selector": False,
        "candidate_count": len(rows),
        "development_count": len(selected),
        "development_sample": selected,
        "excluded": exclusions,
        "response": response,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "global_selection_manifest.blinded.json",
        {
            **{key: deepcopy(item) for key, item in value.items() if key != "development_sample"},
            "development_sample": [
                {
                    key: deepcopy(item)
                    for key, item in row.items()
                    if key not in {"private_arm", "generation_call_id", "calibration_task_ids"}
                }
                for row in selected
            ],
        },
    )
    return selected, value


def _initialize_records(runtime: V41Runtime, selected: Sequence[Mapping[str, Any]]) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        records = {}
        for source in selected:
            task_id = str(source["candidate_id"])
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": task_id,
                "arm": "treatment",
                "slate_index": int(source["parent_slate_index"]),
                "parent_slate_index": int(source["parent_slate_index"]),
                "source_development_priority": int(source["source_development_priority"]),
                "selector_global_priority": int(source["selector_global_priority"]),
                "search_mode": "concept_led",
                "kernel": deepcopy(source["kernel"]),
                "editorial_brief": deepcopy(source["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["v431_pipeline"] = V431_PIPELINE
        runtime.state["v431_design"] = V431_DESIGN
        runtime.state["no_new_idea_calls"] = True
        runtime.save()


def _plan_records(runtime: V41Runtime) -> None:
    records = runtime.state["semantic_records"]

    def one(task_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        record = deepcopy(records[task_id])
        brief = deepcopy(record["editorial_brief"])
        brief["optional_development_suggestion"] = None
        values = {
            "CANDIDATE": {**deepcopy(record["kernel"]), "candidate_id": task_id},
            "EDITORIAL_BRIEF": {**brief, "candidate_id": task_id},
            "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
            "CANDIDATE_ID": task_id,
        }
        try:
            response = runtime.call(
                prompt_name="concrete_planner",
                composed_with=("v43_episode_joint_plan", "v431_quality_plan"),
                call_id=f"{task_id}-v431-quality-plan",
                values=values,
                parent_ids=[task_id],
                role="concrete_planner",
                arm_id="episode-joint-quality-authoring",
                expected={"candidate_id": task_id},
            )
            nested = response.get("concrete_plan")
            if nested is not None and nested.get("candidate_id") != task_id:
                raise EpisodeJointQualityError("planner changed the nested candidate ID")
            if (
                nested is not None
                and nested.get("status") == "abort"
                and response["property_status"] != "abort"
            ):
                raise EpisodeJointQualityError("an aborted plan must use abort property status")
            leaked = [
                label
                for label in PRIVATE_LABELS
                if label in json.dumps(nested, ensure_ascii=False).casefold()
            ] if nested is not None else []
            if leaked:
                raise EpisodeJointQualityError(f"plan leaked private labels: {leaked}")
            return task_id, response, None
        except Exception as exc:
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, record in records.items() if "planning_result" not in record
    ]
    results = parallel_map(
        one,
        pending,
        min(len(pending), int(runtime.config["model"]["concurrency"])),
    ) if pending else []
    for task_id, result, error in results:
        if error or result is None:
            _record_stage(
                runtime,
                task_id,
                "planning",
                status="planning_operational_failure",
                planning_error=error,
            )
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
        directory = runtime.run_dir / "plans" / "treatment"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / f"{task_id}.json", result)


def apply_realization_cap(runtime: V41Runtime) -> dict[str, Any]:
    path = runtime.run_dir / "realization_selection_manifest.json"
    if path.exists():
        return read_json(path)
    ready = sorted(
        (
            record
            for record in runtime.state["semantic_records"].values()
            if record.get("status") == "plan_ready"
            and record.get("concrete_plan", {}).get("status") == "ready"
        ),
        key=lambda item: (int(item["selector_global_priority"]), item["task_id"]),
    )
    maximum = int(runtime.config["authoring_quality"]["maximum_realizations"])
    realized_ids = [item["task_id"] for item in ready[:maximum]]
    reserve_ids = [item["task_id"] for item in ready[maximum:]]
    for task_id in reserve_ids:
        _record_stage(
            runtime,
            task_id,
            "realization_selection",
            status="authored_reserve",
            authored_reserve_reason=(
                "ready plan retained beyond the fixed six-realization pilot cap"
            ),
        )
    value = {
        "schema_version": 1,
        "maximum_realizations": maximum,
        "ready_plan_count": len(ready),
        "realization_candidate_ids": realized_ids,
        "authored_reserve_candidate_ids": reserve_ids,
        "ranking_source": "frozen global selector quality order",
        "no_post_plan_model_ranking_call": True,
    }
    atomic_write_json(path, value)
    return value


def _progress(
    runtime: V41Runtime,
    *,
    stage: str,
    kernels: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    value = {
        "pipeline": V431_PIPELINE,
        "run_id": runtime.run_id,
        "completed": False,
        "current_safe_stage": stage,
        "frozen_kernel_count": len(kernels),
        "selected_development_count": len(selected),
        "new_idea_calls": 0,
        "call_accounting": runtime.accounting(),
        "production_promotion_occurred": False,
        "good_task_bank_mutated": False,
        "resume_command": (
            "python3 -m arcgen.lib v4-3-1-experiment --config "
            "config/v4_3_1_exp_episode_joint_quality.yaml --through-phase complete"
        ),
    }
    atomic_write_json(runtime.run_dir / "progress.json", value)
    return value


def _mean_metric(tasks: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [item["grid_metrics"][key] for item in tasks if item.get("grid_metrics")]
    return round(sum(values) / len(values), 6) if values else None


def _render_report(report: Mapping[str, Any]) -> str:
    funnel = report["funnel"]
    comparison = report["comparison_with_v43_treatment"]
    lines = [
        "# ARCGEN V4.3.1 — episode-joint authoring-quality pilot",
        "",
        f"Run: `{report['run_id']}`  ",
        f"Recommendation: **{report['recommendation']}**",
        "",
        "This pilot generated no new ideas. It globally reselected the exact 24 checksummed V4.3 treatment kernels, strengthened only the concrete architecture gate, and reused the existing realization, technical, blind, and fixed-anchor editorial stack unchanged.",
        "",
        "## Funnel",
        "",
        f"- Frozen kernels / globally selected: {funnel['frozen_kernels']} / {funnel['selected']}",
        f"- Plans ready / aborted or failed / authored reserves: {funnel['plans_ready']} / {funnel['plans_not_ready']} / {funnel['authored_reserve']}",
        f"- Realized / technical pass / blind clear: {funnel['realized']} / {funnel['technical_pass']} / {funnel['blind_clear']}",
        f"- Core / reserve / ambiguous / rejected-or-other: {funnel['core']} / {funnel['reserve']} / {funnel['ambiguous']} / {funnel['other']}",
        "",
        "## What changed",
        "",
        "- One selector compared all 24 kernels, so sibling detection crossed the three original slate boundaries.",
        "- The planner had to find a perceptually legible and consequential embodiment of the joint inference, while retaining the right to drop it or abort.",
        "- No new judge, score, generator instruction, realization freedom, or acceptance relaxation was introduced.",
        "",
        "## Comparison with frozen V4.3 treatment",
        "",
        f"- Prior treatment: {comparison['parent_selected']} developed, {comparison['parent_core']} core, {comparison['parent_reserve']} reserve, core yield {comparison['parent_core_yield']:.3f}.",
        f"- V4.3.1: {comparison['pilot_realized']} realized, {comparison['pilot_core']} core, {comparison['pilot_reserve']} reserve, core yield {comparison['pilot_core_yield']:.3f}.",
        f"- Mean maximum dimension, prior / pilot: {comparison['parent_mean_maximum_dimension']} / {comparison['pilot_mean_maximum_dimension']}.",
        f"- Genuine episode-joint completions: {report['efficacy']['genuine_count']} of {funnel['realized']}.",
        f"- Diversity summary: `{report['diversity']['summary']}`",
        "",
        "## Calls and review",
        "",
        f"- New idea calls: 0",
        f"- Experimental logical calls: {report['call_accounting']['experimental_calls']}",
        f"- Active model-minutes: {report['call_accounting']['active_model_minutes']}",
        f"- Wall-clock minutes: {report['timing']['wall_clock_minutes']}",
        f"- Split-board human-review candidates: {report['human_review_routing']['split_board_candidate_count']}",
        f"- Blinded review sheet: `{report['human_review']['review_sheet']}`",
        "",
        "## Task outcomes",
        "",
        "| Task | Prior | Joint audit | Technical | Blind | Intrinsic | Portfolio | Max dim |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for task in report["tasks"]:
        audit = task.get("mechanism_audit") or {}
        lines.append(
            f"| `{task['task_id']}` | `{task['parent_outcome']}` | `{audit.get('status', 'not_audited')}` | "
            f"`{task['technical_pass']}` | `{task['blind_clear']}` | `{task.get('intrinsic_quality') or task['terminal_status']}` | "
            f"`{task.get('portfolio_placement') or 'not_applicable'}` | {task['grid_metrics']['maximum_dimension'] if task.get('grid_metrics') else ''} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["recommendation_reason"],
            "",
            "No feature was promoted and no task was committed. The V4.2 milestone and completed V4.3 run were verified before and after this pilot.",
            "",
        ]
    )
    return "\n".join(lines)


def run_episode_joint_quality_pilot(
    config: dict[str, Any], *, through_phase: str = "complete"
) -> dict[str, Any]:
    if through_phase not in {"selection", "plans", "complete"}:
        raise EpisodeJointQualityError("through_phase must be selection, plans, or complete")
    if config.get("v4_3_1_pipeline") != V431_PIPELINE or config.get("v4_3_1_design") != V431_DESIGN:
        raise EpisodeJointQualityError("the isolated V4.3.1 config is required")
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise EpisodeJointQualityError("experimental tasks cannot be committed automatically")

    run_dir = Path(config["paths"]["runs"]) / config["run"]["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    start_path = run_dir / "experiment_start.json"
    if not start_path.exists():
        atomic_write_json(start_path, {"started_at_epoch_seconds": round(time.time(), 3)})
    protected_before = _protected_hashes(config)
    kernels, parent_manifest = _freeze_parent_kernels(config, run_dir)
    design = {
        "schema_version": 1,
        "pipeline": V431_PIPELINE,
        "design": V431_DESIGN,
        "parent_manifest": parent_manifest,
        "v42_parent": deepcopy(config["_v4_2_baseline_verification"]),
        "model_and_effort_settings": deepcopy(config["model"]["roles"]),
        "intervention": ["global_cross_slate_selection", "architecture_quality_gate"],
        "unchanged_downstream": True,
        "protected_production_state": protected_before,
    }
    runtime = V41Runtime(config, design)
    _install_extensions(runtime)
    if runtime.state.get("completed") and (run_dir / "report.json").is_file():
        return read_json(run_dir / "report.json")

    selected, selection = _select_kernels(runtime, kernels)
    _initialize_records(runtime, selected)
    if through_phase == "selection":
        return _progress(runtime, stage="global_selection_frozen", kernels=kernels, selected=selected)

    _plan_records(runtime)
    realization_selection = apply_realization_cap(runtime)
    if through_phase == "plans":
        return _progress(runtime, stage="plans_and_realization_cap_frozen", kernels=kernels, selected=selected)

    _realize_records(runtime)
    budgets = _budgets(runtime)
    _run_technical(runtime, budgets)
    _run_blind(runtime, budgets)
    _single_instance_boards(runtime)
    audit = _mechanism_audit(runtime)
    diversity = _diversity_audit(runtime, audit)
    human = _human_pack(runtime, audit)

    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda item: int(item["selector_global_priority"]),
    )
    realized_records = [item for item in records if item.get("complete_task")]
    quality = _arm_quality(realized_records)
    audit_by_id = {item["task_id"]: item for item in audit}
    status_counts = Counter(item.get("status", "unknown") for item in records)
    core = sum(item.get("intrinsic_quality") == "core_quality" for item in records)
    reserve = sum(item.get("intrinsic_quality") == "reserve_quality" for item in records)
    technical_pass = sum(
        bool(item.get("technical_results"))
        and item["technical_results"][-1].get("passed") is True
        for item in records
    )
    blind_clear = sum(item.get("status") in {"blind_pass", "editorial_complete"} for item in records)
    plans_ready = sum(item.get("concrete_plan", {}).get("status") == "ready" for item in records)
    genuine = sum(item.get("status") == "genuine" for item in audit)

    source = Path(config["_v4_3_source_verification"]["source_run"])
    parent_report = read_json(source / "report.json")
    parent_tasks = {
        item["task_id"]: item
        for item in parent_report["tasks"]
        if item.get("private_arm") == "treatment"
    }
    parent_quality = parent_report["quality_noninferiority"]["treatment"]
    tasks = []
    split_board = []
    for item in records:
        judgments = item.get("editorial_judgments") or []
        recommendations = {
            judgment.get("intrinsic_recommendation") for judgment in judgments
        }
        if len(recommendations) > 1 and item.get("complete_task"):
            split_board.append(item["task_id"])
        parent = parent_tasks.get(item["task_id"])
        tasks.append(
            {
                "task_id": item["task_id"],
                "selector_global_priority": item["selector_global_priority"],
                "parent_outcome": (
                    parent.get("intrinsic_quality") or parent.get("terminal_status")
                    if parent
                    else "not_previously_developed"
                ),
                "planner_property_status": item.get("planning_result", {}).get("property_status"),
                "mechanism_audit": audit_by_id.get(item["task_id"]),
                "terminal_status": item.get("status"),
                "technical_pass": bool(item.get("technical_results"))
                and item["technical_results"][-1].get("passed") is True,
                "blind_clear": item.get("status") in {"blind_pass", "editorial_complete"},
                "intrinsic_quality": item.get("intrinsic_quality"),
                "portfolio_placement": item.get("portfolio_placement"),
                "editorial_judgments": judgments,
                "split_board_human_review_candidate": item["task_id"] in split_board,
                "grid_metrics": _task_metrics(item["complete_task"])
                if item.get("complete_task")
                else None,
                "task_json": str(run_dir / "tasks" / "treatment" / f"{item['task_id']}.arc.json")
                if item.get("complete_task")
                else None,
                "plan_json": str(run_dir / "plans" / "treatment" / f"{item['task_id']}.json"),
            }
        )

    accounting = runtime.accounting()
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    if verify_v43_source(config) != config["_v4_3_source_verification"]:
        raise EpisodeJointQualityError("frozen V4.3 source changed during the pilot")
    baseline_after = verify_v42_baseline(
        config["_repository_root"], config["paths"]["v4_2_baseline_manifest"]
    )
    if baseline_after != config["_v4_2_baseline_verification"]:
        raise EpisodeJointQualityError("frozen V4.2 baseline changed during the pilot")
    started = read_json(start_path)["started_at_epoch_seconds"]
    completed = round(time.time(), 3)

    realized_task_rows = [item for item in tasks if item.get("grid_metrics")]
    parent_task_rows = list(parent_tasks.values())
    pilot_core_yield = core / len(realized_records) if realized_records else 0.0
    recommendation = "continue_testing"
    if not realized_records:
        recommendation = "reject"
        reason = "The intervention produced no completed tasks."
    elif genuine < len(realized_records):
        reason = (
            "The routing/architecture intervention did not preserve genuine episode-joint inference in every completed task; inspect the dropped cases before any fresh-supply trial."
        )
    elif pilot_core_yield > float(parent_quality["core_yield"]):
        reason = (
            "The frozen-input pilot improved intrinsic core yield while preserving mechanical, hidden-output, and blind checks. Test the same controls on fresh minority-rate supply before considering integration."
        )
    else:
        reason = (
            "The intervention did not clearly improve core yield over the frozen V4.3 treatment. Retain the current baseline and inspect the task-level authoring results before another iteration."
        )
    report = {
        "schema_version": 1,
        "pipeline": V431_PIPELINE,
        "run_id": runtime.run_id,
        "experimental_design": V431_DESIGN,
        "completed": True,
        "frozen_source": parent_manifest,
        "frozen_v42_baseline": deepcopy(config["_v4_2_baseline_verification"]),
        "funnel": {
            "frozen_kernels": len(kernels),
            "selected": len(selected),
            "globally_excluded_same_procedure_siblings": sum(
                item["reason"] == "same_minimal_procedure_sibling"
                for item in selection["excluded"]
            ),
            "plans_ready": plans_ready,
            "plans_not_ready": len(records) - plans_ready,
            "authored_reserve": len(realization_selection["authored_reserve_candidate_ids"]),
            "realized": len(realized_records),
            "technical_pass": technical_pass,
            "blind_clear": blind_clear,
            "core": core,
            "reserve": reserve,
            "ambiguous": status_counts["ambiguous"],
            "other": len(records) - core - reserve - status_counts["ambiguous"],
        },
        "selection": {
            "manifest": str(run_dir / "global_selection_manifest.private.json"),
            "quality_order": selection["response"]["quality_order"],
            "selected_candidate_ids": [item["candidate_id"] for item in selected],
            "exclusions": selection["excluded"],
            "source_hidden": True,
        },
        "architecture": {
            "quality_gate_added": True,
            "schema_changed": False,
            "model_or_effort_changed": False,
            "property_status_counts": dict(
                sorted(
                    Counter(
                        item.get("planning_result", {}).get("property_status", "failed")
                        for item in records
                    ).items()
                )
            ),
            "realization_selection": realization_selection,
        },
        "efficacy": {
            "genuine_count": genuine,
            "audit_status_counts": dict(sorted(Counter(item["status"] for item in audit).items())),
            "audit_non_gating": True,
        },
        "quality": quality,
        "diversity": diversity,
        "comparison_with_v43_treatment": {
            "parent_selected": int(parent_quality["candidate_count"]),
            "parent_core": int(parent_quality["core_count"]),
            "parent_reserve": int(parent_quality["reserve_count"]),
            "parent_core_yield": float(parent_quality["core_yield"]),
            "pilot_realized": len(realized_records),
            "pilot_core": core,
            "pilot_reserve": reserve,
            "pilot_core_yield": pilot_core_yield,
            "parent_unique_positive_mechanisms": parent_report["diversity"]["summary"]["unique_positive_mechanism_instances"],
            "pilot_unique_positive_mechanisms": diversity["summary"]["unique_positive_mechanism_instances"],
            "parent_mean_maximum_dimension": _mean_metric(parent_task_rows, "maximum_dimension"),
            "pilot_mean_maximum_dimension": _mean_metric(realized_task_rows, "maximum_dimension"),
            "paired_input_not_paired_output": True,
        },
        "human_review_routing": {
            "non_gating": True,
            "split_board_candidate_count": len(split_board),
            "split_board_candidate_ids": split_board,
        },
        "human_review": human,
        "call_accounting": accounting,
        "timing": {
            "started_at_epoch_seconds": started,
            "completed_at_epoch_seconds": completed,
            "wall_clock_minutes": round((completed - started) / 60.0, 3),
        },
        "tasks": tasks,
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "new_idea_calls": 0,
        "downstream_stack_unchanged": True,
        "production_promotion_occurred": False,
        "good_task_bank_mutated": False,
        "protected_production_state_unchanged": protected_before == protected_after,
    }
    atomic_write_json(run_dir / "comparison_with_v43_treatment.json", report["comparison_with_v43_treatment"])
    atomic_write_json(run_dir / "human_review_routing.json", report["human_review_routing"])
    atomic_write_json(run_dir / "call_and_timing_ledger.json", accounting)
    atomic_write_json(
        run_dir / "promotion_recommendation.json",
        {
            "recommendation": recommendation,
            "production_promotion_occurred": False,
            "good_task_bank_mutated": False,
        },
    )
    return runtime.finish(report, _render_report(report))


def rebuild_v431_report(run: str | Path) -> dict[str, Any]:
    path = Path(run).resolve() / "report.json"
    if not path.is_file():
        raise EpisodeJointQualityError(f"V4.3.1 report not found: {path}")
    report = read_json(path)
    if report.get("pipeline") != V431_PIPELINE:
        raise EpisodeJointQualityError("run is not a V4.3.1 quality pilot")
    atomic_write_text(path.parent / "report.md", _render_report(report))
    return report


def v431_human_review_status(run: str | Path) -> dict[str, Any]:
    directory = Path(run).resolve() / "human_review"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise EpisodeJointQualityError(f"V4.3.1 review pack not found: {manifest}")
    value = read_json(manifest)
    return {
        "run": str(Path(run).resolve()),
        "task_count": value["task_count"],
        "review_sheet": str(directory / value["contact_sheet"]),
        "review_form": str(directory / "review_form.json"),
        "answer_appendix": str(directory / "answer_rule_appendix.md"),
        "source_and_arm_hidden": value["source_and_arm_hidden"],
    }


__all__ = [
    "EpisodeJointQualityError",
    "GLOBAL_SELECTOR_SCHEMA",
    "apply_realization_cap",
    "choose_global_development",
    "parse_v431_global_selector",
    "rebuild_v431_report",
    "run_episode_joint_quality_pilot",
    "v431_human_review_status",
]
