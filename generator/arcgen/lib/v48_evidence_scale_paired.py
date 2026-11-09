"""Paired qualification for evidence-preserving functional scale.

The selected-kernel cohort is frozen before either arm begins authoring.  Both
arms receive the same kernel, editorial brief, palette policy, models, efforts,
realizer, technical verifier, blind solver, escalation policy, and boards.
Only the final planner addendum differs:

* ``v45_original_scale`` uses the saved V4.5 functional-scale prompt.
* ``evidence_preserving_scale`` uses the conservative evidence-preserving
  prompt introduced for this experiment.

Post-blind state is frozen before either arm invokes editorial boards so the
same tasks can later support the boardless-throughput study.  Nothing in this
module can commit tasks or mutate the production reservoir/bank.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any

from .config import ConfigError
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import (
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
)
from .v4_pipeline import V4ControlPipeline
from .v41_experiments import V41Runtime, _anonymous_id
from .v41_integration import _assert_hashes_unchanged, _protected_hashes
from .v41_palette import palette_equivalent_hash
from .v41_supported_departure import (
    _realize_records,
    _record_stage,
    _run_blind,
    _run_technical,
)
from .v41_test_challenge import _board as _editorial_boards
from .v44_abstraction_led import (
    _install_extensions as _install_v44_extensions,
)
from .v44_abstraction_led import (
    _normalize_palette_claims,
)
from .v44_functional_scale import (
    SCALE_PLAN_SCHEMA,
    _budgets,
    _install_scale_extension,
    _private_leaks,
    _transport,
    parse_scale_plan,
    task_scale_metrics,
)
from .v44_functional_scale import (
    _arm_summary as _original_arm_summary,
)
from .v44_functional_scale import (
    _genuinely_expanded as _original_genuinely_expanded,
)
from .v44_scale_config import load_v44_scale_config
from .validation import validate_arc_task

PIPELINE = "arcgen-v4.8-exp-evidence-preserving-scale-paired"
VERSION = "arcgen-v4.8-evidence-preserving-scale-paired-1.0"
ARMS = ("v45_original_scale", "evidence_preserving_scale")
EVIDENCE_PROMPT = "v4_8_evidence_preserving_functional_scale_planning_addendum.md"


class EvidenceScaleError(RuntimeError):
    """Raised when the paired scale experiment loses isolation or provenance."""


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"evidence-scale config must be a mapping: {path}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != "1.0":
        raise ConfigError(f"configuration must use {PIPELINE} version 1.0")
    base = Path(str(raw.get("base_scale_config", "")))
    if not base.is_absolute():
        base = (source.parent / base).resolve()
    config = load_v44_scale_config(base)
    root = Path(config["_repository_root"]).resolve()
    selected = Path(str(raw.get("selected_cohort_manifest", "")))
    if not selected.is_absolute():
        selected = (root / selected).resolve()

    settings = raw.get("paired_qualification")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        raise ConfigError("paired_qualification must be explicitly enabled")
    for field in (
        "same_selected_kernels",
        "same_downstream_models_and_efforts",
        "same_palette_assignment",
        "arm_labels_hidden_downstream",
        "post_blind_frozen_before_boards",
        "full_editorial_review",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"paired_qualification.{field} must be true")
    for field in (
        "automatic_promotion",
        "automatic_task_commit",
        "production_state_mutation_allowed",
    ):
        if settings.get(field) is not False:
            raise ConfigError(f"paired_qualification.{field} must be false")
    for field, minimum in {
        "minimum_selected_lineages": 1,
        "maximum_selected_lineages": 1,
        "arm_concurrency": 1,
        "global_model_concurrency": 2,
        "escalation_calls_per_slate_per_arm": 0,
        "minimum_genuine_expansions": 1,
        "maximum_additional_ambiguity_cases": 0,
    }.items():
        value = settings.get(field)
        if type(value) is not int or value < minimum:
            raise ConfigError(f"paired_qualification.{field} must be >= {minimum}")
    if settings["maximum_selected_lineages"] < settings["minimum_selected_lineages"]:
        raise ConfigError("maximum_selected_lineages cannot be below minimum")
    if settings["global_model_concurrency"] % len(ARMS):
        raise ConfigError("global_model_concurrency must divide evenly across arms")
    for field in ("core_yield_ratio_floor", "core_or_reserve_yield_ratio_floor"):
        value = settings.get(field)
        if type(value) not in {int, float} or not 0 <= float(value) <= 1:
            raise ConfigError(f"paired_qualification.{field} must be in [0,1]")
    run = raw.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str):
        raise ConfigError("run.id is required")
    if run.get("mock") not in {True, False}:
        raise ConfigError("run.mock must be boolean")
    if run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("experiment cannot automatically commit tasks")
    if type(raw.get("seed")) is not int:
        raise ConfigError("seed must be an integer")

    config = deepcopy(config)
    config["paired_qualification"] = deepcopy(dict(settings))
    config["seed"] = int(raw["seed"])
    config["model"] = deepcopy(config["model"])
    config["model"]["concurrency"] = int(settings["global_model_concurrency"])
    config["functional_scale"] = deepcopy(config["functional_scale"])
    config["functional_scale"]["escalation_calls_per_slate"] = int(
        settings["escalation_calls_per_slate_per_arm"]
    )
    config["run"] = {**deepcopy(config["run"]), **deepcopy(dict(run))}
    config["_evidence_scale_config_path"] = str(source)
    config["_evidence_scale_base_config"] = str(base)
    config["_selected_cohort_manifest_path"] = str(selected)
    return config


def _snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value) for key, value in config.items() if not key.startswith("_")
    }


def _implementation_manifest(
    config: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    active = Path(config["_evidence_scale_config_path"])
    relatives = (
        "arcgen.lib/v48_evidence_scale_paired.py",
        "arcgen.lib/v44_functional_scale.py",
        "arcgen.lib/v41_supported_departure.py",
        str(active.relative_to(root)),
        "prompts/v4_concrete_planner.md",
        "prompts/v4_4_concrete_quality_planning_addendum.md",
        "prompts/v4_4_functional_scale_planning_addendum.md",
        f"prompts/{EVIDENCE_PROMPT}",
        "prompts/v4_task_realizer.md",
        "prompts/v4_technical_verifier.md",
        "prompts/v4_blind_solver.md",
        "prompts/v4_editorial_board.md",
    )
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "selected_cohort_manifest": {
            "path": config["_selected_cohort_manifest_path"],
            "sha256": _sha(config["_selected_cohort_manifest_path"]),
        },
        "files": {relative: _sha(root / relative) for relative in relatives},
        "one_variable_only": "final functional-scale planner addendum text",
        "wrapper_schema_identical": True,
    }
    path = run_dir / "implementation_manifest.json"
    if path.exists() and read_json(path) != value:
        raise EvidenceScaleError("implementation changed during resume")
    if not path.exists():
        atomic_write_json(path, value)
    return value


def _write_immutable(path: Path, value: Any, *, label: str) -> None:
    if path.exists():
        if read_json(path) != value:
            raise EvidenceScaleError(f"{label} changed during resume")
        return
    atomic_write_json(path, value)


def _load_selected(config: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    frozen_path = run_dir / "selected_kernel_manifest.private.json"
    if frozen_path.exists():
        return deepcopy(read_json(frozen_path)["development_sample"])
    source_path = Path(config["_selected_cohort_manifest_path"])
    source = read_json(source_path)
    if (
        source.get("completed") is not True
        or source.get("selection_only") is not True
        or source.get("stopped_before_planning") is not True
    ):
        raise EvidenceScaleError("selected cohort is not complete before planning")
    rows: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    for slate in source.get("slates", []):
        artifact = slate.get("selector_source_manifest")
        if not isinstance(artifact, Mapping):
            raise EvidenceScaleError(
                "selected cohort slate lacks selector source manifest"
            )
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha(path) != artifact.get("sha256"):
            raise EvidenceScaleError("selected cohort source artifact changed")
        value = read_json(path)
        if value.get("selection_stopped_before_planning") is not True:
            raise EvidenceScaleError("source selection crossed the planning boundary")
        source_artifacts.append({"path": str(path), "sha256": _sha(path)})
        for raw in value.get("selected", []):
            rows.append(
                {
                    "source_candidate_id": str(raw["candidate_id"]),
                    "slate_index": int(raw["slate_index"]),
                    "source_development_priority": int(raw["selector_global_priority"]),
                    "search_mode": str(raw["search_mode"]),
                    "private_kernel_author_provider": str(
                        raw["private_kernel_author_provider"]
                    ),
                    "kernel": deepcopy(raw["kernel"]),
                    "editorial_brief": deepcopy(raw["editorial_brief"]),
                }
            )
    ids = [row["source_candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise EvidenceScaleError("selected candidate repeats across frozen slates")
    rows.sort(
        key=lambda row: (
            int(row["slate_index"]),
            int(row["source_development_priority"]),
        )
    )
    maximum = int(config["paired_qualification"]["maximum_selected_lineages"])
    if len(rows) > maximum:
        # A deterministic prefix by complete selected-slate order changes no
        # within-slate ranking and is fixed before arm assignment.
        rows = rows[:maximum]
    minimum = int(config["paired_qualification"]["minimum_selected_lineages"])
    if len(rows) < minimum:
        raise EvidenceScaleError(
            f"selectors retained {len(rows)} lineages; {minimum} required; do not lower quality"
        )
    frozen = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "selection_completed_before_arm_assignment": True,
        "selector_received_scale_prompt": False,
        "selected_cohort_manifest": {
            "path": str(source_path),
            "sha256": _sha(source_path),
        },
        "source_artifacts": source_artifacts,
        "development_sample": rows,
    }
    _write_immutable(frozen_path, frozen, label="selected-kernel manifest")
    atomic_write_json(
        run_dir / "selected_kernel_manifest.public.json",
        {
            "schema_version": 1,
            "selection_completed_before_arm_assignment": True,
            "selector_received_scale_prompt": False,
            "selected_count": len(rows),
            "search_mode_counts": dict(
                sorted(Counter(row["search_mode"] for row in rows).items())
            ),
        },
    )
    return deepcopy(rows)


def _inspect_selected_cohort(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only preflight for the frozen selection handoff."""

    source_path = Path(config["_selected_cohort_manifest_path"])
    if not source_path.is_file():
        return {"available": False, "ready": False, "selected_count": 0}
    source = read_json(source_path)
    if (
        source.get("completed") is not True
        or source.get("selection_only") is not True
        or source.get("stopped_before_planning") is not True
    ):
        raise EvidenceScaleError("selected cohort is not complete before planning")
    ids: list[str] = []
    artifacts = []
    for slate in source.get("slates", []):
        artifact = slate.get("selector_source_manifest")
        if not isinstance(artifact, Mapping):
            raise EvidenceScaleError(
                "selected cohort slate lacks selector source manifest"
            )
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha(path) != artifact.get("sha256"):
            raise EvidenceScaleError("selected cohort source artifact changed")
        value = read_json(path)
        if value.get("selection_stopped_before_planning") is not True:
            raise EvidenceScaleError("source selection crossed the planning boundary")
        ids.extend(str(row["candidate_id"]) for row in value.get("selected", []))
        artifacts.append({"path": str(path), "sha256": _sha(path)})
    if len(ids) != len(set(ids)):
        raise EvidenceScaleError("selected candidate repeats across frozen slates")
    maximum = int(config["paired_qualification"]["maximum_selected_lineages"])
    count = min(len(ids), maximum)
    minimum = int(config["paired_qualification"]["minimum_selected_lineages"])
    return {
        "available": True,
        "ready": count >= minimum,
        "selected_count": count,
        "source_selected_count": len(ids),
        "slate_count": len(source.get("slates", [])),
        "source_manifest_sha256": _sha(source_path),
        "selector_artifacts": artifacts,
        "planning_calls": source.get("funnel", {}).get("planning_calls"),
        "realization_calls": source.get("funnel", {}).get("realization_calls"),
        "verification_calls": source.get("funnel", {}).get("verification_calls"),
        "editorial_calls": source.get("funnel", {}).get("editorial_calls"),
    }


def _aliases(
    selected: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, dict[str, str]]:
    rng = Random(seed + 480045)
    used: set[str] = set()
    return {
        str(row["source_candidate_id"]): {arm: _anonymous_id(rng, used) for arm in ARMS}
        for row in sorted(selected, key=lambda item: str(item["source_candidate_id"]))
    }


def _install_evidence_extension(runtime: V41Runtime) -> None:
    root = Path(runtime.config["_repository_root"])
    text, version, digest = load_prompt(root / "prompts" / EVIDENCE_PROMPT)
    runtime.prompts.entries["evidence_scale_addendum"] = {
        "filename": EVIDENCE_PROMPT,
        "text": text,
        "version": version,
        "hash": digest,
        "stage": "v44_functional_scale_plan",
    }
    destination = runtime.run_dir / "prompts" / "evidence_scale"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / EVIDENCE_PROMPT
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise EvidenceScaleError("evidence-scale prompt changed during resume")
    if not target.exists():
        atomic_write_text(target, text)
    manifest = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "prompt": {"filename": EVIDENCE_PROMPT, "version": version, "sha256": digest},
        "wrapper_schema_sha256": content_hash(SCALE_PLAN_SCHEMA),
        "one_variable_only": "planner addendum text",
    }
    _write_immutable(
        runtime.run_dir / "evidence_scale_extension_manifest.json",
        manifest,
        label="evidence-scale extension manifest",
    )
    original = runtime.invoker.output_parser

    def parse(stage: str, data: Any) -> Any:
        if stage == "v44_functional_scale_plan":
            return parse_scale_plan(data)
        return original(stage, data)

    runtime.invoker.output_parser = parse
    runtime.invoker.output_schemas["v44_functional_scale_plan"] = _transport(
        SCALE_PLAN_SCHEMA
    )
    runtime.invoker._write_schemas()


def _arm_config(config: Mapping[str, Any], outer: Path, arm: str) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["paths"] = deepcopy(config["paths"])
    value["paths"]["runs"] = str(outer / "execution")
    value["run"] = {**deepcopy(config["run"]), "id": arm}
    value["model"] = deepcopy(config["model"])
    value["model"]["concurrency"] = int(
        config["paired_qualification"]["global_model_concurrency"]
    ) // len(ARMS)
    return value


def _initialize_arm(
    runtime: V41Runtime,
    *,
    selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, Mapping[str, str]],
    arm: str,
) -> None:
    with runtime.lock:
        if "semantic_records" in runtime.state:
            return
        counterpart = ARMS[1] if arm == ARMS[0] else ARMS[0]
        records: dict[str, Any] = {}
        for row in selected:
            source = str(row["source_candidate_id"])
            task_id = str(aliases[source][arm])
            records[task_id] = {
                "task_id": task_id,
                "source_candidate_id": source,
                "paired_lineage_id": content_hash({"source_candidate_id": source}),
                "paired_counterpart_task_id": aliases[source][counterpart],
                "arm": "single",
                "scale_arm": arm,
                "slate_index": int(row["slate_index"]),
                "source_development_priority": int(row["source_development_priority"]),
                "search_mode": row["search_mode"],
                "private_kernel_author_provider": row["private_kernel_author_provider"],
                "kernel": deepcopy(row["kernel"]),
                "editorial_brief": deepcopy(row["editorial_brief"]),
                "status": "selected",
                "stage_history": [],
                "repair_count": 0,
            }
        runtime.state["semantic_records"] = records
        runtime.state["evidence_scale_arm"] = arm
        runtime.save()


def _plan_records(runtime: V41Runtime, arm: str) -> None:
    records = runtime.state["semantic_records"]
    prompt = (
        "functional_scale_addendum"
        if arm == "v45_original_scale"
        else "evidence_scale_addendum"
    )

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
            result = runtime.call(
                prompt_name="concrete_planner",
                composed_with=("v44_quality_plan", prompt),
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
                    raise EvidenceScaleError("nested plan changed candidate ID")
                plan = _normalize_palette_claims(plan)
                result["concrete_plan"] = plan
                leaks = _private_leaks(plan)
                if leaks:
                    raise EvidenceScaleError(
                        f"downstream plan leaked private labels: {leaks}"
                    )
            if (
                plan is not None
                and plan.get("status") == "abort"
                and result.get("scale_decision") != "abort"
            ):
                raise EvidenceScaleError("aborted plan must use abort scale decision")
            return task_id, result, None
        except Exception as exc:  # noqa: BLE001 - preserve per-candidate progress
            return task_id, None, str(exc)

    pending = [
        task_id for task_id, row in records.items() if "planning_result" not in row
    ]
    results = (
        parallel_map(
            one,
            pending,
            min(len(pending), int(runtime.config["model"]["concurrency"])),
        )
        if pending
        else []
    )
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
        plan = result.get("concrete_plan")
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


def _safe_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sanitized = []
    for source in records:
        row = deepcopy(dict(source))
        if row.get("concrete_plan") is None:
            row["concrete_plan"] = {}
        if row.get("deterministic_diagnostics") is None:
            row["deterministic_diagnostics"] = {}
        sanitized.append(row)
    return _original_arm_summary(sanitized)


def _genuinely_expanded(record: Mapping[str, Any]) -> bool:
    row = deepcopy(dict(record))
    for field in ("planning_result", "realization"):
        if row.get(field) is None:
            row[field] = {}
    return _original_genuinely_expanded(row)


def _freeze_post_blind(runtime: V41Runtime, arm: str) -> dict[str, Any]:
    path = runtime.run_dir / "post_blind.private.json"
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "arm": arm,
        "frozen_before_editorial_boards": True,
        "editorial_calls_at_freeze": sum(
            item.get("role") == "editorial_board"
            for item in runtime.state.get("call_ledger", {}).values()
        ),
        "records": deepcopy(runtime.state["semantic_records"]),
        "call_ledger": deepcopy(runtime.state.get("call_ledger", {})),
    }
    if value["editorial_calls_at_freeze"]:
        raise EvidenceScaleError("editorial work crossed the post-blind freeze")
    _write_immutable(path, value, label=f"{arm} post-blind freeze")
    return value


def _materialize_record(
    outer: Path,
    runtime: V41Runtime,
    arm: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(record["task_id"])
    root = outer / "final_state_exports" / arm
    complete = record.get("complete_task")
    task_path: Path | None = None
    exact_hash: str | None = None
    palette_hash: str | None = None
    metrics: dict[str, Any] | None = None
    if complete is not None:
        validation = validate_arc_task(complete, require_test_outputs=True)
        if not validation.valid:
            raise EvidenceScaleError(
                f"invalid terminal task {arm}/{task_id}: {'; '.join(validation.errors)}"
            )
        task_path = root / "tasks" / f"{task_id}.arc.json"
        atomic_write_json(task_path, complete)
        exact_hash = content_hash(complete)
        palette_hash = palette_equivalent_hash(complete)
        metrics = task_scale_metrics(complete)
    plan_path: Path | None = None
    if isinstance(record.get("concrete_plan"), Mapping):
        plan_path = root / "plans" / f"{task_id}.json"
        payload = deepcopy(record.get("planning_result") or {})
        payload["candidate_id"] = task_id
        payload["concrete_plan"] = deepcopy(record["concrete_plan"])
        atomic_write_json(plan_path, payload)
    authoring_path = root / "authoring" / f"{task_id}.authoring.json"
    atomic_write_json(
        authoring_path,
        {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "arm": arm,
            "terminal_semantic_record": deepcopy(dict(record)),
            "source_state": str(runtime.state_path),
            "source_state_sha256": _sha(runtime.state_path),
        },
    )
    technical = record.get("technical_results") or []
    latest = technical[-1] if technical else {}
    return {
        "task_id": task_id,
        "source_candidate_id": record["source_candidate_id"],
        "paired_lineage_id": record["paired_lineage_id"],
        "paired_counterpart_task_id": record["paired_counterpart_task_id"],
        "private_arm": arm,
        "search_mode": record["search_mode"],
        "slate_index": record["slate_index"],
        "development_priority": record["source_development_priority"],
        "source_provider_private": record["private_kernel_author_provider"],
        "scale_decision": (record.get("planning_result") or {}).get("scale_decision"),
        "scale_reason": (record.get("planning_result") or {}).get("scale_reason"),
        "functionally_expanded": _genuinely_expanded(record),
        "status": record.get("status"),
        "realized": complete is not None,
        "technical_pass": latest.get("passed") is True,
        "hidden_output_exact": latest.get("hidden_output_match"),
        "blind_clear": record.get("status") in {"blind_pass", "editorial_complete"},
        "serious_ambiguity": record.get("status") == "ambiguous"
        or record.get("intrinsic_quality") == "ambiguous",
        "repair_count": int(record.get("repair_count") or 0),
        "intrinsic_quality": record.get("intrinsic_quality"),
        "portfolio_placement": record.get("portfolio_placement"),
        "scale_metrics": metrics,
        "final_exact_task_hash": exact_hash,
        "palette_equivalent_identity_hash": palette_hash,
        "task_json": str(task_path) if task_path else None,
        "plan_json": str(plan_path) if plan_path else None,
        "authoring_json": str(authoring_path),
        "technical_results": deepcopy(technical),
        "blind_results": deepcopy(record.get("blind_results") or []),
        "editorial_judgments": deepcopy(record.get("editorial_judgments") or []),
    }


def _arm_markdown(arm: str, summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# {arm}",
            "",
            f"Selected: {summary['selected']}",
            f"Realized: {summary['realized']}",
            f"Technical pass: {summary['technical_pass']}",
            f"Blind clear: {summary['blind_clear']}",
            f"Core / reserve / ambiguous: {summary['core']} / {summary['reserve']} / {summary['ambiguous']}",
            "",
            "Post-blind state was frozen before the unchanged editorial boards ran.",
            "This experimental arm cannot promote or commit tasks.",
            "",
        ]
    )


def _run_arm(
    config: Mapping[str, Any],
    outer: Path,
    selected: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, Mapping[str, str]],
    arm: str,
) -> dict[str, Any]:
    arm_config = _arm_config(config, outer, arm)
    runtime = V41Runtime(
        arm_config,
        {
            "schema_version": 1,
            "pipeline": PIPELINE,
            "version": VERSION,
            "arm": arm,
            "selected_manifest": str(outer / "selected_kernel_manifest.private.json"),
            "selected_manifest_sha256": _sha(
                outer / "selected_kernel_manifest.private.json"
            ),
            "private_arm_hidden_downstream": True,
            "automatic_task_commit": False,
        },
    )
    _install_v44_extensions(runtime)
    if arm == "v45_original_scale":
        _install_scale_extension(runtime)
    else:
        _install_evidence_extension(runtime)
    _initialize_arm(runtime, selected=selected, aliases=aliases, arm=arm)
    post_blind_path = runtime.run_dir / "post_blind.private.json"
    if not post_blind_path.exists():
        _plan_records(runtime, arm)
        _realize_records(runtime)
        budgets = _budgets(runtime)
        _run_technical(runtime, budgets)
        _run_blind(runtime, budgets)
        _freeze_post_blind(runtime, arm)
    if not runtime.state.get("completed"):
        _editorial_boards(runtime)
    records = sorted(
        runtime.state["semantic_records"].values(),
        key=lambda row: (
            int(row["slate_index"]),
            int(row["source_development_priority"]),
        ),
    )
    summary = _safe_summary(records)
    rows = [_materialize_record(outer, runtime, arm, row) for row in records]
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "arm": arm,
        "completed": True,
        "post_blind_freeze": {
            "path": str(post_blind_path),
            "sha256": _sha(post_blind_path),
        },
        "summary": summary,
        "tasks": rows,
        "private_arm_hidden_downstream": True,
        "automatic_task_commit": False,
    }
    if runtime.state.get("completed") and (runtime.run_dir / "report.json").exists():
        prior = read_json(runtime.run_dir / "report.json")
        if prior != report:
            raise EvidenceScaleError(f"{arm} completed report changed")
        completed = prior
    else:
        completed = runtime.finish(report, _arm_markdown(arm, summary))
    return {"runtime": runtime, "records": records, "report": completed}


def _paired_outcomes(outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {
        arm: {str(row["source_candidate_id"]): row for row in output["records"]}
        for arm, output in outputs.items()
    }
    if set(by_arm[ARMS[0]]) != set(by_arm[ARMS[1]]):
        raise EvidenceScaleError("paired arms do not contain identical lineages")
    rows = []
    for source in sorted(by_arm[ARMS[0]]):
        left = by_arm[ARMS[0]][source]
        right = by_arm[ARMS[1]][source]
        rows.append(
            {
                "source_candidate_id": source,
                "paired_lineage_id": left["paired_lineage_id"],
                "v45_task_id": left["task_id"],
                "evidence_task_id": right["task_id"],
                "v45_scale_decision": (left.get("planning_result") or {}).get(
                    "scale_decision"
                ),
                "evidence_scale_decision": (right.get("planning_result") or {}).get(
                    "scale_decision"
                ),
                "v45_core": left.get("intrinsic_quality") == "core_quality",
                "evidence_core": right.get("intrinsic_quality") == "core_quality",
                "v45_useful": left.get("intrinsic_quality")
                in {"core_quality", "reserve_quality"},
                "evidence_useful": right.get("intrinsic_quality")
                in {"core_quality", "reserve_quality"},
                "v45_ambiguity": left.get("status") == "ambiguous"
                or left.get("intrinsic_quality") == "ambiguous",
                "evidence_ambiguity": right.get("status") == "ambiguous"
                or right.get("intrinsic_quality") == "ambiguous",
                "v45_functionally_expanded": _genuinely_expanded(left),
                "evidence_functionally_expanded": _genuinely_expanded(right),
                "v45_repair_count": int(left.get("repair_count") or 0),
                "evidence_repair_count": int(right.get("repair_count") or 0),
            }
        )
    return {
        "paired_lineage_count": len(rows),
        "both_core": sum(row["v45_core"] and row["evidence_core"] for row in rows),
        "v45_only_core": sum(
            row["v45_core"] and not row["evidence_core"] for row in rows
        ),
        "evidence_only_core": sum(
            row["evidence_core"] and not row["v45_core"] for row in rows
        ),
        "neither_core": sum(
            not row["v45_core"] and not row["evidence_core"] for row in rows
        ),
        "v45_ambiguities": sum(row["v45_ambiguity"] for row in rows),
        "evidence_ambiguities": sum(row["evidence_ambiguity"] for row in rows),
        "v45_genuine_expansions": sum(row["v45_functionally_expanded"] for row in rows),
        "evidence_genuine_expansions": sum(
            row["evidence_functionally_expanded"] for row in rows
        ),
        "rows": rows,
    }


def _quality_gate(
    config: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Any],
) -> dict[str, Any]:
    settings = config["paired_qualification"]
    control = outputs[ARMS[0]]["report"]["summary"]
    treatment = outputs[ARMS[1]]["report"]["summary"]
    core_ratio = (
        treatment["core_yield"] / control["core_yield"]
        if control.get("core_yield")
        else None
    )
    useful_ratio = (
        treatment["core_or_reserve_yield"] / control["core_or_reserve_yield"]
        if control.get("core_or_reserve_yield")
        else None
    )
    hidden_mismatch = sum(
        row.get("realized")
        and row.get("technical_pass")
        and row.get("hidden_output_exact") is not True
        for row in outputs[ARMS[1]]["report"]["tasks"]
    )
    artificial = sum(
        (
            read_json(row["authoring_json"])["terminal_semantic_record"]
            .get("realization", {})
            .get("authoring_metadata", {})
            .get("canvas_use_classification")
            == "artificially_inflated"
        )
        for row in outputs[ARMS[1]]["report"]["tasks"]
        if row.get("authoring_json")
    )
    checks = {
        "zero_hidden_output_mismatch_after_technical_pass": hidden_mismatch == 0,
        "ambiguity_not_worse_than_control_plus_bound": paired["evidence_ambiguities"]
        <= paired["v45_ambiguities"]
        + int(settings["maximum_additional_ambiguity_cases"]),
        "core_yield_noninferior": core_ratio is not None
        and core_ratio >= float(settings["core_yield_ratio_floor"]),
        "core_or_reserve_yield_noninferior": useful_ratio is not None
        and useful_ratio >= float(settings["core_or_reserve_yield_ratio_floor"]),
        "minimum_genuine_expansions": paired["evidence_genuine_expansions"]
        >= int(settings["minimum_genuine_expansions"]),
        "zero_artificial_inflation": artificial == 0,
    }
    return {
        "provisional_pass": all(checks.values()),
        "checks": checks,
        "core_yield_ratio": core_ratio,
        "core_or_reserve_yield_ratio": useful_ratio,
        "treatment_hidden_output_mismatches": hidden_mismatch,
        "treatment_artificially_inflated": artificial,
        "automatic_promotion_occurred": False,
        "requires_independent_quality_review": True,
    }


def _render(outer: Path, outputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    destination = outer / "renders"
    result: dict[str, Any] = {}
    for arm in ARMS:
        images = []
        tasks = []
        for record in outputs[arm]["records"]:
            if not record.get("complete_task"):
                continue
            task_id = str(record["task_id"])
            rule = str(record["realization"]["authoring_metadata"]["intended_rule"])
            path = destination / arm / f"{task_id}.png"
            image = render_episode(
                record["complete_task"],
                title=task_id,
                subtitle=f"{record['search_mode']} · lineage {record['paired_lineage_id'][:10]}",
                rule=rule,
                destination=path,
            )
            images.append((task_id, image))
            tasks.append({"task_id": task_id, "render": str(path)})
        contact = destination / f"{arm}_contact_sheet.png"
        if images:
            build_contact_sheet(images, contact)
        result[arm] = {
            "task_count": len(tasks),
            "contact_sheet": str(contact) if images else None,
            "tasks": tasks,
        }
    atomic_write_json(destination / "manifest.json", result)
    return result


def _markdown(report: Mapping[str, Any]) -> str:
    left = report["arms"][ARMS[0]]["summary"]
    right = report["arms"][ARMS[1]]["summary"]
    paired = report["paired_outcomes"]
    gate = report["quality_gate"]
    return "\n".join(
        [
            "# Evidence-preserving functional-scale qualification",
            "",
            "The same selected kernels were authored in both arms. Only the final planner addendum differed.",
            "",
            "| Arm | Selected | Realized | Technical | Blind clear | Core | Reserve | Ambiguous | Genuine expansion |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| Original V4.5 scale | {left['selected']} | {left['realized']} | {left['technical_pass']} | {left['blind_clear']} | {left['core']} | {left['reserve']} | {left['ambiguous']} | {paired['v45_genuine_expansions']} |",
            f"| Evidence-preserving scale | {right['selected']} | {right['realized']} | {right['technical_pass']} | {right['blind_clear']} | {right['core']} | {right['reserve']} | {right['ambiguous']} | {paired['evidence_genuine_expansions']} |",
            "",
            f"Provisional gate: **{'pass' if gate['provisional_pass'] else 'fail'}**.",
            "",
            "No automatic promotion or task-bank commitment occurred. Post-blind state was frozen before boards for later boardless analysis.",
            "",
        ]
    )


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    selected = Path(config["_selected_cohort_manifest_path"])
    cohort = _inspect_selected_cohort(config)
    root = Path(config["_repository_root"])
    original_text, original_version, original_hash = load_prompt(
        root / "prompts/v4_4_functional_scale_planning_addendum.md"
    )
    treatment_text, treatment_version, treatment_hash = load_prompt(
        root / "prompts" / EVIDENCE_PROMPT
    )
    return {
        "valid": cohort["ready"],
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "run_id": config["run"]["id"],
        "selected_cohort_manifest": str(selected),
        "selected_cohort": cohort,
        "one_variable_only": "final functional-scale planner addendum text",
        "prompt_arms": {
            "v45_original_scale": {
                "version": original_version,
                "sha256": original_hash,
                "characters": len(original_text),
            },
            "evidence_preserving_scale": {
                "version": treatment_version,
                "sha256": treatment_hash,
                "characters": len(treatment_text),
            },
        },
        "wrapper_schema_sha256": content_hash(SCALE_PLAN_SCHEMA),
        "model_roles": deepcopy(config["model"]["roles"]),
        "automatic_task_commit": False,
    }


def run(config: dict[str, Any], *, authorize_model_calls: bool) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise EvidenceScaleError("real model calls require --authorize-model-calls")
    outer = Path(config["paths"]["runs"]) / config["run"]["id"]
    outer.mkdir(parents=True, exist_ok=True)
    report_path = outer / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report.get("pipeline") != PIPELINE:
            raise EvidenceScaleError("existing run belongs to another pipeline")
        return report
    snapshot = _snapshot(config)
    _write_immutable(
        outer / "config.snapshot.json", snapshot, label="configuration snapshot"
    )
    _implementation_manifest(config, outer)
    started_path = outer / "experiment_start.json"
    if not started_path.exists():
        atomic_write_json(
            started_path, {"started_at_epoch_seconds": round(time.time(), 3)}
        )
    protected_before = _protected_hashes(config)
    selected = _load_selected(config, outer)
    aliases_path = outer / "paired_aliases.private.json"
    aliases = (
        read_json(aliases_path)["aliases"]
        if aliases_path.exists()
        else _aliases(selected, int(config["seed"]))
    )
    _write_immutable(
        aliases_path,
        {
            "schema_version": 1,
            "arms_private": True,
            "same_lineage_in_both_arms": True,
            "aliases": aliases,
        },
        label="paired alias manifest",
    )

    def one(arm: str) -> tuple[str, dict[str, Any]]:
        return arm, _run_arm(config, outer, selected, aliases, arm)

    outputs = dict(
        parallel_map(
            one,
            list(ARMS),
            min(len(ARMS), int(config["paired_qualification"]["arm_concurrency"])),
        )
    )
    left_palette = {
        row["source_candidate_id"]: (row.get("palette_treatment") or {}).get(
            "requested_background_color"
        )
        for row in outputs[ARMS[0]]["records"]
    }
    right_palette = {
        row["source_candidate_id"]: (row.get("palette_treatment") or {}).get(
            "requested_background_color"
        )
        for row in outputs[ARMS[1]]["records"]
    }
    if left_palette != right_palette:
        raise EvidenceScaleError("paired palette assignments differ")
    paired = _paired_outcomes(outputs)
    gate = _quality_gate(config, outputs, paired)
    renders = _render(outer, outputs)
    protected_after = _protected_hashes(config)
    _assert_hashes_unchanged(protected_before, protected_after)
    completed = round(time.time(), 3)
    started = float(read_json(started_path)["started_at_epoch_seconds"])
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "run_id": config["run"]["id"],
        "completed": True,
        "experimental_design": {
            "selected_lineages": len(selected),
            "same_kernel_and_brief": True,
            "same_palette_assignment": True,
            "same_downstream_models_and_efforts": True,
            "arm_labels_hidden_downstream": True,
            "post_blind_frozen_before_boards": True,
            "one_variable_only": "final functional-scale planner addendum text",
            "intention_to_treat": True,
        },
        "arms": {arm: outputs[arm]["report"] for arm in ARMS},
        "paired_outcomes": paired,
        "quality_gate": gate,
        "renders": renders,
        "call_accounting": {arm: outputs[arm]["runtime"].accounting() for arm in ARMS},
        "timing": {
            "started_at_epoch_seconds": started,
            "completed_at_epoch_seconds": completed,
            "wall_clock_minutes": round((completed - started) / 60.0, 3),
        },
        "production_state_unchanged": protected_before == protected_after,
        "automatic_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "independent_quality_review_pending": True,
    }
    atomic_write_json(report_path, report)
    atomic_write_text(outer / "report.md", _markdown(report))
    atomic_write_json(
        outer / "non_mutation_attestation.json",
        {"before": protected_before, "after": protected_after, "unchanged": True},
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "report"))
    parser.add_argument("--config", default="config/v4_8_evidence_scale_paired_01.yaml")
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise EvidenceScaleError("--run is required")
            value = read_json(Path(args.run) / "report.json")
        else:
            config = load_config(args.config)
            value = (
                validate_config(config)
                if args.command == "validate"
                else run(config, authorize_model_calls=args.authorize_model_calls)
            )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI reports structured experiment failures
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "PIPELINE",
    "VERSION",
    "EvidenceScaleError",
    "load_config",
    "run",
    "validate_config",
]
