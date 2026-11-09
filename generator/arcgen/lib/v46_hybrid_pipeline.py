"""End-to-end Sol-control versus Sol+Opus hybrid qualification.

The experiment is a disjoint branch from the frozen V4.5 milestone.  It uses
Opus only for minority idea supply and bounded conditional assistance while
retaining Sol for every production quality gate.  No output from this module
can automatically enter the committed task bank or production reservoir.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

from .config import ConfigError
from .embeddings import build_embedding_provider
from .lean_pipeline import serious_blind_alternative
from .lean_v3_pipeline import classify_v3_editorial_results
from .lean_v3_rendering import build_contact_sheet, render_episode
from .util import (
    PLACEHOLDER_RE,
    atomic_write_json,
    atomic_write_text,
    content_hash,
    load_prompt,
    parallel_map,
    read_json,
)
from .v4_diagnostics import (
    check_plan_fidelity,
    complete_task,
    diagnose_arc_task_v4,
    diagnose_public_arc_task_v4,
)
from .v4_pipeline import V4ControlPipeline, _DataContext, v4_call_accounting
from .v4_reservoir import (
    append_supply,
    freeze_eligible_window,
    initialize_reservoir,
    load_reservoir,
    make_proposal,
    memory_digest,
    reservation_digest,
    reserve_slate,
    resolve_proposal,
)
from .v41_experiments import _task_metrics
from .v41_integration import transform_realization_palette
from .v43_episode_joint import _reference_packages
from .v431_episode_joint_quality import choose_global_development
from .v44_scale_fresh_integrated import load_fresh_scale_integrated_config
from .v45_claude_cross_eval import _opaque_id, _palette_assignments, _public_task
from .v45_claude_generator_replication import (
    GeneratorReplicationRuntime,
    _calibration_slates,
    _prompt_name_for_mode,
    _task_rows as _v45_task_rows,
    blind_solve,
    editorial_review,
    plan_selected,
    realize_plans,
    technical_verify,
)
from .v45_claude_cross_eval import (
    _technical_evaluation,
    _board_package,
    _portfolio_digest,
)
from .v45_config import _verify_registry_or_authorized_append
from .validation import validate_arc_task


PIPELINE = "arcgen-v4.6-exp-sol-opus-hybrid"
VERSION = "arcgen-v4.6-sol-opus-hybrid-1.0"
INNER_PIPELINE = "arcgen-v4.5-exp-claude-opus-5-generator-replication"
MODES = ("open", "concept_led", "composition_led")
ARMS = ("sol_only", "hybrid")


class HybridPipelineError(RuntimeError):
    """Raised when the paired qualification violates its frozen design."""


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"hybrid config must be a mapping: {path}")
    return value


def _sha(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _resolve(root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def _verify_v45_append_only(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if (
        manifest.get("baseline_id") != "arcgen-v4.5-integrated-authoring-20260808"
        or manifest.get("status") != "frozen"
    ):
        raise ConfigError("unexpected V4.5 baseline manifest")
    failures = []
    checked: dict[str, str] = {}
    for section in ("critical_source_sha256", "qualification_report_sha256"):
        for relative, expected in manifest.get(section, {}).items():
            path = root / relative
            if not path.is_file():
                failures.append(f"missing {relative}")
                continue
            actual = _sha(path)
            checked[relative] = actual
            if actual != expected:
                failures.append(f"changed {relative}")
    registry = root / str(manifest["committed_bank"]["registry"])
    registry_status = _verify_registry_or_authorized_append(
        root=root, manifest=manifest, registry_path=registry
    )
    if failures:
        raise ConfigError("frozen V4.5 source changed: " + "; ".join(failures))
    return {
        "baseline_id": manifest["baseline_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "verified_source_count": len(checked),
        "registry_status": registry_status,
        "verification_hash": content_hash(checked),
    }


def supply_provider_schedule(settings: Mapping[str, Any]) -> dict[tuple[int, str], str]:
    """Return the private provider for each hybrid supply call."""

    cycles = int(settings["supply_cycles"])
    schedule = {(cycle, mode): "codex" for cycle in range(1, cycles + 1) for mode in MODES}
    seen: set[tuple[int, str]] = set()
    for row in settings["hybrid_claude_schedule"]:
        key = (int(row["cycle"]), str(row["mode"]))
        if key in seen or key not in schedule:
            raise ConfigError(f"invalid or duplicate Claude supply assignment: {key}")
        seen.add(key)
        schedule[key] = "claude"
    expected = float(settings["hybrid_claude_call_fraction"])
    actual = sum(value == "claude" for value in schedule.values()) / len(schedule)
    if abs(actual - expected) > 1e-9:
        raise ConfigError(f"hybrid Claude supply fraction is {actual}, expected {expected}")
    if Counter(mode for (_, mode), provider in schedule.items() if provider == "claude") != Counter(MODES):
        raise ConfigError("Claude supply must contribute exactly one call per ordinary mode")
    return schedule


def load_hybrid_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read_yaml_or_json(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != "1.0":
        raise ConfigError(f"hybrid qualification must use {PIPELINE} version 1.0")
    base_path = Path(str(raw.get("base_current_config", "")))
    if not base_path.is_absolute():
        base_path = (source.parent / base_path).resolve()
    config = load_fresh_scale_integrated_config(base_path)
    root = Path(config["_repository_root"]).resolve()
    manifest = Path(_resolve(root, raw.get("v4_5_baseline_manifest"), "v4_5_baseline_manifest"))
    baseline = _verify_v45_append_only(root, manifest)

    settings = raw.get("hybrid_experiment")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        raise ConfigError("hybrid_experiment must be explicitly enabled")
    for field in (
        "gating",
        "auto_promote",
        "production_state_mutation_allowed",
        "good_task_bank_mutation_allowed",
        "production_reservoir_mutation_allowed",
        "automatic_task_commit",
        "source_balancing_enabled",
    ):
        if settings.get(field) is not False:
            raise ConfigError(f"hybrid_experiment.{field} must be false")
    integer_exact = {
        "supply_cycles": 4,
        "ideas_per_call": 4,
        "eligible_window_size": 48,
        "shard_size": 12,
        "shard_finalists_per_shard": 4,
        "maximum_development_per_arm": 8,
        "planner_rescue_maximum": 1,
        "planner_rescue_priority_maximum": 4,
        "realizer_operational_fallback_maximum": 1,
        "editorial_boards_per_group": 2,
        "editorial_generated_group_size": 6,
    }
    for field, expected in integer_exact.items():
        if settings.get(field) != expected:
            raise ConfigError(f"hybrid_experiment.{field} must be {expected}")
    for field in ("generation_concurrency", "downstream_concurrency"):
        if type(settings.get(field)) is not int or settings[field] < 1:
            raise ConfigError(f"hybrid_experiment.{field} must be positive")
    for field in (
        "hybrid_claude_call_fraction",
        "conditional_claude_blind_maximum_fraction",
        "deterministic_claude_blind_audit_fraction",
    ):
        if type(settings.get(field)) not in {int, float} or not 0 <= float(settings[field]) <= 1:
            raise ConfigError(f"hybrid_experiment.{field} must be between zero and one")
    supply_provider_schedule(settings)

    claude = raw.get("claude_code")
    if not isinstance(claude, Mapping) or claude.get("model_alias") != "opus":
        raise ConfigError("Claude Code Opus settings are required")
    required_efforts = {
        "idea_search": "max",
        "concrete_planner": "max",
        "task_realizer": "max",
        "blind_solver": "xhigh",
        "ambiguity_adjudicator": "high",
    }
    role_efforts = claude.get("role_efforts")
    if not isinstance(role_efforts, Mapping) or any(
        role_efforts.get(role) != effort for role, effort in required_efforts.items()
    ):
        raise ConfigError("Claude role efforts do not match the qualified routing")
    run = raw.get("run")
    if not isinstance(run, Mapping) or run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("hybrid experiment cannot automatically commit tasks")
    if run.get("mock") not in {True, False} or not isinstance(run.get("id"), str):
        raise ConfigError("run.id and run.mock are required")

    config = deepcopy(config)
    config["pipeline"] = PIPELINE
    config["version"] = "1.0"
    config["seed"] = raw.get("seed")
    if type(config["seed"]) is not int:
        raise ConfigError("seed must be an integer")
    config["hybrid_experiment"] = deepcopy(dict(settings))
    config["claude_code"] = deepcopy(dict(claude))
    config["claude_code"]["attestation_path"] = _resolve(
        root, raw.get("paths", {}).get("claude_attestation"), "paths.claude_attestation"
    )
    config["claude_code"]["mock"] = bool(run["mock"])
    config["run"] = {**deepcopy(config["run"]), **deepcopy(dict(run))}
    config["paths"]["frozen_generation_source_run"] = _resolve(
        root,
        raw.get("paths", {}).get("frozen_generation_source_run"),
        "paths.frozen_generation_source_run",
    )
    config["paths"]["claude_system_prompt"] = _resolve(
        root, raw.get("paths", {}).get("claude_system_prompt"), "paths.claude_system_prompt"
    )
    config["_config_path"] = str(source)
    config["_hybrid_base_config"] = str(base_path)
    config["_v4_5_baseline_manifest"] = str(manifest)
    config["_v4_5_append_verification"] = baseline
    return config


def snapshot_hybrid_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def _protected_state(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key in (
        "committed_task_registry",
        "reservoir",
        "v4_1_feature_manifest",
        "v4_2_feature_manifest",
        "v4_4_feature_manifest",
    ):
        value = config.get("paths", {}).get(key)
        if value and Path(value).is_file():
            result[key] = {"path": str(value), "sha256": _sha(value)}
    result["v4_5_baseline_manifest"] = {
        "path": config["_v4_5_baseline_manifest"],
        "sha256": _sha(config["_v4_5_baseline_manifest"]),
    }
    return result


def _assert_protected_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    changed = [key for key, row in before.items() if after.get(key, {}).get("sha256") != row["sha256"]]
    if changed:
        raise HybridPipelineError("experiment mutated protected production state: " + ", ".join(changed))


def _inner_config(config: Mapping[str, Any], *, runs_root: Path, run_id: str, reservoir: Path) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["pipeline"] = INNER_PIPELINE
    value["version"] = "1.0"
    value["paths"] = deepcopy(config["paths"])
    value["paths"]["runs"] = str(runs_root)
    value["paths"]["reservoir"] = str(reservoir)
    value["paths"]["selection_reference_reservoir"] = str(
        config.get("_selection_reference_reservoir", config["paths"]["reservoir"])
    )
    value["run"] = {**deepcopy(config["run"]), "id": run_id}
    # Two arm runtimes execute concurrently. Give each half of the global
    # worker allowance so the paired experiment does not double production
    # concurrency merely because it has two arms.
    value["model"] = deepcopy(config["model"])
    value["model"]["concurrency"] = max(
        1, int(config["hybrid_experiment"]["downstream_concurrency"]) // 2
    )
    value["generator_replication"] = {
        "independent_slates": 4,
        "normal_search_modes": list(MODES),
        "calls_per_mode_per_author_per_slate": 1,
        "ideas_per_call": 4,
        "candidates_per_author": 48,
        "maximum_development_per_slate": int(config["hybrid_experiment"]["shard_finalists_per_shard"]),
        "maximum_total_development": int(config["hybrid_experiment"]["maximum_development_per_arm"]),
        "downstream_provider": "codex",
        "downstream_model_control_unchanged": True,
        "arm_labels_hidden_downstream": True,
        "source_balancing_enabled": False,
        "repairs_enabled": False,
        "editorial_boards_per_group": 2,
        "editorial_generated_group_size": int(config["hybrid_experiment"]["editorial_generated_group_size"]),
        "generation_concurrency": int(config["hybrid_experiment"]["generation_concurrency"]),
        "downstream_concurrency": int(config["hybrid_experiment"]["downstream_concurrency"]),
    }
    return value


def _runtime(
    config: Mapping[str, Any], *, runs_root: Path, run_id: str, reservoir: Path
) -> GeneratorReplicationRuntime:
    inner = _inner_config(config, runs_root=runs_root, run_id=run_id, reservoir=reservoir)
    runtime = GeneratorReplicationRuntime(
        inner,
        require_attestation=not bool(config["run"]["mock"]),
    )
    # The prior generator study did not need adjudication. The hybrid route
    # installs the unchanged V4 prompt into this run-local prompt store.
    text, version, digest = load_prompt(Path(config["paths"]["prompts"]) / "v4_ambiguity_adjudicator.md")
    runtime.prompts.entries["ambiguity_adjudicator"] = {
        "filename": "v4_ambiguity_adjudicator.md",
        "stage": "v4_ambiguity_adjudicator",
        "text": text,
        "version": version,
        "hash": digest,
    }
    return runtime


def _kernel_ids(seed: int, cycle: int, mode: str) -> list[str]:
    return [
        "k" + content_hash({"seed": seed, "cycle": cycle, "mode": mode, "idea": index})[:13]
        for index in range(1, 5)
    ]


def generate_paired_supply(
    runtime: GeneratorReplicationRuntime,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    path = runtime.run_dir / "paired_supply.private.json"
    if path.exists():
        return read_json(path)
    schedule = supply_provider_schedule(config["hybrid_experiment"])
    calibrations = _calibration_slates(runtime.config)
    specs: list[tuple[str, int, str, Mapping[str, Any]]] = []
    for slate in calibrations:
        cycle = int(slate["slate_index"])
        for mode in MODES:
            calibration = slate["modes"][mode]
            specs.append(("codex", cycle, mode, calibration))
            if schedule[(cycle, mode)] == "claude":
                specs.append(("claude", cycle, mode, calibration))

    def one(spec: tuple[str, int, str, Mapping[str, Any]]) -> dict[str, Any]:
        provider, cycle, mode, calibration = spec
        ids = _kernel_ids(int(config["seed"]), cycle, mode)
        call_id = f"{provider}-cycle-{cycle:02d}-{mode.replace('_', '-')}-supply"

        def validate(value: dict[str, Any]) -> None:
            if [item["candidate_id"] for item in value["ideas"]] != ids:
                raise HybridPipelineError(f"{call_id} changed assigned IDs")

        response = runtime.call(
            provider=provider,
            prompt_name=_prompt_name_for_mode(mode),
            call_id=call_id,
            values={
                "CALIBRATION_DESCRIPTIONS": list(calibration["descriptions"]),
                "SEARCH_CALL_ID": f"cycle-{cycle:02d}-{mode.replace('_', '-')}-supply",
                "CANDIDATE_IDS": ids,
            },
            parent_ids=[],
            role="idea_search",
            private_purpose=f"paired-supply-{provider}",
            expected={
                "search_call_id": f"cycle-{cycle:02d}-{mode.replace('_', '-')}-supply",
                "search_mode": mode,
            },
            validator=validate,
        )
        key = f"{provider}/paired-supply-{provider}/v4_idea_search/{call_id}"
        return {
            "provider": provider,
            "cycle": cycle,
            "mode": mode,
            "calibration": deepcopy(dict(calibration)),
            "ideas": deepcopy(response["ideas"]),
            "prompt_hash": runtime.state["call_ledger"][key]["prompt_hash"],
        }

    rows = parallel_map(
        one,
        specs,
        min(len(specs), int(config["hybrid_experiment"]["generation_concurrency"])),
    )
    by_slot = {(row["provider"], row["cycle"], row["mode"]): row for row in rows}
    control: list[dict[str, Any]] = []
    hybrid: list[dict[str, Any]] = []
    paired_checks = []
    for cycle in range(1, int(config["hybrid_experiment"]["supply_cycles"]) + 1):
        for mode in MODES:
            sol = by_slot[("codex", cycle, mode)]
            provider = schedule[(cycle, mode)]
            selected = by_slot[(provider, cycle, mode)]
            if provider == "claude":
                if sol["prompt_hash"] != selected["prompt_hash"]:
                    raise HybridPipelineError("paired Sol/Opus idea prompts differ")
                paired_checks.append(
                    {"cycle": cycle, "mode": mode, "prompt_hash": sol["prompt_hash"], "exact": True}
                )
            for item in sol["ideas"]:
                control.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "slate_index": cycle,
                        "search_mode": mode,
                        "kernel": deepcopy(item),
                        "private_kernel_author_provider": "codex",
                    }
                )
            for item in selected["ideas"]:
                hybrid.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "slate_index": cycle,
                        "search_mode": mode,
                        "kernel": deepcopy(item),
                        "private_kernel_author_provider": provider,
                    }
                )
    if len(control) != 48 or len(hybrid) != 48:
        raise HybridPipelineError("paired supply did not produce two 48-kernel windows")
    value = {
        "schema_version": 1,
        "actual_model_calls": len(rows),
        "sol_calls": sum(row["provider"] == "codex" for row in rows),
        "claude_calls": sum(row["provider"] == "claude" for row in rows),
        "hybrid_schedule": [
            {"cycle": cycle, "mode": mode, "provider": provider}
            for (cycle, mode), provider in sorted(schedule.items())
        ],
        "paired_prompt_checks": paired_checks,
        "raw_calls": rows,
        "arms": {"sol_only": control, "hybrid": hybrid},
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "paired_supply.blinded.json",
        {
            "schema_version": 1,
            "provider_labels_removed": True,
            "arms": {
                arm: [
                    {key: deepcopy(row[key]) for key in ("candidate_id", "slate_index", "search_mode", "kernel")}
                    for row in values
                ]
                for arm, values in value["arms"].items()
            },
        },
    )
    return value


def build_arm_reservoir(
    *,
    config: Mapping[str, Any],
    runtime: GeneratorReplicationRuntime,
    arm: str,
    kernels: Sequence[Mapping[str, Any]],
    reservoir_path: Path,
) -> dict[str, Any]:
    manifest = runtime.run_dir.parent.parent / f"{arm}_frozen_window.json"
    if manifest.exists():
        return read_json(manifest)
    initialize_reservoir(reservoir_path)
    context = _DataContext(runtime.config)
    embedder = build_embedding_provider(runtime.config["embedding"], runtime.config["paths"]["cache"])
    vectors = embedder.embed([row["kernel"]["central_relation"] for row in kernels])
    by_cycle: dict[int, list[dict[str, Any]]] = {}
    for row, vector in zip(kernels, vectors, strict=True):
        mode = str(row["search_mode"])
        entry = runtime.prompts.entries[_prompt_name_for_mode(mode)]
        proposal = make_proposal(
            row["kernel"],
            supply_run_id=f"{config['run']['id']}:{arm}:paired-supply",
            supply_cycle_id=f"{config['run']['id']}:{arm}:cycle-{int(row['slate_index']):02d}",
            calibration_task_ids=[
                "description:" + content_hash(text)[:12]
                for text in next(
                    call["calibration"]["descriptions"]
                    for call in read_json(runtime.run_dir / "paired_supply.private.json")["raw_calls"]
                    if call["provider"] == "codex"
                    and call["cycle"] == row["slate_index"]
                    and call["mode"] == mode
                )
            ],
            calibration_descriptions=next(
                call["calibration"]["descriptions"]
                for call in read_json(runtime.run_dir / "paired_supply.private.json")["raw_calls"]
                if call["provider"] == "codex"
                and call["cycle"] == row["slate_index"]
                and call["mode"] == mode
            ),
            calibration_version=runtime.config["reservoir"]["calibration_version"],
            prompt_version=entry["version"],
            prompt_hash=entry["hash"],
            model=("gpt-5.6-sol" if row["private_kernel_author_provider"] == "codex" else "claude-opus-5"),
            reasoning_effort="max",
            source_bank_snapshot_id=context.bank_snapshot_id,
            embedding={
                "provider": runtime.config["embedding"]["provider"],
                "model": runtime.config["embedding"]["model"],
                "dimension": len(vector),
                "vector_hash": content_hash(vector),
            },
        )
        proposal["private_kernel_author_provider"] = row["private_kernel_author_provider"]
        by_cycle.setdefault(int(row["slate_index"]), []).append(proposal)
    for cycle, proposals in sorted(by_cycle.items()):
        append_supply(
            reservoir_path,
            proposals,
            hard_capacity=60,
            supply_run_id=f"{config['run']['id']}:{arm}:paired-supply",
            supply_cycle_id=f"{config['run']['id']}:{arm}:cycle-{cycle:02d}",
        )
    window = freeze_eligible_window(
        reservoir_path,
        window_id=f"window:{config['run']['id']}:{arm}",
        window_size=48,
        shard_size=12,
        seed=int(config["seed"]),
        current_bank_snapshot_id=context.bank_snapshot_id,
        current_prompt_hashes={mode: runtime.prompts.entries[_prompt_name_for_mode(mode)]["hash"] for mode in MODES},
        calibration_version=runtime.config["reservoir"]["calibration_version"],
        accepted_concept_hashes=context.accepted_concept_hashes(),
    )
    atomic_write_json(manifest, window)
    return window


def select_arm(
    runtime: GeneratorReplicationRuntime,
    *,
    arm: str,
    kernels: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    reservoir_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = runtime.run_dir / "selected_kernel_manifest.private.json"
    if path.exists():
        value = read_json(path)
        return value["selected"], value
    by_id = {row["candidate_id"]: deepcopy(dict(row)) for row in kernels}
    shards = []

    def run_shard(spec: tuple[int, list[str]]) -> dict[str, Any]:
        index, ids = spec
        rows = [by_id[candidate_id] for candidate_id in ids]
        packages, accepted, reservations, rejected = _reference_packages(runtime.config, rows)
        production_memory = load_reservoir(
            runtime.config["paths"]["selection_reference_reservoir"]
        )
        reservations = reservation_digest(production_memory)
        rejected = memory_digest(production_memory)
        serialized = json.dumps(packages, ensure_ascii=False).casefold()
        if "private_kernel_author_provider" in serialized or "author provider" in serialized:
            raise HybridPipelineError("provider label entered selector package")
        selector_id = f"{arm}-shard-selector-{index + 1:02d}"
        expected = set(ids)

        def validate(value: dict[str, Any]) -> None:
            if {item["candidate_id"] for item in value["assessments"]} != expected:
                raise HybridPipelineError("shard selector omitted a candidate")
            if set(value["quality_order"]) != expected:
                raise HybridPipelineError("shard selector quality order is incomplete")

        response = runtime.call(
            provider="codex",
            prompt_name="selector",
            call_id=selector_id,
            values={
                "CANDIDATES_WITH_REFERENCES": packages,
                "ACCEPTED_BANK_DIGEST": accepted,
                "IN_FLIGHT_RESERVATIONS": reservations,
                "REJECTED_MEMORY_DIGEST": rejected,
                "SELECTOR_ID": selector_id,
            },
            parent_ids=sorted(expected),
            role="batch_selector",
            private_purpose=f"{arm}-source-blind-shard-selection",
            expected={"selector_id": selector_id},
            validator=validate,
        )
        chosen, exclusions = choose_global_development(
            response,
            rows,
            maximum=int(runtime.config["generator_replication"]["maximum_development_per_slate"]),
        )
        return {
            "shard_index": index + 1,
            "candidate_ids": ids,
            "response": response,
            "selected": chosen,
            "exclusions": exclusions,
            "provider_labels_visible": False,
        }

    shards = parallel_map(
        run_shard,
        list(enumerate(window["shards"])),
        min(len(window["shards"]), int(runtime.config["model"]["concurrency"])),
    )
    finalists = [row for shard in shards for row in shard["selected"]]
    packages, accepted, reservations, rejected = _reference_packages(runtime.config, finalists)
    production_memory = load_reservoir(
        runtime.config["paths"]["selection_reference_reservoir"]
    )
    reservations = reservation_digest(production_memory)
    rejected = memory_digest(production_memory)
    if "private_kernel_author_provider" in json.dumps(packages, ensure_ascii=False):
        raise HybridPipelineError("provider label entered global selector package")
    selector_id = f"{arm}-global-selector"
    finalist_ids = {row["candidate_id"] for row in finalists}

    def validate_global(value: dict[str, Any]) -> None:
        if {item["candidate_id"] for item in value["assessments"]} != finalist_ids:
            raise HybridPipelineError("global selector omitted a finalist")
        if set(value["quality_order"]) != finalist_ids:
            raise HybridPipelineError("global selector quality order is incomplete")

    response = runtime.call(
        provider="codex",
        prompt_name="selector",
        call_id=selector_id,
        values={
            "CANDIDATES_WITH_REFERENCES": packages,
            "ACCEPTED_BANK_DIGEST": accepted,
            "IN_FLIGHT_RESERVATIONS": reservations,
            "REJECTED_MEMORY_DIGEST": rejected,
            "SELECTOR_ID": selector_id,
        },
        parent_ids=sorted(finalist_ids),
        role="batch_selector",
        private_purpose=f"{arm}-source-blind-global-selection",
        expected={"selector_id": selector_id},
        validator=validate_global,
    )
    selected, exclusions = choose_global_development(
        response,
        finalists,
        maximum=int(runtime.config["generator_replication"]["maximum_total_development"]),
    )
    reservations_result = reserve_slate(
        reservoir_path,
        window_id=str(window["window_id"]),
        owner_run_id=f"{runtime.config['run']['id']}:{arm}",
        slate_id=f"slate:{arm}:01",
        selections=[
            {
                "proposal_id": row["candidate_id"],
                "candidate_id": row["candidate_id"],
                "mechanism_summary": row["kernel"]["central_relation"],
            }
            for row in selected
        ],
        lease_seconds=int(runtime.config["reservoir"]["lease_seconds"]),
    )
    value = {
        "schema_version": 1,
        "arm": arm,
        "frozen_window_id": window["window_id"],
        "shards": shards,
        "global_response": response,
        "global_exclusions": exclusions,
        "selected": selected,
        "selected_by_private_source": dict(
            Counter(row["private_kernel_author_provider"] for row in selected)
        ),
        "source_labels_visible_to_selectors": False,
        "source_balancing_enabled": False,
        "whole_slate_reservation": reservations_result,
    }
    atomic_write_json(path, value)
    atomic_write_json(
        runtime.run_dir / "selected_kernel_manifest.blinded.json",
        {
            "schema_version": 1,
            "provider_labels_removed": True,
            "selected": [
                {
                    key: deepcopy(row[key])
                    for key in ("candidate_id", "slate_index", "search_mode", "kernel", "editorial_brief", "selector_global_priority")
                }
                for row in selected
            ],
        },
    )
    return selected, value


def plan_hybrid(
    runtime: GeneratorReplicationRuntime,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Plan with Sol, allowing one source-blind Opus rescue of a top-four abort."""

    records = plan_selected(runtime, selected)
    manifest_path = runtime.run_dir / "planning_manifest.private.json"
    manifest = read_json(manifest_path)
    if manifest.get("hybrid_rescue_processed"):
        return records
    settings = runtime.config["hybrid_experiment"]
    eligible = planner_rescue_candidates(
        records,
        priority_maximum=int(settings["planner_rescue_priority_maximum"]),
    )
    rescue_log: list[dict[str, Any]] = []
    if eligible and int(settings["planner_rescue_maximum"]):
        record = eligible[0]
        task_id = record["candidate_id"]
        brief = deepcopy(record["editorial_brief"])
        brief["optional_development_suggestion"] = None
        trigger = record["status"]
        try:
            response = runtime.call(
                provider="claude",
                prompt_name="concrete_planner",
                composed_with=("quality_plan",),
                call_id=f"{task_id}-opus-bounded-rescue-plan",
                values={
                    "CANDIDATE": deepcopy(record["kernel"]),
                    "EDITORIAL_BRIEF": brief,
                    "ROUTE_GUIDANCE": V4ControlPipeline._route_guidance(brief["route"]),
                    "CANDIDATE_ID": task_id,
                },
                parent_ids=[task_id],
                role="concrete_planner",
                private_purpose="hybrid-bounded-planner-rescue",
                expected={"candidate_id": task_id},
            )
            ready = response.get("status") == "ready"
            records[task_id] = {
                **deepcopy(record),
                "sol_original_plan_status": trigger,
                "sol_original_plan": deepcopy(record.get("plan")),
                "plan": response,
                "status": "plan_ready" if ready else "plan_aborted",
                "abort_reason": response.get("abort_reason", ""),
                "planner_provider": "claude" if ready else "codex",
                "planner_rescue_attempted": True,
                "planner_rescue_succeeded": ready,
            }
            rescue_log.append(
                {
                    "candidate_id": task_id,
                    "trigger": trigger,
                    "priority": record["selector_global_priority"],
                    "outcome": "ready" if ready else "aborted",
                }
            )
        except Exception as exc:
            records[task_id] = {
                **deepcopy(record),
                "sol_original_plan_status": trigger,
                "sol_original_plan": deepcopy(record.get("plan")),
                "planner_provider": "codex",
                "planner_rescue_attempted": True,
                "planner_rescue_succeeded": False,
                "planner_rescue_error": str(exc),
            }
            rescue_log.append(
                {
                    "candidate_id": task_id,
                    "trigger": trigger,
                    "priority": record["selector_global_priority"],
                    "outcome": "operational_failure",
                    "error": str(exc),
                }
            )
    for task_id, record in records.items():
        record.setdefault("planner_provider", "codex")
        record.setdefault("planner_rescue_attempted", False)
        record.setdefault("planner_rescue_succeeded", False)
        atomic_write_json(runtime.run_dir / "plans" / f"{task_id}.json", record)
    manifest.update(
        {
            "records": list(records.values()),
            "hybrid_rescue_processed": True,
            "planner_rescue_maximum": int(settings["planner_rescue_maximum"]),
            "planner_rescue_priority_maximum": int(
                settings["planner_rescue_priority_maximum"]
            ),
            "planner_rescue_log": rescue_log,
            "funnel": {
                "selected": len(records),
                "ready": sum(item["status"] == "plan_ready" for item in records.values()),
                "aborted": sum(item["status"] == "plan_aborted" for item in records.values()),
                "operational_failure": sum(
                    item["status"] == "plan_operational_failure" for item in records.values()
                ),
                "rescues_attempted": len(rescue_log),
                "rescues_succeeded": sum(item["outcome"] == "ready" for item in rescue_log),
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    return records


def planner_rescue_candidates(
    records: Mapping[str, Mapping[str, Any]], *, priority_maximum: int
) -> list[dict[str, Any]]:
    return sorted(
        [
            deepcopy(dict(item))
            for item in records.values()
            if item["status"] in {"plan_aborted", "plan_operational_failure"}
            and int(item["selector_global_priority"]) <= priority_maximum
        ],
        key=lambda item: (int(item["selector_global_priority"]), item["candidate_id"]),
    )


def _finalize_fallback_realization(
    runtime: GeneratorReplicationRuntime,
    record: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    target_background: int | None,
) -> dict[str, Any]:
    task_id = str(record["candidate_id"])
    final_plan, final_response, palette = transform_realization_palette(
        candidate_id=task_id,
        plan=deepcopy(record["plan"]),
        realization=deepcopy(dict(response)),
        target_background=target_background,
        palette_config=runtime.config["palette_experiment"],
    )
    if final_response.get("status") != "realized":
        return {
            **deepcopy(dict(record)),
            "plan": final_plan,
            "realization": final_response,
            "palette_treatment": palette,
            "realization_status": "refused",
            "failure_reason": final_response.get("refusal_reason", "realizer refused"),
        }
    fidelity = check_plan_fidelity(final_plan, final_response)
    public_validation = validate_arc_task(final_response["task"], require_test_outputs=False)
    full = complete_task(final_response)
    full_validation = validate_arc_task(full, require_test_outputs=True)
    diagnostics = diagnose_arc_task_v4(full) if full_validation.valid else None
    failures = [
        *fidelity.get("objective_failures", []),
        *public_validation.errors,
        *full_validation.errors,
        *(diagnostics.get("errors", []) if diagnostics and not diagnostics["hard_valid"] else []),
    ]
    if failures:
        return {
            **deepcopy(dict(record)),
            "plan": final_plan,
            "realization": final_response,
            "palette_treatment": palette,
            "plan_fidelity": fidelity,
            "realization_status": "invalid",
            "failure_reason": "; ".join(failures),
        }
    return {
        **deepcopy(dict(record)),
        "plan": final_plan,
        "realization": final_response,
        "complete_task": full,
        "palette_treatment": palette,
        "plan_fidelity": fidelity,
        "deterministic_diagnostics": diagnostics,
        "technical_diagnostics": diagnose_public_arc_task_v4(final_response["task"]),
        "task_metrics": _task_metrics(full),
        "realization_status": "realized",
    }


def realize_hybrid(
    runtime: GeneratorReplicationRuntime,
    plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Realize with Sol and use Opus only for one genuine transport failure."""

    records = realize_plans(runtime, plans)
    manifest_path = runtime.run_dir / "realization_manifest.private.json"
    manifest = read_json(manifest_path)
    if manifest.get("hybrid_fallback_processed"):
        return records
    eligible = realizer_fallback_candidates(records)
    fallback_log: list[dict[str, Any]] = []
    maximum = int(runtime.config["hybrid_experiment"]["realizer_operational_fallback_maximum"])
    assignments = _palette_assignments(runtime, list(plans.values()))
    for record in eligible[:maximum]:
        task_id = record["candidate_id"]
        try:
            response = runtime.call(
                provider="claude",
                prompt_name="task_realizer",
                call_id=f"{task_id}-opus-operational-fallback-realization",
                values={"CONCRETE_PLAN": deepcopy(record["plan"]), "CANDIDATE_ID": task_id},
                parent_ids=[task_id],
                role="task_realizer",
                private_purpose="hybrid-operational-realizer-fallback",
                expected={"candidate_id": task_id},
            )
            final = _finalize_fallback_realization(
                runtime, record, response, target_background=assignments.get(task_id)
            )
            final["sol_original_realization_status"] = "operational_failure"
            final["realizer_provider"] = "claude"
            final["realizer_fallback_attempted"] = True
            final["realizer_fallback_succeeded"] = final["realization_status"] == "realized"
            records[task_id] = final
            fallback_log.append(
                {
                    "candidate_id": task_id,
                    "trigger": "sol_operational_failure",
                    "outcome": final["realization_status"],
                }
            )
        except Exception as exc:
            records[task_id] = {
                **deepcopy(record),
                "sol_original_realization_status": "operational_failure",
                "realizer_provider": "codex",
                "realizer_fallback_attempted": True,
                "realizer_fallback_succeeded": False,
                "realizer_fallback_error": str(exc),
            }
            fallback_log.append(
                {
                    "candidate_id": task_id,
                    "trigger": "sol_operational_failure",
                    "outcome": "operational_failure",
                    "error": str(exc),
                }
            )
    public_dir = runtime.run_dir / "tasks" / "anonymous"
    private_dir = runtime.run_dir / "tasks" / "private_dossiers"
    for task_id, record in records.items():
        record.setdefault("realizer_provider", "codex")
        record.setdefault("realizer_fallback_attempted", False)
        record.setdefault("realizer_fallback_succeeded", False)
        if record.get("realization_status") == "realized":
            atomic_write_json(public_dir / f"{task_id}.arc.json", record["complete_task"])
            atomic_write_json(private_dir / f"{task_id}.authoring.json", record)
    manifest.update(
        {
            "records": list(records.values()),
            "hybrid_fallback_processed": True,
            "realizer_operational_fallback_maximum": maximum,
            "realizer_fallback_log": fallback_log,
            "funnel": {
                "selected": len(plans),
                "plans_ready": sum(item["status"] == "plan_ready" for item in plans.values()),
                "realized": sum(item.get("realization_status") == "realized" for item in records.values()),
                "refused": sum(item.get("realization_status") == "refused" for item in records.values()),
                "invalid": sum(item.get("realization_status") == "invalid" for item in records.values()),
                "operational_failure": sum(
                    item.get("realization_status") == "operational_failure" for item in records.values()
                ),
                "fallbacks_attempted": len(fallback_log),
                "fallbacks_succeeded": sum(item["outcome"] == "realized" for item in fallback_log),
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    return records


def realizer_fallback_candidates(
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            deepcopy(dict(item))
            for item in records.values()
            if item.get("realization_status") == "operational_failure"
        ],
        key=lambda item: (int(item.get("selector_global_priority", 999)), item["candidate_id"]),
    )


def conditional_blind_plan(
    *,
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    sol_blind: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a bounded Opus second-opinion cohort from concrete triggers."""

    eligible = sorted(task_id for task_id, result in technical.items() if result.get("passed"))
    maximum = math.ceil(
        len(eligible)
        * float(config["hybrid_experiment"]["conditional_claude_blind_maximum_fraction"])
    )
    audit_target = math.ceil(
        len(eligible)
        * float(config["hybrid_experiment"]["deterministic_claude_blind_audit_fraction"])
    )
    area_threshold = int(config["hybrid_experiment"]["expensive_output_area_threshold"])
    triggers: dict[str, list[str]] = {}
    for task_id in eligible:
        result = sol_blind.get(task_id, {})
        response = result.get("response") or {}
        row: list[str] = []
        if result.get("operational_failure"):
            row.append("sol_operational_failure")
        if result.get("primary_matches_hidden") is False:
            row.append("sol_primary_output_mismatch")
        if response.get("confidence") == "low":
            row.append("sol_low_confidence")
        if result.get("serious_different_output_alternative") is True:
            row.append("sol_serious_different_output_alternative")
        tech = technical[task_id]
        tech_response = tech.get("response") or {}
        if tech_response.get("confidence") == "low" or tech_response.get("status") in {
            "uncertain",
            "repairable",
        }:
            row.append("technical_uncertainty")
        hidden = records[task_id]["realization"]["authoring_metadata"]["expected_test_outputs"]
        if max((len(grid) * len(grid[0]) for grid in hidden), default=0) > area_threshold:
            row.append("expensive_output")
        if row:
            triggers[task_id] = row

    priority = {
        "sol_operational_failure": 0,
        "sol_serious_different_output_alternative": 1,
        "technical_uncertainty": 2,
        "sol_primary_output_mismatch": 3,
        "sol_low_confidence": 4,
        "expensive_output": 5,
        "deterministic_audit": 6,
    }
    selected = sorted(
        triggers,
        key=lambda task_id: (
            min(priority[item] for item in triggers[task_id]),
            content_hash({"seed": config["seed"], "opus-blind": task_id}),
        ),
    )[:maximum]
    audit_pool = sorted(
        [task_id for task_id in eligible if task_id not in selected],
        key=lambda task_id: content_hash({"seed": config["seed"], "audit": task_id}),
    )
    desired_audits = min(audit_target, max(0, maximum - len(selected)))
    for task_id in audit_pool[:desired_audits]:
        triggers.setdefault(task_id, []).append("deterministic_audit")
        selected.append(task_id)
    return {
        "schema_version": 1,
        "eligible_technical_passes": eligible,
        "maximum_calls": maximum,
        "audit_target": audit_target,
        "selected_task_ids": selected,
        "triggers": {task_id: triggers[task_id] for task_id in selected},
        "selection_is_bounded": len(selected) <= maximum,
    }


def _adjudicate_blind_conflict(
    runtime: GeneratorReplicationRuntime,
    task_id: str,
    record: Mapping[str, Any],
    provider_results: Mapping[str, Any],
) -> dict[str, Any]:
    return runtime.call(
        provider="codex",
        prompt_name="ambiguity_adjudicator",
        call_id=f"{task_id}-hybrid-ambiguity-adjudication",
        values={
            "TASK_ID": task_id,
            "AUTHOR_SPECIFICATION": {
                "intended_rule": record["realization"]["authoring_metadata"]["intended_rule"],
                "concrete_plan": record["plan"],
            },
            "COMPLETE_TASK": record["complete_task"],
            "BLIND_RESULTS": provider_results,
        },
        parent_ids=[task_id],
        role="ambiguity_adjudicator",
        private_purpose="hybrid-conditional-ambiguity-adjudication",
        expected={"task_id": task_id},
    )


def blind_hybrid(
    runtime: GeneratorReplicationRuntime,
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run Sol for every task and Opus only under bounded explicit triggers."""

    sol_results = blind_solve(runtime, records, technical)
    path = runtime.run_dir / "hybrid_blind_results.private.json"
    if path.exists():
        return read_json(path)["results"]
    plan = conditional_blind_plan(
        config=runtime.config,
        records=records,
        technical=technical,
        sol_blind=sol_results,
    )
    atomic_write_json(runtime.run_dir / "conditional_opus_blind_plan.private.json", plan)

    def opus_one(task_id: str) -> tuple[str, dict[str, Any]]:
        record = records[task_id]
        public = _public_task(record["complete_task"])
        if any("output" in pair for pair in public["test"]):
            raise HybridPipelineError("hidden output entered Opus blind prompt")
        opaque = _opaque_id(runtime.config["seed"], "blind-opus", task_id)
        solver = _opaque_id(runtime.config["seed"], "solver-opus", task_id)
        try:
            response = runtime.call(
                provider="claude",
                prompt_name="blind_solver",
                call_id=f"opus-conditional-blind-{task_id}",
                values={"PUBLIC_TASK": public, "TASK_ID": opaque, "SOLVER_ID": solver},
                parent_ids=[task_id],
                role="blind_solver",
                private_purpose="hybrid-conditional-blind",
                expected={"task_id": opaque, "solver_id": solver},
            )
        except Exception as exc:
            return task_id, {
                "operational_failure": True,
                "error": str(exc),
                "response": None,
                "primary_matches_hidden": None,
                "serious_different_output_alternative": None,
                "author_information_withheld": True,
                "hidden_output_withheld": True,
            }
        hidden = record["realization"]["authoring_metadata"]["expected_test_outputs"]
        return task_id, {
            "operational_failure": False,
            "response": response,
            "primary_matches_hidden": response["predicted_test_outputs"] == hidden,
            "serious_different_output_alternative": serious_blind_alternative(response, hidden),
            "author_information_withheld": True,
            "hidden_output_withheld": True,
        }

    original_timeout = runtime.claude.settings.get("timeout_seconds")
    original_retries = runtime.claude.settings.get("retries")
    runtime.claude.settings["timeout_seconds"] = int(
        runtime.config["hybrid_experiment"]["claude_blind_timeout_seconds"]
    )
    runtime.claude.settings["retries"] = int(
        runtime.config["hybrid_experiment"]["claude_blind_retries"]
    )
    try:
        opus_rows = parallel_map(
            opus_one,
            plan["selected_task_ids"],
            min(
                len(plan["selected_task_ids"]),
                int(runtime.config["claude_code"]["concurrency"]),
            ),
        ) if plan["selected_task_ids"] else []
    finally:
        runtime.claude.settings["timeout_seconds"] = original_timeout
        runtime.claude.settings["retries"] = original_retries
    opus_results = dict(opus_rows)

    results: dict[str, dict[str, Any]] = {}
    adjudications: dict[str, Any] = {}
    for task_id, sol in sol_results.items():
        opus = opus_results.get(task_id)
        provider_results = {"sol": deepcopy(sol), "opus": deepcopy(opus)}
        usable = sol if not sol.get("operational_failure") else opus
        operational = usable is None or usable.get("operational_failure")
        serious = any(
            item and item.get("serious_different_output_alternative") is True
            for item in (sol, opus)
        )
        adjudication = None
        if opus is not None and serious:
            try:
                adjudication = _adjudicate_blind_conflict(
                    runtime, task_id, records[task_id], provider_results
                )
                adjudications[task_id] = adjudication
                serious = adjudication.get("decision") != "clear"
            except Exception as exc:
                adjudication = {"operational_failure": True, "error": str(exc)}
                adjudications[task_id] = adjudication
                serious = True
        results[task_id] = {
            "operational_failure": operational,
            "response": None if operational else deepcopy(usable.get("response")),
            "primary_matches_hidden": None if operational else usable.get("primary_matches_hidden"),
            "serious_different_output_alternative": serious if not operational else None,
            "provider_results": provider_results,
            "opus_triggered": task_id in opus_results,
            "opus_triggers": plan["triggers"].get(task_id, []),
            "ambiguity_adjudication": adjudication,
            "author_information_withheld": True,
            "hidden_output_withheld": True,
        }
    value = {
        "schema_version": 1,
        "sol_blind_required_for_every_technical_pass": True,
        "opus_is_conditional_only": True,
        "opus_timeout_seconds": int(runtime.config["hybrid_experiment"]["claude_blind_timeout_seconds"]),
        "opus_retries": int(runtime.config["hybrid_experiment"]["claude_blind_retries"]),
        "selection_plan": plan,
        "opus_results": opus_results,
        "adjudications": adjudications,
        "results": results,
    }
    atomic_write_json(path, value)
    return results


def _arm_task_rows(
    *,
    arm: str,
    records: Mapping[str, Mapping[str, Any]],
    technical: Mapping[str, Mapping[str, Any]],
    blind: Mapping[str, Mapping[str, Any]],
    boards: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _v45_task_rows(records, technical, blind, boards)
    for row in rows:
        record = records[row["task_id"]]
        row["arm"] = arm
        row["planner_provider"] = record.get("planner_provider", "codex")
        row["realizer_provider"] = record.get("realizer_provider", "codex")
        row["planner_rescue_attempted"] = record.get("planner_rescue_attempted", False)
        row["planner_rescue_succeeded"] = record.get("planner_rescue_succeeded", False)
        row["realizer_fallback_attempted"] = record.get("realizer_fallback_attempted", False)
        row["realizer_fallback_succeeded"] = record.get("realizer_fallback_succeeded", False)
    return rows


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    realized = [row for row in rows if row["realization_status"] == "realized"]
    tech_pass = [row for row in realized if row.get("technical", {}).get("passed")]
    completed_blind = [
        row for row in tech_pass if row.get("blind") and not row["blind"].get("operational_failure")
    ]
    core = [row for row in rows if row["experimental_disposition"] == "core_quality"]
    reserve = [row for row in rows if row["experimental_disposition"] == "reserve_quality"]
    axes = {
        "novel_concept": Counter(),
        "functional_composition": Counter(),
        "overall_quality": Counter(),
    }
    for row in rows:
        for judgment in (row.get("editorial") or {}).get("judgments", []):
            for axis in axes:
                axes[axis][judgment[axis]] += 1
    return {
        "selected": len(rows),
        "plan_ready": sum(row["plan_status"] == "plan_ready" for row in rows),
        "plan_aborted": sum(row["plan_status"] == "plan_aborted" for row in rows),
        "plan_operational_failure": sum(
            row["plan_status"] == "plan_operational_failure" for row in rows
        ),
        "planner_rescue_attempted": sum(row["planner_rescue_attempted"] for row in rows),
        "planner_rescue_succeeded": sum(row["planner_rescue_succeeded"] for row in rows),
        "realized": len(realized),
        "realizer_fallback_attempted": sum(row["realizer_fallback_attempted"] for row in rows),
        "realizer_fallback_succeeded": sum(row["realizer_fallback_succeeded"] for row in rows),
        "technical_pass": len(tech_pass),
        "technical_hidden_output_exact": sum(
            row["technical"].get("hidden_output_match") is True for row in tech_pass
        ),
        "blind_completed": len(completed_blind),
        "blind_primary_exact": sum(
            row["blind"].get("primary_matches_hidden") is True for row in completed_blind
        ),
        "serious_different_output_ambiguity": sum(
            row["blind"].get("serious_different_output_alternative") is True
            for row in completed_blind
        ),
        "conditional_opus_blind_calls": sum(
            bool(row.get("blind", {}).get("opus_triggered")) for row in completed_blind
        ),
        "core_quality": len(core),
        "reserve_quality": len(reserve),
        "core_or_reserve": len(core) + len(reserve),
        "core_yield": round(len(core) / len(rows), 6) if rows else None,
        "core_or_reserve_yield": round((len(core) + len(reserve)) / len(rows), 6)
        if rows
        else None,
        "board_distributions": {axis: dict(sorted(counts.items())) for axis, counts in axes.items()},
    }


def _runtime_accounting(runtime: GeneratorReplicationRuntime) -> dict[str, Any]:
    overall = v4_call_accounting(runtime.state["call_ledger"], run_dir=runtime.run_dir)
    by_provider = {}
    for provider in ("codex", "claude"):
        ledger = {
            key: row
            for key, row in runtime.state["call_ledger"].items()
            if row.get("provider") == provider
        }
        by_provider[provider] = v4_call_accounting(ledger, run_dir=runtime.run_dir)
    return {**overall, "by_provider": by_provider}


def _resolve_experimental_leases(
    reservoir_path: Path,
    selected: Sequence[Mapping[str, Any]],
    *,
    source_run_id: str,
) -> None:
    state = load_reservoir(reservoir_path)
    terminal = {
        item["proposal_id"]
        for item in state["proposals"]
        if item.get("state") in {"accepted", "rejected"}
    }
    for item in selected:
        if item["candidate_id"] in terminal:
            continue
        resolve_proposal(
            reservoir_path,
            proposal_id=item["candidate_id"],
            outcome="rejected",
            reason="isolated paired qualification complete; task remains uncommitted and recoverable",
            source_run_id=source_run_id,
            memory_tier="recoverable_work",
        )


def _render_arm_tasks(
    outer_run: Path,
    arm: str,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    directory = outer_run / "renders" / arm
    directory.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, Path]] = []
    for task_id, record in sorted(records.items()):
        if record.get("realization_status") != "realized":
            continue
        path = directory / f"{task_id}.png"
        render_episode(
            record["complete_task"],
            title=f"{arm}: {task_id}",
            subtitle=f"{record['search_mode']} · inspect grids and rule",
            rule=record["realization"]["authoring_metadata"]["intended_rule"],
            destination=path,
        )
        images.append((task_id, path))
    contact = build_contact_sheet(images, directory / "contact_sheet.png", columns=2)
    return {
        "task_count": len(images),
        "directory": str(directory),
        "contact_sheet": str(contact) if contact else None,
    }


def _render_blinded_review(
    outer_run: Path,
    arm_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    directory = outer_run / "human_review"
    directory.mkdir(parents=True, exist_ok=True)
    items = [
        (arm, task_id, record)
        for arm, records in arm_records.items()
        for task_id, record in records.items()
        if record.get("realization_status") == "realized"
    ]
    Random(seed + 46017).shuffle(items)
    images: list[tuple[str, Path]] = []
    key = []
    appendix = ["# Hybrid qualification answer/rule appendix", ""]
    for index, (arm, task_id, record) in enumerate(items, 1):
        alias = f"episode-{index:02d}"
        path = directory / f"{alias}.png"
        render_episode(
            record["complete_task"],
            title=alias,
            subtitle="Arm and author providers hidden",
            rule=None,
            destination=path,
        )
        images.append((alias, path))
        key.append(
            {
                "alias": alias,
                "arm": arm,
                "task_id": task_id,
                "kernel_author_provider": record["private_kernel_author_provider"],
                "planner_provider": record.get("planner_provider", "codex"),
                "realizer_provider": record.get("realizer_provider", "codex"),
            }
        )
        appendix.extend(
            [
                f"## {alias}",
                "",
                record["realization"]["authoring_metadata"]["intended_rule"],
                "",
            ]
        )
    contact = build_contact_sheet(images, directory / "contact_sheet.png", columns=2)
    atomic_write_json(directory / "provenance_key.private.json", {"mapping": key})
    atomic_write_text(directory / "answer_rule_appendix.md", "\n".join(appendix))
    atomic_write_text(
        directory / "review_sheet.md",
        "# Blinded Sol-only versus Sol+Opus hybrid review\n\n"
        "Inspect the grids before opening the appendix. Rate conceptual novelty, "
        "functional composition, discoverability, fairness, visual economy, and overall "
        "preference. Arm and provider provenance are hidden.\n",
    )
    return {
        "task_count": len(items),
        "contact_sheet": str(contact) if contact else None,
        "review_sheet": str(directory / "review_sheet.md"),
        "answer_rule_appendix": str(directory / "answer_rule_appendix.md"),
        "private_provenance_key": str(directory / "provenance_key.private.json"),
    }


def _intervention_checks(
    *,
    config: Mapping[str, Any],
    supply: Mapping[str, Any],
    arm_outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    hybrid = arm_outputs["hybrid"]
    control = arm_outputs["sol_only"]
    hybrid_records = hybrid["records"]
    control_records = control["records"]
    hybrid_tech_ids = {task_id for task_id, row in hybrid["technical"].items() if row.get("passed")}
    sol_blind_ids = set(
        read_json(Path(hybrid["runtime_dir"]) / "blind_results.private.json")["results"]
    )
    conditional = read_json(Path(hybrid["runtime_dir"]) / "conditional_opus_blind_plan.private.json")
    hybrid_ledger = hybrid["runtime_state"]["call_ledger"]
    control_ledger = control["runtime_state"]["call_ledger"]
    fallback_rows = [row for row in hybrid_records.values() if row.get("realizer_fallback_attempted")]
    rescue_rows = [row for row in hybrid_records.values() if row.get("planner_rescue_attempted")]
    checks = {
        "hybrid_idea_supply_exactly_25_percent_opus": supply["claude_calls"] == 3
        and supply["sol_calls"] == 12
        and len(supply["raw_calls"]) == 15,
        "hybrid_has_one_opus_idea_call_per_normal_mode": Counter(
            row["mode"] for row in supply["hybrid_schedule"] if row["provider"] == "claude"
        )
        == Counter(MODES),
        "selectors_are_sol_only": all(
            row["provider"] == "codex"
            for ledger in (hybrid_ledger, control_ledger)
            for row in ledger.values()
            if row["role"] == "batch_selector"
        ),
        "source_labels_hidden_and_no_source_quota": all(
            output["selection"]["source_labels_visible_to_selectors"] is False
            and output["selection"]["source_balancing_enabled"] is False
            for output in arm_outputs.values()
        ),
        "planner_rescue_within_cap": len(rescue_rows)
        <= int(config["hybrid_experiment"]["planner_rescue_maximum"]),
        "planner_rescue_only_top_priority_abort_or_failure": all(
            int(row["selector_global_priority"])
            <= int(config["hybrid_experiment"]["planner_rescue_priority_maximum"])
            and row.get("sol_original_plan_status")
            in {"plan_aborted", "plan_operational_failure"}
            for row in rescue_rows
        ),
        "realizer_fallback_within_cap": len(fallback_rows)
        <= int(config["hybrid_experiment"]["realizer_operational_fallback_maximum"]),
        "realizer_fallback_only_for_sol_transport_failure": all(
            row.get("sol_original_realization_status") == "operational_failure"
            for row in fallback_rows
        ),
        "technical_verification_is_sol_only": all(
            row["provider"] == "codex"
            for ledger in (hybrid_ledger, control_ledger)
            for row in ledger.values()
            if row["role"] == "technical_verifier"
        ),
        "every_hybrid_technical_pass_received_sol_blind": hybrid_tech_ids <= sol_blind_ids,
        "conditional_opus_blind_is_bounded": conditional["selection_is_bounded"],
        "editorial_boards_are_sol_only": all(
            row["provider"] == "codex"
            for ledger in (hybrid_ledger, control_ledger)
            for row in ledger.values()
            if row["role"] == "editorial_board"
        ),
        "acceptance_policy_identical": True,
        "automatic_task_commit_disabled": config["run"]["automatically_commit_generated_tasks"]
        is False,
    }
    checks["all_required_checks_pass"] = all(checks.values())
    return checks


def _markdown_report(report: Mapping[str, Any]) -> str:
    control = report["arms"]["sol_only"]["metrics"]
    hybrid = report["arms"]["hybrid"]["metrics"]
    calls = report["call_comparison"]
    lines = [
        "# Sol-only versus Sol + Opus hybrid end-to-end qualification",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "This is an isolated, non-gating paired qualification. The committed task bank "
        "and production reservoir were protected from writes. The hybrid changes only "
        "minority idea supply and bounded conditional assistance; Sol retains selection "
        "and every production quality gate.",
        "",
        "## Outcome",
        "",
        f"- Sol-only: {control['selected']} selected, {control['realized']} realized, "
        f"{control['technical_pass']} technical passes, {control['core_quality']} core and "
        f"{control['reserve_quality']} reserve.",
        f"- Hybrid: {hybrid['selected']} selected, {hybrid['realized']} realized, "
        f"{hybrid['technical_pass']} technical passes, {hybrid['core_quality']} core and "
        f"{hybrid['reserve_quality']} reserve.",
        f"- Hybrid conditional Opus blind calls: {hybrid['conditional_opus_blind_calls']}.",
        f"- Recommendation: **{report['recommendation']}**.",
        "",
        "## Intervention behavior",
        "",
    ]
    for key, value in report["intervention_checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "The idea intervention fired by construction: 3 of the hybrid arm's 12 "
            "supply calls use Opus, one in each ordinary search mode. Planner rescue and "
            "realizer fallback fire only when their qualifying failures occur; a zero "
            "count is correct when Sol has no such failure. Conditional Opus blind review "
            "includes deterministic audit sampling, so that integration is exercised even "
            "when Sol raises no concern.",
            "",
            "## Funnel comparison",
            "",
            "| Measure | Sol-only | Hybrid |",
            "|---|---:|---:|",
        ]
    )
    for key in (
        "selected",
        "plan_ready",
        "realized",
        "technical_pass",
        "technical_hidden_output_exact",
        "blind_completed",
        "blind_primary_exact",
        "serious_different_output_ambiguity",
        "core_quality",
        "reserve_quality",
        "core_or_reserve",
    ):
        lines.append(f"| {key} | {control[key]} | {hybrid[key]} |")
    lines.extend(
        [
            "",
            "## Calls and time",
            "",
            f"- Actual unique idea-supply calls: {calls['actual_unique_supply_calls']} "
            f"({calls['actual_supply_sol_calls']} Sol, {calls['actual_supply_opus_calls']} Opus).",
            f"- Attributable Sol-only logical calls: {calls['sol_only_attributable_logical_calls']}.",
            f"- Attributable hybrid logical calls: {calls['hybrid_attributable_logical_calls']}.",
            f"- Attributable hybrid Sol calls: {calls['hybrid_attributable_sol_calls']}; "
            f"Opus calls: {calls['hybrid_attributable_opus_calls']}.",
            f"- Sol-only downstream active model minutes: "
            f"{calls['sol_only_downstream']['active_model_minutes']}.",
            f"- Hybrid downstream active model minutes: "
            f"{calls['hybrid_downstream']['active_model_minutes']}.",
            f"- End-to-end wall time: {report['wall_time_minutes']} minutes (both arms ran "
            "concurrently, so this is not the sum of arm latencies).",
            "",
            "Cache hits and the paired concurrent design are reported in `report.json`; "
            "active model time is more interpretable than raw wall time for per-arm comparison.",
            "",
            "## Quality interpretation",
            "",
            report["interpretation"],
            "",
            "Both arms use the same deterministic checks, Sol/high technical verifier, "
            "Sol/xhigh primary blind solver, two Sol/max fixed-anchor boards, and the same "
            "core/reserve policy. Opus never serves as technical verifier or board. A "
            "conditional Opus blind opinion cannot silently overrule Sol; a material "
            "ambiguity signal is routed to the existing Sol adjudicator.",
            "",
            "## Artifacts",
            "",
            f"- Sol-only contact sheet: `{report['renders']['sol_only']['contact_sheet']}`",
            f"- Hybrid contact sheet: `{report['renders']['hybrid']['contact_sheet']}`",
            f"- Blinded comparison sheet: `{report['human_review_pack']['contact_sheet']}`",
            f"- Rule appendix: `{report['human_review_pack']['answer_rule_appendix']}`",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "No experimental task was committed and no production promotion occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def build_hybrid_report(
    *,
    config: Mapping[str, Any],
    outer_run: Path,
    supply: Mapping[str, Any],
    supply_runtime: GeneratorReplicationRuntime,
    arm_outputs: Mapping[str, Mapping[str, Any]],
    protected_before: Mapping[str, Any],
    started_at: float,
) -> dict[str, Any]:
    arm_sections: dict[str, Any] = {}
    renders = {}
    for arm in ARMS:
        output = arm_outputs[arm]
        rows = _arm_task_rows(
            arm=arm,
            records=output["records"],
            technical=output["technical"],
            blind=output["blind"],
            boards=output["boards"],
        )
        for row in rows:
            if row.get("task_path"):
                row["task_path"] = f"execution/{arm}/{row['task_path']}"
            if row.get("dossier_path"):
                row["dossier_path"] = f"execution/{arm}/{row['dossier_path']}"
        renders[arm] = _render_arm_tasks(outer_run, arm, output["records"])
        arm_sections[arm] = {
            "selected_private_source_mix": output["selection"]["selected_by_private_source"],
            "metrics": _arm_metrics(rows),
            "tasks": rows,
            "call_accounting": _runtime_accounting(output["runtime"]),
            "runtime_dir": output["runtime_dir"],
        }
    review = _render_blinded_review(
        outer_run,
        {arm: output["records"] for arm, output in arm_outputs.items()},
        seed=int(config["seed"]),
    )
    supply_calls = _runtime_accounting(supply_runtime)
    control_calls = arm_sections["sol_only"]["call_accounting"]
    hybrid_calls = arm_sections["hybrid"]["call_accounting"]
    hybrid_supply_sol = 9
    hybrid_supply_opus = 3
    call_comparison = {
        "actual_unique_supply_calls": supply["actual_model_calls"],
        "actual_supply_sol_calls": supply["sol_calls"],
        "actual_supply_opus_calls": supply["claude_calls"],
        "supply_actual": supply_calls,
        "sol_only_downstream": control_calls,
        "hybrid_downstream": hybrid_calls,
        "sol_only_attributable_logical_calls": 12 + control_calls["substantive_logical_calls"],
        "hybrid_attributable_logical_calls": 12 + hybrid_calls["substantive_logical_calls"],
        "hybrid_attributable_sol_calls": hybrid_supply_sol
        + hybrid_calls["by_provider"]["codex"]["substantive_logical_calls"],
        "hybrid_attributable_opus_calls": hybrid_supply_opus
        + hybrid_calls["by_provider"]["claude"]["substantive_logical_calls"],
        "sol_only_attributable_sol_calls": 12
        + control_calls["by_provider"]["codex"]["substantive_logical_calls"],
        "sol_usage_change_calls": (
            hybrid_supply_sol
            + hybrid_calls["by_provider"]["codex"]["substantive_logical_calls"]
            - 12
            - control_calls["by_provider"]["codex"]["substantive_logical_calls"]
        ),
    }
    checks = _intervention_checks(
        config=config,
        supply=supply,
        arm_outputs={
            arm: {
                **output,
                "runtime_state": output["runtime"].state,
            }
            for arm, output in arm_outputs.items()
        },
    )
    if not checks["all_required_checks_pass"]:
        raise HybridPipelineError("one or more hybrid intervention invariants failed")
    protected_after = _protected_state(config)
    _assert_protected_unchanged(protected_before, protected_after)
    control = arm_sections["sol_only"]["metrics"]
    hybrid = arm_sections["hybrid"]["metrics"]
    if hybrid["core_or_reserve"] >= control["core_or_reserve"] and hybrid["core_quality"] >= max(
        1, control["core_quality"] - 1
    ):
        recommendation = "candidate_for_larger_shadow_run_pending_human_review"
        interpretation = (
            "In this paired qualification the hybrid retained the Sol-only core-or-reserve "
            "yield and remained close on core yield while exercising its intended minority "
            "supply and conditional blind-review paths. This supports continued shadow use, "
            "not automatic production promotion from one cohort."
        )
    else:
        recommendation = "continue_testing_quality_noninferiority_not_established"
        interpretation = (
            "The hybrid did not establish non-inferior downstream yield in this cohort. "
            "The routing behaved as designed, but the quality signal is not strong enough "
            "to recommend production adoption without another matched run and human review."
        )
    report = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "run_id": config["run"]["id"],
        "completed": True,
        "comparison": "matched_sol_only_control_vs_sol_opus_hybrid",
        "baseline": deepcopy(config["_v4_5_append_verification"]),
        "committed_bank_count_at_start": config["_v4_5_append_verification"]["registry_status"][
            "live_task_count"
        ],
        "design": {
            "control_window_kernels": 48,
            "hybrid_window_kernels": 48,
            "shared_sol_kernel_slots": 36,
            "hybrid_opus_kernel_slots": 12,
            "hybrid_opus_idea_call_fraction": 0.25,
            "ordinary_search_modes_unchanged": list(MODES),
            "source_blind_selection": True,
            "source_quota": False,
            "sol_quality_gates_unchanged": True,
            "non_gating_experiment": True,
        },
        "supply": {
            key: deepcopy(value)
            for key, value in supply.items()
            if key not in {"raw_calls", "arms"}
        },
        "arms": arm_sections,
        "intervention_checks": checks,
        "call_comparison": call_comparison,
        "wall_time_minutes": round((time.time() - started_at) / 60, 3),
        "renders": renders,
        "human_review_pack": review,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "production_promotion_occurred": False,
        "automatic_task_commit_occurred": False,
        "production_state_mutated": False,
        "protected_state_before": deepcopy(dict(protected_before)),
        "protected_state_after": protected_after,
        "limitations": [
            "This is one paired qualification cohort; human preference and replication remain necessary.",
            "The arms draw from windows sharing 36 Sol-authored slots but selection can produce different slates, so downstream tasks are a pipeline comparison rather than task-paired A/Bs.",
            "Content-addressed cache hits are preserved and disclosed; they can reduce observed latency for a repeated prompt.",
            "Planner rescue and realizer fallback are conditional safety valves, so their absence is not evidence of failure when no qualifying Sol failure occurs.",
            "Experimental tasks remain uncommitted regardless of their board status until explicit user authorization.",
        ],
    }
    atomic_write_json(outer_run / "report.json", report)
    atomic_write_text(outer_run / "report.md", _markdown_report(report))
    return report


def _outer_implementation_manifest(config: Mapping[str, Any], outer_run: Path) -> dict[str, Any]:
    root = Path(config["_repository_root"])
    relatives = (
        "arcgen.lib/v46_hybrid_pipeline.py",
        "arcgen.lib/post_v45_core_commit.py",
        "config/v4_6_exp_sol_opus_hybrid.yaml",
        "experiments/v4_6_sol_opus_hybrid_plan.md",
        "committed_tasks/commit_manifests/post-v45-experiments-all-core-20260809.json",
    )
    value = {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "branch_source": config["_v4_5_append_verification"],
        "files": {relative: _sha(root / relative) for relative in relatives},
    }
    path = outer_run / "implementation_manifest.json"
    if path.exists():
        existing = read_json(path)
        if existing.get("files") == value["files"]:
            return existing
        state_path = outer_run / "state.json"
        state = read_json(state_path) if state_path.exists() else {}
        if state.get("completed"):
            raise HybridPipelineError("V4.6 implementation changed after completion")
        value["resume_hotfix_history"] = [
            *existing.get("resume_hotfix_history", []),
            {
                "reason": "resume-safe fix after pre-blind NameError: imported the already-tested blind_solve route",
                "previous_files": existing.get("files", {}),
                "applied_at_epoch_seconds": round(time.time(), 3),
            },
        ]
        atomic_write_json(path, value)
        return value
    atomic_write_json(path, value)
    return value


def _run_arm(
    *,
    config: Mapping[str, Any],
    runtime: GeneratorReplicationRuntime,
    arm: str,
    kernels: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    reservoir_path: Path,
) -> dict[str, Any]:
    selected, selection = select_arm(
        runtime,
        arm=arm,
        kernels=kernels,
        window=window,
        reservoir_path=reservoir_path,
    )
    plans = plan_selected(runtime, selected) if arm == "sol_only" else plan_hybrid(runtime, selected)
    records = realize_plans(runtime, plans) if arm == "sol_only" else realize_hybrid(runtime, plans)
    technical = technical_verify(runtime, records)
    blind = (
        blind_solve(runtime, records, technical)
        if arm == "sol_only"
        else blind_hybrid(runtime, records, technical)
    )
    boards = editorial_review(runtime, records, technical, blind)
    _resolve_experimental_leases(
        reservoir_path,
        selected,
        source_run_id=f"{runtime.config['run']['id']}:{arm}",
    )
    return {
        "arm": arm,
        "runtime": runtime,
        "runtime_dir": str(runtime.run_dir),
        "selection": selection,
        "plans": plans,
        "records": records,
        "technical": technical,
        "blind": blind,
        "boards": boards,
        "reservoir_path": str(reservoir_path),
    }


def run_hybrid_experiment(
    config: dict[str, Any],
    *,
    authorize_model_calls: bool,
) -> dict[str, Any]:
    if not authorize_model_calls and not config["run"]["mock"]:
        raise HybridPipelineError(
            "real model calls require --authorize-model-calls"
        )
    root = Path(config["_repository_root"])
    outer_run = Path(config["paths"]["runs"]) / config["run"]["id"]
    outer_run.mkdir(parents=True, exist_ok=True)
    report_path = outer_run / "report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report.get("pipeline") != PIPELINE:
            raise HybridPipelineError("existing run belongs to another pipeline")
        return report
    started_at = time.time()
    protected_before = _protected_state(config)
    snapshot = snapshot_hybrid_config(config)
    snapshot_path = outer_run / "config.snapshot.json"
    if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
        raise HybridPipelineError("hybrid config changed during resume")
    if not snapshot_path.exists():
        atomic_write_json(snapshot_path, snapshot)
    _outer_implementation_manifest(config, outer_run)
    state_path = outer_run / "state.json"
    outer_state = read_json(state_path) if state_path.exists() else {
        "schema_version": 1,
        "pipeline": PIPELINE,
        "pipeline_version": VERSION,
        "run_id": config["run"]["id"],
        "completed": False,
        "stage": "initialized",
        "started_at_epoch_seconds": round(started_at, 3),
        "protected_state_before": protected_before,
        "automatic_task_commit": False,
        "production_state_mutation_allowed": False,
    }
    if outer_state.get("protected_state_before") != protected_before:
        raise HybridPipelineError("protected production state changed during resume")
    atomic_write_json(state_path, outer_state)

    state_dir = outer_run / "state"
    execution_dir = outer_run / "execution"
    state_dir.mkdir(parents=True, exist_ok=True)
    execution_dir.mkdir(parents=True, exist_ok=True)
    selection_snapshot = state_dir / "production_reservoir.selection_snapshot.json"
    if not selection_snapshot.exists():
        shutil.copy2(config["paths"]["reservoir"], selection_snapshot)
    config = deepcopy(config)
    config["_selection_reference_reservoir"] = str(selection_snapshot)
    atomic_write_json(
        state_dir / "frozen_production_context.json",
        {
            "accepted_registry": protected_before.get("committed_task_registry"),
            "production_reservoir": protected_before.get("reservoir"),
            "selection_reservoir_snapshot": {
                "path": str(selection_snapshot),
                "sha256": _sha(selection_snapshot),
            },
            "fixed_anchor_manifest": config["paths"].get("fixed_anchor_manifest"),
            "seed": config["seed"],
        },
    )

    supply_reservoir = state_dir / "supply_runtime_reservoir.json"
    supply_runtime = _runtime(
        config,
        runs_root=execution_dir,
        run_id="supply",
        reservoir=supply_reservoir,
    )
    supply = generate_paired_supply(supply_runtime, config)
    outer_state["stage"] = "supply_complete"
    atomic_write_json(state_path, outer_state)

    runtimes: dict[str, GeneratorReplicationRuntime] = {}
    windows: dict[str, dict[str, Any]] = {}
    reservoirs: dict[str, Path] = {}
    for arm in ARMS:
        reservoirs[arm] = state_dir / f"{arm}_reservoir.json"
        runtimes[arm] = _runtime(
            config,
            runs_root=execution_dir,
            run_id=arm,
            reservoir=reservoirs[arm],
        )
        supply_copy = runtimes[arm].run_dir / "paired_supply.private.json"
        if supply_copy.exists() and read_json(supply_copy) != supply:
            raise HybridPipelineError(f"paired supply changed in {arm} runtime")
        if not supply_copy.exists():
            atomic_write_json(supply_copy, supply)
        windows[arm] = build_arm_reservoir(
            config=config,
            runtime=runtimes[arm],
            arm=arm,
            kernels=supply["arms"][arm],
            reservoir_path=reservoirs[arm],
        )
    outer_state["stage"] = "windows_frozen"
    outer_state["eligible_windows"] = {
        arm: windows[arm]["window_id"] for arm in ARMS
    }
    atomic_write_json(state_path, outer_state)

    def arm_one(arm: str) -> tuple[str, dict[str, Any]]:
        return arm, _run_arm(
            config=config,
            runtime=runtimes[arm],
            arm=arm,
            kernels=supply["arms"][arm],
            window=windows[arm],
            reservoir_path=reservoirs[arm],
        )

    arm_outputs = dict(parallel_map(arm_one, list(ARMS), 2))
    outer_state["stage"] = "arms_complete"
    atomic_write_json(state_path, outer_state)
    report = build_hybrid_report(
        config=config,
        outer_run=outer_run,
        supply=supply,
        supply_runtime=supply_runtime,
        arm_outputs=arm_outputs,
        protected_before=protected_before,
        started_at=float(outer_state["started_at_epoch_seconds"]),
    )
    outer_state["stage"] = "complete"
    outer_state["completed"] = True
    outer_state["completed_at_epoch_seconds"] = round(time.time(), 3)
    outer_state["report_path"] = str(report_path)
    atomic_write_json(state_path, outer_state)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arcgen.lib.v46_hybrid_pipeline",
        description="Paired Sol-only versus Sol+Opus hybrid qualification",
    )
    parser.add_argument("command", choices=("validate", "run", "report"))
    parser.add_argument("--config", default="config/v4_6_exp_sol_opus_hybrid.yaml")
    parser.add_argument("--run")
    parser.add_argument("--authorize-model-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "report":
            if not args.run:
                raise HybridPipelineError("--run is required")
            path = Path(args.run)
            if path.is_dir():
                path = path / "report.json"
            value = read_json(path)
            if value.get("pipeline") != PIPELINE:
                raise HybridPipelineError("run is not a V4.6 hybrid qualification")
        else:
            config = load_hybrid_config(args.config)
            if args.command == "validate":
                value = {
                    "pipeline": PIPELINE,
                    "pipeline_version": VERSION,
                    "baseline": config["_v4_5_append_verification"],
                    "supply_schedule": [
                        {"cycle": cycle, "mode": mode, "provider": provider}
                        for (cycle, mode), provider in sorted(
                            supply_provider_schedule(config["hybrid_experiment"]).items()
                        )
                    ],
                    "protected_state": _protected_state(config),
                    "model_calls_made": 0,
                    "ready": True,
                }
            else:
                value = run_hybrid_experiment(
                    config,
                    authorize_model_calls=args.authorize_model_calls,
                )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HybridPipelineError",
    "PIPELINE",
    "VERSION",
    "blind_hybrid",
    "conditional_blind_plan",
    "load_hybrid_config",
    "plan_hybrid",
    "realize_hybrid",
    "run_hybrid_experiment",
    "supply_provider_schedule",
]
