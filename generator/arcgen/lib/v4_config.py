"""Configuration and frozen-parent validation for ARCGEN V4.

V4 is an additive branch from the saved V3.1.2 implementation.  Production
and every experiment use separate identifiers; loading an experimental
configuration never makes it eligible for the production runner.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

from .config import ConfigError
from .util import content_hash, read_json


CONTROL_PIPELINE_ID = "arcgen-v4-control"
V4_PIPELINE_VERSION = "arcgen-v4-control-4.0.0"
SEARCH_MODES = ("open", "concept_led", "composition_led")
EXPERIMENT_PIPELINE_IDS = {
    "arcgen-v4-exp-plan-review-shadow",
    "arcgen-v4-exp-plan-review-treatment",
    "arcgen-v4-exp-answer-aware",
    "arcgen-v4-exp-idea-effort",
    "arcgen-v4-exp-realizer-effort",
    "arcgen-v4-exp-reservoir-curation-effort",
}
EXPERIMENT_KINDS = {
    "plan_review_shadow",
    "plan_review_treatment",
    "answer_aware",
    "idea_effort",
    "realizer_effort",
    "reservoir_curation_effort",
}
PRODUCTION_ROLE_CONTROL = {
    "idea_search": ("gpt-5.6-sol", "max"),
    "batch_selector": ("gpt-5.6-sol", "max"),
    "concrete_planner": ("gpt-5.6-sol", "max"),
    "task_realizer": ("gpt-5.6-sol", "max"),
    "technical_verifier": ("gpt-5.6-sol", "high"),
    "blind_solver": ("gpt-5.6-sol", "xhigh"),
    "ambiguity_adjudicator": ("gpt-5.6-sol", "high"),
    "targeted_repair": ("gpt-5.6-sol", "max"),
    "editorial_board": ("gpt-5.6-sol", "max"),
}
NORMAL_STAGE_LIMITS = {
    "batch_selector": 5,
    "concrete_planner": 10,
    "task_realizer": 8,
    "technical_verifier": 8,
    "blind_solver": 8,
    # Two boards review each bounded group.  Normally eight realized tasks
    # form two groups (four calls).  The hard role cap also covers the explicit
    # maximum-wait rule, whose worst case is eight one-task timeout flushes.
    "editorial_board": 16,
}
NORMAL_CONTROL_MAXIMUM = sum(NORMAL_STAGE_LIMITS.values())


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve(base: Path, value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    candidate = Path(value)
    return str((candidate if candidate.is_absolute() else base / candidate).resolve())


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ConfigError(f"{field} must be a number >= {minimum}")
    return float(value)


def verify_v4_parent(repository_root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("manifest_id") != "arcgen-v4-parent-v3.1.2-20260802":
        raise ConfigError("unexpected V4 parent manifest")
    failures: list[str] = []
    for relative, expected in manifest.get("frozen_files", {}).items():
        source = root / relative
        if not source.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"changed {relative}: {actual} != {expected}")
    if failures:
        raise ConfigError("V4 frozen V3.1.2 parent changed: " + " | ".join(failures))
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": content_hash(manifest),
        "source_snapshot": manifest["source_snapshot"],
        "verified_file_count": len(manifest["frozen_files"]),
        "committed_task_count_at_branch": manifest["mutable_branch_state"][
            "committed_task_count_at_branch"
        ],
    }


def _validate_role_settings(model: Mapping[str, Any]) -> None:
    efforts = {"low", "medium", "high", "xhigh", "max"}
    for key in ("executable", "name", "reasoning_effort"):
        if not isinstance(model.get(key), str) or not str(model[key]).strip():
            raise ConfigError(f"model.{key} must be non-empty")
    if model["reasoning_effort"] not in efforts:
        raise ConfigError("model.reasoning_effort is invalid")
    _integer(model.get("timeout_seconds"), "model.timeout_seconds", 1)
    _integer(model.get("concurrency"), "model.concurrency", 1)
    _integer(model.get("retries"), "model.retries")
    roles = model.get("roles")
    if not isinstance(roles, dict):
        raise ConfigError("model.roles must be a mapping")
    for role, (expected_model, expected_effort) in PRODUCTION_ROLE_CONTROL.items():
        setting = roles.get(role)
        if not isinstance(setting, dict):
            raise ConfigError(f"model.roles.{role} must be configured")
        if setting.get("model") != expected_model or setting.get("reasoning_effort") != expected_effort:
            raise ConfigError(
                f"production role {role} must remain {expected_model}/{expected_effort}"
            )
        if "timeout_seconds" in setting:
            _integer(setting["timeout_seconds"], f"model.roles.{role}.timeout_seconds", 1)
        if "retries" in setting:
            _integer(setting["retries"], f"model.roles.{role}.retries")


def _validate_experiment(config: dict[str, Any], *, allow_experiment: bool) -> None:
    section = config.get("experimental")
    if not isinstance(section, dict):
        raise ConfigError("experimental must be a mapping")
    enabled = section.get("enabled")
    pipeline = config["pipeline"]
    if pipeline == CONTROL_PIPELINE_ID:
        if enabled is not False or section.get("kind") != "none":
            raise ConfigError("v4_control must disable every experimental treatment")
        if section.get("gating") is not False or section.get("auto_promote") is not False:
            raise ConfigError("V4 control cannot gate on or auto-promote experiments")
        return
    if pipeline not in EXPERIMENT_PIPELINE_IDS:
        raise ConfigError(f"unknown V4 pipeline identifier: {pipeline}")
    if not allow_experiment:
        raise ConfigError("experimental configuration cannot run through the V4 control command")
    if enabled is not True or section.get("kind") not in EXPERIMENT_KINDS:
        raise ConfigError("experimental configuration must name and enable one known treatment")
    if section.get("gating") is not False or section.get("auto_promote") is not False:
        raise ConfigError("V4 experiments must remain non-gating and cannot auto-promote")
    expected = "arcgen-v4-exp-" + str(section["kind"]).replace("_", "-")
    if pipeline != expected:
        raise ConfigError(f"pipeline {pipeline} does not match experiment kind {section['kind']}")


def load_v4_config(path: str | Path, *, allow_experiment: bool = False) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"V4 configuration not found: {source}")
    config = _read(source)
    extends = config.get("extends")
    if extends is not None:
        if not isinstance(extends, str) or not extends.strip():
            raise ConfigError("extends must be a non-empty relative configuration path")
        parent = Path(extends)
        parent = (source.parent / parent).resolve() if not parent.is_absolute() else parent.resolve()
        if parent == source:
            raise ConfigError("V4 configuration cannot extend itself")
        inherited = _read(parent)
        if inherited.get("extends") is not None:
            raise ConfigError("nested V4 configuration inheritance is not supported")
        config = _deep_merge(inherited, config)
    if str(config.get("version")) != "4.0":
        raise ConfigError("V4 configuration version must be 4.0")
    _validate_experiment(config, allow_experiment=allow_experiment)
    if type(config.get("seed")) is not int:
        raise ConfigError("seed must be an integer")
    for section in (
        "paths", "model", "embedding", "reservoir", "selection",
        "editorial_queue", "budget", "run", "experimental",
    ):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{section} must be a mapping")

    base = source.parent.parent if source.parent.name == "config" else source.parent
    for key in (
        "annotations", "tasks", "prompts", "runs", "cache",
        "fixed_anchor_manifest", "parent_manifest", "committed_task_registry",
        "generated_reference_policy", "reservoir", "historical_case_manifest",
    ):
        config["paths"][key] = _resolve(base, config["paths"].get(key), f"paths.{key}")

    model = config["model"]
    if model.get("provider") not in {"codex_cli", "mock"}:
        raise ConfigError("model.provider must be codex_cli or mock")
    _validate_role_settings(model)

    embedding = config["embedding"]
    if not isinstance(embedding.get("provider"), str) or not embedding["provider"]:
        raise ConfigError("embedding.provider must be configured")
    if not isinstance(embedding.get("model"), str) or not embedding["model"]:
        raise ConfigError("embedding.model must be behind the provider interface")

    reservoir = config["reservoir"]
    low = _integer(reservoir.get("low_watermark"), "reservoir.low_watermark", 1)
    high = _integer(reservoir.get("high_watermark"), "reservoir.high_watermark", 1)
    hard = _integer(reservoir.get("hard_capacity"), "reservoir.hard_capacity", 1)
    if not low <= high <= hard:
        raise ConfigError("reservoir watermarks must satisfy low <= high <= hard capacity")
    _integer(reservoir.get("lease_seconds"), "reservoir.lease_seconds", 60)
    _integer(reservoir.get("replenishment_cycles_per_command"), "reservoir.replenishment_cycles_per_command", 1)
    policy = reservoir.get("per_mode_supply_calls")
    if not isinstance(policy, dict) or set(policy) != set(SEARCH_MODES):
        raise ConfigError("reservoir.per_mode_supply_calls must name the three V4 modes")
    if any(_integer(value, f"reservoir.per_mode_supply_calls.{mode}", 1) != 1 for mode, value in policy.items()):
        raise ConfigError("each replenishment cycle must use one call from each mode")

    selection = config["selection"]
    window = _integer(selection.get("eligible_window_size"), "selection.eligible_window_size", 1)
    shard = _integer(selection.get("shard_size"), "selection.shard_size", 1)
    if window % shard:
        raise ConfigError("eligible window size must be divisible by shard size")
    if window > hard:
        raise ConfigError("eligible window cannot exceed reservoir capacity")
    _integer(selection.get("shard_selector_max_selected"), "selection.shard_selector_max_selected", 1)
    global_max = _integer(selection.get("global_selector_max_selected"), "selection.global_selector_max_selected", 1)
    max_plans = _integer(selection.get("max_plans"), "selection.max_plans", 1)
    max_realizations = _integer(selection.get("max_realizations"), "selection.max_realizations", 1)
    if global_max > max_plans or max_realizations > max_plans:
        raise ConfigError("selection caps must satisfy global <= plans and realizations <= plans")
    _integer(selection.get("references_per_candidate"), "selection.references_per_candidate", 1)
    if selection.get("search_modes") != list(SEARCH_MODES):
        raise ConfigError("V4 control must retain open, concept_led, composition_led")
    if _integer(selection.get("fixed_anchor_count"), "selection.fixed_anchor_count", 1) != 6:
        raise ConfigError("V4 must keep the fixed six-anchor panel")

    queue = config["editorial_queue"]
    minimum = _integer(queue.get("minimum_generated_tasks"), "editorial_queue.minimum_generated_tasks", 1)
    maximum = _integer(queue.get("maximum_generated_tasks"), "editorial_queue.maximum_generated_tasks", 1)
    if not 4 <= minimum <= maximum <= 6:
        raise ConfigError("editorial groups must normally contain four to six generated tasks")
    _integer(queue.get("maximum_wait_seconds"), "editorial_queue.maximum_wait_seconds", 1)

    budget = config["budget"]
    if budget.get("normal_stage_limits") != NORMAL_STAGE_LIMITS:
        raise ConfigError("V4 normal stage limits differ from the production control")
    if budget.get("normal_control_maximum") != NORMAL_CONTROL_MAXIMUM:
        raise ConfigError(f"budget.normal_control_maximum must equal {NORMAL_CONTROL_MAXIMUM}")
    if budget.get("escalation_call_limit") != 5:
        raise ConfigError("V4 escalation limit must remain five")
    if budget.get("hard_control_maximum") != NORMAL_CONTROL_MAXIMUM + 5:
        raise ConfigError("V4 hard control maximum must equal normal plus escalation")
    _integer(budget.get("reservoir_calls_per_cycle"), "budget.reservoir_calls_per_cycle", 3)
    if budget["reservoir_calls_per_cycle"] != 3:
        raise ConfigError("a replenishment cycle must use exactly three supply calls")

    run = config["run"]
    if not isinstance(run.get("id"), str) or not run["id"].strip():
        raise ConfigError("run.id must be non-empty")
    if run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("automatic generated-task commits are forbidden")

    verify_v4_parent(base, config["paths"]["parent_manifest"])
    config["_config_path"] = str(source)
    config["_repository_root"] = str(base.resolve())
    return config


def snapshot_v4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def v4_call_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    window = int(config["selection"]["eligible_window_size"])
    shards = window // int(config["selection"]["shard_size"])
    maximum_realized = int(config["selection"]["max_realizations"])
    queue_size = int(config["editorial_queue"]["maximum_generated_tasks"])
    typical_groups = (
        maximum_realized
        + int(config["editorial_queue"]["minimum_generated_tasks"])
        - 1
    ) // int(config["editorial_queue"]["minimum_generated_tasks"])
    hard_groups = maximum_realized
    return {
        "pipeline": config["pipeline"],
        "production_control": config["pipeline"] == CONTROL_PIPELINE_ID,
        "idea_generation_on_development_critical_path": False,
        "reservoir_supply_calls_per_cycle": 3,
        "eligible_window_size": window,
        "shard_selector_calls": shards,
        "global_selector_calls": 1,
        "maximum_plans": int(config["selection"]["max_plans"]),
        "maximum_realizations": maximum_realized,
        "typical_maximum_editorial_groups": typical_groups,
        "hard_maximum_editorial_groups_with_timeout_flushes": hard_groups,
        "typical_maximum_editorial_board_calls": typical_groups * 2,
        "hard_maximum_editorial_board_calls_with_timeout_flushes": hard_groups * 2,
        "normal_stage_limits": deepcopy(NORMAL_STAGE_LIMITS),
        "normal_control_maximum": NORMAL_CONTROL_MAXIMUM,
        "conditional_escalation_maximum": int(config["budget"]["escalation_call_limit"]),
        "hard_control_maximum": int(config["budget"]["hard_control_maximum"]),
        "experimental_treatment": deepcopy(config["experimental"]),
    }


__all__ = [
    "CONTROL_PIPELINE_ID",
    "EXPERIMENT_KINDS",
    "EXPERIMENT_PIPELINE_IDS",
    "NORMAL_CONTROL_MAXIMUM",
    "NORMAL_STAGE_LIMITS",
    "PRODUCTION_ROLE_CONTROL",
    "SEARCH_MODES",
    "V4_PIPELINE_VERSION",
    "load_v4_config",
    "snapshot_v4_config",
    "v4_call_plan",
    "verify_v4_parent",
]
