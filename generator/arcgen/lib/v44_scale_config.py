"""Configuration and frozen-parent checks for the V4.4 scale experiment."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json

from .config import ConfigError
from .v44_control_supplement import load_v44_control_supplement_config
from .v45_config import verify_v44_baseline


V44_SCALE_PIPELINE = "arcgen-v4.4-exp-functional-scale"
V44_SCALE_VERSION = "arcgen-v4.4-functional-scale-experiment-1.0"


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"functional-scale configuration must be a mapping: {path}")
    return value


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def load_v44_scale_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read(source)
    if raw.get("pipeline") != V44_SCALE_PIPELINE or str(raw.get("version")) != "4.4-scale-experiment":
        raise ConfigError(f"functional-scale config must use pipeline {V44_SCALE_PIPELINE}")
    base_path = Path(str(raw.get("base_v4_4_config", "")))
    base_path = (source.parent / base_path).resolve() if not base_path.is_absolute() else base_path.resolve()
    base = load_v44_control_supplement_config(base_path)
    root = Path(base["_repository_root"]).resolve()
    manifest = Path(str(raw.get("v4_4_baseline_manifest", "")))
    manifest = (root / manifest).resolve() if not manifest.is_absolute() else manifest.resolve()
    baseline = verify_v44_baseline(root, manifest)

    settings = raw.get("functional_scale")
    if not isinstance(settings, Mapping):
        raise ConfigError("functional_scale settings are required")
    if settings.get("enabled") is not True:
        raise ConfigError("the experiment must be explicitly enabled")
    if settings.get("intervention_stage") != "concrete_planner_after_selection":
        raise ConfigError("the scale intervention may occur only after selection")
    continuation_value = raw.get("continuation_of")
    supplemental_extension = raw.get("supplemental_extension")
    independent_slates = _integer(
        settings.get("independent_slates"), "functional_scale.independent_slates", 1
    )
    if continuation_value is None and independent_slates != 3:
        raise ConfigError("the qualification uses exactly three independent slates")
    if continuation_value is not None and independent_slates != 1:
        if not isinstance(supplemental_extension, Mapping):
            raise ConfigError(
                "a multi-slate continuation requires an explicit supplemental_extension authorization"
            )
        if supplemental_extension.get("explicitly_authorized") is not True:
            raise ConfigError("supplemental_extension.explicitly_authorized must be true")
        if supplemental_extension.get("authorization_record") in (None, ""):
            raise ConfigError("supplemental_extension.authorization_record is required")
        authorized_slates = _integer(
            supplemental_extension.get("maximum_new_slates"),
            "supplemental_extension.maximum_new_slates",
            1,
        )
        if independent_slates > authorized_slates:
            raise ConfigError(
                "functional_scale.independent_slates exceeds the authorized supplemental cap"
            )
    if _integer(settings.get("maximum_development_per_slate"), "functional_scale.maximum_development_per_slate", 1) != 8:
        raise ConfigError("each planned slate develops up to eight candidates")
    if _integer(settings.get("eligible_window_size"), "functional_scale.eligible_window_size", 1) != 48:
        raise ConfigError("each selector window must contain 48 proposals")
    fraction = settings.get("treatment_fraction")
    if type(fraction) not in {int, float} or float(fraction) != 0.5:
        raise ConfigError("treatment_fraction must be 0.5")
    _integer(settings.get("minimum_completed_per_arm"), "functional_scale.minimum_completed_per_arm", 1)
    _integer(settings.get("maximum_additional_slates"), "functional_scale.maximum_additional_slates")
    _integer(settings.get("escalation_calls_per_slate"), "functional_scale.escalation_calls_per_slate")
    _integer(settings.get("adoption_minimum_functionally_expanded"), "functional_scale.adoption_minimum_functionally_expanded", 1)
    for field in (
        "full_editorial_review",
        "arm_labels_hidden_downstream",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"functional_scale.{field} must be true")
    for field in (
        "other_optional_planning_hints_enabled",
        "paired_task_variants",
        "automatic_promotion",
        "automatic_task_commit",
        "production_state_mutation_allowed",
    ):
        if settings.get(field) is not False:
            raise ConfigError(f"functional_scale.{field} must be false")
    for field in ("core_yield_ratio_floor", "core_or_reserve_yield_ratio_floor"):
        value = settings.get(field)
        if type(value) not in {int, float} or not 0 < float(value) <= 1:
            raise ConfigError(f"functional_scale.{field} must be in (0, 1]")

    config = deepcopy(base)
    config["v4_4_scale_pipeline"] = V44_SCALE_PIPELINE
    config["functional_scale"] = deepcopy(dict(settings))
    config["functional_scale"]["slate_index_offset"] = 0
    config["seed"] = raw.get("seed", base["seed"])
    if type(config["seed"]) is not int:
        raise ConfigError("seed must be an integer")
    config["run"] = _merge(base["run"], raw.get("run", {}))
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("experimental tasks cannot be committed automatically")
    config["selection"]["eligible_window_size"] = 48
    config["selection"]["max_plans"] = 10
    config["selection"]["max_realizations"] = 8
    config["_config_path"] = str(source)
    config["_v4_4_scale_config_path"] = str(source)
    config["_v4_4_base_config"] = str(base_path)
    config["_v4_4_baseline_verification"] = baseline
    if continuation_value is not None:
        parent = Path(str(continuation_value))
        parent = (root / parent).resolve() if not parent.is_absolute() else parent.resolve()
        parent_report_path = parent / "report.json"
        parent_reservoir = parent / "selection_sandbox" / "reservoir.json"
        if not parent_report_path.is_file() or not parent_reservoir.is_file():
            raise ConfigError("continuation_of must name a completed functional-scale run")
        parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
        if (
            parent_report.get("pipeline") != V44_SCALE_PIPELINE
            or parent_report.get("completed") is not True
        ):
            raise ConfigError("continuation parent is not a completed functional-scale run")
        if parent_report.get("frozen_v4_4_baseline") != baseline:
            raise ConfigError("continuation parent used a different frozen V4.4 baseline")
        config["paths"]["reservoir"] = str(parent_reservoir)
        parent_slate_count = parent_report.get("funnel", {}).get("slates")
        if type(parent_slate_count) is not int or parent_slate_count < 1:
            raise ConfigError("continuation parent has no valid slate count")
        parent_task_slate_indices = [
            task.get("slate_index")
            for task in parent_report.get("tasks", [])
            if type(task.get("slate_index")) is int
        ]
        # A continuation can itself be the parent of a later explicitly
        # authorized supplement.  Use the actual task ledger rather than the
        # number of slates local to that parent so slate identifiers never
        # restart and reservation owners cannot collide.
        parent_max_slate_index = max(parent_task_slate_indices, default=parent_slate_count)
        if parent_max_slate_index < parent_slate_count:
            raise ConfigError("continuation parent has inconsistent slate indices")
        config["functional_scale"]["slate_index_offset"] = parent_max_slate_index
        config["_v4_4_scale_continuation"] = {
            "parent_run": str(parent),
            "parent_run_id": parent_report.get("run_id"),
            "parent_report_sha256": sha256(parent_report_path.read_bytes()).hexdigest(),
            "parent_max_slate_index": parent_max_slate_index,
            "supplemental_extension": deepcopy(supplemental_extension),
        }
    return config


def snapshot_v44_scale_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


__all__ = [
    "V44_SCALE_PIPELINE",
    "V44_SCALE_VERSION",
    "load_v44_scale_config",
    "snapshot_v44_scale_config",
]
