"""Configuration for the V4-to-current integrated portfolio qualification.

This is deliberately an experiment layered on the exact frozen V4.4 baseline.
It cannot promote features, mutate the good-task bank, or reinterpret the
unpromoted functional-scale result as production policy.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import json

from .config import ConfigError
from .util import read_json
from .v44_control_supplement import load_v44_control_supplement_config
from .v45_config import verify_v44_baseline


INTEGRATED_PIPELINE = "arcgen-v4-integrated-qualification"
INTEGRATED_VERSION = "arcgen-v4-integrated-qualification-1.0"

ROUTES = {
    "ordinary_control",
    "ordinary_current",
    "logical_leap",
    "structural_transfer",
    "functional_scale",
    "episode_joint_inference",
    "abstraction_led",
}


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"integrated qualification config must be a mapping: {path}")
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _fraction(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not 0 <= float(value) <= 1:
        raise ConfigError(f"{field} must be between zero and one")
    return float(value)


def load_integrated_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read(source)
    if raw.get("pipeline") != INTEGRATED_PIPELINE:
        raise ConfigError(f"integrated config must use pipeline {INTEGRATED_PIPELINE}")
    if str(raw.get("version")) != "1.0":
        raise ConfigError("integrated qualification config must use version 1.0")

    base_path = Path(str(raw.get("base_v4_4_config", "")))
    if not base_path.is_absolute():
        base_path = (source.parent / base_path).resolve()
    base = load_v44_control_supplement_config(base_path)
    root = Path(base["_repository_root"]).resolve()

    baseline_path = Path(str(raw.get("v4_4_baseline_manifest", "")))
    if not baseline_path.is_absolute():
        baseline_path = (root / baseline_path).resolve()
    baseline = verify_v44_baseline(root, baseline_path)

    settings = raw.get("integrated_qualification")
    if not isinstance(settings, Mapping):
        raise ConfigError("integrated_qualification settings are required")
    fixed = {
        "independent_slates": 8,
        "eligible_window_size": 48,
        "maximum_development_per_slate": 8,
        "ordinary_control_per_slate": 2,
        "maximum_special_source_per_slate": 1,
        "maximum_planning_invitations_per_slate": 2,
        "maximum_interventions_per_candidate": 1,
    }
    for field, expected in fixed.items():
        actual = _integer(settings.get(field), f"integrated_qualification.{field}", 1)
        if actual != expected:
            raise ConfigError(
                f"integrated_qualification.{field} is fixed at {expected} for qualification"
            )
    if _integer(settings.get("maximum_extension_slates"), "integrated_qualification.maximum_extension_slates") != 2:
        raise ConfigError("qualification may extend by at most two slates")
    if _integer(settings.get("audit_batch_size"), "integrated_qualification.audit_batch_size", 1) > 16:
        raise ConfigError("mechanism audit batches cannot exceed 16 tasks")
    if _integer(settings.get("escalation_calls_per_slate"), "integrated_qualification.escalation_calls_per_slate") != 5:
        raise ConfigError("the frozen five-call escalation ceiling must be retained")
    for field in (
        "arm_labels_hidden_downstream",
        "full_v4_verification",
        "full_editorial_review",
        "one_answer_policy",
        "palette_crosses_safe_current_routes",
        "selector_quality_precedes_routing",
    ):
        if settings.get(field) is not True:
            raise ConfigError(f"integrated_qualification.{field} must be true")
    for field in (
        "forced_feature_stacking",
        "automatic_promotion",
        "automatic_task_commit",
        "production_state_mutation_allowed",
        "answer_aware_verifier_enabled",
    ):
        if settings.get(field) is not False:
            raise ConfigError(f"integrated_qualification.{field} must be false")

    route_cycle = settings.get("planning_route_cycle")
    expected_cycle = [
        ["logical_leap", "functional_scale"],
        ["logical_leap", "structural_transfer"],
        ["structural_transfer", "functional_scale"],
        ["logical_leap", "functional_scale"],
    ]
    if route_cycle != expected_cycle:
        raise ConfigError("planning_route_cycle differs from the preregistered four-slate schedule")
    if settings.get("special_source_cycle") != [
        "episode_joint_inference",
        "abstraction_led",
    ]:
        raise ConfigError("special_source_cycle must alternate episode-joint and abstraction-led")

    bands = settings.get("rolling_core_bands")
    if not isinstance(bands, Mapping):
        raise ConfigError("rolling_core_bands are required")
    expected_bands = {
        "ordinary_unforced_minimum": (0.50, 1.0),
        "logical_leap": (0.10, 0.25),
        "structural_transfer": (0.05, 0.20),
        "episode_joint_inference": (0.10, 0.20),
        "abstraction_led": (0.10, 0.25),
    }
    normalized_bands: dict[str, Any] = {}
    for name, bounds in expected_bands.items():
        value = bands.get(name)
        if name == "ordinary_unforced_minimum":
            minimum = _fraction(value, f"rolling_core_bands.{name}")
            if minimum < bounds[0]:
                raise ConfigError("ordinary/unforced core share must remain at least 50%")
            normalized_bands[name] = minimum
            continue
        if not isinstance(value, Mapping):
            raise ConfigError(f"rolling_core_bands.{name} must be a mapping")
        lower = _fraction(value.get("lower"), f"rolling_core_bands.{name}.lower")
        upper = _fraction(value.get("upper"), f"rolling_core_bands.{name}.upper")
        ceiling = _fraction(value.get("soft_ceiling"), f"rolling_core_bands.{name}.soft_ceiling")
        if not lower <= upper <= ceiling or (lower, ceiling) != bounds:
            raise ConfigError(f"rolling_core_bands.{name} differs from the frozen plan")
        normalized_bands[name] = {"lower": lower, "upper": upper, "soft_ceiling": ceiling}

    gates = settings.get("quality_gates")
    if not isinstance(gates, Mapping):
        raise ConfigError("quality_gates are required")
    if _fraction(gates.get("core_yield_ratio_floor"), "quality_gates.core_yield_ratio_floor") != 0.8:
        raise ConfigError("core yield ratio floor must be 0.8")
    if _fraction(gates.get("core_or_reserve_yield_ratio_floor"), "quality_gates.core_or_reserve_yield_ratio_floor") != 0.9:
        raise ConfigError("core-or-reserve yield ratio floor must be 0.9")
    if _integer(gates.get("maximum_core_serious_ambiguities"), "quality_gates.maximum_core_serious_ambiguities") != 0:
        raise ConfigError("core may contain no unresolved serious ambiguity")
    if _integer(gates.get("maximum_artificially_inflated_scale_tasks"), "quality_gates.maximum_artificially_inflated_scale_tasks") != 0:
        raise ConfigError("artificial scale inflation is not permitted")

    feature_manifest = read_json(root / "config/v4_4_feature_manifest.json")
    accepted = feature_manifest.get("features", {})
    for name in ("palette_diversity", "logical_leap", "episode_joint_inference", "abstraction_led"):
        if not str(accepted.get(name, {}).get("status", "")).startswith("accepted"):
            raise ConfigError(f"accepted V4.4 feature is missing: {name}")

    scale_report_path = Path(str(raw.get("functional_scale_evidence", "")))
    if not scale_report_path.is_absolute():
        scale_report_path = (root / scale_report_path).resolve()
    scale_report = read_json(scale_report_path)
    if scale_report.get("recommendation") != "continue_testing":
        raise ConfigError("functional scale must remain an unpromoted continue-testing feature")
    if scale_report.get("automatic_promotion_occurred") is not False:
        raise ConfigError("functional-scale evidence unexpectedly promoted itself")

    config = deepcopy(base)
    config["integrated_pipeline"] = INTEGRATED_PIPELINE
    config["integrated_qualification"] = deepcopy(dict(settings))
    config["integrated_qualification"]["rolling_core_bands"] = normalized_bands
    config["seed"] = raw.get("seed", base["seed"])
    if type(config["seed"]) is not int:
        raise ConfigError("seed must be an integer")
    config["run"] = {**deepcopy(base["run"]), **deepcopy(raw.get("run", {}))}
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("integrated qualification cannot commit generated tasks")
    config["selection"]["eligible_window_size"] = 48
    config["selection"]["max_realizations"] = 8
    config["selection"]["maximum_optional_development_suggestions"] = 3
    config.setdefault("structural_transfer", {})["audit_batch_size"] = int(settings["audit_batch_size"])
    config["logical_leap"]["audit_batch_size"] = int(settings["audit_batch_size"])
    config["episode_joint"]["audit_batch_size"] = int(settings["audit_batch_size"])
    config["abstraction_led"]["audit_batch_size"] = int(settings["audit_batch_size"])
    config["functional_scale"] = {
        "escalation_calls_per_slate": int(settings["escalation_calls_per_slate"]),
    }
    config["_config_path"] = str(source)
    config["_integrated_config_path"] = str(source)
    config["_v4_4_scale_config_path"] = str(source)
    config["_integrated_base_config"] = str(base_path)
    config["_v4_4_baseline_verification"] = baseline
    config["_v4_4_feature_manifest_snapshot"] = feature_manifest
    config["_functional_scale_evidence"] = {
        "path": str(scale_report_path),
        "run_id": scale_report.get("run_id"),
        "recommendation": scale_report.get("recommendation"),
        "promoted": False,
    }
    return config


def snapshot_integrated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


__all__ = [
    "INTEGRATED_PIPELINE",
    "INTEGRATED_VERSION",
    "ROUTES",
    "load_integrated_config",
    "snapshot_integrated_config",
]
