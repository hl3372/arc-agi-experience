"""Configuration and frozen-control checks for isolated ARCGEN V4.1 studies."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

from .config import ConfigError
from .util import content_hash, read_json
from .v4_config import PRODUCTION_ROLE_CONTROL, load_v4_config
from .v41_features import (
    FEATURE_ORDER,
    FEATURE_TO_FLAG,
    accepted_feature_flags,
    assert_prior_decisions_complete,
    load_feature_manifest,
)


V41_CONTROL_PIPELINE = "arcgen-v4.1-control"
V41_VERSION = "arcgen-v4.1-experiments-4.1.0"
EXPERIMENT_PIPELINES = {
    "palette_diversity": "arcgen-v4.1-exp-palette",
    "supported_departure": "arcgen-v4.1-exp-supported-departure",
    "episode_joint_inference": "arcgen-v4.1-exp-episode-joint",
    "abstraction_led": "arcgen-v4.1-exp-abstraction-led",
}
INTEGRATION_PIPELINES = {
    "palette_diversity": "arcgen-v4.1-integration-palette",
    "supported_departure": "arcgen-v4.1-integration-curveball",
    "episode_joint_inference": "arcgen-v4.1-integration-episode-joint",
    "abstraction_led": "arcgen-v4.1-integration-abstraction",
}
FLAG_KEYS = tuple(FEATURE_TO_FLAG.values())


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"V4.1 configuration root must be a mapping: {path}")
    return value


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve(root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def verify_v41_parent(repository_root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    manifest = read_json(Path(manifest_path).resolve())
    if manifest.get("manifest_id") != "arcgen-v4.1-parent-v4-qualified-20260803":
        raise ConfigError("unexpected V4.1 frozen parent manifest")
    failures = []
    for relative, expected in manifest.get("frozen_files", {}).items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"changed {relative}: {actual} != {expected}")
    if failures:
        raise ConfigError("frozen V4 control changed: " + " | ".join(failures))
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": content_hash(manifest),
        "verified_file_count": len(manifest["frozen_files"]),
        "source_pipeline": manifest["source_pipeline"],
        "source_run": manifest["source_run"],
    }


def _positive_integer(value: Any, field: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def load_v41_config(
    path: str | Path,
    *,
    allow_experiment: bool = False,
    allow_integration: bool = False,
) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read(source)
    if raw.get("extends") is not None:
        parent = Path(str(raw["extends"]))
        parent = (source.parent / parent).resolve() if not parent.is_absolute() else parent.resolve()
        inherited = _read(parent)
        if inherited.get("extends") is not None:
            raise ConfigError("nested V4.1 configuration inheritance is not supported")
        raw = _merge(inherited, raw)
    if str(raw.get("version")) != "4.1":
        raise ConfigError("V4.1 configuration version must be 4.1")
    repository_root = source.parent.parent if source.parent.name == "config" else source.parent
    v4_source = Path(str(raw.get("extends_v4_control", "")))
    v4_source = (source.parent / v4_source).resolve() if not v4_source.is_absolute() else v4_source.resolve()
    v4 = load_v4_config(v4_source)
    if v4["pipeline"] != "arcgen-v4-control":
        raise ConfigError("V4.1 must branch from the V4 control configuration")

    config = deepcopy(v4)
    config.update({key: deepcopy(value) for key, value in raw.items() if key not in {
        "extends", "extends_v4_control", "paths", "run"
    }})
    config["pipeline"] = raw.get("pipeline")
    config["seed"] = raw.get("seed", v4["seed"])
    config["paths"] = deepcopy(v4["paths"])
    for key, value in raw.get("paths", {}).items():
        config["paths"][key] = _resolve(repository_root, value, f"paths.{key}")
    config["run"] = _merge(v4["run"], raw.get("run", {}))
    config["model"] = deepcopy(v4["model"])
    config["embedding"] = deepcopy(v4["embedding"])
    config["reservoir"] = deepcopy(v4["reservoir"])
    config["selection"] = deepcopy(v4["selection"])
    config["editorial_queue"] = deepcopy(v4["editorial_queue"])
    config["budget"] = deepcopy(v4["budget"])

    for section in ("v4_1_features", "experimental", "integration", "palette_experiment", "semantic_experiments"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{section} must be a mapping")
    flags = config["v4_1_features"]
    if set(flags) != set(FLAG_KEYS) or any(type(flags[key]) is not bool for key in FLAG_KEYS):
        raise ConfigError("v4_1_features must contain exactly four boolean flags")
    if config["run"].get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("V4.1 experimental task commits are forbidden")
    if config["model"]["roles"] != v4["model"]["roles"]:
        raise ConfigError("V4.1 model/effort routing differs from frozen V4")
    for role, (model, effort) in PRODUCTION_ROLE_CONTROL.items():
        if config["model"]["roles"][role] != {"model": model, "reasoning_effort": effort}:
            raise ConfigError(f"V4.1 role {role} differs from V4 control")

    parent = verify_v41_parent(repository_root, config["paths"]["v4_1_parent_manifest"])
    feature_manifest = load_feature_manifest(config["paths"]["v4_1_feature_manifest"])
    accepted = accepted_feature_flags(feature_manifest)
    experimental = config["experimental"]
    integration = config["integration"]
    if experimental.get("auto_promote") is not False or experimental.get("non_gating") is not True:
        raise ConfigError("V4.1 experiments must be non-gating and cannot auto-promote")
    if experimental.get("production_state_mutation_allowed") is not False:
        raise ConfigError("V4.1 experiments cannot mutate production state")

    feature = experimental.get("feature")
    effective = dict(accepted)
    pipeline = config["pipeline"]
    if pipeline == V41_CONTROL_PIPELINE:
        if experimental.get("enabled") is not False or feature != "none":
            raise ConfigError("V4.1 control must disable every experiment")
        if any(flags.values()):
            raise ConfigError("all V4.1 control feature flags must be off")
        if integration.get("enabled") is not False:
            raise ConfigError("V4.1 control cannot be an integration run")
        effective = {key: False for key in FLAG_KEYS}
    elif experimental.get("enabled") is True:
        if not allow_experiment:
            raise ConfigError("V4.1 experiment requires the experimental runner")
        if feature not in FEATURE_ORDER or pipeline != EXPERIMENT_PIPELINES[feature]:
            raise ConfigError("V4.1 experiment pipeline and feature do not match")
        assert_prior_decisions_complete(feature_manifest, feature)
        treatment_flag = FEATURE_TO_FLAG[feature]
        configured = {key for key, enabled in flags.items() if enabled}
        if configured != {treatment_flag}:
            raise ConfigError("experimental config must enable only its isolated treatment flag")
        effective[treatment_flag] = True
        if integration.get("enabled") is not False:
            raise ConfigError("experiment and integration modes cannot be combined")
    elif integration.get("enabled") is True:
        if not allow_integration:
            raise ConfigError("V4.1 integration requires the integration runner")
        stage = integration.get("stage")
        if stage not in FEATURE_ORDER or pipeline != INTEGRATION_PIPELINES[stage]:
            raise ConfigError("V4.1 integration stage and pipeline do not match")
        assert_prior_decisions_complete(feature_manifest, stage)
        if feature_manifest["features"][stage]["status"] != "accepted":
            raise ConfigError(f"integration feature {stage} has not been explicitly accepted")
        enabled = {key for key, value in flags.items() if value}
        expected = {
            FEATURE_TO_FLAG[item] for item in FEATURE_ORDER[: FEATURE_ORDER.index(stage) + 1]
            if feature_manifest["features"][item]["status"] == "accepted"
        }
        if enabled != expected:
            raise ConfigError("integration flags do not match the explicitly accepted baseline")
        effective = dict(accepted)
    else:
        raise ConfigError("unknown or inactive V4.1 pipeline configuration")

    palette = config["palette_experiment"]
    colors = palette.get("allowed_background_colors")
    if colors != list(range(10)) or palette.get("keep_black_eligible") is not True:
        raise ConfigError("palette study must retain every ARC color including black")
    for key in ("replay_sample_size", "technical_subset_size", "blind_subset_size", "editorial_subset_size"):
        _positive_integer(palette.get(key), f"palette_experiment.{key}")
    if palette["replay_sample_size"] < 12:
        raise ConfigError("palette replay sample must contain at least 12 tasks")
    if palette["technical_subset_size"] > palette["replay_sample_size"] or palette["blind_subset_size"] > palette["replay_sample_size"]:
        raise ConfigError("palette review subset exceeds replay sample")
    weights = float(palette.get("real_reference_weight", -1)) + float(palette.get("diversity_weight", -1))
    if abs(weights - 1.0) > 1e-9:
        raise ConfigError("palette reference and diversity weights must sum to one")

    config["_config_path"] = str(source)
    config["_repository_root"] = str(repository_root.resolve())
    config["_v4_control_config"] = str(v4_source)
    config["_v4_parent_verification"] = parent
    config["_feature_manifest_snapshot"] = feature_manifest
    config["_effective_features"] = effective
    return config


def snapshot_v41_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


__all__ = [
    "EXPERIMENT_PIPELINES",
    "FLAG_KEYS",
    "INTEGRATION_PIPELINES",
    "V41_CONTROL_PIPELINE",
    "V41_VERSION",
    "load_v41_config",
    "snapshot_v41_config",
    "verify_v41_parent",
]
