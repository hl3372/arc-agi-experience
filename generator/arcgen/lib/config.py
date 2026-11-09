"""Configuration loading and validation for the ARC generation benchmark."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json


class ConfigError(ValueError):
    """Raised when a benchmark configuration is incomplete or inconsistent."""


DEFAULTS: dict[str, Any] = {
    "version": 1,
    "seed": 20260722,
    "paths": {
        "annotations": "annotations",
        "tasks": "evaluation",
        "prompts": "prompts",
        "runs": "runs",
        "cache": ".arcgen_cache",
    },
    "model": {
        "provider": "codex_cli",
        "executable": "codex",
        "name": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "timeout_seconds": 1800,
        "retries": 1,
        "concurrency": 4,
        "stage_overrides": {},
    },
    "embedding": {
        "provider": "hashing_tfidf",
        "model": "arc-text-hash-v1",
        "dimensions": 1536,
        "word_ngrams": [1, 2],
        "char_ngrams": [3, 5],
    },
    "experiment": {
        "calibration_count": 90,
        "anchor_count": 30,
        "ideas_per_call": 4,
        "calls_per_mode": 2,
        "calibration_examples_per_call": 5,
        "dedup_similarity_threshold": 0.70,
        "max_unique_ideas": 14,
        "novelty_screen_min_per_mode": 4,
        "novelty_reference_count": 50,
        "novelty_scout_count": 2,
        "max_development_candidates": 9,
        "max_per_mode_for_development": 3,
        "anchor_sample_count": 5,
        "ranking_judges": 3,
        "max_ranking_batch": 16,
    },
    "ambiguity": {
        "policy": "bounded_two_guess_fallback_v2",
        "max_training_evidence_repairs": 1,
        "max_bounded_two_guess_fraction": 0.40,
        "minimum_one_guess_before_fallback": 2,
    },
    "validation": {
        "route": "legacy",
        "description_aware": {
            "checkers": [
                {
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                },
                {
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "xhigh",
                },
            ],
            "timeout_seconds": 300,
            "retries": 0,
            "blind_audit_count": 0,
            "blind_audit_model": "gpt-5.6-sol",
            "blind_audit_effort": "high",
            "blind_audit_timeout_seconds": 300,
        },
    },
    "ranking": {
        "route": "full_batch",
        "minibatch": {
            "generated_per_batch": 4,
            "rankers_per_batch": 3,
            "partition_seed": 20260724,
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "timeout_seconds": 420,
            "retries": 0,
        },
    },
    "run": {"id": "mvp-20260722-real", "mock": False},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(text)
    except ModuleNotFoundError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                "PyYAML is unavailable and the configuration is not JSON-compatible YAML"
            ) from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, default, validate, and path-resolve a version-1 configuration."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"configuration file not found: {source}")
    config = _merge(DEFAULTS, _read_yaml_or_json(source))
    if config.get("version") != 1:
        raise ConfigError("only configuration version 1 is supported")
    if not isinstance(config.get("seed"), int):
        raise ConfigError("seed must be an integer")
    for section in (
        "paths",
        "model",
        "embedding",
        "experiment",
        "ambiguity",
        "validation",
        "ranking",
        "run",
    ):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{section} must be a mapping")

    base = source.parent.parent if source.parent.name == "config" else source.parent
    for key in ("annotations", "tasks", "prompts", "runs", "cache"):
        raw = config["paths"].get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError(f"paths.{key} must be a non-empty string")
        candidate = Path(raw)
        config["paths"][key] = str(
            (candidate if candidate.is_absolute() else base / candidate).resolve()
        )

    model = config["model"]
    if model.get("provider") not in {"codex_cli", "mock"}:
        raise ConfigError("model.provider must be codex_cli or mock")
    for key in ("executable", "name", "reasoning_effort"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ConfigError(f"model.{key} must be a non-empty string")
    if model["reasoning_effort"] not in {"low", "medium", "high", "xhigh", "max"}:
        raise ConfigError("model.reasoning_effort is invalid")
    for key in ("timeout_seconds", "concurrency"):
        if not isinstance(model.get(key), int) or model[key] < 1:
            raise ConfigError(f"model.{key} must be a positive integer")
    if not isinstance(model.get("retries"), int) or model["retries"] < 0:
        raise ConfigError("model.retries must be a non-negative integer")
    stage_overrides = model.get("stage_overrides", {})
    if not isinstance(stage_overrides, dict):
        raise ConfigError("model.stage_overrides must be a mapping")
    for stage, override in stage_overrides.items():
        if not isinstance(stage, str) or not stage or not isinstance(override, dict):
            raise ConfigError(
                "each model.stage_overrides entry must map a stage name to a mapping"
            )
        unknown = set(override) - {"timeout_seconds", "retries"}
        if unknown:
            raise ConfigError(
                f"model.stage_overrides.{stage} has unsupported keys: "
                + ", ".join(sorted(unknown))
            )
        if (
            "timeout_seconds" in override
            and (
                type(override["timeout_seconds"]) is not int
                or override["timeout_seconds"] < 1
            )
        ):
            raise ConfigError(
                f"model.stage_overrides.{stage}.timeout_seconds must be positive"
            )
        if (
            "retries" in override
            and (
                type(override["retries"]) is not int
                or override["retries"] < 0
            )
        ):
            raise ConfigError(
                f"model.stage_overrides.{stage}.retries must be non-negative"
            )

    exp = config["experiment"]
    if exp["calibration_count"] + exp["anchor_count"] != 120:
        raise ConfigError("calibration_count plus anchor_count must equal 120")
    if exp["ideas_per_call"] != 4 or exp["calls_per_mode"] != 2:
        raise ConfigError("the MVP requires exactly two four-idea calls per mode")
    if exp["novelty_reference_count"] != 50 or exp["novelty_scout_count"] != 2:
        raise ConfigError("the MVP novelty audit requires two scouts over 50 references")
    if exp["max_ranking_batch"] > 16:
        raise ConfigError("ranking batch size cannot exceed 16")
    screen_minimum = exp.get("novelty_screen_min_per_mode")
    if type(screen_minimum) is not int or screen_minimum < 0:
        raise ConfigError(
            "experiment.novelty_screen_min_per_mode must be a non-negative integer"
        )
    ambiguity = config["ambiguity"]
    if ambiguity.get("policy") != "bounded_two_guess_fallback_v2":
        raise ConfigError(
            "ambiguity.policy must be bounded_two_guess_fallback_v2"
        )
    repair_count = ambiguity.get("max_training_evidence_repairs")
    if type(repair_count) is not int or repair_count not in {0, 1}:
        raise ConfigError("ambiguity.max_training_evidence_repairs must be 0 or 1")
    bounded_fraction = ambiguity.get("max_bounded_two_guess_fraction")
    if (
        type(bounded_fraction) not in (int, float)
        or not 0 <= float(bounded_fraction) < 0.5
    ):
        raise ConfigError(
            "ambiguity.max_bounded_two_guess_fraction must be in [0, 0.5)"
        )
    minimum_unique = ambiguity.get("minimum_one_guess_before_fallback")
    if type(minimum_unique) is not int or minimum_unique < 1:
        raise ConfigError(
            "ambiguity.minimum_one_guess_before_fallback must be a positive integer"
        )
    validation = config["validation"]
    if validation.get("route") not in {"legacy", "description_aware_v1"}:
        raise ConfigError(
            "validation.route must be legacy or description_aware_v1"
        )
    informed = validation.get("description_aware")
    if not isinstance(informed, dict):
        raise ConfigError("validation.description_aware must be a mapping")
    checkers = informed.get("checkers")
    if not isinstance(checkers, list) or len(checkers) != 2:
        raise ConfigError(
            "validation.description_aware.checkers must contain exactly two settings"
        )
    for index, checker in enumerate(checkers):
        if not isinstance(checker, dict):
            raise ConfigError(
                f"validation.description_aware.checkers[{index}] must be a mapping"
            )
        if not isinstance(checker.get("model"), str) or not checker["model"]:
            raise ConfigError(
                f"validation.description_aware.checkers[{index}].model is required"
            )
        if checker.get("reasoning_effort") not in {
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ConfigError(
                f"validation.description_aware.checkers[{index}].reasoning_effort is invalid"
            )
    if checkers[0] == checkers[1]:
        raise ConfigError(
            "validation.description_aware checkers must use distinct configurations"
        )
    for key in ("timeout_seconds", "blind_audit_timeout_seconds"):
        if type(informed.get(key)) is not int or informed[key] < 1:
            raise ConfigError(
                f"validation.description_aware.{key} must be positive"
            )
    if type(informed.get("retries")) is not int or informed["retries"] < 0:
        raise ConfigError(
            "validation.description_aware.retries must be non-negative"
        )
    if (
        type(informed.get("blind_audit_count")) is not int
        or informed["blind_audit_count"] < 0
    ):
        raise ConfigError(
            "validation.description_aware.blind_audit_count must be non-negative"
        )
    if (
        not isinstance(informed.get("blind_audit_model"), str)
        or not informed["blind_audit_model"]
    ):
        raise ConfigError(
            "validation.description_aware.blind_audit_model is required"
        )
    if informed.get("blind_audit_effort") not in {
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ConfigError(
            "validation.description_aware.blind_audit_effort is invalid"
        )
    ranking = config["ranking"]
    if ranking.get("route") not in {
        "full_batch",
        "anchor_minibatch_v1",
    }:
        raise ConfigError(
            "ranking.route must be full_batch or anchor_minibatch_v1"
        )
    minibatch = ranking.get("minibatch")
    if not isinstance(minibatch, dict):
        raise ConfigError("ranking.minibatch must be a mapping")
    for key in (
        "generated_per_batch",
        "rankers_per_batch",
        "partition_seed",
        "timeout_seconds",
    ):
        if type(minibatch.get(key)) is not int or minibatch[key] < 1:
            raise ConfigError(
                f"ranking.minibatch.{key} must be positive"
            )
    if minibatch["rankers_per_batch"] != 3:
        raise ConfigError(
            "ranking.minibatch.rankers_per_batch must equal 3"
        )
    if (
        not isinstance(minibatch.get("model"), str)
        or not minibatch["model"]
    ):
        raise ConfigError("ranking.minibatch.model is required")
    if minibatch.get("reasoning_effort") not in {
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ConfigError(
            "ranking.minibatch.reasoning_effort is invalid"
        )
    if (
        type(minibatch.get("retries")) is not int
        or minibatch["retries"] < 0
    ):
        raise ConfigError(
            "ranking.minibatch.retries must be non-negative"
        )
    config["_config_path"] = str(source)
    return config


def snapshot_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable configuration without loader-private keys."""

    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}
