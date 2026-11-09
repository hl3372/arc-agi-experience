"""Configuration boundary for the lean authoring pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json

from .config import ConfigError


PIPELINE_ID = "arcgen-lean-authoring-v1"
SEARCH_MODES = ("open", "concept_led", "composition_led")
NORMAL_STAGE_LIMITS = {
    "idea_search": 3,
    "batch_selector": 1,
    "episode_architect": 5,
    "task_realizer": 4,
    "technical_verifier": 4,
    "blind_solver": 4,
    "editorial_board": 2,
}
NORMAL_CALL_LIMIT = sum(NORMAL_STAGE_LIMITS.values())


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return value


def _positive(value: Any, path: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        adjective = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{path} must be a {adjective} integer")
    return value


def _resolve(base: Path, value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty path")
    candidate = Path(value)
    return str((candidate if candidate.is_absolute() else base / candidate).resolve())


def load_lean_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"lean configuration file not found: {source}")
    config = _read(source)
    if config.get("pipeline") != PIPELINE_ID:
        raise ConfigError(f"pipeline must be {PIPELINE_ID!r}")
    if config.get("version") != 1:
        raise ConfigError("lean configuration version must be 1")
    if type(config.get("seed")) is not int:
        raise ConfigError("seed must be an integer")
    for section in ("paths", "model", "embedding", "experiment", "budget", "run"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{section} must be a mapping")

    base = source.parent.parent if source.parent.name == "config" else source.parent
    for key in (
        "annotations",
        "tasks",
        "prompts",
        "runs",
        "cache",
        "fixed_anchor_manifest",
        "legacy_concept_archive",
        "lean_concept_archive",
    ):
        config["paths"][key] = _resolve(
            base, config["paths"].get(key), f"paths.{key}"
        )

    model = config["model"]
    if model.get("provider") not in {"codex_cli", "mock"}:
        raise ConfigError("model.provider must be codex_cli or mock")
    for key in ("executable", "name", "reasoning_effort"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ConfigError(f"model.{key} must be a non-empty string")
    allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
    if model["reasoning_effort"] not in allowed_efforts:
        raise ConfigError("model.reasoning_effort is invalid")
    _positive(model.get("timeout_seconds"), "model.timeout_seconds")
    _positive(model.get("concurrency"), "model.concurrency")
    _positive(model.get("retries"), "model.retries", allow_zero=True)
    roles = model.get("roles", {})
    if not isinstance(roles, dict):
        raise ConfigError("model.roles must be a mapping")
    for role, setting in roles.items():
        if not isinstance(role, str) or not isinstance(setting, dict):
            raise ConfigError("each model.roles entry must map a name to a mapping")
        unknown = set(setting) - {
            "model",
            "reasoning_effort",
            "timeout_seconds",
            "retries",
        }
        if unknown:
            raise ConfigError(
                f"model.roles.{role} has unsupported keys: {', '.join(sorted(unknown))}"
            )
        if "model" in setting and (
            not isinstance(setting["model"], str) or not setting["model"].strip()
        ):
            raise ConfigError(f"model.roles.{role}.model must be non-empty")
        if setting.get("reasoning_effort", model["reasoning_effort"]) not in allowed_efforts:
            raise ConfigError(f"model.roles.{role}.reasoning_effort is invalid")
        if "timeout_seconds" in setting:
            _positive(setting["timeout_seconds"], f"model.roles.{role}.timeout_seconds")
        if "retries" in setting:
            _positive(setting["retries"], f"model.roles.{role}.retries", allow_zero=True)

    embedding = config["embedding"]
    if not isinstance(embedding.get("provider"), str) or not embedding["provider"].strip():
        raise ConfigError("embedding.provider must be configured")
    if not isinstance(embedding.get("model"), str) or not embedding["model"].strip():
        raise ConfigError("embedding.model must be configured, not hard-coded in code")

    exp = config["experiment"]
    expected = {
        "idea_search_calls": 3,
        "ideas_per_call": 4,
        "calibration_examples_per_call": 4,
        "selector_references_per_candidate": 3,
        "max_architectures": 5,
        "max_realizations": 4,
        "fixed_anchor_count": 6,
        "editorial_boards": 2,
    }
    for key, required in expected.items():
        if exp.get(key) != required:
            raise ConfigError(f"experiment.{key} must equal {required}")
    pool_count = _positive(
        exp.get("generator_visible_pool_count"),
        "experiment.generator_visible_pool_count",
    )
    if pool_count + exp["fixed_anchor_count"] > 120:
        raise ConfigError("generator-visible pool and fixed anchors exceed the 120-task bank")
    if exp.get("search_modes") != list(SEARCH_MODES):
        raise ConfigError(
            "experiment.search_modes must be open, concept_led, composition_led exactly"
        )

    budget = config["budget"]
    if budget.get("normal_call_limit") != NORMAL_CALL_LIMIT:
        raise ConfigError(
            f"budget.normal_call_limit must equal the fixed normal plan ({NORMAL_CALL_LIMIT})"
        )
    if budget.get("escalation_call_limit") != 5:
        raise ConfigError("budget.escalation_call_limit must equal 5")
    if budget.get("hard_substantive_call_limit") != 28:
        raise ConfigError("budget.hard_substantive_call_limit must equal 28")
    if budget["normal_call_limit"] + budget["escalation_call_limit"] != budget[
        "hard_substantive_call_limit"
    ]:
        raise ConfigError("normal and escalation budgets do not add to the hard limit")

    run = config["run"]
    if not isinstance(run.get("id"), str) or not run["id"].strip():
        raise ConfigError("run.id must be a non-empty string")
    if type(run.get("mock", False)) is not bool:
        raise ConfigError("run.mock must be boolean")
    if type(run.get("legacy_audit", True)) is not bool:
        raise ConfigError("run.legacy_audit must be boolean")

    config["_config_path"] = str(source)
    return config


def snapshot_lean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in config.items()
        if not key.startswith("_")
    }


__all__ = [
    "NORMAL_CALL_LIMIT",
    "NORMAL_STAGE_LIMITS",
    "PIPELINE_ID",
    "SEARCH_MODES",
    "load_lean_config",
    "snapshot_lean_config",
]
