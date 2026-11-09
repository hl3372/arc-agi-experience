"""Configuration and frozen-baseline checks for the V4.5 Claude study."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json

from .claude_code import ALLOWED_EFFORTS, inspect_claude_code, sanitized_auth_status
from .config import ConfigError
from .util import content_hash, read_json
from .v44_scale_fresh_integrated import load_fresh_scale_integrated_config
from .v45_config import _verify_registry_or_authorized_append


PIPELINE = "arcgen-v4.5-exp-claude-opus-5-cross-eval"
VERSION = "arcgen-v4.5-claude-opus-5-cross-eval-1.0"
BASELINE_ID = "arcgen-v4.5-integrated-authoring-20260808"
EXPECTED_OPUS_REGEX = r"^claude-opus-5(?:$|[-.])"

ROLE_EFFORTS = {
    "idea_search": "max",
    "batch_selector": "max",
    "concrete_planner": "max",
    "task_realizer": "max",
    "technical_verifier": "high",
    "blind_solver": "xhigh",
    "editorial_board": "max",
}


def _read(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"Claude experiment configuration must be a mapping: {path}")
    return value


def _resolve(root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve())


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_v45_integrated_baseline(
    repository_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify the saved V4.5 milestone without touching production state."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("baseline_id") != BASELINE_ID or manifest.get("status") != "frozen":
        raise ConfigError("unexpected or unfrozen V4.5 baseline manifest")
    failures: list[str] = []
    checked: dict[str, str] = {}
    for section in ("critical_source_sha256", "qualification_report_sha256"):
        rows = manifest.get(section)
        if not isinstance(rows, Mapping):
            failures.append(f"missing {section}")
            continue
        for relative, expected in rows.items():
            source = root / str(relative)
            if not source.is_file():
                failures.append(f"missing {relative}")
                continue
            actual = _sha(source)
            checked[str(relative)] = actual
            if actual != expected:
                failures.append(f"changed {relative}: {actual} != {expected}")
    bank = manifest.get("committed_bank", {})
    registry = root / str(bank.get("registry", ""))
    if not registry.is_file():
        failures.append("V4.5 committed registry is missing")
    else:
        try:
            registry_status = _verify_registry_or_authorized_append(
                root=root,
                manifest=manifest,
                registry_path=registry,
            )
        except Exception as exc:
            failures.append(f"V4.5 committed registry is not an authorized append: {exc}")
            registry_status = None
    snapshot = manifest.get("snapshot", {})
    archive = root / str(snapshot.get("archive", ""))
    checksum = root / str(snapshot.get("archive_checksum", ""))
    if not archive.is_file() or not checksum.is_file():
        failures.append("V4.5 restore archive or checksum is missing")
    if failures:
        raise ConfigError("frozen V4.5 baseline changed: " + " | ".join(failures))
    return {
        "baseline_id": BASELINE_ID,
        "manifest_path": str(path),
        "manifest_sha256": _sha(path),
        "verified_artifact_count": len(checked) + 3,
        "committed_task_count": bank.get("task_count"),
        "live_committed_task_count": (
            registry_status.get("live_task_count") if registry_status else None
        ),
        "registry_status": registry_status,
        "qualification_run": manifest.get("qualification", {}).get("run"),
        "snapshot_archive": str(archive.resolve()),
        "verification_hash": content_hash(checked),
    }


def load_v45_claude_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    raw = _read(source)
    if raw.get("pipeline") != PIPELINE or str(raw.get("version")) != "1.0":
        raise ConfigError(f"Claude study must use pipeline {PIPELINE} version 1.0")
    base_path = Path(str(raw.get("base_current_config", "")))
    if not base_path.is_absolute():
        base_path = (source.parent / base_path).resolve()
    base = load_fresh_scale_integrated_config(base_path)
    root = Path(base["_repository_root"]).resolve()
    manifest_path = Path(
        _resolve(root, raw.get("v4_5_baseline_manifest"), "v4_5_baseline_manifest")
    )
    baseline = verify_v45_integrated_baseline(root, manifest_path)

    experiment = raw.get("claude_cross_eval")
    if not isinstance(experiment, Mapping) or experiment.get("enabled") is not True:
        raise ConfigError("claude_cross_eval must be explicitly enabled")
    required_false = (
        "gating",
        "auto_promote",
        "production_state_mutation_allowed",
        "good_task_bank_mutation_allowed",
        "reservoir_mutation_allowed",
        "reservation_mutation_allowed",
        "rejection_memory_mutation_allowed",
        "automatic_task_commit",
    )
    for field in required_false:
        if experiment.get(field) is not False:
            raise ConfigError(f"claude_cross_eval.{field} must be false")
    if experiment.get("codex_selector_gating") is not True:
        raise ConfigError("the source-blind Codex selector must remain the gating selector")
    if experiment.get("claude_selector_shadow") is not True:
        raise ConfigError("the Claude selector must remain a non-gating shadow")
    if experiment.get("crossed_evaluation") is not True:
        raise ConfigError("both providers must evaluate every completed generated task")
    if experiment.get("arm_labels_hidden_downstream") is not True:
        raise ConfigError("author-provider labels must remain hidden downstream")
    if experiment.get("same_prompts_and_schemas") is not True:
        raise ConfigError("both providers must use identical prompts and schemas")
    if experiment.get("same_calibration_descriptions") is not True:
        raise ConfigError("paired idea calls must use identical calibration descriptions")
    if experiment.get("normal_search_modes") != ["open", "concept_led", "composition_led"]:
        raise ConfigError("the experiment must retain all three ordinary V4 search modes")
    for field, expected, minimum, maximum in (
        ("ideas_per_call", 4, 4, 4),
        ("calls_per_mode_per_author", 1, 1, 1),
        ("maximum_development_per_author", 4, 1, 8),
        ("maximum_total_development", 8, 2, 16),
        ("board_calls_per_evaluator", 2, 2, 2),
    ):
        value = experiment.get(field)
        if type(value) is not int or not minimum <= value <= maximum or value != expected:
            raise ConfigError(f"claude_cross_eval.{field} must be {expected}")
    if experiment.get("repairs_enabled") is not False:
        raise ConfigError("repairs are disabled in the initial attribution study")
    observed_failures = experiment.get("observed_terminal_failures", [])
    if not isinstance(observed_failures, list):
        raise ConfigError("observed_terminal_failures must be a list")
    for row in observed_failures:
        if not isinstance(row, Mapping):
            raise ConfigError("each observed terminal failure must be a mapping")
        if row.get("provider") not in {"codex", "claude"}:
            raise ConfigError("observed terminal failure provider is invalid")
        if row.get("stage") not in {"v4_blind_solver"}:
            raise ConfigError("only exhausted blind calls may be inherited")
        if not all(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("call_id", "source_run", "error")
        ):
            raise ConfigError("observed terminal failure provenance is incomplete")
        source_run = Path(str(row["source_run"]))
        if not source_run.is_absolute():
            source_run = root / source_run
        state_path = source_run / "state.json"
        if not state_path.is_file():
            raise ConfigError("observed terminal failure source state is missing")
        source_state = read_json(state_path)
        matching = [
            item
            for item in source_state.get("call_ledger", {}).values()
            if item.get("provider") == row["provider"]
            and item.get("stage") == row["stage"]
            and item.get("call_id") == row["call_id"]
            and item.get("status") == "failed"
        ]
        exhausted = (
            len(matching) == 1
            and (
                int(matching[0].get("retry_calls", 0)) >= 1
                or "attempt 2:" in str(matching[0].get("error", ""))
            )
        )
        if not exhausted:
            raise ConfigError(
                "observed terminal failure is not backed by an exhausted source call"
            )
    recovery_runs = experiment.get("recovery_source_runs", [])
    if not isinstance(recovery_runs, list) or not all(
        isinstance(item, str) and item.strip() for item in recovery_runs
    ):
        raise ConfigError("recovery_source_runs must be a list of run paths")
    for item in recovery_runs:
        source_run = Path(item)
        if not source_run.is_absolute():
            source_run = root / source_run
        state_path = source_run / "state.json"
        if not state_path.is_file():
            raise ConfigError(f"recovery source state is missing: {source_run}")
        source_state = read_json(state_path)
        if source_state.get("pipeline") != PIPELINE:
            raise ConfigError("recovery source belongs to another pipeline")

    claude = raw.get("claude_code")
    if not isinstance(claude, Mapping):
        raise ConfigError("claude_code settings are required")
    if claude.get("model_alias") != "opus":
        raise ConfigError("Claude must be requested through the exact 'opus' alias")
    if claude.get("expected_model_regex") != EXPECTED_OPUS_REGEX:
        raise ConfigError("Claude model resolution must be restricted to Opus 5")
    role_efforts = claude.get("role_efforts")
    if role_efforts != ROLE_EFFORTS:
        raise ConfigError("Claude role efforts must mirror the V4.5 Sol control")
    required_efforts = claude.get("required_attestation_efforts")
    if required_efforts != ["high", "xhigh", "max"]:
        raise ConfigError("attestation must cover high, xhigh, and max")
    if any(value not in ALLOWED_EFFORTS for value in role_efforts.values()):
        raise ConfigError("Claude role effort is invalid")
    for field in ("timeout_seconds", "concurrency", "attestation_maximum_age_seconds"):
        if type(claude.get(field)) is not int or claude[field] < 1:
            raise ConfigError(f"claude_code.{field} must be a positive integer")
    if type(claude.get("retries")) is not int or claude["retries"] < 0:
        raise ConfigError("claude_code.retries must be non-negative")
    budget = claude.get("max_budget_usd_per_call")
    if budget is not None and (type(budget) not in {int, float} or budget <= 0):
        raise ConfigError("claude_code.max_budget_usd_per_call must be positive or null")

    codex_expected = {
        "idea_search": ("gpt-5.6-sol", "max"),
        "batch_selector": ("gpt-5.6-sol", "max"),
        "concrete_planner": ("gpt-5.6-sol", "max"),
        "task_realizer": ("gpt-5.6-sol", "max"),
        "technical_verifier": ("gpt-5.6-sol", "high"),
        "blind_solver": ("gpt-5.6-sol", "xhigh"),
        "editorial_board": ("gpt-5.6-sol", "max"),
    }
    for role, (model, effort) in codex_expected.items():
        if base["model"]["roles"].get(role) != {
            "model": model,
            "reasoning_effort": effort,
        }:
            raise ConfigError(f"frozen Codex control routing changed for {role}")

    run = raw.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"].strip():
        raise ConfigError("run.id is required")
    if run.get("automatically_commit_generated_tasks") is not False:
        raise ConfigError("the Claude experiment cannot commit generated tasks")
    if run.get("mock") not in {True, False}:
        raise ConfigError("run.mock must be boolean")

    config = deepcopy(base)
    config["version"] = "1.0"
    config["pipeline"] = PIPELINE
    config["seed"] = raw.get("seed")
    if type(config["seed"]) is not int:
        raise ConfigError("seed must be an integer")
    config["claude_cross_eval"] = deepcopy(dict(experiment))
    config["claude_code"] = deepcopy(dict(claude))
    config["run"] = {**deepcopy(base["run"]), **deepcopy(dict(run))}
    for key, value in raw.get("paths", {}).items():
        config["paths"][key] = _resolve(root, value, f"paths.{key}")
    required_paths = (
        "prompts",
        "runs",
        "cache",
        "committed_task_registry",
        "fixed_anchor_manifest",
        "reservoir",
        "frozen_generation_source_run",
        "historical_case_manifest",
        "claude_system_prompt",
        "claude_attestation",
    )
    missing_paths = [key for key in required_paths if key not in config["paths"]]
    if missing_paths:
        raise ConfigError("missing experiment paths: " + ", ".join(missing_paths))
    config["experimental"] = {
        "enabled": True,
        "kind": "claude_opus_5_cross_authoring_and_evaluation",
        "gating": False,
        "auto_promote": False,
        "arm_id": "private",
    }
    config["claude_code"]["mock"] = bool(config["run"]["mock"])
    config["claude_code"]["system_prompt_path"] = config["paths"]["claude_system_prompt"]
    config["claude_code"]["attestation_path"] = config["paths"]["claude_attestation"]
    config["_config_path"] = str(source)
    config["_repository_root"] = str(root)
    config["_base_current_config"] = str(base_path)
    config["_v4_5_baseline_manifest"] = str(manifest_path)
    config["_v4_5_baseline_verification"] = baseline
    return config


def validate_claude_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate installation/auth/settings only; never sends a model prompt."""

    if config.get("run", {}).get("mock"):
        return {
            "mock": True,
            "model_calls_made": 0,
            "role_efforts": deepcopy(config["claude_code"]["role_efforts"]),
        }
    capabilities = inspect_claude_code(config["claude_code"]["executable"])
    auth = sanitized_auth_status(config["claude_code"]["executable"])
    if not auth["logged_in"]:
        raise ConfigError("Claude Code is not authenticated")
    return {
        "mock": False,
        "model_calls_made": 0,
        "capabilities": capabilities.as_dict(),
        "auth": auth,
        "requested_model_alias": config["claude_code"]["model_alias"],
        "expected_resolved_model_regex": config["claude_code"]["expected_model_regex"],
        "role_efforts": deepcopy(config["claude_code"]["role_efforts"]),
        "live_model_attestation_required_before_run": True,
        "required_attestation_efforts": deepcopy(
            config["claude_code"]["required_attestation_efforts"]
        ),
    }


def snapshot_v45_claude_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


__all__ = [
    "BASELINE_ID",
    "EXPECTED_OPUS_REGEX",
    "PIPELINE",
    "ROLE_EFFORTS",
    "VERSION",
    "load_v45_claude_config",
    "snapshot_v45_claude_config",
    "validate_claude_environment",
    "verify_v45_integrated_baseline",
]
